"""
Stress Detection Module with Automatic Baseline Calibration.
Designed for AR glasses - no keyboard input required.
"""
import cv2
import numpy as np
import time
from collections import deque
from stress_classifier import StressClassifier

class StressDetectorModule:
    """
    Calculates stress score by comparing current metrics against a baseline.
    Automatically captures baseline when face is stable and rPPG is reliable.
    """
    
    # Weights for stress calculation
    WEIGHT_HEART_RATE = 0.4
    WEIGHT_AU_STRESS = 0.4
    WEIGHT_BLINK_RATE = 0.2
    
    # Calibration settings
    STABILITY_WAIT_TIME = 2.0  # Seconds of stable face before calibration starts
    CALIBRATION_DURATION = 5.0  # Seconds to capture baseline
    
    # Stress-related Action Units
    STRESS_AUS = ['AU1', 'AU2', 'AU4']  # Brow movements
    POSITIVE_AUS = ['AU12']  # Smile (reduces stress score)
    
    def __init__(self):
        # Baseline storage
        self.baseline = {
            'heart_rate': None,
            'action_units': {},
            'blink_rate': None,
            'captured': False
        }
        
        # Calibration state
        self.calibration_state = 'idle'  # idle, waiting, capturing, complete
        self.face_stable_since = None
        self.calibration_start_time = None
        
        # Calibration buffers
        self.cal_hr_buffer = []
        self.cal_au_buffer = []
        self.cal_blink_buffer = []
        
        # Running metrics
        self.stress_score = 0.0
        self.hr_deviation = 0.0
        self.au_deviation = 0.0
        
        # Blink tracking
        self.blink_buffer = deque(maxlen=300)  # ~10 seconds at 30fps
        self.last_blink_state = False
        self.blink_count = 0
        
        # Smoothing
        self.stress_history = deque(maxlen=15)
        
        # Advanced classification
        self.classifier = StressClassifier()
        self.stress_category = "Unknown"
        self.stress_trend = "Stable"
        self.stress_type = "None"
        
    def initialize(self, shared_state):
        pass
    
    def _check_stability(self, shared_state):
        """Check if face is stable and rPPG signal is reliable."""
        if not shared_state.get('face_detected'):
            return False
        
        # Check rPPG stability (from bpm_buffer std_dev)
        hr = shared_state.get('heart_rate_bpm', 0)
        if hr < 50:  # Not enough rPPG data yet
            return False
        
        return True
    
    def _update_calibration(self, shared_state):
        """Manage the automatic calibration state machine."""
        current_time = time.time()
        is_stable = self._check_stability(shared_state)
        
        if self.calibration_state == 'idle':
            if is_stable:
                self.calibration_state = 'waiting'
                self.face_stable_since = current_time
                
        elif self.calibration_state == 'waiting':
            if not is_stable:
                self.calibration_state = 'idle'
                self.face_stable_since = None
            elif current_time - self.face_stable_since >= self.STABILITY_WAIT_TIME:
                # Start calibration
                self.calibration_state = 'capturing'
                self.calibration_start_time = current_time
                self.cal_hr_buffer = []
                self.cal_au_buffer = []
                self.cal_blink_buffer = []
                # Suppress messages in minimal mode
                if not shared_state.get('minimal_mode', False):
                    print("[OK] Starting baseline calibration...")
                
        elif self.calibration_state == 'capturing':
            if not is_stable:
                # Abort calibration
                self.calibration_state = 'idle'
                self.calibration_start_time = None
                # Suppress messages in minimal mode
                if not shared_state.get('minimal_mode', False):
                    print("[WARN] Calibration aborted - face lost")
            else:
                # Collect data
                hr = shared_state.get('heart_rate_bpm', 0)
                aus = shared_state.get('action_units', {})
                blink = aus.get('AU45', 0)  # Blink AU
                
                if hr > 0:
                    self.cal_hr_buffer.append(hr)
                if aus:
                    self.cal_au_buffer.append(aus.copy())
                self.cal_blink_buffer.append(1 if blink > 0.5 else 0)
                
                # Check if calibration is complete
                elapsed = current_time - self.calibration_start_time
                if elapsed >= self.CALIBRATION_DURATION:
                    self._finalize_baseline(shared_state)
                    
    def _finalize_baseline(self, shared_state=None):
        """Calculate and store baseline values."""
        if len(self.cal_hr_buffer) > 0:
            self.baseline['heart_rate'] = np.median(self.cal_hr_buffer)
        
        if len(self.cal_au_buffer) > 0:
            # Average each AU across all frames
            au_keys = self.cal_au_buffer[0].keys()
            for key in au_keys:
                values = [frame.get(key, 0) for frame in self.cal_au_buffer]
                self.baseline['action_units'][key] = np.mean(values)
        
        if len(self.cal_blink_buffer) > 0:
            # Calculate blink rate (blinks per second)
            blink_count = sum(self.cal_blink_buffer)
            self.baseline['blink_rate'] = blink_count / self.CALIBRATION_DURATION
        
        self.baseline['captured'] = True
        self.calibration_state = 'complete'
        
        # Suppress messages in minimal mode (calibration happens silently)
        if shared_state and not shared_state.get('minimal_mode', False):
            print(f"[OK] Baseline captured!")
            print(f"  HR: {self.baseline['heart_rate']:.1f} BPM")
            print(f"  AUs: {len(self.baseline['action_units'])} tracked")
        
    def _calculate_stress(self, shared_state):
        """Calculate stress score based on deviation from baseline."""
        if not self.baseline['captured']:
            return 0.0
        
        hr = shared_state.get('heart_rate_bpm', 0)
        aus = shared_state.get('action_units', {})
        
        # 1. Heart Rate Deviation (normalized 0-1)
        baseline_hr = self.baseline['heart_rate'] or 70
        hr_diff = abs(hr - baseline_hr)
        self.hr_deviation = min(hr_diff / 30.0, 1.0)  # 30 BPM diff = max stress
        
        # 2. Action Unit Deviation (stress indicators)
        au_stress_total = 0.0
        au_count = 0
        
        for au in self.STRESS_AUS:
            current = aus.get(au, 0)
            baseline = self.baseline['action_units'].get(au, 0)
            deviation = max(0, current - baseline)  # Only count increases
            au_stress_total += deviation
            au_count += 1
        
        # Subtract positive emotions (smile reduces stress)
        for au in self.POSITIVE_AUS:
            current = aus.get(au, 0)
            baseline = self.baseline['action_units'].get(au, 0)
            positive_deviation = max(0, current - baseline)
            au_stress_total -= positive_deviation * 0.5
        
        self.au_deviation = max(0, min(au_stress_total / (au_count * 0.5), 1.0))
        
        # 3. Blink Rate Factor (higher blink rate = more stress)
        blink_factor = 0.0
        current_blink = aus.get('AU45', 0)
        is_blinking = current_blink > 0.5
        
        # Detect blink transitions
        if is_blinking and not self.last_blink_state:
            self.blink_count += 1
        self.last_blink_state = is_blinking
        self.blink_buffer.append(1 if is_blinking else 0)
        
        if self.baseline['blink_rate'] and len(self.blink_buffer) > 60:
            # Calculate current blink rate over last 5 seconds
            recent_blinks = sum(list(self.blink_buffer)[-150:])
            current_rate = recent_blinks / 5.0
            baseline_rate = self.baseline['blink_rate']
            if baseline_rate > 0:
                blink_factor = min((current_rate / baseline_rate - 1.0) * 0.5, 1.0)
                blink_factor = max(0, blink_factor)
        
        # Combine with weights
        raw_stress = (
            self.hr_deviation * self.WEIGHT_HEART_RATE +
            self.au_deviation * self.WEIGHT_AU_STRESS +
            blink_factor * self.WEIGHT_BLINK_RATE
        )
        
        # Smooth the result
        self.stress_history.append(raw_stress)
        smoothed = np.median(list(self.stress_history))
        
        return min(max(smoothed, 0.0), 1.0)
    
    def get_calibration_progress(self):
        """Returns calibration progress as 0.0 to 1.0."""
        if self.calibration_state == 'capturing' and self.calibration_start_time:
            elapsed = time.time() - self.calibration_start_time
            return min(elapsed / self.CALIBRATION_DURATION, 1.0)
        elif self.calibration_state == 'complete':
            return 1.0
        return 0.0
    
    def process(self, shared_state):
        """Main processing - update calibration and calculate stress."""
        # Only attempt calibration if not already done
        if not self.baseline['captured']:
            self._update_calibration(shared_state)
        
        # Calculate stress if baseline is captured
        self.stress_score = self._calculate_stress(shared_state)
        
        # Advanced classification
        if self.baseline['captured']:
            timestamp = shared_state.get('timestamp', time.time())
            analysis = self.classifier.analyze(self.stress_score, timestamp)
            
            self.stress_category = analysis['stress_category']
            self.stress_trend = analysis['stress_trend']
            self.stress_type = analysis['stress_type']
        else:
            self.stress_category = "Calibrating"
            self.stress_trend = "Stable"
            self.stress_type = "None"
        
        # Update shared state
        shared_state['stress_score'] = self.stress_score
        shared_state['stress_category'] = self.stress_category
        shared_state['stress_trend'] = self.stress_trend
        shared_state['stress_type'] = self.stress_type
        shared_state['baseline_captured'] = self.baseline['captured']
        shared_state['calibration_state'] = self.calibration_state
        shared_state['calibration_progress'] = self.get_calibration_progress()
        shared_state['stress_events'] = self.classifier.detected_events
    
    def render(self, frame, shared_state):
        """Render stress indicator and calibration UI."""
        # Check if minimal mode is enabled
        minimal_mode = shared_state.get('minimal_mode', False)
        
        if minimal_mode:
            # In minimal mode, don't render anything (AR controller handles it)
            return frame
        
        h, w = frame.shape[:2]
        
        # Position for stress display (right side)
        x_pos = w - 200
        y_pos = 30
        
        # Calibration UI
        if not self.baseline['captured']:
            self._render_calibration_ui(frame, shared_state, x_pos, y_pos)
        else:
            self._render_stress_indicator(frame, shared_state, x_pos, y_pos)
        
        return frame
    
    def _render_calibration_ui(self, frame, shared_state, x, y):
        """Render the calibration progress UI (AR-friendly)."""
        progress = self.get_calibration_progress()
        state = self.calibration_state
        
        # Background panel (semi-transparent)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 10, y - 20), (x + 190, y + 80), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        if state == 'idle':
            text = "Scanning..."
            color = (100, 100, 100)
            cv2.putText(frame, text, (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            cv2.putText(frame, "Look at subject", (x, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
        elif state == 'waiting':
            text = "Locking..."
            color = (0, 255, 255)  # Yellow
            cv2.putText(frame, text, (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            cv2.putText(frame, "Hold steady", (x, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
        elif state == 'capturing':
            text = "CALIBRATING"
            color = (0, 200, 255)  # Orange
            cv2.putText(frame, text, (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Progress bar
            bar_width = 160
            bar_height = 12
            bar_x = x
            bar_y = y + 40
            
            # Background
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (60, 60, 60), -1)
            # Progress fill
            fill_width = int(bar_width * progress)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
            # Border
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), 1)
            
            # Percentage text
            pct_text = f"{int(progress * 100)}%"
            cv2.putText(frame, pct_text, (x + 70, y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def _render_stress_indicator(self, frame, shared_state, x, y):
        """Render the enhanced stress level indicator with 5-level colors, trends, and analytics."""
        stress = self.stress_score
        
        # Background panel (larger for more info)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 10, y - 20), (x + 190, y + 140), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "STRESS LEVEL", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        # 5-level color coding
        if stress < 0.20:
            color = (0, 255, 0)  # Green - Very Low
            label = "VERY LOW"
        elif stress < 0.40:
            color = (0, 255, 100)  # Light Green - Low
            label = "LOW"
        elif stress < 0.60:
            color = (0, 255, 255)  # Yellow - Moderate
            label = "MODERATE"
        elif stress < 0.80:
            color = (0, 150, 255)  # Orange - High
            label = "HIGH"
        else:
            color = (0, 0, 255)  # Red - Very High
            label = "VERY HIGH"
        
        # Stress bar with level markers
        bar_width = 160
        bar_height = 20
        bar_x = x
        bar_y = y + 15
        
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (60, 60, 60), -1)
        
        # Level threshold markers
        for threshold, marker_color in [(0.2, (0, 255, 0)), (0.4, (0, 255, 100)), 
                                        (0.6, (0, 255, 255)), (0.8, (0, 150, 255))]:
            marker_x = int(bar_x + bar_width * threshold)
            cv2.line(frame, (marker_x, bar_y), (marker_x, bar_y + bar_height), marker_color, 1)
        
        # Fill with gradient effect
        fill_width = int(bar_width * stress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), color, -1)
        
        # Border
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), 1)
        
        # Label and score
        cv2.putText(frame, label, (x, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        score_text = f"{stress:.0%}"
        cv2.putText(frame, score_text, (x + 110, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Trend indicator
        trend = self.stress_trend
        trend_y = y + 70
        if trend == "Increasing":
            trend_color = (0, 0, 255)  # Red
            trend_symbol = "↑"
        elif trend == "Decreasing":
            trend_color = (0, 255, 0)  # Green
            trend_symbol = "↓"
        elif trend == "Fluctuating":
            trend_color = (0, 255, 255)  # Yellow
            trend_symbol = "~"
        else:
            trend_color = (200, 200, 200)  # Gray
            trend_symbol = "->"
        
        cv2.putText(frame, f"Trend: {trend_symbol} {trend}", (x, trend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, trend_color, 1)
        
        # Stress type
        stress_type = self.stress_type
        if stress_type != "None":
            type_color = (0, 150, 255) if stress_type == "Acute" else (0, 100, 255)
            cv2.putText(frame, f"Type: {stress_type}", (x, trend_y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, type_color, 1)
        
        # Component breakdown
        cv2.putText(frame, f"HR: +{self.hr_deviation:.0%}  AU: +{self.au_deviation:.0%}", 
                   (x, y + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        
        # Event count
        event_count = len(self.classifier.detected_events)
        if event_count > 0:
            cv2.putText(frame, f"Events: {event_count}", (x, y + 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 100, 100), 1)
        
        # Mini stress history graph
        self._draw_mini_timeline(frame, x, y + 135, 160, 15)
    
    def reset_baseline(self):
        """Reset baseline for recalibration."""
        self.baseline = {
            'heart_rate': None,
            'action_units': {},
            'blink_rate': None,
            'captured': False
        }
        self.calibration_state = 'idle'
        self.calibration_start_time = None
        self.face_stable_since = None
        self.classifier.reset()
        print("[OK] Baseline reset - ready to recalibrate")
    
    def get_stress_statistics(self):
        """Get current stress statistics from classifier."""
        return self.classifier.get_stress_statistics()
    
    def _draw_mini_timeline(self, frame, x, y, width, height):
        """Draw mini stress timeline graph."""
        if len(self.classifier.stress_history) < 2:
            return
        
        # Get recent history
        history = list(self.classifier.stress_history)[-30:]  # Last 30 points
        if len(history) < 2:
            return
        
        # Background
        cv2.rectangle(frame, (x, y), (x + width, y + height), (30, 30, 30), -1)
        
        # Draw graph
        for i in range(len(history) - 1):
            x1 = int(x + (i / len(history)) * width)
            y1 = int(y + height - (history[i] * height))
            x2 = int(x + ((i + 1) / len(history)) * width)
            y2 = int(y + height - (history[i + 1] * height))
            
            # Color based on stress level
            stress_val = (history[i] + history[i + 1]) / 2
            if stress_val < 0.2:
                line_color = (0, 255, 0)
            elif stress_val < 0.4:
                line_color = (0, 255, 100)
            elif stress_val < 0.6:
                line_color = (0, 255, 255)
            elif stress_val < 0.8:
                line_color = (0, 150, 255)
            else:
                line_color = (0, 0, 255)
            
            cv2.line(frame, (x1, y1), (x2, y2), line_color, 1)
        
        # Border
        cv2.rectangle(frame, (x, y), (x + width, y + height), (100, 100, 100), 1)
    
    def cleanup(self):
        pass
