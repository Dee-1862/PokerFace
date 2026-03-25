import cv2
import numpy as np
from collections import deque
from scipy import signal

class RPPGModule:
    """
    Standard Green-Channel rPPG (Reverted to original working logic).
    Uses Bandpass Filtering + FFT on the Mean Green Signal.
    """
    def __init__(self, buffer_size=150, fps=30, enable_validation=False):
        self.buffer_size = buffer_size
        self.fps = fps
        self.signal_buffer = deque(maxlen=buffer_size)
        self.bpm_buffer = deque(maxlen=10) # Smoothing buffer
        self.current_bpm = 0
        
        # Standard forehead and cheek landmarks
        self.ROI_IDXS = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                         397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136]
        
        # CLAHE for lighting normalization (Essential for Green Channel method)
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        
        # Validation support
        self.enable_validation = enable_validation
        self.validator = None
        if enable_validation:
            try:
                from heart_rate_validator import HeartRateValidator
                self.validator = HeartRateValidator()
                print("[OK] Heart rate validation enabled")
            except ImportError:
                print("[WARN] Validation module not available")
                self.enable_validation = False

    def initialize(self, shared_state):
        pass

    def _normalize_lighting(self, frame):
        """Apply CLAHE to stabilize lighting."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = self.clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def process(self, shared_state):
        if not shared_state.get('face_detected'):
            return

        # 1. Normalize Lighting (Crucial for Green Channel stability)
        frame = self._normalize_lighting(shared_state['frame'])
        landmarks = shared_state['landmarks']
        h, w = shared_state['frame_dimensions']

        # 2. Extract ROI Mask
        mask = np.zeros((h, w), dtype=np.uint8)
        points = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in self.ROI_IDXS])
        cv2.fillPoly(mask, [points], 255)

        # 3. Extract Mean Green Channel
        # cv2.mean returns (B, G, R, Alpha), we want index 1 (Green)
        mean_green = cv2.mean(frame, mask=mask)[1]
        self.signal_buffer.append(mean_green)

        # 4. Calculate BPM
        self.current_bpm = self._calculate_heart_rate()
        shared_state['heart_rate_bpm'] = self.current_bpm

    def _calculate_heart_rate(self):
        """
        Exact logic from your previous working code:
        Detrend -> Bandpass Filter -> FFT -> Peak Finding -> Smoothing
        """
        # Need enough data (at least 2 seconds approx)
        if len(self.signal_buffer) < 60:
            return 0
        
        # Get signal data
        sig = np.array(self.signal_buffer)
        
        # 1. Detrend (remove DC component and slow drift)
        sig = signal.detrend(sig)
        
        # 2. Apply bandpass filter (0.7-3 Hz, which is 48-180 BPM)
        nyquist = self.fps / 2
        low = 0.7 / nyquist
        high = 3.0 / nyquist
        
        try:
            b, a = signal.butter(5, [low, high], btype='band')
            sig_filtered = signal.filtfilt(b, a, sig)
        except ValueError:
            return self.current_bpm # Return last known good value if filter fails
        
        # 3. FFT to find dominant frequency
        fft = np.fft.rfft(sig_filtered)
        freqs = np.fft.rfftfreq(len(sig_filtered), 1.0 / self.fps)
        
        # 4. Find peak in valid range (50-150 BPM)
        bpm_range = (freqs >= 50/60) & (freqs <= 150/60)
        
        if not np.any(bpm_range):
            return 0
        
        power = np.abs(fft[bpm_range])
        peak_freq_idx = np.argmax(power)
        peak_freq = freqs[bpm_range][peak_freq_idx]
        
        # 5. Convert to BPM
        raw_bpm = peak_freq * 60
        raw_bpm = np.clip(raw_bpm, 50, 150)
        
        # 6. Outlier Rejection & Smoothing
        if len(self.bpm_buffer) > 0:
            recent_bpm = np.median(list(self.bpm_buffer))
            # Reject if change is too drastic (>20 BPM jump)
            if abs(raw_bpm - recent_bpm) > 20:
                return int(recent_bpm)
        
        self.bpm_buffer.append(raw_bpm)
        
        # Return median of buffer
        if len(self.bpm_buffer) >= 3:
            return int(np.median(list(self.bpm_buffer)))
        else:
            return int(raw_bpm)

    def render(self, frame, shared_state):
        # Check if minimal mode is enabled
        minimal_mode = shared_state.get('minimal_mode', False)
        if minimal_mode:
            # In minimal mode, don't render anything (AR controller handles it)
            return frame
        
        # Draw BPM text
        text = f"HEART RATE: {self.current_bpm} BPM"
        
        # Stability color
        if len(self.bpm_buffer) >= 5:
            std_dev = np.std(list(self.bpm_buffer))
            if std_dev < 5:
                color = (0, 255, 0) # Green (Stable)
                status = "Stable"
            else:
                color = (0, 255, 255) # Yellow (Unstable)
                status = "Stabilizing"
        else:
            color = (0, 255, 255)
            status = "Initializing"

        cv2.putText(frame, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Draw small status text
        cv2.putText(frame, status, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Validation status
        if self.enable_validation and self.validator:
            if self.validator.is_recording:
                val_text = f"VALIDATING: {len(self.validator.rppg_readings)} readings"
                cv2.putText(frame, val_text, (20, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                # Show summary if enough data
                if len(self.validator.rppg_readings) >= 10:
                    summary = self.validator.get_summary_string()
                    cv2.putText(frame, summary, (20, 140), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Disclaimer
        h, w = frame.shape[:2]
        cv2.putText(frame, "Research Tool - Not Medical Device", (20, h-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
        
        # Optional: Draw the graph if you want (logic from your snippet)
        self._draw_graph(frame)
        
        return frame
    
    def start_validation(self):
        """Start validation recording."""
        if self.validator:
            self.validator.start_recording()
            return True
        return False
    
    def stop_validation(self):
        """Stop validation recording and print metrics."""
        if self.validator:
            self.validator.stop_recording()
            self.validator.print_metrics()
            return True
        return False
    
    def add_validation_reading(self, apple_watch_bpm):
        """Add Apple Watch reading for validation."""
        if self.validator and self.validator.is_recording:
            return self.validator.add_reading(apple_watch_bpm, self.current_bpm)
        return False
    
    def get_validation_metrics(self):
        """Get current validation metrics."""
        if self.validator:
            return self.validator.calculate_metrics()
        return None
    
    def save_validation_results(self, filepath=None):
        """Save validation results to file."""
        if self.validator:
            return self.validator.save_results(filepath)
        return None

    def _draw_graph(self, frame):
        if len(self.signal_buffer) < 2: return
        
        h, w = frame.shape[:2]
        graph_h = 60
        margin = 20
        
        # Background
        cv2.rectangle(frame, (margin, h - graph_h - margin), (w - margin, h - margin), (50, 50, 50), -1)
        
        sig = np.array(self.signal_buffer)
        # Normalize to 0-1 for drawing
        if np.max(sig) - np.min(sig) == 0: return
        sig = (sig - np.min(sig)) / (np.max(sig) - np.min(sig))
        
        for i in range(len(sig) - 1):
            x1 = int(margin + (i / self.buffer_size) * (w - 2 * margin))
            y1 = int((h - margin) - sig[i] * graph_h)
            x2 = int(margin + ((i + 1) / self.buffer_size) * (w - 2 * margin))
            y2 = int((h - margin) - sig[i+1] * graph_h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    def cleanup(self):
        pass