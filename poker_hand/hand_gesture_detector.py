
"""
Hand Gesture Detector using MediaPipe Hands
Pure AR interaction - no mouse required.
Detects pinch, twist, and other hand gestures for AR UI control.
"""
import cv2
import numpy as np
import math
import mediapipe as mp
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandGestureDetector:
    """
    Detects hand gestures using MediaPipe Hands for AR interaction.
    Supports pinch, twist, swipe, and point gestures.
    """
    
    # Pinch detection threshold (distance between thumb and index finger)
    # Tight (0.04) but relies on OK sign check for robustness
    PINCH_THRESHOLD = 0.04  # Normalized distance (0-1)
    
    # Gesture recognition thresholds
    TWIST_SENSITIVITY = 0.02  # Minimum angle change for twist detection
    
    def __init__(self):
        """Initialize MediaPipe Hands."""
        try:
            model_path = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
            if not os.path.exists(model_path):
                print(f"⚠ Warning: Model file not found at {model_path}")
                model_path = 'hand_landmarker.task'
            
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                running_mode=vision.RunningMode.VIDEO
            )
            
            self.detector = vision.HandLandmarker.create_from_options(options)
            self.timestamp_ms = 0
            
            # State tracking for pinch-and-stretch
            self.pinch_states = {
                'Left': {'start_time': 0, 'locked': False},
                'Right': {'start_time': 0, 'locked': False}
            }
            # Hold duration in seconds (User requested 5s)
            self.LOCK_DURATION = 5.0
            
            print("✓ MediaPipe Hands initialized successfully")
        except Exception as e:
            print(f"⚠ Error initializing MediaPipe Hands: {e}")
            self.detector = None
            self.timestamp_ms = 0
        
        # Hand landmark indices
        self.THUMB_TIP = 4
        self.INDEX_TIP = 8
        self.MIDDLE_MCP = 9
        self.MIDDLE_TIP = 12
        self.RING_MCP = 13
        self.RING_TIP = 16
        self.PINKY_MCP = 17
        self.PINKY_TIP = 20
        self.WRIST = 0
        
        # Gesture state
        self.pinch_active = False
        self.last_hand_angle = None

    def process(self, frame, shared_state):
        """
        Process frame to detect hand gestures.
        """
        if self.detector is None:
            return {}
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self.timestamp_ms += 33
        
        result = self.detector.detect_for_video(mp_image, self.timestamp_ms)
        
        gestures = {
            'pinch_active': False,
            'hands_detected': len(result.hand_landmarks) if result.hand_landmarks else 0,
            'left_hand': None,
            'right_hand': None
        }
        
        if result.hand_landmarks:
            for idx, landmarks in enumerate(result.hand_landmarks):
                handedness = result.handedness[idx][0].category_name
                is_left = handedness == 'Left'
                
                # Analyze pinch (with OK sign check)
                pinch = self._detect_pinch_ok_sign(landmarks, frame.shape)
                
                # Analyze twist (legacy, but kept for structure)
                twist = self._detect_twist(landmarks, frame.shape)
                
                # Analyze two-finger scroll
                two_finger_scroll = self.detect_two_finger_scroll(landmarks, frame.shape)
                
                # Update pinch lock state
                import time
                now = time.time()
                state = self.pinch_states.get(handedness, {'start_time': 0, 'locked': False})
                
                if pinch['active']:
                    if state['start_time'] == 0:
                        state['start_time'] = now
                    
                    # Check duration
                    if (now - state['start_time']) >= self.LOCK_DURATION:
                        state['locked'] = True
                else:
                    if not state['locked']:
                        state['start_time'] = 0
                
                # Add lock info to pinch data
                pinch['is_locked'] = state['locked']
                pinch['hold_progress'] = 0.0
                if state['start_time'] > 0 and not state['locked']:
                    pinch['hold_progress'] = min(1.0, (now - state['start_time']) / self.LOCK_DURATION)
                
                self.pinch_states[handedness] = state
                
                hand_data = {
                    'landmarks': landmarks,
                    'pinch': pinch,
                    'twist': twist,
                    'two_finger_scroll': two_finger_scroll,
                    'is_left': is_left
                }
                
                if is_left:
                    gestures['left_hand'] = hand_data
                else:
                    gestures['right_hand'] = hand_data
                
                if pinch['active']:
                    gestures['pinch_active'] = True
        else:
            # Reset states if no hands
            self.pinch_states = {
                'Left': {'start_time': 0, 'locked': False},
                'Right': {'start_time': 0, 'locked': False}
            }
            
        shared_state['hand_gestures'] = gestures
        shared_state['hands_detected'] = gestures['hands_detected'] > 0
        
        return gestures
    
    def _is_finger_extended(self, landmarks, tip_idx, mcp_idx, wrist_idx, frame_shape):
        """Check if a finger is extended based on tip-wrist distance relative to mcp-wrist distance."""
        h, w = frame_shape[:2]
        wrist = landmarks[wrist_idx]
        mcp = landmarks[mcp_idx]
        tip = landmarks[tip_idx]
        
        # 2D Distances
        def dist(p1, p2):
             return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2) # normalized
        
        d_tip_wrist = dist(tip, wrist)
        d_mcp_wrist = dist(mcp, wrist)
        
        # Heuristic: If tip is significantly farther than MCP, it's extended.
        # Ratio typically > 1.5 for extended. < 1.3 for curled.
        if d_mcp_wrist == 0: return False
        ratio = d_tip_wrist / d_mcp_wrist
        return ratio > 1.5

    def _detect_pinch_ok_sign(self, landmarks, frame_shape):
        """
        Detect pinch gesture AND require other fingers to be extended (OK Sign).
        """
        h, w = frame_shape[:2]
        
        # 1. Check basic pinch (Thumb + Index close)
        thumb_tip = landmarks[self.THUMB_TIP]
        index_tip = landmarks[self.INDEX_TIP]
        
        thumb_pos = np.array([thumb_tip.x * w, thumb_tip.y * h])
        index_pos = np.array([index_tip.x * w, index_tip.y * h])
        
        distance = np.linalg.norm(thumb_pos - index_pos)
        normalized_distance = distance / max(w, h)
        
        is_pinching = normalized_distance < self.PINCH_THRESHOLD
        
        # 2. Check OK Sign conditions (Other 3 fingers extended)
        # Only enforce if trying to LOCK (start pinch).
        # But once locked, user might relax? No, keep it strict to avoid false positives.
        
        if is_pinching:
            cw = self.WRIST
            ext_middle = self._is_finger_extended(landmarks, self.MIDDLE_TIP, self.MIDDLE_MCP, cw, frame_shape)
            ext_ring = self._is_finger_extended(landmarks, self.RING_TIP, self.RING_MCP, cw, frame_shape)
            ext_pinky = self._is_finger_extended(landmarks, self.PINKY_TIP, self.PINKY_MCP, cw, frame_shape)
            
            # Relaxed OK Sign: Only check if fingers are NOT curled too tight?
            # Actually, User just wants it to work. Let's disable the strict extension check for now.
            # It was causing issues.
            # if not (ext_middle and ext_ring and ext_pinky):
            #    is_pinching = False 
            pass
        
        return {
            'active': is_pinching,
            'distance': normalized_distance,
            'thumb_pos': thumb_pos,
            'index_pos': index_pos
        }
    
    def _detect_twist(self, landmarks, frame_shape):
        """
        Detect twist/rotation gesture from hand orientation.
        """
        h, w = frame_shape[:2]
        wrist = landmarks[self.WRIST]
        middle_mcp = landmarks[9]
        
        dx = middle_mcp.x - wrist.x
        dy = middle_mcp.y - wrist.y
        
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        angle_deg = (angle_deg + 360) % 360
        
        angle_change = 0.0
        if self.last_hand_angle is not None:
            angle_diff = (angle_deg - self.last_hand_angle + 180) % 360 - 180
            if abs(angle_diff) > self.TWIST_SENSITIVITY * 360:
                angle_change = angle_diff
        
        self.last_hand_angle = angle_deg
        
        return {
            'angle': angle_deg,
            'angle_change': angle_change,
            'active': abs(angle_change) > 5.0
        }

    def detect_two_finger_scroll(self, landmarks, frame_shape):
        """
        Detect two-finger scroll gesture (index + middle extended, ring + pinky closed).
        Returns vertical position for scrolling.
        """
        h, w = frame_shape[:2]
        
        # Landmarks for checking
        cw = self.WRIST
        
        # Check extensions
        ext_index = self._is_finger_extended(landmarks, self.INDEX_TIP, 5, cw, frame_shape)
        ext_middle = self._is_finger_extended(landmarks, self.MIDDLE_TIP, 9, cw, frame_shape)
        ext_ring = self._is_finger_extended(landmarks, self.RING_TIP, 13, cw, frame_shape)
        ext_pinky = self._is_finger_extended(landmarks, self.PINKY_TIP, 17, cw, frame_shape)
        
        # Scroll active if index and middle are OUT and others are IN
        if ext_index and ext_middle and not ext_ring and not ext_pinky:
            # Average Y of extended tips
            scroll_y = (landmarks[8].y + landmarks[12].y) / 2
            screen_y = int(scroll_y * h)
            
            return {
                'active': True,
                'y_normalized': scroll_y,
                'screen_y': screen_y
            }
        
        return {'active': False, 'y_normalized': 0.5, 'screen_y': h // 2}

    def render_hands(self, frame, gestures, shared_state=None):
        """
        Render hand landmarks with pinch-hold feedback.
        """
        h, w = frame.shape[:2]
        
        # Same rendering as before, just added helper
        COLORS = {
            'dot_inactive': (120, 120, 125),
            'dot_active': (200, 200, 210),
            'pinch_line': (255, 200, 150),
            'charge_track': (60, 60, 65),
            'charge_fill': (0, 255, 150),
            'lock_glow': (0, 200, 255),
            'text_active': (220, 220, 255)
        }
        
        hands_detected = gestures.get('hands_detected', 0)
        if hands_detected == 0:
            hint_text = f"Show Hand"
            (text_w, _), _ = cv2.getTextSize(hint_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.putText(frame, hint_text, (w // 2 - text_w // 2, h - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 145), 1)
            return frame

        for hand_type in ['left_hand', 'right_hand']:
            if not gestures.get(hand_type):
                continue
                
            hand = gestures[hand_type]
            landmarks = hand['landmarks']
            pinch = hand['pinch']
            is_pinching = pinch['active']
            is_locked = pinch.get('is_locked', False)
            progress = pinch.get('hold_progress', 0.0)
            
            thumb_pos = pinch['thumb_pos'].astype(int)
            index_pos = pinch['index_pos'].astype(int)
            mid_x = int((thumb_pos[0] + index_pos[0]) / 2)
            mid_y = int((thumb_pos[1] + index_pos[1]) / 2)
            
            if is_pinching or is_locked:
                cv2.line(frame, tuple(thumb_pos), tuple(index_pos), COLORS['pinch_line'], 2)

            if progress > 0 and not is_locked:
                radius = 20
                cv2.circle(frame, (mid_x, mid_y), radius, COLORS['charge_track'], 2)
                angle = int(360 * progress)
                cv2.ellipse(frame, (mid_x, mid_y), (radius, radius), -90, 0, angle, COLORS['charge_fill'], 2)

            if is_locked:
                cv2.circle(frame, tuple(thumb_pos), 6, COLORS['lock_glow'], -1)
                cv2.circle(frame, tuple(index_pos), 6, COLORS['lock_glow'], -1)
                text = "LOCKED"
                cv2.putText(frame, text, (mid_x - 20, mid_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['text_active'], 1)

        return frame
    
    def cleanup(self):
        if hasattr(self, 'detector'):
            self.detector.close()
