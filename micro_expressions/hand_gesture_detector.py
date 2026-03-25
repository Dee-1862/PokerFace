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
    PINCH_THRESHOLD = 0.08  # Increased for back-of-hand support

    # ... (rest of class)

    # Gesture recognition thresholds
    TWIST_SENSITIVITY = 0.02  # Minimum angle change for twist detection
    
    def __init__(self):
        """Initialize MediaPipe Hands."""
        try:
            model_path = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
            if not os.path.exists(model_path):
                print(f"[WARN] Warning: Model file not found at {model_path}")
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
            # Hold duration in seconds (User requested 4-5s)
            self.LOCK_DURATION = 4.0
            
            print("[OK] MediaPipe Hands initialized successfully")
        except Exception as e:
            print(f"[WARN] Error initializing MediaPipe Hands: {e}")
            self.detector = None
            self.timestamp_ms = 0
        
        # Hand landmark indices
        self.THUMB_TIP = 4
        self.INDEX_TIP = 8
        self.MIDDLE_TIP = 12
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
                
                # Analyze pinch
                pinch = self._detect_pinch(landmarks, frame.shape)
                
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
                    # If locked, we stay locked? No, allow re-lock.
                    # But for "Stretch", we pinch -> hold -> stretch (fingers separate).
                    # So when fingers separate (pinch not active), if we were locked, we stay locked?
                    # Let's say we stay locked as long as hand is detected?
                    # Or better: Once locked, it stays locked until hand is lost or explicitly reset.
                    # Actually, for "pinch ... then increase gap", pinch['active'] becomes False.
                    # So we MUST persist 'locked' even if pinch['active'] is False.
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
    
    def _detect_pinch(self, landmarks, frame_shape):
        """
        Detect pinch gesture (thumb and index finger together).
        """
        h, w = frame_shape[:2]
        
        # Get thumb tip and index tip positions
        thumb_tip = landmarks[self.THUMB_TIP]
        index_tip = landmarks[self.INDEX_TIP]
        
        # Convert to pixel coordinates
        thumb_pos = np.array([thumb_tip.x * w, thumb_tip.y * h])
        index_pos = np.array([index_tip.x * w, index_tip.y * h])
        
        # Calculate distance (normalized by frame size)
        distance = np.linalg.norm(thumb_pos - index_pos)
        normalized_distance = distance / max(w, h)
        
        # Pinch is active if distance is below threshold
        is_pinching = normalized_distance < self.PINCH_THRESHOLD
        
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
        
        # Finger tip landmarks
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        # MCP (knuckle) landmarks
        index_mcp = landmarks[5]
        middle_mcp = landmarks[9]
        ring_mcp = landmarks[13]
        pinky_mcp = landmarks[17]
        
        # Check if fingers are extended (tip above MCP in y-axis)
        index_up = index_tip.y < (index_mcp.y - 0.05)
        middle_up = middle_tip.y < (middle_mcp.y - 0.05)
        ring_down = ring_tip.y > (ring_mcp.y + 0.02)
        pinky_down = pinky_tip.y > (pinky_mcp.y + 0.02)
        
        if index_up and middle_up and ring_down and pinky_down:
            # Calculate average Y position of extended fingers
            scroll_y = (index_tip.y + middle_tip.y) / 2
            screen_y = int(scroll_y * h)
            
            return {
                'active': True,
                'y_normalized': scroll_y,  # 0.0 to 1.0
                'screen_y': screen_y,
                'direction': 'up' if scroll_y < 0.4 else 'down' if scroll_y > 0.6 else 'neutral'
            }
        
        return {'active': False, 'y_normalized': 0.5, 'screen_y': h // 2, 'direction': 'neutral'}
    
    def detect_two_hand_pinch_twist(self, gestures):
        """
        Detect two-hand pinch and twist gesture.
        """
        left_hand = gestures.get('left_hand')
        right_hand = gestures.get('right_hand')
        
        if not left_hand or not right_hand:
            return {'active': False, 'rotation_angle': 0.0}
        
        if not (left_hand['pinch']['active'] and right_hand['pinch']['active']):
            return {'active': False, 'rotation_angle': 0.0}
        
        left_center = left_hand['pinch']['thumb_pos']
        right_center = right_hand['pinch']['thumb_pos']
        
        dx = right_center[0] - left_center[0]
        dy = right_center[1] - left_center[1]
        
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        angle_deg = (angle_deg + 360) % 360
        
        return {
            'active': True,
            'rotation_angle': angle_deg,
            'left_pos': left_center,
            'right_pos': right_center
        }

    def render_hands(self, frame, gestures, shared_state=None, custom_labels=None):
        """
        Render hand landmarks with pinch-hold feedback.
        custom_labels: dict with 'left' and 'right' keys for text.
        """
        h, w = frame.shape[:2]
        
        if custom_labels is None:
            custom_labels = {'left': 'FACE', 'right': 'HEART'}

        COLORS = {
            'dot_inactive': (120, 120, 125),
            'dot_active': (200, 200, 210),
            'pinch_line': (255, 200, 150),
            'pinch_guide': (100, 100, 100), # Faint guide
            'pinch_circle': (255, 180, 120),
            'charge_track': (60, 60, 65),
            'charge_fill': (0, 255, 150),
            'lock_glow': (0, 200, 255),
            'text_active': (220, 220, 255)
        }
        
        hands_detected = gestures.get('hands_detected', 0)
        if hands_detected == 0:
            hint_text = f"Pinch & Hold ({int(self.LOCK_DURATION)}s)"
            (text_w, _), _ = cv2.getTextSize(hint_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.putText(frame, hint_text, (w // 2 - text_w // 2, h - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 145), 1)
            return frame

        # Render hands
        for hand_type in ['left_hand', 'right_hand']:
            if not gestures.get(hand_type):
                continue
                
            hand = gestures[hand_type]
            landmarks = hand['landmarks']
            pinch = hand['pinch']
            is_pinching = pinch['active']
            is_locked = pinch.get('is_locked', False)
            progress = pinch.get('hold_progress', 0.0)
            
            # Dots
            key_indices = [0, 4, 8, 12, 16, 20]
            dot_color = COLORS['dot_active'] if is_pinching else COLORS['dot_inactive']
            for idx in key_indices:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                cv2.circle(frame, (x, y), 4, dot_color, -1)
            
            thumb_pos = pinch['thumb_pos'].astype(int)
            index_pos = pinch['index_pos'].astype(int)
            mid_x = int((thumb_pos[0] + index_pos[0]) / 2)
            mid_y = int((thumb_pos[1] + index_pos[1]) / 2)
            
            # 1. Connected line 
            if is_pinching or is_locked:
                cv2.line(frame, tuple(thumb_pos), tuple(index_pos), COLORS['pinch_line'], 2)
            else:
                # Draw faint guide line to help user position
                cv2.line(frame, tuple(thumb_pos), tuple(index_pos), COLORS['pinch_guide'], 1)
                
                # Show distance value for debugging back-of-hand
                dist = pinch.get('distance', 0)
                if isinstance(dist, (int, float)):
                    cv2.putText(frame, f"{dist:.2f}", (mid_x, mid_y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS['pinch_guide'], 1)

            # 2. Charging Ring
            if progress > 0 and not is_locked:
                radius = 20
                cv2.circle(frame, (mid_x, mid_y), radius, COLORS['charge_track'], 2)
                angle = int(360 * progress)
                cv2.ellipse(frame, (mid_x, mid_y), (radius, radius), -90, 0, angle, COLORS['charge_fill'], 2)
                
                secs = max(0, self.LOCK_DURATION * (1 - progress))
                cv2.putText(frame, f"{secs:.1f}s", (mid_x - 10, mid_y + 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS['text_active'], 1)

            # 3. Locked / Adjust Mode
            if is_locked:
                # Strong glow
                cv2.circle(frame, tuple(thumb_pos), 6, COLORS['lock_glow'], -1)
                cv2.circle(frame, tuple(index_pos), 6, COLORS['lock_glow'], -1)
                cv2.line(frame, tuple(thumb_pos), tuple(index_pos), COLORS['lock_glow'], 3)
                
                # Show label
                label_key = 'left' if hand_type == 'left_hand' else 'right'
                label = custom_labels.get(label_key, "")
                
                # Get level
                level_str = ""
                if shared_state and label in ['FACE', 'HEART']: # Only show % for these specific modes
                    level = shared_state.get('face_button_level', 0) if hand_type == 'left_hand' else shared_state.get('heart_button_level', 0)
                    level_str = f" {level}%"
                
                text = f"{label}{level_str}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(frame, (mid_x - tw//2 - 4, mid_y - 28), (mid_x + tw//2 + 4, mid_y - 10), (40,40,40), -1)
                cv2.putText(frame, text, (mid_x - tw//2, mid_y - 16), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['text_active'], 1)

        return frame
    
    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, 'detector'):
            self.detector.close()

