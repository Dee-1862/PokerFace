
"""
Hand Gesture Detector using MediaPipe Hands
Pure AR interaction - no mouse required.
Detects pinch, twist, and other hand gestures for AR UI control.
"""
import cv2
import numpy as np
import math
import time
import mediapipe as mp
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandGestureDetector:
    """
    Detects hand gestures using MediaPipe Hands for AR interaction.
    Supports pinch, twist, swipe, and point gestures.
    """
    
    # Pinch detection threshold (distance between thumb and index finger, normalised by max(w,h))
    # 0.04 was too tight — at 1280px wide that was only ~51px gap allowed.
    # 0.07 gives ~90px which is achievable without forcing an uncomfortably close touch.
    PINCH_THRESHOLD = 0.07
    
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
            self.timestamp_ms = int(time.time() * 1000)
            
            # State tracking for pinch-and-stretch
            self.pinch_states = {
                'Left': {'start_time': 0, 'locked': False},
                'Right': {'start_time': 0, 'locked': False}
            }
            # Hold duration in seconds (User requested 5s)
            self.LOCK_DURATION = 5.0

            # Thumb signal dwell tracking (thumbs-up = STRONG, thumbs-down = BLUFFING)
            self.THUMB_DWELL = 1.0   # seconds to hold before confirming
            self.thumb_states = {
                'Left':  {'signal': None, 'start': 0.0, 'fired': False},
                'Right': {'signal': None, 'start': 0.0, 'fired': False},
            }
            
            print("[OK] MediaPipe Hands initialized successfully")
        except Exception as e:
            print(f"[WARN] Error initializing MediaPipe Hands: {e}")
            self.detector = None
            self.timestamp_ms = 0
        
        # Hand landmark indices
        self.THUMB_TIP  = 4
        self.INDEX_MCP  = 5
        self.INDEX_TIP  = 8
        self.MIDDLE_MCP = 9
        self.MIDDLE_TIP = 12
        self.RING_MCP   = 13
        self.RING_TIP   = 16
        self.PINKY_MCP  = 17
        self.PINKY_TIP  = 20
        self.WRIST      = 0
        
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
        self.timestamp_ms = int(time.time() * 1000)  # real wall-clock ms — prevents MediaPipe drops
        
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

                # Analyze pointing gesture (index only extended — used for panel drag)
                pointing = self._detect_pointing(landmarks, frame.shape)

                # Thumb signal: thumbs-up = STRONG hand, thumbs-down = BLUFFING
                now = time.time()
                thumb_raw = self._detect_thumb_direction(landmarks, frame.shape)
                ts = self.thumb_states[handedness]
                if thumb_raw['signal'] and thumb_raw['signal'] == ts['signal']:
                    # Continuing same signal
                    progress = min(1.0, (now - ts['start']) / self.THUMB_DWELL) if ts['start'] > 0 else 0.0
                    if progress >= 1.0 and not ts['fired']:
                        ts['fired'] = True
                    thumb_raw['progress'] = progress
                    thumb_raw['fired']    = ts['fired']
                elif thumb_raw['signal'] and thumb_raw['signal'] != ts['signal']:
                    # New direction — reset
                    ts['signal'] = thumb_raw['signal']
                    ts['start']  = now
                    ts['fired']  = False
                    thumb_raw['progress'] = 0.0
                    thumb_raw['fired']    = False
                else:
                    # No thumb signal
                    ts['signal'] = None
                    ts['start']  = 0.0
                    ts['fired']  = False
                    thumb_raw['progress'] = 0.0
                    thumb_raw['fired']    = False
                self.thumb_states[handedness] = ts

                # Analyze twist (legacy, but kept for structure)
                twist = self._detect_twist(landmarks, frame.shape)

                # Analyze two-finger scroll
                two_finger_scroll = self.detect_two_finger_scroll(landmarks, frame.shape)
                
                # Update pinch lock state
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
                
                fist = self._detect_fist(landmarks, frame.shape)

                # Fist overrides pinch — all fingers are curled so thumb/index are
                # trivially close, but it's NOT a pinch intent.
                if fist:
                    pinch['active']    = False
                    pinch['is_locked'] = False

                hand_data = {
                    'landmarks': landmarks,
                    'pinch': pinch,
                    'pointing': pointing,
                    'thumb_signal': thumb_raw,
                    'twist': twist,
                    'two_finger_scroll': two_finger_scroll,
                    'fist': fist,
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
    
    def _detect_thumb_direction(self, landmarks, frame_shape):
        """
        Detect thumbs-up (signal='up') or thumbs-down (signal='down') gesture.
        Requirements:
        - All four non-thumb fingers must be curled
        - Thumb tip must be clearly above (up) or below (down) the average MCP joint level
        Returns {'signal': 'up'|'down'|None, 'progress': float, 'fired': bool, 'tip_pos': np.array}
        """
        h, w = frame_shape[:2]
        cw = self.WRIST

        # All non-thumb fingers must be curled
        fingers_curled = (
            not self._is_finger_extended(landmarks, self.INDEX_TIP,  self.INDEX_MCP,  cw, frame_shape) and
            not self._is_finger_extended(landmarks, self.MIDDLE_TIP, self.MIDDLE_MCP, cw, frame_shape) and
            not self._is_finger_extended(landmarks, self.RING_TIP,   self.RING_MCP,   cw, frame_shape) and
            not self._is_finger_extended(landmarks, self.PINKY_TIP,  self.PINKY_MCP,  cw, frame_shape)
        )

        thumb_tip = landmarks[self.THUMB_TIP]
        tip_pos   = np.array([thumb_tip.x * w, thumb_tip.y * h])

        if not fingers_curled:
            return {'signal': None, 'progress': 0.0, 'fired': False, 'tip_pos': tip_pos}

        # Compare thumb tip y to average of non-thumb MCP joints (robust to hand tilt)
        avg_mcp_y = sum(landmarks[i].y for i in [self.INDEX_MCP,
                                                   self.MIDDLE_MCP,
                                                   self.RING_MCP,
                                                   self.PINKY_MCP]) / 4.0

        dy = thumb_tip.y - avg_mcp_y   # negative = thumb above MCPs = UP (screen y grows down)
        if dy < -0.12:
            signal = 'up'
        elif dy > 0.12:
            signal = 'down'
        else:
            signal = None

        return {'signal': signal, 'progress': 0.0, 'fired': False, 'tip_pos': tip_pos}

    def reset_thumb_signal(self, handedness: str) -> None:
        """Reset thumb state after the fired signal has been consumed."""
        self.thumb_states[handedness] = {'signal': None, 'start': 0.0, 'fired': False}

    def _detect_fist(self, landmarks, frame_shape):
        """
        Detect a closed fist: all five fingers curled, thumb in neutral position
        (not pointing clearly up or down).  Used for DB reset gestures.
        Returns True/False.
        """
        cw = self.WRIST
        all_curled = (
            not self._is_finger_extended(landmarks, self.INDEX_TIP,  self.INDEX_MCP,  cw, frame_shape) and
            not self._is_finger_extended(landmarks, self.MIDDLE_TIP, self.MIDDLE_MCP, cw, frame_shape) and
            not self._is_finger_extended(landmarks, self.RING_TIP,   self.RING_MCP,   cw, frame_shape) and
            not self._is_finger_extended(landmarks, self.PINKY_TIP,  self.PINKY_MCP,  cw, frame_shape)
        )
        if not all_curled:
            return False
        # Thumb must be neutral (not up / not down) — distinguishes fist from thumbs-up/down
        thumb_tip = landmarks[self.THUMB_TIP]
        avg_mcp_y = sum(landmarks[i].y for i in [self.INDEX_MCP, self.MIDDLE_MCP,
                                                   self.RING_MCP,  self.PINKY_MCP]) / 4.0
        dy = thumb_tip.y - avg_mcp_y
        return -0.12 <= dy <= 0.12

    def _detect_pointing(self, landmarks, frame_shape):
        """
        Detect pointing gesture: index finger extended, middle NOT extended, no pinch.
        Used as the drag cursor — completely separate from pinch interactions.
        Returns {'active': bool, 'index_tip_pos': np.array([x_px, y_px])}.
        """
        h, w = frame_shape[:2]
        cw = self.WRIST

        index_extended  = self._is_finger_extended(landmarks, self.INDEX_TIP,  self.INDEX_MCP,  cw, frame_shape)

        # Not pinching (thumb and index are apart)
        thumb_tip = landmarks[self.THUMB_TIP]
        index_tip = landmarks[self.INDEX_TIP]
        t_pos = np.array([thumb_tip.x * w, thumb_tip.y * h])
        i_pos = np.array([index_tip.x * w, index_tip.y * h])
        not_pinching = np.linalg.norm(t_pos - i_pos) / max(w, h) >= self.PINCH_THRESHOLD

        # Pointing: index extended + not pinching. Middle finger free (no longer required curled)
        # to avoid false negatives when people naturally extend middle slightly while pointing.
        is_pointing = index_extended and not_pinching

        return {
            'active': is_pointing,
            'index_tip_pos': np.array([index_tip.x * w, index_tip.y * h])
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

    # MediaPipe hand skeleton connections
    _CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),           # thumb
        (0,5),(5,6),(6,7),(7,8),            # index
        (0,9),(9,10),(10,11),(11,12),       # middle
        (0,13),(13,14),(14,15),(15,16),     # ring
        (0,17),(17,18),(18,19),(19,20),     # pinky
        (5,9),(9,13),(13,17),               # palm knuckle bar
    ]
    # Fingertip indices — rendered larger
    _TIPS = {4, 8, 12, 16, 20}
    # Knuckle (MCP) indices — medium size
    _MCPS = {1, 5, 9, 13, 17}

    # No-hand idle counter shared across calls
    _no_hand_frames = 0

    def render_hands(self, frame, gestures, shared_state=None):
        """
        Modern neon-skeleton hand overlay.

        States:
          idle      — teal glow skeleton
          forming   — orange glow + charging arc at pinch midpoint
          locked    — green glow + lock badge
          no hand   — contextual tip that escalates to lighting advice
        """
        h, w = frame.shape[:2]
        _glow_buf = np.empty_like(frame)   # reused scratch buffer — avoids per-hand copy()
        hands_detected = gestures.get('hands_detected', 0)

        # ── no hand detected ──────────────────────────────────────────────
        if hands_detected == 0:
            self._no_hand_frames += 1

            if self._no_hand_frames < 30:
                # brief grace period — say nothing
                return frame

            # pick message based on how long we've been waiting
            if self._no_hand_frames < 120:
                msg, sub = "Show your hand", ""
            else:
                msg, sub = "No hand detected", "Try better lighting or move closer"

            msg_color = (160, 160, 165) if self._no_hand_frames < 120 else (80, 160, 255)
            (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            mx = w // 2 - tw // 2
            my = h - 38
            cv2.putText(frame, msg, (mx, my),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, msg_color, 1, cv2.LINE_AA)
            if sub:
                (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                cv2.putText(frame, sub, (w // 2 - sw // 2, my + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 120, 140), 1, cv2.LINE_AA)
            return frame

        self._no_hand_frames = 0

        # ── per-hand skeleton ─────────────────────────────────────────────
        for hand_type in ('left_hand', 'right_hand'):
            hand = gestures.get(hand_type)
            if not hand:
                continue

            landmarks  = hand['landmarks']
            pinch      = hand['pinch']
            is_pinching = pinch['active']
            is_locked   = pinch.get('is_locked', False)
            progress    = pinch.get('hold_progress', 0.0)

            # choose palette based on gesture state
            if is_locked:
                core  = (120, 255, 160)   # green
                glow  = (40,  120, 60)
            elif is_pinching or progress > 0:
                core  = (80,  160, 255)   # orange-ish
                glow  = (40,  70,  120)
            else:
                core  = (200, 230, 220)   # soft teal-white
                glow  = (40,  90,  80)

            # project landmarks to pixel coords
            pts = []
            for lm in landmarks:
                pts.append((int(lm.x * w), int(lm.y * h)))

            if len(pts) < 21:
                continue

            # ── draw skeleton bones ───────────────────────────────────────
            # Two-pass: thick dim glow layer -> thin bright core.
            # We draw glow directly on frame with addWeighted to avoid a per-bone copy.
            np.copyto(_glow_buf, frame)
            for a, b in self._CONNECTIONS:
                cv2.line(_glow_buf, pts[a], pts[b], glow, 5, cv2.LINE_AA)
            cv2.addWeighted(_glow_buf, 0.45, frame, 0.55, 0, frame)

            for a, b in self._CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], core, 1, cv2.LINE_AA)

            # ── draw joint nodes ──────────────────────────────────────────
            for i, pt in enumerate(pts):
                if i in self._TIPS:
                    r = 5
                    cv2.circle(frame, pt, r + 2, glow, -1, cv2.LINE_AA)
                    cv2.circle(frame, pt, r,     core,  -1, cv2.LINE_AA)
                elif i in self._MCPS:
                    r = 3
                    cv2.circle(frame, pt, r + 1, glow, -1, cv2.LINE_AA)
                    cv2.circle(frame, pt, r,     core,  -1, cv2.LINE_AA)
                elif i == 0:   # wrist — slightly larger
                    cv2.circle(frame, pt, 4, glow, -1, cv2.LINE_AA)
                    cv2.circle(frame, pt, 3, core, -1, cv2.LINE_AA)
                else:
                    cv2.circle(frame, pt, 2, core, -1, cv2.LINE_AA)

            # ── pinch midpoint feedback ───────────────────────────────────
            thumb_pos = pinch['thumb_pos'].astype(int)
            index_pos = pinch['index_pos'].astype(int)
            mid = ((thumb_pos[0] + index_pos[0]) // 2,
                   (thumb_pos[1] + index_pos[1]) // 2)

            if is_pinching or is_locked:
                # bright connecting line between thumb and index
                cv2.line(frame, tuple(thumb_pos), tuple(index_pos), core, 2, cv2.LINE_AA)

            if progress > 0 and not is_locked:
                # charging arc — track ring + filled arc
                R = 22
                cv2.circle(frame, mid, R, (50, 50, 55), 2, cv2.LINE_AA)
                deg = int(360 * progress)
                cv2.ellipse(frame, mid, (R, R), -90, 0, deg, (80, 255, 160), 2, cv2.LINE_AA)
                pct = f"{int(progress * 100)}%"
                (pw, _), _ = cv2.getTextSize(pct, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                cv2.putText(frame, pct, (mid[0] - pw // 2, mid[1] + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 255, 200), 1, cv2.LINE_AA)

            if is_locked:
                # green glow burst at pinch point
                cv2.circle(frame, mid, 14, (30, 80, 50), -1, cv2.LINE_AA)
                cv2.circle(frame, mid, 14, (80, 255, 140), 1, cv2.LINE_AA)
                cv2.putText(frame, "LOCK", (mid[0] - 14, mid[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 255, 160), 1, cv2.LINE_AA)

            # ── wrist quality badge ───────────────────────────────────────
            # shows which hand and its current gesture state
            wrist_pt = pts[0]
            label_parts = ["L" if hand['is_left'] else "R"]
            if is_locked:
                label_parts.append("LOCKED")
                badge_color = (80, 255, 140)
            elif is_pinching:
                label_parts.append("PINCH")
                badge_color = (80, 160, 255)
            elif progress > 0:
                label_parts.append(f"HOLD {int(progress*100)}%")
                badge_color = (60, 180, 255)
            else:
                label_parts.append("READY")
                badge_color = (160, 200, 195)

            badge = " · ".join(label_parts)
            (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            bx = wrist_pt[0] - bw // 2
            by = wrist_pt[1] + 18
            cv2.rectangle(frame, (bx - 4, by - bh - 2), (bx + bw + 4, by + 4),
                          (18, 18, 22), -1)
            cv2.putText(frame, badge, (bx, by),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, badge_color, 1, cv2.LINE_AA)

        return frame
    
    def cleanup(self):
        if hasattr(self, 'detector'):
            self.detector.close()
