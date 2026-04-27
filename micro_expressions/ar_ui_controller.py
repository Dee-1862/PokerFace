"""
Apple-like AR UI Controller with Snap-to-Level Rotation
Clean, minimalistic interface with discrete level snapping.

Button 1 (Top-Left): Face Actions - snaps to 0%, 10%, 20%... 100% (controls expression count)
Button 2 (Top-Right): Heart/Stress - snaps to 0%, 50%, 100% (controls display detail)
"""
import cv2
import numpy as np
import math
import time
from collections import deque


class ARButton:
    """AR button with snap-to-level rotation."""
    
    def __init__(self, position, icon_type, snap_levels, radius=24):
        self.position = position
        self.icon_type = icon_type
        self.snap_levels = snap_levels  # List of valid levels (e.g., [0, 10, 20...100] or [0, 50, 100])
        self.radius = radius
        self.center = None
        
        # Rotation state - now represents actual percentage (0-100)
        self.level = 0  # Current snapped level
        self.raw_rotation = 0.0  # Raw accumulated rotation for smooth control
        self.last_gesture_angle = None
        self.gesture_active = False
        
        # Sensitivity: degrees of rotation needed to move 1% 
        self.sensitivity = 3.0  # 360 degrees = ~120% change
        
    def set_center(self, frame_width, frame_height, margin=35):
        """Calculate button center based on frame size."""
        if self.position == 'top-left':
            self.center = (margin + self.radius, margin + self.radius)
        elif self.position == 'top-right':
            self.center = (frame_width - margin - self.radius, margin + self.radius)
    
    def update_from_pinch_distance(self, distance):
        """Update level based on pinch distance (stretch)."""
        # Map distance range to [0, 100].
        # min_dist = just above pinch threshold (fingers still close after lock).
        # max_dist = comfortable fully-spread distance (~190px at 1280px wide).
        # Old max_dist was 0.25 (320px) which was unreachable for most people.
        min_dist = 0.07
        max_dist = 0.15
        
        # Normalize to 0-1
        normalized = (distance - min_dist) / (max_dist - min_dist)
        normalized = max(0.0, min(1.0, normalized))
        
        # Map to 0-100
        target_value = normalized * 100
        
        # Snap to nearest valid level
        self.raw_rotation = target_value
        self._snap_to_level()
    
    def _snap_to_level(self):
        """Snap raw_rotation to the nearest valid level."""
        # Find nearest snap level
        nearest = min(self.snap_levels, key=lambda x: abs(x - self.raw_rotation))
        self.level = nearest
    
    def get_expansion(self):
        """Get expansion as 0.0 to 1.0."""
        return self.level / 100.0
    
    def get_glow_color(self):
        """Get glow color based on level."""
        if self.level == 0:
            return (60, 60, 60)
        
        ratio = self.level / 100.0
        r = int(100 + ratio * 80)
        g = int(120 + ratio * 40)
        b = int(180 + ratio * 75)
        return (b, g, r)


class ARUIController:
    """
    Apple-like AR interface with Pinch-and-Stretch control.
    - Face Button: Stretch to set 0-100%
    - Heart Button: Stretch to set 0, 50, 100%
    """
    
    # Apple-style clear glass palette (BGR) - high-contrast text on translucent glass
    COLORS = {
        'bg_dark':         (10, 10, 14),     # deep shadow
        'bg_panel':        (44, 42, 50),     # tint wash
        'bg_panel_top':    (54, 52, 62),
        'bg_panel_bottom': (32, 30, 40),
        'panel_edge':      (240, 238, 248),  # bright glass rim (near white)
        'panel_border':    (155, 152, 172),  # visible border
        # Text - brighter, sharper for clear glass
        'text_primary':    (255, 255, 255),  # pure white
        'text_secondary':  (215, 213, 226),
        'text_tertiary':   (165, 162, 178),
        # Apple-system accent colours (BGR)
        'accent_blue':     (255, 200, 100),  # systemBlue
        'accent_cyan':     (245, 225, 130),  # systemTeal
        'accent_green':    (120, 230, 145),  # systemGreen
        'accent_red':      (90, 100, 255),   # systemRed
        'accent_orange':   (90, 170, 255),   # systemOrange
        'accent_yellow':   (90, 220, 255),   # systemYellow
        'accent_violet':   (240, 130, 200),  # systemPink/Purple
        'divider':         (90, 86, 104),
        'divider_soft':    (60, 58, 72),
    }

    STRESS_COLORS = {
        'very_low':  (140, 230, 130),   # soft green
        'low':       (160, 210, 100),
        'moderate':  (220, 210, 110),   # cyan-yellow
        'high':      (95, 165, 250),    # warm orange
        'very_high': (95, 95, 245),     # red
    }
    
    # All facial expressions
    ALL_EXPRESSIONS = [
        ('AU12', 'Smile'),
        ('AU4', 'Frown'),
        ('AU1', 'Brow Raise'),
        ('AU5', 'Eye Wide'),
        ('AU6', 'Cheek Raise'),
        ('AU15', 'Lip Down'),
        ('AU23', 'Lip Tight'),
        ('AU26', 'Jaw Open'),
        ('AU45', 'Blink'),
        ('AU2', 'Outer Brow'),
    ]
    
    def __init__(self, button_position='top-right'):
        # Face button: snaps to 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
        face_levels = list(range(0, 110, 10))  # [0, 10, 20, ..., 100]
        self.face_button = ARButton('top-left', 'face', face_levels, radius=22)

        # Heart button: snaps to 0, 50, 100
        heart_levels = [0, 50, 100]
        self.heart_button = ARButton('top-right', 'heart', heart_levels, radius=22)

        self.active_button = None

        # Cached data
        self.stress_data = {
            'score': 0.0,
            'category': 'Unknown',
            'trend': 'Stable',
            'type': 'None',
            'heart_rate': 0,
        }
        self.action_units = {}

        # Heart rate history for graph
        self.hr_history = deque(maxlen=120)

        # Store pinch positions for percentage display
        self.left_pinch_pos = None
        self.right_pinch_pos = None

        # Panel position manager (injected from outside)
        self._pm = None

        # Last-rendered rects for drag detection: (x, y, w, h)
        self.hr_analytics_rect = None
        self.hr_graph_rect = None
        self.expressions_rect = None

        # Smoothed face bbox so panels can anchor to the face and follow it.
        # Populated each frame from shared_state['landmarks'].
        self._face_bbox_ema = None  # (x, y, w, h) floats
        self._face_bbox_alpha = 0.18  # EMA factor (lower = smoother but laggier)

    def set_positions(self, pm) -> None:
        """Inject a PanelPositionManager so panels use saved positions."""
        self._pm = pm

    def _update_face_bbox(self, landmarks, w, h):
        """EMA-smooth the face bounding box so panels follow the face without jitter."""
        if not landmarks:
            return
        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]
        bx = max(0, min(xs))
        by = max(0, min(ys))
        bw = max(1, max(xs) - bx)
        bh = max(1, max(ys) - by)
        if self._face_bbox_ema is None:
            self._face_bbox_ema = (bx, by, bw, bh)
            return
        a = self._face_bbox_alpha
        ex, ey, ew, eh = self._face_bbox_ema
        self._face_bbox_ema = (
            (1 - a) * ex + a * bx,
            (1 - a) * ey + a * by,
            (1 - a) * ew + a * bw,
            (1 - a) * eh + a * bh,
        )

    def _face_bbox(self):
        """Return integer (x, y, w, h) face bbox or None."""
        if self._face_bbox_ema is None:
            return None
        return tuple(int(v) for v in self._face_bbox_ema)
        
    def process_hand_gestures(self, shared_state):
        """Process hand gestures for AR interaction (Pinch-and-Stretch)."""
        gestures = shared_state.get('hand_gestures', {})
        
        self.left_pinch_pos = None
        self.right_pinch_pos = None
        
        if not gestures or gestures.get('hands_detected', 0) == 0:
            self.active_button = None
            return
        
        left_hand = gestures.get('left_hand')
        right_hand = gestures.get('right_hand')
        
        # Left Hand -> Face Button
        if left_hand:
            pinch = left_hand['pinch']
            is_locked = pinch.get('is_locked', False)
            
            # If locked (held for 4s), use distance to set level
            if is_locked:
                distance = pinch.get('distance', 0)
                self.face_button.update_from_pinch_distance(distance)
                self.active_button = 'face'
            
            # Track position for feedback
            if pinch['active'] or is_locked:
                self.left_pinch_pos = (pinch['thumb_pos'] + pinch['index_pos']) / 2

        # Right Hand -> Heart Button
        if right_hand:
            pinch = right_hand['pinch']
            is_locked = pinch.get('is_locked', False)
            
            if is_locked:
                distance = pinch.get('distance', 0)
                self.heart_button.update_from_pinch_distance(distance)
                self.active_button = 'heart'
            
            if pinch['active'] or is_locked:
                self.right_pinch_pos = (pinch['thumb_pos'] + pinch['index_pos']) / 2
    
    def update_data(self, shared_state):
        """Update cached data from shared state."""
        self.stress_data = {
            'score': shared_state.get('stress_score', 0.0),
            'category': shared_state.get('stress_category', 'Unknown'),
            'trend': shared_state.get('stress_trend', 'Stable'),
            'type': shared_state.get('stress_type', 'None'),
            'heart_rate': shared_state.get('heart_rate_bpm', 0),
        }
        
        hr = self.stress_data['heart_rate']
        if hr > 0:
            self.hr_history.append(hr)
        
        self.action_units = shared_state.get('action_units', {})
        
        # Store button levels in shared_state for hand renderer
        shared_state['face_button_level'] = self.face_button.level
        shared_state['heart_button_level'] = self.heart_button.level
        shared_state['left_pinch_pos'] = self.left_pinch_pos
        shared_state['right_pinch_pos'] = self.right_pinch_pos
    
    def render(self, frame, shared_state):
        """Render the Apple-like AR UI."""
        h, w = frame.shape[:2]

        self.face_button.set_center(w, h)
        self.heart_button.set_center(w, h)

        self.process_hand_gestures(shared_state)
        self.update_data(shared_state)
        self._update_face_bbox(shared_state.get('landmarks'), w, h)
        
        # Render bottom graph only at 100%
        if self.heart_button.level >= 100:
            self._render_bottom_graph(frame, w, h)
        
        # Render face button and panel
        self._render_face_button(frame)
        
        # Render heart button and side panel
        self._render_heart_button(frame, w, h)
        
        return frame
    
    def _render_face_button(self, frame):
        """Render face button with level indicator."""
        button = self.face_button
        cx, cy = button.center
        glow_color = button.get_glow_color()

        self._draw_circle_button(frame, cx, cy, button.radius, glow_color, button.level > 0)

        # Face icon
        self._draw_face_icon(frame, cx, cy, button.radius - 8)

        # Level arc indicator
        if button.level > 0:
            self._draw_level_arc(frame, cx, cy, button.radius + 4, button.level)

        # Level badge below button
        self._draw_level_badge(frame, cx, cy + button.radius + 18, button.level)

        # Render expressions panel
        if button.level > 0:
            self._render_expressions_panel(frame, button)

    def _draw_circle_button(self, frame, cx, cy, radius, accent_color, active):
        """Apple-style frosted glass orb: heavy backdrop blur, neutral tint,
        bright rim, top specular highlight, soft drop shadow, optional glow."""
        h_f, w_f = frame.shape[:2]

        # ---- 1. Soft drop shadow (blurred circle below) ----
        sh_r = radius + 4
        sx0 = max(0, cx - sh_r - 4); sy0 = max(0, cy + 6 - sh_r - 4)
        sx1 = min(w_f, cx + sh_r + 5); sy1 = min(h_f, cy + 6 + sh_r + 5)
        if sx1 > sx0 and sy1 > sy0:
            shmask_h, shmask_w = sy1 - sy0, sx1 - sx0
            shmask = np.zeros((shmask_h, shmask_w), dtype=np.uint8)
            cv2.circle(shmask, (cx - sx0, cy + 6 - sy0), sh_r, 255, -1, cv2.LINE_AA)
            shmask = cv2.GaussianBlur(shmask, (21, 21), 0)
            sr = frame[sy0:sy1, sx0:sx1]
            dark = np.full_like(sr, self.COLORS['bg_dark'])
            a = (shmask.astype(np.float32) / 255.0 * 0.55)[..., None]
            sr[:] = (sr.astype(np.float32) * (1 - a) + dark.astype(np.float32) * a).astype(np.uint8)

        # ---- 2. Outer glow (only when active) ----
        if active:
            for r_off, ga in ((14, 0.08), (8, 0.14), (4, 0.20)):
                glow_r = radius + r_off
                x0 = max(0, cx - glow_r - 1); y0 = max(0, cy - glow_r - 1)
                x1 = min(w_f, cx + glow_r + 2); y1 = min(h_f, cy + glow_r + 2)
                if x1 <= x0 or y1 <= y0:
                    continue
                region = frame[y0:y1, x0:x1].copy()
                cv2.circle(region, (cx - x0, cy - y0), glow_r,
                           accent_color, -1, cv2.LINE_AA)
                cv2.addWeighted(region, ga, frame[y0:y1, x0:x1], 1 - ga, 0,
                                frame[y0:y1, x0:x1])

        # ---- 3. Clear glass body (almost no blur, whisper tint) ----
        x0 = max(0, cx - radius); y0 = max(0, cy - radius)
        x1 = min(w_f, cx + radius + 1); y1 = min(h_f, cy + radius + 1)
        if x1 > x0 and y1 > y0:
            region = frame[y0:y1, x0:x1]
            blurred = cv2.GaussianBlur(region, (5, 5), 0)
            tint = np.full_like(region, self.COLORS['bg_panel'])
            tinted = (blurred.astype(np.float32) * 0.85 + tint.astype(np.float32) * 0.15)
            tinted = np.clip(tinted, 0, 255).astype(np.uint8)

            # Vertical sheen for spherical illusion
            h_r, w_r = region.shape[:2]
            ramp = np.linspace(1.10, 0.88, h_r, dtype=np.float32).reshape(h_r, 1, 1)
            sheened = np.clip(tinted.astype(np.float32) * ramp, 0, 255).astype(np.uint8)
            composite = cv2.addWeighted(sheened, 0.55, region, 0.45, 0)

            mask = np.zeros((h_r, w_r), dtype=np.uint8)
            cv2.circle(mask, (cx - x0, cy - y0), radius, 255, -1, cv2.LINE_AA)
            mb = mask.astype(bool)
            np.copyto(region, composite, where=mb[..., None])

            # ---- 4. Specular highlight (small bright arc top-left) ----
            spec_layer = np.zeros_like(region, dtype=np.float32)
            cv2.ellipse(spec_layer, (cx - x0 - 2, cy - y0 - 2),
                        (radius - 4, radius - 7), 0, 200, 320,
                        self.COLORS['panel_edge'], 2, cv2.LINE_AA)
            spec_blur = cv2.GaussianBlur(spec_layer, (5, 5), 0)
            region[:] = np.clip(region.astype(np.float32) + spec_blur * 0.7, 0, 255).astype(np.uint8)

        # ---- 5. Outer ring in accent colour ----
        cv2.circle(frame, (cx, cy), radius, accent_color, 2, cv2.LINE_AA)
        # Faint inner highlight ring
        cv2.circle(frame, (cx, cy), radius - 1, self.COLORS['panel_edge'], 1, cv2.LINE_AA)
    
    def _draw_face_icon(self, frame, cx, cy, size):
        """Draw minimalist face icon."""
        color = self.COLORS['text_secondary']
        cv2.circle(frame, (cx, cy), size, color, 1)
        eye_y = cy - size // 4
        eye_offset = size // 3
        cv2.circle(frame, (cx - eye_offset, eye_y), 2, color, -1)
        cv2.circle(frame, (cx + eye_offset, eye_y), 2, color, -1)
        smile_y = cy + size // 4
        cv2.ellipse(frame, (cx, smile_y), (size // 3, size // 6), 0, 0, 180, color, 1)
    
    def _draw_level_arc(self, frame, cx, cy, radius, level):
        """Draw arc showing current level (0-100%) with track + filled progress."""
        # Track ring (very faint)
        cv2.ellipse(frame, (cx, cy), (radius, radius), -90, 0, 360,
                    self.COLORS['divider_soft'], 2, cv2.LINE_AA)
        # Filled progress
        arc_angle = int(360 * level / 100)
        if arc_angle > 0:
            cv2.ellipse(frame, (cx, cy), (radius, radius), -90, 0, arc_angle,
                        self.COLORS['accent_cyan'], 2, cv2.LINE_AA)

    def _draw_level_badge(self, frame, cx, cy, level):
        """Compact frosted pill with the percentage value."""
        text = f"{level}%"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)

        pill_w = text_w + 18
        pill_h = text_h + 10
        pill_x = cx - pill_w // 2
        pill_y = cy - pill_h // 2

        self._draw_glass_panel(frame, pill_x, pill_y, pill_w, pill_h,
                               alpha=0.7, radius=pill_h // 2)

        cv2.putText(frame, text, (pill_x + 9, pill_y + pill_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    self.COLORS['text_primary'], 1, cv2.LINE_AA)
    
    def _render_expressions_panel(self, frame, button):
        """Render expressions based on level (10% = 1 expr, 100% = 10 expr)."""
        if not self.action_units:
            return
        
        cx, cy = button.center

        # Number of expressions = level / 10 (10% = 1, 100% = 10)
        num_expressions = button.level // 10
        if num_expressions == 0:
            return

        visible_expressions = self.ALL_EXPRESSIONS[:num_expressions]

        # Panel dimensions
        panel_width = 160
        row_height = 24
        panel_height = 28 + len(visible_expressions) * row_height

        # Anchor to face bbox (LEFT of the face) so the panel sticks to the head as it moves.
        # Falls back to button-relative position when no face is detected.
        h_f, w_f = frame.shape[:2]
        fbbox = self._face_bbox()
        if fbbox is not None:
            fbx, fby, fbw, fbh = fbbox
            panel_x = max(10, fbx - panel_width - 24)
            panel_y = max(10, min(h_f - panel_height - 10,
                                  fby + (fbh - panel_height) // 2))
        else:
            default_x = max(10, cx - 10)
            default_y = cy + button.radius + 40
            panel_x, panel_y = default_x, default_y

        # Record rect for drag detection
        self.expressions_rect = (panel_x, panel_y, panel_width, panel_height)

        # Glass panel
        self._draw_glass_panel(frame, panel_x, panel_y, panel_width, panel_height,
                               accent=self.COLORS['accent_violet'])

        _radius = 18  # match _draw_glass_panel default
        # Title - clipped so it never escapes the rounded corners
        title = f"EXPRESSIONS ({num_expressions})"
        self._put_text_clipped(frame, title, (panel_x + 14, panel_y + 18),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                               self.COLORS['text_secondary'], 1,
                               clip_x=panel_x, clip_y=panel_y,
                               clip_w=panel_width, clip_h=panel_height,
                               clip_radius=_radius)

        # Divider (kept inside the rounded region by the radius padding)
        cv2.line(frame, (panel_x + 14, panel_y + 24),
                 (panel_x + panel_width - 14, panel_y + 24),
                 self.COLORS['divider_soft'], 1, cv2.LINE_AA)
        
        # Expressions
        y_offset = panel_y + 40
        for au_code, name in visible_expressions:
            value = self.action_units.get(au_code, 0)
            self._draw_expression_row(frame, panel_x + 14, y_offset,
                                      panel_width - 28, name, value,
                                      clip=(panel_x, panel_y, panel_width, panel_height, _radius))
            y_offset += row_height
    
    def _draw_expression_row(self, frame, x, y, width, name, value, clip=None):
        """Single expression row: name + rounded gradient progress bar + percentage."""
        name_color = self.COLORS['text_primary'] if value > 0.3 else self.COLORS['text_secondary']
        if clip is not None:
            cx, cy, cw, ch, cr = clip
            self._put_text_clipped(frame, name, (x, y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.34, name_color, 1,
                                   clip_x=cx, clip_y=cy, clip_w=cw, clip_h=ch,
                                   clip_radius=cr)
        else:
            cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                        name_color, 1, cv2.LINE_AA)

        bar_width  = 48
        bar_height = 6
        bar_x = x + width - bar_width - 30
        bar_y = y - 5
        radius = bar_height // 2

        # Rounded track
        self._draw_rounded_rect(frame, bar_x, bar_y, bar_width, bar_height, radius,
                                self.COLORS['divider_soft'], thickness=-1)

        # Rounded filled portion (Apple system colours)
        fill_width = int(bar_width * min(value, 1.0))
        if fill_width > radius:
            if value > 0.5:
                hi = self.COLORS['accent_green']
            elif value > 0.25:
                hi = self.COLORS['accent_blue']
            else:
                hi = self.COLORS['text_secondary']
            lo = (hi[0] // 2, hi[1] // 2, hi[2] // 2)

            h_f, w_f = frame.shape[:2]
            x0 = max(0, bar_x); y0 = max(0, bar_y)
            x1 = min(w_f, bar_x + fill_width); y1 = min(h_f, bar_y + bar_height)
            if x1 > x0 and y1 > y0:
                grad = self._horizontal_gradient(y1 - y0, x1 - x0, lo, hi)
                fill_mask = self._rounded_rect_mask(y1 - y0, x1 - x0, radius)
                region = frame[y0:y1, x0:x1]
                fm = fill_mask.astype(bool)
                np.copyto(region, grad, where=fm[..., None])

        pct_text = f"{value:.0%}"
        if clip is not None:
            cx, cy, cw, ch, cr = clip
            self._put_text_clipped(frame, pct_text, (bar_x + bar_width + 4, y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.30,
                                   self.COLORS['text_secondary'], 1,
                                   clip_x=cx, clip_y=cy, clip_w=cw, clip_h=ch,
                                   clip_radius=cr)
        else:
            cv2.putText(frame, pct_text, (bar_x + bar_width + 4, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30,
                        self.COLORS['text_secondary'], 1, cv2.LINE_AA)
    
    def _render_heart_button(self, frame, frame_w, frame_h):
        """Render heart button with level indicator."""
        button = self.heart_button
        cx, cy = button.center
        glow_color = button.get_glow_color()

        self._draw_circle_button(frame, cx, cy, button.radius, glow_color, button.level > 0)

        # Heart icon
        self._draw_heart_icon(frame, cx, cy, button.radius - 8)

        # Level arc indicator
        if button.level > 0:
            self._draw_level_arc(frame, cx, cy, button.radius + 4, button.level)

        # Level badge below button
        self._draw_level_badge(frame, cx, cy + button.radius + 18, button.level)

        # Side panel (shows at 50% or 100%)
        if button.level >= 50:
            self._render_analytics_side_panel(frame, button, frame_w)
    
    def _draw_heart_icon(self, frame, cx, cy, size):
        """Draw heart icon."""
        hr = self.stress_data['heart_rate']
        if hr > 0:
            color = self.STRESS_COLORS['high'] if hr > 100 else self.STRESS_COLORS['moderate'] if hr > 80 else self.STRESS_COLORS['low']
        else:
            color = self.COLORS['text_secondary']
        
        scale = size / 10
        heart_coords = [(0, 3), (-5, -2), (-5, -5), (-3, -7), (0, -5), (3, -7), (5, -5), (5, -2), (0, 3)]
        points = np.array([[int(cx + hx * scale), int(cy + hy * scale)] for hx, hy in heart_coords], np.int32)
        cv2.fillPoly(frame, [points], color)
    
    def _render_analytics_side_panel(self, frame, button, frame_w):
        """Render HR and stress analytics anchored to the upper-right of the face."""
        panel_width = 175
        # Panel height depends on level
        panel_height = 110 if button.level >= 50 else 60

        h_f = frame.shape[0]
        fbbox = self._face_bbox()
        if fbbox is not None:
            fbx, fby, fbw, fbh = fbbox
            # Right of the face, anchored to its TOP edge (opponent panel takes the lower half)
            panel_x = min(frame_w - panel_width - 10, fbx + fbw + 24)
            panel_y = max(10, min(h_f - panel_height - 10, fby - 10))
        else:
            panel_x = frame_w - panel_width - 15
            panel_y = button.center[1] + button.radius + 40

        # Record rect for drag detection
        self.hr_analytics_rect = (panel_x, panel_y, panel_width, panel_height)

        self._draw_glass_panel(frame, panel_x, panel_y, panel_width, panel_height,
                               accent=self.COLORS['accent_red'])

        _radius = 18
        _clip = (panel_x, panel_y, panel_width, panel_height, _radius)

        y_offset = panel_y + 14

        # Heart Rate section
        self._put_text_clipped(frame, "HEART RATE", (panel_x + 16, y_offset + 6),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                               self.COLORS['text_tertiary'], 1,
                               clip_x=_clip[0], clip_y=_clip[1],
                               clip_w=_clip[2], clip_h=_clip[3], clip_radius=_clip[4])
        
        hr = self.stress_data['heart_rate']
        hr_text = f"{hr}" if hr > 0 else "--"
        hr_color = self._get_hr_color(hr)
        # Soft glow behind the number for visual depth
        self._put_text_clipped(frame, hr_text, (panel_x + 17, y_offset + 33),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                               (hr_color[0] // 4, hr_color[1] // 4, hr_color[2] // 4),
                               3,
                               clip_x=_clip[0], clip_y=_clip[1],
                               clip_w=_clip[2], clip_h=_clip[3], clip_radius=_clip[4])
        self._put_text_clipped(frame, hr_text, (panel_x + 16, y_offset + 32),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, hr_color, 2,
                               clip_x=_clip[0], clip_y=_clip[1],
                               clip_w=_clip[2], clip_h=_clip[3], clip_radius=_clip[4])
        self._put_text_clipped(frame, "BPM", (panel_x + 16 + len(hr_text) * 18, y_offset + 32),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                               self.COLORS['text_tertiary'], 1,
                               clip_x=_clip[0], clip_y=_clip[1],
                               clip_w=_clip[2], clip_h=_clip[3], clip_radius=_clip[4])

        y_offset += 45

        # Divider
        cv2.line(frame, (panel_x + 16, y_offset), (panel_x + panel_width - 16, y_offset),
                 self.COLORS['divider_soft'], 1, cv2.LINE_AA)
        y_offset += 8

        # Stress section
        self._put_text_clipped(frame, "STRESS", (panel_x + 16, y_offset + 6),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                               self.COLORS['text_tertiary'], 1,
                               clip_x=_clip[0], clip_y=_clip[1],
                               clip_w=_clip[2], clip_h=_clip[3], clip_radius=_clip[4])

        score = self.stress_data['score']
        category = self.stress_data['category']
        stress_color = self._get_stress_color(score)

        self._put_text_clipped(frame, category, (panel_x + 16, y_offset + 28),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, stress_color, 1,
                               clip_x=_clip[0], clip_y=_clip[1],
                               clip_w=_clip[2], clip_h=_clip[3], clip_radius=_clip[4])

        score_text = f"{score:.0%}"
        (score_w, _), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(frame, score_text, (panel_x + panel_width - score_w - 12, y_offset + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, stress_color, 1, cv2.LINE_AA)
    
    def _render_bottom_graph(self, frame, frame_w, frame_h):
        """Render full-width HR graph — stable Y axis, raised above mini-HUD zone."""
        if len(self.hr_history) < 2:
            return

        graph_height = 60
        graph_width  = frame_w - 40
        default_x    = 20
        default_y    = frame_h - graph_height - 96

        if self._pm:
            graph_x, graph_y = self._pm.get('hr_graph', default_x, default_y)
        else:
            graph_x, graph_y = default_x, default_y

        # Record rect for drag detection
        self.hr_graph_rect = (graph_x, graph_y, graph_width, graph_height)

        # Background glass panel (with shadow + gradient + accent edge)
        self._draw_glass_panel(frame, graph_x, graph_y, graph_width, graph_height,
                               accent=self.COLORS['accent_red'])

        # Header
        cv2.putText(frame, "HEART RATE", (graph_x + 10, graph_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    self.COLORS['text_tertiary'], 1, cv2.LINE_AA)
        hr = self.stress_data['heart_rate']
        if hr > 0:
            hr_text = f"{hr} BPM"
            (hr_w, _), _ = cv2.getTextSize(hr_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            hr_col = self._get_hr_color(hr)
            # subtle text glow
            cv2.putText(frame, hr_text,
                        (graph_x + graph_width - hr_w - 9, graph_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (hr_col[0] // 4, hr_col[1] // 4, hr_col[2] // 4),
                        2, cv2.LINE_AA)
            cv2.putText(frame, hr_text,
                        (graph_x + graph_width - hr_w - 10, graph_y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, hr_col, 1, cv2.LINE_AA)

        history = list(self.hr_history)
        graph_area_y = graph_y + 18
        graph_area_h = graph_height - 22

        # Stable Y range
        mean_hr = sum(history) / len(history)
        span    = max(20, max(history) - min(history) + 10)
        lo      = max(30, mean_hr - span / 2)
        hi      = lo + span

        # Refined gridlines (dotted look via short segments)
        for frac in (0.0, 0.5, 1.0):
            gy  = int(graph_area_y + graph_area_h * (1 - frac))
            bpm = int(lo + (hi - lo) * frac)
            x_start = graph_x + 38
            x_end   = graph_x + graph_width - 5
            grid_col = (52, 50, 62) if frac == 0.5 else (44, 42, 54)
            # Dashed grid: 6px on, 6px off
            for gx in range(x_start, x_end, 12):
                cv2.line(frame, (gx, gy), (min(gx + 6, x_end), gy),
                         grid_col, 1, cv2.LINE_AA)
            cv2.putText(frame, str(bpm), (graph_x + 5, gy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.26,
                        self.COLORS['text_tertiary'], 1, cv2.LINE_AA)

        # Plot points
        pts = []
        for i, v in enumerate(history):
            px = int(graph_x + 42 + (i / max(len(history) - 1, 1))
                     * (graph_width - 52))
            norm = max(0.0, min(1.0, (v - lo) / (hi - lo)))
            py   = int(graph_area_y + graph_area_h * (1 - norm))
            pts.append((px, py))

        # Soft area fill under the line (gradient transparency)
        if len(pts) >= 2:
            poly = pts + [
                (pts[-1][0], graph_area_y + graph_area_h),
                (pts[0][0],  graph_area_y + graph_area_h),
            ]
            poly_arr = np.array(poly, dtype=np.int32)
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [poly_arr], 255)

            h_f, w_f = frame.shape[:2]
            xs = [p[0] for p in pts]
            x0, x1 = max(0, min(xs)), min(w_f, max(xs) + 1)
            y0, y1 = max(0, graph_area_y), min(h_f, graph_area_y + graph_area_h)
            if x1 > x0 and y1 > y0:
                col = self._get_hr_color(history[-1])
                top_col    = (col[0] // 2, col[1] // 2, col[2] // 2)
                bot_col    = self.COLORS['bg_panel_bottom']
                area_grad  = self._vertical_gradient(y1 - y0, x1 - x0, top_col, bot_col)
                region     = frame[y0:y1, x0:x1]
                local_mask = mask[y0:y1, x0:x1]
                blend = cv2.addWeighted(area_grad, 0.42, region, 0.58, 0)
                np.copyto(region, blend, where=local_mask[..., None].astype(bool))

        # Plot line on top
        for i in range(len(pts) - 1):
            cv2.line(frame, pts[i], pts[i + 1],
                     self._get_hr_color(history[i]), 2, cv2.LINE_AA)

        # Pulse dot at the end (with glow ring)
        if pts:
            end_col = self._get_hr_color(history[-1])
            cv2.circle(frame, pts[-1], 7,
                       (end_col[0] // 3, end_col[1] // 3, end_col[2] // 3),
                       -1, cv2.LINE_AA)
            cv2.circle(frame, pts[-1], 4, end_col, -1, cv2.LINE_AA)
            cv2.circle(frame, pts[-1], 2,
                       self.COLORS['text_primary'], -1, cv2.LINE_AA)
    
    def _get_hr_color(self, hr):
        """Get color for heart rate value."""
        if hr > 100:
            return self.STRESS_COLORS['high']
        elif hr > 80:
            return self.STRESS_COLORS['moderate']
        return self.STRESS_COLORS['low']
    
    @staticmethod
    def _put_text_clipped(frame, text, org, font, scale, color, thickness=1,
                          line_type=cv2.LINE_AA,
                          clip_x=None, clip_y=None, clip_w=None, clip_h=None,
                          clip_radius=0):
        """Render text but clip to a rounded-rectangle area so it never leaks past
        the panel border. Falls back to plain putText if no clip is given."""
        if clip_x is None:
            cv2.putText(frame, text, org, font, scale, color, thickness, line_type)
            return

        h_f, w_f = frame.shape[:2]
        cx0 = max(0, clip_x); cy0 = max(0, clip_y)
        cx1 = min(w_f, clip_x + clip_w); cy1 = min(h_f, clip_y + clip_h)
        if cx1 <= cx0 or cy1 <= cy0:
            return

        # Render text into a temp buffer the size of the clipping region,
        # offset by where the text would have appeared inside the region.
        tx = org[0] - cx0
        ty = org[1] - cy0
        layer = np.zeros((cy1 - cy0, cx1 - cx0, 3), dtype=np.uint8)
        cv2.putText(layer, text, (tx, ty), font, scale, color, thickness, line_type)

        # Build a clipping mask: rounded region (in clip coords)
        rx0 = clip_x - cx0; ry0 = clip_y - cy0
        rmask_full = ARUIController._rounded_rect_mask(clip_h, clip_w, clip_radius)
        rmask = np.zeros((cy1 - cy0, cx1 - cx0), dtype=np.uint8)
        # Copy rmask_full into rmask at (ry0, rx0)
        ix0 = max(0, rx0); iy0 = max(0, ry0)
        ix1 = min(rmask.shape[1], rx0 + clip_w)
        iy1 = min(rmask.shape[0], ry0 + clip_h)
        if ix1 > ix0 and iy1 > iy0:
            sx = ix0 - rx0; sy = iy0 - ry0
            rmask[iy0:iy1, ix0:ix1] = rmask_full[sy:sy + (iy1 - iy0),
                                                 sx:sx + (ix1 - ix0)]

        # Composite: where layer has ink AND mask is set, copy layer onto frame
        ink = (layer.sum(axis=-1) > 0)
        clip_ok = rmask.astype(bool)
        composite_mask = ink & clip_ok
        target = frame[cy0:cy1, cx0:cx1]
        np.copyto(target, layer, where=composite_mask[..., None])

    @staticmethod
    def _rounded_rect_mask(h, w, radius):
        """Return a (h, w) uint8 mask with rounded corners (255 inside, 0 outside)."""
        radius = max(0, min(radius, h // 2, w // 2))
        mask = np.zeros((h, w), dtype=np.uint8)
        if radius == 0:
            mask[:] = 255
            return mask
        cv2.rectangle(mask, (radius, 0), (w - radius, h), 255, -1)
        cv2.rectangle(mask, (0, radius), (w, h - radius), 255, -1)
        cv2.circle(mask, (radius, radius), radius, 255, -1, cv2.LINE_AA)
        cv2.circle(mask, (w - radius - 1, radius), radius, 255, -1, cv2.LINE_AA)
        cv2.circle(mask, (radius, h - radius - 1), radius, 255, -1, cv2.LINE_AA)
        cv2.circle(mask, (w - radius - 1, h - radius - 1), radius, 255, -1, cv2.LINE_AA)
        return mask

    @staticmethod
    def _draw_rounded_rect(frame, x, y, width, height, radius, color, thickness=1):
        """Stroke a rounded rectangle using 4 lines + 4 arcs (anti-aliased)."""
        radius = max(0, min(radius, height // 2, width // 2))
        if thickness < 0:
            cv2.rectangle(frame, (x + radius, y), (x + width - radius, y + height),
                          color, -1)
            cv2.rectangle(frame, (x, y + radius), (x + width, y + height - radius),
                          color, -1)
            cv2.circle(frame, (x + radius, y + radius), radius, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (x + width - radius, y + radius), radius, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (x + radius, y + height - radius), radius, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (x + width - radius, y + height - radius), radius, color, -1, cv2.LINE_AA)
            return
        cv2.line(frame, (x + radius, y), (x + width - radius, y),
                 color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x + radius, y + height), (x + width - radius, y + height),
                 color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x, y + radius), (x, y + height - radius),
                 color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x + width, y + radius), (x + width, y + height - radius),
                 color, thickness, cv2.LINE_AA)
        cv2.ellipse(frame, (x + radius, y + radius), (radius, radius),
                    180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(frame, (x + width - radius, y + radius), (radius, radius),
                    270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(frame, (x + radius, y + height - radius), (radius, radius),
                    90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(frame, (x + width - radius, y + height - radius), (radius, radius),
                    0, 0, 90, color, thickness, cv2.LINE_AA)

    def _draw_glass_panel(self, frame, x, y, width, height,
                          alpha=0.40, accent=None, radius=18):
        """Apple Vision Pro-style clear glass panel.

        Subtle blur (camera content stays recognisable through the glass),
        light neutral tint, full-edge rim light, top specular highlight,
        soft rounded drop shadow.
        """
        h_f, w_f = frame.shape[:2]
        radius = max(0, min(radius, height // 2, width // 2))

        # ---- 1. Drop shadow: large soft offset, blurred ----
        sh_off_x, sh_off_y = 3, 8
        sh_pad = 14
        sx0 = max(0, x + sh_off_x - sh_pad); sy0 = max(0, y + sh_off_y - sh_pad)
        sx1 = min(w_f, x + width + sh_off_x + sh_pad)
        sy1 = min(h_f, y + height + sh_off_y + sh_pad)
        if sx1 > sx0 and sy1 > sy0:
            shmask_h, shmask_w = sy1 - sy0, sx1 - sx0
            shmask = np.zeros((shmask_h, shmask_w), dtype=np.uint8)
            shx = (x + sh_off_x) - sx0
            shy = (y + sh_off_y) - sy0
            inner = self._rounded_rect_mask(height, width, radius + 4)
            iy0 = max(0, shy); ix0 = max(0, shx)
            iy1 = min(shmask_h, shy + height); ix1 = min(shmask_w, shx + width)
            src_y0 = iy0 - shy; src_x0 = ix0 - shx
            src_y1 = src_y0 + (iy1 - iy0); src_x1 = src_x0 + (ix1 - ix0)
            if iy1 > iy0 and ix1 > ix0:
                shmask[iy0:iy1, ix0:ix1] = inner[src_y0:src_y1, src_x0:src_x1]
            shmask = cv2.GaussianBlur(shmask, (29, 29), 0)
            sr = frame[sy0:sy1, sx0:sx1]
            dark = np.full_like(sr, self.COLORS['bg_dark'])
            a = (shmask.astype(np.float32) / 255.0 * 0.50)[..., None]
            sr[:] = (sr.astype(np.float32) * (1 - a) + dark.astype(np.float32) * a).astype(np.uint8)

        # ---- 2. Body region with subtle blur ----
        x0 = max(0, x); y0 = max(0, y)
        x1 = min(w_f, x + width); y1 = min(h_f, y + height)
        if x1 <= x0 or y1 <= y0:
            return
        h, w = y1 - y0, x1 - x0
        region = frame[y0:y1, x0:x1]

        # Almost no blur - just 5px to soften pixel-level noise. Camera stays clear.
        blurred = cv2.GaussianBlur(region, (5, 5), 0)

        # ---- 3. Whisper-thin tint (camera dominates) ----
        # blur 88%, tint 12% - panel feels like clean glass with a cool wash
        tint = np.array(self.COLORS['bg_panel'], dtype=np.float32)
        tinted = blurred.astype(np.float32) * 0.88 + tint * 0.12
        tinted = np.clip(tinted, 0, 255).astype(np.uint8)

        # Subtle vertical sheen (brighter top, slightly darker bottom)
        ramp = np.linspace(1.06, 0.94, h, dtype=np.float32).reshape(h, 1, 1)
        sheened = np.clip(tinted.astype(np.float32) * ramp, 0, 255).astype(np.uint8)

        # Composite at lower alpha so MORE of the live camera shows through
        composite = cv2.addWeighted(sheened, alpha, region, 1 - alpha, 0)

        # ---- 4. Rounded mask ----
        mask = self._rounded_rect_mask(h, w, radius)
        mask_bool = mask.astype(bool)
        np.copyto(region, composite, where=mask_bool[..., None])

        # ---- 5. Bright rim light around the FULL rounded edge ----
        # Two-pass: outer (faint, 2px) + inner (sharp, 1px) for an Apple-style edge
        rim_outer = np.zeros_like(region)
        self._draw_rounded_rect(rim_outer, 1, 1, w - 2, h - 2, max(1, radius - 1),
                                self.COLORS['panel_edge'], thickness=1)
        rim_a = (rim_outer.sum(axis=-1) > 0).astype(np.float32) * 0.45
        rim_a = (rim_a * (mask.astype(np.float32) / 255.0))[..., None]
        region[:] = (region.astype(np.float32) * (1 - rim_a) +
                     rim_outer.astype(np.float32) * rim_a).astype(np.uint8)

        # ---- 6. Specular highlight: bright soft band along the top ----
        # Creates the "light hitting curved glass" feel
        spec_h = max(2, int(h * 0.35))
        spec_strip = np.zeros((spec_h, w, 3), dtype=np.float32)
        spec_ramp = np.linspace(0.18, 0.0, spec_h, dtype=np.float32).reshape(spec_h, 1, 1)
        spec_color = np.array(self.COLORS['panel_edge'], dtype=np.float32)
        spec_strip[:] = spec_color * spec_ramp
        spec_y0, spec_y1 = 0, spec_h
        sub = region[spec_y0:spec_y1].astype(np.float32) + spec_strip
        sub_clipped = np.clip(sub, 0, 255).astype(np.uint8)
        # Apply only where mask is set
        mask_top = mask[spec_y0:spec_y1].astype(bool)
        np.copyto(region[spec_y0:spec_y1], sub_clipped, where=mask_top[..., None])

        # ---- 7. Outer border (crisp 1px) ----
        self._draw_rounded_rect(frame, x, y, width, height, radius,
                                self.COLORS['panel_border'], thickness=1)

        # NOTE: accent parameter no longer used - the dated "tab pill" is removed.
        # Accent is communicated through content text colour instead.
        del accent

    def _vertical_gradient(self, h, w, top_bgr, bot_bgr):
        """Return a (h, w, 3) uint8 image with a vertical BGR gradient."""
        top = np.array(top_bgr, dtype=np.float32)
        bot = np.array(bot_bgr, dtype=np.float32)
        ramp = np.linspace(0.0, 1.0, h, dtype=np.float32).reshape(h, 1)
        col = (1 - ramp) * top + ramp * bot   # (h, 3)
        img = np.broadcast_to(col[:, None, :], (h, w, 3)).astype(np.uint8).copy()
        return img

    def _horizontal_gradient(self, h, w, left_bgr, right_bgr):
        """Return a (h, w, 3) uint8 image with a horizontal BGR gradient."""
        left = np.array(left_bgr, dtype=np.float32)
        right = np.array(right_bgr, dtype=np.float32)
        ramp = np.linspace(0.0, 1.0, w, dtype=np.float32).reshape(1, w)
        col_r = (1 - ramp) * left[0] + ramp * right[0]
        col_g = (1 - ramp) * left[1] + ramp * right[1]
        col_b = (1 - ramp) * left[2] + ramp * right[2]
        img = np.stack([
            np.broadcast_to(col_r, (h, w)),
            np.broadcast_to(col_g, (h, w)),
            np.broadcast_to(col_b, (h, w)),
        ], axis=-1).astype(np.uint8)
        return img
    
    def _get_stress_color(self, score):
        """Get color for stress score."""
        if score < 0.20:
            return self.STRESS_COLORS['very_low']
        elif score < 0.40:
            return self.STRESS_COLORS['low']
        elif score < 0.60:
            return self.STRESS_COLORS['moderate']
        elif score < 0.80:
            return self.STRESS_COLORS['high']
        return self.STRESS_COLORS['very_high']
