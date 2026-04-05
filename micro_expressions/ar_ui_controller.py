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
    
    # Apple-like colors
    COLORS = {
        'bg_dark': (25, 25, 28),
        'bg_panel': (38, 38, 42),
        'text_primary': (255, 255, 255),
        'text_secondary': (155, 155, 160),
        'text_tertiary': (100, 100, 105),
        'accent_blue': (255, 149, 0),
        'accent_green': (80, 200, 120),
        'accent_red': (80, 80, 255),
        'accent_orange': (80, 165, 255),
        'divider': (55, 55, 60),
    }
    
    STRESS_COLORS = {
        'very_low': (120, 220, 100),
        'low': (130, 200, 80),
        'moderate': (80, 220, 255),
        'high': (80, 140, 255),
        'very_high': (80, 80, 255),
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

    def set_positions(self, pm) -> None:
        """Inject a PanelPositionManager so panels use saved positions."""
        self._pm = pm
        
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
        
        # Outer glow when active
        if button.level > 0:
            glow_radius = int(button.radius + 6)
            overlay = frame.copy()
            cv2.circle(overlay, (cx, cy), glow_radius, glow_color, -1)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        
        # Main button
        cv2.circle(frame, (cx, cy), button.radius, self.COLORS['bg_panel'], -1)
        cv2.circle(frame, (cx, cy), button.radius, glow_color, 2)
        
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
        """Draw arc showing current level (0-100%)."""
        arc_angle = int(360 * level / 100)
        color = (180, 200, 220)
        cv2.ellipse(frame, (cx, cy), (radius, radius), -90, 0, arc_angle, color, 2)
    
    def _draw_level_badge(self, frame, cx, cy, level):
        """Draw level percentage badge."""
        text = f"{level}%"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        
        # Background pill
        pill_w = text_w + 12
        pill_h = text_h + 8
        pill_x = cx - pill_w // 2
        pill_y = cy - pill_h // 2
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h), 
                     self.COLORS['bg_panel'], -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        cv2.rectangle(frame, (pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h), 
                     self.COLORS['divider'], 1)
        
        # Text
        cv2.putText(frame, text, (pill_x + 6, pill_y + pill_h - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLORS['text_primary'], 1)
    
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

        default_x = max(10, cx - 10)
        default_y = cy + button.radius + 40
        if self._pm:
            panel_x, panel_y = self._pm.get('expressions', default_x, default_y)
        else:
            panel_x, panel_y = default_x, default_y

        # Record rect for drag detection
        self.expressions_rect = (panel_x, panel_y, panel_width, panel_height)

        # Glass panel
        self._draw_glass_panel(frame, panel_x, panel_y, panel_width, panel_height)
        
        # Title
        title = f"EXPRESSIONS ({num_expressions})"
        cv2.putText(frame, title, (panel_x + 10, panel_y + 16), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.33, self.COLORS['text_tertiary'], 1)
        
        # Divider
        cv2.line(frame, (panel_x + 8, panel_y + 22), 
                (panel_x + panel_width - 8, panel_y + 22), self.COLORS['divider'], 1)
        
        # Expressions
        y_offset = panel_y + 38
        for au_code, name in visible_expressions:
            value = self.action_units.get(au_code, 0)
            self._draw_expression_row(frame, panel_x + 10, y_offset, panel_width - 20, name, value)
            y_offset += row_height
    
    def _draw_expression_row(self, frame, x, y, width, name, value):
        """Draw single expression row."""
        name_color = self.COLORS['text_primary'] if value > 0.3 else self.COLORS['text_secondary']
        cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, name_color, 1)
        
        bar_width = 50
        bar_height = 5
        bar_x = x + width - bar_width - 28
        bar_y = y - 4
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     self.COLORS['divider'], -1)
        
        fill_width = int(bar_width * min(value, 1.0))
        if fill_width > 0:
            bar_color = self.COLORS['accent_green'] if value > 0.5 else self.COLORS['accent_blue'] if value > 0.25 else self.COLORS['text_tertiary']
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), bar_color, -1)
        
        cv2.putText(frame, f"{value:.0%}", (bar_x + bar_width + 4, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.28, self.COLORS['text_tertiary'], 1)
    
    def _render_heart_button(self, frame, frame_w, frame_h):
        """Render heart button with level indicator."""
        button = self.heart_button
        cx, cy = button.center
        glow_color = button.get_glow_color()
        
        # Outer glow when active
        if button.level > 0:
            glow_radius = int(button.radius + 6)
            overlay = frame.copy()
            cv2.circle(overlay, (cx, cy), glow_radius, glow_color, -1)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        
        # Main button
        cv2.circle(frame, (cx, cy), button.radius, self.COLORS['bg_panel'], -1)
        cv2.circle(frame, (cx, cy), button.radius, glow_color, 2)
        
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
        """Render HR and stress analytics on the right side."""
        panel_width = 175
        default_x = frame_w - panel_width - 15
        default_y = button.center[1] + button.radius + 40

        if self._pm:
            panel_x, panel_y = self._pm.get('hr_analytics', default_x, default_y)
        else:
            panel_x, panel_y = default_x, default_y

        # Panel height depends on level
        panel_height = 110 if button.level >= 50 else 60

        # Record rect for drag detection
        self.hr_analytics_rect = (panel_x, panel_y, panel_width, panel_height)

        self._draw_glass_panel(frame, panel_x, panel_y, panel_width, panel_height)
        
        y_offset = panel_y + 12
        
        # Heart Rate section
        cv2.putText(frame, "HEART RATE", (panel_x + 12, y_offset + 6), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.COLORS['text_tertiary'], 1)
        
        hr = self.stress_data['heart_rate']
        hr_text = f"{hr}" if hr > 0 else "--"
        hr_color = self._get_hr_color(hr)
        cv2.putText(frame, hr_text, (panel_x + 12, y_offset + 32), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, hr_color, 2)
        cv2.putText(frame, "BPM", (panel_x + 12 + len(hr_text) * 18, y_offset + 32), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.COLORS['text_tertiary'], 1)
        
        y_offset += 45
        
        # Divider
        cv2.line(frame, (panel_x + 10, y_offset), (panel_x + panel_width - 10, y_offset), 
                self.COLORS['divider'], 1)
        y_offset += 8
        
        # Stress section
        cv2.putText(frame, "STRESS", (panel_x + 12, y_offset + 6), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.COLORS['text_tertiary'], 1)
        
        score = self.stress_data['score']
        category = self.stress_data['category']
        stress_color = self._get_stress_color(score)
        
        cv2.putText(frame, category, (panel_x + 12, y_offset + 28), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, stress_color, 1)
        
        score_text = f"{score:.0%}"
        (score_w, _), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(frame, score_text, (panel_x + panel_width - score_w - 12, y_offset + 28), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, stress_color, 1)
    
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

        # Background glass panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (graph_x, graph_y),
                      (graph_x + graph_width, graph_y + graph_height),
                      self.COLORS['bg_dark'], -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        cv2.rectangle(frame, (graph_x, graph_y),
                      (graph_x + graph_width, graph_y + graph_height),
                      self.COLORS['divider'], 1)

        # Header
        cv2.putText(frame, "HEART RATE", (graph_x + 10, graph_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.COLORS['text_tertiary'], 1)
        hr = self.stress_data['heart_rate']
        if hr > 0:
            hr_text = f"{hr} BPM"
            (hr_w, _), _ = cv2.getTextSize(hr_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.putText(frame, hr_text,
                        (graph_x + graph_width - hr_w - 10, graph_y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, self._get_hr_color(hr), 1)

        history = list(self.hr_history)
        graph_area_y = graph_y + 18
        graph_area_h  = graph_height - 22

        # Stable Y range: centre on the rolling mean ± 15 BPM floor
        mean_hr  = sum(history) / len(history)
        span     = max(20, max(history) - min(history) + 10)   # min 20 BPM span
        lo       = max(30, mean_hr - span / 2)
        hi       = lo + span

        # Subtle gridlines at lo, mid, hi
        for frac in (0.0, 0.5, 1.0):
            gy  = int(graph_area_y + graph_area_h * (1 - frac))
            bpm = int(lo + (hi - lo) * frac)
            cv2.line(frame, (graph_x + 38, gy),
                     (graph_x + graph_width - 5, gy), (42, 42, 48), 1)
            cv2.putText(frame, str(bpm), (graph_x + 5, gy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.26,
                        self.COLORS['text_tertiary'], 1)

        # Plot line
        pts = []
        for i, v in enumerate(history):
            px = int(graph_x + 42 + (i / max(len(history) - 1, 1))
                     * (graph_width - 52))
            norm = max(0.0, min(1.0, (v - lo) / (hi - lo)))
            py   = int(graph_area_y + graph_area_h * (1 - norm))
            pts.append((px, py))

        for i in range(len(pts) - 1):
            cv2.line(frame, pts[i], pts[i + 1],
                     self._get_hr_color(history[i]), 2, cv2.LINE_AA)

        if pts:
            cv2.circle(frame, pts[-1], 4,
                       self._get_hr_color(history[-1]), -1, cv2.LINE_AA)
    
    def _get_hr_color(self, hr):
        """Get color for heart rate value."""
        if hr > 100:
            return self.STRESS_COLORS['high']
        elif hr > 80:
            return self.STRESS_COLORS['moderate']
        return self.STRESS_COLORS['low']
    
    def _draw_glass_panel(self, frame, x, y, width, height, alpha=0.9):
        """Draw glass panel."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), self.COLORS['bg_panel'], -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.rectangle(frame, (x, y), (x + width, y + height), self.COLORS['divider'], 1)
    
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
