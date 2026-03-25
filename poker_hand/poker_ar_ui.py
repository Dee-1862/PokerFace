
import cv2
import numpy as np
from collections import deque

class PokerARUI:
    """
    AR UI Controller for Poker Assistant.
    - Level 0: Minimal overlay (bounding boxes only)
    - Level 50: Win Probability + Outs Count
    - Level 100: Win Probability + Outs + Scrollable Top Winning Hands
    """
    
    COLORS = {
        'bg_dark': (25, 25, 28),
        'bg_panel': (38, 38, 42),
        'text_primary': (255, 255, 255),
        'text_secondary': (180, 180, 185),
        'text_highlight': (0, 255, 200),
        'win_good': (0, 200, 100),
        'win_ok': (0, 200, 255),
        'win_bad': (50, 50, 255),
        'card_bg': (245, 245, 245),
        'card_outline': (200, 200, 200),
    }

    def __init__(self, frame_width, frame_height):
        self.w = frame_width
        self.h = frame_height

        # UI State
        self.level = 0  # 0 to 100

        # Dual Scroll Offsets
        self.my_hands_scroll = 0
        self.opp_hands_scroll = 0
        self.max_scroll_my = 0
        self.max_scroll_opp = 0
        self.panel_h = 450  # Standardized height

        # Data
        self.win_probability = 0.0
        self.outs_count = 0
        self.my_hands = []   # Hero winning hands
        self.opp_hands = []  # Opponent winning hands
        self.board_cards = []

        # Interaction
        self.last_pinch_y = None

        # Panel position manager (injected from outside)
        self._pm = None

        # Last-rendered rects for drag detection: (x, y, w, h)
        self.win_panel_rect = None

    def set_positions(self, pm) -> None:
        """Inject a PanelPositionManager so panels use saved positions."""
        self._pm = pm

    def update_level(self, level):
        """Update UI expansion level (0-100%)."""
        self.level = max(0, min(100, level))

    def update_data(self, win_prob, outs, hands_data, board_cards=None):
        """Update game data."""
        self.win_probability = win_prob
        self.outs_count = outs
        self.board_cards = board_cards or []
        
        if isinstance(hands_data, dict):
            self.my_hands = hands_data.get('my_hands', [])
            self.opp_hands = hands_data.get('opp_hands', [])
        else:
            self.my_hands = hands_data
            self.opp_hands = []
            
        # Recalculate max scrolls
        row_h = 45 
        content_h = (self.panel_h // 2) - 45 # Space for items in each half
        
        self.max_scroll_my = max(0, len(self.my_hands) * row_h - content_h)
        self.max_scroll_opp = max(0, len(self.opp_hands) * row_h - content_h)
        
        # Clamp current scroll to new limits to prevent empty views
        self.my_hands_scroll = min(self.my_hands_scroll, self.max_scroll_my)
        self.opp_hands_scroll = min(self.opp_hands_scroll, self.max_scroll_opp)

    def handle_scroll(self, curr_y):
        """Handle scrolling using screen Y position."""
        if self.last_pinch_y is not None:
            dy = curr_y - self.last_pinch_y
            
            # Intuitive split: Top half of screen scrolls Hero, Bottom slips Opponent
            mid_point = self.h // 2
            
            if curr_y < mid_point:
                # Scroll MY HANDS
                self.my_hands_scroll += dy * 3.0
                self.my_hands_scroll = max(0, min(self.max_scroll_my, self.my_hands_scroll))
            else:
                # Scroll OPPONENT HANDS
                self.opp_hands_scroll += dy * 3.0
                self.opp_hands_scroll = max(0, min(self.max_scroll_opp, self.opp_hands_scroll))
        
        self.last_pinch_y = curr_y

    def reset_interaction(self):
        self.last_pinch_y = None

    def render(self, frame):
        """Render UI overlay."""
        # Persistent: Locked Board Visuals (Top Right)
        if self.board_cards:
            self._render_board_panel(frame)

        # Level 50+: Win Probability Panel (Bottom Left)
        if self.level >= 50:
            self._render_win_panel(frame)

        # Level 100: Dual Winning Hands Panels (Bottom Right)
        if self.level >= 100:
            self._render_winning_hands_panel(frame)

        return frame

    def _render_board_panel(self, frame):
        """Render Locked Board visually in Top Right."""
        # Dynamic width with specific padding
        card_count = len(self.board_cards)
        right_padding = 25 
        left_padding = 15
        spacing = 52
        
        panel_w = max(280, left_padding + (card_count * spacing) + right_padding)
        panel_h = 110
        
        # Fixed Right edge (10px margin for 720p safety), so as panel_w grows, 'x' moves LEFT
        x = self.w - panel_w - 10
        y = 20
        
        # Dark Glass
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x+panel_w, y+panel_h), self.COLORS['bg_dark'], -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        cv2.rectangle(frame, (x, y), (x+panel_w, y+panel_h), (0, 165, 255), 2) # Orange
        
        # Header with street label
        streets = {0: '', 3: 'FLOP', 4: 'TURN', 5: 'RIVER'}
        street_label = streets.get(card_count, f'{card_count}C')
        header = f"BOARD  {street_label}" if street_label else "BOARD"
        cv2.putText(frame, header, (x+20, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 165, 255), 1, cv2.LINE_AA)
        
        # Render cards - fixed relative to the panel left
        for i, card_str in enumerate(sorted(list(self.board_cards))):
            cx = x + left_padding + (i * spacing)
            cy = y + 40
            self._draw_mini_card(frame, cx, cy, card_str, scale=1.15)

    def _render_win_panel(self, frame):
        """Render main win probability indicator (BOTTOM LEFT)."""
        if self.win_probability > 60:
            color = self.COLORS['win_good']
        elif self.win_probability > 30:
            color = self.COLORS['win_ok']
        else:
            color = self.COLORS['win_bad']

        scale = 0.8 + (self.level - 50) / 100.0 * 0.4
        text = f"{self.win_probability:.1f}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.0 * scale, 3)

        default_x = 30
        default_y = self.h - 50
        if self._pm:
            x, y = self._pm.get('poker_win', default_x, default_y)
        else:
            x, y = default_x, default_y

        # Record rect (top-left corner + size) for drag detection
        self.win_panel_rect = (x - 15, y - th - 30, tw + 180, th + 55)

        cv2.rectangle(frame, (x-15, y-th-30), (x+tw+165, y+25), self.COLORS['bg_panel'], -1)
        cv2.rectangle(frame, (x-15, y-th-30), (x+tw+165, y+25), color, 2)

        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2.0 * scale, color, 3)
        cv2.putText(frame, "WIN CHANCE", (x, y-th-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['text_secondary'], 1)
        cv2.putText(frame, f"OUTS: {self.outs_count}", (x+tw+30, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.COLORS['text_highlight'], 2)

    def _render_winning_hands_panel(self, frame):
        """Render dual scrollable lists: My Hands (Top) and Opponent Hands (Bottom)."""
        panel_w = 440
        panel_h = self.panel_h
        
        # Position with tighter margin for 720p
        x = self.w - panel_w - 10
        y = self.h - panel_h - 10
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x+panel_w, y+panel_h), self.COLORS['bg_dark'], -1)
        cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)
        cv2.rectangle(frame, (x, y), (x+panel_w, y+panel_h), (80, 80, 85), 2)

        half_h = panel_h // 2
        
        # 1. MY POSSIBLE HANDS (TOP HALF)
        self._render_hand_section(frame, x, y, panel_w, half_h, "MY WINNING HANDS", self.my_hands, self.my_hands_scroll)
        
        # Divider
        cv2.line(frame, (x+10, y+half_h), (x+panel_w-10, y+half_h), (100, 100, 110), 2)
        
        # 2. OPPONENT WINNING HANDS (BOTTOM HALF)
        self._render_hand_section(frame, x, y + half_h, panel_w, half_h, "OPPONENT WINNING HANDS", self.opp_hands, self.opp_hands_scroll)

    def _render_hand_section(self, frame, x, y, w, h, title, hands, scroll_offset):
        """Render a section of the hands list."""
        cv2.putText(frame, title, (x+15, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['text_highlight'], 1)
        
        content_y_start = y + 40
        content_h = h - 50
        row_h = 45
        
        start_idx = int(scroll_offset // row_h)
        visible_count = int(content_h // row_h) + 1
        
        for i in range(start_idx, min(len(hands), start_idx + visible_count)):
            data = hands[i]
            draw_y = content_y_start + (i * row_h) - int(scroll_offset)
            
            if draw_y < content_y_start - 10 or draw_y + row_h > y + h + 15:
                continue
            
            # Highlight Glow if very high prob?
            if data.get('prob', 0) > 0.4:
                cv2.rectangle(frame, (x+10, draw_y+5), (x+w-10, draw_y+row_h-5), (0, 60, 0), -1)

            cv2.putText(frame, data['name'][:15], (x+15, draw_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLORS['text_primary'], 1)
            
            prob_pct = f"{data.get('prob', 0)*100:.1f}%"
            cv2.putText(frame, prob_pct, (x+125, draw_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLORS['win_ok'], 1)
            
            # Cards Visuals - Scale down to fit in 45px row
            cards = data.get('cards', [])
            card_start_x = x + 185
            for j, card_str in enumerate(cards[:5]): # Show best 5 cards
                cx = card_start_x + (j * 40) # Standard spacing
                cy = draw_y + 6
                self._draw_mini_card(frame, cx, cy, card_str, scale=0.7)

    def _draw_mini_card(self, frame, x, y, card_str, scale=1.0):
        """Draw a visually minimal but professional card."""
        cw, ch = int(40 * scale), int(56 * scale)
        
        if len(card_str) < 2: return
        rank = card_str[:-1]
        suit = card_str[-1].lower()

        # Subtle Shadow
        cv2.rectangle(frame, (x+2, y+2), (x+cw+2, y+ch+2), (10, 10, 15), -1)

        # White Background with border
        cv2.rectangle(frame, (x, y), (x+cw, y+ch), self.COLORS['card_bg'], -1)
        cv2.rectangle(frame, (x, y), (x+cw, y+ch), self.COLORS['card_outline'], 1)
        
        color = (200, 20, 20) if suit in ['h', 'd'] else (20, 20, 20)
        
        # Rank Text (bold-like)
        font_scale = 0.55 * scale
        cv2.putText(frame, rank, (x+int(4*scale), y+int(18*scale)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
        
        # Suit Symbol (Graphic positions - simplified for small scales)
        sx, sy = x + int(cw/2), y + int(ch*0.65)
        sym_size = int(8 * scale)
        
        if suit == 'h': # Heart
             pts = np.array([[sx, sy+sym_size], [sx-sym_size, sy], [sx-int(sym_size*0.6), sy-int(sym_size*0.8)], 
                            [sx, sy-int(sym_size*0.3)], [sx+int(sym_size*0.6), sy-int(sym_size*0.8)], [sx+sym_size, sy]], np.int32)
             cv2.fillPoly(frame, [pts], color)
        elif suit == 'd': # Diamond
             pts = np.array([[sx, sy-sym_size], [sx-sym_size, sy], [sx, sy+sym_size], [sx+sym_size, sy]], np.int32)
             cv2.fillPoly(frame, [pts], color)
        elif suit == 's': # Spade
             cv2.circle(frame, (sx-int(4*scale), sy+int(6*scale)), int(5*scale), color, -1)
             cv2.circle(frame, (sx+int(4*scale), sy+int(6*scale)), int(5*scale), color, -1)
             pts = np.array([[sx, sy-int(8*scale)], [sx-int(9*scale), sy+int(6*scale)], [sx+int(9*scale), sy+int(6*scale)]], np.int32)
             cv2.fillPoly(frame, [pts], color)
             cv2.rectangle(frame, (sx-1, sy+int(6*scale)), (sx+1, sy+int(14*scale)), color, -1)
        elif suit == 'c': # Club
             cv2.circle(frame, (sx, sy-int(2*scale)), int(4*scale), color, -1)
             cv2.circle(frame, (sx-int(5*scale), sy+int(5*scale)), int(4*scale), color, -1)
             cv2.circle(frame, (sx+int(5*scale), sy+int(5*scale)), int(4*scale), color, -1)
             cv2.rectangle(frame, (sx-1, sy+int(6*scale)), (sx+1, sy+int(12*scale)), color, -1)
