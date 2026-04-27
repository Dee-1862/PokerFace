"""
Unified AR System
Integrates poker_main.py and micro_expressions/main.py

Context Detection:
- Face detected -> Show micro_expressions UI (stress, heart rate, FACS)
- Cards detected -> Show poker_hand UI (equity, outs, scrollable hands)
- Hand gestures work for both contexts (pinch, scroll)

All original UIs are preserved exactly as they were.
"""

import cv2
import sys
import time
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / '.env')
except ImportError:
    pass
try:
    import config as _cfg
except ImportError:
    _cfg = None  # type: ignore[assignment]

def _c(name, default):
    return getattr(_cfg, name, default) if _cfg else default

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent / 'poker_hand'))
sys.path.insert(0, str(Path(__file__).parent / 'micro_expressions'))

# === POKER HAND IMPORTS ===
from ultralytics import YOLO  # type: ignore[reportAttributeAccessIssue]
from treys import Card, Evaluator, Deck

# Import poker modules (from poker_hand folder)
from poker_ar_ui import PokerARUI

# === MICRO EXPRESSIONS IMPORTS ===
from engine import StressDetectionEngine
from ar_ui_controller import ARUIController

# === ADAPTIVE LEARNING IMPORTS ===
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from adaptive_learning import AdaptiveLearningSystem, PanelPositionManager
    ADAPTIVE_LEARNING_AVAILABLE = True
    print("[OK] Adaptive Learning module available")
except ImportError as e:
    ADAPTIVE_LEARNING_AVAILABLE = False
    AdaptiveLearningSystem = None   # type: ignore[misc, assignment]
    PanelPositionManager = None     # type: ignore[misc, assignment]
    print(f"[WARN] Adaptive Learning module not available: {e}")

# === SHARED IMPORTS ===
# Hand gesture detector will be imported after path setup
HandGestureDetector = None  # Will be loaded dynamically


# =============================================
# === POKER CALCULATION FUNCTIONS (from poker_main.py) ===
# =============================================

_PANEL_TINT   = (38, 36, 42)     # cool neutral
_PANEL_BORDER = (128, 124, 144)  # crisp visible border
_PANEL_EDGE   = (230, 228, 240)  # bright glass rim
_PANEL_DARK   = (10, 10, 14)     # shadow colour


def _compute_face_bbox(landmarks, w, h):
    """Return (x, y, w, h) in pixel coords for the face from MediaPipe landmarks."""
    if not landmarks:
        return None
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x0 = max(0, int(min(xs)))
    y0 = max(0, int(min(ys)))
    x1 = min(w, int(max(xs)))
    y1 = min(h, int(max(ys)))
    return (x0, y0, x1 - x0, y1 - y0)


# Smoothed face bbox (EMA) - module-level so it persists across frames
_FACE_BBOX_EMA = {'x': None, 'y': None, 'w': None, 'h': None}

def _smoothed_face_bbox(landmarks, w, h, alpha=0.18):
    """EMA-smoothed face bbox. Returns (x, y, w, h) or None if no face."""
    bbox = _compute_face_bbox(landmarks, w, h)
    if bbox is None:
        return None
    bx, by, bw, bh = bbox
    if _FACE_BBOX_EMA['x'] is None:
        _FACE_BBOX_EMA.update({'x': bx, 'y': by, 'w': bw, 'h': bh})
        return bbox
    _FACE_BBOX_EMA['x'] = (1 - alpha) * _FACE_BBOX_EMA['x'] + alpha * bx
    _FACE_BBOX_EMA['y'] = (1 - alpha) * _FACE_BBOX_EMA['y'] + alpha * by
    _FACE_BBOX_EMA['w'] = (1 - alpha) * _FACE_BBOX_EMA['w'] + alpha * bw
    _FACE_BBOX_EMA['h'] = (1 - alpha) * _FACE_BBOX_EMA['h'] + alpha * bh
    return (
        int(_FACE_BBOX_EMA['x']),
        int(_FACE_BBOX_EMA['y']),
        int(_FACE_BBOX_EMA['w']),
        int(_FACE_BBOX_EMA['h']),
    )


def _put_text_clipped(frame, text, org, font, scale, color, thickness=1,
                      line_type=cv2.LINE_AA,
                      clip_x=None, clip_y=None, clip_w=None, clip_h=None,
                      clip_radius=0):
    """Render text but clip to a rounded-rectangle area so it never leaks past
    the panel border."""
    if clip_x is None:
        cv2.putText(frame, text, org, font, scale, color, thickness, line_type)
        return
    h_f, w_f = frame.shape[:2]
    cx0 = max(0, clip_x); cy0 = max(0, clip_y)
    cx1 = min(w_f, clip_x + clip_w); cy1 = min(h_f, clip_y + clip_h)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    tx = org[0] - cx0
    ty = org[1] - cy0
    layer = np.zeros((cy1 - cy0, cx1 - cx0, 3), dtype=np.uint8)
    cv2.putText(layer, text, (tx, ty), font, scale, color, thickness, line_type)

    rx0 = clip_x - cx0; ry0 = clip_y - cy0
    rmask_full = _rounded_mask(clip_h, clip_w, clip_radius)
    rmask = np.zeros((cy1 - cy0, cx1 - cx0), dtype=np.uint8)
    ix0 = max(0, rx0); iy0 = max(0, ry0)
    ix1 = min(rmask.shape[1], rx0 + clip_w)
    iy1 = min(rmask.shape[0], ry0 + clip_h)
    if ix1 > ix0 and iy1 > iy0:
        sx = ix0 - rx0; sy = iy0 - ry0
        rmask[iy0:iy1, ix0:ix1] = rmask_full[sy:sy + (iy1 - iy0),
                                             sx:sx + (ix1 - ix0)]
    ink = (layer.sum(axis=-1) > 0)
    clip_ok = rmask.astype(bool)
    composite_mask = ink & clip_ok
    target = frame[cy0:cy1, cx0:cx1]
    np.copyto(target, layer, where=composite_mask[..., None])


def _rounded_mask(h, w, radius):
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


def _stroke_rounded_rect(frame, x, y, w_box, h_box, radius, color, thickness=1):
    radius = max(0, min(radius, h_box // 2, w_box // 2))
    if thickness < 0:
        cv2.rectangle(frame, (x + radius, y), (x + w_box - radius, y + h_box), color, -1)
        cv2.rectangle(frame, (x, y + radius), (x + w_box, y + h_box - radius), color, -1)
        cv2.circle(frame, (x + radius, y + radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x + w_box - radius, y + radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x + radius, y + h_box - radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x + w_box - radius, y + h_box - radius), radius, color, -1, cv2.LINE_AA)
        return
    cv2.line(frame, (x + radius, y), (x + w_box - radius, y), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x + radius, y + h_box), (x + w_box - radius, y + h_box), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x, y + radius), (x, y + h_box - radius), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x + w_box, y + radius), (x + w_box, y + h_box - radius), color, thickness, cv2.LINE_AA)
    cv2.ellipse(frame, (x + radius, y + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(frame, (x + w_box - radius, y + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(frame, (x + radius, y + h_box - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(frame, (x + w_box - radius, y + h_box - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)


def _draw_polished_panel(frame, x, y, w_box, h_box, accent=None, alpha=0.40, radius=18):
    """Apple-style clear glass panel - subtle blur, light tint, rounded corners,
    bright rim, top specular highlight, soft drop shadow."""
    del accent  # accent tab removed - looked dated
    h_f, w_f = frame.shape[:2]
    radius = max(0, min(radius, h_box // 2, w_box // 2))

    # ---- 1. Drop shadow (large, soft, blurred) ----
    sh_off_x, sh_off_y = 3, 8
    sh_pad = 14
    sx0 = max(0, x + sh_off_x - sh_pad); sy0 = max(0, y + sh_off_y - sh_pad)
    sx1 = min(w_f, x + w_box + sh_off_x + sh_pad)
    sy1 = min(h_f, y + h_box + sh_off_y + sh_pad)
    if sx1 > sx0 and sy1 > sy0:
        shmask_h, shmask_w = sy1 - sy0, sx1 - sx0
        shmask = np.zeros((shmask_h, shmask_w), dtype=np.uint8)
        shx = (x + sh_off_x) - sx0
        shy = (y + sh_off_y) - sy0
        inner = _rounded_mask(h_box, w_box, radius + 4)
        iy0 = max(0, shy); ix0 = max(0, shx)
        iy1 = min(shmask_h, shy + h_box); ix1 = min(shmask_w, shx + w_box)
        src_y0 = iy0 - shy; src_x0 = ix0 - shx
        src_y1 = src_y0 + (iy1 - iy0); src_x1 = src_x0 + (ix1 - ix0)
        if iy1 > iy0 and ix1 > ix0:
            shmask[iy0:iy1, ix0:ix1] = inner[src_y0:src_y1, src_x0:src_x1]
        shmask = cv2.GaussianBlur(shmask, (29, 29), 0)
        sr = frame[sy0:sy1, sx0:sx1]
        dark = np.full_like(sr, _PANEL_DARK)
        a = (shmask.astype(np.float32) / 255.0 * 0.50)[..., None]
        sr[:] = (sr.astype(np.float32) * (1 - a) + dark.astype(np.float32) * a).astype(np.uint8)

    # ---- 2. Body: SUBTLE blur + light tint ----
    x0 = max(0, x); y0 = max(0, y)
    x1 = min(w_f, x + w_box); y1 = min(h_f, y + h_box)
    if x1 <= x0 or y1 <= y0:
        return
    h, w = y1 - y0, x1 - x0
    region = frame[y0:y1, x0:x1]

    # Almost no blur - just 5px to soften noise. Camera shows through clearly.
    blurred = cv2.GaussianBlur(region, (5, 5), 0)

    tint = np.array(_PANEL_TINT, dtype=np.float32)
    tinted = blurred.astype(np.float32) * 0.88 + tint * 0.12
    tinted = np.clip(tinted, 0, 255).astype(np.uint8)

    ramp = np.linspace(1.06, 0.94, h, dtype=np.float32).reshape(h, 1, 1)
    sheened = np.clip(tinted.astype(np.float32) * ramp, 0, 255).astype(np.uint8)

    composite = cv2.addWeighted(sheened, alpha, region, 1 - alpha, 0)

    # ---- 3. Rounded mask ----
    mask = _rounded_mask(h, w, radius)
    mb = mask.astype(bool)
    np.copyto(region, composite, where=mb[..., None])

    # ---- 4. Full rim light ----
    rim_outer = np.zeros_like(region)
    _stroke_rounded_rect(rim_outer, 1, 1, w - 2, h - 2, max(1, radius - 1),
                         _PANEL_EDGE, thickness=1)
    rim_a = (rim_outer.sum(axis=-1) > 0).astype(np.float32) * 0.45
    rim_a = (rim_a * (mask.astype(np.float32) / 255.0))[..., None]
    region[:] = (region.astype(np.float32) * (1 - rim_a) +
                 rim_outer.astype(np.float32) * rim_a).astype(np.uint8)

    # ---- 5. Top specular highlight ----
    spec_h = max(2, int(h * 0.35))
    spec_strip = np.zeros((spec_h, w, 3), dtype=np.float32)
    spec_ramp = np.linspace(0.18, 0.0, spec_h, dtype=np.float32).reshape(spec_h, 1, 1)
    spec_strip[:] = np.array(_PANEL_EDGE, dtype=np.float32) * spec_ramp
    sub = region[0:spec_h].astype(np.float32) + spec_strip
    sub_clipped = np.clip(sub, 0, 255).astype(np.uint8)
    mask_top = mask[0:spec_h].astype(bool)
    np.copyto(region[0:spec_h], sub_clipped, where=mask_top[..., None])

    # ---- 6. Crisp outer border ----
    _stroke_rounded_rect(frame, x, y, w_box, h_box, radius,
                         _PANEL_BORDER, thickness=1)


def to_treys(label):
    """Convert YOLO label to Treys format."""
    if len(label) < 2: return None
    rank = label[:-1]
    suit = label[-1].lower()
    if rank == '10': rank = 'T'
    if suit not in ['h', 'd', 'c', 's']: return None
    if rank not in ['2','3','4','5','6','7','8','9','T','J','Q','K','A']: return None
    return f"{rank}{suit}"

# Load Preflop Equity Table
PREFLOP_EQUITY = {}
try:
    import csv
    import os
    equity_path = os.path.join(os.path.dirname(__file__), 'poker_hand', 'preflop_equity.csv')
    with open(equity_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            PREFLOP_EQUITY[row['hand']] = float(row['equity'])
    print(f"[OK] Loaded {len(PREFLOP_EQUITY)} preflop hands")
except Exception as e:
    print(f"! Preflop table load failed: {e}")

# Global Evaluator
try:
    POKER_EVALUATOR = Evaluator()
    print("[OK] Poker Evaluator initialized")
except Exception as e:
    print(f"! Evaluator init failed: {e}")
    POKER_EVALUATOR = None

def normalize_hand(my_hand):
    """Convert ['Ah', 'Kc'] to 'AKo' or 'AKs' for lookup."""
    if len(my_hand) != 2:
        return None
    ranks = [c[0] for c in my_hand]
    suits = [c[1] for c in my_hand]
    rank_order = 'AKQJT98765432'
    sorted_ranks = sorted(ranks, key=lambda r: rank_order.index(r))
    suited = 's' if suits[0] == suits[1] else 'o'
    if sorted_ranks[0] == sorted_ranks[1]:
        return sorted_ranks[0] + sorted_ranks[1]
    return sorted_ranks[0] + sorted_ranks[1] + suited

def get_best_5_cards(all_cards_ints):
    """Find the best 5-card hand from >=5 cards."""
    from itertools import combinations
    evaluator = POKER_EVALUATOR
    if not evaluator: return [Card.int_to_str(c) for c in all_cards_ints[:5]]
    
    min_score = float('inf')
    best_hand = None
    for five_cards in combinations(all_cards_ints, 5):
        score = evaluator.evaluate(list(five_cards), [])
        if score < min_score:
            min_score = score
            best_hand = list(five_cards)
    if best_hand:
        return [Card.int_to_str(c) for c in best_hand]
    return [Card.int_to_str(c) for c in all_cards_ints[:5]]

def exhaustive_enumeration(my_hand, board_cards):
    """Evaluate ALL possible outcomes (100% accurate)."""
    from itertools import combinations
    import random

    evaluator = POKER_EVALUATOR
    if evaluator is None:
        return 0.0, 0, {'my_hands': [], 'opp_hands': []}
    hero_hand = [Card.new(c) for c in my_hand]
    board = [Card.new(c) for c in board_cards]
    
    deck = Deck()
    known = hero_hand + board
    for card in known:
        if card in deck.cards:
            deck.cards.remove(card)
    
    cards_needed = 5 - len(board)
    
    wins = 0
    total = 0
    hero_hand_counts = defaultdict(int)
    opp_hand_counts = defaultdict(int)
    hero_hand_examples = {}
    opp_hand_examples = {}
    
    for runout in combinations(deck.cards, cards_needed):
        remaining = [c for c in deck.cards if c not in runout]
        opp_sample = list(combinations(remaining, 2))
        if len(opp_sample) > 200:
            opp_sample = random.sample(opp_sample, 200)
        
        for opp_hand in opp_sample:
            total += 1
            full_board = board + list(runout)
            
            try:
                hero_score = evaluator.evaluate(full_board, hero_hand)
                opp_score = evaluator.evaluate(full_board, list(opp_hand))
                
                hero_class_idx = evaluator.get_rank_class(hero_score)
                hero_class_str = evaluator.class_to_string(hero_class_idx)
                opp_class_idx = evaluator.get_rank_class(opp_score)
                opp_class_str = evaluator.class_to_string(opp_class_idx)
                
                if hero_score < opp_score:
                    wins += 1
                    hero_hand_counts[hero_class_str] += 1
                    if hero_class_str not in hero_hand_examples:
                        hero_hand_examples[hero_class_str] = get_best_5_cards(hero_hand + full_board)
                else:
                    opp_hand_counts[opp_class_str] += 1
                    if opp_class_str not in opp_hand_examples:
                        opp_hand_examples[opp_class_str] = get_best_5_cards(list(opp_hand) + full_board)
            except:
                pass
    
    equity = (wins / total * 100) if total > 0 else 0.0
    
    my_hands = []
    for h_name, count in hero_hand_counts.items():
        prob = count / total if total > 0 else 0
        if prob > 0.01:
            my_hands.append({'name': h_name, 'prob': prob, 'cards': hero_hand_examples.get(h_name, [])[:7]})
    my_hands.sort(key=lambda x: x['prob'], reverse=True)
    
    opp_hands = []
    for h_name, count in opp_hand_counts.items():
        prob = count / total if total > 0 else 0
        if prob > 0.01:
            opp_hands.append({'name': h_name, 'prob': prob, 'cards': opp_hand_examples.get(h_name, [])[:7]})
    opp_hands.sort(key=lambda x: x['prob'], reverse=True)
    
    return equity, 0, {'my_hands': my_hands, 'opp_hands': opp_hands}

def evaluate_river(my_hand, board_cards):
    """River: Evaluate against all possible opponent hands."""
    from itertools import combinations
    evaluator = POKER_EVALUATOR
    if evaluator is None:
        return 0.0, 0, {'my_hands': [], 'opp_hands': []}

    hero_hand = [Card.new(c) for c in my_hand]
    board = [Card.new(c) for c in board_cards]
    
    deck = Deck()
    known = hero_hand + board
    for card in known:
        if card in deck.cards:
            deck.cards.remove(card)
    
    wins = 0
    total = 0
    hero_hand_counts = defaultdict(int)
    opp_hand_counts = defaultdict(int)
    hero_hand_examples = {}
    opp_hand_examples = {}
    
    for opp_hand in combinations(deck.cards, 2):
        total += 1
        try:
            hero_score = evaluator.evaluate(board, hero_hand)
            opp_score = evaluator.evaluate(board, list(opp_hand))
            
            hero_class_idx = evaluator.get_rank_class(hero_score)
            hero_class_str = evaluator.class_to_string(hero_class_idx)
            opp_class_idx = evaluator.get_rank_class(opp_score)
            opp_class_str = evaluator.class_to_string(opp_class_idx)
            
            if hero_score < opp_score:
                wins += 1
                hero_hand_counts[hero_class_str] += 1
                if hero_class_str not in hero_hand_examples:
                    hero_hand_examples[hero_class_str] = get_best_5_cards(hero_hand + board)
            else:
                opp_hand_counts[opp_class_str] += 1
                if opp_class_str not in opp_hand_examples:
                    opp_hand_examples[opp_class_str] = get_best_5_cards(list(opp_hand) + board)
        except:
            pass
    
    equity = (wins / total * 100) if total > 0 else 0.0
    
    my_hands = []
    for h_name, count in hero_hand_counts.items():
        prob = count / total if total > 0 else 0
        if prob > 0.01:
            my_hands.append({'name': h_name, 'prob': prob, 'cards': hero_hand_examples.get(h_name, [])[:7]})
    my_hands.sort(key=lambda x: x['prob'], reverse=True)
    
    opp_hands = []
    for h_name, count in opp_hand_counts.items():
        prob = count / total if total > 0 else 0
        if prob > 0.01:
            opp_hands.append({'name': h_name, 'prob': prob, 'cards': opp_hand_examples.get(h_name, [])[:7]})
    opp_hands.sort(key=lambda x: x['prob'], reverse=True)
    
    return equity, 0, {'my_hands': my_hands, 'opp_hands': opp_hands}

def calculate_equity_fast(my_hand, board_cards):
    """Hybrid equity calculator."""
    if POKER_EVALUATOR is None:
        return 0.0, 0, {'my_hands': [], 'opp_hands': []}
    
    if not my_hand or len(my_hand) != 2:
        return 0.0, 0, {'my_hands': [], 'opp_hands': []}
    
    board_size = len(board_cards)
    
    try:
        if board_size == 0:
            hand_key = normalize_hand(my_hand)
            if hand_key and hand_key in PREFLOP_EQUITY:
                equity = PREFLOP_EQUITY[hand_key]
                is_pair = my_hand[0][0] == my_hand[1][0]
                my_hands = [{'name': 'Pair' if is_pair else 'High Card', 'prob': 1.0, 'cards': my_hand}]
                opp_hands = [
                    {'name': 'Pair', 'prob': 0.06, 'cards': ['As', 'Ac']},
                    {'name': 'High Card', 'prob': 0.94, 'cards': ['Ah', 'Kh']}
                ]
                return equity, 0, {'my_hands': my_hands, 'opp_hands': opp_hands}
            else:
                return 50.0, 0, {'my_hands': [], 'opp_hands': []}
        
        elif board_size in [3, 4]:
            return exhaustive_enumeration(my_hand, board_cards)
        
        else:
            return evaluate_river(my_hand, board_cards)
            
    except Exception as e:
        print(f"[ERROR] calculate_equity_fast: {e}")
        return 0.0, 0, {'my_hands': [], 'opp_hands': []}


# =============================================
# === UNIFIED MAIN LOOP ===
# =============================================

def _render_mini_poker_hud(frame, registered_hand, board_cards, equity, poker_ar_ui, w, h):
    """
    Compact bottom-bar overlay shown in face/hybrid contexts when a hand is saved.
    Shows hole cards, board cards (with street label), and current win equity.
    """
    valid_hand = [c for c in registered_hand if c]
    if not valid_hand:
        return

    # --- geometry ---
    bar_h = 72
    bar_y = h - bar_h - 6
    bar_x = 6
    bar_w = w - 12

    overlay = frame.copy()
    cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (20, 20, 24), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (180, 100, 255), 1)

    # Section label
    cv2.putText(frame, "MY HAND", (bar_x + 10, bar_y + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 100, 255), 1, cv2.LINE_AA)

    # Hole cards
    card_scale = 0.75
    card_w_px = int(40 * card_scale)
    cx = bar_x + 10
    for card_str in valid_hand:
        poker_ar_ui._draw_mini_card(frame, cx, bar_y + 18, card_str, scale=card_scale)
        cx += card_w_px + 4

    # Board cards section
    if board_cards:
        streets = {3: 'FLOP', 4: 'TURN', 5: 'RIVER'}
        n = len(board_cards)
        street = streets.get(n, f'{n}C')
        bx = cx + 14
        cv2.putText(frame, street, (bx, bar_y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 165, 255), 1, cv2.LINE_AA)
        bx_card = bx
        for card_str in sorted(board_cards):
            poker_ar_ui._draw_mini_card(frame, bx_card, bar_y + 18, card_str, scale=card_scale)
            bx_card += card_w_px + 4

    # Equity indicator (right side)
    if equity > 0:
        eq_color = (0, 200, 100) if equity > 60 else (0, 200, 255) if equity > 35 else (80, 80, 255)
        eq_text = f"{equity:.1f}%"
        (tw, _), _ = cv2.getTextSize(eq_text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
        ex = bar_x + bar_w - tw - 14
        cv2.putText(frame, "WIN", (ex, bar_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 155), 1, cv2.LINE_AA)
        cv2.putText(frame, eq_text, (ex, bar_y + 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, eq_color, 2, cv2.LINE_AA)


def _rects_overlap(r1, r2, margin: int = 12) -> bool:
    """Return True if two (x, y, w, h) rects overlap (with a small margin)."""
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    return not (x1 + w1 + margin <= x2 or x2 + w2 + margin <= x1 or
                y1 + h1 + margin <= y2 or y2 + h2 + margin <= y1)


def main():
    # === ARGUMENT PARSING ===
    parser = argparse.ArgumentParser(description="Unified AR System: Poker + Micro Expressions")
    import os
    _default_cam = int(os.environ.get("CAMERA_INDEX", _c('CAMERA_INDEX', 0)))
    parser.add_argument('--camera', '-c', type=int, default=_default_cam,
                        help='Camera device ID (default from .env CAMERA_INDEX)')
    parser.add_argument('--debug', action='store_true', 
                        help='Enable debug mode')
    args = parser.parse_args()
    
    camera_id = args.camera
    
    print("Unified AR System: Poker + Micro Expressions")
    print("=" * 60)
    print(f"Using Camera: {camera_id}")
    print("Context Detection:")
    print("  - Face detected -> Micro Expressions UI (stress, HR, FACS)")
    print("  - Cards detected -> Poker Hand UI (equity, outs, hands)")
    print("  - Hand gestures (pinch, scroll) work in both contexts")
    print("=" * 60)
    
    # === INITIALIZE YOLO MODEL (Poker) ===
    try:
        model_path = Path(__file__).parent / 'poker_hand' / 'poker_v2.pt'
        if model_path.exists():
            model = YOLO(str(model_path))
        else:
            model = YOLO(str(Path(__file__).parent / 'poker_hand' / 'poker_v1.pt'))
        print("[OK] YOLO Poker Model loaded")
    except Exception as e:
        print(f"[WARN] Failed to load YOLO model: {e}")
        model = None
    
    # === INITIALIZE STRESS DETECTION ENGINE (Micro Expressions) ===
    config = {
        'camera_id': camera_id,
        'debug_mode': args.debug,
        'enable_validation': False,
        'is_video_file': False,
        'minimal_mode': True  # AR mode
    }
    
    engine = StressDetectionEngine(config)
    
    # Load micro expression modules
    def load_module_instance(name, config):
        if name == 'face':
            from face_detection import FaceDetectionModule
            return FaceDetectionModule()
        elif name == 'rppg':
            from rppg_heart_rate import RPPGModule
            return RPPGModule(enable_validation=False)
        elif name == 'facs':
            from facs_action_units import FACSModule
            return FACSModule()
        elif name == 'stress':
            from stress_detector import StressDetectorModule
            return StressDetectorModule()
        return None
    
    # Attach modules
    pipeline_order = ['face', 'rppg', 'facs', 'stress']
    for module_name in pipeline_order:
        try:
            module = load_module_instance(module_name, config)
            if module:
                engine.attach_module(module_name, module)
                print(f"[OK] Module loaded: {module_name.upper()}")
        except Exception as e:
            print(f"[WARN] Error loading {module_name}: {e}")
    
    # === INITIALIZE AR CONTROLLERS ===
    # Micro expressions AR UI
    micro_ar_controller = ARUIController(button_position='top-right')
    
    # Hand gesture detector (shared) - import dynamically
    global HandGestureDetector
    gesture_detector = None
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "hand_gesture_detector",
            str(Path(__file__).parent / 'poker_hand' / 'hand_gesture_detector.py')
        )
        if spec is not None and spec.loader is not None:
            hgd_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(hgd_module)
            HandGestureDetector = hgd_module.HandGestureDetector
            gesture_detector = HandGestureDetector()
            print("[OK] Hand Gesture Detector initialized")
    except Exception as e:
        print(f"[WARN] Error initializing hand detector: {e}")
    
    # === CAMERA SETUP ===
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    cv2.namedWindow("Unified AR System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Unified AR System", 1280, 720)
    
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Cannot open camera")
        return
    
    h, w = frame.shape[:2]
    
    # === POKER AR UI ===
    poker_ar_ui = PokerARUI(w, h)
    
    # === STATE VARIABLES ===
    # Poker state (from poker_main.py)
    card_history = defaultdict(int)
    finalized_cards = {}
    STABILITY_THRESHOLD    = _c('CARD_STABILITY_THRESHOLD', 4)
    FINALIZE_THRESHOLD     = _c('CARD_FINALIZE_THRESHOLD',  12)
    CARD_FADE_TIMEOUT      = _c('CARD_FADE_TIMEOUT',        60)
    NO_CARDS_RESET_TIMEOUT = _c('CARD_RESET_TIMEOUT',       90)
    frame_count = 0
    zero_card_frames = 0
    
    registered_hand = []
    reg_lock = False
    board_cards = set()
    board_lock_once = False
    
    equity_cache = {'hand': None, 'board': None, 'result': None}
    eq, outs, top = 0.0, 0, {'my_hands': [], 'opp_hands': []}  # global equity state

    # Frame-skip counters for heavy inference
    # YOLO: every 2 frames (was every frame at imgsz=1280 -> ~150ms; now imgsz=640 every 2 frames)
    # Micro modules: every 3 frames (face/rPPG/FACS/stress don't change fast enough to need per-frame)
    YOLO_SKIP   = _c('YOLO_SKIP',   2)
    MODULE_SKIP = _c('MODULE_SKIP', 3)
    _last_detected_list: list = []
    _last_current_frame_cards: set = set()

    # Context state — smooth transitions
    # Contexts: 'poker' | 'face' | 'hybrid' (face + saved hand) | 'none'
    current_context = 'none'
    pending_context = 'none'
    pending_frames = 0
    HYSTERESIS_FRAMES = _c('HYSTERESIS_FRAMES', 8)
    context_alpha = 1.0        # 0->1 fade-in for current context panels
    CONTEXT_FADE_SPEED = 0.07  # increment per frame (~14 frames to fully fade in)
    prev_display_frame = None  # last rendered frame used for cross-fade

    # Per-street locked verdict.
    # Committed once every time a new community card appears (flop/turn/river).
    # In face-only mode: committed on thumb-gesture showdown.
    # Never fluctuates within a street.
    _locked_verdict  = None  # dict: {prediction, p_bluff, confidence, mode_tag, street} or None
    _last_board_size = 0     # track board_cards size to detect street changes
    MIN_PRED_SAMPLES = _c('MIN_PRED_SAMPLES', 2)

    # Showdown gesture state
    _SHOWDOWN_COOLDOWN   = _c('SHOWDOWN_COOLDOWN_SECONDS', 3.0)
    _last_showdown_time  = 0.0
    _showdown_flash      = None  # (label_str, timestamp) — brief on-screen confirmation

    # === PANEL DRAG STATE ===
    _panel_positions  = PanelPositionManager() if PanelPositionManager else None
    _panel_rects      = {}   # {name: (x, y, w, h)} — updated each frame after rendering
    _drag_panel       = None # name of panel being dragged, or None
    _drag_hand        = None # 'left_hand' | 'right_hand' doing the drag
    _drag_offset_x    = 0
    _drag_offset_y    = 0
    _drag_orig_x      = 0   # position before drag started (for collision revert)
    _drag_orig_y      = 0
    _drag_hover_name   = None  # panel name whose handle the finger is hovering over
    _drag_hover_start  = 0.0
    _drag_hover_grace  = 0     # frames of lost pointing before hover resets (forgiveness)
    DRAG_HOVER_GRACE_FRAMES = 6
    DRAG_DWELL         = 0.5   # seconds to dwell on handle circle before drag activates
    # DB reset gesture-hold state
    # Right-hand fist 3s -> reset current player
    # Both-hands fist 5s -> wipe all profiles
    _db_reset_user_start  = 0.0
    _db_reset_all_start   = 0.0

    # === INITIALIZE ADAPTIVE LEARNING ===
    learner = None
    if ADAPTIVE_LEARNING_AVAILABLE and AdaptiveLearningSystem is not None:
        try:
            learner = AdaptiveLearningSystem()
            print("[OK] Adaptive Learning System initialized")
        except Exception as e:
            print(f"[WARN] Adaptive Learning init failed: {e}")

    # Wire position manager to UI components so panels remember their positions
    if _panel_positions is not None:
        micro_ar_controller.set_positions(_panel_positions)
        poker_ar_ui.set_positions(_panel_positions)
        print("[OK] Panel position manager wired to UI components")

    print("\n>> Starting unified detection loop...")
    print("  Press 'q' to quit, 'c' to clear poker board")
    print("  Press 'b' to force baseline recalibration (face mode)")
    print("  Press 'r' to reset all panel positions to defaults")
    print("  Showdown: thumbs-up (strong hand) or thumbs-down (bluffing), hold 1s")
    print("  Drag panels: point index finger at panel handle circle, hold 0.3s to grab")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        h, w = frame.shape[:2]
        
        # === UPDATE SHARED STATE FOR MICRO EXPRESSIONS ===
        engine.shared_state.update({
            'frame': frame,
            'frame_dimensions': (h, w),
            'timestamp': time.time(),
            'is_video_file': False,
            'minimal_mode': True
        })

        # === PROCESS MICRO EXPRESSION MODULES (throttled — every MODULE_SKIP frames) ===
        # Face/rPPG/FACS/stress results are stable enough; running every frame just burns CPU.
        if frame_count % MODULE_SKIP == 0:
            for name, module in engine.modules.items():
                module.process(engine.shared_state)

        face_detected = engine.shared_state.get('face_detected', False)

        # === PROCESS POKER DETECTION (throttled — every YOLO_SKIP frames, imgsz=640) ===
        # imgsz=640 is YOLO's native training resolution — equally accurate, ~4× faster than 1280.
        # We reuse the previous frame's detections on skipped frames.
        if frame_count % YOLO_SKIP == 0:
            _last_detected_list = []
            _last_current_frame_cards = set()
            if model:
                results = model(frame, verbose=False,
                               conf=_c('YOLO_CONF', 0.45),
                               iou=_c('YOLO_IOU', 0.15),
                               imgsz=_c('YOLO_IMGSZ', 640))
                for r in results:
                    for box in r.boxes:
                        lbl = model.names[int(box.cls[0])]
                        conf = float(box.conf[0])
                        bbox = box.xyxy[0].tolist()
                        _last_current_frame_cards.add(lbl)
                        _last_detected_list.append({'label': lbl, 'confidence': conf, 'box': bbox})

        detected_list = _last_detected_list
        current_frame_cards = _last_current_frame_cards
        cards_detected = len(detected_list) > 0
        
        # === CONTEXT DETECTION (hysteresis + smooth fade) ===
        # Cards + face in view  -> hybrid_poker (full poker UI + opponent panel side by side)
        # Cards alone           -> poker
        # Face + saved hand     -> hybrid (stress UI + mini poker bar)
        # Face alone            -> face
        # Nothing               -> none
        if cards_detected and face_detected:
            voted = 'hybrid_poker'
        elif cards_detected:
            voted = 'poker'
        elif face_detected:
            voted = 'hybrid' if registered_hand else 'face'
        else:
            voted = 'none'

        if voted == current_context:
            pending_frames = 0
            pending_context = voted
        else:
            if voted == pending_context:
                pending_frames += 1
            else:
                pending_context = voted
                pending_frames = 1

            if pending_frames >= HYSTERESIS_FRAMES:
                if current_context != pending_context:
                    current_context = pending_context
                    context_alpha = 0.0
                    print(f"[Context] -> {current_context.upper()}")
                pending_frames = 0

        # Advance fade-in each frame
        context_alpha = min(1.0, context_alpha + CONTEXT_FADE_SPEED)
        
        # === PROCESS HAND GESTURES ===
        gestures = {}
        if gesture_detector:
            gestures = gesture_detector.process(frame, engine.shared_state)
        
        # Helper for grip rejection
        def is_touching_card(point, cards):
            px, py = point
            for c in cards:
                x1, y1, x2, y2 = c['box']
                if (x1-20 < px < x2+20) and (y1-20 < py < y2+20):
                    return True
            return False
        
        # === EQUITY CALCULATION (runs whenever a hand is saved, context-independent) ===
        if registered_hand:
            eq_board = list(board_cards)
            hand_key = tuple(sorted(x for x in registered_hand if x is not None))
            board_key = tuple(sorted(eq_board)) if eq_board else None
            if (equity_cache['hand'] == hand_key and equity_cache['board'] == board_key
                    and equity_cache['result'] is not None):
                eq, outs, top = equity_cache['result']
            else:
                eq, outs, top = calculate_equity_fast(registered_hand, eq_board)
                equity_cache['hand'] = hand_key  # type: ignore[typeddict-unknown-key]
                equity_cache['board'] = board_key  # type: ignore[typeddict-unknown-key]
                equity_cache['result'] = (eq, outs, top)  # type: ignore[typeddict-unknown-key]
            poker_ar_ui.update_data(eq, outs, top, board_cards=board_cards)

        # === RENDER BASED ON CONTEXT ===
        display_frame = frame.copy()

        if current_context == 'poker':
            # === POKER CONTEXT: Full poker_main.py logic ===
            
            # Filter gestures touching cards
            lh = gestures.get('left_hand')
            if lh and lh['pinch']['active']:
                thumb_pt = lh['pinch']['thumb_pos']
                if is_touching_card(thumb_pt, detected_list):
                    lh['pinch']['active'] = False
                    lh['pinch']['is_locked'] = False
            
            # Render hands
            if gesture_detector:
                display_frame = gesture_detector.render_hands(display_frame, gestures, {})
            
            # Left Hand: Register hand
            if lh and lh['pinch']['is_locked']:
                if not reg_lock:
                    candidates = list(finalized_cards.values())
                    if len(candidates) >= 2:
                        candidates.sort(key=lambda x: (x['box'][2]-x['box'][0])*(x['box'][3]-x['box'][1]), reverse=True)
                        registered_hand = [to_treys(x['label']) for x in candidates[:2]]
                        reg_lock = True
            else:
                reg_lock = False
            
            # Right Hand: Board Lock + Scroll
            rh = gestures.get('right_hand')
            
            if rh and rh['pinch']['active']:
                if rh['pinch']['is_locked']:
                    if not board_lock_once:
                        added_count = 0
                        for lbl, data in finalized_cards.items():
                            t = to_treys(lbl)
                            if t and t not in board_cards and t not in registered_hand:
                                board_cards.add(t)
                                added_count += 1
                        if added_count > 0:
                            print(f"Added {added_count} cards to board: {board_cards}")
                        board_lock_once = True
                else:
                    board_lock_once = False
            else:
                board_lock_once = False
            
            # Two-Finger Scroll
            if rh and rh.get('two_finger_scroll', {}).get('active'):
                poker_ar_ui.handle_scroll(rh['two_finger_scroll']['screen_y'])
            else:
                poker_ar_ui.reset_interaction()
            
            # UI Visibility
            if registered_hand:
                poker_ar_ui.update_level(100)
            else:
                poker_ar_ui.update_level(0)
            
            # === Stability Logic (from poker_main.py) ===
            for lbl in current_frame_cards:
                card_history[lbl] += 1
            
            for lbl in list(card_history.keys()):
                if lbl not in current_frame_cards and lbl not in finalized_cards:
                    card_history[lbl] -= 1
                    if card_history[lbl] <= 0:
                        del card_history[lbl]
            
            for c in detected_list:
                lbl = c['label']
                if lbl not in finalized_cards and STABILITY_THRESHOLD <= card_history[lbl] <= FINALIZE_THRESHOLD:
                    finalized_cards[lbl] = {
                        'label': lbl,
                        'confidence': c['confidence'],
                        'box': c['box'],
                        'last_seen': frame_count
                    }
            
            for c in detected_list:
                if c['label'] in finalized_cards:
                    finalized_cards[c['label']].update({
                        'confidence': max(c['confidence'], finalized_cards[c['label']]['confidence']),
                        'box': c['box'],
                        'last_seen': frame_count
                    })
            
            if len(detected_list) == 0:
                zero_card_frames += 1
            else:
                zero_card_frames = 0
            
            # Full reset when no cards visible for a while
            if zero_card_frames > NO_CARDS_RESET_TIMEOUT:
                finalized_cards.clear()
                card_history.clear()
                if zero_card_frames == NO_CARDS_RESET_TIMEOUT + 1:
                    print("Visuals Reset (No cards visible)")
            
            # Expire individual cards that haven't been seen recently
            expired_cards = [lbl for lbl, data in finalized_cards.items() 
                            if frame_count - data.get('last_seen', 0) > CARD_FADE_TIMEOUT]
            for lbl in expired_cards:
                del finalized_cards[lbl]
            
            # Build stable cards
            stable_cards = {}
            for k, v in finalized_cards.items():
                stable_cards[k] = v
            for c in detected_list:
                lbl = c['label']
                if lbl not in finalized_cards and card_history[lbl] >= STABILITY_THRESHOLD:
                    if lbl not in stable_cards or c['confidence'] > stable_cards[lbl]['confidence']:
                        stable_cards[lbl] = c
            
            stable_list = list(stable_cards.values())
            
            # === RENDER POKER UI ===
            display_frame = poker_ar_ui.render(display_frame)
            
            # Draw card boxes
            rh_set = set(registered_hand)
            
            for c in stable_list:
                t = to_treys(c['label'])
                x1, y1, x2, y2 = map(int, c['box'])
                
                is_reg = t in rh_set
                is_locked = t in board_cards
                
                if is_reg:
                    color = (0, 255, 0)
                    status = "ME"
                    thick = 3
                elif is_locked:
                    color = (0, 165, 255)
                    status = "LOCKED"
                    thick = 3
                else:
                    color = (255, 200, 0)
                    status = "DETECTING"
                    thick = 1
                
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thick)
                cv2.putText(display_frame, f"{c['label']} {status}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # UI Hints
            if rh and rh['pinch']['active']:
                if not rh['pinch']['is_locked']:
                    prog = rh['pinch'].get('hold_progress', 0)
                    pt = tuple(rh['pinch']['index_pos'].astype(int))
                    cv2.putText(display_frame, f"HOLD TO ADD... {int(prog*100)}%", 
                               (pt[0]+20, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            if not board_cards and not registered_hand:
                cv2.putText(display_frame, "1. Left Pinch: Save Hand", (20, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.putText(display_frame, "2. Right Pinch (5s): Add Board Cards", (20, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.putText(display_frame, "3. Press 'c' to Clear Board", (20, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Draw saved hand panel — mini card visuals
            valid_hand = [c for c in registered_hand if c]
            if valid_hand:
                card_w, card_h = 42, 58
                spacing = 50
                panel_w = 16 + len(valid_hand) * spacing + 10
                panel_h = 90
                px, py = 8, 8
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (25, 25, 28), -1)
                cv2.addWeighted(overlay, 0.85, display_frame, 0.15, 0, display_frame)
                cv2.rectangle(display_frame, (px, py), (px + panel_w, py + panel_h), (0, 210, 80), 2)
                cv2.putText(display_frame, "MY HAND", (px + 10, py + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 210, 80), 1, cv2.LINE_AA)
                visible_cards = {to_treys(c['label']) for c in stable_list}
                for i, card_str in enumerate(valid_hand):
                    cx = px + 10 + i * spacing
                    cy = py + 26
                    poker_ar_ui._draw_mini_card(display_frame, cx, cy, card_str, scale=1.05)
                    # Dim badge when card not currently visible in camera
                    if card_str not in visible_cards:
                        cv2.rectangle(display_frame, (cx, cy), (cx + card_w, cy + card_h),
                                      (0, 0, 100), 1)
            else:
                cv2.putText(display_frame, "Left Pinch: Save Hand", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1, cv2.LINE_AA)
            
            if not board_cards and registered_hand:
                cv2.putText(display_frame, "Right Pinch (5s) to Add Board", (w - 450, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        elif current_context in ('face', 'hybrid', 'hybrid_poker'):
            # === FACE / HYBRID / HYBRID_POKER CONTEXT ===

            if current_context == 'hybrid_poker':
                # In hybrid_poker: pure poker UI only — no face panels, no opponent read.
                # (Learner still runs silently below to keep accumulating signal data.)

                # --- Gesture state for cards ---
                lh = gestures.get('left_hand')
                if lh and lh['pinch']['active']:
                    thumb_pt = lh['pinch']['thumb_pos']
                    if is_touching_card(thumb_pt, detected_list):
                        lh['pinch']['active'] = False
                        lh['pinch']['is_locked'] = False
                if gesture_detector:
                    display_frame = gesture_detector.render_hands(display_frame, gestures, {})
                if lh and lh['pinch']['is_locked']:
                    if not reg_lock:
                        candidates = list(finalized_cards.values())
                        if len(candidates) >= 2:
                            candidates.sort(key=lambda x: (x['box'][2]-x['box'][0])*(x['box'][3]-x['box'][1]), reverse=True)
                            registered_hand = [to_treys(x['label']) for x in candidates[:2]]
                            reg_lock = True
                else:
                    reg_lock = False
                rh = gestures.get('right_hand')
                if rh and rh['pinch']['active']:
                    if rh['pinch']['is_locked']:
                        if not board_lock_once:
                            for lbl, data in finalized_cards.items():
                                t = to_treys(lbl)
                                if t and t not in board_cards and t not in registered_hand:
                                    board_cards.add(t)
                            board_lock_once = True
                    else:
                        board_lock_once = False
                else:
                    board_lock_once = False
                if rh and rh.get('two_finger_scroll', {}).get('active'):
                    poker_ar_ui.handle_scroll(rh['two_finger_scroll']['screen_y'])
                else:
                    poker_ar_ui.reset_interaction()
                poker_ar_ui.update_level(100 if registered_hand else 0)
                # Stability / finalize
                for lbl in current_frame_cards:
                    card_history[lbl] += 1
                for lbl in list(card_history.keys()):
                    if lbl not in current_frame_cards and lbl not in finalized_cards:
                        card_history[lbl] -= 1
                        if card_history[lbl] <= 0:
                            del card_history[lbl]
                for c in detected_list:
                    lbl = c['label']
                    if lbl not in finalized_cards and STABILITY_THRESHOLD <= card_history[lbl] <= FINALIZE_THRESHOLD:
                        finalized_cards[lbl] = {'label': lbl, 'confidence': c['confidence'], 'box': c['box'], 'last_seen': frame_count}
                for c in detected_list:
                    if c['label'] in finalized_cards:
                        finalized_cards[c['label']].update({'confidence': max(c['confidence'], finalized_cards[c['label']]['confidence']), 'box': c['box'], 'last_seen': frame_count})
                if len(detected_list) == 0:
                    zero_card_frames += 1
                else:
                    zero_card_frames = 0
                if zero_card_frames > NO_CARDS_RESET_TIMEOUT:
                    finalized_cards.clear(); card_history.clear()
                expired_cards = [lbl for lbl, data in finalized_cards.items() if frame_count - data.get('last_seen', 0) > CARD_FADE_TIMEOUT]
                for lbl in expired_cards:
                    del finalized_cards[lbl]
                stable_cards = {}
                for k, v in finalized_cards.items():
                    stable_cards[k] = v
                for c in detected_list:
                    lbl = c['label']
                    if lbl not in finalized_cards and card_history[lbl] >= STABILITY_THRESHOLD:
                        if lbl not in stable_cards or c['confidence'] > stable_cards[lbl]['confidence']:
                            stable_cards[lbl] = c
                stable_list = list(stable_cards.values())

                # --- LAYER 2: poker corner panels (equity, board, hands) ---
                display_frame = poker_ar_ui.render(display_frame)

                # --- LAYER 3: card bounding boxes — drawn LAST so always on top ---
                rh_set = set(registered_hand)
                for c in stable_list:
                    t = to_treys(c['label'])
                    x1, y1, x2, y2 = map(int, c['box'])
                    is_reg = t in rh_set; is_locked = t in board_cards
                    color = (0, 255, 0) if is_reg else (0, 165, 255) if is_locked else (255, 200, 0)
                    status = "ME" if is_reg else "LOCKED" if is_locked else "DETECTING"
                    thick = 3 if (is_reg or is_locked) else 2
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thick)
                    # Bold label with background for legibility over face panels
                    _lbl_txt = f"{c['label']} {status}"
                    (lw, lh_), _ = cv2.getTextSize(_lbl_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    cv2.rectangle(display_frame, (x1, y1-22), (x1+lw+6, y1), (0,0,0), -1)
                    cv2.putText(display_frame, _lbl_txt, (x1+3, y1-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            else:
                # --- FACE / HYBRID: Micro Expressions UI ---
                display_frame = micro_ar_controller.render(display_frame, engine.shared_state)
                if gesture_detector and gestures:
                    display_frame = gesture_detector.render_hands(display_frame, gestures, engine.shared_state)

            # === ADAPTIVE LEARNING INTEGRATION ===
            # (runs for face, hybrid, and hybrid_poker whenever face is detected)
            if learner:
                face_landmarks = engine.shared_state.get('landmarks')
                if face_landmarks:
                    learner.on_face_detected(face_landmarks)
                else:
                    learner.on_face_lost()

                hr        = engine.shared_state.get('heart_rate_bpm', 70)
                stress    = engine.shared_state.get('stress_score', 0)
                au_values = engine.shared_state.get('action_units', {}).copy()
                # Inject blink-rate and HRV deviations as synthetic AU entries so
                # they flow through BaselineExtractor → au_delta → PersonalityModel
                # without any interface changes.  PersonalityModel will accumulate
                # their bluff correlations alongside HR and stress slopes.
                au_values['blink_rate'] = engine.shared_state.get('blink_factor_raw',  0.0)
                au_values['hrv']        = engine.shared_state.get('hrv_deviation_raw', 0.0)

                # Flag: suppress all face-analysis UI when cards are on screen
                _cards_mode = (current_context == 'hybrid_poker')

                if learner.baseline.is_calibrating:
                    learner.add_calibration_sample(hr, stress, au_values)
                    if not _cards_mode:
                        progress = learner.baseline.get_calibration_progress()
                        # Calibration progress bar — top-centre
                        bar_x = w // 2 - 140
                        cv2.rectangle(display_frame, (bar_x, 55), (bar_x + 280, 80), (30, 30, 35), -1)
                        cv2.rectangle(display_frame, (bar_x, 55), (bar_x + 280, 80), (0, 210, 210), 1)
                        fill = int(2.8 * progress)
                        cv2.rectangle(display_frame, (bar_x, 55), (bar_x + fill, 80), (0, 210, 210), -1)
                        cv2.putText(display_frame, f"CALIBRATING  {progress}%",
                                    (bar_x + 8, 73),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 22), 1, cv2.LINE_AA)

                elif learner.baseline.has_baseline() and learner.current_player_id:
                    # Always call get_prediction to keep current_deviation fresh (needed
                    # by get_heuristic_prediction and the bandit's context bucket).
                    pred = learner.get_prediction(hr, stress, au_values)
                    _samples = pred.get('samples', 0) if pred else 0
                    enough_data = _samples >= MIN_PRED_SAMPLES

                    # === STREET-CHANGE DETECTION: lock one prediction per community card reveal ===
                    _cur_board_size = len(board_cards) if current_context == 'hybrid_poker' else 0
                    if _cur_board_size > _last_board_size:
                        _streets_map = {3: 'FLOP', 4: 'TURN', 5: 'RIVER'}
                        _street_name = _streets_map.get(_cur_board_size, f'{_cur_board_size}C')
                        if enough_data and pred:
                            _vp_s     = pred.get('p_bluff', 0.5)
                            _v_mode_s = f"TRAINED  {_samples}obs"
                            _v_prof_s = pred.get('personality_profile')
                        else:
                            _hf_s     = learner.get_heuristic_prediction()
                            _vp_s     = _hf_s['p_bluff'] if _hf_s else 0.5
                            _v_mode_s = "SIGNAL"
                            _v_prof_s = None
                        _locked_verdict = {
                            'prediction': 'BLUFFING' if _vp_s > 0.5 else 'STRONG HAND',
                            'p_bluff':    _vp_s,
                            'confidence': abs(_vp_s - 0.5) * 2,
                            'mode_tag':   _v_mode_s,
                            'profile':    _v_prof_s,
                            'street':     _street_name,
                        }
                    elif _cur_board_size == 0 and _last_board_size > 0:
                        _locked_verdict = None
                    _last_board_size = _cur_board_size

                    # === HYBRID_POKER COMPACT VERDICT OVERLAY ===
                    if _cards_mode and learner.current_player_id:
                        _hp_pw, _hp_ph = 220, 90
                        _hp_x = w - _hp_pw - 15
                        _hp_y = 10

                        _hp_clip = (_hp_x, _hp_y, _hp_pw, _hp_ph, 18)
                        if _locked_verdict:
                            _hp_pred   = _locked_verdict['prediction']
                            _hp_col    = (90, 100, 255) if _hp_pred == 'BLUFFING' else (120, 230, 145)
                            _hp_street = _locked_verdict.get('street', '')
                            _draw_polished_panel(display_frame, _hp_x, _hp_y,
                                                 _hp_pw, _hp_ph, accent=_hp_col)

                            _put_text_clipped(display_frame, f"OPPONENT  [{_hp_street}]",
                                              (_hp_x + 16, _hp_y + 19),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                                              (215, 213, 226), 1,
                                              clip_x=_hp_clip[0], clip_y=_hp_clip[1],
                                              clip_w=_hp_clip[2], clip_h=_hp_clip[3],
                                              clip_radius=_hp_clip[4])
                            _hp_glow = (_hp_col[0] // 4, _hp_col[1] // 4, _hp_col[2] // 4)
                            _put_text_clipped(display_frame, _hp_pred,
                                              (_hp_x + 17, _hp_y + 53),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                              _hp_glow, 3,
                                              clip_x=_hp_clip[0], clip_y=_hp_clip[1],
                                              clip_w=_hp_clip[2], clip_h=_hp_clip[3],
                                              clip_radius=_hp_clip[4])
                            _put_text_clipped(display_frame, _hp_pred,
                                              (_hp_x + 16, _hp_y + 52),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                              _hp_col, 2,
                                              clip_x=_hp_clip[0], clip_y=_hp_clip[1],
                                              clip_w=_hp_clip[2], clip_h=_hp_clip[3],
                                              clip_radius=_hp_clip[4])
                            _hp_conf = int(_locked_verdict['confidence'] * 100)
                            _put_text_clipped(display_frame, f"{_hp_conf}% conf  {_locked_verdict['mode_tag']}",
                                              (_hp_x + 16, _hp_y + 73),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.26,
                                              (165, 162, 178), 1,
                                              clip_x=_hp_clip[0], clip_y=_hp_clip[1],
                                              clip_w=_hp_clip[2], clip_h=_hp_clip[3],
                                              clip_radius=_hp_clip[4])
                        else:
                            _draw_polished_panel(display_frame, _hp_x, _hp_y,
                                                 _hp_pw, _hp_ph, accent=(155, 152, 172))
                            _put_text_clipped(display_frame, "OPPONENT",
                                              (_hp_x + 16, _hp_y + 19),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                                              (215, 213, 226), 1,
                                              clip_x=_hp_clip[0], clip_y=_hp_clip[1],
                                              clip_w=_hp_clip[2], clip_h=_hp_clip[3],
                                              clip_radius=_hp_clip[4])
                            _buf_f = len(learner._hand_buffer)
                            _put_text_clipped(display_frame,
                                              "READING..." if _buf_f < 15 else f"READING  {_buf_f}f",
                                              (_hp_x + 16, _hp_y + 52),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                                              (165, 162, 178), 1,
                                              clip_x=_hp_clip[0], clip_y=_hp_clip[1],
                                              clip_w=_hp_clip[2], clip_h=_hp_clip[3],
                                              clip_radius=_hp_clip[4])
                            _put_text_clipped(display_frame, "locks on flop / turn / river",
                                              (_hp_x + 16, _hp_y + 73),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.24,
                                              (130, 128, 146), 1,
                                              clip_x=_hp_clip[0], clip_y=_hp_clip[1],
                                              clip_w=_hp_clip[2], clip_h=_hp_clip[3],
                                              clip_radius=_hp_clip[4])

                    # === THUMB GESTURE: record showdown (face contexts only) ===
                    if not _cards_mode:
                        _now_sd = time.time()
                        if gestures and gesture_detector and (_now_sd - _last_showdown_time) > _SHOWDOWN_COOLDOWN:
                            for _hk_sd in ('left_hand', 'right_hand'):
                                _hd_sd = gestures.get(_hk_sd)
                                if not _hd_sd:
                                    continue
                                _ts = _hd_sd.get('thumb_signal', {})
                                if _ts.get('fired') and _ts.get('signal') in ('up', 'down'):
                                    _was_bluffing = (_ts['signal'] == 'down')

                                    # Compute the one verdict for this hand right before
                                    # resetting the buffer — uses the settled 80th-pct values.
                                    if enough_data and pred:
                                        _vp      = pred.get('p_bluff', 0.5)
                                        _v_mode  = f"TRAINED  {_samples}obs"
                                        _v_prof  = pred.get('personality_profile')
                                    else:
                                        _heur_final = learner.get_heuristic_prediction()
                                        _vp     = _heur_final['p_bluff'] if _heur_final else 0.5
                                        _v_mode = "SIGNAL"
                                        _v_prof = None
                                    _locked_verdict = {
                                        'prediction': 'BLUFFING' if _vp > 0.5 else 'STRONG HAND',
                                        'p_bluff':    _vp,
                                        'confidence': abs(_vp - 0.5) * 2,
                                        'mode_tag':   _v_mode,
                                        'profile':    _v_prof,
                                    }

                                    learner.on_showdown(was_bluffing=_was_bluffing)
                                    learner.start_hand()  # reset buffer for next hand
                                    _last_showdown_time = _now_sd
                                    _showdown_flash = (
                                        "BLUFFING recorded" if _was_bluffing else "STRONG HAND recorded",
                                        _now_sd
                                    )
                                    _hn = 'Left' if _hk_sd == 'left_hand' else 'Right'
                                    gesture_detector.reset_thumb_signal(_hn)
                                    break

                        # ── Opponent panel ────────────────────────────────────────────
                        # Anchored to the RIGHT of the face, but BELOW the HR/Stress panel
                        # (which lives at face-top-right). This avoids overlap.
                        _opp_pw = 230
                        _opp_ph = 120
                        _fbbox = _smoothed_face_bbox(face_landmarks, w, h)
                        if _fbbox is not None:
                            fbx, fby, fbw, fbh = _fbbox
                            # Right of face, lower half (analytics panel takes the upper half)
                            _opp_dx = min(w - _opp_pw - 10, fbx + fbw + 24)
                            _opp_dy = max(10, min(h - _opp_ph - 10,
                                                  fby + fbh - _opp_ph + 20))
                        else:
                            _opp_dx = w - _opp_pw - 15
                            _opp_dy = 360
                        panel_x, panel_y = _opp_dx, _opp_dy
                        panel_w, panel_h = _opp_pw, _opp_ph
                        _panel_rects['opponent'] = (panel_x, panel_y, panel_w, panel_h)

                        _opp_clip = (panel_x, panel_y, panel_w, panel_h, 18)
                        if _locked_verdict:
                            _pred_label = _locked_verdict['prediction']
                            _pred_color = (90, 100, 255) if _pred_label == 'BLUFFING' else (120, 230, 145)
                            _draw_polished_panel(display_frame, panel_x, panel_y,
                                                 panel_w, panel_h, accent=_pred_color)

                            _put_text_clipped(display_frame, "OPPONENT READ",
                                              (panel_x + 18, panel_y + 19),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                                              (215, 213, 226), 1,
                                              clip_x=_opp_clip[0], clip_y=_opp_clip[1],
                                              clip_w=_opp_clip[2], clip_h=_opp_clip[3],
                                              clip_radius=_opp_clip[4])
                            _put_text_clipped(display_frame, _locked_verdict['mode_tag'],
                                              (panel_x + panel_w - 88, panel_y + 19),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.26,
                                              (165, 162, 178), 1,
                                              clip_x=_opp_clip[0], clip_y=_opp_clip[1],
                                              clip_w=_opp_clip[2], clip_h=_opp_clip[3],
                                              clip_radius=_opp_clip[4])

                            # Pred label with subtle glow
                            _glow = (_pred_color[0] // 4, _pred_color[1] // 4, _pred_color[2] // 4)
                            _put_text_clipped(display_frame, _pred_label,
                                              (panel_x + 19, panel_y + 49),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                              _glow, 3,
                                              clip_x=_opp_clip[0], clip_y=_opp_clip[1],
                                              clip_w=_opp_clip[2], clip_h=_opp_clip[3],
                                              clip_radius=_opp_clip[4])
                            _put_text_clipped(display_frame, _pred_label,
                                              (panel_x + 18, panel_y + 48),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                              _pred_color, 2,
                                              clip_x=_opp_clip[0], clip_y=_opp_clip[1],
                                              clip_w=_opp_clip[2], clip_h=_opp_clip[3],
                                              clip_radius=_opp_clip[4])

                            # Confidence bar (rounded)
                            _conf_pct = int(_locked_verdict['confidence'] * 100)
                            _bar_x  = panel_x + 18
                            _bar_y  = panel_y + 72
                            _bar_w  = panel_w - 36
                            _bar_h  = 5
                            _bar_r  = _bar_h // 2
                            _stroke_rounded_rect(display_frame, _bar_x, _bar_y,
                                                 _bar_w, _bar_h, _bar_r,
                                                 (60, 58, 72), thickness=-1)
                            _fill_w = int(_bar_w * _locked_verdict['confidence'])
                            if _fill_w > _bar_r:
                                _stroke_rounded_rect(display_frame, _bar_x, _bar_y,
                                                     _fill_w, _bar_h, _bar_r,
                                                     _pred_color, thickness=-1)
                            _put_text_clipped(display_frame, f"{_conf_pct}% CONFIDENCE",
                                              (panel_x + 18, panel_y + 92),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.30,
                                              (215, 213, 226), 1,
                                              clip_x=_opp_clip[0], clip_y=_opp_clip[1],
                                              clip_w=_opp_clip[2], clip_h=_opp_clip[3],
                                              clip_radius=_opp_clip[4])

                            if _locked_verdict.get('profile'):
                                _put_text_clipped(display_frame,
                                                  _locked_verdict['profile'][:34],
                                                  (panel_x + 18, panel_y + 107),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                                                  (165, 162, 178), 1,
                                                  clip_x=_opp_clip[0], clip_y=_opp_clip[1],
                                                  clip_w=_opp_clip[2], clip_h=_opp_clip[3],
                                                  clip_radius=_opp_clip[4])

                        else:
                            # READING - collecting data for this hand
                            _draw_polished_panel(display_frame, panel_x, panel_y,
                                                 panel_w, panel_h, accent=(155, 152, 172))

                            _put_text_clipped(display_frame, "OPPONENT READ",
                                              (panel_x + 18, panel_y + 19),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                                              (215, 213, 226), 1,
                                              clip_x=_opp_clip[0], clip_y=_opp_clip[1],
                                              clip_w=_opp_clip[2], clip_h=_opp_clip[3],
                                              clip_radius=_opp_clip[4])

                            _buf_frames = len(learner._hand_buffer)
                            _read_tag = "READING..." if _buf_frames < 15 else f"READING  {_buf_frames}f"
                            _put_text_clipped(display_frame, _read_tag,
                                              (panel_x + 18, panel_y + 50),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                              (165, 162, 178), 1,
                                              clip_x=_opp_clip[0], clip_y=_opp_clip[1],
                                              clip_w=_opp_clip[2], clip_h=_opp_clip[3],
                                              clip_radius=_opp_clip[4])

                            _put_text_clipped(display_frame,
                                              "thumbs-up: strong  thumbs-down: bluff",
                                              (panel_x + 18, panel_y + 108),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.26,
                                              (130, 128, 146), 1,
                                              clip_x=_opp_clip[0], clip_y=_opp_clip[1],
                                              clip_w=_opp_clip[2], clip_h=_opp_clip[3],
                                              clip_radius=_opp_clip[4])

                        # Draw thumb gesture progress on screen
                        if gestures:
                            for _hk_td in ('left_hand', 'right_hand'):
                                _hd_td = gestures.get(_hk_td)
                                if not _hd_td:
                                    continue
                                _ts2 = _hd_td.get('thumb_signal', {})
                                if _ts2.get('signal') and _ts2.get('progress', 0) > 0:
                                    _tp = _ts2['tip_pos']
                                    _tx, _ty = int(_tp[0]), int(_tp[1])
                                    _prog = _ts2['progress']
                                    _sc = (110, 215, 130) if _ts2['signal'] == 'up' else (95, 95, 245)
                                    # outer faint track
                                    cv2.ellipse(display_frame, (_tx, _ty), (22, 22),
                                                -90, 0, 360, (44, 42, 52), 2, cv2.LINE_AA)
                                    # progress arc
                                    cv2.ellipse(display_frame, (_tx, _ty), (22, 22),
                                                -90, 0, int(360 * _prog), _sc, 3, cv2.LINE_AA)
                                    # glow dot
                                    cv2.circle(display_frame, (_tx, _ty), 9,
                                               (_sc[0] // 4, _sc[1] // 4, _sc[2] // 4),
                                               -1, cv2.LINE_AA)
                                    cv2.circle(display_frame, (_tx, _ty), 6, _sc, -1, cv2.LINE_AA)
                                    _lbl2 = "STRONG" if _ts2['signal'] == 'up' else "BLUFF"
                                    cv2.putText(display_frame, _lbl2,
                                                (_tx + 26, _ty + 5),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, _sc, 1, cv2.LINE_AA)

                        # Fist label — suppress any pinch visual and show FIST instead
                        if gestures:
                            for _hk_fi in ('left_hand', 'right_hand'):
                                _hd_fi = gestures.get(_hk_fi)
                                if _hd_fi and _hd_fi.get('fist'):
                                    _lms_fi = _hd_fi['landmarks']
                                    _wx = int(_lms_fi[0].x * w)
                                    _wy = int(_lms_fi[0].y * h)
                                    cv2.putText(display_frame, "FIST",
                                                (_wx - 20, _wy - 18),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 255), 2, cv2.LINE_AA)

                        # Showdown flash confirmation (1.5s)
                        if _showdown_flash and (time.time() - _showdown_flash[1]) < 1.5:
                            _fl = _showdown_flash[0]
                            (fw, fh), _ = cv2.getTextSize(_fl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            _fx = w // 2 - fw // 2
                            _fy = h // 2
                            cv2.rectangle(display_frame, (_fx - 10, _fy - fh - 8),
                                          (_fx + fw + 10, _fy + 10), (20, 20, 26), -1)
                            cv2.putText(display_frame, _fl, (_fx, _fy),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 150), 2, cv2.LINE_AA)
                        elif _showdown_flash and (time.time() - _showdown_flash[1]) >= 1.5:
                            _showdown_flash = None

            # === MINI POKER HUD (hybrid only — hybrid_poker uses full poker UI instead) ===
            if current_context == 'hybrid' and registered_hand:
                _render_mini_poker_hud(display_frame, registered_hand, board_cards, eq, poker_ar_ui, w, h)

        else:
            # === NO CONTEXT: Show hints ===
            cv2.putText(display_frame, "Point camera at:", (w//2 - 150, h//2 - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            cv2.putText(display_frame, "- Poker Cards for Hand Analysis", (w//2 - 180, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.putText(display_frame, "- Face for Stress Detection", (w//2 - 180, h//2 + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
            if gesture_detector:
                display_frame = gesture_detector.render_hands(display_frame, gestures, {})
            # Still show mini poker HUD if a hand was registered
            if registered_hand:
                _render_mini_poker_hud(display_frame, registered_hand, board_cards, eq, poker_ar_ui, w, h)
        
        # === PINCH HINT (cards visible — bottom-left, above context pill) ===
        if current_context in ('poker', 'hybrid_poker'):
            _hints = [
                ("L pinch", "save my hand", (80, 200, 255)),
                ("R pinch", "add board cards", (100, 255, 140)),
            ]
            _hint_x, _hint_y0 = 10, h - 95
            _hint_line_h = 22
            _hint_box_h = len(_hints) * _hint_line_h + 10
            _hint_box_w = 230
            _hov = display_frame.copy()
            cv2.rectangle(_hov, (_hint_x, _hint_y0 - 6),
                          (_hint_x + _hint_box_w, _hint_y0 + _hint_box_h - 6),
                          (20, 20, 24), -1)
            cv2.addWeighted(_hov, 0.75, display_frame, 0.25, 0, display_frame)
            cv2.rectangle(display_frame, (_hint_x, _hint_y0 - 6),
                          (_hint_x + _hint_box_w, _hint_y0 + _hint_box_h - 6),
                          (60, 60, 70), 1)
            for _i, (_key, _desc, _col) in enumerate(_hints):
                _ly = _hint_y0 + _i * _hint_line_h + 12
                cv2.putText(display_frame, _key,
                            (_hint_x + 8, _ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, _col, 1, cv2.LINE_AA)
                cv2.putText(display_frame, f"-> {_desc}",
                            (_hint_x + 72, _ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 170), 1, cv2.LINE_AA)

        # === CROSS-FADE TRANSITION ===
        if context_alpha < 1.0 and prev_display_frame is not None:
            display_frame = cv2.addWeighted(
                prev_display_frame, 1.0 - context_alpha,
                display_frame, context_alpha, 0
            )

        # === CONTEXT INDICATOR (small pill, bottom-right) ===
        context_colors = {
            'poker': (80, 165, 255),
            'face': (200, 220, 110),
            'hybrid': (220, 110, 165),
            'hybrid_poker': (220, 200, 80),
            'none': (118, 116, 130),
        }
        ctx_color = context_colors.get(current_context, (118, 116, 130))
        ctx_labels = {'poker': 'CARDS', 'face': 'FACE', 'hybrid': 'FACE+HAND', 'hybrid_poker': 'CARDS+FACE', 'none': 'IDLE'}
        ctx_label = ctx_labels.get(current_context, current_context.upper())
        (tw, th), _ = cv2.getTextSize(ctx_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        pill_x, pill_y = w - tw - 24, h - 18

        # Pill geometry (full rounded pill)
        _px0, _py0 = pill_x - 18, pill_y - th - 8
        _px1, _py1 = pill_x + tw + 10, pill_y + 8
        _pw, _ph = _px1 - _px0, _py1 - _py0
        _radius = _ph // 2

        # Frosted glass body (pill-shaped: radius == half-height)
        _draw_polished_panel(display_frame, _px0, _py0, _pw, _ph,
                             accent=None, alpha=0.55, radius=_radius)

        # Status dot (left of label)
        _dot_x = _px0 + 12
        _dot_y = (_py0 + _py1) // 2
        cv2.circle(display_frame, (_dot_x, _dot_y), 5,
                   (ctx_color[0] // 4, ctx_color[1] // 4, ctx_color[2] // 4),
                   -1, cv2.LINE_AA)
        cv2.circle(display_frame, (_dot_x, _dot_y), 3, ctx_color, -1, cv2.LINE_AA)

        # Label
        cv2.putText(display_frame, ctx_label, (pill_x + 4, pill_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, ctx_color, 1, cv2.LINE_AA)

        # Pending-switch progress bar (thin strip at top)
        if pending_frames > 0 and pending_context != current_context:
            bar_w = int(w * pending_frames / HYSTERESIS_FRAMES)
            bar_color = context_colors.get(pending_context, (200, 200, 200))
            cv2.rectangle(display_frame, (0, 0), (bar_w, 3), bar_color, -1)

        # === COLLECT PANEL RECTS FROM UI CONTROLLERS ===
        if micro_ar_controller.hr_analytics_rect:
            _panel_rects['hr_analytics'] = micro_ar_controller.hr_analytics_rect
        if micro_ar_controller.hr_graph_rect:
            _panel_rects['hr_graph'] = micro_ar_controller.hr_graph_rect
        if micro_ar_controller.expressions_rect:
            _panel_rects['expressions'] = micro_ar_controller.expressions_rect
        if poker_ar_ui.win_panel_rect:
            _panel_rects['poker_win'] = poker_ar_ui.win_panel_rect

        # Drag-handle circles removed: panels now anchor to the face automatically,
        # so manual drag is disabled. Keep _drag_handles empty so the drag-logic
        # block below safely no-ops.
        _drag_handles = {}

        # === PANEL DRAG LOGIC — pointing gesture only (index finger, no pinch) ===
        if _panel_positions and gestures:
            for _hk in ('left_hand', 'right_hand'):
                _hd = gestures.get(_hk)
                _pt = _hd.get('pointing', {}) if _hd else {}

                if not _pt.get('active'):
                    # Pointing dropped — apply grace period before resetting hover/drag
                    if _drag_hand == _hk and _drag_panel:
                        # Active drag: grace period before drop
                        _drag_hover_grace += 1
                        if _drag_hover_grace > DRAG_HOVER_GRACE_FRAMES:
                            _pr = _panel_rects.get(_drag_panel)
                            if _pr:
                                _cur = _panel_positions.get(_drag_panel, _pr[0], _pr[1])
                                _nr  = (_cur[0], _cur[1], _pr[2], _pr[3])
                                _reverted = False
                                for _opn, _or in _panel_rects.items():
                                    if _opn != _drag_panel and _rects_overlap(_nr, _or):
                                        _panel_positions.set(_drag_panel, _drag_orig_x, _drag_orig_y)
                                        _reverted = True
                                        break
                                if not _reverted:
                                    _panel_positions.save()
                            _drag_panel = None
                            _drag_hand  = None
                            _drag_hover_grace = 0
                    elif _drag_hover_name:
                        # Hovering but not dragging — grace period before hover reset
                        _drag_hover_grace += 1
                        if _drag_hover_grace > DRAG_HOVER_GRACE_FRAMES:
                            _drag_hover_name  = None
                            _drag_hover_grace = 0
                    continue

                # Pointing is active — reset grace counter
                _drag_hover_grace = 0

                _tip = _pt['index_tip_pos']
                _mx, _my = int(_tip[0]), int(_tip[1])

                if _drag_panel and _drag_hand == _hk:
                    # Move the panel — clamp to screen
                    _pr = _panel_rects.get(_drag_panel)
                    if _pr:
                        _pw, _ph = _pr[2], _pr[3]
                        _nx = max(0, min(w - _pw, _mx - _drag_offset_x))
                        _ny = max(0, min(h - _ph, _my - _drag_offset_y))
                        _panel_positions.set(_drag_panel, _nx, _ny)
                        # "Moving" label on the handle
                        _hhx, _hhy = _nx + _pw // 2, _ny
                        cv2.circle(display_frame, (_hhx, _hhy), 13, (0, 180, 220), -1)
                        cv2.putText(display_frame, "DROP",
                                    (_hhx - 12, _hhy + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)
                        # Draw fingertip cursor
                        cv2.circle(display_frame, (_mx, _my), 6, (0, 200, 255), -1, cv2.LINE_AA)

                else:
                    # Show fingertip cursor for this hand (both hands always get a cursor)
                    cv2.circle(display_frame, (_mx, _my), 5, (180, 200, 255), -1, cv2.LINE_AA)

                    if _drag_hand is None:
                        # Check if fingertip is near a handle (50px radius)
                        _hover = None
                        for _pn2, (_hx2, _hy2) in _drag_handles.items():
                            if ((_mx - _hx2) ** 2 + (_my - _hy2) ** 2) ** 0.5 < 50:
                                _hover = _pn2
                                break

                        if _hover:
                            if _drag_hover_name == _hover:
                                if time.time() - _drag_hover_start >= DRAG_DWELL:
                                    # Activate drag — keep _drag_hover_name so arc shows 100%
                                    _pr           = _panel_rects[_hover]
                                    _drag_orig_x  = _pr[0]
                                    _drag_orig_y  = _pr[1]
                                    _drag_panel   = _hover
                                    _drag_hand    = _hk
                                    _drag_offset_x = _mx - _pr[0]
                                    _drag_offset_y = _my - _pr[1]
                            else:
                                _drag_hover_name  = _hover
                                _drag_hover_start = time.time()
                        elif not _drag_hover_name:
                            pass  # nothing to clear

        # Only store the frame for cross-fade when a transition is active or imminent
        if context_alpha < 1.0 or pending_frames > HYSTERESIS_FRAMES - 3:
            prev_display_frame = display_frame.copy()

        # === DB RESET — fist gestures ===
        # Right fist held 3s  -> reset current player
        # Both fists held 5s  -> wipe all profiles
        _now_key = time.time()
        if gestures and learner:
            _rh = gestures.get('right_hand')
            _lh = gestures.get('left_hand')
            _right_fist = bool(_rh and _rh.get('fist'))
            _left_fist  = bool(_lh and _lh.get('fist'))
            _both_fist  = _right_fist and _left_fist

            # Both-fist timer (takes priority — checked first)
            if _both_fist:
                if _db_reset_all_start == 0.0:
                    _db_reset_all_start = _now_key
                _db_reset_user_start = 0.0  # cancel single-hand timer
            else:
                if _db_reset_all_start > 0:
                    _db_reset_all_start = 0.0  # broke the gesture

            # Single right-fist timer (only when left isn't also fist)
            if _right_fist and not _left_fist:
                if _db_reset_user_start == 0.0:
                    _db_reset_user_start = _now_key
            else:
                if not _both_fist:
                    _db_reset_user_start = 0.0

            # Draw progress bar and fire
            if _db_reset_all_start > 0:
                _prog_a = min(1.0, (_now_key - _db_reset_all_start) / 5.0)
                cv2.rectangle(display_frame, (0, h - 8), (int(w * _prog_a), h), (0, 30, 220), -1)
                cv2.putText(display_frame, f"  WIPE ALL PROFILES  {int(_prog_a*100)}%",
                            (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 120, 255), 1, cv2.LINE_AA)
                if _prog_a >= 1.0:
                    learner.delete_all_profiles()
                    _locked_verdict = None
                    _db_reset_all_start = 0.0
                    print("[DB] ALL profiles wiped")
            elif _db_reset_user_start > 0:
                _prog_u = min(1.0, (_now_key - _db_reset_user_start) / 3.0)
                cv2.rectangle(display_frame, (0, h - 8), (int(w * _prog_u), h), (60, 60, 200), -1)
                cv2.putText(display_frame, f"  Reset this player  {int(_prog_u*100)}%",
                            (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 255), 1, cv2.LINE_AA)
                if _prog_u >= 1.0:
                    learner.reset_current_player()
                    _locked_verdict = None
                    _db_reset_user_start = 0.0
                    print("[DB] Current player reset")

        cv2.imshow("Unified AR System", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            board_cards.clear()
            registered_hand.clear()
            print("Board and hand cleared manually (Press 'c')")
        elif key == ord('b') and learner and current_context in ('face', 'hybrid'):
            learner.start_calibration()
            print("[Adaptive] Manual recalibration started")
        elif key == ord('r') and _panel_positions:
            _panel_positions.reset_all()
            print("[Panels] All panel positions reset to defaults")
        # s/n/d/D keyboard showdown/reset removed — use thumb/fist gestures
    
    cap.release()
    cv2.destroyAllWindows()
    
    if gesture_detector:
        gesture_detector.cleanup()
    
    for module in engine.modules.values():
        if hasattr(module, 'cleanup'):
            module.cleanup()
    
    # Cleanup adaptive learning
    if learner:
        learner.close()
        print("[Adaptive] Learning system closed")


if __name__ == "__main__":
    main()
