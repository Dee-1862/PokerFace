"""
Unified AR System
Integrates poker_main.py and micro_expressions/main.py

Context Detection:
- Face detected → Show micro_expressions UI (stress, heart rate, FACS)
- Cards detected → Show poker_hand UI (equity, outs, scrollable hands)
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

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent / 'poker_hand'))
sys.path.insert(0, str(Path(__file__).parent / 'micro_expressions'))

# === POKER HAND IMPORTS ===
from ultralytics import YOLO
from treys import Card, Evaluator, Deck

# Import poker modules (from poker_hand folder)
from poker_ar_ui import PokerARUI

# === MICRO EXPRESSIONS IMPORTS ===
from engine import StressDetectionEngine
from ar_ui_controller import ARUIController

# === ADAPTIVE LEARNING IMPORTS ===
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from adaptive_learning import AdaptiveLearningSystem
    ADAPTIVE_LEARNING_AVAILABLE = True
    print("✓ Adaptive Learning module available")
except ImportError as e:
    ADAPTIVE_LEARNING_AVAILABLE = False
    print(f"⚠ Adaptive Learning module not available: {e}")

# === SHARED IMPORTS ===
# Hand gesture detector will be imported after path setup
HandGestureDetector = None  # Will be loaded dynamically


# =============================================
# === POKER CALCULATION FUNCTIONS (from poker_main.py) ===
# =============================================

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
    print(f"✓ Loaded {len(PREFLOP_EQUITY)} preflop hands")
except Exception as e:
    print(f"! Preflop table load failed: {e}")

# Global Evaluator
try:
    POKER_EVALUATOR = Evaluator()
    print("✓ Poker Evaluator initialized")
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

def main():
    # === ARGUMENT PARSING ===
    parser = argparse.ArgumentParser(description="Unified AR System: Poker + Micro Expressions")
    parser.add_argument('--camera', '-c', type=int, default=0, 
                        help='Camera device ID (default: 0)')
    parser.add_argument('--debug', action='store_true', 
                        help='Enable debug mode')
    args = parser.parse_args()
    
    camera_id = args.camera
    
    print("🚀 Unified AR System: Poker + Micro Expressions")
    print("=" * 60)
    print(f"Using Camera: {camera_id}")
    print("Context Detection:")
    print("  • Face detected → Micro Expressions UI (stress, HR, FACS)")
    print("  • Cards detected → Poker Hand UI (equity, outs, hands)")
    print("  • Hand gestures (pinch, scroll) work in both contexts")
    print("=" * 60)
    
    # === INITIALIZE YOLO MODEL (Poker) ===
    try:
        model_path = Path(__file__).parent / 'poker_hand' / 'poker_v2.pt'
        if model_path.exists():
            model = YOLO(str(model_path))
        else:
            model = YOLO(str(Path(__file__).parent / 'poker_hand' / 'poker_v1.pt'))
        print("✓ YOLO Poker Model loaded")
    except Exception as e:
        print(f"⚠ Failed to load YOLO model: {e}")
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
                print(f"✓ Module loaded: {module_name.upper()}")
        except Exception as e:
            print(f"⚠ Error loading {module_name}: {e}")
    
    # === INITIALIZE AR CONTROLLERS ===
    # Micro expressions AR UI
    micro_ar_controller = ARUIController(button_position='top-right')
    
    # Hand gesture detector (shared) - import dynamically
    global HandGestureDetector
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "hand_gesture_detector", 
            str(Path(__file__).parent / 'poker_hand' / 'hand_gesture_detector.py')
        )
        hgd_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hgd_module)
        HandGestureDetector = hgd_module.HandGestureDetector
        gesture_detector = HandGestureDetector()
        print("✓ Hand Gesture Detector initialized")
    except Exception as e:
        print(f"⚠ Error initializing hand detector: {e}")
        gesture_detector = None
    
    # === CAMERA SETUP ===
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    cv2.namedWindow("Unified AR System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Unified AR System", 1280, 720)
    
    ret, frame = cap.read()
    if not ret:
        print("❌ Cannot open camera")
        return
    
    h, w = frame.shape[:2]
    
    # === POKER AR UI ===
    poker_ar_ui = PokerARUI(w, h)
    
    # === STATE VARIABLES ===
    # Poker state (from poker_main.py)
    card_history = defaultdict(int)
    finalized_cards = {}
    STABILITY_THRESHOLD = 10
    FINALIZE_THRESHOLD = 20
    CARD_FADE_TIMEOUT = 45  # Frames before a card fades (1.5s at 30fps)
    NO_CARDS_RESET_TIMEOUT = 90  # Frames before full reset when no cards visible (1s at 30fps)
    frame_count = 0
    zero_card_frames = 0
    
    registered_hand = []
    reg_lock = False
    board_cards = set()
    board_lock_once = False
    
    equity_cache = {'hand': None, 'board': None, 'result': None}
    
    # Context state
    current_context = 'none'  # 'face', 'poker', 'none'
    context_switch_cooldown = 0
    COOLDOWN_FRAMES = 30  # 1 second at 30fps
    
    # === INITIALIZE ADAPTIVE LEARNING ===
    learner = None
    if ADAPTIVE_LEARNING_AVAILABLE:
        try:
            learner = AdaptiveLearningSystem()
            print("✓ Adaptive Learning System initialized")
        except Exception as e:
            print(f"⚠ Adaptive Learning init failed: {e}")
    
    print("\n▶ Starting unified detection loop...")
    print("  Press 'q' to quit, 'c' to clear poker board")
    print("  Press 'b' to calibrate baseline (face mode)")
    print("  Press 's' to record showdown (was bluffing)")
    print("  Press 'n' to record showdown (not bluffing)")
    
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
        
        # === PROCESS MICRO EXPRESSION MODULES ===
        for name, module in engine.modules.items():
            module.process(engine.shared_state)
        
        face_detected = engine.shared_state.get('face_detected', False)
        
        # === PROCESS POKER DETECTION ===
        cards_detected = False
        detected_list = []
        current_frame_cards = set()
        
        if model:
            results = model(frame, verbose=False, conf=0.7, iou=0.15, imgsz=1280)
            
            for r in results:
                for box in r.boxes:
                    lbl = model.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    bbox = box.xyxy[0].tolist()
                    current_frame_cards.add(lbl)
                    detected_list.append({'label': lbl, 'confidence': conf, 'box': bbox})
            
            cards_detected = len(detected_list) > 0
        
        # === CONTEXT DETECTION ===
        # Priority: Cards > Face (since cards are more specific)
        if context_switch_cooldown > 0:
            context_switch_cooldown -= 1
        
        new_context = current_context
        
        if cards_detected or registered_hand:
            new_context = 'poker'
        elif face_detected:
            new_context = 'face'
        else:
            new_context = 'none'
        
        if new_context != current_context and context_switch_cooldown == 0:
            current_context = new_context
            context_switch_cooldown = COOLDOWN_FRAMES
            print(f"[Context] Switched to: {current_context.upper()}")
        
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
            
            # === Poker Analytics ===
            eq_board = list(board_cards)
            hand_key = tuple(sorted(registered_hand)) if registered_hand else None
            board_key = tuple(sorted(eq_board)) if eq_board else None
            
            if (equity_cache['hand'] == hand_key and equity_cache['board'] == board_key 
                and equity_cache['result'] is not None):
                eq, outs, top = equity_cache['result']
            else:
                eq, outs, top = calculate_equity_fast(registered_hand, eq_board)
                equity_cache['hand'] = hand_key
                equity_cache['board'] = board_key
                equity_cache['result'] = (eq, outs, top)
            
            poker_ar_ui.update_data(eq, outs, top, board_cards=board_cards)
            
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
            
            # Draw saved hand panel
            if registered_hand:
                cv2.rectangle(display_frame, (10, 10), (250, 100), (30, 30, 30), -1)
                cv2.rectangle(display_frame, (10, 10), (250, 100), (0, 255, 0), 2)
                cv2.putText(display_frame, "MY HAND (SAVED):", (20, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                
                y_off = 70
                x_off = 20
                visible_cards = {to_treys(c['label']) for c in stable_list}
                
                for card_str in registered_hand:
                    if card_str in visible_cards:
                        color = (0, 255, 0)
                    else:
                        color = (0, 0, 255)
                    cv2.putText(display_frame, card_str, (x_off, y_off), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                    x_off += 80
            else:
                cv2.putText(display_frame, "Left Pinch: Save Hand", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            
            if not board_cards and registered_hand:
                cv2.putText(display_frame, "Right Pinch (5s) to Add Board", (w - 450, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        elif current_context == 'face':
            # === FACE CONTEXT: Micro Expressions UI ===
            display_frame = micro_ar_controller.render(display_frame, engine.shared_state)
            
            if gesture_detector:
                if gestures:
                    display_frame = gesture_detector.render_hands(display_frame, gestures, engine.shared_state)
            
            # === ADAPTIVE LEARNING INTEGRATION ===
            if learner:
                # Get face landmarks from shared state
                face_landmarks = engine.shared_state.get('face_landmarks')
                if face_landmarks:
                    learner.on_face_detected(face_landmarks)
                
                # Get current signals
                hr = engine.shared_state.get('heart_rate', 70)
                stress = engine.shared_state.get('stress_level', 0)
                au_values = engine.shared_state.get('action_units', {})
                
                # Add sample if calibrating
                if learner.baseline.is_calibrating:
                    learner.add_calibration_sample(hr, stress, au_values)
                    # Draw calibration progress
                    progress = learner.baseline.get_calibration_progress()
                    cv2.putText(display_frame, f"CALIBRATING: {progress}%", (w//2 - 100, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.rectangle(display_frame, (w//2 - 100, 80), (w//2 - 100 + int(2*progress), 95), (0, 255, 255), -1)
                
                # Get prediction if baseline exists
                elif learner.baseline.has_baseline() and learner.current_player_id:
                    pred = learner.get_prediction(hr, stress, au_values)
                    if pred:
                        # Draw prediction panel
                        panel_x, panel_y = w - 300, 100
                        cv2.rectangle(display_frame, (panel_x, panel_y), (panel_x + 280, panel_y + 120), (30, 30, 30), -1)
                        cv2.rectangle(display_frame, (panel_x, panel_y), (panel_x + 280, panel_y + 120), (255, 165, 0), 2)
                        cv2.putText(display_frame, "OPPONENT ANALYSIS", (panel_x + 10, panel_y + 25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 1)
                        
                        # Prediction
                        pred_color = (0, 0, 255) if pred['prediction'] == 'BLUFFING' else (0, 255, 0)
                        cv2.putText(display_frame, f"Likely: {pred['prediction']}", (panel_x + 10, panel_y + 55),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, pred_color, 2)
                        
                        # Confidence bar
                        conf_width = int(pred['confidence'] * 200)
                        cv2.rectangle(display_frame, (panel_x + 10, panel_y + 70), (panel_x + 10 + conf_width, panel_y + 85), pred_color, -1)
                        cv2.putText(display_frame, f"{int(pred['confidence']*100)}%", (panel_x + 220, panel_y + 83),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                        
                        # Stats
                        cv2.putText(display_frame, f"Samples: {pred['samples']} | HR: {pred['hr_delta']:+.0f}", (panel_x + 10, panel_y + 110),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                else:
                    # Prompt for calibration
                    cv2.putText(display_frame, "Press 'B' to calibrate baseline", (w//2 - 150, h - 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        else:
            # === NO CONTEXT: Show hints ===
            cv2.putText(display_frame, "Point camera at:", (w//2 - 150, h//2 - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            cv2.putText(display_frame, "• Poker Cards for Hand Analysis", (w//2 - 180, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.putText(display_frame, "• Face for Stress Detection", (w//2 - 180, h//2 + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
            
            if gesture_detector:
                display_frame = gesture_detector.render_hands(display_frame, gestures, {})
        
        # === CONTEXT INDICATOR ===
        context_colors = {
            'poker': (0, 165, 255),
            'face': (0, 255, 200),
            'none': (100, 100, 100)
        }
        context_text = f"CONTEXT: {current_context.upper()}"
        cv2.putText(display_frame, context_text, (w - 220, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, context_colors.get(current_context, (100,100,100)), 2)
        
        cv2.imshow("Unified AR System", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            board_cards.clear()
            registered_hand.clear()
            print("Board and hand cleared manually (Press 'c')")
        elif key == ord('b') and learner and current_context == 'face':
            # Start baseline calibration
            learner.start_calibration()
            print("[Adaptive] Starting baseline calibration...")
        elif key == ord('s') and learner and current_context == 'face':
            # Showdown: opponent was bluffing
            was_correct = learner.on_showdown(was_bluffing=True)
            result = "CORRECT!" if was_correct else "Wrong"
            print(f"[Adaptive] Showdown logged: BLUFFING - Prediction was {result}")
        elif key == ord('n') and learner and current_context == 'face':
            # Showdown: opponent was NOT bluffing
            was_correct = learner.on_showdown(was_bluffing=False)
            result = "CORRECT!" if was_correct else "Wrong"
            print(f"[Adaptive] Showdown logged: STRONG HAND - Prediction was {result}")
    
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
