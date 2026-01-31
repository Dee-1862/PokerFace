
"""
Unified Poker Detection System
(Logic from simple_card_detector.py + AR Features)
"""

import cv2
import argparse
from ultralytics import YOLO
from collections import defaultdict
from treys import Card, Evaluator, Deck
from poker_ar_ui import PokerARUI
from hand_gesture_detector import HandGestureDetector
import numpy as np

# Helper: Convert YOLO label to Treys
def to_treys(label):
    if len(label) < 2: return None
    rank = label[:-1]
    suit = label[-1].lower()
    if rank == '10': rank = 'T'
    # Validate suit and rank
    if suit not in ['h', 'd', 'c', 's']: return None
    if rank not in ['2','3','4','5','6','7','8','9','T','J','Q','K','A']: return None
    return f"{rank}{suit}"

# Load Preflop Equity Table
PREFLOP_EQUITY = {}
try:
    import csv
    with open('preflop_equity.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            PREFLOP_EQUITY[row['hand']] = float(row['equity'])
    print(f"✓ Loaded {len(PREFLOP_EQUITY)} preflop hands")
except Exception as e:
    print(f"! Preflop table load failed: {e}")

# Global Evaluator (expensive to initialize, so do it once)
try:
    POKER_EVALUATOR = Evaluator()
    print("✓ Poker Evaluator initialized")
except Exception as e:
    print(f"! Evaluator init failed: {e}")
    POKER_EVALUATOR = None

def normalize_hand(my_hand):
    """Convert ['Ah', 'Kc'] to 'AKo' or 'AKs' for lookup"""
    if len(my_hand) != 2:
        return None
    
    # Extract ranks and suits
    ranks = [c[0] for c in my_hand]
    suits = [c[1] for c in my_hand]
    
    # Sort ranks by poker value
    rank_order = 'AKQJT98765432'
    sorted_ranks = sorted(ranks, key=lambda r: rank_order.index(r))
    
    # Determine if suited
    suited = 's' if suits[0] == suits[1] else 'o'
    
    # Handle pairs
    if sorted_ranks[0] == sorted_ranks[1]:
        return sorted_ranks[0] + sorted_ranks[1]
    
    # Non-pairs
    return sorted_ranks[0] + sorted_ranks[1] + suited

# Helper: Fast Equity Calculator (Hybrid Approach)
def calculate_equity_fast(my_hand, board_cards):
    """
    Hybrid equity calculator:
    - Preflop (0 board): Lookup table (instant)
    - Flop/Turn (3-4 board): Exhaustive enumeration (exact)
    - River (5 board): Direct evaluation (instant)
    """
    
    if POKER_EVALUATOR is None:
        return 0.0, 0, {'my_hands': [], 'opp_hands': []}
    
    if not my_hand or len(my_hand) != 2:
        return 0.0, 0, {'my_hands': [], 'opp_hands': []}
    
    board_size = len(board_cards)
    
    try:
        # PREFLOP: Lookup table
        if board_size == 0:
            hand_key = normalize_hand(my_hand)
            if hand_key and hand_key in PREFLOP_EQUITY:
                equity = PREFLOP_EQUITY[hand_key]
                print(f"[Preflop Lookup] {hand_key}: {equity:.1f}%")
                # Return basic hand types for preflop
                is_pair = my_hand[0][0] == my_hand[1][0]
                my_hands = [
                    {'name': 'Pair' if is_pair else 'High Card', 'prob': 1.0, 'cards': my_hand}
                ]
                opp_hands = [
                    {'name': 'Pair', 'prob': 0.06, 'cards': ['As', 'Ac']},
                    {'name': 'High Card', 'prob': 0.94, 'cards': ['Ah', 'Kh']}
                ]
                return equity, 0, {'my_hands': my_hands, 'opp_hands': opp_hands}
            else:
                print(f"[Preflop] Hand {my_hand} not in table, using 50%")
                return 50.0, 0, {'my_hands': [], 'opp_hands': []}
        
        # FLOP/TURN: Exhaustive enumeration
        elif board_size in [3, 4]:
            return exhaustive_enumeration(my_hand, board_cards)
        
        # RIVER: Direct showdown evaluation
        else:
            return evaluate_river(my_hand, board_cards)
            
    except Exception as e:
        print(f"[ERROR] calculate_equity_fast: {e}")
        return 0.0, 0, {'my_hands': [], 'opp_hands': []}

def get_best_5_cards(all_cards_ints):
    """Find the best 5-card hand from >=5 cards."""
    from itertools import combinations
    evaluator = POKER_EVALUATOR
    if not evaluator: return [Card.int_to_str(c) for c in all_cards_ints[:5]]
    
    min_score = float('inf')
    best_hand = None
    
    # Try all combinations of 5
    for five_cards in combinations(all_cards_ints, 5):
        score = evaluator.evaluate(list(five_cards), [])
        if score < min_score:
            min_score = score
            best_hand = list(five_cards)
            
    if best_hand:
        return [Card.int_to_str(c) for c in best_hand]
    return [Card.int_to_str(c) for c in all_cards_ints[:5]]

def exhaustive_enumeration(my_hand, board_cards):
    """Evaluate ALL possible outcomes (100% accurate)"""
    from itertools import combinations
    
    evaluator = POKER_EVALUATOR
    
    # Convert to Treys ints
    hero_hand = [Card.new(c) for c in my_hand]
    board = [Card.new(c) for c in board_cards]
    
    # Build remaining deck
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
    
    print(f"[Exhaustive] Evaluating {len(deck.cards)} remaining cards, need {cards_needed} more")
    
    hero_hand_examples = {}
    opp_hand_examples = {}
    
    # ALL possible runouts
    for runout in combinations(deck.cards, cards_needed):
        remaining = [c for c in deck.cards if c not in runout]
        
        # Sample opponent hands (limit to avoid explosion)
        opp_sample = list(combinations(remaining, 2))
        if len(opp_sample) > 200:  # Limit for performance
            import random
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
                        # Store BEST 5 cards
                        hero_hand_examples[hero_class_str] = get_best_5_cards(hero_hand + full_board)
                else:
                    opp_hand_counts[opp_class_str] += 1
                    if opp_class_str not in opp_hand_examples:
                        opp_hand_examples[opp_class_str] = get_best_5_cards(list(opp_hand) + full_board)
            except:
                pass
    
    equity = (wins / total * 100) if total > 0 else 0.0
    
    # Format hero hands
    my_hands = []
    for h_name, count in hero_hand_counts.items():
        prob = count / total if total > 0 else 0
        if prob > 0.01:
            my_hands.append({
                'name': h_name, 
                'prob': prob,
                'cards': hero_hand_examples.get(h_name, [])[:7] 
            })
    my_hands.sort(key=lambda x: x['prob'], reverse=True)
    
    # Format opponent hands
    opp_hands = []
    for h_name, count in opp_hand_counts.items():
        prob = count / total if total > 0 else 0
        if prob > 0.01:
            opp_hands.append({
                'name': h_name, 
                'prob': prob,
                'cards': opp_hand_examples.get(h_name, [])[:7]
            })
    opp_hands.sort(key=lambda x: x['prob'], reverse=True)
    
    print(f"[Exhaustive] {equity:.1f}% ({wins}/{total}), My: {len(my_hands)}, Opp: {len(opp_hands)}")
    
    return equity, 0, {'my_hands': my_hands, 'opp_hands': opp_hands}

def evaluate_river(my_hand, board_cards):
    """River: Just evaluate against all possible opponent hands"""
    evaluator = POKER_EVALUATOR
    
    hero_hand = [Card.new(c) for c in my_hand]
    board = [Card.new(c) for c in board_cards]
    
    deck = Deck()
    known = hero_hand + board
    for card in known:
        if card in deck.cards:
            deck.cards.remove(card)
    
    from itertools import combinations
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
            my_hands.append({
                'name': h_name, 
                'prob': prob,
                'cards': hero_hand_examples.get(h_name, [])[:7]
            })
    my_hands.sort(key=lambda x: x['prob'], reverse=True)
    
    opp_hands = []
    for h_name, count in opp_hand_counts.items():
        prob = count / total if total > 0 else 0
        if prob > 0.01:
            opp_hands.append({
                'name': h_name, 
                'prob': prob,
                'cards': opp_hand_examples.get(h_name, [])[:7]
            })
    opp_hands.sort(key=lambda x: x['prob'], reverse=True)
    
    print(f"[River] {equity:.1f}%, My: {len(my_hands)}, Opp: {len(opp_hands)}")
    
    return equity, 0, {'my_hands': my_hands, 'opp_hands': opp_hands}

def main():
    print("🚀 Unified Poker System (Simple Logic + AR)")
    
    # 1. Load Model (Simple)
    try:
        model = YOLO('poker_v2.pt')
    except:
        print("Fallback to v1")
        model = YOLO('poker_v1.pt')
        
    # 2. Setup AR
    try:
        gesture_detector = HandGestureDetector()
        ar_active = True
    except:
        gesture_detector = None
        ar_active = False
        
    cap = cv2.VideoCapture(0)
    # Use 720p for better performance and screen fit
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    cv2.namedWindow("Poker AR (Simple)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Poker AR (Simple)", 1280, 720)
    
    ret, frame = cap.read()
    if not ret: return
    h, w = frame.shape[:2]
    ar_ui = PokerARUI(w, h)
    
    # --- LOGIC FROM simple_card_detector.py ---
    card_history = defaultdict(int)
    finalized_cards = {}
    # Stability Tuning
    STABILITY_THRESHOLD = 10 # Increase to 10 for solid lock
    FINALIZE_THRESHOLD = 20
    FADE = 60 # Keep cards for 2 seconds after loss
    frame_count = 0
    zero_card_frames = 0
    
    # AR State
    registered_hand = [] 
    reg_lock = False
    
    # Cumulative Board State
    board_cards = set()
    board_lock_once = False
    
    # Equity Cache (prevent recalculation)
    equity_cache = {
        'hand': None,
        'board': None,
        'result': None
    }
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        
        # 1. AR Hand (Process First)
        gestures = {}
        # We need detection detections for Grip Rejection, but we haven't run YOLO yet.
        # Let's run YOLO first!
        
        # --- 2. Simple Detection Loop (Moved Up) ---
        results = model(frame, verbose=False, conf=0.5, iou=0.15, imgsz=1280)
        
        current_frame = set()
        detected_list = []
        
        for r in results:
            for box in r.boxes:
                lbl = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                bbox = box.xyxy[0].tolist()
                
                current_frame.add(lbl)
                detected_list.append({'label':lbl, 'confidence':conf, 'box':bbox})
        
        # Helper: Check if point is in any card box
        def is_touching_card(point, cards):
             px, py = point
             for c in cards:
                 x1, y1, x2, y2 = c['box']
                 # Pad box slightly
                 if (x1-20 < px < x2+20) and (y1-20 < py < y2+20):
                     return True
             return False

        # --- 1. AR Gestures (Now with Grip Rejection) ---
        if ar_active:
            gestures = gesture_detector.process(frame, {})
            # Filter gestures intersecting with cards
            
            # Left Hand
            lh = gestures.get('left_hand')
            if lh and lh['pinch']['active']:
                 thumb_pt = lh['pinch']['thumb_pos']
                 if is_touching_card(thumb_pt, detected_list):
                     lh['pinch']['active'] = False
                     lh['pinch']['is_locked'] = False
                     
            frame = gesture_detector.render_hands(frame, gestures, {})
            
            # Left Hand Registration
            if lh and lh['pinch']['is_locked']:
                if not reg_lock:
                    candidates = list(finalized_cards.values())
                    if len(candidates) >= 2:
                        candidates.sort(key=lambda x: (x['box'][2]-x['box'][0])*(x['box'][3]-x['box'][1]), reverse=True)
                        registered_hand = [to_treys(x['label']) for x in candidates[:2]]
                        reg_lock = True
            else:
                reg_lock = False
            
            # Right Hand: Board Lock (Pinch) + Scroll (Two-Finger)
            rh = gestures.get('right_hand')
            
            # 1. Pinch: Lock board cards (5s hold)
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

            # 2. Two-Finger Scroll: Navigate winning hands (independent of pinch)
            if rh and rh.get('two_finger_scroll', {}).get('active'):
                ar_ui.handle_scroll(rh['two_finger_scroll']['screen_y'])
            else:
                ar_ui.reset_interaction()
        
        # UI Visiblity: Always show if hand is registered
        if registered_hand:
            ar_ui.update_level(100)
        else:
            ar_ui.update_level(0)

        # --- Stability Logic ---
                
        # History Update
        for lbl in current_frame:
            card_history[lbl] += 1
        
        for lbl in list(card_history.keys()):
            if lbl not in current_frame and lbl not in finalized_cards:
                card_history[lbl] -= 1
                if card_history[lbl] <= 0: del card_history[lbl]
                
        # Finalize
        for c in detected_list:
            lbl = c['label']
            if lbl not in finalized_cards and STABILITY_THRESHOLD <= card_history[lbl] <= FINALIZE_THRESHOLD:
                finalized_cards[lbl] = {
                    'label': lbl, 
                    'confidence': c['confidence'],
                    'box': c['box'],
                    'last_seen': frame_count
                }
                
        # Update Finalized
        for c in detected_list:
            if c['label'] in finalized_cards:
                finalized_cards[c['label']].update({
                    'confidence': max(c['confidence'], finalized_cards[c['label']]['confidence']),
                    'box': c['box'],
                    'last_seen': frame_count
                })
        
        # Sticky Board Logic:
        # We DO NOT expire finalized cards based on time anymore (Infinity Fade).
        # We reset EVERYTHING if no cards are seen for 3 seconds (90 frames).
        if len(detected_list) == 0:
            zero_card_frames += 1
        else:
            zero_card_frames = 0
            
        if zero_card_frames > 90: # 3 seconds
             finalized_cards.clear()
             card_history.clear()
             # DO NOT clear board_cards (persistence for Flop/Turn/River)
             if zero_card_frames == 91:
                 print("Visuals Reset (No cards visible)")
        
        # Build Display List (Stable Cards)
        stable_cards = {}
        for k, v in finalized_cards.items(): stable_cards[k] = v
        for c in detected_list:
            lbl = c['label']
            if lbl not in finalized_cards and card_history[lbl] >= STABILITY_THRESHOLD:
                if lbl not in stable_cards or c['confidence'] > stable_cards[lbl]['confidence']:
                    stable_cards[lbl] = c
        
        stable_list = list(stable_cards.values())
        
        # 3. Poker Analytics (Use Persistent Board + Caching)
        eq_board = list(board_cards) 
        
        # Check cache to avoid recalculation
        hand_key = tuple(sorted(registered_hand)) if registered_hand else None
        board_key = tuple(sorted(eq_board)) if eq_board else None
        
        if (equity_cache['hand'] == hand_key and equity_cache['board'] == board_key 
            and equity_cache['result'] is not None):
            # Use cached result
            eq, outs, top = equity_cache['result']
        else:
            # Calculate fresh (only when cards change)
            eq, outs, top = calculate_equity_fast(registered_hand, eq_board)
            equity_cache['hand'] = hand_key
            equity_cache['board'] = board_key
            equity_cache['result'] = (eq, outs, top)
        
        ar_ui.update_data(eq, outs, top, board_cards=board_cards)
        
        # 4. Render
        frame = ar_ui.render(frame)
        
        # Draw Cards (Auto-Lock Visualization)
        # We draw what we SEE (stable_list).
        # But we color it based on whether it is "Locked" in board_cards.
        
        rh_set = set(registered_hand)
        
        for c in stable_list:
            t = to_treys(c['label'])
            x1, y1, x2, y2 = map(int, c['box'])
            
            is_reg = t in rh_set
            is_locked = t in board_cards
            
            if is_reg:
                color = (0, 255, 0) # Green (My Hand)
                status = "ME"
                thick = 3
            elif is_locked:
                color = (0, 165, 255) # Orange (Locked Board)
                status = "LOCKED"
                thick = 3
            else:
                color = (255, 200, 0) # Blue/Cyan (Candidate)
                status = "DETECTING"
                thick = 1
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
            cv2.putText(frame, f"{c['label']} {status}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        # UI Hints
        if rh and rh['pinch']['active']:
             if not rh['pinch']['is_locked']:
                 # Progress Bar?
                 prog = rh['pinch'].get('hold_progress', 0)
                 # Show "HOLD TO LOCK" near hand
                 pt = tuple(rh['pinch']['index_pos'].astype(int))
                 cv2.putText(frame, f"HOLD TO ADD... {int(prog*100)}%", (pt[0]+20, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                 
        if not board_cards and not registered_hand:
             cv2.putText(frame, "1. Left Pinch: Save Hand", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
             cv2.putText(frame, "2. Right Pinch (5s): Add Board Cards", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
             cv2.putText(frame, "3. Press 'c' to Clear Board", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
        # Draw Saved Hand UI (Persistent Display)
        if registered_hand:
            # Draw a panel in top-left
            cv2.rectangle(frame, (10, 10), (250, 100), (30, 30, 30), -1)
            cv2.rectangle(frame, (10, 10), (250, 100), (0, 255, 0), 2)
            cv2.putText(frame, "MY HAND (SAVED):", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            # Check visibility of each card
            y_off = 70
            x_off = 20
            
            visible_cards = {to_treys(c['label']) for c in stable_list}
            
            for card_str in registered_hand:
                # Color code: Green if visible, Red if hidden
                if card_str in visible_cards:
                    color = (0, 255, 0) # Green (Live)
                else:
                    color = (0, 0, 255) # Red (Hidden)
                
                cv2.putText(frame, card_str, (x_off, y_off), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                x_off += 80
        else:
            # Hint text
            cv2.putText(frame, "Left Pinch: Save Hand", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            
        if not board_cards and registered_hand:
             cv2.putText(frame, "Right Pinch (5s) to Add Board", (w - 450, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
        cv2.imshow("Poker AR (Simple)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): 
            break
        elif key == ord('c'):
            board_cards.clear()
            print("Board cleared manually (Press 'c')")
        
    cap.release()
    cv2.destroyAllWindows()
    if gesture_detector: gesture_detector.cleanup()
    
if __name__ == "__main__":
    main()
