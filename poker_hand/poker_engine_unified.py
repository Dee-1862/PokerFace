
import cv2
import numpy as np
from ultralytics import YOLO
from treys import Card, Evaluator, Deck
from collections import defaultdict

class PokerAssistantUnified:
    """
    Unified Poker Assistant combining best features from V1 and V2:
    - V1's ROI validation for false positive filtering
    - V1's IoU-based deduplication
    - V2's color-based suit correction
    - Enhanced debugging
    - Lower detection thresholds
    """
    def __init__(self, model_path='poker_v1.pt', debug=False):
        print(f"♠️ Loading Unified Poker Brain from {model_path}...")
        try:
            self.model = YOLO(model_path)
            # Model verification
            print(f"✓ Model loaded successfully")
            print(f"✓ Model classes: {len(self.model.names)}")
            if len(self.model.names) > 0:
                sample_classes = list(self.model.names.values())[:10]
                print(f"✓ Sample classes: {sample_classes}")
                if len(self.model.names) == 52:
                    print(f"✓ Model has correct number of classes (52 cards)")
                else:
                    print(f"⚠ Warning: Expected 52 classes, got {len(self.model.names)}")
        except Exception as e:
            print(f"⚠ Error loading model: {e}")
            raise e

        self.evaluator = Evaluator()
        self.debug = debug
        
        # State
        self.last_known_state = []
        self.cached_result = (0.0, 0, [])
        self.history = defaultdict(int)
        self.registered_hand_cards = []
        
        # Config
        self.MIN_WHITE_RATIO = 0.10  # Card must be at least 10% white/bright
        self.STABILITY_THRESHOLD = 3  # Frames to confirm
        
        # Debug counters
        self.frame_count = 0
        self.last_debug_time = 0

    def verify_card_roi(self, frame, box):
        """
        OpenCV verification to reject non-card objects.
        Checks if the region contains white/gray-ish pixels (Paper color).
        """
        x1, y1, x2, y2 = box
        h, w = frame.shape[:2]
        
        # Clamp
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return False  # Too small
        
        roi = frame[y1:y2, x1:x2]
        
        # Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Define 'White' range (Cards are usually white/off-white)
        lower_white = np.array([0, 0, 120])
        upper_white = np.array([180, 50, 255])
        
        mask = cv2.inRange(hsv, lower_white, upper_white)
        ratio = cv2.countNonZero(mask) / (roi.size / 3)
        
        return ratio > self.MIN_WHITE_RATIO

    def check_suit_color(self, frame, box, label):
        """
        Forces suit correctness based on pixel color analysis (from V2).
        Returns corrected label or original label.
        """
        x1, y1, x2, y2 = box
        h, w = frame.shape[:2]
        margin = 5
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
            return label
        
        # Focus on bottom-right region where suit symbols typically are
        roi_h, roi_w = roi.shape[:2]
        suit_roi = roi[int(roi_h * 0.6):, int(roi_w * 0.5):]
        
        if suit_roi.size == 0:
            suit_roi = roi
        
        # Convert to HSV
        hsv = cv2.cvtColor(suit_roi, cv2.COLOR_BGR2HSV)
        
        # Red detection
        lower_red1 = np.array([0, 50, 30])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 30])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 | mask2
        
        # Black detection
        black_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 50]))
        
        total_pixels = suit_roi.shape[0] * suit_roi.shape[1]
        red_ratio = cv2.countNonZero(red_mask) / total_pixels if total_pixels > 0 else 0
        black_ratio = cv2.countNonZero(black_mask) / total_pixels if total_pixels > 0 else 0
        
        # Current label suit
        current_suit = label[-1].lower()
        is_red_suit = current_suit in ['h', 'd']
        is_black_suit = current_suit in ['c', 's']
        
        # Correction logic
        if is_red_suit and red_ratio < 0.015 and black_ratio > 0.10:
            if current_suit == 'h':
                return label[:-1] + 'c'  # Hearts -> Clubs
            elif current_suit == 'd':
                return label[:-1] + 's'  # Diamonds -> Spades
            
        if is_black_suit and red_ratio > 0.03 and black_ratio < 0.05:
            if current_suit == 'c':
                return label[:-1] + 'h'  # Clubs -> Hearts
            elif current_suit == 's':
                return label[:-1] + 'd'  # Spades -> Diamonds
            
        return label

    def parse_yolo_output(self, results, frame):
        """
        Unified Parser: YOLO -> ROI Validation -> Color Correction -> IoU Deduplication
        """
        candidates = []
        raw_detections = []
        roi_rejected = 0
        color_corrected = 0
        
        for r in results:
            for box in r.boxes:
                # 1. Basic Info
                cls_id = int(box.cls[0])
                raw_label = self.model.names[cls_id]
                conf = float(box.conf[0])
                
                # 2. Coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                box_tuple = (x1, y1, x2, y2)
                
                # Debug: Track raw detections
                raw_detections.append({
                    'label': raw_label,
                    'conf': conf,
                    'box': box_tuple
                })
                
                # 3. Treys Name
                treys_name = raw_label.replace("10", "T")
                if len(treys_name) > 1:
                    treys_name = treys_name[0].upper() + treys_name[1].lower()

                # --- HYBRID VALIDATION ---
                # ROI validation (from V1)
                if self.verify_card_roi(frame, box_tuple):
                    # Color correction (from V2)
                    original_treys = treys_name
                    treys_name = self.check_suit_color(frame, box_tuple, treys_name)
                    if treys_name != original_treys:
                        color_corrected += 1
                    
                    candidates.append({
                        'raw': raw_label,
                        'treys': treys_name,
                        'box': box_tuple,
                        'conf': conf,
                        'center_y': (y1+y2)/2
                    })
                else:
                    roi_rejected += 1
        
        # --- SPATIAL DEDUPLICATION (from V1) ---
        kept_candidates = []
        candidates.sort(key=lambda x: x['conf'], reverse=True)
        
        for c in candidates:
            is_new = True
            for k in kept_candidates:
                # IoU Calculation
                xA = max(c['box'][0], k['box'][0])
                yA = max(c['box'][1], k['box'][1])
                xB = min(c['box'][2], k['box'][2])
                yB = min(c['box'][3], k['box'][3])
                interArea = max(0, xB - xA) * max(0, yB - yA)
                cArea = (c['box'][2]-c['box'][0]) * (c['box'][3]-c['box'][1])
                kArea = (k['box'][2]-k['box'][0]) * (k['box'][3]-k['box'][1])
                
                unionArea = cArea + kArea - interArea
                iou = interArea / unionArea if unionArea > 0 else 0
                
                if iou > 0.70:
                    is_new = False
                    break
            
            if is_new:
                kept_candidates.append(c)
        
        # Debug output
        import time
        current_time = time.time()
        if self.debug or (current_time - self.last_debug_time > 2.0):
            print(f"\n[DEBUG Unified Frame {self.frame_count}]")
            print(f"  Raw YOLO detections: {len(raw_detections)}")
            if raw_detections:
                top3 = sorted(raw_detections, key=lambda x: x['conf'], reverse=True)[:3]
                top3_str = [(d['label'], f"{d['conf']:.2f}") for d in top3]
                print(f"  Top 3 raw: {top3_str}")
            print(f"  After ROI validation: {len(candidates)} (rejected: {roi_rejected})")
            print(f"  Color corrections: {color_corrected}")
            print(f"  After deduplication: {len(kept_candidates)}")
            if kept_candidates:
                final_str = [(c['treys'], f"{c['conf']:.2f}") for c in kept_candidates]
                print(f"  Final cards: {final_str}")
            self.last_debug_time = current_time
        
        self.frame_count += 1
        return kept_candidates

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        
        # Lower thresholds for better detection
        results = self.model(frame, verbose=False, conf=0.05, iou=0.30)
        
        detected_cards = self.parse_yolo_output(results, frame)
        my_hand, board = self.separate_hand_and_board(detected_cards, h)
        equity, outs, top_hands = self.calculate_equity(my_hand, board)
        
        return detected_cards, my_hand, board, equity, outs, top_hands

    def register_hand(self, detected_cards):
        if len(detected_cards) < 2:
            return False
        sorted_cards = sorted(detected_cards, 
                             key=lambda x: (x['box'][2]-x['box'][0]) * (x['box'][3]-x['box'][1]), 
                             reverse=True)
        self.registered_hand_cards = [c['treys'] for c in sorted_cards[:2]]
        print(f"🃏 Registered Unified Hand: {self.registered_hand_cards}")
        self.last_known_state = []
        return True

    def separate_hand_and_board(self, detected_cards, frame_height):
        my_hand = []
        board = []
        if self.registered_hand_cards:
            for c in detected_cards:
                if c['treys'] in self.registered_hand_cards:
                    my_hand.append(c)
                else:
                    board.append(c)
            # Fallback if registration missed
            if len(my_hand) < 2:
                thresh = frame_height * 0.6
                for c in board[:]:
                    if c['center_y'] > thresh:
                        board.remove(c)
                        my_hand.append(c)
        else:
            thresh = frame_height * 0.6
            for c in detected_cards:
                if c['center_y'] > thresh:
                    my_hand.append(c)
                else:
                    board.append(c)
        
        # Sort board L-R
        board.sort(key=lambda x: x['box'][0])
        # Limit hand to 2 best
        if len(my_hand) > 2:
            my_hand.sort(key=lambda x: (x['box'][2]-x['box'][0]) * (x['box'][3]-x['box'][1]), reverse=True)
            board.extend(my_hand[2:])
            my_hand = my_hand[:2]
            
        return my_hand, board

    def calculate_equity(self, hand_objs, board_objs):
        if len(hand_objs) != 2:
            return 0.0, 0, []
        try:
            seen = set()
            mc = []
            for c in hand_objs:
                v = Card.new(c['treys'])
                if v not in seen:
                    seen.add(v)
                    mc.append(v)
            if len(mc) != 2:
                return 0.0, 0, []
            
            bc = []
            for c in board_objs:
                v = Card.new(c['treys'])
                if v not in seen:
                    seen.add(v)
                    bc.append(v)
                
            state_key = sorted([c['treys'] for c in hand_objs + board_objs])
            if state_key == self.last_known_state:
                return self.cached_result
            
            deck = Deck()
            known = mc + bc
            for k in known:
                if k in deck.cards:
                    deck.cards.remove(k)
                
            wins = 0
            valid = 0
            stats = {}
            for _ in range(500):  # More iterations for accuracy
                deck.shuffle()
                draw = deck.draw(5 - len(bc))
                sb = bc + draw
                opp = deck.draw(2)
                try:
                    ms = self.evaluator.evaluate(sb, mc)
                    os = self.evaluator.evaluate(sb, opp)
                    if ms < os:
                        wins += 1
                        cls = self.evaluator.class_to_string(self.evaluator.get_rank_class(ms))
                        stats[cls] = stats.get(cls, 0) + 1
                    valid += 1
                except:
                    pass
                deck.cards.extend(draw)
                deck.cards.extend(opp)
            
            eq = (wins/valid*100) if valid else 0
            outs = int(eq/10) if len(bc) >= 3 and eq < 100 else 0
            
            th = [{'name': k, 'prob': v/valid, 'cards': []} for k, v in stats.items()]
            th.sort(key=lambda x: x['prob'], reverse=True)
            
            res = (eq, outs, th)
            self.last_known_state = state_key
            self.cached_result = res
            return res
        except:
            return 0.0, 0, []

