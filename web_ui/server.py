"""
FastAPI WebSocket server that runs the existing face/rPPG/FACS/stress pipeline
in a background thread and broadcasts:
    - latest webcam frame as JPEG (binary message)
    - latest metrics as JSON (text message)

The browser receives both, displays the frame as a CSS background, and renders
the glass-UI panels with `backdrop-filter` for true frosted-glass refraction.

Run:
    python web_ui/server.py
Open:
    http://localhost:8000              (laptop)
    http://<laptop-ip>:8000             (phone on the same Wi-Fi)
"""
import asyncio
import csv
import json
import os
import sys
import threading
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from itertools import combinations
from pathlib import Path

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

# Reuse the existing pipeline
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'micro_expressions'))
sys.path.insert(0, str(ROOT / 'poker_hand'))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
except ImportError:
    pass

from face_detection import FaceDetectionModule
from rppg_heart_rate import RPPGModule
from facs_action_units import FACSModule
from stress_detector import StressDetectorModule
from hand_gesture_detector import HandGestureDetector

# Optional config (tunables); fall back to literals if missing.
try:
    import config as _CONFIG
    def _c(name, default):
        return getattr(_CONFIG, name, default)
except Exception:
    def _c(name, default):
        return default

# Optional poker stack — if either YOLO or treys is missing, the pipeline still
# runs without card detection.
try:
    from ultralytics import YOLO  # type: ignore
    YOLO_AVAILABLE = True
except Exception as _e:
    print(f"[server] YOLO unavailable, poker disabled: {_e}")
    YOLO_AVAILABLE = False

try:
    from treys import Card, Evaluator, Deck  # type: ignore
    TREYS_AVAILABLE = True
except Exception as _e:
    print(f"[server] treys unavailable, equity disabled: {_e}")
    TREYS_AVAILABLE = False


CAMERA_INDEX = int(os.environ.get('CAMERA_INDEX', '0'))
JPEG_QUALITY = 70
TARGET_FPS = 25

# === Poker equity helpers (ported from unified_ar_system.py) ===
PREFLOP_EQUITY: dict = {}
POKER_EVALUATOR = None
if TREYS_AVAILABLE:
    try:
        equity_path = ROOT / 'poker_hand' / 'preflop_equity.csv'
        with open(equity_path, 'r') as f:
            for row in csv.DictReader(f):
                PREFLOP_EQUITY[row['hand']] = float(row['equity'])
        print(f"[server] Loaded {len(PREFLOP_EQUITY)} preflop hands")
    except Exception as e:
        print(f"[server] preflop table load failed: {e}")
    try:
        POKER_EVALUATOR = Evaluator()
        print("[server] Poker evaluator initialised")
    except Exception as e:
        print(f"[server] Poker evaluator init failed: {e}")


def to_treys(label):
    """Convert YOLO class label (e.g. '10H') to treys format (e.g. 'Th')."""
    if not label or len(label) < 2:
        return None
    rank = label[:-1]
    suit = label[-1].lower()
    if rank == '10':
        rank = 'T'
    if suit not in ('h', 'd', 'c', 's'):
        return None
    if rank not in ('2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'):
        return None
    return f"{rank}{suit}"


def normalize_hand(my_hand):
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
    if not POKER_EVALUATOR:
        return [Card.int_to_str(c) for c in all_cards_ints[:5]]
    min_score = float('inf')
    best_hand = None
    for five_cards in combinations(all_cards_ints, 5):
        score = POKER_EVALUATOR.evaluate(list(five_cards), [])
        if score < min_score:
            min_score = score
            best_hand = list(five_cards)
    if best_hand:
        return [Card.int_to_str(c) for c in best_hand]
    return [Card.int_to_str(c) for c in all_cards_ints[:5]]


def exhaustive_enumeration(my_hand, board_cards):
    """Flop/turn equity: enumerate runouts, sample opponent hands."""
    import random
    hero_hand = [Card.new(c) for c in my_hand]
    board = [Card.new(c) for c in board_cards]
    deck = Deck()
    for c in hero_hand + board:
        if c in deck.cards:
            deck.cards.remove(c)
    cards_needed = 5 - len(board)
    wins = 0
    total = 0
    hero_hand_counts: dict = defaultdict(int)
    opp_hand_counts: dict = defaultdict(int)
    hero_hand_examples: dict = {}
    opp_hand_examples: dict = {}
    for runout in combinations(deck.cards, cards_needed):
        remaining = [c for c in deck.cards if c not in runout]
        opp_sample = list(combinations(remaining, 2))
        if len(opp_sample) > 200:
            opp_sample = random.sample(opp_sample, 200)
        full_board = board + list(runout)
        for opp_hand in opp_sample:
            total += 1
            try:
                hero_score = POKER_EVALUATOR.evaluate(full_board, hero_hand)
                opp_score = POKER_EVALUATOR.evaluate(full_board, list(opp_hand))
                hero_class = POKER_EVALUATOR.class_to_string(POKER_EVALUATOR.get_rank_class(hero_score))
                opp_class = POKER_EVALUATOR.class_to_string(POKER_EVALUATOR.get_rank_class(opp_score))
                if hero_score < opp_score:
                    wins += 1
                    hero_hand_counts[hero_class] += 1
                    if hero_class not in hero_hand_examples:
                        hero_hand_examples[hero_class] = get_best_5_cards(hero_hand + full_board)
                else:
                    opp_hand_counts[opp_class] += 1
                    if opp_class not in opp_hand_examples:
                        opp_hand_examples[opp_class] = get_best_5_cards(list(opp_hand) + full_board)
            except Exception:
                pass
    equity = (wins / total * 100) if total > 0 else 0.0
    my_hands = []
    for name, count in hero_hand_counts.items():
        prob = count / total if total > 0 else 0
        if prob > 0.01:
            my_hands.append({'name': name, 'prob': prob,
                             'cards': hero_hand_examples.get(name, [])[:7]})
    my_hands.sort(key=lambda x: x['prob'], reverse=True)
    opp_hands = []
    for name, count in opp_hand_counts.items():
        prob = count / total if total > 0 else 0
        if prob > 0.01:
            opp_hands.append({'name': name, 'prob': prob,
                              'cards': opp_hand_examples.get(name, [])[:7]})
    opp_hands.sort(key=lambda x: x['prob'], reverse=True)
    return equity, 0, {'my_hands': my_hands, 'opp_hands': opp_hands}


def evaluate_river(my_hand, board_cards):
    hero_hand = [Card.new(c) for c in my_hand]
    board = [Card.new(c) for c in board_cards]
    deck = Deck()
    for c in hero_hand + board:
        if c in deck.cards:
            deck.cards.remove(c)
    wins = 0
    total = 0
    hero_hand_counts: dict = defaultdict(int)
    opp_hand_counts: dict = defaultdict(int)
    hero_hand_examples: dict = {}
    opp_hand_examples: dict = {}
    for opp_hand in combinations(deck.cards, 2):
        total += 1
        try:
            hero_score = POKER_EVALUATOR.evaluate(board, hero_hand)
            opp_score = POKER_EVALUATOR.evaluate(board, list(opp_hand))
            hero_class = POKER_EVALUATOR.class_to_string(POKER_EVALUATOR.get_rank_class(hero_score))
            opp_class = POKER_EVALUATOR.class_to_string(POKER_EVALUATOR.get_rank_class(opp_score))
            if hero_score < opp_score:
                wins += 1
                hero_hand_counts[hero_class] += 1
                if hero_class not in hero_hand_examples:
                    hero_hand_examples[hero_class] = get_best_5_cards(hero_hand + board)
            else:
                opp_hand_counts[opp_class] += 1
                if opp_class not in opp_hand_examples:
                    opp_hand_examples[opp_class] = get_best_5_cards(list(opp_hand) + board)
        except Exception:
            pass
    equity = (wins / total * 100) if total > 0 else 0.0
    my_hands = []
    for name, count in hero_hand_counts.items():
        prob = count / total if total > 0 else 0
        if prob > 0.01:
            my_hands.append({'name': name, 'prob': prob,
                             'cards': hero_hand_examples.get(name, [])[:7]})
    my_hands.sort(key=lambda x: x['prob'], reverse=True)
    opp_hands = []
    for name, count in opp_hand_counts.items():
        prob = count / total if total > 0 else 0
        if prob > 0.01:
            opp_hands.append({'name': name, 'prob': prob,
                              'cards': opp_hand_examples.get(name, [])[:7]})
    opp_hands.sort(key=lambda x: x['prob'], reverse=True)
    return equity, 0, {'my_hands': my_hands, 'opp_hands': opp_hands}


def calculate_equity_fast(my_hand, board_cards):
    """Hybrid equity: lookup preflop, enumerate flop/turn, evaluate river."""
    if POKER_EVALUATOR is None or not my_hand or len(my_hand) != 2:
        return 0.0, 0, {'my_hands': [], 'opp_hands': []}
    if any(c is None for c in my_hand):
        return 0.0, 0, {'my_hands': [], 'opp_hands': []}
    board_size = len(board_cards)
    try:
        if board_size == 0:
            key = normalize_hand(my_hand)
            if key and key in PREFLOP_EQUITY:
                eq = PREFLOP_EQUITY[key]
                is_pair = my_hand[0][0] == my_hand[1][0]
                return eq, 0, {
                    'my_hands': [{'name': 'Pair' if is_pair else 'High Card',
                                  'prob': 1.0, 'cards': my_hand}],
                    'opp_hands': [
                        {'name': 'Pair', 'prob': 0.06, 'cards': ['As', 'Ac']},
                        {'name': 'High Card', 'prob': 0.94, 'cards': ['Ah', 'Kh']},
                    ],
                }
            return 50.0, 0, {'my_hands': [], 'opp_hands': []}
        elif board_size in (3, 4):
            return exhaustive_enumeration(my_hand, board_cards)
        else:
            return evaluate_river(my_hand, board_cards)
    except Exception as e:
        print(f"[server] equity error: {e}")
        return 0.0, 0, {'my_hands': [], 'opp_hands': []}

# Shared state between the capture thread and the WebSocket handlers
_state = {
    'frame_jpeg': None,        # bytes
    'metrics': {},             # dict
    'last_capture_time': 0.0,
    'lock': threading.Lock(),
}

# Ordered set of connected WebSocket clients (the asyncio loop owns these)
_clients: 'set[WebSocket]' = set()
_loop: asyncio.AbstractEventLoop | None = None


def _compute_face_bbox_norm(landmarks):
    """Return normalised bbox dict {x, y, w, h} in 0-1 frame coords, or None."""
    if not landmarks:
        return None
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x0 = max(0.0, min(xs))
    y0 = max(0.0, min(ys))
    x1 = min(1.0, max(xs))
    y1 = min(1.0, max(ys))
    return {'x': x0, 'y': y0, 'w': x1 - x0, 'h': y1 - y0}


def _extract_hand_state(hand_data, frame_w, frame_h):
    """Convert HandGestureDetector output to a small JSON-safe dict.
    Coordinates are normalised 0-1 to the frame so the browser can map them
    to its viewport regardless of resolution.
    """
    if not hand_data:
        return None
    pinch = hand_data.get('pinch') or {}
    # tip positions are pixel coords; normalise back to 0-1
    tx = ty = ix = iy = None
    if pinch.get('thumb_pos') is not None:
        tx = float(pinch['thumb_pos'][0]) / max(1, frame_w)
        ty = float(pinch['thumb_pos'][1]) / max(1, frame_h)
    if pinch.get('index_pos') is not None:
        ix = float(pinch['index_pos'][0]) / max(1, frame_w)
        iy = float(pinch['index_pos'][1]) / max(1, frame_h)
    # Midpoint between thumb tip and index tip = where the user is "grabbing"
    mid_x = (tx + ix) / 2 if (tx is not None and ix is not None) else None
    mid_y = (ty + iy) / 2 if (ty is not None and iy is not None) else None

    # Two-finger scroll gesture (peace sign — index + middle up). Used by the
    # browser to scroll the My/Opp hand-rank lists hands-free.
    scroll = hand_data.get('two_finger_scroll') or {}
    scroll_payload = {
        'active': bool(scroll.get('active', False)),
        'y':      float(scroll.get('y_normalized', 0.5)),
    }

    return {
        'pinch':         bool(pinch.get('active', False)),
        'is_locked':     bool(pinch.get('is_locked', False)),
        'hold_progress': float(pinch.get('hold_progress', 0.0)),
        'distance':      float(pinch.get('distance', 0.0)),
        'is_left':       bool(hand_data.get('is_left', False)),
        'thumb':         None if tx is None else {'x': tx, 'y': ty},
        'index':         None if ix is None else {'x': ix, 'y': iy},
        'mid':           None if mid_x is None else {'x': mid_x, 'y': mid_y},
        'scroll':        scroll_payload,
    }


def _open_camera(index):
    """Open the camera, trying the most reliable Windows backend first.

    DroidCam and other virtual cameras on Windows behave best with the
    DirectShow backend. If that fails we fall back to MSMF, then any.
    """
    backends = []
    if sys.platform.startswith('win'):
        backends = [
            (cv2.CAP_DSHOW, 'DirectShow'),
            (cv2.CAP_MSMF,  'Media Foundation'),
            (cv2.CAP_ANY,   'Default'),
        ]
    else:
        backends = [(cv2.CAP_ANY, 'Default')]

    for api, name in backends:
        cap = cv2.VideoCapture(index, api)
        if cap.isOpened():
            print(f"[server] Camera {index} opened via {name}")
            return cap
        cap.release()
    return None


def _list_cameras(max_index=5):
    """Return a list of indices that have a working camera (Windows-friendly)."""
    found = []
    api = cv2.CAP_DSHOW if sys.platform.startswith('win') else cv2.CAP_ANY
    for i in range(max_index):
        cap = cv2.VideoCapture(i, api)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                found.append(i)
            cap.release()
    return found


def capture_loop():
    """Background thread: capture frame, run pipeline, encode, store."""
    print(f"[server] Opening camera index {CAMERA_INDEX} ...")
    cap = _open_camera(CAMERA_INDEX)
    if cap is None:
        print(f"[server] ERROR: cannot open camera {CAMERA_INDEX}")
        avail = _list_cameras()
        if avail:
            print(f"[server] Available camera indices on this machine: {avail}")
            print(f"[server] -> set CAMERA_INDEX in .env to one of those values.")
        else:
            print("[server] No cameras detected. If using DroidCam, make sure")
            print("        the DroidCam Client is connected to the phone.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Warm-up: most webcams (and DroidCam in particular) deliver a few black or
    # garbage frames before the real video starts. Discard them, then sample.
    print("[server] Warming up camera (skipping first 8 frames) ...")
    for _ in range(8):
        cap.read()
        time.sleep(0.05)

    ok, sample = cap.read()
    if ok and sample is not None:
        mb, mg, mr = sample.mean(axis=(0, 1))[:3]
        peak = max(mb, mg, mr)
        trough = min(mb, mg, mr)
        saturation = (peak - trough) / max(1.0, peak)
        print(f"[server] Warm frame {sample.shape[1]}x{sample.shape[0]} "
              f"mean BGR=({mb:.0f},{mg:.0f},{mr:.0f})")

        if saturation > 0.6 and mg == peak:
            print("[server] WARNING: frame is dominated by GREEN (DroidCam 'no signal').")
            print("         DroidCam Client is connected to a virtual camera but the")
            print("         phone is not actually sending video. To fix:")
            print("         1. On the PHONE: open the DroidCam app and tap 'Start'.")
            print("         2. Make sure the phone and laptop are on the SAME Wi-Fi.")
            print("         3. In DroidCam Client (laptop), enter the phone's IP shown")
            print("            on the phone screen, then click the play button.")
            print("         4. The DroidCam Client window should now show LIVE phone video.")
            print("         5. Restart this server so it picks up the live stream.")
        elif saturation > 0.6:
            print(f"[server] WARNING: frame is a single colour (saturation={saturation:.2f}).")
            print("         This is a 'no signal' pattern. Check the camera source.")
        elif peak < 8:
            print("[server] WARNING: frame is almost pure BLACK. Check camera privacy")
            print("         settings, or DroidCam phone-side may not be sending video.")

    print("[server] Initialising pipeline modules ...")
    face_mod   = FaceDetectionModule()
    rppg_mod   = RPPGModule(buffer_size=300, fps=30)
    facs_mod   = FACSModule()
    stress_mod = StressDetectorModule()
    try:
        hand_mod = HandGestureDetector()
        if hand_mod.detector is None:
            hand_mod = None
            print("[server] Hand detector failed to load (drag will be disabled)")
        else:
            print("[server] Hand detector ready (pinch-to-drag enabled)")
    except Exception as e:
        hand_mod = None
        print(f"[server] Hand detector init error: {e}")

    # === Poker (YOLO) initialisation ===
    yolo_model = None
    if YOLO_AVAILABLE:
        try:
            v2 = ROOT / 'poker_hand' / 'poker_v2.pt'
            v1 = ROOT / 'poker_hand' / 'poker_v1.pt'
            model_path = v2 if v2.exists() else v1
            yolo_model = YOLO(str(model_path))
            print(f"[server] YOLO poker model loaded ({model_path.name})")
        except Exception as e:
            print(f"[server] YOLO model load failed: {e}")
            yolo_model = None

    # Poker state (mirrors unified_ar_system.py)
    card_history: dict = defaultdict(int)
    finalized_cards: dict = {}
    STABILITY_THRESHOLD    = _c('CARD_STABILITY_THRESHOLD', 4)
    FINALIZE_THRESHOLD     = _c('CARD_FINALIZE_THRESHOLD',  12)
    CARD_FADE_TIMEOUT      = _c('CARD_FADE_TIMEOUT',        60)
    NO_CARDS_RESET_TIMEOUT = _c('CARD_RESET_TIMEOUT',       90)
    YOLO_SKIP              = _c('YOLO_SKIP',                2)
    YOLO_CONF              = _c('YOLO_CONF',                0.45)
    YOLO_IOU               = _c('YOLO_IOU',                 0.15)
    YOLO_IMGSZ             = _c('YOLO_IMGSZ',               640)
    HYSTERESIS_FRAMES      = _c('HYSTERESIS_FRAMES',        8)
    zero_card_frames       = 0
    registered_hand: list  = []
    reg_lock               = False
    board_cards: set       = set()
    board_lock_once        = False
    equity_cache: dict     = {'hand': None, 'board': None, 'result': None}
    eq_state               = {'eq': 0.0, 'outs': 0, 'top': {'my_hands': [], 'opp_hands': []}}
    last_detected_list: list = []
    last_current_frame_cards: set = set()
    # Context with hysteresis: 'poker' | 'hybrid_poker' | 'face' | 'hybrid' | 'none'
    current_context  = 'none'
    pending_context  = 'none'
    pending_frames   = 0

    # Extra-hold confirmation for poker actions, on top of the 1.5s pinch lock.
    # When holding a card, your natural grip looks like a pinch and would
    # otherwise fire after 1.5s. We require the lock to be SUSTAINED for an
    # additional N frames, so a true poker action takes ~3s of held pinch but
    # passively gripping a card never reaches the threshold.
    POKER_HOLD_CONFIRM_FRAMES = _c('POKER_HOLD_CONFIRM_FRAMES', 36)  # ~1.5s @ 25fps
    lh_lock_frames = 0
    rh_lock_frames = 0

    # Visual-only bbox tracker: every YOLO hit updates an EMA-smoothed box for
    # that label; the box persists for ~1.2s after the last hit so it doesn't
    # vanish on a single dropped detection. Decoupled from stability/finalize
    # so even unstable detections produce a steady on-screen box.
    recent_boxes: dict = {}            # {label: {'box': [x1,y1,x2,y2], 'frame': N, 'conf': float}}
    BOX_FADE_FRAMES   = _c('BOX_FADE_FRAMES', 30)
    BOX_SMOOTH_ALPHA  = _c('BOX_SMOOTH_ALPHA', 0.4)   # weight of the new observation

    shared = {
        'frame': None,
        'frame_dimensions': (0, 0),
        'face_detected': False,
        'landmarks': None,
        'timestamp': 0,
        'is_video_file': False,
        'frame_number': 0,
    }

    # Initialise modules
    for mod in (face_mod, rppg_mod, facs_mod, stress_mod):
        if hasattr(mod, 'initialize'):
            try:
                mod.initialize(shared)
            except Exception as e:
                print(f"[server] init {type(mod).__name__} failed: {e}")

    print("[server] Capture loop running.")
    frame_period = 1.0 / TARGET_FPS
    next_t = time.monotonic()
    last_diag_t = time.monotonic()

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        h, w = frame.shape[:2]
        shared.update({
            'frame': frame,
            'frame_dimensions': (h, w),
            'timestamp': time.time(),
            'frame_number': shared['frame_number'] + 1,
        })

        # Every 3 seconds, log a one-line diagnostic so the user can see what
        # the pipeline is actually receiving without enabling debug-level logs.
        now_t = time.monotonic()
        if now_t - last_diag_t > 3.0:
            mb, mg, mr = frame.mean(axis=(0, 1))[:3]
            face_ok = shared.get('face_detected', False)
            hr_val  = shared.get('heart_rate_bpm', 0)
            print(f"[server] frame#{shared['frame_number']} "
                  f"mean BGR=({mb:.0f},{mg:.0f},{mr:.0f}) "
                  f"face={face_ok} hr={hr_val} clients={len(_clients)}")
            last_diag_t = now_t

        # Face / rPPG / FACS / stress pipeline. Skipped entirely when in cards
        # mode — the face panels are hidden then anyway, and MediaPipe face
        # landmarker + scalp-FFT rPPG are by far the heaviest CPU users in
        # this loop. This is what makes the fans quiet down when cards take
        # over the screen.
        in_cards_mode = current_context in ('poker', 'hybrid_poker')
        if not in_cards_mode:
            for mod in (face_mod, rppg_mod, facs_mod, stress_mod):
                try:
                    mod.process(shared)
                except Exception as e:
                    # Don't let one module take down the whole loop
                    print(f"[server] {type(mod).__name__}.process error: {e}")
        else:
            # While cards rule the screen, mark the face state as not-detected
            # so the context vote keeps voting for poker / hybrid_poker until
            # cards genuinely leave the frame.
            shared['face_detected'] = False

        # Hand detector takes (frame, shared) - different signature
        if hand_mod is not None:
            try:
                hand_mod.process(frame, shared)
            except Exception as e:
                print(f"[server] HandGestureDetector.process error: {e}")

        gestures = shared.get('hand_gestures', {}) or {}

        # === YOLO card detection (adaptive cadence) ===
        # When cards have been seen recently, run YOLO on its tight cadence
        # (every YOLO_SKIP frames). When the frame is idle (no cards anywhere
        # for a while), drop to a slow polling cadence — YOLO inference is
        # ~50 ms per call and burns the GPU even when finding nothing.
        face_detected = bool(shared.get('face_detected'))
        idle_yolo = (not recent_boxes) and (not finalized_cards) and (current_context not in ('poker', 'hybrid_poker'))
        yolo_cadence = _c('YOLO_SKIP_IDLE', 12) if idle_yolo else YOLO_SKIP
        if yolo_model is not None and (shared['frame_number'] % yolo_cadence == 0):
            last_detected_list = []
            last_current_frame_cards = set()
            try:
                results = yolo_model(frame, verbose=False,
                                     conf=YOLO_CONF, iou=YOLO_IOU, imgsz=YOLO_IMGSZ)
                for r in results:
                    for box in r.boxes:
                        lbl = yolo_model.names[int(box.cls[0])]
                        conf = float(box.conf[0])
                        bbox = box.xyxy[0].tolist()
                        last_current_frame_cards.add(lbl)
                        last_detected_list.append({'label': lbl, 'confidence': conf, 'box': bbox})

                        # EMA-smooth the visual bbox so the on-screen rectangle
                        # doesn't snap on every YOLO frame.
                        prev = recent_boxes.get(lbl)
                        if prev is None:
                            smoothed = list(bbox)
                        else:
                            a = BOX_SMOOTH_ALPHA
                            old = prev['box']
                            smoothed = [
                                old[0] + a * (bbox[0] - old[0]),
                                old[1] + a * (bbox[1] - old[1]),
                                old[2] + a * (bbox[2] - old[2]),
                                old[3] + a * (bbox[3] - old[3]),
                            ]
                        recent_boxes[lbl] = {
                            'box':   smoothed,
                            'frame': shared['frame_number'],
                            'conf':  conf,
                        }
            except Exception as e:
                print(f"[server] YOLO inference error: {e}")

        # Expire visual boxes whose last YOLO hit was too long ago. This is
        # what gives the bbox its ~1.2s persistence through dropped frames.
        if recent_boxes:
            cur_frame = shared['frame_number']
            stale = [lbl for lbl, d in recent_boxes.items()
                     if cur_frame - d['frame'] > BOX_FADE_FRAMES]
            for lbl in stale:
                del recent_boxes[lbl]

        detected_list = last_detected_list
        current_frame_cards = last_current_frame_cards
        cards_detected = len(detected_list) > 0

        # For the context vote, treat any card YOLO has seen in the last
        # ~1.8s as still "in play". This is the single most important knob
        # for stopping the cards-dashboard from flickering: with strict
        # YOLO_CONF, current-frame detections drop out constantly, but
        # `recent_boxes` persists across BOX_FADE_FRAMES (~45 frames).
        # We only leave cards mode after the cards genuinely vanish for that
        # whole window, not on a single missed YOLO frame.
        cards_in_play = cards_detected or bool(finalized_cards) or bool(recent_boxes)

        # === Context vote with hysteresis (cards always beat face) ===
        if cards_in_play and face_detected:
            voted = 'hybrid_poker'
        elif cards_in_play:
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
                    print(f"[server] context -> {current_context.upper()}")
                pending_frames = 0

        # === Card stability tracking + gesture-driven hand/board state ===
        # (only when cards are involved or already-saved hand needs upkeep)
        if cards_detected or finalized_cards:
            # Grip rejection: only reject when BOTH thumb and index tips sit on
            # the same card — that's when the user is physically gripping the
            # card. A pinch where only one fingertip is near a card (or
            # neither) is treated as an intentional gesture so the loader can
            # build up.
            def _reject_grip(hand):
                if not hand:
                    return
                pinch = hand.get('pinch') or {}
                if not pinch.get('active'):
                    return
                tp = pinch.get('thumb_pos')
                ip = pinch.get('index_pos')
                if tp is None or ip is None:
                    return
                tx, ty = float(tp[0]), float(tp[1])
                ix, iy = float(ip[0]), float(ip[1])
                for c in detected_list:
                    x1, y1, x2, y2 = c['box']
                    thumb_on  = (x1 - 12 < tx < x2 + 12) and (y1 - 12 < ty < y2 + 12)
                    index_on  = (x1 - 12 < ix < x2 + 12) and (y1 - 12 < iy < y2 + 12)
                    if thumb_on and index_on:
                        pinch['active']    = False
                        pinch['is_locked'] = False
                        return

            lh = gestures.get('left_hand')
            rh = gestures.get('right_hand')
            _reject_grip(lh)
            _reject_grip(rh)

            # Sustained-lock counters: the detector flips is_locked=True after
            # 1.5s of pinch. We require it to STAY locked for an extra
            # POKER_HOLD_CONFIRM_FRAMES on top of that before firing — so a
            # passive grip on a card never reaches the threshold.
            lh_locked_now = bool(lh and lh.get('pinch', {}).get('is_locked'))
            rh_locked_now = bool(rh and rh.get('pinch', {}).get('is_locked'))
            lh_lock_frames = lh_lock_frames + 1 if lh_locked_now else 0
            rh_lock_frames = rh_lock_frames + 1 if rh_locked_now else 0

            # Build a broad pool of "good enough to register" cards: finalized
            # first (most stable), then stable-but-not-yet-finalized detections,
            # then high-confidence raw detections. This way registration fires
            # as long as YOLO has plausibly seen 2 cards during the hold —
            # without requiring 12 perfectly-stable frames each.
            def _registration_pool():
                pool = list(finalized_cards.values())
                seen = set(c['label'] for c in pool)
                # Tier 2: detections that have crossed STABILITY_THRESHOLD
                for c in detected_list:
                    if c['label'] not in seen and card_history.get(c['label'], 0) >= STABILITY_THRESHOLD:
                        pool.append(c)
                        seen.add(c['label'])
                # Tier 3: high-confidence detections seen this frame
                for c in detected_list:
                    if c['label'] not in seen and c.get('confidence', 0) >= 0.55:
                        pool.append(c)
                        seen.add(c['label'])
                return pool

            # Left pinch lock + confirm -> save the two largest visible cards as hero hand
            if lh_lock_frames >= POKER_HOLD_CONFIRM_FRAMES:
                if not reg_lock:
                    candidates = _registration_pool()
                    if len(candidates) >= 2:
                        candidates.sort(
                            key=lambda x: (x['box'][2] - x['box'][0]) * (x['box'][3] - x['box'][1]),
                            reverse=True,
                        )
                        new_hand = [to_treys(x['label']) for x in candidates[:2]]
                        new_hand = [h for h in new_hand if h]
                        if len(new_hand) == 2:
                            registered_hand = new_hand
                            reg_lock = True
                            print(f"[server] hero hand registered: {registered_hand}")
                        else:
                            print(f"[server] register skipped: invalid labels {[c['label'] for c in candidates[:2]]}")
                    else:
                        print(f"[server] register skipped: only {len(candidates)} card(s) visible")
            else:
                reg_lock = False

            # Right pinch lock + confirm -> add visible cards (not in hand) to the board
            if rh_lock_frames >= POKER_HOLD_CONFIRM_FRAMES:
                if not board_lock_once:
                    added = 0
                    pool = _registration_pool()
                    for c in pool:
                        t = to_treys(c['label'])
                        if t and t not in board_cards and t not in registered_hand:
                            board_cards.add(t)
                            added += 1
                    if added:
                        print(f"[server] board +{added} -> {sorted(board_cards)}")
                    board_lock_once = True
            else:
                board_lock_once = False
        else:
            # No cards anywhere -> reset the confirmation counters
            lh_lock_frames = 0
            rh_lock_frames = 0

            # Stability counters
            for lbl in current_frame_cards:
                card_history[lbl] += 1
            for lbl in list(card_history.keys()):
                if lbl not in current_frame_cards and lbl not in finalized_cards:
                    card_history[lbl] -= 1
                    if card_history[lbl] <= 0:
                        del card_history[lbl]

            # Promote stable detections to finalized
            for c in detected_list:
                lbl = c['label']
                if (lbl not in finalized_cards
                        and STABILITY_THRESHOLD <= card_history[lbl] <= FINALIZE_THRESHOLD):
                    finalized_cards[lbl] = {
                        'label': lbl,
                        'confidence': c['confidence'],
                        'box': c['box'],
                        'last_seen': shared['frame_number'],
                    }
            for c in detected_list:
                if c['label'] in finalized_cards:
                    finalized_cards[c['label']].update({
                        'confidence': max(c['confidence'], finalized_cards[c['label']]['confidence']),
                        'box': c['box'],
                        'last_seen': shared['frame_number'],
                    })

            if not detected_list:
                zero_card_frames += 1
            else:
                zero_card_frames = 0

            # Full reset (clears finalized + history; KEEPS board_cards for street persistence)
            if zero_card_frames > NO_CARDS_RESET_TIMEOUT:
                finalized_cards.clear()
                card_history.clear()

            # Expire individual stale finalized cards
            expired = [lbl for lbl, data in finalized_cards.items()
                       if shared['frame_number'] - data.get('last_seen', 0) > CARD_FADE_TIMEOUT]
            for lbl in expired:
                del finalized_cards[lbl]


        # === Equity (cached on hand+board key) ===
        if registered_hand and POKER_EVALUATOR is not None:
            eq_board = list(board_cards)
            hand_key = tuple(sorted(x for x in registered_hand if x is not None))
            board_key = tuple(sorted(eq_board)) if eq_board else None
            if (equity_cache['hand'] == hand_key and equity_cache['board'] == board_key
                    and equity_cache['result'] is not None):
                eq_v, outs_v, top_v = equity_cache['result']
            else:
                eq_v, outs_v, top_v = calculate_equity_fast(registered_hand, eq_board)
                equity_cache['hand'] = hand_key
                equity_cache['board'] = board_key
                equity_cache['result'] = (eq_v, outs_v, top_v)
            eq_state['eq'] = float(eq_v)
            eq_state['outs'] = int(outs_v)
            eq_state['top'] = top_v

        # Encode frame
        ok2, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok2:
            continue

        # Build metrics payload
        au = shared.get('action_units', {}) or {}
        hands = {
            'left':  _extract_hand_state(gestures.get('left_hand'),  w, h),
            'right': _extract_hand_state(gestures.get('right_hand'), w, h),
        }

        # Visible cards (normalised bbox + status) for overlay rendering.
        # We emit a box for every card YOLO sees this frame AND every card
        # that's still finalized (last-known box). This way registered/locked
        # cards always get a green box even if they were registered via the
        # "high confidence raw detection" tier and never made it into
        # finalized_cards.
        rh_set = set(x for x in registered_hand if x)
        cards_payload = []
        emitted = set()

        def _emit_card(label, box, conf):
            if label in emitted:
                return
            emitted.add(label)
            t = to_treys(label)
            status = 'me' if t in rh_set else ('locked' if t in board_cards else 'detecting')
            x1, y1, x2, y2 = box
            cards_payload.append({
                'label': label,
                'treys': t,
                'status': status,
                'conf':  float(conf or 0.0),
                'bbox':  {
                    'x': float(x1) / max(1, w),
                    'y': float(y1) / max(1, h),
                    'w': float(x2 - x1) / max(1, w),
                    'h': float(y2 - y1) / max(1, h),
                },
            })

        # Tier 1: every label YOLO has hit recently (EMA-smoothed, persists
        # for BOX_FADE_FRAMES even when the current frame had no detection).
        for lbl, data in recent_boxes.items():
            _emit_card(lbl, data['box'], data['conf'])
        # Tier 2: finalized cards beyond the recent-box window — keeps a
        # stale-but-valid box on screen for committed cards (registered/locked)
        # for the longer CARD_FADE_TIMEOUT window.
        for lbl, c in finalized_cards.items():
            _emit_card(lbl, c['box'], c.get('confidence'))

        # Hold-progress for both pinches — covers the full sequence
        # (1.5s pinch-lock + N-frame confirmation). Maps to 0..1 continuously
        # so the floating loader fills smoothly across the whole gesture.
        def _combined_hold(hand, lock_frames):
            if not hand:
                return 0.0
            pinch = hand.get('pinch') or {}
            if not pinch.get('active'):
                return 0.0
            if pinch.get('is_locked'):
                confirm = min(1.0, lock_frames / max(1, POKER_HOLD_CONFIRM_FRAMES))
                return 0.5 + 0.5 * confirm
            return 0.5 * float(pinch.get('hold_progress', 0.0))

        rh_hold = _combined_hold(gestures.get('right_hand'), rh_lock_frames)
        lh_hold = _combined_hold(gestures.get('left_hand'),  lh_lock_frames)

        metrics = {
            'fps':             round(1.0 / max(1e-3, time.monotonic() - next_t + frame_period), 1),
            'frame_w':         w,
            'frame_h':         h,
            'context':         current_context,
            'face_detected':   face_detected,
            'face_bbox':       _compute_face_bbox_norm(shared.get('landmarks')),
            'hr_bpm':          int(shared.get('heart_rate_bpm', 0) or 0),
            'hrv_rmssd':       shared.get('hrv_rmssd', None),
            'stress_score':    float(shared.get('stress_score', 0) or 0),
            'stress_category': shared.get('stress_category', 'Unknown'),
            'is_calibrating':  bool(getattr(stress_mod, 'is_calibrating', False)),
            'calibration_pct': int(getattr(stress_mod, 'calibration_progress', 0) * 100)
                                  if hasattr(stress_mod, 'calibration_progress') else 0,
            'action_units':    {k: float(v) for k, v in au.items() if isinstance(v, (int, float))},
            'hands':           hands,
            # Poker payload
            'cards_detected':  cards_detected,
            'cards':           cards_payload,
            'registered_hand': [c for c in registered_hand if c],
            'board_cards':     sorted(board_cards),
            'equity':          eq_state['eq'],
            'outs':            eq_state['outs'],
            'my_hands':        eq_state['top'].get('my_hands', []),
            'opp_hands':       eq_state['top'].get('opp_hands', []),
            'rh_hold':         rh_hold,
            'lh_hold':         lh_hold,
        }

        with _state['lock']:
            _state['frame_jpeg'] = buf.tobytes()
            _state['metrics'] = metrics
            _state['last_capture_time'] = time.time()

        # Schedule broadcast on the asyncio loop
        if _loop and _clients:
            _loop.call_soon_threadsafe(_schedule_broadcast)

        # Pace
        next_t += frame_period
        sleep = next_t - time.monotonic()
        if sleep > 0:
            time.sleep(sleep)
        else:
            next_t = time.monotonic()


def _schedule_broadcast():
    """Run on the asyncio loop. Fan out frame + metrics to every connected client."""
    if not _clients:
        return
    asyncio.create_task(_broadcast_to_clients())


async def _broadcast_to_clients():
    with _state['lock']:
        frame_bytes = _state['frame_jpeg']
        metrics = _state['metrics']
    if frame_bytes is None:
        return
    payload_text = json.dumps(metrics)
    dead = []
    for ws in list(_clients):
        try:
            await ws.send_bytes(frame_bytes)
            await ws.send_text(payload_text)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _clients.discard(ws)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _loop
    _loop = asyncio.get_running_loop()
    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()
    print("[server] Background capture thread started.")
    yield
    print("[server] Shutting down.")


app = FastAPI(lifespan=lifespan)
STATIC_DIR = Path(__file__).resolve().parent / 'static'
app.mount('/static', StaticFiles(directory=str(STATIC_DIR)), name='static')


@app.get('/')
async def index():
    from fastapi.responses import FileResponse
    return FileResponse(str(STATIC_DIR / 'index.html'))


@app.websocket('/ws')
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    _clients.add(ws)
    print(f"[server] client connected ({len(_clients)} total)")
    try:
        while True:
            # We don't expect messages from client right now, just keep alive
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _clients.discard(ws)
        print(f"[server] client disconnected ({len(_clients)} total)")


if __name__ == '__main__':
    import uvicorn
    host = os.environ.get('WEB_UI_HOST', '0.0.0.0')
    port = int(os.environ.get('WEB_UI_PORT', '8000'))
    print(f"[server] listening on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level='warning')
