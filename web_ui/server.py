"""
FastAPI WebSocket server for the Stoned web UI.

The browser captures the camera locally (via getUserMedia) and streams JPEG
frames to this server. The server runs the existing Python pipeline
(face / rPPG / FACS / stress / hand) on each frame and sends back the
metrics as JSON. The browser displays its own camera and overlays the
glass UI using the metrics.

This means:
  - No DroidCam needed. The phone (or laptop) talks directly to its own
    camera via the standard browser API.
  - The laptop still does the heavy ML; only the metrics travel back to
    the phone.

Run:
    python web_ui/server.py
Open on phone (same Wi-Fi):
    http://<laptop-ip>:8000
"""
import asyncio
import json
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Reuse the existing pipeline
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'micro_expressions'))

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

# Optional: full bluff-detection / adaptive learning stack. Loaded lazily so
# the server still runs if these modules are missing or fail to import.
try:
    from adaptive_learning import AdaptiveLearningSystem
except Exception as _e:
    print(f"[server] AdaptiveLearningSystem import failed: {_e}")
    AdaptiveLearningSystem = None

# Equity engine (treys). Optional - server still runs without it; equity
# panel + hand-rank lists just stay empty in that case.
try:
    from poker_hand.poker_engine import calculate_equity as _calc_equity
except Exception as _e:
    print(f"[server] poker_engine import failed: {_e}")
    _calc_equity = None


# -----------------------------------------------------------------------------
# Helpers (shared across client pipelines)
# -----------------------------------------------------------------------------

def _bbox_iou(a, b):
    """Intersection-over-union of two [x1, y1, x2, y2] bboxes. Used for
    cross-label spatial dedup when YOLO outputs two different labels for the
    same physical card."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0); iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1); iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _to_treys(label):
    """Convert a YOLO card label to Treys-format ('10s' -> 'Ts', 'Jh' -> 'Jh').
    Returns None if the label isn't a valid card. Mirrors the helper used in
    unified_ar_system.py."""
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


def _compute_face_bbox_norm(landmarks):
    if not landmarks:
        return None
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x0 = max(0.0, min(xs))
    y0 = max(0.0, min(ys))
    x1 = min(1.0, max(xs))
    y1 = min(1.0, max(ys))
    return {'x': x0, 'y': y0, 'w': x1 - x0, 'h': y1 - y0}


def _extract_hand_state(hand_data, frame_w, frame_h, ema_state=None, alpha=0.35):
    """Convert HandGestureDetector output to a small JSON-safe dict, with
    optional EMA smoothing on thumb / index / mid positions to stop the
    on-screen pinch loader from jittering with raw MediaPipe noise.

    ema_state: a dict that this function reads & writes for per-key EMA.
               Pass {} the first time; reuse for subsequent frames.
               If None, no smoothing is applied (raw positions).
    """
    if not hand_data:
        # Hand lost - clear smoothing state for this hand
        if ema_state is not None:
            ema_state.clear()
        return None
    pinch = hand_data.get('pinch') or {}
    tx = ty = ix = iy = None
    if pinch.get('thumb_pos') is not None:
        tx = float(pinch['thumb_pos'][0]) / max(1, frame_w)
        ty = float(pinch['thumb_pos'][1]) / max(1, frame_h)
    if pinch.get('index_pos') is not None:
        ix = float(pinch['index_pos'][0]) / max(1, frame_w)
        iy = float(pinch['index_pos'][1]) / max(1, frame_h)
    mid_x = (tx + ix) / 2 if (tx is not None and ix is not None) else None
    mid_y = (ty + iy) / 2 if (ty is not None and iy is not None) else None

    def _ema(key, x, y):
        """Return EMA-smoothed (x, y), updating ema_state in place."""
        if x is None or ema_state is None:
            return x, y
        prev = ema_state.get(key)
        if prev is None:
            ema_state[key] = (x, y)
            return x, y
        sx = (1 - alpha) * prev[0] + alpha * x
        sy = (1 - alpha) * prev[1] + alpha * y
        ema_state[key] = (sx, sy)
        return sx, sy

    tx, ty = _ema('thumb', tx, ty)
    ix, iy = _ema('index', ix, iy)
    mid_x, mid_y = _ema('mid', mid_x, mid_y)

    is_locked = bool(pinch.get('is_locked', False))
    hold = float(pinch.get('hold_progress', 0.0) or 0.0)
    # When the pinch has just locked, the underlying detector resets
    # hold_progress to 0. Show a full ring (1.0) so the loader visibly
    # completes instead of snapping back.
    if is_locked:
        hold = 1.0

    return {
        'pinch':         bool(pinch.get('active', False)),
        'distance':      float(pinch.get('distance', 0.0)),
        'is_left':       bool(hand_data.get('is_left', False)),
        'hold_progress': hold,
        'is_locked':     is_locked,
        'thumb':         None if tx is None else {'x': tx, 'y': ty},
        'index':         None if ix is None else {'x': ix, 'y': iy},
        'mid':           None if mid_x is None else {'x': mid_x, 'y': mid_y},
    }


# -----------------------------------------------------------------------------
# Per-client pipeline
# -----------------------------------------------------------------------------

class ClientPipeline:
    """One pipeline per connected browser. Modules are loaded lazily on the
    first frame so that connecting doesn't pay the MediaPipe init cost
    until a real frame actually arrives."""

    def __init__(self):
        self.shared = {
            'frame': None,
            'frame_dimensions': (0, 0),
            'face_detected': False,
            'landmarks': None,
            'timestamp': 0,
            'is_video_file': False,
            'frame_number': 0,
        }
        self.face_mod = None
        self.rppg_mod = None
        self.facs_mod = None
        self.stress_mod = None
        self.hand_mod = None
        self.learner  = None        # AdaptiveLearningSystem (optional)
        self._last_thumb_t = 0.0    # debounce for thumbs gestures
        self._showdown_pending = None   # 'BLUFFING' | 'STRONG' set when client sends a manual showdown
        # Tracks the face identity from the previous frame so we can flush
        # rPPG + stress buffers when the user pans to a new person.
        self._last_player_id = None
        # YOLO card detection (loaded lazily, optional)
        self.yolo = None
        self.yolo_attempted = False
        self.yolo_frame_skip = 2          # run YOLO every Nth frame
        self.frame_count = 0
        self.initialized = False
        self.last_log_t = 0.0

        # ---- Card stability machine (ported from unified_ar_system.py) ----
        # A label must appear in STABILITY_THRESHOLD consecutive frames
        # before it gets a stable box. Stable boxes use EMA smoothing on
        # their bbox so they don't jitter. Boxes fade out after
        # CARD_FADE_TIMEOUT frames of absence; the whole pool resets after
        # NO_CARDS_RESET_TIMEOUT frames of no detections at all.
        self.STABILITY_THRESHOLD     = 4
        self.FINALIZE_THRESHOLD      = 12
        self.CARD_FADE_TIMEOUT       = 60
        self.NO_CARDS_RESET_TIMEOUT  = 90
        self.BOX_SMOOTH_ALPHA        = 0.4   # EMA weight on new YOLO box
        self.card_history     = {}           # label -> consecutive-seen counter
        self.finalized_cards  = {}           # label -> {bbox, conf, last_seen}
        self.zero_card_frames = 0

        # ---- Hand-position smoothing ----
        # MediaPipe hand landmarks jitter frame-to-frame. Smoothing the
        # thumb / index / mid points before sending to the browser stops
        # the on-screen pinch loader from wobbling. Per-hand state.
        self.HAND_POS_EMA_ALPHA = 0.35    # lower = smoother, laggier
        self.hand_pos_ema = {'left': {}, 'right': {}}

        # ---- Manually-saved poker state (driven by browser buttons /
        #      pinch-locks). Treys-format strings ('Ah', 'Ts', etc.). ----
        self.registered_hand = []   # max 2 hole cards
        self.board_cards     = []   # max 5 community cards
        # Rising-edge detection for pinch-locks (so a held lock fires once).
        self._lh_lock_prev = False
        self._rh_lock_prev = False
        # Browser-initiated card commands (save/lock/reset). Drained on the
        # next pipeline frame.
        self._card_action_queue = []
        # Equity cache. A flop enumeration is ~150k treys evaluations
        # (~6s) - too slow to run on the frame loop, so we recompute in a
        # background thread and just emit the most recent result.
        self._equity_cache = {
            'key':       None,    # ((hand_tuple), (board_tuple)) of last result
            'equity':    0.0,
            'outs':      0,
            'my_hands':  [],
            'opp_hands': [],
            'computing': False,
        }
        self._equity_lock = threading.Lock()
        self._equity_running_key = None    # the key currently being computed

    def _init_modules(self):
        print("[pipeline] Initialising modules for new client ...")
        self.face_mod   = FaceDetectionModule()
        self.rppg_mod   = RPPGModule(buffer_size=300, fps=30)
        self.facs_mod   = FACSModule()
        self.stress_mod = StressDetectorModule()
        try:
            hm = HandGestureDetector()
            self.hand_mod = hm if hm.detector is not None else None
        except Exception as e:
            print(f"[pipeline] HandGestureDetector init failed: {e}")
            self.hand_mod = None

        for mod in (self.face_mod, self.rppg_mod, self.facs_mod, self.stress_mod):
            if hasattr(mod, 'initialize'):
                try:
                    mod.initialize(self.shared)
                except Exception as e:
                    print(f"[pipeline] init {type(mod).__name__} failed: {e}")

        # Adaptive learning (optional - bluff prediction + thumbs showdowns)
        if AdaptiveLearningSystem is not None:
            try:
                self.learner = AdaptiveLearningSystem()
                print("[pipeline] AdaptiveLearningSystem ready")
            except Exception as e:
                print(f"[pipeline] AdaptiveLearningSystem init failed: {e}")
                self.learner = None

        self.initialized = True
        print("[pipeline] Modules ready")

    def _init_yolo(self):
        """Load the poker-card YOLO model lazily on the first frame so the
        client doesn't pay the import cost until they actually need it."""
        if self.yolo_attempted:
            return
        self.yolo_attempted = True
        try:
            from ultralytics import YOLO
            model_path = ROOT / 'poker_hand' / 'poker_v1.pt'
            if not model_path.exists():
                print(f"[pipeline] YOLO model missing at {model_path} - "
                      "card detection disabled.")
                return
            self.yolo = YOLO(str(model_path))
            print(f"[pipeline] YOLO loaded ({len(self.yolo.names)} classes)")
        except Exception as e:
            print(f"[pipeline] YOLO load failed: {e}")
            self.yolo = None

    def process_sync(self, jpeg_bytes):
        """Decode the frame, run all modules, return the metrics dict."""
        if not self.initialized:
            self._init_modules()

        nparr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return None

        h, w = frame.shape[:2]
        self.shared.update({
            'frame': frame,
            'frame_dimensions': (h, w),
            'timestamp': time.time(),
            'frame_number': self.shared['frame_number'] + 1,
        })

        # Periodic diagnostic so the user can confirm in the terminal that
        # frames are flowing without enabling debug-level logs.
        now_t = time.monotonic()
        if now_t - self.last_log_t > 3.0:
            mb, mg, mr = frame.mean(axis=(0, 1))[:3]
            face_ok = self.shared.get('face_detected', False)
            hr_val  = self.shared.get('heart_rate_bpm', 0)
            print(f"[pipeline] frame#{self.shared['frame_number']} "
                  f"{w}x{h} mean BGR=({mb:.0f},{mg:.0f},{mr:.0f}) "
                  f"face={face_ok} hr={hr_val}")
            self.last_log_t = now_t

        for mod in (self.face_mod, self.rppg_mod, self.facs_mod, self.stress_mod):
            try:
                mod.process(self.shared)
            except Exception as e:
                print(f"[pipeline] {type(mod).__name__}.process error: {e}")

        if self.hand_mod is not None:
            try:
                self.hand_mod.process(frame, self.shared)
            except Exception as e:
                print(f"[pipeline] HandGestureDetector.process error: {e}")

        # ---- Adaptive learning (bluff detection) ----
        if self.learner is not None:
            self._drive_learner()

        # YOLO card detection + stability machine (frame-skipped)
        self._init_yolo()
        self.frame_count += 1
        if self.yolo and self.frame_count % self.yolo_frame_skip == 0:
            self._run_yolo_and_stabilise(frame)

        # ---- Card save / board lock (pinch + button driven) ----
        self._drive_card_state()

        return self._build_metrics(w, h)

    def _drive_card_state(self):
        """Watch pinch-lock rising edges and drain any browser-issued card
        commands. Left pinch-lock saves the visible cards as the registered
        hand; right pinch-lock locks the visible cards into the board."""
        gestures = self.shared.get('hand_gestures', {}) or {}
        lh = gestures.get('left_hand')  or {}
        rh = gestures.get('right_hand') or {}
        lh_locked = bool((lh.get('pinch') or {}).get('is_locked', False))
        rh_locked = bool((rh.get('pinch') or {}).get('is_locked', False))
        if lh_locked and not self._lh_lock_prev:
            self._save_hand_from_visible()
        if rh_locked and not self._rh_lock_prev:
            self._lock_board_from_visible()
        self._lh_lock_prev = lh_locked
        self._rh_lock_prev = rh_locked

        while self._card_action_queue:
            action = self._card_action_queue.pop(0)
            if action == 'save_hand':
                self._save_hand_from_visible()
            elif action == 'lock_board':
                self._lock_board_from_visible()
            elif action == 'reset_hand':
                if self.registered_hand:
                    print("[pipeline] hand reset")
                self.registered_hand = []
            elif action == 'reset_board':
                if self.board_cards:
                    print("[pipeline] board reset")
                self.board_cards = []
            elif action == 'reset_all':
                if self.registered_hand or self.board_cards:
                    print("[pipeline] hand + board reset")
                self.registered_hand = []
                self.board_cards = []

    def _save_hand_from_visible(self):
        """Pick the 2 largest currently-finalised cards as the hole cards."""
        cards = list(self.finalized_cards.values())
        if len(cards) < 2:
            print(f"[pipeline] save_hand ignored: only {len(cards)} stable card(s)")
            return False
        cards.sort(
            key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]),
            reverse=True,
        )
        picked = []
        for c in cards[:2]:
            t = _to_treys(c['label'])
            if t:
                picked.append(t)
        if len(picked) != 2:
            print("[pipeline] save_hand ignored: bad treys conversion")
            return False
        self.registered_hand = picked
        print(f"[pipeline] hand registered: {picked}")
        return True

    def _lock_board_from_visible(self):
        """Add all currently-finalised cards (excluding the hand) to the board."""
        added = 0
        for c in self.finalized_cards.values():
            t = _to_treys(c['label'])
            if not t:
                continue
            if t in self.board_cards or t in self.registered_hand:
                continue
            if len(self.board_cards) >= 5:
                break
            self.board_cards.append(t)
            added += 1
        if added:
            print(f"[pipeline] board locked +{added} -> {self.board_cards}")
        return added > 0

    def queue_card_action(self, action):
        """Called from the WS handler for browser button presses."""
        if action in ('save_hand', 'lock_board',
                      'reset_hand', 'reset_board', 'reset_all'):
            self._card_action_queue.append(action)

    def _run_yolo_and_stabilise(self, frame):
        """Run YOLO and update the stability machine. Cards must appear in
        STABILITY_THRESHOLD consecutive frames before they finalise; finalised
        boxes are EMA-smoothed and persist for CARD_FADE_TIMEOUT frames after
        the card disappears."""
        # Pull thresholds from config.py if available (user-tunable).
        try:
            import config as _cfg
            conf_th = float(getattr(_cfg, 'YOLO_CONF', 0.7))
            iou_th  = float(getattr(_cfg, 'YOLO_IOU',  0.15))
            imgsz   = int(getattr(_cfg, 'YOLO_IMGSZ', 640))
        except Exception:
            conf_th, iou_th, imgsz = 0.7, 0.15, 640

        try:
            results = self.yolo(frame, verbose=False, conf=conf_th,
                                iou=iou_th, imgsz=imgsz)
        except Exception as e:
            print(f"[pipeline] YOLO inference error: {e}")
            return

        # 1. Per-label dedup: keep highest-confidence detection per label.
        #    Stops the same card label from appearing twice.
        current = {}
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                lbl  = self.yolo.names[int(box.cls[0])]
                conf = float(box.conf[0])
                bb   = box.xyxy[0].tolist()  # [x1, y1, x2, y2] px
                if lbl not in current or conf > current[lbl]['confidence']:
                    current[lbl] = {'confidence': conf, 'bbox': bb}

        # 2. Cross-label spatial dedup: when two DIFFERENT labels share most
        #    of the same bounding box (IoU > 0.45), the model is confused
        #    between two visually-similar cards (e.g. 6c vs 6s). Keep the
        #    higher-confidence one only - that's almost always the right pick.
        OVERLAP_DEDUP_IOU = 0.45
        items = sorted(current.items(),
                       key=lambda kv: kv[1]['confidence'], reverse=True)
        kept = {}
        for lbl, c in items:
            keep = True
            for klbl, kc in kept.items():
                if _bbox_iou(c['bbox'], kc['bbox']) > OVERLAP_DEDUP_IOU:
                    keep = False
                    break
            if keep:
                kept[lbl] = c
        current = kept

        # Increment counter for labels seen this frame
        for lbl in current:
            self.card_history[lbl] = min(self.card_history.get(lbl, 0) + 1,
                                          self.FINALIZE_THRESHOLD)
        # Decay counter for labels NOT seen this frame and not yet finalised
        for lbl in list(self.card_history.keys()):
            if lbl not in current and lbl not in self.finalized_cards:
                self.card_history[lbl] -= 1
                if self.card_history[lbl] <= 0:
                    del self.card_history[lbl]

        # Promote labels that have hit STABILITY_THRESHOLD into finalised
        for lbl, c in current.items():
            if (lbl not in self.finalized_cards
                    and self.card_history.get(lbl, 0) >= self.STABILITY_THRESHOLD):
                self.finalized_cards[lbl] = {
                    'label':      lbl,
                    'confidence': c['confidence'],
                    'bbox':       list(c['bbox']),
                    'last_seen':  self.frame_count,
                }

        # Update finalised entries (EMA on bbox, max on confidence)
        a = self.BOX_SMOOTH_ALPHA
        for lbl, c in current.items():
            if lbl in self.finalized_cards:
                fin = self.finalized_cards[lbl]
                fbbox = fin['bbox']
                nbbox = c['bbox']
                # EMA: smoothed = (1-a)*old + a*new
                fin['bbox'] = [
                    (1 - a) * fbbox[i] + a * nbbox[i] for i in range(4)
                ]
                fin['confidence'] = max(fin['confidence'], c['confidence'])
                fin['last_seen']  = self.frame_count

        # Track empty-frame streak; full reset after long absence
        if not current:
            self.zero_card_frames += 1
        else:
            self.zero_card_frames = 0
        if self.zero_card_frames > self.NO_CARDS_RESET_TIMEOUT:
            if self.finalized_cards or self.card_history:
                print("[pipeline] no cards seen recently - resetting card pool")
            self.finalized_cards.clear()
            self.card_history.clear()

        # Expire individual cards that have been gone too long
        expired = [lbl for lbl, d in self.finalized_cards.items()
                   if self.frame_count - d['last_seen'] > self.CARD_FADE_TIMEOUT]
        for lbl in expired:
            del self.finalized_cards[lbl]

    def _refresh_equity(self):
        """Spawn a background equity calc when hand/board state changes.
        Cheap on every frame (one dict comparison + lock) while no state has
        changed; on a change, fires off a worker thread and lets the existing
        cached result stay live until the new one comes in."""
        if _calc_equity is None or len(self.registered_hand) != 2:
            with self._equity_lock:
                if self._equity_cache['key'] is not None or self._equity_cache['computing']:
                    self._equity_cache.update({
                        'key': None, 'equity': 0.0, 'outs': 0,
                        'my_hands': [], 'opp_hands': [], 'computing': False,
                    })
                self._equity_running_key = None
            return
        key = (tuple(sorted(self.registered_hand)),
               tuple(sorted(self.board_cards)))
        with self._equity_lock:
            if self._equity_cache['key'] == key:
                return            # already have the right answer cached
            if self._equity_running_key == key:
                return            # already computing - just wait
            self._equity_running_key = key
            self._equity_cache['computing'] = True

        hand  = list(self.registered_hand)
        board = list(self.board_cards)

        def _worker():
            try:
                eq, outs, breakdown = _calc_equity(hand, board)
            except Exception as e:
                print(f"[pipeline] equity calc error: {e}")
                with self._equity_lock:
                    if self._equity_running_key == key:
                        self._equity_running_key = None
                        self._equity_cache['computing'] = False
                return
            with self._equity_lock:
                # Drop the result if state moved on while we were computing.
                # The newer change will already have spawned its own worker.
                if self._equity_running_key != key:
                    return
                self._equity_cache.update({
                    'key':       key,
                    'equity':    float(eq),
                    'outs':      int(outs),
                    'my_hands':  breakdown.get('my_hands', []),
                    'opp_hands': breakdown.get('opp_hands', []),
                    'computing': False,
                })
                self._equity_running_key = None

        threading.Thread(target=_worker, daemon=True).start()

    def _build_metrics(self, w, h):
        # Refresh poker equity cache (no-op if hand+board unchanged).
        # Heavy calc runs in a background thread; we just read the latest.
        self._refresh_equity()
        with self._equity_lock:
            eq_snapshot = {
                'equity':    float(self._equity_cache.get('equity', 0.0)),
                'outs':      int(self._equity_cache.get('outs', 0)),
                'my_hands':  list(self._equity_cache.get('my_hands', [])),
                'opp_hands': list(self._equity_cache.get('opp_hands', [])),
                'computing': bool(self._equity_cache.get('computing', False)),
            }
        au = self.shared.get('action_units', {}) or {}
        gestures = self.shared.get('hand_gestures', {}) or {}
        hands = {
            'left':  _extract_hand_state(
                gestures.get('left_hand'),  w, h,
                ema_state=self.hand_pos_ema['left'],
                alpha=self.HAND_POS_EMA_ALPHA,
            ),
            'right': _extract_hand_state(
                gestures.get('right_hand'), w, h,
                ema_state=self.hand_pos_ema['right'],
                alpha=self.HAND_POS_EMA_ALPHA,
            ),
        }
        # Cards from the stability machine (EMA-smoothed, only labels that
        # have survived STABILITY_THRESHOLD frames). The browser expects the
        # bbox as {x, y, w, h} (normalised) and a `status` string.
        cards = []
        for c in self.finalized_cards.values():
            x1, y1, x2, y2 = c['bbox']
            cards.append({
                'label':      c['label'],
                'treys':      _to_treys(c['label']),
                'confidence': c['confidence'],
                'bbox':       {
                    'x': x1 / w,
                    'y': y1 / h,
                    'w': (x2 - x1) / w,
                    'h': (y2 - y1) / h,
                },
                'status':     'stable',
            })

        face_detected = bool(self.shared.get('face_detected'))
        # Context priority mirrors the OpenCV unified_ar_system: any cards
        # detected -> poker mode (or hybrid_poker if a face is also seen);
        # otherwise face mode if a face is seen; otherwise none.
        if cards and face_detected:
            context = 'hybrid_poker'
        elif cards:
            context = 'poker'
        elif face_detected:
            context = 'face'
        else:
            context = 'none'

        # Bluff verdict + learner calibration state
        verdict = None
        learner_calibrating = False
        learner_calib_pct = 0
        player_id = None
        if self.learner is not None:
            try:
                verdict = self._build_verdict()
            except Exception:
                verdict = None
            try:
                learner_calibrating = bool(self.learner.baseline.is_calibrating)
                learner_calib_pct   = int(self.learner.baseline.get_calibration_progress() or 0)
                player_id = self.learner.current_player_id
            except Exception:
                pass

        return {
            'frame_w':         w,
            'frame_h':         h,
            'context':         context,
            'face_detected':   face_detected,
            'face_bbox':       _compute_face_bbox_norm(self.shared.get('landmarks')),
            'hr_bpm':          int(self.shared.get('heart_rate_bpm', 0) or 0),
            'hrv_rmssd':       self.shared.get('hrv_rmssd', None),
            'stress_score':    float(self.shared.get('stress_score', 0) or 0),
            'stress_category': self.shared.get('stress_category', 'Unknown'),
            # Calibration: prefer the learner's baseline if active (it's the
            # one driving bluff predictions); fall back to stress detector.
            'is_calibrating':  learner_calibrating or bool(getattr(self.stress_mod, 'is_calibrating', False)),
            'calibration_pct': learner_calib_pct or (
                int(getattr(self.stress_mod, 'calibration_progress', 0) * 100)
                if hasattr(self.stress_mod, 'calibration_progress') else 0
            ),
            # Bluff verdict (None until baseline + at least one prediction frame)
            'verdict':         verdict,
            'player_id':       player_id,
            'action_units':    {k: float(v) for k, v in au.items() if isinstance(v, (int, float))},
            'hands':           hands,
            'cards':           cards,
            # Pinch-loader animation: drives the circular progress around each
            # hand's mid-point as the user holds the pinch. 0..1.
            'lh_hold':         (hands['left']  or {}).get('hold_progress', 0.0),
            'rh_hold':         (hands['right'] or {}).get('hold_progress', 0.0),
            # Manually-saved poker state + computed equity (cached, runs in
            # a background thread when hand+board changes; see _refresh_equity).
            'registered_hand': list(self.registered_hand),
            'board_cards':     list(self.board_cards),
            'equity':          eq_snapshot['equity'],
            'outs':            eq_snapshot['outs'],
            'my_hands':        eq_snapshot['my_hands'],
            'opp_hands':       eq_snapshot['opp_hands'],
            'equity_computing': eq_snapshot['computing'],
        }

    def _drive_learner(self):
        """Per-frame: feed the AdaptiveLearningSystem, detect thumbs gestures
        for showdowns, and respond to manual STRONG/BLUFF button presses
        from the browser."""
        learner = self.learner
        if learner is None:
            return

        face_detected = bool(self.shared.get('face_detected'))
        landmarks     = self.shared.get('landmarks')
        hr            = self.shared.get('heart_rate_bpm', 70) or 70
        stress        = self.shared.get('stress_score', 0) or 0
        au            = self.shared.get('action_units', {}) or {}
        # Inject blink + HRV as synthetic AUs (matches unified_ar_system).
        au_values = dict(au)
        au_values['blink_rate'] = self.shared.get('blink_factor_raw',  0.0)
        au_values['hrv']        = self.shared.get('hrv_deviation_raw', 0.0)

        if face_detected and landmarks:
            try:
                learner.on_face_detected(landmarks)
            except Exception as e:
                print(f"[pipeline] learner.on_face_detected error: {e}")

            # ---- Face-identity-change reset ----
            # The learner identifies players via their face embedding. When
            # the user pans to a new opponent, current_player_id flips to a
            # different value. The rPPG green/BPM buffers and the stress
            # baseline still belong to the previous person, so the new
            # person's HR + stress would be wrong for ~10 s. Flush both
            # whenever the identity actually changes (not on first lock).
            new_pid = learner.current_player_id
            if new_pid and new_pid != self._last_player_id:
                if self._last_player_id is not None:
                    print(f"[pipeline] face switch "
                          f"{self._last_player_id[:8]} -> {new_pid[:8]} "
                          "(flushing rPPG + stress)")
                    try:
                        if self.rppg_mod and hasattr(self.rppg_mod, 'reset'):
                            self.rppg_mod.reset()
                    except Exception as e:
                        print(f"[pipeline] rppg reset error: {e}")
                    try:
                        if self.stress_mod and hasattr(self.stress_mod, 'reset'):
                            self.stress_mod.reset()
                    except Exception as e:
                        print(f"[pipeline] stress reset error: {e}")
                self._last_player_id = new_pid

            # Calibration vs prediction split
            if learner.baseline.is_calibrating:
                try:
                    learner.add_calibration_sample(hr, stress, au_values)
                except Exception as e:
                    print(f"[pipeline] add_calibration_sample error: {e}")
            elif learner.baseline.has_baseline() and learner.current_player_id:
                try:
                    learner.get_prediction(hr, stress, au_values)
                except Exception as e:
                    print(f"[pipeline] learner.get_prediction error: {e}")
        else:
            try:
                learner.on_face_lost()
            except Exception:
                pass

        # ---- Thumbs gestures from camera-visible hands ----
        gestures = self.shared.get('hand_gestures', {}) or {}
        now_t = time.time()
        if (face_detected and learner.current_player_id
                and (now_t - self._last_thumb_t) > 3.0):
            for side in ('left_hand', 'right_hand'):
                hd = gestures.get(side)
                if not hd:
                    continue
                ts = hd.get('thumb_signal') or {}
                if ts.get('fired') and ts.get('signal') in ('up', 'down'):
                    was_bluffing = (ts['signal'] == 'down')
                    self._record_showdown(was_bluffing)
                    self._last_thumb_t = now_t
                    if self.hand_mod is not None:
                        hn = 'Left' if side == 'left_hand' else 'Right'
                        try: self.hand_mod.reset_thumb_signal(hn)
                        except Exception: pass
                    break

        # ---- Browser-initiated showdown (button press) ----
        if self._showdown_pending is not None:
            was_bluffing = (self._showdown_pending == 'BLUFFING')
            self._showdown_pending = None
            if (now_t - self._last_thumb_t) > 1.0 and learner.current_player_id:
                self._record_showdown(was_bluffing)
                self._last_thumb_t = now_t

    def _record_showdown(self, was_bluffing):
        if not self.learner:
            return
        try:
            self.learner.on_showdown(was_bluffing=was_bluffing)
            self.learner.start_hand()
            label = "BLUFFING" if was_bluffing else "STRONG HAND"
            print(f"[pipeline] showdown recorded: {label}")
        except Exception as e:
            print(f"[pipeline] on_showdown error: {e}")

    def _build_verdict(self):
        """Return the current bluff verdict to send in metrics, or None if
        nothing meaningful to display yet (no baseline, no player, etc.)."""
        learner = self.learner
        if not learner or not learner.current_player_id:
            return None
        # Use the trained model when there's enough data, else heuristic.
        try:
            pred = learner.get_prediction(
                self.shared.get('heart_rate_bpm', 70) or 70,
                self.shared.get('stress_score', 0) or 0,
                self.shared.get('action_units', {}) or {},
            )
        except Exception:
            pred = None
        samples = (pred or {}).get('samples', 0) if pred else 0
        MIN_PRED_SAMPLES = 8
        if pred and samples >= MIN_PRED_SAMPLES:
            p_bluff = float(pred.get('p_bluff', 0.5))
            mode    = f"TRAINED  {samples}obs"
            profile = pred.get('personality_profile')
        else:
            heur = None
            try:
                heur = learner.get_heuristic_prediction()
            except Exception:
                heur = None
            if not heur:
                return None
            p_bluff = float(heur.get('p_bluff', 0.5))
            mode    = "SIGNAL"
            profile = None
        return {
            'prediction':  'BLUFFING' if p_bluff > 0.5 else 'STRONG',
            'p_bluff':     p_bluff,
            'confidence':  abs(p_bluff - 0.5) * 2,
            'mode_tag':    mode,
            'profile':     profile,
            'samples':     int(samples or 0),
        }

    def queue_showdown(self, was_bluffing):
        """Called from the WS handler when the browser sends a STRONG/BLUFF
        button press. Picked up on the next pipeline frame."""
        self._showdown_pending = 'BLUFFING' if was_bluffing else 'STRONG'

    def cleanup(self):
        if self.hand_mod and hasattr(self.hand_mod, 'cleanup'):
            try:
                self.hand_mod.cleanup()
            except Exception:
                pass
        for mod in (self.face_mod, self.rppg_mod, self.facs_mod, self.stress_mod):
            if mod and hasattr(mod, 'cleanup'):
                try:
                    mod.cleanup()
                except Exception:
                    pass


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[server] Ready. Open http://<laptop-ip>:8000 on a phone "
          "(same Wi-Fi) and allow camera access.")
    yield
    print("[server] Shutting down.")


app = FastAPI(lifespan=lifespan)
STATIC_DIR = Path(__file__).resolve().parent / 'static'
app.mount('/static', StaticFiles(directory=str(STATIC_DIR)), name='static')


@app.get('/')
async def index():
    return FileResponse(str(STATIC_DIR / 'index.html'))


@app.websocket('/ws')
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    pipeline = ClientPipeline()
    print("[server] client connected")
    loop = asyncio.get_running_loop()
    try:
        while True:
            msg = await ws.receive()
            if msg.get('type') == 'websocket.disconnect':
                break

            frame_bytes = msg.get('bytes')
            if frame_bytes:
                metrics = await loop.run_in_executor(
                    None, pipeline.process_sync, frame_bytes
                )
                if metrics:
                    try:
                        await ws.send_text(json.dumps(metrics))
                    except Exception:
                        break
                continue

            text = msg.get('text')
            if text:
                # Browser-initiated commands (button presses). Best-effort
                # JSON parse; ignore non-JSON text (keep-alive pings).
                try:
                    cmd = json.loads(text)
                except Exception:
                    continue
                if not isinstance(cmd, dict):
                    continue
                ctype = cmd.get('type')
                if ctype == 'showdown':
                    pipeline.queue_showdown(bool(cmd.get('was_bluffing')))
                elif ctype == 'card_action':
                    pipeline.queue_card_action(str(cmd.get('action') or ''))
    except WebSocketDisconnect:
        pass
    finally:
        pipeline.cleanup()
        print("[server] client disconnected")


def _ensure_self_signed_cert(cert_dir: Path):
    """Generate a self-signed TLS cert for localhost + the current LAN IP if
    one isn't already cached. Browsers will warn the first time you visit
    (because the cert isn't trusted by the OS), but after you accept once,
    the page is in a secure context and getUserMedia works.
    """
    cert_path = cert_dir / 'server.crt'
    key_path  = cert_dir / 'server.key'
    cert_dir.mkdir(parents=True, exist_ok=True)
    if cert_path.exists() and key_path.exists():
        return str(cert_path), str(key_path)

    print("[server] Generating self-signed certificate ...")
    try:
        from datetime import datetime, timedelta
        import ipaddress
        import socket
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
    except ImportError:
        raise RuntimeError(
            "HTTPS mode requires the 'cryptography' package. "
            "Run:  pip install cryptography"
        )

    # Best-effort detection of the LAN IP so the cert SAN matches whatever
    # the phone actually uses to reach the laptop. The trick: connect a UDP
    # socket somewhere - the OS picks the right outbound interface, no
    # packets are actually sent.
    lan_ip = '127.0.0.1'
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        lan_ip = s.getsockname()[0]
        s.close()
    except Exception:
        pass

    san = [
        x509.DNSName('localhost'),
        x509.IPAddress(ipaddress.IPv4Address('127.0.0.1')),
    ]
    if lan_ip != '127.0.0.1':
        try:
            san.append(x509.IPAddress(ipaddress.IPv4Address(lan_ip)))
        except Exception:
            pass

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'stoned-local')])
    cert = (x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.utcnow() - timedelta(days=1))
        .not_valid_after(datetime.utcnow() + timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        .sign(key, hashes.SHA256()))

    with open(cert_path, 'wb') as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    with open(key_path, 'wb') as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))
    print(f"[server] Cert written to {cert_path} (LAN IP: {lan_ip})")
    return str(cert_path), str(key_path)


if __name__ == '__main__':
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description='Stoned web UI server')
    parser.add_argument('--https', action='store_true',
        help='Run with HTTPS using a self-signed cert (required for camera '
             'access from a phone over LAN; not needed for laptop localhost).')
    parser.add_argument('--port', type=int,
        default=int(os.environ.get('WEB_UI_PORT', '8000')),
        help='Port to bind to (default: 8000).')
    parser.add_argument('--host', default=os.environ.get('WEB_UI_HOST', '0.0.0.0'),
        help='Host/interface to bind to (default: 0.0.0.0).')
    args = parser.parse_args()

    use_https = args.https or os.environ.get('WEB_UI_HTTPS') == '1'

    if use_https:
        cert_dir = Path(__file__).resolve().parent / '.cert'
        cert_path, key_path = _ensure_self_signed_cert(cert_dir)
        print(f"[server] listening on https://{args.host}:{args.port}")
        print(f"[server] -> open https://localhost:{args.port} on the laptop")
        print(f"[server] -> open https://<laptop-ip>:{args.port} on the phone")
        print(f"[server] First visit: accept the 'not secure' warning (self-signed cert).")
        uvicorn.run(app, host=args.host, port=args.port, log_level='warning',
                    ssl_keyfile=key_path, ssl_certfile=cert_path)
    else:
        print(f"[server] listening on http://{args.host}:{args.port}")
        print(f"[server] CAMERA NOTE: phones cannot use the camera over plain http://.")
        print(f"[server] Restart with --https to allow phone camera access.")
        uvicorn.run(app, host=args.host, port=args.port, log_level='warning')
