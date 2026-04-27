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


# -----------------------------------------------------------------------------
# Helpers (shared across client pipelines)
# -----------------------------------------------------------------------------

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


def _extract_hand_state(hand_data, frame_w, frame_h):
    if not hand_data:
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
    return {
        'pinch':    bool(pinch.get('active', False)),
        'distance': float(pinch.get('distance', 0.0)),
        'is_left':  bool(hand_data.get('is_left', False)),
        'thumb':    None if tx is None else {'x': tx, 'y': ty},
        'index':    None if ix is None else {'x': ix, 'y': iy},
        'mid':      None if mid_x is None else {'x': mid_x, 'y': mid_y},
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
        # YOLO card detection (loaded lazily, optional)
        self.yolo = None
        self.yolo_attempted = False
        self.yolo_frame_skip = 2          # run YOLO every Nth frame
        self.last_yolo_cards = []         # persisted between skipped frames
        self.frame_count = 0
        self.initialized = False
        self.last_log_t = 0.0

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

        # YOLO card detection (frame-skipped for performance)
        self._init_yolo()
        self.frame_count += 1
        if self.yolo and self.frame_count % self.yolo_frame_skip == 0:
            try:
                results = self.yolo(frame, verbose=False, conf=0.5,
                                    iou=0.15, imgsz=640)
                cards = []
                for r in results:
                    if r.boxes is None:
                        continue
                    for box in r.boxes:
                        lbl  = self.yolo.names[int(box.cls[0])]
                        conf = float(box.conf[0])
                        bb   = box.xyxy[0].tolist()  # [x1, y1, x2, y2] px
                        cards.append({'label': lbl, 'confidence': conf, 'bbox': bb})
                self.last_yolo_cards = cards
            except Exception as e:
                print(f"[pipeline] YOLO inference error: {e}")

        return self._build_metrics(w, h)

    def _build_metrics(self, w, h):
        au = self.shared.get('action_units', {}) or {}
        gestures = self.shared.get('hand_gestures', {}) or {}
        hands = {
            'left':  _extract_hand_state(gestures.get('left_hand'),  w, h),
            'right': _extract_hand_state(gestures.get('right_hand'), w, h),
        }
        # Cards as normalised bboxes so the browser can map them through the
        # same object-fit-cover transform used for the face bbox.
        cards = []
        for c in self.last_yolo_cards:
            x1, y1, x2, y2 = c['bbox']
            cards.append({
                'label':      c['label'],
                'treys':      _to_treys(c['label']),
                'confidence': c['confidence'],
                'bbox':       [x1 / w, y1 / h, x2 / w, y2 / h],
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
            'is_calibrating':  bool(getattr(self.stress_mod, 'is_calibrating', False)),
            'calibration_pct': int(getattr(self.stress_mod, 'calibration_progress', 0) * 100)
                                if hasattr(self.stress_mod, 'calibration_progress') else 0,
            'action_units':    {k: float(v) for k, v in au.items() if isinstance(v, (int, float))},
            'hands':           hands,
            'cards':           cards,
            # Poker-engine placeholders: the full engine (gesture-based hand
            # registration, equity, hand-rank lists) is a separate follow-up.
            # Browser handles empty arrays gracefully (shows hint state).
            'registered_hand': [],
            'board_cards':     [],
            'equity':          0.0,
            'outs':            0,
            'my_hands':        [],
            'opp_hands':       [],
            'lh_hold':         0.0,
            'rh_hold':         0.0,
        }

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
                # CPU-bound work goes to a worker thread so the asyncio loop
                # can keep handling other websockets and the network buffer
                # doesn't back up.
                metrics = await loop.run_in_executor(
                    None, pipeline.process_sync, frame_bytes
                )
                if metrics:
                    try:
                        await ws.send_text(json.dumps(metrics))
                    except Exception:
                        break
            # Text messages from the browser (e.g. ping) are intentionally
            # ignored - the server doesn't need them.
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
