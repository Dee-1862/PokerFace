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
import json
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

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


CAMERA_INDEX = int(os.environ.get('CAMERA_INDEX', '0'))
JPEG_QUALITY = 70
TARGET_FPS = 25

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

        for mod in (face_mod, rppg_mod, facs_mod, stress_mod):
            try:
                mod.process(shared)
            except Exception as e:
                # Don't let one module take down the whole loop
                print(f"[server] {type(mod).__name__}.process error: {e}")

        # Encode frame
        ok2, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok2:
            continue

        # Build metrics payload
        au = shared.get('action_units', {}) or {}
        metrics = {
            'fps':             round(1.0 / max(1e-3, time.monotonic() - next_t + frame_period), 1),
            'frame_w':         w,
            'frame_h':         h,
            'face_detected':   bool(shared.get('face_detected')),
            'face_bbox':       _compute_face_bbox_norm(shared.get('landmarks')),
            'hr_bpm':          int(shared.get('heart_rate_bpm', 0) or 0),
            'hrv_rmssd':       shared.get('hrv_rmssd', None),
            'stress_score':    float(shared.get('stress_score', 0) or 0),
            'stress_category': shared.get('stress_category', 'Unknown'),
            'is_calibrating':  bool(getattr(stress_mod, 'is_calibrating', False)),
            'calibration_pct': int(getattr(stress_mod, 'calibration_progress', 0) * 100)
                                  if hasattr(stress_mod, 'calibration_progress') else 0,
            'action_units':    {k: float(v) for k, v in au.items() if isinstance(v, (int, float))},
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
