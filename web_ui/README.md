# web_ui - Browser front-end

A web-based front-end for the face / rPPG / FACS / stress pipeline. Renders an
Apple Vision Pro-style frosted glass UI in the browser using real CSS
`backdrop-filter` so the glass actually refracts the live camera feed.

**Architecture in one line:** the browser owns the camera, the laptop owns the
ML. The phone (or laptop) browser captures its own camera with the standard
`getUserMedia` Web API, sends JPEG frames to the laptop server over a
WebSocket, and the server runs the existing Python pipeline on each frame and
sends back face / HR / HRV / stress / AU / hand-gesture metrics. The browser
draws its own camera feed and overlays the glass panels using the metrics.

Result:

- **No DroidCam needed.** The browser asks for camera permission once.
- **Phone displays its own camera with no lag** - the camera feed never makes
  a network round-trip.
- **Laptop still does the heavy lifting** - MediaPipe face landmarks, POS rPPG,
  FACS action units, stress score, hand-gesture detection. All in Python,
  unchanged from the OpenCV app.

This does **not** replace the OpenCV `unified_ar_system.py`. Both can live
side by side - run whichever you prefer.

---

## Layout

```
web_ui/
  server.py            FastAPI app + per-client pipeline + WebSocket endpoint
  static/
    index.html         <video> element + glass panels
    style.css          backdrop-filter glass styling
    app.js             getUserMedia, capture-and-upload loop, panel rendering
    manifest.json      PWA manifest (lets you "Add to Home Screen")
    icon.svg           App icon
  README.md            (this file)
```

---

## Run on the laptop

```bash
# 1. Install deps (one time)
pip install -r requirements.txt

# 2a. For phone access -> START WITH HTTPS (recommended)
python web_ui/server.py --https

# 2b. For laptop-only testing -> plain HTTP is fine
python web_ui/server.py
```

The `--https` flag auto-generates a self-signed certificate the first time
you run it (cached in `web_ui/.cert/`). Browsers only allow camera access
on **HTTPS** or **localhost** - so phones over LAN need HTTPS.

When started with `--https`, you will see:

```
[server] Generating self-signed certificate ...
[server] Cert written to ...\web_ui\.cert\server.crt (LAN IP: 192.168.1.42)
[server] listening on https://0.0.0.0:8000
[server] -> open https://localhost:8000 on the laptop
[server] -> open https://<laptop-ip>:8000 on the phone
[server] First visit: accept the 'not secure' warning (self-signed cert).
```

Open the URL from any modern browser:

- **Laptop:** `https://localhost:8000` -- accept the self-signed warning once.
- **Phone (same Wi-Fi):** `https://<laptop-ip>:8000` -- same, accept once.

To find the laptop's local IP:
- Windows: `ipconfig` -> "IPv4 Address" under your Wi-Fi adapter.
- macOS / Linux: `ifconfig | grep "inet "`.

The Windows firewall may pop up the first run asking to allow Python
through. Choose "Private networks" so phones on the LAN can reach it.

---

## Why the security warning the first time?

The certificate is **self-signed**: created by your laptop, not signed by a
public certificate authority. Browsers don't recognise it, so they show
"Your connection is not private" the first visit.

- **Chrome / Edge:** click "Advanced" -> "Proceed to ... (unsafe)".
- **Safari (iPhone):** click "Show details" -> "visit this website" -> confirm.

Once accepted, the browser remembers the cert for that hostname and doesn't
ask again. After acceptance the connection is in a "secure context" and
`getUserMedia` works. The traffic is encrypted; the warning is only because
the cert isn't from a known CA.

The cert is cached in `web_ui/.cert/` (gitignored) and reused on every
subsequent run. Delete that folder to regenerate (e.g. if your LAN IP
changed).

---

## Why the camera permission prompt?

After accepting the self-signed cert, the browser prompts for camera access.
This is the standard `navigator.mediaDevices.getUserMedia` call - same as
Google Meet, Zoom web, etc. Tap **Allow**.

If you blocked it by mistake, tap the camera icon in the address bar (or
your phone browser's site-info menu) and re-allow it.

---

## Install on the phone as a fullscreen app (PWA)

The page ships with a PWA manifest, so once you have it loading on the phone
you can install it on the home screen. The icon launches the AR overlay
fullscreen, no browser bars, like a native app.

### Android (Chrome / Edge)

1. With the laptop running the server, open `http://<laptop-ip>:8000` on the
   phone in **Chrome** (or Edge).
2. Chrome shows an "Install" / "Add to Home screen" prompt. If it doesn't,
   tap the browser **menu (3 dots)** -> **Install app** (or **Add to Home
   Screen**).
3. Confirm. A "Stoned" icon with a green heart appears on the home screen.
4. Tap the icon. Fullscreen, no Chrome UI - just the camera feed and the
   glass panels.

### iPhone / iPad (Safari)

1. Open `http://<laptop-ip>:8000` in **Safari**.
2. Tap **Share** -> **Add to Home Screen** -> **Add**.
3. Tap the icon. Fullscreen, no Safari UI.

---

## Environment variables (`.env` at repo root)

| Variable      | Default     | Effect                            |
|---------------|-------------|-----------------------------------|
| `WEB_UI_HOST` | `0.0.0.0`   | Address to bind the server to     |
| `WEB_UI_PORT` | `8000`      | Port to bind the server to        |

`CAMERA_INDEX` is no longer used by the web UI (the camera is opened by the
browser, not the Python process). It still affects `unified_ar_system.py`.

---

## What is shown

- **Status pill** (top centre): connection state.
- **Calibration banner**: appears the first ~12 s while the stress baseline
  is being captured. Progress bar fills as it completes.
- **Heart Rate / HRV / Stress panel**: anchored to the right of the face.
- **Expressions panel**: anchored to the left of the face. 10 Action Units
  with intensifying green-to-yellow-to-red bars.
- **Heart-rate graph**: full-width strip at the bottom. ~10 seconds of HR
  history.

Side panels hide automatically when no face is detected and re-anchor when it
comes back.

---

## Gestures

- **Right hand pinch over a panel**, hold for 2 seconds, then move your hand
  -> drag the panel. Release to drop. Position is saved per panel.
- **Left hand pinch over a panel**, hold for 2 seconds, then move up/down ->
  resize that specific panel. Release to lock. Size is saved per panel.

The 2-second hold filters out incidental brief pinches that happen during
normal hand movement. To make pinches register faster, change `PINCH_HOLD_MS`
in `web_ui/static/app.js`.

Panels auto-resolve overlap (push down when they would collide) and remember
positions across reloads via `localStorage`.

Reset all positions / sizes:

```js
localStorage.removeItem('stoned.panelOffsets.v1');
localStorage.removeItem('stoned.panelScales.v2');
```

Then refresh.

---

## How it talks to the server

WebSocket at `/ws`. Two message types:

- **Browser -> server:** binary JPEG frame (one per ~70 ms, ~15 fps,
  capped at 960px wide, JPEG quality 0.7).
- **Server -> browser:** JSON metrics (one reply per frame received).

The server uses an asyncio thread pool to run the Python pipeline so the
event loop never blocks. Each connected browser gets its own pipeline
instance (face mod, rPPG mod, FACS mod, stress mod, hand mod) so multiple
clients don't share state.

---

## Limits / caveats

- The OpenCV `unified_ar_system.py` includes poker card detection, gesture
  control (drag / pinch / showdowns), bluff predictions, and the Claude
  integration. This web UI is **face metrics only**. The poker side and the
  adaptive-learning bandit are not surfaced here yet.
- HTTPS or localhost is required for the camera to work (browser policy).
- A bad Wi-Fi connection between phone and laptop will throttle the upload
  frame rate. The pipeline still runs at whatever rate frames arrive.
- iOS Safari has flakier long-lived WebSocket support than Chrome on
  Android - if you see frequent reconnects on iPhone, try Chrome for iOS.

---

## Stop

`Ctrl+C` in the terminal running `web_ui/server.py`. The browser tab will
show "Reconnecting" and re-attempt every 1.5 s.
