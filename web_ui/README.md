# web_ui - Browser front-end

A web-based front-end for the face / rPPG / FACS / stress pipeline. Renders an
Apple Vision Pro-style frosted glass UI in the browser using real CSS
`backdrop-filter` so the glass actually refracts the live camera feed.

The Python pipeline (face landmarking, rPPG, action units, stress score) runs
unchanged in the background. A FastAPI WebSocket server pushes the latest
JPEG frame and the metrics JSON to any browser tab / phone connected to the
laptop.

This does **not** replace the OpenCV `unified_ar_system.py` app. Both can live
side by side - run whichever you prefer.

---

## Layout

```
web_ui/
  server.py            FastAPI app + capture thread + WebSocket broadcaster
  static/
    index.html         Page layout (camera image + glass panels)
    style.css          Glass styling (backdrop-filter blur + saturate)
    app.js             WebSocket client, DOM updates, face-anchored panel positioning
  README.md            (this file)
```

---

## Run on the laptop

```bash
# 1. Install deps (one time)
pip install -r requirements.txt

# 2. Pick which camera to use (DroidCam usually shows up as index 1 or 2)
#    Set this in .env at the repo root:
#    CAMERA_INDEX=1
# 3. Start the server
python web_ui/server.py
```

Open `http://localhost:8000` in any modern browser (Chrome, Edge, Safari,
Firefox). You should see the live camera with the glass panels appearing
around your face once it is detected.

The server binds to `0.0.0.0` by default, so if the laptop and phone are on
the **same Wi-Fi**, you can also open it from the phone:

1. On the laptop, find your local IP:
   - Windows: `ipconfig` (look for "IPv4 Address" under your Wi-Fi adapter)
   - macOS / Linux: `ifconfig | grep "inet "`
2. On the phone browser open `http://<laptop-ip>:8000`
   (e.g. `http://192.168.1.42:8000`)

The Windows firewall may pop up on first run asking to allow Python through -
choose "Private networks" so phones on the LAN can reach it.

---

## Use DroidCam to make the phone the camera

DroidCam exposes the phone's camera as a regular Windows webcam. The web UI
does not know or care - it just opens the camera at `CAMERA_INDEX`.

1. Install **DroidCam** on the phone (Play Store / App Store).
2. Install the **DroidCam Client** on the laptop:
   <https://www.dev47apps.com/>
3. On the laptop, run DroidCam Client and connect to the phone over Wi-Fi or USB.
4. The phone camera now shows up as a webcam. To find which index it took:
   ```bash
   python -c "import cv2; [print(i, cv2.VideoCapture(i).isOpened()) for i in range(5)]"
   ```
   Each open camera prints `True`. DroidCam usually grabs index `1` or `2`.
5. Set that in `.env`:
   ```
   CAMERA_INDEX=1
   ```
6. Start the web server: `python web_ui/server.py`
7. The web UI now uses the phone as its camera. You can view the UI in any
   other browser - laptop, second phone, tablet, anything on the same Wi-Fi.

---

## Environment variables

Read from `.env` at the repo root via `python-dotenv`:

| Variable        | Default         | Effect                                |
|-----------------|-----------------|---------------------------------------|
| `CAMERA_INDEX`  | `0`             | Which webcam to capture from          |
| `WEB_UI_HOST`   | `0.0.0.0`       | Address to bind the server to         |
| `WEB_UI_PORT`   | `8000`          | Port to bind the server to            |

---

## What is shown

- **Status pill** (top centre): connection state.
- **Calibration banner**: appears for ~12 s on first detection, while the
  baseline is being captured. The progress bar fills as it completes.
- **Heart Rate / HRV / Stress panel**: anchored to the right of the face.
  Updates live.
- **Expressions panel**: anchored to the left of the face. Lists the 10
  Action Units the system tracks; each row has a progress bar that lights up
  green when the AU intensity exceeds 50%.
- **Heart-rate graph**: full-width strip at the bottom. Last ~7-10 seconds of
  HR history plotted as a line.

When no face is detected, the side panels hide automatically. They reappear
and re-anchor when the face comes back.

---

## How the glass actually works

The Python server only does pipeline work. It pushes:

1. Binary JPEG frames (~25 fps, 1280x720, JPEG quality 70).
2. Text JSON metrics (every frame): heart rate, stress score, AU intensities,
   normalised face bbox.

The browser places the JPEG as a full-screen `<img>` underneath the page, then
draws each panel as a regular `<div>` styled with:

```css
backdrop-filter: blur(22px) saturate(160%);
background: rgba(28, 28, 36, 0.32);
border: 1px solid rgba(255, 255, 255, 0.18);
border-radius: 18px;
```

The browser's GPU does the blur natively. That is why this version finally
looks like Vision Pro instead of looking like a smudged OpenCV rectangle.

The JS positions the panels using the `face_bbox` from the metrics JSON and
the inverse of the `object-fit: cover` mapping, so panels follow your face as
it moves across the screen. Panel position changes are CSS-transitioned with
a 180 ms cubic-bezier so the movement feels glided rather than jumpy.

---

## Limits / caveats

- The OpenCV `unified_ar_system.py` includes poker card detection, gesture
  control (drag / pinch / showdowns), bluff predictions, and the Claude
  integration. The web UI in this folder is **face metrics only**. The poker
  side and the adaptive-learning bandit are not surfaced here yet.
- The camera can only be opened by one process at a time on Windows. If you
  start the OpenCV app first and then `web_ui/server.py`, the second one will
  fail to open the camera. Run one or the other.
- DroidCam's free version downsamples to 480p. The paid version supports HD;
  rPPG works much better at HD because the forehead patch has more pixels to
  average.

---

## Stop

Ctrl+C in the terminal running `web_ui/server.py`. Refresh the browser tab if
it stays on a stale frame.
