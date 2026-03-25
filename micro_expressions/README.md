# micro_expressions - Physiological Signal Extraction and AR UI

Reads a person's heart rate, stress level, and facial muscle activity from a standard webcam - no wearable device required. Combines these signals into a calibrated stress score and displays them as an AR overlay.

---

## How the Pipeline Runs

Each camera frame flows through four modules in sequence. Each module reads from and writes to a shared `engine.shared_state` dictionary so any module can see any other module's output.

```mermaid
flowchart TD
    CAM([Camera Frame])

    subgraph M1 ["Step 1 - Find the face"]
        FACE["face_detection.py<br/>MediaPipe FaceLandmarker<br/>Outputs 468 landmarks in 3D<br/>and a face_detected flag"]
    end

    subgraph M2 ["Step 2 - Measure heart rate"]
        RPPG["rppg_heart_rate.py<br/>Samples a 40x40 patch of forehead skin<br/>Tracks green channel brightness over time<br/>FFT finds the pulse frequency -> BPM"]
    end

    subgraph M3 ["Step 3 - Read facial muscles"]
        FACS["facs_action_units.py<br/>Measures 6 Action Units from landmark geometry<br/>AU1 AU2 AU4 AU12 AU26 AU45"]
    end

    subgraph M4 ["Step 4 - Score stress"]
        STRESS["stress_detector.py<br/>Compares current readings to personal baseline<br/>Outputs a 0-100 stress score<br/>and a Calm / Mild / Moderate / High / Extreme label"]
    end

    subgraph UI ["Display"]
        AR["ar_ui_controller.py<br/>HR gauge, stress bar,<br/>AU display, HR history graph"]
    end

    CAM --> M1 --> M2 --> M3 --> M4 --> UI
```

---

## How Heart Rate is Measured Without a Sensor

The skin over a blood vessel changes colour very slightly with each heartbeat - too subtle for the eye but detectable in video. This technique is called **remote photoplethysmography (rPPG)**.

```mermaid
flowchart LR
    A([Forehead region<br/>40x40 pixels]) --> B["Track mean green<br/>channel value<br/>each frame"]
    B --> C["300-frame rolling window<br/>~10 seconds of signal"]
    C --> D["Bandpass filter<br/>0.5 to 3.5 Hz<br/>= 30 to 210 BPM"]
    D --> E["FFT - find<br/>dominant frequency"]
    E --> F([Frequency x 60<br/>= BPM estimate])
```

Accuracy is best under natural or LED lighting. Fluorescent tubes flicker at mains frequency and can create artefacts in the signal.

---

## The Six Action Units

Action Units (AUs) come from the Facial Action Coding System - a standardised way to describe facial expressions in terms of individual muscle movements rather than labels like "angry" or "surprised".

```mermaid
flowchart LR
    subgraph Upper ["Upper Face"]
        AU1["AU1<br/>Inner brow raiser<br/>Worry, sadness"]
        AU2["AU2<br/>Outer brow raiser<br/>Surprise"]
        AU4["AU4<br/>Brow lowerer<br/>Concentration, anger"]
        AU45["AU45<br/>Blink rate<br/>Nervousness, fatigue"]
    end
    subgraph Lower ["Lower Face"]
        AU12["AU12<br/>Lip corner puller<br/>Smile"]
        AU26["AU26<br/>Jaw drop<br/>Surprise, relaxation"]
    end
```

Each AU is computed geometrically from landmark distances, so no pre-trained AU classifier is needed.

---

## How the Stress Score is Calculated

The score is a weighted sum of deviations from the player's **personal baseline** - not from population averages. A person with a naturally fast blink rate is not flagged as stressed just because they blink often.

```mermaid
flowchart TD
    B["baseline_extractor.py<br/>Stores mean and std dev<br/>for each signal"]

    subgraph Deviations ["Compute z-score deviations"]
        D1["HR delta<br/>(HR_now - HR_mean) / HR_std"]
        D2["AU4 delta<br/>brow furrow vs baseline"]
        D3["AU1+AU2 delta<br/>brow raise vs baseline"]
        D4["AU45 delta<br/>blink rate vs baseline"]
    end

    subgraph Weights ["Weighted combination"]
        W["Stress Score =<br/>  0.35 x HR delta<br/>+ 0.25 x AU4 delta<br/>+ 0.20 x AU1+AU2 delta<br/>+ 0.20 x AU45 delta<br/><br/>Clamped to 0 to 100"]
    end

    subgraph Label ["stress_classifier.py"]
        L["0-20  -> Calm<br/>20-40 -> Mild<br/>40-60 -> Moderate<br/>60-80 -> High<br/>80-100 -> Extreme"]
    end

    B --> Deviations --> Weights --> Label
```

### Baseline Calibration

When the system first sees a new person it runs a 30-second calibration window (a progress bar appears on screen). During this window it records the resting values for HR, stress score, and all six AUs. After calibration the stress score becomes meaningful for that individual.

Press `b` to force a new calibration at any time.

---

## What Is Displayed on Screen

```mermaid
flowchart LR
    subgraph Panels ["AR Panels - all draggable"]
        P1["HR Gauge<br/>Circular arc and BPM number<br/>with trend arrow"]
        P2["Stress Bar<br/>Colour shifts green to yellow to red<br/>with level label"]
        P3["Action Unit Display<br/>6 horizontal bars<br/>labelled AU1-AU45"]
        P4["HR History Graph<br/>60-second rolling chart"]
        P5["Expressions Panel<br/>AU intensities as numbers"]
    end
```

Each panel has a small circle handle at its top centre. Point your index finger at it and hold 0.5 seconds to grab and drag it. Positions are saved automatically.

---

## Module Reference

| File | What it does |
|------|-------------|
| `engine.py` | Coordinates the pipeline. Modules register here and run in order each frame. |
| `face_detection.py` | MediaPipe FaceLandmarker - finds face, outputs 468 3D landmarks. |
| `rppg_heart_rate.py` | rPPG heart rate estimation from forehead skin colour. |
| `facs_action_units.py` | Geometric AU detection from landmark distances. |
| `stress_detector.py` | Baseline calibration, z-score deviation, 0-100 stress score. |
| `stress_classifier.py` | Converts numeric score to Calm/Mild/Moderate/High/Extreme. Detects spikes. |
| `ar_ui_controller.py` | Draws all AR panels on the live video frame. Handles drag state. |
| `hand_gesture_detector.py` | Gesture detector used in standalone mode. |
| `video_processor.py` | Run the same pipeline on a pre-recorded video file, export CSV. |
| `heart_rate_validator.py` | Compare rPPG readings to a manually entered reference value. |
| `stress_analytics.py` | Offline charts (timeline, distribution) from exported CSV data. |
| `main.py` | Standalone entry point - runs face analysis without poker integration. |

---

## Standalone Usage

```bash
python micro_expressions/main.py
```

| Key | Action |
|-----|--------|
| `q` | Quit |
| `b` | Force new baseline calibration |
| `v` | Start heart rate validation recording |
| `s` | Stop recording and print correlation result |
| Any number | Add a manual BPM reference reading for validation |

---

## Technical Notes

- rPPG requires a reasonably still subject. Heavy movement blurs the skin-colour signal.
- The stress score is **relative** - it only becomes meaningful after the calibration window completes.
- AU detection is approximate (geometry-based, not model-based). Clear expressions are detected reliably. Subtle micro-expressions are less consistent.
- `face_landmarker.task` and `hand_landmarker.task` must be present in this directory for standalone use.
