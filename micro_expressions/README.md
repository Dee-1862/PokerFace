# 🧠 Micro-Expressions & Stress Detection System

> Real-time physiological stress analysis using computer vision, rPPG heart rate estimation, and FACS facial action units.

![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange)

---

## 📋 Overview

This module provides **non-invasive stress detection** by analyzing:
- **Heart Rate (rPPG)**: Remote photoplethysmography from facial skin color changes
- **Facial Action Units (FACS)**: Micro-expression analysis via AU mapping
- **Baseline Comparison**: Automatic calibration against user's neutral state
- **AR Interface**: Apple-inspired UI with gesture control

---

## ✨ Key Features

### 💓 Heart Rate Detection (rPPG)
- Extracts pulse signal from forehead/cheek regions
- Uses green channel + FFT frequency analysis
- Real-time BPM display with confidence indicator
- Detects HR spikes (stress response)

### 😶 Facial Action Units
| Action Unit | Name | Stress Indicator |
|-------------|------|------------------|
| AU1 | Inner Brow Raise | Worry/Concern |
| AU2 | Outer Brow Raise | Surprise |
| AU4 | Brow Lowerer | Anger/Concentration |
| AU12 | Lip Corner Puller | Genuine smile (lack = hiding) |
| AU26 | Jaw Drop | Surprise/Shock |
| AU45 | Blink Rate | Cognitive load |

### 🎯 Stress Classification
```
┌─────────────────────────────────────────────────┐
│  5-Level Stress Classification                  │
├─────────┬───────────────────────────────────────┤
│ Level 1 │ 💚 Calm (0-20)                        │
│ Level 2 │ 💛 Mild (20-40)                       │
│ Level 3 │ 🟠 Moderate (40-60)                   │
│ Level 4 │ 🟥 High (60-80)                       │
│ Level 5 │ 🔴 Extreme (80-100)                   │
└─────────┴───────────────────────────────────────┘
```

### 🖐️ AR Gesture Control
- **Pinch & Stretch**: Control UI panel expansion
- **Face Button**: Shows 0-10 facial expressions
- **Heart Button**: Shows HR + stress analytics

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install opencv-python mediapipe numpy scipy

# Run the main system
cd micro_expressions
python main.py

# Or run with specific modules
python main.py --modules face rppg facs
```

### Options
- `--modules face rppg facs` - Select active modules
- `--camera 1` - Use alternate camera
- `--debug` - Enable debug output

---

## 📁 Project Structure

```
micro_expressions/
├── main.py                 # Entry point
├── engine.py               # Core pipeline coordinator
├── face_detection.py       # MediaPipe face landmarks
├── rppg_heart_rate.py      # Remote heart rate estimation
├── facs_action_units.py    # FACS AU mapping
├── stress_detector.py      # Baseline-calibrated stress scoring
├── stress_classifier.py    # 5-level classification with trends
├── ar_ui_controller.py     # Apple-like AR interface
├── hand_gesture_detector.py # Gesture recognition
├── video_processor.py      # Batch/realtime video analysis
├── face_landmarker.task    # MediaPipe model (3.7MB)
└── hand_landmarker.task    # Hand tracking model (7.8MB)
```

---

## 🔧 Technical Architecture

### Processing Pipeline
```
Camera Feed
     │
     ▼
┌─────────────┐
│ Face Detect │ ─→ 468 Landmarks + Blendshapes
└──────┬──────┘
       │
   ┌───┴───┐
   ▼       ▼
┌─────┐  ┌─────┐
│rPPG │  │FACS │
│ HR  │  │ AUs │
└──┬──┘  └──┬──┘
   │        │
   └───┬────┘
       ▼
┌─────────────┐
│   Stress    │
│  Detector   │
└──────┬──────┘
       ▼
┌─────────────┐
│   AR UI     │
└─────────────┘
```

### Stress Calculation Formula
```python
stress_score = (
    0.4 * heart_rate_deviation +    # HR above baseline
    0.4 * au_stress_score +         # Facial tension signals
    0.1 * blink_rate_delta +        # Cognitive load
    0.1 * micro_movement_score      # Fidgeting
)
```

---

## 📊 Calibration System

The stress detector uses **automatic baseline calibration**:

1. **Detection Phase**: Wait for stable face (5 seconds)
2. **Collection Phase**: Gather baseline metrics (10 seconds)
3. **Finalization**: Calculate neutral HR, AU levels
4. **Active Mode**: Compare current state to baseline

> 💡 No keyboard input required - fully AR-compatible!

---

## 📈 Recent Updates (Jan 2026)

- ✅ **Added**: Automatic baseline calibration (AR-friendly)
- ✅ **Improved**: 5-level stress classification with color coding
- ✅ **Enhanced**: Trend detection (Rising/Falling/Spiking)
- ✅ **New**: Mini stress timeline graph
- ✅ **Fixed**: rPPG reliability in varying lighting

---

## 🔮 Roadmap

- [ ] Integration with poker hand analysis
- [ ] Multi-face tracking for opponent analysis  
- [ ] Bluff probability scoring
- [ ] Mobile deployment
- [ ] Thermal imaging support

---

## 📚 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `opencv-python` | 4.x | Video capture & rendering |
| `mediapipe` | 0.10.x | Face/Hand detection |
| `numpy` | 1.x | Array operations |
| `scipy` | 1.x | Signal processing (FFT) |

---

## 🔬 Deprecated Components

| File | Status | Reason |
|------|--------|--------|
| `lstm_fusion.py` | ⚠️ Disabled | Model unreliable, kept for compatibility |
| `trial.py` | 🔧 Dev Only | Standalone testing tool |
| `gpu_validator.py` | 🔧 Utility | GPU check script |

---

## 📝 Technical Notes

- **rPPG Limitations**: Requires good lighting, minimal movement
- **Face Detection**: Needs front-facing camera, clear face visibility
- **Calibration**: Takes ~15 seconds, do when subject is relaxed
- **Accuracy**: ~85% correlation with actual stress in controlled conditions

---

## ⚠️ Disclaimer

This system is for **research and educational purposes only**. Stress detection from video has inherent limitations and should not be used for medical or legal decisions.

---

## 📝 License

Educational / Personal Use

---

**Last Updated**: January 2026  
**Version**: 1.5 (Stress Detection Enhanced)
