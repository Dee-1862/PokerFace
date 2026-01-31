# AR Poker + Stress Detection Integration Plan

## Executive Summary

This plan outlines how to integrate your **poker hand analysis** (`poker_hand/`) with **micro-expressions/stress detection** (`micro_expressions/`) into a wearable AR system for real-time bluff detection during poker games.

---

## Current System Analysis

### Poker Hand Module (`poker_hand/`)
| Component | Purpose | Key Files |
|-----------|---------|-----------|
| Card Detection | YOLO-based playing card recognition | `poker_v1.pt`, `poker_v2.pt` |
| Hand Analysis | Treys library for equity calculation | `poker_main.py` |
| Gesture Control | Pinch-to-lock, scroll gestures | `hand_gesture_detector.py` |
| AR UI | Win probability, winning hands display | `poker_ar_ui.py` |

### Micro-Expressions Module (`micro_expressions/`)
| Component | Purpose | Key Files |
|-----------|---------|-----------|
| Face Detection | MediaPipe landmarks & blendshapes | `face_detection.py` |
| Heart Rate (rPPG) | Remote photoplethysmography | `rppg_heart_rate.py` |
| FACS Analysis | Facial Action Units mapping | `facs_action_units.py` |
| Stress Detection | Baseline-calibrated stress scoring | `stress_detector.py` |
| AR UI | Expression panels, HR graphs | `ar_ui_controller.py` |

---

## Hardware Options

### Option A: DIY Raspberry Pi AR Glasses (Budget: $150-300)

**Pros:**
- Full customization control
- Open-source, hackable
- Low cost

**Cons:**
- Limited processing power (Raspberry Pi Zero 2W)
- Requires offloading heavy ML to a phone/laptop
- 3D printing and assembly required

**Components:**
```
┌─────────────────────────────────────────┐
│  Raspberry Pi Zero 2W          ~$15    │
│  Pi Camera Module (5MP)        ~$25    │
│  Micro OLED Display (0.96")    ~$15    │
│  LiPo Battery (3.7V 2000mAh)   ~$10    │
│  3D Printed Frame              ~$20    │
│  Reflective Lens/Prism         ~$10    │
│  Misc (Wires, PCB, etc.)       ~$15    │
│─────────────────────────────────────────│
│  TOTAL                         ~$110   │
└─────────────────────────────────────────┘
```

> ⚠️ **Warning:** The Pi Zero 2W cannot run YOLO + MediaPipe locally. You'll need a **companion device** (phone/laptop) for ML inference.

---

### Option B: Commercial AR Glasses (Budget: $700-1000)

| Model | Price | Camera | Processing | Best For |
|-------|-------|--------|------------|----------|
| **Xreal Air 2 Ultra** | $699 | 2x cameras | Tethered to phone/PC | Developer-focused, 6DOF |
| **TCL RayNeo X2** | ~$700 | 16MP camera | Snapdragon XR2 | Standalone, real-time translation |
| **Vuzix Blade 2** | $999 | 8MP camera | Android-based | Enterprise, streaming |

**Recommended: Xreal Air 2 Ultra**
- Best developer SDK support
- Can tether to Android phone for processing
- Camera access via SDK

---

### Option C: Hybrid Smartphone Approach (Recommended for MVP)

> 💡 **Tip:** Best for rapid prototyping and proving the concept before investing in hardware.

**Setup:**
```
┌─────────────────────────────────────────────────────────┐
│  SMARTPHONE (Processing Unit)                           │
│  ├─ Camera: Captures table + opponent faces             │
│  ├─ ML: Runs YOLO + MediaPipe + Stress Detection        │
│  └─ Display: Shows AR overlay on screen                 │
│                                                         │
│  Mount Options:                                         │
│  1. Chest-mounted phone holder (hands-free)             │
│  2. Tripod beside table                                 │
│  3. Clip-on phone mount for glasses                     │
└─────────────────────────────────────────────────────────┘
```

**Why this is smarter:**
1. Your phone already has a powerful GPU
2. No custom hardware development
3. Can use existing codebase with minimal changes
4. Easily pivot to real AR glasses later

---

## Software Architecture

### Unified Pipeline

```
Camera Feed
     │
     ▼
┌────────────────┐
│  Frame Router  │
└───────┬────────┘
        │
   ┌────┴────┐
   │         │
   ▼         ▼
┌──────┐  ┌──────┐
│ YOLO │  │ Face │
│ Cards│  │ Mesh │
└───┬──┘  └───┬──┘
    │         │
    ▼         ▼
┌──────┐  ┌───────┐
│Poker │  │Stress │
│Engine│  │Detect │
└───┬──┘  └───┬───┘
    │         │
    └────┬────┘
         ▼
   ┌───────────┐
   │  Decision │
   │   Fusion  │
   └─────┬─────┘
         ▼
   ┌───────────┐
   │ Unified   │
   │  AR UI    │
   └───────────┘
```

### Proposed File Structure

```
Stone/
├── poker_ar_unified/           # NEW unified module
│   ├── main.py                 # Entry point
│   ├── unified_engine.py       # Coordinates both systems
│   ├── card_detector.py        # From poker_hand
│   ├── face_analyzer.py        # From micro_expressions
│   ├── bluff_detector.py       # NEW: Combines stress + equity
│   ├── ar_ui_unified.py        # NEW: Single unified UI
│   └── hardware/
│       ├── camera_source.py    # Abstract camera (USB/Pi/Phone)
│       └── display_output.py   # Abstract display (Screen/AR)
├── poker_hand/                 # Original (reference)
└── micro_expressions/          # Original (reference)
```

---

## Bluff Detection Logic

### Decision Matrix

| Your Equity | Opponent Stress | Opponent HR Delta | Suggested Action |
|-------------|-----------------|-------------------|------------------|
| High (>70%) | Low (<30) | Normal | **BET/RAISE** - You're likely winning |
| High (>70%) | High (>70) | Elevated (+15 BPM) | **SLOW PLAY** - They might fold |
| Low (<30%) | Low (<30) | Normal | **FOLD** - They likely have it |
| Low (<30%) | High (>70) | Elevated | **CALL/BLUFF CATCH** - They're nervous |
| Medium | Spiking | Sudden jump | **ALERT** - Possible bluff detected |

### Confidence Scoring

```python
def calculate_bluff_probability(stress_score, hr_delta, au_signals):
    """
    Returns 0-100 probability that opponent is bluffing.
    
    Key indicators:
    - AU12 (Lip Corner Push) suppression = hiding smile
    - AU4 (Brow Lowerer) + AU1 (Inner Brow Raise) = worry
    - HR spike 10-20 BPM above baseline = stress response
    - Rapid AU45 (Blink) = cognitive load
    """
    bluff_score = 0
    
    if stress_score > 60:
        bluff_score += 30
    if hr_delta > 15:
        bluff_score += 25
    if au_signals.get('AU4', 0) > 0.3 and au_signals.get('AU1', 0) > 0.2:
        bluff_score += 20
    if au_signals.get('AU45', 0) > 0.4:  # Excessive blinking
        bluff_score += 15
    if au_signals.get('AU12', 0) < 0.1:  # Suppressed smile
        bluff_score += 10
    
    return min(100, bluff_score)
```

---

## Implementation Phases

### Phase 1: Software Integration (Week 1-2)

- [ ] Create `poker_ar_unified/` directory
- [ ] Build `unified_engine.py` to run both pipelines in parallel
- [ ] Implement `bluff_detector.py` with decision matrix
- [ ] Design unified AR UI showing equity + stress + bluff probability
- [ ] Test on laptop with USB webcam

### Phase 2: Mobile Optimization (Week 3-4)

- [ ] Port to Android using Kivy or Flutter + Python bridge
- [ ] Optimize YOLO model (use YOLOv8n or export to TFLite)
- [ ] Reduce MediaPipe load (use `facemesh_lite`)
- [ ] Implement phone-as-camera streaming

### Phase 3: Hardware Prototype (Week 5-8)

**Option A Path (DIY):**
- [ ] Build Raspberry Pi camera unit
- [ ] Implement RTSP streaming to laptop/phone
- [ ] 3D print glasses frame
- [ ] Add OLED micro-display

**Option B Path (Commercial):**
- [ ] Purchase Xreal Air 2 Ultra
- [ ] Develop Android companion app
- [ ] Implement Xreal SDK camera access
- [ ] Build overlay rendering pipeline

### Phase 4: Field Testing (Week 9-12)

- [ ] Test at home poker games
- [ ] Calibrate stress baselines for different lighting
- [ ] Tune bluff detection thresholds
- [ ] Iterate on UI/UX feedback

---

## Improvement Ideas

### Software Enhancements

1. **Multi-Face Tracking**: Track stress for ALL opponents at the table
2. **Betting Pattern Memory**: Store opponent history for pattern recognition
3. **Voice Hints**: Subtle audio cues via bone conduction headphones
4. **Preflop Range Analysis**: Show opponent likely holdings based on position

### Hardware Innovations

1. **Bone Conduction Audio**: Whisper hints without visible earbuds
2. **Ring Controller**: Discreet finger ring for input instead of gestures
3. **Thermal Camera Add-on**: Detect micro-sweating on face
4. **Eye Tracking**: Know where opponent is looking (their cards vs yours)

### Blackjack Extension

1. **Card Counting Overlay**: Running count + true count display
2. **Deviations Chart**: Basic strategy exceptions based on count
3. **Heat Detection**: Warn when pit boss is watching
4. **Shuffle Tracking**: Identify card clumps through shuffles

---

## License

This project is for educational and personal use only. Using such systems in casinos or regulated poker rooms may be illegal.

---

**Last Updated**: January 2026
