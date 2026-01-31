# 🃏 Poker Hand AR Assistant

> Real-time poker hand analysis with augmented reality overlays, gesture control, and winning hand predictions.

![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/License-Educational-yellow)

---

## 📋 Overview

This system uses **YOLO-based computer vision** to detect playing cards in real-time and provides:
- **Equity Calculation**: Exact win probability using exhaustive enumeration (Flop/Turn) or direct evaluation (River)
- **Winning Hands Display**: Visual breakdown of possible winning combinations for you AND your opponent
- **Gesture Control**: Pinch-to-lock cards, two-finger scroll through hand lists
- **AR UI**: Premium dark-glass interface with animated elements

---

## ✨ Key Features

### 🎯 Card Detection
- **YOLO v8** trained on 24,000+ card images
- **52 classes** (all standard playing cards)
- **Stability filtering**: Cards must be visible for 10+ frames before locking

### 📊 Equity Engine
| Stage | Method | Accuracy | Speed |
|-------|--------|----------|-------|
| Preflop | Lookup Table (169 hands) | 100% | Instant |
| Flop/Turn | Exhaustive Enumeration | 100% | ~200ms |
| River | Direct Evaluation | 100% | Instant |

### 🖐️ Gesture Control
| Gesture | Action |
|---------|--------|
| **Left Pinch** | Register your hole cards |
| **Right Pinch (5s hold)** | Lock board cards |
| **Two-Finger Scroll** | Navigate winning hands list |
| **Press 'c'** | Clear board |

### 🎨 AR Interface
- **Dual Winning Hands Panels**: Your hands (top) vs Opponent's hands (bottom)
- **Best 5 Cards Display**: Logical poker hand combinations, not 7-card clutter
- **Dynamic Board Panel**: Expands left as cards are added
- **Color-Coded Equity**: Green (>60%), Yellow (30-60%), Red (<30%)

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install opencv-python ultralytics treys mediapipe numpy

# Run the system
cd poker_hand
python poker_main.py
```

### Controls
- `q` - Quit
- `c` - Clear locked board cards
- Left hand pinch - Save your hole cards
- Right hand pinch (hold 5s) - Add visible cards to board

---

## 📁 Project Structure

```
poker_hand/
├── poker_main.py           # Main entry point (unified pipeline)
├── poker_ar_ui.py          # AR UI controller with dual panels
├── hand_gesture_detector.py # MediaPipe gesture recognition
├── poker_v1.pt             # Trained YOLO model
├── poker_v2.pt             # Alternate model
├── preflop_equity.csv      # 169-hand lookup table
├── df_preflop_hand_distrib.csv # Extended preflop dataset
└── Training Dataset/       # YOLO training data (24K images)
```

---

## 🔧 Technical Details

### Card Detection Pipeline
```
Frame → YOLO Inference → Stability Filter (10 frames) → Finalization (20 frames)
                                    ↓
                           Hand/Board Separation
                                    ↓
                            Equity Calculation
                                    ↓
                             AR Overlay Render
```

### Key Algorithms
1. **Best-5 Card Selection**: From 7 available cards (2 hole + 5 board), finds the mathematically strongest 5-card combination
2. **Exhaustive Enumeration**: On Flop/Turn, evaluates ALL possible runouts and opponent hands
3. **Two-Finger Scroll Detection**: Index + Middle extended, Ring + Pinky closed

---

## � Recent Updates (Jan 2026)

- ✅ **Fixed**: Board overflow issue - cards now stay within panel
- ✅ **Added**: Best-5 card logic for user-friendly hand display
- ✅ **Improved**: Two-finger scroll gesture reliability
- ✅ **Enhanced**: Mini-card visuals with suit symbols and shadows
- ✅ **Optimized**: Resolution set to 720p for better screen fit

---

## 🔮 Roadmap

- [ ] Integration with micro-expressions stress detection
- [ ] Multi-opponent tracking
- [ ] Voice hints via bone conduction
- [ ] Mobile (Android) deployment
- [ ] AR glasses hardware support

---

## � Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `opencv-python` | 4.x | Video capture & rendering |
| `ultralytics` | 8.x | YOLO inference |
| `treys` | 0.1.x | Poker hand evaluation |
| `mediapipe` | 0.10.x | Hand gesture detection |
| `numpy` | 1.x | Array operations |

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. Using such systems in regulated poker rooms or casinos may be illegal. Use responsibly.

---

## 📝 License

Educational / Personal Use

---

**Last Updated**: January 2026  
**Version**: 2.0 (AR Enhanced)
