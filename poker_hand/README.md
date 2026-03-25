# poker_hand - Card Detection, Equity Calculation, and AR UI

Points a webcam at playing cards and identifies them using a custom-trained AI model. Once your hole cards are saved, it calculates your probability of winning against a random opponent and displays the result as an augmented reality overlay.

---

## How Cards Are Detected

A YOLO object detection model (`poker_v1.pt`) was trained on 24,000 images of all 52 standard playing cards. It runs on every other camera frame for performance.

Raw detections are noisy. A card might flicker in and out for a frame or two. A stability buffer filters this out:

```mermaid
flowchart TD
    CAM([Camera Frame])
    YOLO["poker_engine_unified.py<br/>YOLO model - conf = 0.45<br/>Detect all 52 card classes"]
    BUF{Seen for<br/>4+ frames?}
    IGNORE["Ignore<br/>Too brief to be real"]
    SHOW["Show detection box<br/>Yellow border<br/>Label and confidence"]
    LOCK{Seen for<br/>12+ frames?}
    STABLE["Stable card<br/>Kept even if briefly hidden<br/>up to 60 frames / ~2 seconds"]

    CAM --> YOLO --> BUF
    BUF -->|No| IGNORE
    BUF -->|Yes| SHOW --> LOCK
    LOCK -->|No| SHOW
    LOCK -->|Yes| STABLE
```

**Card border colours on screen:**

| Colour | Meaning |
|--------|---------|
| Yellow | Detected, not yet saved |
| Green | Saved as your hole cards |
| Orange | Locked as a board card |

---

## How Equity Is Calculated

Once you save your hole cards, the system continuously calculates your probability of winning against a random opponent hand. The method depends on how many board cards are known:

```mermaid
flowchart TD
    HAND([Your hole cards saved])
    BOARD{How many<br/>board cards?}
    PRE["Preflop<br/>Lookup table<br/>169 canonical hand types<br/>Result: instant"]
    FLOP["Flop or Turn<br/>Exhaustive enumeration<br/>Sample 200 random opponent pairs<br/>from remaining deck<br/>Result: ~50ms"]
    RIVER["River<br/>Direct evaluation<br/>All possible opponent pairs<br/>using Treys evaluator<br/>Result: ~10ms"]
    OUT([Win %<br/>Top hand combinations<br/>for you and opponent])

    HAND --> BOARD
    BOARD -->|0 cards| PRE --> OUT
    BOARD -->|3-4 cards| FLOP --> OUT
    BOARD -->|5 cards| RIVER --> OUT
```

The output includes win equity as a percentage plus the top 3 winning hand types for both you and a random opponent, shown in scrollable panels on the right side of the screen.

---

## How to Use It (Step by Step)

```mermaid
sequenceDiagram
    actor You
    participant Cam as Camera + System

    You->>Cam: Hold your hole cards up to camera
    Note over Cam: Cards detected, yellow boxes appear
    Note over Cam: After 12 frames they lock in (stable)

    You->>Cam: Left pinch gesture, hold until locked
    Note over Cam: Hole cards saved, boxes turn green<br/>Preflop equity appears immediately

    You->>Cam: Reveal flop (3 board cards)
    You->>Cam: Right pinch, hold 5 seconds
    Note over Cam: Board cards locked, boxes turn orange<br/>Flop equity recalculated

    You->>Cam: Right hand two-finger scroll
    Note over Cam: Scroll through winning hand combinations

    You->>Cam: Press C to clear board and start next hand
```

---

## What Appears on Screen

```mermaid
flowchart LR
    subgraph Panels ["poker_ar_ui.py - AR Panels"]
        P1["Win % panel<br/>Large percentage<br/>Colour: green above 50%<br/>yellow near 50%<br/>red below 35%"]
        P2["Board panel<br/>Mini card thumbnails<br/>of locked board cards"]
        P3["My Hand panel<br/>Top-left corner<br/>Mini thumbnails of<br/>your hole cards"]
        P4["Winning Hands panels<br/>Scrollable list<br/>Top 3 hand types for<br/>you and opponent"]
    end
```

All panels are draggable. Point your index finger at the circle handle at the top of a panel and hold for 0.5 seconds to grab it.

---

## Gestures in Cards Mode

| Gesture | Action |
|---------|--------|
| Left pinch, hold until locked | Save visible cards as your hole cards |
| Right pinch, hold 5 seconds | Lock visible cards as board cards |
| Two-finger scroll (right hand, vertical) | Scroll winning hand combination panels |
| `c` key | Clear board and saved hand |

---

## Module Reference

| File | What it does |
|------|-------------|
| `poker_main.py` | Standalone entry point. Also contains the equity calculation functions (`calculate_equity_fast`, `exhaustive_enumeration`, `evaluate_river`) imported by the unified system. |
| `poker_ar_ui.py` | Draws all poker AR panels on the live frame. Handles scroll interaction. |
| `poker_engine_unified.py` | YOLO model wrapper. Manages card stability buffering and finalization. |
| `hand_gesture_detector.py` | Full gesture detector. Handles pinch, pinch-lock, two-finger scroll, thumbs, fist, and pointing. |
| `simple_card_detector.py` | Minimal standalone card detector for testing the YOLO model. |
| `preflop_equity.csv` | Win percentage lookup table for all 169 canonical starting hands. |
| `df_preflop_hand_distrib.csv` | Extended preflop distribution data. |
| `poker_v1.pt` | Custom-trained YOLOv8 model. 52 card classes. |

---

## Standalone Usage

```bash
python poker_hand/poker_main.py
```

| Key | Action |
|-----|--------|
| `q` | Quit |
| `c` | Clear board and saved hand |

---

## Technical Notes

- Cards work best when held flat, well-lit, and not overlapping each other.
- Glare on glossy card surfaces reduces detection confidence. Matte-finish cards perform better.
- The model handles standard poker card designs. Non-standard decks may not detect reliably.
- Equity calculation assumes heads-up (2 players). Multi-way pot equity is not computed.
- `face_landmarker.task` is present in this folder so `poker_main.py` can run standalone without needing the root directory on the path.
