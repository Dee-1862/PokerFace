# adaptive_learning - Opponent Profiling and Bluff Detection

This module watches an opponent's face over time and builds a personal model of how they behave when bluffing versus when they have a strong hand. The more showdowns you record with a thumbs gesture, the sharper the prediction becomes.

All data is stored locally in a SQLite database. Nothing leaves the machine.

---

## The Learning Cycle

Here is the full journey from "new opponent sits down" to "confident bluff prediction":

```mermaid
flowchart TD
    A([Opponent's face<br/>appears on camera])
    B["face_embedder.py<br/>Convert face geometry to<br/>a 128-number identity vector"]
    C{Match in<br/>database?}
    D["Create new player profile<br/>in SQLite"]
    E["baseline_extractor.py<br/>30-second window<br/>Record resting HR, stress,<br/>facial muscle readings"]
    F["Load existing profile<br/>and all learned state"]
    G["baseline_extractor.py<br/>Compute deviation<br/>How different is this frame<br/>from their resting state?"]
    H["opponent_model.py<br/>Context bucket lookup<br/>personality + stress level + HR level"]
    I["Thompson Sampling<br/>Draw from probability distribution<br/>Output: p_bluff (0 to 1)"]
    J([Display: BLUFFING<br/>or STRONG HAND])
    K{Showdown<br/>gesture?}
    L["Extract peak signal anomaly<br/>from entire hand buffer<br/>90th-percentile deviation"]
    M["Update bandit alpha/beta<br/>Update personality model<br/>Save to SQLite"]

    A --> B --> C
    C -->|No match| D --> E --> G
    C -->|Similarity >= 0.85| F --> G
    G --> H --> I --> J
    J --> K
    K -->|Thumbs-up / down| L --> M
    M -->|Next hand| G
```

---

## How Player Recognition Works

The system never stores a photo. Instead it measures 64 distances between facial landmark pairs - things like the gap between your eyes relative to your face width, or how your nose width compares to your jaw. These 64 distances form a 128-number vector (each pair gives a horizontal and vertical component).

When a face appears, the system computes cosine similarity against every stored vector. A score of 0.85 or above is considered the same person.

```mermaid
flowchart LR
    A([MediaPipe<br/>468 face landmarks]) --> B["Select 64 landmark pairs<br/>eye spacing, nose, jaw, etc."]
    B --> C["Normalise by face width<br/>Scale-invariant distances"]
    C --> D["128-dimensional vector"]
    D --> E{Cosine similarity<br/>>= 0.85?}
    E -->|Yes| F([Returning player<br/>Load profile])
    E -->|No| G([New player<br/>Create profile])
```

---

## What the Baseline Calibration Does

The stress score and heart rate mean nothing in isolation - a naturally high heart rate is not a tell. The system spends the first 30 seconds with a new player recording their resting state readings. Everything after that is measured as a deviation from that personal baseline.

```mermaid
sequenceDiagram
    participant Sys as System
    participant DB as Database

    Note over Sys: New player - calibration starts
    loop 30 seconds (~900 frames)
        Sys->>Sys: Record HR, stress score, AU1..AU6
    end
    Sys->>Sys: Compute mean and std dev for each signal
    Sys->>DB: Save baseline
    Note over Sys,DB: Future readings become z-scores:<br/>(observed - mean) / std_dev
```

After calibration, a reading of `hr_delta = +2.1` means the opponent's heart rate is 2.1 standard deviations above their own normal - a meaningful signal regardless of their absolute BPM.

---

## How the Bluff Prediction Works

The prediction engine is a **Thompson Sampling contextual bandit** - a lightweight reinforcement learning approach that maintains uncertainty and explores rather than committing too early.

### Context Buckets

Each prediction is made in a specific context defined by three factors:

| Factor | Possible values |
|--------|----------------|
| Personality type | aggressive, tight, unpredictable, standard |
| Stress state | calm, mild, moderate, high, extreme |
| Heart rate state | normal, elevated, high |

These combine into a bucket name like `aggressive_high_elevated`. Each bucket has its own independent model.

### The Beta Distribution

For each bucket the system maintains two numbers: `alpha` (number of times this bucket was observed and the player was bluffing) and `beta` (times they were not bluffing). The bluff probability is sampled from a Beta(alpha, beta) distribution.

```mermaid
flowchart TD
    A([Showdown recorded:<br/>thumb-up = strong hand<br/>thumb-down = bluff]) --> B["Identify active bucket<br/>e.g. aggressive_high_elevated"]
    B --> C{Was bluffing?}
    C -->|Yes - thumbs-down| D["alpha = alpha + 1"]
    C -->|No - thumbs-up| E["beta = beta + 1"]
    D & E --> F["Next prediction:<br/>sample from Beta(alpha, beta)<br/>Higher alpha = more likely bluff"]
```

Before any showdowns the system falls back to a heuristic estimate based purely on how much the current signals deviate from baseline. The panel labels this mode "SIGNAL" versus "TRAINED Nobs".

---

## Signal Buffer and Peak Extraction

Rather than capturing physiology only at the moment you make a thumbs gesture, the system buffers every frame's deviation readings from when you started a hand until you show the gesture. At showdown it extracts the **90th-percentile peak** - the most extreme anomaly that appeared at any point during the hand.

```mermaid
flowchart LR
    A([Hand starts<br/>start_hand called]) --> B["Buffer fills:<br/>hr_delta, stress_delta,<br/>au_delta per frame"]
    B --> C([Thumbs gesture<br/>showdown recorded])
    C --> D["Sort absolute values<br/>Find 90th percentile threshold"]
    D --> E["Find the single frame<br/>with the biggest deviation<br/>at or above that threshold"]
    E --> F["Peak HR delta<br/>Peak stress delta<br/>Peak AU delta<br/>Where in hand it occurred"]
    F --> G([Saved to<br/>showdowns table])
```

This means if a player flinched for two frames in the middle of the hand, that flinch is captured - even if they composed themselves by the time the cards were shown.

---

## Database Schema

```mermaid
erDiagram
    players {
        text player_id PK
        blob embedding
        real first_seen
        real last_seen
        int session_count
        int showdown_count
    }
    baselines {
        text player_id PK
        text baseline_data
        real created_at
    }
    bandit_state {
        text player_id PK
        text alpha_data
        text beta_data
        real updated_at
    }
    personality_state {
        text player_id PK
        text state_data
        real updated_at
    }
    showdowns {
        int id PK
        text player_id
        text context_bucket
        text prediction
        text actual_result
        int was_correct
        int frames_sampled
        real hr_variance
        real stress_variance
        real peak_frame_pct
        real timestamp
    }

    players ||--o| baselines : "has"
    players ||--o| bandit_state : "has"
    players ||--o| personality_state : "has"
    players ||--o{ showdowns : "recorded"
```

---

## Module Reference

| File | Responsibility |
|------|---------------|
| `adaptive_learning_system.py` | The only file the rest of the codebase calls. Orchestrates everything below. |
| `face_embedder.py` | Face landmark -> 128-dim identity vector |
| `baseline_extractor.py` | 30-second calibration, z-score deviation calculation |
| `opponent_model.py` | Thompson Sampling bandit - predict and update |
| `personality_model.py` | Per-player traits: bluff rate, stress correlation, consistency |
| `prior_generator.py` | First-guess parameters before any showdowns exist |
| `profile_store.py` | All SQLite reads and writes |
| `panel_positions.py` | Save/load draggable panel screen positions |
