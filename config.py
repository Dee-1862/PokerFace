# config.py
# Tunable parameters. Edit here to adjust detection, calibration, and gesture behaviour.
# Personal/secret values (paths, keys) go in .env instead.

# --- YOLO card detection ---
YOLO_CONF   = 0.7    # minimum detection confidence
YOLO_IOU    = 0.15   # NMS intersection-over-union threshold
YOLO_IMGSZ  = 640    # inference image size (multiple of 32; 640 = native training res)

# --- Card tracking ---
CARD_STABILITY_THRESHOLD = 4    # consecutive frames before a card is shown
CARD_FINALIZE_THRESHOLD  = 12   # consecutive frames before a card is locked in
CARD_FADE_TIMEOUT        = 60   # frames of absence before a card fades (~2s at 30fps)
CARD_RESET_TIMEOUT       = 90   # frames of zero cards before the board resets

# --- Frame skipping ---
YOLO_SKIP       = 2    # run YOLO every Nth frame when cards are present
YOLO_SKIP_IDLE  = 12   # slow polling cadence when no cards anywhere (saves GPU)
MODULE_SKIP     = 3    # run face/rPPG/FACS/stress every Nth frame
HYSTERESIS_FRAMES = 8  # frames of agreement required before a context switch commits

# --- Card box visualisation ---
BOX_FADE_FRAMES   = 90    # frames a card box stays drawn after the last YOLO hit (~3.6s,
                          # outlasts the ~3s right-pinch lock so briefly-seen cards can
                          # still be captured into the green/locked state)
BOX_SMOOTH_ALPHA  = 0.4   # EMA weight on new YOLO box (lower = smoother, laggier)
BOX_ANCHOR_FRAMES = 15    # minimum frames a freshly-added box is protected from eviction

# Lock-pool recency: how recently a label must have been detected to be
# eligible for hand-save / board-lock. Tighter than the visual fade so
# stale ghosts don't get swept into the lock action.
LOCK_RECENCY_FRAMES = 10  # ~0.4s

# --- Adaptive learning ---
BASELINE_MAX_AGE_DAYS    = 7      # days before a saved baseline is discarded
CALIBRATION_FRAMES       = 90     # frames to collect during baseline calibration
HAND_BUFFER_MAX          = 4500   # max frames in the per-hand signal buffer (~150s at 30fps)
NOISY_HR_VAR_THRESHOLD   = 100.0  # hr_variance above this = unreliable rPPG signal
TIMESERIES_DOWNSAMPLE    = 10     # store every Nth frame in the DB timeseries
MIN_HAND_BUFFER_FRAMES   = 15     # minimum buffer frames before a heuristic prediction fires
MIN_PRED_SAMPLES         = 8      # showdowns needed before the trained model replaces heuristic

# --- Player identity ---
FACE_SIMILARITY_THRESHOLD = 0.85  # cosine similarity to match a face to an existing profile

# --- Gestures ---
THUMB_DWELL_SECONDS       = 0.4   # seconds to hold thumb gesture before it fires
SHOWDOWN_COOLDOWN_SECONDS = 3.0   # minimum seconds between two showdown recordings
PINCH_LOCK_DURATION       = 1     # seconds to hold a pinch to lock/register a card
# Pinch detection threshold: normalised distance between thumb tip and index
# tip relative to max(frame_w, frame_h). Larger = easier to register a pinch.
# 0.08 = strict (hand close to camera), 0.12-0.15 = phone back-cam friendly.
PINCH_THRESHOLD           = 0.13

# --- Camera resolution ---
CAMERA_WIDTH  = 1280
CAMERA_HEIGHT = 720
