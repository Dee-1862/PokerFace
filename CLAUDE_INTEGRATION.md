# Claude API Integration Plan

This document describes five planned integrations of the Claude API into the adaptive learning pipeline. Each section covers what the system currently does, what the problem is, and exactly how the integration changes it.

No integration touches the camera loop. All Claude calls are asynchronous background threads that fire between hands or at session boundaries. Real-time performance is unaffected.

All data sent to Claude is numeric signal data only - AU intensities, heart rate readings, stress scores, and showdown statistics. No images, no face embeddings, no personally identifying information ever leaves the device.

---

## Integration 1 - Baseline Quality Filtering

### What the system does today

`baseline_extractor.py` runs a calibration window when a new face is detected. It collects samples for a fixed number of frames and then calls `_finalize_calibration()`, which computes the mean and standard deviation of every sample collected regardless of what the person was doing during that window. All 900 frames are treated equally.

Every z-score deviation computed for the rest of the session is measured against this baseline. A reading of `hr_delta = +2.1` means the opponent's HR is 2.1 standard deviations above whatever was recorded during those 30 seconds.

### The problem

The person being calibrated is rarely sitting perfectly still. They may be talking, laughing, glancing around, or settling into their seat. Any of these activities corrupts the AU and HR readings. A 10-second talking episode inflates AU12 (lip corner) and AU26 (jaw) in the baseline. A head movement corrupts the rPPG signal. Once those corrupted values are locked in as the baseline, the system under-reports stress for the rest of the session because the baseline is already elevated. A genuine tell gets swallowed by a bad reference point.

There is currently no quality check. The code accepts whatever it collected and saves it.

### How it changes after integration

After the calibration window closes, the full sample buffer already held in `self.samples` is serialised to a compact JSON time series and sent to Claude in a background thread. The main loop is not blocked.

Claude receives HR, stress score, and all six AU values per frame and identifies which segments represent genuine resting state. It uses FACS knowledge to interpret AU co-occurrence: AU12 and AU26 spiking together indicates talking or laughing, not a stress tell. AU45 (blink) spiking alongside sudden AU1 and AU2 movement indicates a head turn. It returns the start and end indices of clean segments and a quality verdict.

`_finalize_calibration` then computes mean and std dev over only the clean frames instead of all frames.

If Claude reports that fewer than 40 percent of frames are clean, the system extends the calibration window rather than saving a corrupted baseline. A quality indicator is shown on screen: "Baseline: GOOD - 524 of 847 frames used" or "Baseline: POOR - please stay still for 15 more seconds."

**Before:** average everything, silent failure, bad baseline poisons the session.

**After:** clean frames only, quality score, automatic extension if the window was too noisy.

---

## Integration 2 - Showdown Signal Interpretation

### What the system does today

When a showdown gesture is recorded, `opponent_model.py` identifies the current context bucket (a string like `HH_H_u` combining HR state, stress state, personality slope, and hand bucket) and increments either alpha or beta in the Beta distribution for that bucket. The bandit knows the bucket fired and whether it was a bluff. It has no knowledge of what the physiological signal actually looked like during the hand.

`profile_store.py` saves hr_variance, stress_variance, and peak_frame_pct to the showdowns table but nothing downstream reads these columns to adjust how signals are weighted per player.

### The problem

Two hands can land in the same context bucket with completely different underlying patterns. In one hand stress ramped up gradually over 40 frames. In another it spiked sharply at the moment of the bet and immediately dropped. Both update the same bucket identically. The bandit has no way to learn that this player's tell is a spike at the bet, not sustained elevation.

`PersonalityModel` tracks stress-bluff slope and HR-bluff slope as running averages but these are global across all hands. There is no per-player weighting of which AU is most predictive or whether the timing of the anomaly within the hand matters.

### How it changes after integration

After a showdown gesture is recorded, the 2-4 anomaly frames already extracted by the signal buffer (the 90th-percentile peak window that the buffer already identifies) are serialised to JSON with their AU values, HR delta, stress delta, and position in the hand timeline. This fires in a background thread.

Claude receives the anomaly frames alongside the context bucket, prediction made, and actual result. It returns a structured interpretation: which AU combination dominated, whether the anomaly was sustained or a brief spike, what expression pattern this represents in FACS terms, and a suggested per-signal weight adjustment for this player.

The interpretation is stored as a new column in the showdowns table. Over several sessions the weight adjustments accumulate in personality_state and the bandit gradually focuses on the signals that are actually predictive for this specific person rather than applying the same fixed weights to everyone.

**Before:** bandit updates a bucket counter, signal shape is discarded, all players weighted identically.

**After:** Claude names the expression pattern, timing is recorded, per-player signal weights adjust over time toward whichever signals are genuinely predictive.

---

## Integration 3 - Cold-Start Prior Generation

### What the system does today

`prior_generator.py` is already stubbed in the codebase. The full `PriorGenerator` class targets Microsoft Phi-3-mini running locally via transformers and torch - a multi-gigabyte download that is disabled by default. The active code falls through to `SimplePriorGenerator`, which applies three hard-coded heuristics based on resting HR and an experience level estimate.

When a new player is seen for the first time, the bandit starts with uniform priors: alpha = 1, beta = 1 for every bucket, meaning it has no initial opinion. For the first two showdowns the system displays a heuristic estimate labelled "SIGNAL". The bandit only takes over after that.

### The problem

With uniform priors, the first two hands carry near-random predictions. Against an unfamiliar player, those are often the most important hands. The three heuristics in `SimplePriorGenerator` are population-level generalisations and do not use the actual physiological data that has just been collected during baseline calibration.

The local Phi-3 approach is correct in intent but wrong in execution - it requires a dependency stack most machines cannot easily run and the model quality for structured JSON output is inconsistent.

### How it changes after integration

`PriorGenerator` is rewritten to call Claude API instead of loading a local model. The call fires once, at the moment a new player profile is created in `profile_store.py`, never during gameplay.

At that point the baseline has just been computed and validated (by Integration 1), giving Claude real data to work with: resting HR, baseline stress level, blink rate, AU resting values, and the personality type inferred from the first few frames. Claude returns a full set of starting alpha and beta values for every context bucket, reflecting what is known about this specific person's physiology before any showdowns have been seen.

`_apply_priors` in `opponent_model.py` already exists and handles loading these values with a configurable pseudo-count. No structural change to the bandit is needed.

**Before:** uniform priors, three hard-coded heuristics, multi-gigabyte local model dependency disabled by default.

**After:** one Claude API call per new player at profile creation, informed priors from real baseline data, no local model dependency.

---

## Integration 4 - Session Debrief

### What the system does today

`PersonalityModel.get_profile_string()` produces a single line assembled from four threshold comparisons - for example "bluffs often; bluffs more under stress". It cannot express street patterns, timing observations, HR versus stress dominance, session-over-session changes, or how reliable specific tells have been.

When the face leaves frame the session ends silently. No summary is generated. The next time the player is recognised, the profile string still shows the same four-word description it showed last session.

### The problem

The showdowns table contains rich data: context bucket per hand, whether the prediction was correct, HR variance, stress variance, peak frame position in the hand, and timestamp. The personality_state table contains bluff base rate, stress-bluff slope, HR-bluff slope, and consistency. The bandit's `get_context_summary()` returns alpha-beta ratios for every bucket the player has been observed in.

None of this is surfaced in a form that is actually useful when you sit down opposite the same player at a new session. You have to remember what you noticed last time or re-learn it from scratch.

### How it changes after integration

When the system detects the opponent's face has been absent for more than 60 seconds, it queries all three tables for that player_id and serialises the data to JSON. This fires in a background thread and writes the result back to a new `session_notes` column in the players table.

Claude receives the full showdown history, personality state, and bandit context summary. It writes a plain-English paragraph in second-person note style, for example:

> "Tight pre-flop, rarely bluffed in early position. Bluffed 4 of 7 times on the river, usually with elevated HR but controlled stress - the HR spike is the stronger tell for this player. Caught 3 of those 4. In the last three hands of the session prediction accuracy dropped, possibly adjusting after being caught. Stress-bluff correlation is weak; focus on HR and AU4 sustained across multiple frames."

The next time the player is recognised, before calibration starts, this note is displayed for 5 seconds on screen and printed to console.

**Before:** four-word profile string, no session history surfaced, must remember everything manually.

**After:** plain-English paragraph written after each session, shown automatically at next recognition, grounded in the actual showdown data rather than hard-coded thresholds.

---

## Integration 5 - Behaviour Shift Detection

### What the system does today

The bandit accumulates alpha and beta indefinitely across all sessions. A showdown from six sessions ago carries the same weight as the one that just happened. There is no mechanism to detect when a player has changed their behaviour between sessions or within a session.

`get_player_stats()` tracks overall prediction accuracy but nothing compares recent accuracy to historical accuracy and nothing acts on a detected drop.

### The problem

If a player has noticed they are being read and consciously suppresses their tells - breathing differently, deliberating longer, keeping expression flat when bluffing - the bandit keeps using a now-invalid model with misplaced confidence. The alpha-beta ratios reflect a player who no longer exists. A full fist-reset (the existing gesture) wipes everything including valid learning. There is no middle path.

A simple rolling accuracy check ("if last 5 accuracy is below 40 percent, warn") can detect that something is wrong but cannot tell whether to recalibrate everything or just the buckets that have degraded.

### How it changes after integration

After every 5th showdown in a session, the system computes rolling accuracy: the last 5 hands versus the lifetime accuracy from `get_player_stats()`. If the gap exceeds 20 percentage points, a Claude check fires in a background thread.

Claude receives the last 10 showdowns split into two windows of 5, the current personality state, and the session number. It determines whether the degradation is natural variance, a genuine style shift, or a tell suppression pattern and returns one of three verdicts:

- `variance` - sample too small to conclude, do nothing
- `partial_recalibration` - decay alpha and beta for the specific buckets that have degraded, preserving buckets that are still accurate
- `full_reset` - suggest wiping the bandit state and starting fresh while keeping personality_state as a prior

The verdict drives a targeted response. Partial recalibration is implemented by multiplying the alpha and beta of affected buckets toward 1 (the uniform prior) by a decay factor, rather than zeroing them. This softens the model in the areas that are failing while keeping the areas that are working.

A one-line notification appears on screen when a verdict other than `variance` is returned: "Opponent may have adjusted - model updating" or "Significant behaviour shift detected - consider recalibration."

**Before:** alpha-beta accumulates forever, no detection of style change, only option is full manual reset via fist gesture.

**After:** rolling accuracy monitored every 5 hands, Claude classifies degradation type, partial decay applied surgically to failing buckets only, screen notification on detection.

---

## Summary

| Integration | Fires when | Blocks camera loop | Data sent to Claude |
|-------------|-----------|-------------------|---------------------|
| 1 - Baseline filtering | Calibration window closes | No | AU + HR + stress time series |
| 2 - Signal interpretation | Showdown gesture recorded | No | 2-4 anomaly frames of AU + HR + stress values |
| 3 - Cold-start priors | New player profile created | No | Baseline stats, AU resting values, personality type |
| 4 - Session debrief | Face absent 60+ seconds | No | Full showdown history, personality state, bandit summary |
| 5 - Behaviour shift | Every 5th showdown in session | No | Last 10 showdowns split into two windows |

## Files affected

| File | Change |
|------|--------|
| `adaptive_learning/baseline_extractor.py` | `_finalize_calibration` uses clean segments from Claude instead of all frames |
| `adaptive_learning/prior_generator.py` | Replace local Phi-3 with Claude API call |
| `adaptive_learning/opponent_model.py` | Read per-player signal weights from personality_state, partial decay for shift detection |
| `adaptive_learning/personality_model.py` | Store per-signal weight adjustments from Integration 2 |
| `adaptive_learning/profile_store.py` | Add `session_notes` column to players table, store Claude interpretation in showdowns |
| `adaptive_learning/adaptive_learning_system.py` | Orchestrate all five background threads, surface debrief note on recognition |
