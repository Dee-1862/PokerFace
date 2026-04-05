"""
Adaptive Learning Integration Module (Multi-Agent Learning Layer)

Orchestrates Identity + Baseline, Personality Inference, and Prediction agents.
Single source of truth: ProfileStore + FaceEmbedder. Works with unified_ar_system
now and poker_ar_unified once Phase 3 migrates.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    import config as _cfg
except ImportError:
    _cfg = None  # type: ignore[assignment]

from .baseline_extractor import BaselineExtractor
from .face_embedder import FaceEmbedder
from .profile_store import ProfileStore
from .opponent_model import OpponentModel
from .prior_generator import SimplePriorGenerator
from .personality_model import PersonalityModel
from .claude_advisor import ClaudeAdvisor


class AdaptiveLearningSystem:
    """
    High-level coordinator for the adaptive opponent learning system.

    Auto-calibration: baseline is captured silently in the background as soon as
    a face is detected.  No user action required.  If a saved baseline exists but
    is older than BASELINE_MAX_AGE_DAYS it is discarded and re-captured.
    """

    BASELINE_MAX_AGE_DAYS = getattr(_cfg, 'BASELINE_MAX_AGE_DAYS', 7)
    _BASELINE_MAX_AGE = BASELINE_MAX_AGE_DAYS * 24 * 3600  # seconds

    def __init__(self, db_path: str = None, use_slm: bool = False):
        """
        Initialize the adaptive learning system.

        Args:
            db_path: Path to SQLite database (None = default)
            use_slm: Whether to use SLM for cold-start priors
        """
        self.baseline = BaselineExtractor()
        self.embedder = FaceEmbedder()
        self.profiles = ProfileStore(db_path)

        prior_gen = SimplePriorGenerator() if not use_slm else None
        self.model = OpponentModel(prior_generator=prior_gen)
        self.personality = PersonalityModel()

        # Claude advisor: agentic signal analyst — initialized lazily so a missing
        # API key doesn't crash the whole system, it just disables Claude predictions.
        try:
            self.claude_advisor: Optional[ClaudeAdvisor] = ClaudeAdvisor(self.profiles)
            print("[AdaptiveLearning] ClaudeAdvisor ready")
        except ValueError as e:
            self.claude_advisor = None
            print(f"[AdaptiveLearning] ClaudeAdvisor disabled: {e}")

        # Current state (single-opponent)
        self.current_player_id = None
        self.current_deviation = None
        self.last_prediction = None
        self._last_hand_strength_bucket: Optional[str] = None
        self.is_tracking = False
        self._needs_auto_calibration = False  # set by _load_player_state

        # Session debrief tracking
        self._face_last_seen: float = 0.0
        self._session_debrief_fired: bool = False

        # Per-hand signal buffer: accumulates ALL frame deviations (not just peaks).
        # Full buffer is passed to ClaudeAdvisor at showdown; peak stats also extracted
        # for the bandit/personality models.
        self._hand_buffer: List[Dict] = []
        self._HAND_BUFFER_MAX          = getattr(_cfg, 'HAND_BUFFER_MAX',          4500)
        self._NOISY_HR_VAR_THRESHOLD   = getattr(_cfg, 'NOISY_HR_VAR_THRESHOLD',  100.0)
        self._TIMESERIES_DOWNSAMPLE    = getattr(_cfg, 'TIMESERIES_DOWNSAMPLE',      10)
        self._hand_start_time: float = time.time()

        print("[AdaptiveLearning] System initialized (identity + personality + prediction)")
    
    def on_face_detected(self, landmarks) -> Optional[str]:
        """
        Called when a face is detected in frame.
        
        Args:
            landmarks: MediaPipe face landmarks (468 points)
            
        Returns:
            Player ID if matched/created, None if invalid
        """
        # Extract embedding
        embedding = self.embedder.get_embedding(landmarks)
        if embedding is None:
            return None
        
        # Find or create player
        player_id = self.profiles.find_or_create_player(embedding)
        
        # If new player (or first detection), load their saved state and count the session
        if player_id != self.current_player_id:
            self._load_player_state(player_id)
            self.profiles.increment_session_count(player_id)
            self.current_player_id = player_id
            self.is_tracking = True

            # Show session notes from last visit if available
            notes = self.profiles.load_session_notes(player_id)
            if notes:
                print(f"\n[AdaptiveLearning] === SESSION NOTES for {player_id[:8]} ===")
                print(notes)
                print("[AdaptiveLearning] ==========================================\n")

            # Reset face-tracking state for new player
            self._face_last_seen = time.time()
            self._session_debrief_fired = False

        # Auto-start baseline calibration silently if needed
        if self._needs_auto_calibration and not self.baseline.is_calibrating:
            self.baseline.start_calibration()
            self._needs_auto_calibration = False
            print("[AdaptiveLearning] Auto-calibration started (background)")

        return player_id
    
    def _load_player_state(self, player_id: str) -> None:
        """Load saved state for a player (baseline, personality, predictor)."""
        self._hand_buffer = []  # fresh buffer for new player
        # Load baseline — discard if older than BASELINE_MAX_AGE_DAYS
        saved_baseline = self.profiles.load_baseline(player_id)
        self._needs_auto_calibration = False
        if saved_baseline:
            age = time.time() - saved_baseline.get('saved_at', 0)
            if age > self._BASELINE_MAX_AGE:
                print(f"[AdaptiveLearning] Baseline stale ({age/3600:.0f}h old), will recalibrate")
                self.baseline.reset()
                self._needs_auto_calibration = True
            else:
                self.baseline.from_dict(saved_baseline)
                print(f"[AdaptiveLearning] Loaded baseline for {player_id[:8]}")
        else:
            self.baseline.reset()
            self._needs_auto_calibration = True
        
        # Load personality state
        personality_data = self.profiles.load_personality_state(player_id)
        self.personality.from_dict(player_id, personality_data)
        if personality_data:
            print(f"[AdaptiveLearning] Loaded personality for {player_id[:8]}")
        
        # Load bandit / predictor state
        alpha, beta = self.profiles.load_bandit_state(player_id)
        if alpha:
            self.model.load_state(player_id, alpha, beta)
            print(f"[AdaptiveLearning] Loaded bandit state for {player_id[:8]}")
        else:
            self.model.initialize_for_new_player(player_id, {
                'baseline_hr': saved_baseline.get('hr', 70) if saved_baseline else 70
            })
    
    def _post_calibration_background(self, samples, player_id):
        """
        Background daemon thread: runs after calibration completes.
        Handles Integration 1 (baseline quality filtering) and
        Integration 3 (cold-start priors).
        """
        import threading  # already imported at module level via stdlib but explicit here for clarity

        # Integration 1: baseline quality refinement
        quality = self.claude_advisor.analyze_baseline(samples)
        if quality and quality.get('clean_ranges'):
            self.baseline.refine_from_clean_ranges(
                quality['clean_ranges'],
                quality['step'],
                quality['original_n'],
                quality['verdict'],
                quality['clean_pct'],
            )
            # Save refined baseline (guard against player switch during analysis)
            if self.current_player_id == player_id:
                baseline_dict = self.baseline.to_dict()
                if baseline_dict:
                    baseline_dict['saved_at'] = time.time()
                    baseline_dict['quality_verdict'] = quality.get('verdict', 'UNKNOWN')
                    self.profiles.save_baseline(player_id, baseline_dict)
                    print(f"[AdaptiveLearning] Refined baseline saved ({quality.get('verdict')})")

        # Integration 3: cold-start priors (only if no bandit data yet)
        existing_alpha, _ = self.profiles.load_bandit_state(player_id)
        if not existing_alpha:
            baseline_dict = self.baseline.to_dict() or {}
            priors = self.claude_advisor.generate_cold_start_priors(baseline_dict, player_id)
            if priors:
                self.model._apply_priors(player_id, priors, pseudo_count=4)
                # Save priors as bandit state
                alpha_for_save = {
                    (player_id, k): v
                    for k, v in self.model.get_state_for_player(player_id)[0].items()
                }
                beta_for_save = {
                    (player_id, k): v
                    for k, v in self.model.get_state_for_player(player_id)[1].items()
                }
                self.profiles.save_bandit_state(player_id, alpha_for_save, beta_for_save)
                print(f"[AdaptiveLearning] Cold-start priors applied for {player_id[:8]}")

    def on_face_lost(self):
        """Call when face is no longer detected. Triggers session debrief after 60s absence."""
        if not self.current_player_id or self._session_debrief_fired:
            return
        now = time.time()
        if self._face_last_seen > 0 and (now - self._face_last_seen) > 60.0:
            self._session_debrief_fired = True
            player_id = self.current_player_id
            import threading
            t = threading.Thread(target=self._generate_debrief, args=(player_id,), daemon=True)
            t.start()

    def _generate_debrief(self, player_id):
        """Background thread: generate and persist a session debrief note via Claude."""
        if not self.claude_advisor:
            return
        showdowns = self.profiles.get_recent_showdowns(player_id, limit=20)
        if not showdowns:
            return
        personality_state = self.personality.get_state(player_id)
        context_summary = self.model.get_context_summary(player_id)
        notes = self.claude_advisor.generate_session_debrief(
            player_id, showdowns, personality_state, context_summary
        )
        if notes:
            self.profiles.save_session_notes(player_id, notes)
            print(f"[AdaptiveLearning] Session debrief saved for {player_id[:8]}")
            print(f"[AdaptiveLearning] Debrief: {notes}")

    def start_calibration(self, duration_frames: int = 90):
        """Manually restart baseline calibration (e.g. from 'b' key)."""
        self.baseline.start_calibration(duration_frames)
        print("[AdaptiveLearning] Manual recalibration started")
    
    def add_calibration_sample(self, hr: float, stress: float, au_values: Dict):
        """Add a sample during calibration."""
        self.baseline.add_sample(hr, stress, au_values)

        # Auto-save when calibration completes
        if not self.baseline.is_calibrating and self.baseline.has_baseline():
            if self.current_player_id:
                baseline_dict = self.baseline.to_dict()
                baseline_dict['saved_at'] = time.time()  # for stale-check on next load
                self.profiles.save_baseline(self.current_player_id, baseline_dict)
                print(f"[AdaptiveLearning] Baseline saved for {self.current_player_id[:8]}")

                # Launch background thread for quality refinement (Integration 1)
                # and cold-start priors (Integration 3)
                if self.claude_advisor is not None:
                    import threading
                    raw_samples = list(self.baseline.samples)
                    player_id = self.current_player_id
                    t = threading.Thread(
                        target=self._post_calibration_background,
                        args=(raw_samples, player_id),
                        daemon=True,
                    )
                    t.start()
    
    def get_prediction(
        self,
        hr: float,
        stress: float,
        au_values: Dict = None,
        hand_strength_bucket: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict]:
        """
        Get prediction for current player state (identity + personality + prediction flow).
        
        Args:
            hr: Current heart rate
            stress: Current stress level
            au_values: Current Action Unit values
            hand_strength_bucket: Optional "weak" | "medium" | "strong" for poker context
            context: Optional dict for context-adjusted deviation (e.g. pot_size, street); used when v2 provides it
            
        Returns:
            Dict with 'prediction', 'confidence', 'p_bluff', 'samples', 'personality_profile', etc., or None
        """
        self._face_last_seen = time.time()
        self._session_debrief_fired = False  # reset when face is actively tracked

        if not self.current_player_id or not self.baseline.has_baseline():
            return None

        current_state = {'hr': hr, 'stress': stress, 'au': au_values or {}}
        self.current_deviation = self.baseline.get_deviation(current_state, context=context)
        if not self.current_deviation:
            return None

        # Buffer this frame's deviation for peak-anomaly extraction at showdown
        self._hand_buffer.append({
            'hr_delta':     self.current_deviation.get('hr_delta', 0),
            'stress_delta': self.current_deviation.get('stress_delta', 0),
            'au_delta':     dict(self.current_deviation.get('au_delta', {})),
        })
        if len(self._hand_buffer) > self._HAND_BUFFER_MAX:
            self._hand_buffer = self._hand_buffer[-self._HAND_BUFFER_MAX:]

        self._last_hand_strength_bucket = hand_strength_bucket
        personality_state = self.personality.get_state(self.current_player_id)
        
        prediction, confidence, p_bluff = self.model.predict(
            self.current_player_id,
            self.current_deviation,
            personality_state=personality_state,
            hand_strength_bucket=hand_strength_bucket,
        )
        self.last_prediction = prediction
        
        samples = self.model.get_sample_count(
            self.current_player_id,
            self.current_deviation,
            personality_state=personality_state,
            hand_strength_bucket=hand_strength_bucket,
        )
        profile_str = self.personality.get_profile_string(self.current_player_id)
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'p_bluff': p_bluff,
            'samples': samples,
            'hr_delta': self.current_deviation.get('hr_delta', 0),
            'stress_delta': self.current_deviation.get('stress_delta', 0),
            'personality_profile': profile_str,
        }
    
    def get_heuristic_prediction(self) -> Optional[Dict]:
        """
        Raw physiological signal reading — no showdown history needed.
        Uses deviation from baseline directly to estimate bluff probability.
        This is the cold-start signal shown while the model is still learning.

        After 5+ showdowns the hardcoded weights are replaced by PersonalityModel's
        learned per-player regression slopes, so the signal that actually predicts
        bluffing for THIS person gets weighted higher automatically.

        Returns a dict compatible with get_prediction() output, plus 'is_heuristic': True.
        """
        if not self.baseline.has_baseline() or not self.current_deviation:
            return None

        dev  = self.current_deviation
        au_d = dev.get('au_delta', {})

        personality      = self.personality.get_state(self.current_player_id) if self.current_player_id else {}
        n_showdowns      = personality.get('n_showdowns', 0)
        using_learned    = n_showdowns >= 5

        # Use peak values from the full hand buffer rather than the current noisy frame.
        # This gives a stable, settled reading instead of a jittery per-frame value.
        if len(self._hand_buffer) >= 15:
            hr_vals     = np.array([f['hr_delta']     for f in self._hand_buffer])
            stress_vals = np.array([f['stress_delta'] for f in self._hand_buffer])
            # 80th-percentile magnitude (catches genuine elevation without reacting to spikes)
            hr_d  = float(np.sign(np.mean(hr_vals))     * np.percentile(np.abs(hr_vals),     80))
            st_d  = float(np.sign(np.mean(stress_vals)) * np.percentile(np.abs(stress_vals), 80))
            # Down-weight HR if it's been jumping around (variance > 40 BPM²)
            hr_var = float(np.var(hr_vals))
            hr_reliability = max(0.0, 1.0 - (hr_var - 10) / 90) if hr_var > 10 else 1.0
        else:
            # Not enough buffer frames yet — use current frame but with reduced HR weight
            hr_d = dev.get('hr_delta', 0)
            st_d = dev.get('stress_delta', 0)
            hr_reliability = 0.4  # conservative until buffer fills

        if using_learned:
            bluff_base = personality.get('bluff_base_rate', 0.5)
            p = bluff_base
            hr_slope = personality.get('hr_bluff_slope', 0.0)
            st_slope = personality.get('stress_bluff_slope', 0.0)
            p += hr_reliability * max(-0.25, min(0.25, hr_d * hr_slope))
            p += max(-0.20, min(0.20, st_d * st_slope))
        else:
            p = 0.5
            p += hr_reliability * min(0.22, max(-0.22, hr_d / 35.0))
            p += min(0.18, max(-0.18, st_d * 0.7))

        # AU contributions — use per-AU peak across the hand buffer when available
        if len(self._hand_buffer) >= 15:
            for au_name in ('AU4', 'AU23', 'AU20', 'AU1', 'blink_rate'):
                au_vals = np.array([f['au_delta'].get(au_name, 0.0) for f in self._hand_buffer])
                au_d[au_name] = float(np.sign(np.mean(au_vals)) * np.percentile(np.abs(au_vals), 80))

        p += au_d.get('AU4',       0) * 0.06
        p += au_d.get('AU23',      0) * 0.06
        p += au_d.get('AU20',      0) * 0.05
        p += au_d.get('AU1',       0) * 0.03
        p += au_d.get('blink_rate', 0) * 0.04
        p = max(0.10, min(0.90, p))

        label = 'BLUFFING' if p > 0.5 else 'STRONG HAND'
        return {
            'prediction':           label,
            'confidence':           abs(p - 0.5) * 2,
            'p_bluff':              p,
            'samples':              0,
            'hr_delta':             hr_d,
            'stress_delta':         st_d,
            'personality_profile':  personality.get('profile_string', None),
            'is_heuristic':         True,
            'using_learned_weights': using_learned,
        }

    def start_hand(self):
        """Reset the per-hand signal buffer. Call at the start of each new hand."""
        self._hand_buffer = []
        self._hand_start_time = time.time()

    def _extract_peak_deviation(self) -> Optional[Dict]:
        """
        Analyse the buffered frames from this hand and return a deviation dict
        that represents the *anomalous* signal rather than the average.

        Strategy:
          - HR and stress: take the sample whose absolute deviation is at the
            90th-percentile (top 10% most extreme frames).  This catches
            genuine spikes while ignoring single-frame noise.
          - AUs: for each action unit take the 90th-percentile of its absolute
            activation across the hand buffer.  Sign is preserved (we keep the
            sign of the sample closest to that magnitude).

        Falls back to current_deviation if the buffer is too short.
        """
        if len(self._hand_buffer) < 5:
            return self.current_deviation  # not enough data, use latest frame

        hr_vals     = np.array([s['hr_delta']     for s in self._hand_buffer])
        stress_vals = np.array([s['stress_delta'] for s in self._hand_buffer])

        # 90th-percentile of absolute value, preserving the sign of the extreme sample
        def _signed_p90(vals):
            abs_vals = np.abs(vals)
            threshold = np.percentile(abs_vals, 90)
            # pick the first sample at or above threshold
            candidates = vals[abs_vals >= threshold]
            if len(candidates) == 0:
                return float(vals[np.argmax(abs_vals)])
            # return the most extreme of the candidates
            return float(candidates[np.argmax(np.abs(candidates))])

        peak_hr     = _signed_p90(hr_vals)
        peak_stress = _signed_p90(stress_vals)

        # Gather all AU names seen during the hand
        all_au_names = set()
        for s in self._hand_buffer:
            all_au_names.update(s['au_delta'].keys())

        peak_au = {}
        for au in all_au_names:
            au_vals = np.array([s['au_delta'].get(au, 0.0) for s in self._hand_buffer])
            peak_au[au] = _signed_p90(au_vals)

        # Variance: how sustained vs. spike was the signal
        hr_var     = float(np.var(hr_vals))
        stress_var = float(np.var(stress_vals))

        # Peak frame position (0.0 = start of hand, 1.0 = end)
        hr_peak_idx = int(np.argmax(np.abs(hr_vals)))
        peak_frame_pct = round(hr_peak_idx / max(1, len(hr_vals) - 1), 3)

        hand_duration = time.time() - self._hand_start_time

        print(f"[AdaptiveLearning] Peak deviation from {len(self._hand_buffer)} frames "
              f"({hand_duration:.0f}s hand): "
              f"HR Δ{peak_hr:+.1f} (var {hr_var:.1f})  stress Δ{peak_stress:+.2f} (var {stress_var:.3f})")
        return {
            'hr_delta':       peak_hr,
            'stress_delta':   peak_stress,
            'au_delta':       peak_au,
            'frames_sampled': len(self._hand_buffer),
            'hr_variance':    hr_var,
            'stress_variance': stress_var,
            'peak_frame_pct': peak_frame_pct,
            'hand_start_time': self._hand_start_time,
            'hand_duration':   round(hand_duration, 1),
        }

    def on_showdown(self, was_bluffing: bool) -> bool:
        """
        Called after showdown: update Personality and Prediction agents, persist to ProfileStore.
        
        Returns:
            True if the prediction was correct
        """
        if not self.current_player_id or not self.current_deviation:
            print("[AdaptiveLearning] Cannot update: no current player or deviation")
            return False

        full_buffer = list(self._hand_buffer)
        hand_deviation = self._extract_peak_deviation()
        self._hand_buffer = []  # reset for next hand

        hr_noisy = hand_deviation.get('hr_variance', 0) > self._NOISY_HR_VAR_THRESHOLD
        if hr_noisy:
            print(f"[AdaptiveLearning] WARNING: HR variance {hand_deviation['hr_variance']:.1f} exceeds "
                  f"threshold {self._NOISY_HR_VAR_THRESHOLD} — skipping bandit/personality update (noisy rPPG)")

        personality_state = self.personality.get_state(self.current_player_id)

        if not hr_noisy:
            self.personality.update(self.current_player_id, hand_deviation, was_bluffing)
        pers_data = self.personality.to_dict(self.current_player_id)
        if pers_data is not None:
            self.profiles.save_personality_state(self.current_player_id, pers_data)

        was_correct = self.model.update_with_prediction_result(
            self.current_player_id,
            hand_deviation,
            self.last_prediction or "STRONG",
            was_bluffing,
            personality_state=personality_state,
            hand_strength_bucket=self._last_hand_strength_bucket,
        ) if not hr_noisy else (self.last_prediction == ("BLUFFING" if was_bluffing else "STRONG"))

        alpha, beta = self.model.get_state_for_player(self.current_player_id)
        self.profiles.save_bandit_state(
            self.current_player_id,
            {(self.current_player_id, k): v for k, v in alpha.items()},
            {(self.current_player_id, k): v for k, v in beta.items()},
        )

        n = self._TIMESERIES_DOWNSAMPLE
        sampled = full_buffer[::n]
        au_names = sorted({au for f in sampled for au in f.get('au_delta', {})})
        timeseries = {
            'hr':     [f.get('hr_delta', 0.0)     for f in sampled],
            'stress': [f.get('stress_delta', 0.0) for f in sampled],
            'au':     {au: [f.get('au_delta', {}).get(au, 0.0) for f in sampled] for au in au_names},
            'downsample_factor': n,
            'original_frames': len(full_buffer),
        }

        context_bucket = self.model._get_context_bucket(hand_deviation)
        showdown_id = self.profiles.log_showdown(
            self.current_player_id,
            context_bucket,
            self.last_prediction or "STRONG",
            "BLUFFING" if was_bluffing else "STRONG",
            was_correct,
            hand_deviation,
            timeseries=timeseries,
            claude_result=None,
        )

        result = "CORRECT" if was_correct else "WRONG"
        print(f"[AdaptiveLearning] Showdown logged: {result} (Claude analysis running in background)")

        if self.claude_advisor and full_buffer:
            import threading
            _pid  = self.current_player_id
            _buf  = full_buffer
            _base = self.baseline.to_dict() or {}
            _sdid = showdown_id

            def _run_claude():
                history = self.profiles.get_recent_showdowns(_pid, limit=20)
                cr = self.claude_advisor.analyze_hand(_pid, _buf, _base, history)
                if cr and _sdid is not None:
                    self.profiles.update_showdown_claude_result(_sdid, cr)
                    print(f"[AdaptiveLearning] Claude (async): "
                          f"{cr['prediction']} p={cr['p_bluff']:.2f}")

            threading.Thread(target=_run_claude, daemon=True).start()
        # -------------------------------------------------------------------------

        return was_correct
    
    def get_status(self) -> Dict:
        """Get current system status for UI display."""
        return {
            'player_id': self.current_player_id[:8] if self.current_player_id else None,
            'has_baseline': self.baseline.has_baseline(),
            'is_calibrating': self.baseline.is_calibrating,
            'calibration_progress': self.baseline.get_calibration_progress(),
            'model_accuracy': self.model.get_accuracy(),
            'total_predictions': self.model.predictions_made,
            'is_tracking': self.is_tracking
        }
    
    def get_player_stats(self) -> Optional[Dict]:
        """Get stats for current player."""
        if not self.current_player_id:
            return None
        return self.profiles.get_player_stats(self.current_player_id)
    
    def get_context_summary(self) -> Optional[Dict]:
        """Get learned parameters for current player."""
        if not self.current_player_id:
            return None
        return self.model.get_context_summary(self.current_player_id)
    
    def get_personality_profile(self) -> Optional[str]:
        """Short interpretable personality string for current player (e.g. 'bluffs more under stress')."""
        if not self.current_player_id:
            return None
        return self.personality.get_profile_string(self.current_player_id)
    
    def reset_current_player(self):
        """Reset learning for current player."""
        if self.current_player_id:
            self.model.reset_player(self.current_player_id)
            self.personality._state.pop(self.current_player_id, None)
            self.profiles.delete_player(self.current_player_id)
            self.current_player_id = None
            self.baseline.reset()
    
    def delete_all_profiles(self):
        """Delete all player profiles (privacy reset)."""
        self.profiles.delete_all_profiles()
        self.personality._state.clear()
        self.current_player_id = None
        self.baseline.reset()
        print("[AdaptiveLearning] All profiles deleted")
    
    def close(self):
        """Clean up resources."""
        self.profiles.close()
