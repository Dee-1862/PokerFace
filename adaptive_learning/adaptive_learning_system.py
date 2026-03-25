"""
Adaptive Learning Integration Module (Multi-Agent Learning Layer)

Orchestrates Identity + Baseline, Personality Inference, and Prediction agents.
Single source of truth: ProfileStore + FaceEmbedder. Works with unified_ar_system
now and poker_ar_unified once Phase 3 migrates.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
import time
import numpy as np

from .baseline_extractor import BaselineExtractor
from .face_embedder import FaceEmbedder
from .profile_store import ProfileStore
from .opponent_model import OpponentModel
from .prior_generator import SimplePriorGenerator
from .personality_model import PersonalityModel


class AdaptiveLearningSystem:
    """
    High-level coordinator for the adaptive opponent learning system.

    Auto-calibration: baseline is captured silently in the background as soon as
    a face is detected.  No user action required.  If a saved baseline exists but
    is older than BASELINE_MAX_AGE_DAYS it is discarded and re-captured.
    """

    BASELINE_MAX_AGE_DAYS = 7
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

        # Current state (single-opponent)
        self.current_player_id = None
        self.current_deviation = None
        self.last_prediction = None
        self._last_hand_strength_bucket: Optional[str] = None
        self.is_tracking = False
        self._needs_auto_calibration = False  # set by _load_player_state

        # Per-hand signal buffer: accumulates deviations from start of hand to showdown.
        # On showdown we extract peak (90th-percentile anomaly) instead of a single frame.
        self._hand_buffer: List[Dict] = []
        self._HAND_BUFFER_MAX = 900  # ~30s at 30fps

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
        
        # If new player (or first detection), load their saved state
        if player_id != self.current_player_id:
            self._load_player_state(player_id)
            self.current_player_id = player_id
            self.is_tracking = True

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

        Returns a dict compatible with get_prediction() output, plus 'is_heuristic': True.
        """
        if not self.baseline.has_baseline() or not self.current_deviation:
            return None

        dev = self.current_deviation
        hr_d  = dev.get('hr_delta', 0)
        st_d  = dev.get('stress_delta', 0)
        au_d  = dev.get('au_delta', {})

        # Weighted combination: each signal nudges probability away from neutral 0.5
        p = 0.5
        p += min(0.22, max(-0.22, hr_d  / 35.0))     # HR elevation  → bluffing
        p += min(0.18, max(-0.18, st_d  * 0.7))      # Stress        → bluffing
        p += au_d.get('AU4',  0) * 0.06              # Brow furrow   → nervous
        p += au_d.get('AU23', 0) * 0.06              # Lip tighten   → suppression
        p += au_d.get('AU20', 0) * 0.05              # Lip stretch   → fear/anxiety
        p += au_d.get('AU1',  0) * 0.03              # Inner brow raise (concern)
        p = max(0.10, min(0.90, p))

        label = 'BLUFFING' if p > 0.5 else 'STRONG HAND'
        return {
            'prediction':        label,
            'confidence':        abs(p - 0.5) * 2,
            'p_bluff':           p,
            'samples':           0,
            'hr_delta':          hr_d,
            'stress_delta':      st_d,
            'personality_profile': None,
            'is_heuristic':      True,
        }

    def start_hand(self):
        """Reset the per-hand signal buffer. Call at the start of each new hand."""
        self._hand_buffer = []

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

        print(f"[AdaptiveLearning] Peak deviation from {len(self._hand_buffer)} frames: "
              f"HR Δ{peak_hr:+.1f} (var {hr_var:.1f})  stress Δ{peak_stress:+.2f} (var {stress_var:.3f})")
        return {
            'hr_delta':       peak_hr,
            'stress_delta':   peak_stress,
            'au_delta':       peak_au,
            'frames_sampled': len(self._hand_buffer),
            'hr_variance':    hr_var,
            'stress_variance': stress_var,
            'peak_frame_pct': peak_frame_pct,   # where in the hand the HR peak occurred
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

        # Use peak deviation from the entire hand buffer (anomaly-based), not the current frame
        hand_deviation = self._extract_peak_deviation()
        self._hand_buffer = []  # reset for next hand

        personality_state = self.personality.get_state(self.current_player_id)

        # Update personality from this showdown
        self.personality.update(self.current_player_id, hand_deviation, was_bluffing)
        pers_data = self.personality.to_dict(self.current_player_id)
        if pers_data is not None:
            self.profiles.save_personality_state(self.current_player_id, pers_data)

        # Update predictor (bandit/RL)
        was_correct = self.model.update_with_prediction_result(
            self.current_player_id,
            hand_deviation,
            self.last_prediction or "STRONG",
            was_bluffing,
            personality_state=personality_state,
            hand_strength_bucket=self._last_hand_strength_bucket,
        )

        # Persist bandit state
        alpha, beta = self.model.get_state_for_player(self.current_player_id)
        self.profiles.save_bandit_state(
            self.current_player_id,
            {(self.current_player_id, k): v for k, v in alpha.items()},
            {(self.current_player_id, k): v for k, v in beta.items()},
        )

        # Log showdown
        context_bucket = self.model._get_context_bucket(hand_deviation)
        self.profiles.log_showdown(
            self.current_player_id,
            context_bucket,
            self.last_prediction or "STRONG",
            "BLUFFING" if was_bluffing else "STRONG",
            was_correct,
            hand_deviation,
        )
        
        result = "CORRECT" if was_correct else "WRONG"
        print(f"[AdaptiveLearning] Showdown logged: {result}")
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
