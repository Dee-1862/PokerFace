"""
Opponent Model Module (Prediction RL Agent)

Thompson Sampling contextual bandit for predicting player behavior.
State includes physio bucket, optional personality bucket, and optional hand-strength bucket.
Uses Beta distributions per (player_id, state_bucket); updates at showdown.
"""

import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, Optional, Any
import time


def _personality_bucket(personality_state: Optional[Dict[str, float]]) -> str:
    """Map personality state to one of H/M/L for stress-bluff correlation."""
    if not personality_state or personality_state.get('n_showdowns', 0) < 2:
        return "M"
    slope = personality_state.get('stress_bluff_slope', 0.0) or 0.0
    if slope > 0.2:
        return "H"
    if slope < -0.2:
        return "L"
    return "M"


def _state_bucket(
    deviation: Dict,
    personality_state: Optional[Dict[str, float]] = None,
    hand_strength_bucket: Optional[str] = None,
) -> str:
    """Full state bucket: physio + personality + hand. Use physio-only for backward compatibility when no personality/hand."""
    physio = OpponentModel._get_context_bucket_static(deviation)
    if personality_state is None and not hand_strength_bucket:
        return physio
    pers = _personality_bucket(personality_state)
    hand_b = (hand_strength_bucket or "unknown").lower()[:1]
    if hand_b not in ("w", "m", "s"):
        hand_b = "u"
    return f"{physio}_{pers}_{hand_b}"


class OpponentModel:
    """
    Thompson Sampling contextual bandit for opponent behavior prediction.
    
    The model maintains Beta distribution parameters (alpha, beta) for each
    (player_id, context_bucket) pair. Predictions are made by sampling from
    the posterior, and updates are made after showdowns.
    
    Usage:
        model = OpponentModel()
        
        # Get prediction
        prediction, confidence, p_bluff = model.predict(player_id, deviation)
        
        # Update after showdown
        model.update(player_id, deviation, was_bluffing=True)
        
        # Load persisted state
        model.load_state(player_id, alpha_dict, beta_dict)
    """
    
    # Context bucket thresholds
    HR_HIGH_THRESHOLD = 10      # +10 BPM from baseline = high
    HR_LOW_THRESHOLD = -5       # -5 BPM from baseline = low
    STRESS_HIGH_THRESHOLD = 0.2  # +0.2 from baseline = high
    STRESS_LOW_THRESHOLD = -0.1  # -0.1 from baseline = low
    
    def __init__(self, prior_generator=None):
        """
        Initialize the opponent model.
        
        Args:
            prior_generator: Optional PriorGenerator for cold-start
        """
        # Beta distribution parameters: alpha = successes + 1, beta = failures + 1
        self.alpha = defaultdict(lambda: 1.0)
        self.beta = defaultdict(lambda: 1.0)
        
        # Optional SLM prior generator
        self.prior_generator = prior_generator
        
        # Tracking
        self.predictions_made = 0
        self.correct_predictions = 0
        
    def predict(
        self,
        player_id: str,
        deviation: Dict,
        personality_state: Optional[Dict[str, float]] = None,
        hand_strength_bucket: Optional[str] = None,
    ) -> Tuple[str, float, float]:
        """
        Make a prediction about whether the player is bluffing.
        
        Args:
            player_id: Unique player identifier
            deviation: Dict with 'hr_delta', 'stress_delta', 'au_delta' (context-adjusted when available)
            personality_state: Optional dict from PersonalityModel.get_state (adds personality bucket to state)
            hand_strength_bucket: Optional "weak" | "medium" | "strong" | None (unknown)
            
        Returns:
            Tuple of (prediction, confidence, p_bluff)
        """
        bucket = _state_bucket(deviation, personality_state, hand_strength_bucket)
        key = (player_id, bucket)
        
        # Thompson Sampling: sample from Beta posterior
        p_bluff = np.random.beta(self.alpha[key], self.beta[key])
        
        # Optional: blend with personality base rate when we have few samples (cold start)
        if personality_state and personality_state.get('n_showdowns', 0) >= 2:
            total = self.alpha[key] + self.beta[key] - 2
            if total < 3:  # few observations in this bucket
                blend = 0.4  # weight toward bandit
                base = personality_state.get('bluff_base_rate', 0.5)
                p_bluff = blend * p_bluff + (1 - blend) * base
        
        prediction = "BLUFFING" if p_bluff > 0.5 else "STRONG"
        confidence = abs(p_bluff - 0.5) * 2
        self.predictions_made += 1
        return prediction, confidence, p_bluff
    
    def get_expected_probability(
        self,
        player_id: str,
        deviation: Dict,
        personality_state: Optional[Dict[str, float]] = None,
        hand_strength_bucket: Optional[str] = None,
    ) -> float:
        """Get expected P(bluff) without sampling (mean of Beta)."""
        bucket = _state_bucket(deviation, personality_state, hand_strength_bucket)
        key = (player_id, bucket)
        return self.alpha[key] / (self.alpha[key] + self.beta[key])
    
    def update(
        self,
        player_id: str,
        deviation: Dict,
        was_bluffing: bool,
        personality_state: Optional[Dict[str, float]] = None,
        hand_strength_bucket: Optional[str] = None,
    ) -> None:
        """
        Update the model after a showdown.
        Uses same bucket as predict (physio + personality + hand) so state is consistent.
        """
        bucket = _state_bucket(deviation, personality_state, hand_strength_bucket)
        key = (player_id, bucket)
        if was_bluffing:
            self.alpha[key] += 1
        else:
            self.beta[key] += 1
        print(f"[OpponentModel] Updated {bucket}: alpha={self.alpha[key]:.1f}, beta={self.beta[key]:.1f}")
    
    def update_with_prediction_result(
        self,
        player_id: str,
        deviation: Dict,
        prediction: str,
        was_bluffing: bool,
        personality_state: Optional[Dict[str, float]] = None,
        hand_strength_bucket: Optional[str] = None,
    ) -> bool:
        """Update with explicit prediction tracking; returns True if prediction was correct."""
        was_correct = (prediction == "BLUFFING") == was_bluffing
        if was_correct:
            self.correct_predictions += 1
        self.update(
            player_id, deviation, was_bluffing,
            personality_state=personality_state,
            hand_strength_bucket=hand_strength_bucket,
        )
        return was_correct
    
    def _get_context_bucket(self, deviation: Dict) -> str:
        """Convert continuous deviations to physio bucket (e.g. HH, ML)."""
        return self._get_context_bucket_static(deviation)

    @staticmethod
    def _get_context_bucket_static(deviation: Dict) -> str:
        """Static version for use in _state_bucket."""
        if not deviation:
            return "MM"
        hr_delta = deviation.get('hr_delta', 0)
        stress_delta = deviation.get('stress_delta', 0)
        hr_bucket = "H" if hr_delta > OpponentModel.HR_HIGH_THRESHOLD else ("L" if hr_delta < OpponentModel.HR_LOW_THRESHOLD else "M")
        stress_bucket = "H" if stress_delta > OpponentModel.STRESS_HIGH_THRESHOLD else ("L" if stress_delta < OpponentModel.STRESS_LOW_THRESHOLD else "M")
        return f"{hr_bucket}{stress_bucket}"
    
    def initialize_for_new_player(self, player_id: str, observed_features: Dict = None):
        """
        Initialize priors for a new player using SLM (if available).
        
        Args:
            player_id: Player identifier
            observed_features: Optional dict of initial observations
                {'baseline_hr': 75, 'blink_rate': 12, 'fidgeting': 'low'}
        """
        if self.prior_generator and observed_features:
            try:
                priors = self.prior_generator.generate_prior(observed_features)
                self._apply_priors(player_id, priors)
                print(f"[OpponentModel] Applied SLM priors for player {player_id[:8]}...")
            except Exception as e:
                print(f"[OpponentModel] Failed to generate priors: {e}")
        else:
            print(f"[OpponentModel] No prior generator, using uniform priors")
    
    def _apply_priors(self, player_id: str, priors: Dict[str, float], pseudo_count: int = 5):
        """
        Apply LLM-generated priors to the model.
        
        Args:
            player_id: Player identifier
            priors: Dict mapping bucket -> P(bluffing)
            pseudo_count: Strength of prior (default: 5 pseudo-observations)
        """
        for bucket, p_bluff in priors.items():
            key = (player_id, bucket)
            # Convert probability to pseudo-counts
            self.alpha[key] = p_bluff * pseudo_count + 1
            self.beta[key] = (1 - p_bluff) * pseudo_count + 1
    
    def load_state(self, player_id: str, alpha_dict: Dict[str, float], beta_dict: Dict[str, float]):
        """
        Load persisted state for a player.
        
        Args:
            player_id: Player identifier
            alpha_dict: Dict mapping bucket -> alpha value
            beta_dict: Dict mapping bucket -> beta value
        """
        for bucket, value in alpha_dict.items():
            self.alpha[(player_id, bucket)] = value
        for bucket, value in beta_dict.items():
            self.beta[(player_id, bucket)] = value
        
        print(f"[OpponentModel] Loaded state for player {player_id[:8]}: {len(alpha_dict)} buckets")
    
    def get_state_for_player(self, player_id: str) -> Tuple[Dict, Dict]:
        """
        Get current state for a player (for persistence).
        
        Returns:
            Tuple of (alpha_dict, beta_dict) for the player
        """
        alpha_dict = {k[1]: v for k, v in self.alpha.items() if k[0] == player_id}
        beta_dict = {k[1]: v for k, v in self.beta.items() if k[0] == player_id}
        return alpha_dict, beta_dict
    
    def get_sample_count(
        self,
        player_id: str,
        deviation: Dict = None,
        personality_state: Optional[Dict[str, float]] = None,
        hand_strength_bucket: Optional[str] = None,
    ) -> int:
        """Get observation count for current context or total for player."""
        if deviation is not None:
            bucket = _state_bucket(deviation, personality_state, hand_strength_bucket)
            key = (player_id, bucket)
            return int(self.alpha[key] + self.beta[key]) - 2
        total = 0
        for k, v in self.alpha.items():
            if k[0] == player_id:
                total += int(v + self.beta[k]) - 2
        return total
    
    def get_accuracy(self) -> float:
        """Get overall prediction accuracy."""
        if self.predictions_made == 0:
            return 0.0
        return self.correct_predictions / self.predictions_made
    
    def reset_player(self, player_id: str):
        """Reset all learning for a specific player."""
        keys_to_remove = [k for k in self.alpha.keys() if k[0] == player_id]
        for key in keys_to_remove:
            del self.alpha[key]
            del self.beta[key]
        print(f"[OpponentModel] Reset player {player_id[:8]}")
    
    def get_context_summary(self, player_id: str) -> Dict[str, Dict]:
        """
        Get summary of learned parameters for all contexts.
        
        Returns:
            Dict mapping bucket -> {alpha, beta, p_bluff, samples}
        """
        summary = {}
        for key, alpha_val in self.alpha.items():
            if key[0] == player_id:
                bucket = key[1]
                beta_val = self.beta[key]
                p_bluff = alpha_val / (alpha_val + beta_val)
                samples = int(alpha_val + beta_val) - 2
                summary[bucket] = {
                    'alpha': alpha_val,
                    'beta': beta_val,
                    'p_bluff': p_bluff,
                    'samples': samples
                }
        return summary
