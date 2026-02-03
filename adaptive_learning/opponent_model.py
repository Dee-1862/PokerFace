"""
Opponent Model Module

Thompson Sampling contextual bandit for predicting player behavior.
Uses Beta distributions to model probability of bluffing in each context.
"""

import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, Optional
import time


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
        
    def predict(self, player_id: str, deviation: Dict) -> Tuple[str, float, float]:
        """
        Make a prediction about whether the player is bluffing.
        
        Args:
            player_id: Unique player identifier
            deviation: Dict with 'hr_delta', 'stress_delta', 'au_delta'
            
        Returns:
            Tuple of (prediction, confidence, p_bluff)
            - prediction: "BLUFFING" or "STRONG"
            - confidence: 0.0 to 1.0
            - p_bluff: Raw probability of bluffing
        """
        bucket = self._get_context_bucket(deviation)
        key = (player_id, bucket)
        
        # Thompson Sampling: sample from Beta posterior
        p_bluff = np.random.beta(self.alpha[key], self.beta[key])
        
        # Prediction
        prediction = "BLUFFING" if p_bluff > 0.5 else "STRONG"
        
        # Confidence: how far from 0.5 (uncertainty)
        confidence = abs(p_bluff - 0.5) * 2  # Scale to 0-1
        
        self.predictions_made += 1
        
        return prediction, confidence, p_bluff
    
    def get_expected_probability(self, player_id: str, deviation: Dict) -> float:
        """
        Get expected probability of bluffing (without sampling).
        
        Uses the mean of the Beta distribution instead of sampling.
        Useful for displaying stable UI values.
        """
        bucket = self._get_context_bucket(deviation)
        key = (player_id, bucket)
        
        # Mean of Beta distribution = alpha / (alpha + beta)
        return self.alpha[key] / (self.alpha[key] + self.beta[key])
    
    def update(self, player_id: str, deviation: Dict, was_bluffing: bool):
        """
        Update the model after a showdown.
        
        Args:
            player_id: Player identifier
            deviation: Deviation dict that was active during prediction
            was_bluffing: True if player was actually bluffing
        """
        bucket = self._get_context_bucket(deviation)
        key = (player_id, bucket)
        
        if was_bluffing:
            # Player was bluffing in this context -> increase alpha
            self.alpha[key] += 1
        else:
            # Player was not bluffing -> increase beta
            self.beta[key] += 1
        
        print(f"[OpponentModel] Updated {bucket}: alpha={self.alpha[key]:.1f}, beta={self.beta[key]:.1f}")
    
    def update_with_prediction_result(self, player_id: str, deviation: Dict, 
                                       prediction: str, was_bluffing: bool):
        """
        Update the model with explicit prediction tracking.
        
        Args:
            player_id: Player identifier
            deviation: Deviation dict
            prediction: The prediction that was made ("BLUFFING" or "STRONG")
            was_bluffing: True if player was actually bluffing
        """
        was_correct = (prediction == "BLUFFING") == was_bluffing
        if was_correct:
            self.correct_predictions += 1
        
        self.update(player_id, deviation, was_bluffing)
        
        return was_correct
    
    def _get_context_bucket(self, deviation: Dict) -> str:
        """
        Convert continuous deviations to discrete bucket.
        
        Format: "{HR_bucket}{STRESS_bucket}" e.g., "HH", "ML", "LM"
        """
        if not deviation:
            return "MM"  # Default to medium
        
        hr_delta = deviation.get('hr_delta', 0)
        stress_delta = deviation.get('stress_delta', 0)
        
        # HR bucket
        if hr_delta > self.HR_HIGH_THRESHOLD:
            hr_bucket = "H"
        elif hr_delta < self.HR_LOW_THRESHOLD:
            hr_bucket = "L"
        else:
            hr_bucket = "M"
        
        # Stress bucket
        if stress_delta > self.STRESS_HIGH_THRESHOLD:
            stress_bucket = "H"
        elif stress_delta < self.STRESS_LOW_THRESHOLD:
            stress_bucket = "L"
        else:
            stress_bucket = "M"
        
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
    
    def get_sample_count(self, player_id: str, deviation: Dict = None) -> int:
        """
        Get total observations for a player in current context.
        
        Returns:
            Number of showdowns observed in this context
        """
        if deviation:
            bucket = self._get_context_bucket(deviation)
            key = (player_id, bucket)
            # Total observations = alpha + beta - 2 (prior)
            return int(self.alpha[key] + self.beta[key]) - 2
        else:
            # Total across all contexts
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
