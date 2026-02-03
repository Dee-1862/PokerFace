"""
Adaptive Learning Integration Module

High-level API for integrating adaptive learning into the unified AR system.
Coordinates all components: baseline, embeddings, profiles, and opponent model.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import time

from .baseline_extractor import BaselineExtractor
from .face_embedder import FaceEmbedder
from .profile_store import ProfileStore
from .opponent_model import OpponentModel
from .prior_generator import SimplePriorGenerator


class AdaptiveLearningSystem:
    """
    High-level coordinator for the adaptive opponent learning system.
    
    This class integrates all components and provides a simple API for
    the unified AR system to use.
    
    Usage:
        # Initialize once
        learner = AdaptiveLearningSystem()
        
        # When face detected (every frame):
        learner.on_face_detected(landmarks)
        
        # Add signal samples for baseline:
        learner.add_calibration_sample(hr, stress, au_values)
        
        # Get prediction:
        prediction = learner.get_prediction(current_hr, current_stress, current_au)
        # Returns: {'prediction': 'BLUFFING', 'confidence': 0.72, 'p_bluff': 0.65}
        
        # After showdown:
        learner.on_showdown(was_bluffing=True)
    """
    
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
        
        # Use simple priors by default (no heavy dependencies)
        prior_gen = SimplePriorGenerator() if not use_slm else None
        self.model = OpponentModel(prior_generator=prior_gen)
        
        # Current state
        self.current_player_id = None
        self.current_deviation = None
        self.last_prediction = None
        self.is_tracking = False
        
        print("[AdaptiveLearning] System initialized")
    
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
        
        # If new player, load their saved state
        if player_id != self.current_player_id:
            self._load_player_state(player_id)
            self.current_player_id = player_id
            self.is_tracking = True
        
        return player_id
    
    def _load_player_state(self, player_id: str):
        """Load saved state for a player."""
        # Load baseline
        saved_baseline = self.profiles.load_baseline(player_id)
        if saved_baseline:
            self.baseline.from_dict(saved_baseline)
            print(f"[AdaptiveLearning] Loaded baseline for {player_id[:8]}")
        else:
            self.baseline.reset()
        
        # Load bandit state
        alpha, beta = self.profiles.load_bandit_state(player_id)
        if alpha:
            self.model.load_state(player_id, alpha, beta)
            print(f"[AdaptiveLearning] Loaded bandit state for {player_id[:8]}")
        else:
            # Initialize with priors for new player
            self.model.initialize_for_new_player(player_id, {
                'baseline_hr': saved_baseline.get('hr', 70) if saved_baseline else 70
            })
    
    def start_calibration(self, duration_frames: int = 90):
        """Start baseline calibration (3 seconds at 30fps)."""
        self.baseline.start_calibration(duration_frames)
    
    def add_calibration_sample(self, hr: float, stress: float, au_values: Dict):
        """Add a sample during calibration."""
        self.baseline.add_sample(hr, stress, au_values)
        
        # Auto-save when calibration completes
        if not self.baseline.is_calibrating and self.baseline.has_baseline():
            if self.current_player_id:
                self.profiles.save_baseline(
                    self.current_player_id, 
                    self.baseline.to_dict()
                )
    
    def get_prediction(self, hr: float, stress: float, au_values: Dict = None) -> Optional[Dict]:
        """
        Get prediction for current player state.
        
        Args:
            hr: Current heart rate
            stress: Current stress level
            au_values: Current Action Unit values
            
        Returns:
            Dict with 'prediction', 'confidence', 'p_bluff', 'samples'
            or None if not ready
        """
        if not self.current_player_id or not self.baseline.has_baseline():
            return None
        
        # Compute deviations
        current_state = {
            'hr': hr,
            'stress': stress,
            'au': au_values or {}
        }
        self.current_deviation = self.baseline.get_deviation(current_state)
        
        if not self.current_deviation:
            return None
        
        # Get prediction
        prediction, confidence, p_bluff = self.model.predict(
            self.current_player_id, 
            self.current_deviation
        )
        
        self.last_prediction = prediction
        
        # Get sample count for this context
        samples = self.model.get_sample_count(self.current_player_id, self.current_deviation)
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'p_bluff': p_bluff,
            'samples': samples,
            'hr_delta': self.current_deviation.get('hr_delta', 0),
            'stress_delta': self.current_deviation.get('stress_delta', 0)
        }
    
    def on_showdown(self, was_bluffing: bool) -> bool:
        """
        Called after showdown to update the model.
        
        Args:
            was_bluffing: True if player was actually bluffing
            
        Returns:
            True if the prediction was correct
        """
        if not self.current_player_id or not self.current_deviation:
            print("[AdaptiveLearning] Cannot update: no current player or deviation")
            return False
        
        # Update model
        was_correct = self.model.update_with_prediction_result(
            self.current_player_id,
            self.current_deviation,
            self.last_prediction or "STRONG",
            was_bluffing
        )
        
        # Save updated state
        alpha, beta = self.model.get_state_for_player(self.current_player_id)
        self.profiles.save_bandit_state(self.current_player_id, 
                                        {(self.current_player_id, k): v for k, v in alpha.items()},
                                        {(self.current_player_id, k): v for k, v in beta.items()})
        
        # Log showdown
        context = self.model._get_context_bucket(self.current_deviation)
        self.profiles.log_showdown(
            self.current_player_id,
            context,
            self.last_prediction or "STRONG",
            "BLUFFING" if was_bluffing else "STRONG",
            was_correct,
            self.current_deviation
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
    
    def reset_current_player(self):
        """Reset learning for current player."""
        if self.current_player_id:
            self.model.reset_player(self.current_player_id)
            self.profiles.delete_player(self.current_player_id)
            self.current_player_id = None
            self.baseline.reset()
    
    def delete_all_profiles(self):
        """Delete all player profiles (privacy reset)."""
        self.profiles.delete_all_profiles()
        self.current_player_id = None
        self.baseline.reset()
        print("[AdaptiveLearning] All profiles deleted")
    
    def close(self):
        """Clean up resources."""
        self.profiles.close()
