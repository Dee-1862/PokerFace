# Adaptive Learning Package (Multi-Agent Learning Layer)
"""
Identity + Personality + Prediction agents; single source of truth: ProfileStore + FaceEmbedder.

Modules:
- baseline_extractor: Per-opponent baseline; context-adjusted deviation (v2)
- face_embedder: Face embeddings for player identification
- profile_store: SQLite persistence (baselines, personality_state, bandit_state, showdowns)
- personality_model: Personality Inference Agent (bluff base rate, stress/hr-bluff slopes, consistency)
- opponent_model: Prediction RL Agent (Thompson Sampling, personality + hand bucket in state)
- prior_generator: Cold-start priors
- adaptive_learning_system: Orchestrator (identity → personality → prediction; showdown updates)
"""

from .baseline_extractor import BaselineExtractor
from .face_embedder import FaceEmbedder
from .profile_store import ProfileStore
from .personality_model import PersonalityModel
from .opponent_model import OpponentModel
from .prior_generator import SimplePriorGenerator, PriorGenerator
from .adaptive_learning_system import AdaptiveLearningSystem
from .panel_positions import PanelPositionManager

__all__ = [
    'BaselineExtractor',
    'FaceEmbedder',
    'ProfileStore',
    'PersonalityModel',
    'OpponentModel',
    'SimplePriorGenerator',
    'PriorGenerator',
    'AdaptiveLearningSystem',
    'PanelPositionManager',
]

