# Adaptive Learning Package
"""
Adaptive Opponent Learning System (AOLS)
Option B: Hybrid SLM + Contextual Bandit

Modules:
- baseline_extractor: Captures player's neutral state
- face_embedder: Extracts face embeddings for player identification
- profile_store: SQLite persistence for player profiles
- opponent_model: Thompson Sampling contextual bandit
- prior_generator: SLM-based prior generation for cold-start
- adaptive_learning_system: High-level integration API
"""

from .baseline_extractor import BaselineExtractor
from .face_embedder import FaceEmbedder
from .profile_store import ProfileStore
from .opponent_model import OpponentModel
from .prior_generator import SimplePriorGenerator, PriorGenerator
from .adaptive_learning_system import AdaptiveLearningSystem

__all__ = [
    'BaselineExtractor',
    'FaceEmbedder', 
    'ProfileStore',
    'OpponentModel',
    'SimplePriorGenerator',
    'PriorGenerator',
    'AdaptiveLearningSystem'
]

