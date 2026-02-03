"""
Face Embedder Module

Extracts a 128-dimensional embedding from MediaPipe face landmarks
for player identification and profile matching.
"""

import numpy as np
from typing import List, Tuple, Optional
import hashlib


class FaceEmbedder:
    """
    Extracts face embeddings from MediaPipe landmarks for player identification.
    
    Uses geometric features (distances and ratios between key landmarks)
    to create a unique, scale-invariant representation of a face.
    
    Usage:
        embedder = FaceEmbedder()
        embedding = embedder.get_embedding(landmarks_468)
        similarity = embedder.similarity(emb1, emb2)
    """
    
    # Key landmark pairs for geometric features
    # Format: (landmark_idx_1, landmark_idx_2)
    # These pairs capture stable facial proportions
    LANDMARK_PAIRS = [
        # Eye measurements
        (33, 133),   # Left eye inner to outer corner
        (362, 263),  # Right eye inner to outer corner
        (33, 246),   # Left eye to eyebrow
        (362, 466),  # Right eye to eyebrow
        (159, 145),  # Left eye height
        (386, 374),  # Right eye height
        
        # Nose measurements
        (1, 4),      # Nose bridge length
        (2, 94),     # Nose width at nostrils
        (4, 6),      # Nose tip to base
        (168, 6),    # Nose bridge to tip
        
        # Mouth measurements
        (61, 291),   # Mouth width
        (0, 17),     # Upper lip to lower lip (mouth height)
        (13, 14),    # Upper lip thickness
        (78, 308),   # Mouth corner to corner
        
        # Face outline measurements
        (10, 152),   # Forehead to chin (face height)
        (234, 454),  # Face width at cheekbones
        (127, 356),  # Face width at jaw
        (93, 323),   # Ear to ear width
        
        # Cross-face measurements
        (33, 291),   # Left eye to right mouth corner
        (362, 61),   # Right eye to left mouth corner
        (1, 61),     # Nose to left mouth corner
        (1, 291),    # Nose to right mouth corner
        (10, 1),     # Forehead to nose bridge
        (152, 4),    # Chin to nose tip
        
        # Additional stability features
        (70, 300),   # Cheek landmarks
        (105, 334),  # Under-eye landmarks
        (107, 336),  # Upper cheek
        (187, 411),  # Lower face
        (132, 361),  # Temple region
        (172, 397),  # Jaw line
        
        # More cross-measurements for stability
        (33, 4),     # Left eye to nose tip
        (362, 4),    # Right eye to nose tip
        (61, 0),     # Left mouth to upper lip center
        (291, 0),    # Right mouth to upper lip center
        (234, 152),  # Left cheek to chin
        (454, 152),  # Right cheek to chin
        (10, 234),   # Forehead to left cheek
        (10, 454),   # Forehead to right cheek
        
        # Fine detail pairs
        (55, 285),   # Inner eyebrow points
        (8, 168),    # Nose landmarks
        (57, 287),   # Eye region
        (130, 359),  # Lower eye region
        (243, 463),  # Outer eye region
        (35, 265),   # Eyebrow outer
        (156, 383),  # Cheek inner
        (143, 372),  # Under eye
        (111, 340),  # Mid cheek
        (117, 346),  # Upper cheek region
        
        # Additional ratios
        (164, 393),  # Nose side to cheek
        (188, 412),  # Jaw region
        (210, 430),  # Outer face
        (135, 364),  # Face shape
        (138, 367),  # Face contour
        (58, 288),   # Eye region detail
        (172, 397),  # Lower jaw
        (176, 401),  # Jaw angle
        (148, 377),  # Chin region
        (149, 378),  # Lower face detail
    ]
    
    def __init__(self):
        self.embedding_dim = len(self.LANDMARK_PAIRS) * 2  # Distance + normalized ratio
        print(f"[FaceEmbedder] Initialized with {self.embedding_dim}-dim embeddings")
    
    def get_embedding(self, landmarks) -> Optional[np.ndarray]:
        """
        Extract embedding from MediaPipe face landmarks.
        
        Args:
            landmarks: List of 468 landmarks, each with x, y, z coordinates
                      Can be mediapipe NormalizedLandmarkList or list of dicts/tuples
                      
        Returns:
            Normalized embedding vector (embedding_dim,) or None if invalid
        """
        if landmarks is None or len(landmarks) < 468:
            return None
        
        try:
            # Convert landmarks to numpy array
            points = self._landmarks_to_array(landmarks)
            if points is None:
                return None
            
            # Extract features
            features = []
            
            # Reference distance for normalization (face height)
            face_height = np.linalg.norm(points[10] - points[152])
            if face_height < 0.01:  # Invalid face detection
                return None
            
            for i, j in self.LANDMARK_PAIRS:
                if i >= len(points) or j >= len(points):
                    continue
                    
                # Raw distance
                dist = np.linalg.norm(points[i] - points[j])
                features.append(dist)
                
                # Normalized distance (scale-invariant)
                normalized = dist / face_height
                features.append(normalized)
            
            if len(features) < 10:  # Not enough features
                return None
            
            # Convert to numpy and normalize to unit vector
            embedding = np.array(features, dtype=np.float32)
            norm = np.linalg.norm(embedding)
            
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            print(f"[FaceEmbedder] Error extracting embedding: {e}")
            return None
    
    def _landmarks_to_array(self, landmarks) -> Optional[np.ndarray]:
        """Convert various landmark formats to numpy array."""
        try:
            points = []
            
            for lm in landmarks:
                if hasattr(lm, 'x'):  # MediaPipe NormalizedLandmark
                    points.append([lm.x, lm.y, lm.z if hasattr(lm, 'z') else 0])
                elif isinstance(lm, dict):
                    points.append([lm.get('x', 0), lm.get('y', 0), lm.get('z', 0)])
                elif isinstance(lm, (list, tuple)):
                    if len(lm) >= 2:
                        points.append([lm[0], lm[1], lm[2] if len(lm) > 2 else 0])
                else:
                    continue
            
            if len(points) < 468:
                return None
                
            return np.array(points, dtype=np.float32)
            
        except Exception as e:
            print(f"[FaceEmbedder] Error converting landmarks: {e}")
            return None
    
    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            emb1, emb2: Normalized embedding vectors
            
        Returns:
            Similarity score between 0 and 1 (1 = identical)
        """
        if emb1 is None or emb2 is None:
            return 0.0
        
        # Cosine similarity (embeddings are already normalized)
        return float(np.dot(emb1, emb2))
    
    def get_player_id(self, embedding: np.ndarray) -> str:
        """
        Generate a unique player ID from an embedding.
        
        Args:
            embedding: Face embedding vector
            
        Returns:
            Hex string identifier (first 16 chars of hash)
        """
        if embedding is None:
            return "unknown"
        
        # Hash the embedding for a stable ID
        emb_bytes = embedding.tobytes()
        hash_obj = hashlib.sha256(emb_bytes)
        return hash_obj.hexdigest()[:16]
    
    def is_same_person(self, emb1: np.ndarray, emb2: np.ndarray, threshold: float = 0.85) -> bool:
        """
        Check if two embeddings represent the same person.
        
        Args:
            emb1, emb2: Face embeddings
            threshold: Similarity threshold (default: 0.85)
            
        Returns:
            True if embeddings are similar enough
        """
        return self.similarity(emb1, emb2) >= threshold
