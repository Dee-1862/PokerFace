"""
Baseline Extractor Module

Captures a player's neutral physiological state before cards are dealt.
This baseline is used to compute deviations during gameplay.
"""

import numpy as np
from collections import defaultdict
import time


class BaselineExtractor:
    """
    Captures and stores a player's baseline physiological signals.
    
    The baseline represents the player's "neutral" state and is used
    to compute deviations during gameplay, which are more informative
    than absolute values.
    
    Usage:
        extractor = BaselineExtractor()
        extractor.start_calibration()
        
        # During calibration period (3 seconds):
        while extractor.is_calibrating:
            extractor.add_sample(hr=75, stress=0.3, au_values={'AU12': 0.1, ...})
        
        # After calibration:
        deviation = extractor.get_deviation({'hr': 85, 'stress': 0.5, 'au': {...}})
        # Returns: {'hr_delta': 10, 'stress_delta': 0.2, 'au_delta': {...}}
    """
    
    DEFAULT_CALIBRATION_FRAMES = 90  # ~3 seconds at 30fps
    
    def __init__(self):
        self.samples = []
        self.baseline = None
        self.is_calibrating = False
        self.target_samples = self.DEFAULT_CALIBRATION_FRAMES
        self.calibration_start_time = None
        
    def start_calibration(self, duration_frames=None):
        """
        Begin baseline calibration period.
        
        Args:
            duration_frames: Number of frames to collect (default: 90 = 3 seconds)
        """
        self.samples = []
        self.baseline = None
        self.is_calibrating = True
        self.target_samples = duration_frames or self.DEFAULT_CALIBRATION_FRAMES
        self.calibration_start_time = time.time()
        print(f"[Baseline] Calibration started - collecting {self.target_samples} samples...")
        
    def add_sample(self, hr, stress, au_values):
        """
        Add a sample during calibration.
        
        Args:
            hr: Heart rate in BPM
            stress: Stress level (0.0 - 1.0)
            au_values: Dictionary of Action Unit values {'AU12': 0.5, 'AU6': 0.3, ...}
        """
        if not self.is_calibrating:
            return
            
        self.samples.append({
            'hr': hr if hr and hr > 0 else 70,  # Default HR if not available
            'stress': stress if stress else 0.0,
            'au': au_values.copy() if au_values else {},
            'timestamp': time.time()
        })
        
        # Check if we have enough samples
        if len(self.samples) >= self.target_samples:
            self._finalize_calibration()
    
    def _finalize_calibration(self):
        """Compute baseline from collected samples."""
        if len(self.samples) < 10:  # Minimum samples needed
            print("[Baseline] Not enough samples, calibration failed")
            self.is_calibrating = False
            return
        
        # Compute mean values for baseline
        valid_hr = [s['hr'] for s in self.samples if s['hr'] and s['hr'] > 0]
        
        self.baseline = {
            'hr': np.mean(valid_hr) if valid_hr else 70.0,
            'stress': np.mean([s['stress'] for s in self.samples]),
            'au': self._compute_au_baseline(),
            'sample_count': len(self.samples),
            'calibration_time': time.time() - self.calibration_start_time
        }
        
        self.is_calibrating = False
        print(f"[Baseline] Calibration complete:")
        print(f"  HR baseline: {self.baseline['hr']:.1f} BPM")
        print(f"  Stress baseline: {self.baseline['stress']:.2f}")
        print(f"  AU features: {len(self.baseline['au'])} tracked")
        
    def _compute_au_baseline(self):
        """Compute baseline for all Action Units."""
        au_baseline = defaultdict(list)
        
        for sample in self.samples:
            for au_name, value in sample['au'].items():
                au_baseline[au_name].append(value)
        
        return {au: np.mean(values) for au, values in au_baseline.items()}
    
    def get_deviation(self, current, context=None):
        """
        Compute deviation from baseline. Optionally context-adjusted when v2 provides context.
        
        Args:
            current: Dict with 'hr', 'stress', 'au' keys
            context: Optional dict e.g. {'pot_size': 100, 'street': 'river'} for context-adjusted
                     expectation. When provided, returned deltas can be adjusted by expected delta
                     for this context (v2: expected_hr_delta, expected_stress_delta per context).
                     For now, context is ignored; v2 pipeline will implement lookup/subtraction.
            
        Returns:
            Dict with 'hr_delta', 'stress_delta', 'au_delta' keys or None if no baseline
        """
        if not self.baseline:
            return None
            
        current_hr = current.get('hr', self.baseline['hr'])
        current_stress = current.get('stress', self.baseline['stress'])
        current_au = current.get('au', {})
        
        hr_delta = current_hr - self.baseline['hr']
        stress_delta = current_stress - self.baseline['stress']
        
        # Context-adjusted: subtract expected delta for this context (v2 implements per-context expectations)
        if context:
            expected_hr = self._expected_hr_delta_for_context(context)
            expected_stress = self._expected_stress_delta_for_context(context)
            hr_delta = hr_delta - expected_hr
            stress_delta = stress_delta - expected_stress
        
        au_delta = {}
        for au_name, baseline_val in self.baseline['au'].items():
            current_val = current_au.get(au_name, baseline_val)
            au_delta[au_name] = current_val - baseline_val
        
        return {
            'hr_delta': hr_delta,
            'stress_delta': stress_delta,
            'au_delta': au_delta
        }
    
    def _expected_hr_delta_for_context(self, context):
        """Expected HR delta for context (pot_size, street). v2: populate from session stats or regression."""
        return 0.0
    
    def _expected_stress_delta_for_context(self, context):
        """Expected stress delta for context. v2: populate from session stats."""
        return 0.0
    
    def get_calibration_progress(self):
        """Get calibration progress as percentage (0-100)."""
        if not self.is_calibrating:
            return 100 if self.baseline else 0
        return min(100, int(len(self.samples) / self.target_samples * 100))
    
    def reset(self):
        """Reset the extractor to initial state."""
        self.samples = []
        self.baseline = None
        self.is_calibrating = False
        self.calibration_start_time = None
        
    def has_baseline(self):
        """Check if a valid baseline exists."""
        return self.baseline is not None
    
    def to_dict(self):
        """Serialize baseline for storage."""
        if not self.baseline:
            return None
        return {
            'hr': float(self.baseline['hr']),
            'stress': float(self.baseline['stress']),
            'au': {k: float(v) for k, v in self.baseline['au'].items()},
            'sample_count': self.baseline['sample_count'],
            'calibration_time': self.baseline['calibration_time']
        }
    
    def from_dict(self, data):
        """Load baseline from stored data."""
        if data:
            self.baseline = {
                'hr': data['hr'],
                'stress': data['stress'],
                'au': data['au'],
                'sample_count': data.get('sample_count', 0),
                'calibration_time': data.get('calibration_time', 0)
            }
            self.is_calibrating = False
