"""
Personality Inference Agent

Maintains a persistent, interpretable personality for each opponent inferred from
showdown history and physiology. Updated at each showdown; used to condition
the Prediction Agent. State: bluff base rate, stress-bluff correlation,
HR-bluff correlation, consistency.
"""

from collections import deque
from typing import Dict, Optional, Any
import math


class PersonalityModel:
    """
    Maintains per-opponent personality state from showdown outcomes and physiology.
    No heavy ML; online, sample-efficient updates (running averages, incremental regression).
    """

    DEFAULT_CONSISTENCY_WINDOW = 20  # Last N outcomes for consistency std

    def __init__(self):
        # In-memory state per player_id: personality dict + regression accumulators
        self._state: Dict[str, Dict[str, Any]] = {}

    def get_state(self, player_id: str) -> Dict[str, float]:
        """
        Get current personality state for the player (for Prediction Agent and UI).
        Returns a serializable dict with bluff_base_rate, stress_bluff_slope, hr_bluff_slope, consistency.
        """
        if player_id not in self._state:
            return self._default_state()
        s = self._state[player_id]
        n = s.get('_n', 0)
        if n == 0:
            return self._default_state()

        # Bluff base rate
        bluff_base_rate = s['_sum_bluff'] / n

        # Slopes from incremental regression: y = was_bluff (0/1), x = stress_delta or hr_delta
        stress_bluff_slope = self._slope(
            n, s.get('_sum_stress', 0), s.get('_sum_bluff', 0),
            s.get('_sum_stress_bluff', 0), s.get('_sum_stress_sq', 0)
        )
        hr_bluff_slope = self._slope(
            n, s.get('_sum_hr', 0), s.get('_sum_bluff', 0),
            s.get('_sum_hr_bluff', 0), s.get('_sum_hr_sq', 0)
        )

        # Consistency: 1 / (1 + std(recent bluff outcomes)); high when predictable
        recent = s.get('_recent_bluffs', [])
        if len(recent) >= 2:
            std_bluff = math.sqrt(sum((x - sum(recent) / len(recent)) ** 2 for x in recent) / len(recent))
            consistency = 1.0 / (1.0 + std_bluff)
        else:
            consistency = 0.5  # neutral

        return {
            'bluff_base_rate': float(bluff_base_rate),
            'stress_bluff_slope': float(stress_bluff_slope),
            'hr_bluff_slope': float(hr_bluff_slope),
            'consistency': float(consistency),
            'n_showdowns': n,
        }

    def _slope(self, n: float, sum_x: float, sum_y: float, sum_xy: float, sum_xx: float) -> float:
        """Incremental regression slope (n*sum_xy - sum_x*sum_y) / (n*sum_xx - sum_x^2)."""
        denom = n * sum_xx - sum_x * sum_x
        if denom <= 0:
            return 0.0
        return (n * sum_xy - sum_x * sum_y) / denom

    def _default_state(self) -> Dict[str, float]:
        return {
            'bluff_base_rate': 0.5,
            'stress_bluff_slope': 0.0,
            'hr_bluff_slope': 0.0,
            'consistency': 0.5,
            'n_showdowns': 0,
        }

    def update(self, player_id: str, deviation: Dict, was_bluffing: bool) -> None:
        """
        Update personality from one showdown.
        deviation: dict with 'hr_delta', 'stress_delta' (and optionally 'au_delta').
        was_bluffing: True if opponent was actually bluffing.
        """
        if player_id not in self._state:
            self._state[player_id] = {
                '_n': 0, '_sum_bluff': 0.0, '_sum_stress': 0.0, '_sum_hr': 0.0,
                '_sum_stress_bluff': 0.0, '_sum_hr_bluff': 0.0,
                '_sum_stress_sq': 0.0, '_sum_hr_sq': 0.0,
                '_recent_bluffs': deque(maxlen=self.DEFAULT_CONSISTENCY_WINDOW),
            }
        s = self._state[player_id]
        b = 1.0 if was_bluffing else 0.0
        stress_d = deviation.get('stress_delta', 0.0) or 0.0
        hr_d = deviation.get('hr_delta', 0.0) or 0.0

        n = s['_n'] + 1
        s['_n'] = n
        s['_sum_bluff'] += b
        s['_sum_stress'] += stress_d
        s['_sum_hr'] += hr_d
        s['_sum_stress_bluff'] += stress_d * b
        s['_sum_hr_bluff'] += hr_d * b
        s['_sum_stress_sq'] += stress_d * stress_d
        s['_sum_hr_sq'] += hr_d * hr_d
        s['_recent_bluffs'].append(b)

    def set_state(self, player_id: str, state_dict: Dict[str, Any]) -> None:
        """
        Load persisted state (e.g. from ProfileStore).
        state_dict can be the full internal state or the public get_state output.
        If it's the public output we need to reconstruct accumulators; for now we expect
        the serialized internal state from to_dict().
        """
        if '_n' in state_dict:
            self._state[player_id] = dict(state_dict)
            if '_recent_bluffs' in self._state[player_id] and not isinstance(
                self._state[player_id]['_recent_bluffs'], deque
            ):
                self._state[player_id]['_recent_bluffs'] = deque(
                    self._state[player_id]['_recent_bluffs'],
                    maxlen=self.DEFAULT_CONSISTENCY_WINDOW
                )
        else:
            # Public state only (n_showdowns, etc.) - cannot fully restore regression; reset
            self._state[player_id] = {
                '_n': int(state_dict.get('n_showdowns', 0)),
                '_sum_bluff': 0.0, '_sum_stress': 0.0, '_sum_hr': 0.0,
                '_sum_stress_bluff': 0.0, '_sum_hr_bluff': 0.0,
                '_sum_stress_sq': 0.0, '_sum_hr_sq': 0.0,
                '_recent_bluffs': deque(maxlen=self.DEFAULT_CONSISTENCY_WINDOW),
            }

    def to_dict(self, player_id: str) -> Optional[Dict[str, Any]]:
        """Serialize state for persistence. Returns None if player has no state."""
        if player_id not in self._state:
            return None
        s = self._state[player_id].copy()
        s['_recent_bluffs'] = list(s.get('_recent_bluffs', []))
        return s

    def from_dict(self, player_id: str, data: Optional[Dict[str, Any]]) -> None:
        """Restore state from persisted dict (from ProfileStore)."""
        if not data:
            return
        self.set_state(player_id, data)

    def get_profile_string(self, player_id: str) -> str:
        """Short interpretable profile for UI, e.g. 'bluffs more under stress'."""
        state = self.get_state(player_id)
        if state['n_showdowns'] < 3:
            return "Learning..."
        parts = []
        br = state['bluff_base_rate']
        if br > 0.6:
            parts.append("bluffs often")
        elif br < 0.4:
            parts.append("bluffs rarely")
        if state['stress_bluff_slope'] > 0.2:
            parts.append("bluffs more under stress")
        elif state['stress_bluff_slope'] < -0.2:
            parts.append("bluffs less when stressed")
        if state['consistency'] > 0.6:
            parts.append("fairly consistent")
        if not parts:
            return "Neutral"
        return "; ".join(parts)
