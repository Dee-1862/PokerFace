"""Persistent panel position manager for draggable AR panels."""
import json
from pathlib import Path
from typing import Dict, Tuple


class PanelPositionManager:
    """
    Stores panel (x, y) positions and persists them to disk.
    Renderers call get() for their position; drag logic calls set() + save().
    """
    _SAVE_FILE = Path(__file__).parent / 'panel_positions.json'

    def __init__(self):
        self._positions: Dict[str, list] = {}
        self._load()

    def get(self, name: str, default_x: int, default_y: int) -> Tuple[int, int]:
        """Return saved position, or default if panel has never been moved."""
        if name in self._positions:
            return (int(self._positions[name][0]), int(self._positions[name][1]))
        return (default_x, default_y)

    def set(self, name: str, x: int, y: int) -> None:
        """Update a panel position in memory (call save() to persist)."""
        self._positions[name] = [int(x), int(y)]

    def reset(self, name: str) -> None:
        """Remove override for one panel so it reverts to its default position."""
        self._positions.pop(name, None)
        self._save()

    def reset_all(self) -> None:
        """Remove all overrides."""
        self._positions.clear()
        self._save()

    def save(self) -> None:
        self._save()

    def _load(self) -> None:
        try:
            if self._SAVE_FILE.exists():
                self._positions = json.loads(
                    self._SAVE_FILE.read_text(encoding='utf-8')
                )
        except Exception:
            self._positions = {}

    def _save(self) -> None:
        try:
            self._SAVE_FILE.write_text(
                json.dumps(self._positions, indent=2), encoding='utf-8'
            )
        except Exception as e:
            print(f"[PanelPositions] Save failed: {e}")
