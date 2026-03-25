"""
Profile Store Module

SQLite-based persistence for player profiles, including:
- Face embeddings
- Baseline physiological data
- Bandit learning state (alpha/beta parameters)
"""

import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import time


class ProfileStore:
    """
    SQLite storage for player profiles and learning state.
    
    Usage:
        store = ProfileStore()
        
        # Find or create player
        player_id = store.find_or_create_player(embedding)
        
        # Save/load baseline
        store.save_baseline(player_id, baseline_dict)
        baseline = store.load_baseline(player_id)
        
        # Save/load bandit state
        store.save_bandit_state(player_id, alpha_dict, beta_dict)
        alpha, beta = store.load_bandit_state(player_id)
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize the profile store.
        
        Args:
            db_path: Path to SQLite database. If None, uses default location.
        """
        if db_path is None:
            db_path = Path(__file__).parent / 'profiles.db'
        
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._create_tables()
        print(f"[ProfileStore] Database initialized at {self.db_path}")
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Players table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS players (
                player_id TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                first_seen REAL NOT NULL,
                last_seen REAL NOT NULL,
                session_count INTEGER DEFAULT 1,
                showdown_count INTEGER DEFAULT 0
            )
        ''')
        
        # Baselines table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS baselines (
                player_id TEXT PRIMARY KEY,
                baseline_data TEXT NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY (player_id) REFERENCES players(player_id)
            )
        ''')
        
        # Bandit state table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bandit_state (
                player_id TEXT PRIMARY KEY,
                alpha_data TEXT NOT NULL,
                beta_data TEXT NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY (player_id) REFERENCES players(player_id)
            )
        ''')
        
        # Showdown history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS showdowns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id TEXT NOT NULL,
                context_bucket TEXT NOT NULL,
                prediction TEXT NOT NULL,
                actual_result TEXT NOT NULL,
                was_correct INTEGER NOT NULL,
                deviation_data TEXT,
                frames_sampled INTEGER DEFAULT 0,
                hr_variance REAL DEFAULT 0,
                stress_variance REAL DEFAULT 0,
                peak_frame_pct REAL DEFAULT 0,
                timestamp REAL NOT NULL,
                FOREIGN KEY (player_id) REFERENCES players(player_id)
            )
        ''')
        # Migrate existing DB: add columns if they don't exist yet
        for col, typ in [('frames_sampled', 'INTEGER DEFAULT 0'),
                         ('hr_variance',    'REAL DEFAULT 0'),
                         ('stress_variance','REAL DEFAULT 0'),
                         ('peak_frame_pct', 'REAL DEFAULT 0')]:
            try:
                cursor.execute(f'ALTER TABLE showdowns ADD COLUMN {col} {typ}')
            except sqlite3.OperationalError:
                pass  # column already exists
        
        # Personality state table (learning layer: bluff base rate, stress/hr-bluff slopes, consistency)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS personality_state (
                player_id TEXT PRIMARY KEY,
                state_data TEXT NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY (player_id) REFERENCES players(player_id)
            )
        ''')
        
        self.conn.commit()
    
    def find_or_create_player(self, embedding: np.ndarray, similarity_threshold: float = 0.85) -> str:
        """
        Find existing player by embedding similarity or create new.
        
        Args:
            embedding: Face embedding vector
            similarity_threshold: Minimum similarity to match (default: 0.85)
            
        Returns:
            Player ID (existing or newly created)
        """
        # Check all existing players
        cursor = self.conn.cursor()
        cursor.execute('SELECT player_id, embedding FROM players')
        
        for row in cursor.fetchall():
            stored_id, stored_emb_bytes = row
            stored_emb = np.frombuffer(stored_emb_bytes, dtype=np.float32)
            
            similarity = np.dot(embedding, stored_emb)
            if similarity >= similarity_threshold:
                # Update last seen
                cursor.execute(
                    'UPDATE players SET last_seen = ?, session_count = session_count + 1 WHERE player_id = ?',
                    (time.time(), stored_id)
                )
                self.conn.commit()
                print(f"[ProfileStore] Matched existing player: {stored_id[:8]}... (similarity: {similarity:.3f})")
                return stored_id
        
        # Create new player
        return self._create_player(embedding)
    
    def _create_player(self, embedding: np.ndarray) -> str:
        """Create a new player entry."""
        import hashlib
        
        player_id = hashlib.sha256(embedding.tobytes()).hexdigest()[:16]
        now = time.time()
        
        cursor = self.conn.cursor()
        cursor.execute(
            'INSERT INTO players (player_id, embedding, first_seen, last_seen) VALUES (?, ?, ?, ?)',
            (player_id, embedding.tobytes(), now, now)
        )
        self.conn.commit()
        
        print(f"[ProfileStore] Created new player: {player_id}")
        return player_id
    
    def save_baseline(self, player_id: str, baseline: Dict):
        """Save baseline data for a player."""
        cursor = self.conn.cursor()
        
        baseline_json = json.dumps(baseline)
        
        cursor.execute('''
            INSERT OR REPLACE INTO baselines (player_id, baseline_data, created_at)
            VALUES (?, ?, ?)
        ''', (player_id, baseline_json, time.time()))
        
        self.conn.commit()
    
    def load_baseline(self, player_id: str) -> Optional[Dict]:
        """Load baseline data for a player."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT baseline_data FROM baselines WHERE player_id = ?', (player_id,))
        
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None
    
    def save_bandit_state(self, player_id: str, alpha: Dict, beta: Dict):
        """
        Save bandit learning state (alpha/beta parameters).
        
        Args:
            player_id: Player identifier
            alpha: Dict mapping (player_id, bucket) keys to alpha values
            beta: Dict mapping (player_id, bucket) keys to beta values
        """
        cursor = self.conn.cursor()
        
        # Filter to only this player's state and serialize
        player_alpha = {k[1]: v for k, v in alpha.items() if k[0] == player_id}
        player_beta = {k[1]: v for k, v in beta.items() if k[0] == player_id}
        
        alpha_json = json.dumps(player_alpha)
        beta_json = json.dumps(player_beta)
        
        cursor.execute('''
            INSERT OR REPLACE INTO bandit_state (player_id, alpha_data, beta_data, updated_at)
            VALUES (?, ?, ?, ?)
        ''', (player_id, alpha_json, beta_json, time.time()))
        
        self.conn.commit()
    
    def load_bandit_state(self, player_id: str) -> Tuple[Dict, Dict]:
        """
        Load bandit learning state for a player.
        
        Returns:
            Tuple of (alpha_dict, beta_dict) mapping bucket -> value
        """
        cursor = self.conn.cursor()
        cursor.execute('SELECT alpha_data, beta_data FROM bandit_state WHERE player_id = ?', (player_id,))
        
        row = cursor.fetchone()
        if row:
            alpha = json.loads(row[0])
            beta = json.loads(row[1])
            return alpha, beta
        return {}, {}
    
    def save_personality_state(self, player_id: str, state_data: Dict) -> None:
        """Save personality state for a player (from PersonalityModel.to_dict)."""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO personality_state (player_id, state_data, updated_at)
            VALUES (?, ?, ?)
        ''', (player_id, json.dumps(state_data), time.time()))
        self.conn.commit()
    
    def load_personality_state(self, player_id: str) -> Optional[Dict]:
        """Load personality state for a player. Returns None if not found."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT state_data FROM personality_state WHERE player_id = ?', (player_id,))
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None
    
    def log_showdown(self, player_id: str, context_bucket: str, prediction: str,
                     actual: str, was_correct: bool, deviation: Dict = None):
        """Log a showdown result for analysis."""
        cursor = self.conn.cursor()

        frames   = deviation.get('frames_sampled', 0)    if deviation else 0
        hr_var   = deviation.get('hr_variance',    0.0)  if deviation else 0.0
        st_var   = deviation.get('stress_variance',0.0)  if deviation else 0.0
        pk_pct   = deviation.get('peak_frame_pct', 0.0)  if deviation else 0.0

        cursor.execute('''
            INSERT INTO showdowns (player_id, context_bucket, prediction, actual_result,
                                   was_correct, deviation_data,
                                   frames_sampled, hr_variance, stress_variance, peak_frame_pct,
                                   timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (player_id, context_bucket, prediction, actual,
              1 if was_correct else 0, json.dumps(deviation) if deviation else None,
              frames, hr_var, st_var, pk_pct, time.time()))
        
        # Update showdown count
        cursor.execute(
            'UPDATE players SET showdown_count = showdown_count + 1 WHERE player_id = ?',
            (player_id,)
        )
        
        self.conn.commit()
    
    def get_player_stats(self, player_id: str) -> Dict:
        """Get statistics for a player."""
        cursor = self.conn.cursor()
        
        # Basic info
        cursor.execute('SELECT session_count, showdown_count, first_seen, last_seen FROM players WHERE player_id = ?', 
                      (player_id,))
        row = cursor.fetchone()
        
        if not row:
            return {}
        
        # Showdown accuracy
        cursor.execute('SELECT COUNT(*), SUM(was_correct) FROM showdowns WHERE player_id = ?', (player_id,))
        showdown_row = cursor.fetchone()
        
        total_showdowns = showdown_row[0] or 0
        correct_predictions = showdown_row[1] or 0
        
        return {
            'player_id': player_id,
            'sessions': row[0],
            'showdowns': row[1],
            'first_seen': row[2],
            'last_seen': row[3],
            'prediction_accuracy': correct_predictions / total_showdowns if total_showdowns > 0 else 0.0
        }
    
    def get_all_players(self) -> List[Dict]:
        """Get list of all known players."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT player_id, session_count, showdown_count, last_seen FROM players ORDER BY last_seen DESC')
        
        return [
            {'player_id': row[0], 'sessions': row[1], 'showdowns': row[2], 'last_seen': row[3]}
            for row in cursor.fetchall()
        ]
    
    def delete_player(self, player_id: str):
        """Delete a player and all associated data."""
        cursor = self.conn.cursor()
        
        cursor.execute('DELETE FROM showdowns WHERE player_id = ?', (player_id,))
        cursor.execute('DELETE FROM bandit_state WHERE player_id = ?', (player_id,))
        cursor.execute('DELETE FROM personality_state WHERE player_id = ?', (player_id,))
        cursor.execute('DELETE FROM baselines WHERE player_id = ?', (player_id,))
        cursor.execute('DELETE FROM players WHERE player_id = ?', (player_id,))
        
        self.conn.commit()
        print(f"[ProfileStore] Deleted player: {player_id}")
    
    def delete_all_profiles(self):
        """Delete all player profiles (privacy reset)."""
        cursor = self.conn.cursor()
        
        cursor.execute('DELETE FROM showdowns')
        cursor.execute('DELETE FROM bandit_state')
        cursor.execute('DELETE FROM personality_state')
        cursor.execute('DELETE FROM baselines')
        cursor.execute('DELETE FROM players')
        
        self.conn.commit()
        print("[ProfileStore] All profiles deleted")
    
    def close(self):
        """Close database connection."""
        self.conn.close()
