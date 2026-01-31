"""
Advanced Stress Classification Module
Provides 5-level classification and multi-dimensional stress analysis.
"""
import numpy as np
from collections import deque
from enum import Enum


class StressLevel(Enum):
    """5-level stress classification."""
    VERY_LOW = "Very Low"
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    VERY_HIGH = "Very High"


class StressType(Enum):
    """Type of stress pattern."""
    ACUTE = "Acute"  # Sudden spike
    CHRONIC = "Chronic"  # Sustained elevation
    NONE = "None"


class StressTrend(Enum):
    """Trend direction."""
    INCREASING = "Increasing"
    DECREASING = "Decreasing"
    STABLE = "Stable"
    FLUCTUATING = "Fluctuating"


class StressClassifier:
    """
    Advanced stress classification with 5-level categorization
    and multi-dimensional analysis.
    """
    
    # 5-level thresholds
    THRESHOLD_VERY_LOW = 0.20   # 0-20%
    THRESHOLD_LOW = 0.40         # 20-40%
    THRESHOLD_MODERATE = 0.60    # 40-60%
    THRESHOLD_HIGH = 0.80        # 60-80%
    # 80-100% = Very High
    
    # Trend detection parameters
    TREND_WINDOW_SIZE = 30  # Frames for trend analysis (~1 second at 30fps)
    TREND_SLOPE_THRESHOLD = 0.01  # Minimum slope for trend detection
    
    # Event detection parameters
    EVENT_THRESHOLD = 0.70  # Stress score threshold for event detection
    EVENT_MIN_DURATION = 5  # Minimum frames for event (frames)
    
    def __init__(self):
        """Initialize stress classifier."""
        self.stress_history = deque(maxlen=300)  # ~10 seconds at 30fps
        self.timestamp_history = deque(maxlen=300)
        self.detected_events = []
        self.last_event_end = -1
    
    def classify_stress_level(self, stress_score):
        """
        Classify stress score into 5-level category.
        
        Args:
            stress_score: Stress score (0.0 to 1.0)
        
        Returns:
            StressLevel enum value
        """
        if stress_score < self.THRESHOLD_VERY_LOW:
            return StressLevel.VERY_LOW
        elif stress_score < self.THRESHOLD_LOW:
            return StressLevel.LOW
        elif stress_score < self.THRESHOLD_MODERATE:
            return StressLevel.MODERATE
        elif stress_score < self.THRESHOLD_HIGH:
            return StressLevel.HIGH
        else:
            return StressLevel.VERY_HIGH
    
    def detect_stress_trend(self, current_stress):
        """
        Detect stress trend over recent history.
        
        Args:
            current_stress: Current stress score
        
        Returns:
            StressTrend enum value
        """
        self.stress_history.append(current_stress)
        
        if len(self.stress_history) < self.TREND_WINDOW_SIZE:
            return StressTrend.STABLE
        
        # Get recent window
        recent = list(self.stress_history)[-self.TREND_WINDOW_SIZE:]
        
        # Calculate linear regression slope
        x = np.arange(len(recent))
        y = np.array(recent)
        
        # Simple linear regression
        if len(x) > 1 and np.std(x) > 0:
            slope = np.polyfit(x, y, 1)[0]
        else:
            slope = 0
        
        # Calculate variance (for fluctuating detection)
        variance = np.var(recent)
        variance_threshold = 0.02  # Threshold for fluctuation
        
        # Determine trend
        if abs(slope) < self.TREND_SLOPE_THRESHOLD and variance < variance_threshold:
            return StressTrend.STABLE
        elif variance > variance_threshold:
            return StressTrend.FLUCTUATING
        elif slope > self.TREND_SLOPE_THRESHOLD:
            return StressTrend.INCREASING
        elif slope < -self.TREND_SLOPE_THRESHOLD:
            return StressTrend.DECREASING
        else:
            return StressTrend.STABLE
    
    def detect_stress_type(self, current_stress, history_window=60):
        """
        Detect if stress is acute (sudden spike) or chronic (sustained).
        
        Args:
            current_stress: Current stress score
            history_window: Number of frames to look back
        
        Returns:
            StressType enum value
        """
        if len(self.stress_history) < history_window:
            return StressType.NONE
        
        recent = list(self.stress_history)[-history_window:]
        baseline = np.mean(recent[:-10]) if len(recent) > 10 else np.mean(recent)
        
        # Check if current is significantly above baseline
        if current_stress < baseline + 0.2:  # Not significantly elevated
            return StressType.NONE
        
        # Check if it's a sudden spike (acute) or sustained (chronic)
        elevated_frames = sum(1 for s in recent if s > baseline + 0.2)
        elevation_ratio = elevated_frames / len(recent)
        
        if elevation_ratio > 0.7:  # Sustained elevation
            return StressType.CHRONIC
        elif current_stress > baseline + 0.3:  # Sudden spike
            return StressType.ACUTE
        else:
            return StressType.NONE
    
    def identify_stress_events(self, current_stress, timestamp=None):
        """
        Identify and log significant stress events (spikes).
        
        Args:
            current_stress: Current stress score
            timestamp: Current timestamp (optional)
        
        Returns:
            List of detected events (updated)
        """
        if timestamp is None:
            timestamp = len(self.stress_history)
        
        self.timestamp_history.append(timestamp)
        
        # Check if we're in a high-stress state
        if current_stress >= self.EVENT_THRESHOLD:
            # Check if this is continuation of existing event
            if len(self.detected_events) > 0:
                last_event = self.detected_events[-1]
                # If within 1 second of last event, extend it
                if timestamp - last_event['end_time'] < 1.0:
                    last_event['end_time'] = timestamp
                    last_event['peak_stress'] = max(last_event['peak_stress'], current_stress)
                    last_event['duration'] = last_event['end_time'] - last_event['start_time']
                    return self.detected_events
            
            # Check if we're starting a new event
            if timestamp > self.last_event_end + 2.0:  # At least 2 seconds since last event
                # New event detected
                event = {
                    'start_time': timestamp,
                    'end_time': timestamp,
                    'peak_stress': current_stress,
                    'duration': 0.0,
                    'type': self.detect_stress_type(current_stress).value
                }
                self.detected_events.append(event)
                self.last_event_end = timestamp
        
        return self.detected_events
    
    def get_stress_statistics(self, window_size=300):
        """
        Calculate summary statistics for stress analysis.
        
        Args:
            window_size: Number of frames to analyze
        
        Returns:
            Dictionary with statistics
        """
        if len(self.stress_history) == 0:
            return {
                'avg_stress': 0.0,
                'peak_stress': 0.0,
                'min_stress': 0.0,
                'std_stress': 0.0,
                'time_in_levels': {},
                'total_events': 0
            }
        
        recent = list(self.stress_history)[-window_size:]
        
        # Basic statistics
        stats = {
            'avg_stress': np.mean(recent),
            'peak_stress': np.max(recent),
            'min_stress': np.min(recent),
            'std_stress': np.std(recent)
        }
        
        # Time spent in each level
        time_in_levels = {
            'Very Low': 0,
            'Low': 0,
            'Moderate': 0,
            'High': 0,
            'Very High': 0
        }
        
        for stress in recent:
            level = self.classify_stress_level(stress)
            time_in_levels[level.value] += 1
        
        # Convert to percentages
        total = len(recent)
        for level in time_in_levels:
            time_in_levels[level] = (time_in_levels[level] / total * 100) if total > 0 else 0
        
        stats['time_in_levels'] = time_in_levels
        stats['total_events'] = len(self.detected_events)
        
        return stats
    
    def analyze(self, stress_score, timestamp=None):
        """
        Complete stress analysis: classification, trend, type, and events.
        
        Args:
            stress_score: Current stress score (0.0 to 1.0)
            timestamp: Current timestamp (optional)
        
        Returns:
            Dictionary with complete analysis
        """
        # Classify level
        level = self.classify_stress_level(stress_score)
        
        # Detect trend
        trend = self.detect_stress_trend(stress_score)
        
        # Detect type
        stress_type = self.detect_stress_type(stress_score)
        
        # Identify events
        events = self.identify_stress_events(stress_score, timestamp)
        
        return {
            'stress_score': stress_score,
            'stress_level': level.value,
            'stress_category': level.value,  # Alias for compatibility
            'stress_trend': trend.value,
            'stress_type': stress_type.value,
            'events': events,
            'recent_events': events[-5:] if len(events) > 5 else events  # Last 5 events
        }
    
    def reset(self):
        """Reset classifier history and events."""
        self.stress_history.clear()
        self.timestamp_history.clear()
        self.detected_events.clear()
        self.last_event_end = -1
    
    def get_duration_analysis(self, duration_seconds=None):
        """
        Analyze time spent in each stress level.
        
        Args:
            duration_seconds: Total duration in seconds (optional)
        
        Returns:
            Dictionary with duration breakdown
        """
        if len(self.stress_history) == 0:
            return {}
        
        level_counts = {
            'Very Low': 0,
            'Low': 0,
            'Moderate': 0,
            'High': 0,
            'Very High': 0
        }
        
        for stress in self.stress_history:
            level = self.classify_stress_level(stress)
            level_counts[level.value] += 1
        
        total = len(self.stress_history)
        
        # Convert to percentages and seconds
        duration_analysis = {}
        for level, count in level_counts.items():
            pct = (count / total * 100) if total > 0 else 0
            seconds = (pct / 100 * duration_seconds) if duration_seconds else None
            duration_analysis[level] = {
                'frames': count,
                'percentage': pct,
                'seconds': seconds
            }
        
        return duration_analysis

