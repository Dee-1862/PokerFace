"""
Stress Analytics Module
Provides visualization and statistical analysis of stress data.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from stress_classifier import StressClassifier, StressLevel


class StressAnalytics:
    """
    Analytics and visualization for stress detection data.
    """
    
    def __init__(self):
        """Initialize analytics module."""
        self.classifier = StressClassifier()
    
    def generate_timeline(self, stress_data, output_path=None, title="Stress Timeline"):
        """
        Generate stress timeline visualization.
        
        Args:
            stress_data: List of dictionaries with 'timestamp' and 'stress_score'
            output_path: Path to save image (optional)
            title: Chart title
        
        Returns:
            matplotlib figure
        """
        if not stress_data:
            print("[WARN] No data to visualize")
            return None
        
        # Extract data
        timestamps = [d.get('timestamp', i) for i, d in enumerate(stress_data)]
        stress_scores = [d.get('stress_score', 0) for d in stress_data]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot stress over time
        ax.plot(timestamps, stress_scores, linewidth=2, color='#2E86AB', label='Stress Score')
        
        # Add level thresholds
        ax.axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='Very Low/Low')
        ax.axhline(y=0.4, color='yellow', linestyle='--', alpha=0.5, label='Low/Moderate')
        ax.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Moderate/High')
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='High/Very High')
        
        # Fill areas by stress level
        ax.fill_between(timestamps, 0, 0.2, alpha=0.1, color='green', label='Very Low')
        ax.fill_between(timestamps, 0.2, 0.4, alpha=0.1, color='lightgreen', label='Low')
        ax.fill_between(timestamps, 0.4, 0.6, alpha=0.1, color='yellow', label='Moderate')
        ax.fill_between(timestamps, 0.6, 0.8, alpha=0.1, color='orange', label='High')
        ax.fill_between(timestamps, 0.8, 1.0, alpha=0.1, color='red', label='Very High')
        
        # Highlight stress events
        events = [d for d in stress_data if d.get('stress_score', 0) > 0.7]
        if events:
            event_times = [d.get('timestamp', 0) for d in events]
            event_scores = [d.get('stress_score', 0) for d in events]
            ax.scatter(event_times, event_scores, color='red', s=50, zorder=5, 
                      label='High Stress Events', marker='v')
        
        # Labels and formatting
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Stress Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        
        plt.tight_layout()
        
        # Save if path provided
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"[OK] Timeline saved to: {output_path}")
        
        return fig
    
    def generate_distribution(self, stress_data, output_path=None, title="Stress Distribution"):
        """
        Generate stress distribution histogram.
        
        Args:
            stress_data: List of dictionaries with 'stress_score'
            output_path: Path to save image (optional)
            title: Chart title
        
        Returns:
            matplotlib figure
        """
        if not stress_data:
            print("[WARN] No data to visualize")
            return None
        
        stress_scores = [d.get('stress_score', 0) for d in stress_data]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogram
        ax1.hist(stress_scores, bins=50, color='#2E86AB', edgecolor='black', alpha=0.7)
        ax1.axvline(x=np.mean(stress_scores), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(stress_scores):.2f}')
        ax1.set_xlabel('Stress Score', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Stress Score Distribution', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Level distribution (pie chart)
        level_counts = {
            'Very Low': 0,
            'Low': 0,
            'Moderate': 0,
            'High': 0,
            'Very High': 0
        }
        
        for score in stress_scores:
            if score < 0.2:
                level_counts['Very Low'] += 1
            elif score < 0.4:
                level_counts['Low'] += 1
            elif score < 0.6:
                level_counts['Moderate'] += 1
            elif score < 0.8:
                level_counts['High'] += 1
            else:
                level_counts['Very High'] += 1
        
        # Filter out zero counts
        levels = [k for k, v in level_counts.items() if v > 0]
        counts = [level_counts[k] for k in levels]
        colors = ['green', 'lightgreen', 'yellow', 'orange', 'red']
        level_colors = [colors[i] for i, k in enumerate(['Very Low', 'Low', 'Moderate', 'High', 'Very High']) if k in levels]
        
        ax2.pie(counts, labels=levels, autopct='%1.1f%%', colors=level_colors, startangle=90)
        ax2.set_title('Time in Each Stress Level', fontsize=13, fontweight='bold')
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save if path provided
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"[OK] Distribution saved to: {output_path}")
        
        return fig
    
    def calculate_time_in_range(self, stress_data):
        """
        Calculate time spent in each stress level.
        
        Args:
            stress_data: List of dictionaries with 'stress_score' and 'timestamp'
        
        Returns:
            Dictionary with time breakdown
        """
        if not stress_data:
            return {}
        
        level_times = {
            'Very Low': 0.0,
            'Low': 0.0,
            'Moderate': 0.0,
            'High': 0.0,
            'Very High': 0.0
        }
        
        # Calculate time differences
        for i, data in enumerate(stress_data):
            score = data.get('stress_score', 0)
            timestamp = data.get('timestamp', i)
            
            # Determine time interval
            if i == 0:
                time_interval = 0.033  # Assume 30fps default
            else:
                prev_timestamp = stress_data[i-1].get('timestamp', i-1)
                time_interval = timestamp - prev_timestamp
            
            # Classify and accumulate
            if score < 0.2:
                level_times['Very Low'] += time_interval
            elif score < 0.4:
                level_times['Low'] += time_interval
            elif score < 0.6:
                level_times['Moderate'] += time_interval
            elif score < 0.8:
                level_times['High'] += time_interval
            else:
                level_times['Very High'] += time_interval
        
        # Convert to percentages
        total_time = sum(level_times.values())
        if total_time > 0:
            for level in level_times:
                level_times[level] = {
                    'seconds': level_times[level],
                    'percentage': (level_times[level] / total_time * 100)
                }
        
        return level_times
    
    def identify_peak_moments(self, stress_data, top_n=10):
        """
        Identify peak stress moments.
        
        Args:
            stress_data: List of dictionaries with 'timestamp' and 'stress_score'
            top_n: Number of top moments to return
        
        Returns:
            List of peak moments sorted by stress score
        """
        if not stress_data:
            return []
        
        # Create list of moments with stress scores
        moments = []
        for data in stress_data:
            moments.append({
                'timestamp': data.get('timestamp', 0),
                'stress_score': data.get('stress_score', 0),
                'frame_number': data.get('frame_number', 0)
            })
        
        # Sort by stress score (descending)
        moments.sort(key=lambda x: x['stress_score'], reverse=True)
        
        return moments[:top_n]
    
    def detect_patterns(self, stress_data, window_size=60):
        """
        Detect recurring stress patterns.
        
        Args:
            stress_data: List of dictionaries with 'stress_score'
            window_size: Window size for pattern detection
        
        Returns:
            Dictionary with detected patterns
        """
        if len(stress_data) < window_size * 2:
            return {}
        
        stress_scores = [d.get('stress_score', 0) for d in stress_data]
        
        patterns = {
            'recurring_spikes': 0,
            'gradual_increases': 0,
            'gradual_decreases': 0,
            'oscillations': 0
        }
        
        # Analyze windows
        for i in range(len(stress_scores) - window_size):
            window = stress_scores[i:i+window_size]
            
            # Check for spike pattern
            if max(window) > 0.7 and np.std(window) > 0.15:
                patterns['recurring_spikes'] += 1
            
            # Check for gradual increase
            slope = np.polyfit(range(len(window)), window, 1)[0]
            if slope > 0.005:
                patterns['gradual_increases'] += 1
            elif slope < -0.005:
                patterns['gradual_decreases'] += 1
            
            # Check for oscillation
            if np.std(window) > 0.1 and len(set(np.sign(np.diff(window)))) > 1:
                patterns['oscillations'] += 1
        
        return patterns
    
    def generate_comprehensive_report(self, stress_data, output_dir=None):
        """
        Generate comprehensive analytics report with all visualizations.
        
        Args:
            stress_data: List of dictionaries with stress data
            output_dir: Output directory for saved files
        
        Returns:
            Dictionary with paths to generated files
        """
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = Path.cwd()
        
        generated_files = {}
        
        # Timeline
        timeline_path = output_path / "stress_timeline.png"
        self.generate_timeline(stress_data, str(timeline_path))
        generated_files['timeline'] = str(timeline_path)
        
        # Distribution
        dist_path = output_path / "stress_distribution.png"
        self.generate_distribution(stress_data, str(dist_path))
        generated_files['distribution'] = str(dist_path)
        
        # Statistics
        stats = {
            'time_in_range': self.calculate_time_in_range(stress_data),
            'peak_moments': self.identify_peak_moments(stress_data, top_n=10),
            'patterns': self.detect_patterns(stress_data)
        }
        
        stats_path = output_path / "stress_statistics.json"
        import json
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        generated_files['statistics'] = str(stats_path)
        
        print(f"\n[OK] Comprehensive report generated in: {output_path}")
        
        return generated_files

