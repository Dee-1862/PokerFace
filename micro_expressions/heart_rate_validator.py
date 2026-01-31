"""
Heart Rate Validation Module
Compares rPPG readings with reference device (e.g., Apple Watch) for validation.
Provides statistical metrics: MAE, RMSE, Correlation, Bland-Altman analysis.
"""

import numpy as np
import json
import os
from datetime import datetime
from collections import deque


class HeartRateValidator:
    """
    Validates rPPG heart rate measurements against reference device (Apple Watch).
    Records synchronized readings and calculates validation metrics.
    """
    
    def __init__(self, max_readings=1000):
        """
        Initialize validator.
        
        Args:
            max_readings: Maximum number of readings to store
        """
        self.apple_watch_readings = deque(maxlen=max_readings)
        self.rppg_readings = deque(maxlen=max_readings)
        self.timestamps = deque(maxlen=max_readings)
        self.is_recording = False
        
    def start_recording(self):
        """Start recording validation data."""
        self.is_recording = True
        self.apple_watch_readings.clear()
        self.rppg_readings.clear()
        self.timestamps.clear()
        print("✓ Validation recording started")
    
    def stop_recording(self):
        """Stop recording validation data."""
        self.is_recording = False
        print(f"✓ Validation recording stopped. Collected {len(self.rppg_readings)} readings")
    
    def add_reading(self, apple_watch_bpm, rppg_bpm, timestamp=None):
        """
        Record synchronized readings from both devices.
        
        Args:
            apple_watch_bpm: Heart rate from Apple Watch (reference)
            rppg_bpm: Heart rate from rPPG system
            timestamp: Optional timestamp (defaults to current time)
        """
        if not self.is_recording:
            return False
        
        if apple_watch_bpm is None or rppg_bpm is None:
            return False
        
        if apple_watch_bpm <= 0 or rppg_bpm <= 0:
            return False
        
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        self.apple_watch_readings.append(float(apple_watch_bpm))
        self.rppg_readings.append(float(rppg_bpm))
        self.timestamps.append(timestamp)
        
        return True
    
    def calculate_metrics(self):
        """
        Calculate validation metrics.
        
        Returns:
            dict: Validation metrics including MAE, RMSE, Correlation, Bias, etc.
            Returns None if insufficient data.
        """
        if len(self.apple_watch_readings) < 10:
            return None
        
        aw = np.array(self.apple_watch_readings)
        rppg = np.array(self.rppg_readings)
        
        # Mean Absolute Error (MAE)
        mae = np.mean(np.abs(aw - rppg))
        
        # Root Mean Square Error (RMSE)
        rmse = np.sqrt(np.mean((aw - rppg)**2))
        
        # Pearson Correlation Coefficient
        if np.std(aw) > 0 and np.std(rppg) > 0:
            correlation = np.corrcoef(aw, rppg)[0, 1]
        else:
            correlation = 0.0
        
        # Mean Error (Bias) - positive means rPPG overestimates
        bias = np.mean(rppg - aw)
        
        # Standard Deviation of differences
        std_diff = np.std(rppg - aw)
        
        # Percentage Error
        mean_aw = np.mean(aw)
        percentage_error = (mae / mean_aw) * 100 if mean_aw > 0 else 0
        
        # Bland-Altman Analysis
        mean_values = (aw + rppg) / 2
        differences = rppg - aw
        mean_diff = np.mean(differences)
        std_diff_ba = np.std(differences)
        
        # Limits of Agreement (95%)
        upper_loa = mean_diff + 1.96 * std_diff_ba
        lower_loa = mean_diff - 1.96 * std_diff_ba
        
        # Count readings within ±5 BPM (clinical tolerance)
        within_5_bpm = np.sum(np.abs(differences) <= 5)
        within_10_bpm = np.sum(np.abs(differences) <= 10)
        
        return {
            'sample_size': len(aw),
            'mean_absolute_error': float(mae),
            'rmse': float(rmse),
            'correlation': float(correlation),
            'bias': float(bias),
            'std_difference': float(std_diff),
            'percentage_error': float(percentage_error),
            'bland_altman': {
                'mean_difference': float(mean_diff),
                'upper_loa': float(upper_loa),
                'lower_loa': float(lower_loa),
                'std_difference': float(std_diff_ba)
            },
            'agreement': {
                'within_5_bpm': int(within_5_bpm),
                'within_10_bpm': int(within_10_bpm),
                'within_5_bpm_percent': float((within_5_bpm / len(aw)) * 100),
                'within_10_bpm_percent': float((within_10_bpm / len(aw)) * 100)
            },
            'statistics': {
                'apple_watch_mean': float(np.mean(aw)),
                'apple_watch_std': float(np.std(aw)),
                'rppg_mean': float(np.mean(rppg)),
                'rppg_std': float(np.std(rppg)),
                'apple_watch_range': [float(np.min(aw)), float(np.max(aw))],
                'rppg_range': [float(np.min(rppg)), float(np.max(rppg))]
            }
        }
    
    def print_metrics(self):
        """Print validation metrics in a readable format."""
        metrics = self.calculate_metrics()
        if metrics is None:
            print("\n⚠ Insufficient data for validation (need at least 10 readings)")
            print(f"   Current readings: {len(self.rppg_readings)}")
            return
        
        print("\n" + "="*60)
        print("HEART RATE VALIDATION METRICS")
        print("="*60)
        print(f"Sample Size: {metrics['sample_size']} readings")
        print(f"\nAccuracy Metrics:")
        print(f"  Mean Absolute Error (MAE): {metrics['mean_absolute_error']:.2f} BPM")
        print(f"  Root Mean Square Error (RMSE): {metrics['rmse']:.2f} BPM")
        print(f"  Percentage Error: {metrics['percentage_error']:.2f}%")
        print(f"\nCorrelation:")
        print(f"  Pearson Correlation: r = {metrics['correlation']:.3f}")
        if metrics['correlation'] > 0.7:
            print("  ✓ Strong correlation")
        elif metrics['correlation'] > 0.5:
            print("  ⚠ Moderate correlation")
        else:
            print("  ✗ Weak correlation")
        print(f"\nBias:")
        print(f"  Mean Bias: {metrics['bias']:.2f} BPM", end="")
        if abs(metrics['bias']) < 3:
            print(" (Good)")
        elif abs(metrics['bias']) < 5:
            print(" (Acceptable)")
        else:
            print(" (High bias)")
        print(f"  Standard Deviation of Differences: {metrics['std_difference']:.2f} BPM")
        print(f"\nBland-Altman Analysis:")
        print(f"  Mean Difference: {metrics['bland_altman']['mean_difference']:.2f} BPM")
        print(f"  Upper Limit of Agreement: {metrics['bland_altman']['upper_loa']:.2f} BPM")
        print(f"  Lower Limit of Agreement: {metrics['bland_altman']['lower_loa']:.2f} BPM")
        print(f"\nAgreement:")
        print(f"  Within ±5 BPM: {metrics['agreement']['within_5_bpm']}/{metrics['sample_size']} ({metrics['agreement']['within_5_bpm_percent']:.1f}%)")
        print(f"  Within ±10 BPM: {metrics['agreement']['within_10_bpm']}/{metrics['sample_size']} ({metrics['agreement']['within_10_bpm_percent']:.1f}%)")
        print(f"\nStatistics:")
        print(f"  Apple Watch: {metrics['statistics']['apple_watch_mean']:.1f} ± {metrics['statistics']['apple_watch_std']:.1f} BPM")
        print(f"  rPPG System: {metrics['statistics']['rppg_mean']:.1f} ± {metrics['statistics']['rppg_std']:.1f} BPM")
        print("="*60 + "\n")
    
    def save_results(self, filepath=None):
        """
        Save validation results to JSON file.
        
        Args:
            filepath: Path to save file (defaults to timestamped filename)
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"validation_results_{timestamp}.json"
        
        metrics = self.calculate_metrics()
        if metrics is None:
            print("⚠ No metrics to save - insufficient data")
            return None
        
        # Include raw data
        data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'raw_data': {
                'apple_watch': list(self.apple_watch_readings),
                'rppg': list(self.rppg_readings),
                'timestamps': list(self.timestamps)
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Validation results saved to: {filepath}")
        return filepath
    
    def load_results(self, filepath):
        """Load validation results from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.apple_watch_readings = deque(data['raw_data']['apple_watch'], maxlen=1000)
        self.rppg_readings = deque(data['raw_data']['rppg'], maxlen=1000)
        self.timestamps = deque(data['raw_data']['timestamps'], maxlen=1000)
        
        print(f"✓ Loaded {len(self.rppg_readings)} readings from {filepath}")
        return data['metrics']
    
    def get_summary_string(self):
        """Get a short summary string for display."""
        metrics = self.calculate_metrics()
        if metrics is None:
            return f"Validating... ({len(self.rppg_readings)} readings)"
        
        return (f"MAE: {metrics['mean_absolute_error']:.1f} BPM | "
                f"r={metrics['correlation']:.2f} | "
                f"Bias: {metrics['bias']:+.1f} BPM")


# Standalone validation script
if __name__ == "__main__":
    print("Heart Rate Validator - Manual Entry Mode")
    print("="*60)
    print("Enter readings manually for validation")
    print("Format: apple_watch_bpm,rppg_bpm (or 'q' to quit, 'c' to calculate)")
    print("="*60 + "\n")
    
    validator = HeartRateValidator()
    validator.start_recording()
    
    while True:
        try:
            user_input = input("Enter reading (AW_BPM,rPPG_BPM): ").strip()
            
            if user_input.lower() == 'q':
                break
            elif user_input.lower() == 'c':
                validator.print_metrics()
                continue
            elif ',' in user_input:
                parts = user_input.split(',')
                aw_bpm = float(parts[0].strip())
                rppg_bpm = float(parts[1].strip())
                validator.add_reading(aw_bpm, rppg_bpm)
                print(f"✓ Added: AW={aw_bpm} BPM, rPPG={rppg_bpm} BPM ({len(validator.rppg_readings)} total)")
            else:
                print("⚠ Invalid format. Use: apple_watch_bpm,rppg_bpm")
        
        except ValueError:
            print("⚠ Invalid number format")
        except KeyboardInterrupt:
            break
    
    validator.stop_recording()
    validator.print_metrics()
    
    save = input("\nSave results? (y/n): ").strip().lower()
    if save == 'y':
        validator.save_results()


