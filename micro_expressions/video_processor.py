"""
Video File Processing Module
Supports batch and real-time processing of video files for stress detection.
"""
import cv2
import time
import json
import csv
import numpy as np
from pathlib import Path
from collections import deque
from engine import StressDetectionEngine


class VideoFileProcessor:
    """
    Processes video files for stress detection.
    Supports both batch (offline) and real-time (playback) modes.
    """
    
    def __init__(self, video_path, config=None):
        """
        Initialize video processor.
        
        Args:
            video_path: Path to video file
            config: Configuration dictionary
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.config = config or {}
        self.cap = cv2.VideoCapture(str(self.video_path))
        
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")
        
        # Video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
        
        # Processing state
        self.current_frame = 0
        self.results = []
        self.start_time = None
        
        print(f"Video loaded: {self.video_path.name}")
        print(f"  Resolution: {self.width}x{self.height}")
        print(f"  FPS: {self.fps:.2f}")
        print(f"  Duration: {self.duration:.2f}s ({self.frame_count} frames)")
    
    def process_batch(self, engine, skip_frames=1, progress_callback=None):
        """
        Process entire video in batch mode (offline analysis).
        
        Args:
            engine: StressDetectionEngine instance
            skip_frames: Process every Nth frame (1 = all frames)
            progress_callback: Optional callback(progress, eta) function
        
        Returns:
            List of results dictionaries
        """
        print(f"\n{'='*60}")
        print("BATCH PROCESSING MODE")
        print(f"{'='*60}")
        print(f"Processing every {skip_frames} frame(s)...")
        
        self.results = []
        self.start_time = time.time()
        frame_processed = 0
        
        # Reset video to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Skip frames if needed
            if self.current_frame % skip_frames != 0:
                self.current_frame += 1
                continue
            
            # Update shared state
            h, w = frame.shape[:2]
            engine.shared_state.update({
                'frame': frame,
                'frame_dimensions': (h, w),
                'timestamp': self.current_frame / self.fps if self.fps > 0 else time.time(),
                'frame_number': self.current_frame,
                'is_video_file': True,
                'video_fps': self.fps,
                'video_duration': self.duration
            })
            
            # Process modules
            for name, module in engine.modules.items():
                module.process(engine.shared_state)
            
            # Collect results
            result = {
                'frame_number': self.current_frame,
                'timestamp': self.current_frame / self.fps if self.fps > 0 else 0,
                'heart_rate_bpm': engine.shared_state.get('heart_rate_bpm', 0),
                'stress_score': engine.shared_state.get('stress_score', 0.0),
                'stress_category': engine.shared_state.get('stress_category', 'Unknown'),
                'stress_trend': engine.shared_state.get('stress_trend', 'Stable'),
                'action_units': engine.shared_state.get('action_units', {}).copy(),
                'face_detected': engine.shared_state.get('face_detected', False)
            }
            
            self.results.append(result)
            frame_processed += 1
            
            # Progress update
            if progress_callback and frame_processed % 10 == 0:
                progress = self.current_frame / self.frame_count
                elapsed = time.time() - self.start_time
                eta = (elapsed / progress - elapsed) if progress > 0 else 0
                progress_callback(progress, eta)
            
            # Print progress every 100 frames
            if frame_processed % 100 == 0:
                progress_pct = (self.current_frame / self.frame_count) * 100
                print(f"Progress: {progress_pct:.1f}% ({frame_processed} frames processed)")
            
            self.current_frame += 1
        
        elapsed_time = time.time() - self.start_time
        print(f"\n✓ Batch processing complete!")
        print(f"  Processed: {frame_processed} frames in {elapsed_time:.2f}s")
        print(f"  Average: {elapsed_time/frame_processed*1000:.1f}ms per frame")
        
        return self.results
    
    def process_realtime(self, engine):
        """
        Process video in real-time playback mode with visualization.
        
        Args:
            engine: StressDetectionEngine instance
        
        Returns:
            List of results dictionaries
        """
        print(f"\n{'='*60}")
        print("REAL-TIME PLAYBACK MODE")
        print(f"{'='*60}")
        print("Controls:")
        print("  SPACE: Play/Pause")
        print("  LEFT/RIGHT: Seek backward/forward (5 seconds)")
        print("  +/-: Speed up/down")
        print("  's': Save current frame analysis")
        print("  'a': Toggle enhanced analytics panel")
        print("  'q': Quit")
        print("\nEnhanced Classification Features:")
        print("  - 5-Level Stress Categories (Very Low to Very High)")
        print("  - Trend Detection (Increasing/Decreasing/Stable/Fluctuating)")
        print("  - Stress Type (Acute/Chronic)")
        print("  - Event Detection and Counting")
        print("  - Real-time Heart Rate")
        
        self.results = []
        self.current_frame = 0
        playing = True
        playback_speed = 1.0
        frame_delay = int(1000 / self.fps) if self.fps > 0 else 33
        show_enhanced_panel = True  # Toggle for enhanced classification panel
        result = None  # Store last result for saving
        
        # Reset video to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while True:
            if playing:
                ret, frame = self.cap.read()
                if not ret:
                    # End of video - loop or exit
                    print("End of video reached. Press 'q' to quit or 'r' to restart.")
                    playing = False
                    continue
                
                # Update shared state
                h, w = frame.shape[:2]
                engine.shared_state.update({
                    'frame': frame,
                    'frame_dimensions': (h, w),
                    'timestamp': self.current_frame / self.fps if self.fps > 0 else time.time(),
                    'frame_number': self.current_frame,
                    'is_video_file': True,
                    'video_fps': self.fps,
                    'video_duration': self.duration
                })
                
                # Process modules
                for name, module in engine.modules.items():
                    module.process(engine.shared_state)
                
                # Collect results
                result = {
                    'frame_number': self.current_frame,
                    'timestamp': self.current_frame / self.fps if self.fps > 0 else 0,
                    'heart_rate_bpm': engine.shared_state.get('heart_rate_bpm', 0),
                    'stress_score': engine.shared_state.get('stress_score', 0.0),
                    'stress_category': engine.shared_state.get('stress_category', 'Unknown'),
                    'stress_trend': engine.shared_state.get('stress_trend', 'Stable'),
                    'stress_type': engine.shared_state.get('stress_type', 'None'),
                    'action_units': engine.shared_state.get('action_units', {}).copy(),
                    'face_detected': engine.shared_state.get('face_detected', False)
                }
                self.results.append(result)
                
                # Render
                display_frame = frame.copy()
                for name, module in engine.modules.items():
                    display_frame = module.render(display_frame, engine.shared_state)
                
                # Add enhanced classification panel (if enabled)
                if show_enhanced_panel:
                    self._draw_enhanced_classification(display_frame, engine.shared_state)
                
                # Add playback info
                self._draw_playback_info(display_frame, playing, playback_speed)
                
                cv2.imshow('Video Stress Detection - Real-time Playback', display_frame)
                
                self.current_frame += 1
            else:
                # When paused, still show the last frame with overlays
                if result is not None:
                    ret, frame = self.cap.read()
                    if ret:
                        # Update shared state for rendering
                        h, w = frame.shape[:2]
                        engine.shared_state.update({
                            'frame': frame,
                            'frame_dimensions': (h, w),
                            'timestamp': self.current_frame / self.fps if self.fps > 0 else time.time(),
                            'frame_number': self.current_frame,
                            'is_video_file': True,
                            'video_fps': self.fps,
                            'video_duration': self.duration
                        })
                        
                        # Render (without processing)
                        display_frame = frame.copy()
                        for name, module in engine.modules.items():
                            display_frame = module.render(display_frame, engine.shared_state)
                        
                        # Add enhanced classification panel (if enabled)
                        if show_enhanced_panel:
                            self._draw_enhanced_classification(display_frame, engine.shared_state)
                        
                        # Add playback info
                        self._draw_playback_info(display_frame, playing, playback_speed)
                        
                        cv2.imshow('Video Stress Detection - Real-time Playback', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(int(frame_delay / playback_speed)) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space = play/pause
                playing = not playing
            elif key == ord('+') or key == ord('='):
                playback_speed = min(playback_speed * 1.5, 4.0)
                print(f"Playback speed: {playback_speed:.1f}x")
            elif key == ord('-') or key == ord('_'):
                playback_speed = max(playback_speed / 1.5, 0.25)
                print(f"Playback speed: {playback_speed:.1f}x")
            elif key == ord('s'):  # Save current frame
                if result is not None:
                    self._save_frame_analysis(result)
                else:
                    print("⚠ No frame data to save")
            elif key == ord('a'):  # Toggle enhanced analytics
                show_enhanced_panel = not show_enhanced_panel
                print(f"Enhanced classification panel: {'ON' if show_enhanced_panel else 'OFF'}")
            elif key == 81 or key == 2:  # Left arrow
                self._seek_relative(-5)  # Seek back 5 seconds
            elif key == 83 or key == 3:  # Right arrow
                self._seek_relative(5)  # Seek forward 5 seconds
            elif key == ord('r'):  # Restart
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.current_frame = 0
                playing = True
        
        cv2.destroyAllWindows()
        return self.results
    
    def _seek_relative(self, seconds):
        """Seek video relative to current position."""
        current_pos = self.current_frame / self.fps if self.fps > 0 else 0
        new_pos = max(0, min(current_pos + seconds, self.duration))
        new_frame = int(new_pos * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
        self.current_frame = new_frame
        print(f"Seeked to {new_pos:.1f}s (frame {new_frame})")
    
    def _draw_enhanced_classification(self, frame, shared_state):
        """Draw enhanced classification analytics panel."""
        h, w = frame.shape[:2]
        
        # Get stress data
        stress_score = shared_state.get('stress_score', 0.0)
        stress_category = shared_state.get('stress_category', 'Unknown')
        stress_trend = shared_state.get('stress_trend', 'Stable')
        stress_type = shared_state.get('stress_type', 'None')
        stress_events = shared_state.get('stress_events', [])
        heart_rate = shared_state.get('heart_rate_bpm', 0)
        
        # Position: Top-left corner
        x = 10
        y = 10
        panel_width = 280
        panel_height = 200
        
        # Background panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame, (x, y), (x + panel_width, y + panel_height), (100, 100, 100), 2)
        
        # Title
        cv2.putText(frame, "ENHANCED CLASSIFICATION", (x + 10, y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Stress Category with color coding
        y_offset = y + 50
        category_colors = {
            'Very Low': (0, 255, 0),
            'Low': (0, 255, 100),
            'Moderate': (0, 255, 255),
            'High': (0, 150, 255),
            'Very High': (0, 0, 255),
            'Calibrating': (150, 150, 150),
            'Unknown': (150, 150, 150)
        }
        category_color = category_colors.get(stress_category, (200, 200, 200))
        
        cv2.putText(frame, f"Category: {stress_category}", (x + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, category_color, 2)
        
        # Stress Score
        y_offset += 25
        cv2.putText(frame, f"Score: {stress_score:.1%}", (x + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Trend
        y_offset += 25
        trend_symbols = {
            'Increasing': ('↑', (0, 0, 255)),
            'Decreasing': ('↓', (0, 255, 0)),
            'Fluctuating': ('~', (0, 255, 255)),
            'Stable': ('→', (200, 200, 200))
        }
        symbol, trend_color = trend_symbols.get(stress_trend, ('?', (200, 200, 200)))
        cv2.putText(frame, f"Trend: {symbol} {stress_trend}", (x + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, trend_color, 1)
        
        # Stress Type
        y_offset += 25
        if stress_type != 'None':
            type_color = (0, 150, 255) if stress_type == 'Acute' else (0, 100, 255)
            cv2.putText(frame, f"Type: {stress_type}", (x + 10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, type_color, 1)
        else:
            cv2.putText(frame, "Type: Normal", (x + 10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Heart Rate
        y_offset += 25
        if heart_rate > 0:
            cv2.putText(frame, f"Heart Rate: {heart_rate} BPM", (x + 10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Event Count
        y_offset += 25
        event_count = len(stress_events)
        if event_count > 0:
            cv2.putText(frame, f"Stress Events: {event_count}", (x + 10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
        
        # Recent event indicator
        if stress_score > 0.7:
            y_offset += 25
            cv2.putText(frame, "! HIGH STRESS EVENT !", (x + 10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    def _draw_playback_info(self, frame, playing, speed):
        """Draw playback controls info on frame."""
        h, w = frame.shape[:2]
        
        # Background panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, h - 80), (300, h - 10), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Status
        status = "PLAYING" if playing else "PAUSED"
        color = (0, 255, 0) if playing else (0, 0, 255)
        cv2.putText(frame, status, (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Time info
        current_time = self.current_frame / self.fps if self.fps > 0 else 0
        time_text = f"{current_time:.1f}s / {self.duration:.1f}s"
        cv2.putText(frame, time_text, (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Speed
        speed_text = f"Speed: {speed:.1f}x"
        cv2.putText(frame, speed_text, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def _save_frame_analysis(self, result):
        """Save current frame analysis to file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"frame_analysis_{timestamp}_frame{result['frame_number']}.json"
        
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"✓ Frame analysis saved to: {filename}")
    
    def export_results(self, output_dir=None, formats=['csv', 'json']):
        """
        Export processing results to files.
        
        Args:
            output_dir: Output directory (default: current directory)
            formats: List of formats to export ('csv', 'json', 'both')
        
        Returns:
            Dictionary of exported file paths
        """
        if not self.results:
            print("⚠ No results to export")
            return {}
        
        output_path = Path(output_dir) if output_dir else Path.cwd()
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = self.video_path.stem
        
        exported = {}
        
        # CSV Export
        if 'csv' in formats or 'both' in formats:
            csv_path = output_path / f"{base_name}_results_{timestamp}.csv"
            self._export_csv(csv_path)
            exported['csv'] = str(csv_path)
        
        # JSON Export
        if 'json' in formats or 'both' in formats:
            json_path = output_path / f"{base_name}_results_{timestamp}.json"
            self._export_json(json_path)
            exported['json'] = str(json_path)
        
        # Summary report
        summary_path = output_path / f"{base_name}_summary_{timestamp}.txt"
        self._export_summary(summary_path)
        exported['summary'] = str(summary_path)
        
        print(f"\n✓ Results exported to: {output_path}")
        for fmt, path in exported.items():
            print(f"  {fmt.upper()}: {path}")
        
        return exported
    
    def _export_csv(self, filepath):
        """Export results to CSV format."""
        with open(filepath, 'w', newline='') as f:
            if not self.results:
                return
            
            # Get all AU keys from first result
            au_keys = set()
            for result in self.results:
                if 'action_units' in result:
                    au_keys.update(result['action_units'].keys())
            au_keys = sorted(au_keys)
            
            # CSV headers
            fieldnames = ['frame_number', 'timestamp', 'heart_rate_bpm', 'stress_score', 
                         'stress_category', 'stress_trend', 'face_detected']
            fieldnames.extend([f'au_{au}' for au in au_keys])
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            # Write rows
            for result in self.results:
                row = {
                    'frame_number': result.get('frame_number', 0),
                    'timestamp': result.get('timestamp', 0),
                    'heart_rate_bpm': result.get('heart_rate_bpm', 0),
                    'stress_score': result.get('stress_score', 0.0),
                    'stress_category': result.get('stress_category', 'Unknown'),
                    'stress_trend': result.get('stress_trend', 'Stable'),
                    'face_detected': result.get('face_detected', False)
                }
                
                # Add AU values
                aus = result.get('action_units', {})
                for au in au_keys:
                    row[f'au_{au}'] = aus.get(au, 0.0)
                
                writer.writerow(row)
    
    def _export_json(self, filepath):
        """Export results to JSON format with metadata."""
        export_data = {
            'metadata': {
                'video_file': str(self.video_path),
                'video_resolution': f"{self.width}x{self.height}",
                'video_fps': self.fps,
                'video_duration': self.duration,
                'total_frames': self.frame_count,
                'processed_frames': len(self.results),
                'export_timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            },
            'results': self.results,
            'summary': self._calculate_summary_stats()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def _export_summary(self, filepath):
        """Export summary report as text file."""
        stats = self._calculate_summary_stats()
        
        with open(filepath, 'w') as f:
            f.write("="*60 + "\n")
            f.write("STRESS DETECTION ANALYSIS SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Video: {self.video_path.name}\n")
            f.write(f"Duration: {self.duration:.2f} seconds\n")
            f.write(f"Frames Processed: {len(self.results)}\n\n")
            
            f.write("STRESS STATISTICS\n")
            f.write("-"*60 + "\n")
            f.write(f"Average Stress: {stats['avg_stress']:.2%}\n")
            f.write(f"Peak Stress: {stats['peak_stress']:.2%}\n")
            f.write(f"Min Stress: {stats['min_stress']:.2%}\n")
            f.write(f"Std Deviation: {stats['std_stress']:.2%}\n\n")
            
            f.write("STRESS LEVEL DISTRIBUTION\n")
            f.write("-"*60 + "\n")
            for level, count in stats['level_distribution'].items():
                pct = (count / len(self.results)) * 100 if self.results else 0
                f.write(f"{level}: {count} frames ({pct:.1f}%)\n")
            
            f.write("\nHEART RATE STATISTICS\n")
            f.write("-"*60 + "\n")
            f.write(f"Average BPM: {stats['avg_bpm']:.1f}\n")
            f.write(f"Peak BPM: {stats['peak_bpm']:.1f}\n")
            f.write(f"Min BPM: {stats['min_bpm']:.1f}\n")
            
            if stats.get('peak_stress_moments'):
                f.write("\nPEAK STRESS MOMENTS\n")
                f.write("-"*60 + "\n")
                for moment in stats['peak_stress_moments'][:5]:  # Top 5
                    f.write(f"  {moment['timestamp']:.1f}s: {moment['stress']:.2%}\n")
    
    def _calculate_summary_stats(self):
        """Calculate summary statistics from results."""
        if not self.results:
            return {}
        
        stress_scores = [r.get('stress_score', 0) for r in self.results if r.get('stress_score') is not None]
        bpms = [r.get('heart_rate_bpm', 0) for r in self.results if r.get('heart_rate_bpm', 0) > 0]
        
        # Stress level distribution
        level_dist = {'Very Low': 0, 'Low': 0, 'Moderate': 0, 'High': 0, 'Very High': 0}
        for result in self.results:
            category = result.get('stress_category', 'Unknown')
            if category in level_dist:
                level_dist[category] += 1
        
        # Peak stress moments
        peak_moments = []
        for result in self.results:
            if result.get('stress_score', 0) > 0.7:  # High stress threshold
                peak_moments.append({
                    'timestamp': result.get('timestamp', 0),
                    'stress': result.get('stress_score', 0),
                    'frame': result.get('frame_number', 0)
                })
        peak_moments.sort(key=lambda x: x['stress'], reverse=True)
        
        return {
            'avg_stress': np.mean(stress_scores) if stress_scores else 0,
            'peak_stress': np.max(stress_scores) if stress_scores else 0,
            'min_stress': np.min(stress_scores) if stress_scores else 0,
            'std_stress': np.std(stress_scores) if stress_scores else 0,
            'avg_bpm': np.mean(bpms) if bpms else 0,
            'peak_bpm': np.max(bpms) if bpms else 0,
            'min_bpm': np.min(bpms) if bpms else 0,
            'level_distribution': level_dist,
            'peak_stress_moments': peak_moments
        }
    
    def cleanup(self):
        """Release video capture resources."""
        if self.cap:
            self.cap.release()

