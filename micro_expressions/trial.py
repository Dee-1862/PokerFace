import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from collections import deque
import os
from scipy import signal
# TensorFlow/Keras imports removed - LSTM model has been completely removed
TF_AVAILABLE = False

class RPPGVisualizer:
    def __init__(self, buffer_size=150):
        # Use MediaPipe 0.10+ FaceLandmarker API
        # Get absolute path to model file
        model_path = os.path.join(os.path.dirname(__file__), 'face_landmarker.task')
        
        # Download model if it doesn't exist
        if not os.path.exists(model_path):
            print("Downloading face_landmarker model...")
            import urllib.request
            url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
            urllib.request.urlretrieve(url, model_path)
            print("Model downloaded successfully!")
        
        base_options = python.BaseOptions(model_asset_path=model_path, delegate=python.BaseOptions.Delegate.CPU)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.VIDEO
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
        self.frame_timestamp_ms = 0
        
        # Buffers for signal visualization
        self.buffer_size = buffer_size
        self.signal_buffer = deque(maxlen=buffer_size)
        self.fps = 30  # Approximate FPS for heart rate calculation
        self.current_bpm = 0
        # Buffer for smoothing BPM readings (store last 10 readings)
        self.bpm_buffer = deque(maxlen=10)
        
        # MediaPipe Indices for Forehead and Cheeks
        self.ROI_IDXS = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 
                         397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 
                         172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        
        # ========== FACS/Action Units Mapping ==========
        # MediaPipe landmark indices mapped to key facial points for FACS
        # Based on MediaPipe Face Mesh 468 landmarks
        self.FACS_LANDMARKS = {
            # Brow regions (AU1, AU2, AU4)
            'left_inner_brow': 107,    # Left inner eyebrow
            'left_outer_brow': 70,     # Left outer eyebrow
            'left_brow_center': 63,    # Left brow center
            'right_inner_brow': 336,   # Right inner eyebrow
            'right_outer_brow': 300,   # Right outer eyebrow
            'right_brow_center': 293,  # Right brow center
            
            # Eye regions (AU5, AU6, AU7)
            'left_eye_top': 159,       # Left eye top
            'left_eye_bottom': 145,    # Left eye bottom
            'left_eye_outer': 33,      # Left eye outer corner
            'left_eye_inner': 133,     # Left eye inner corner
            'right_eye_top': 386,      # Right eye top
            'right_eye_bottom': 374,   # Right eye bottom
            'right_eye_outer': 263,    # Right eye outer corner
            'right_eye_inner': 362,    # Right eye inner corner
            
            # Nose region (AU9)
            'nose_tip': 1,             # Nose tip
            'nose_base': 2,            # Nose base
            
            # Mouth region (AU10, AU12, AU15, AU20, AU25, AU26, AU27)
            'mouth_left_corner': 61,   # Left mouth corner
            'mouth_right_corner': 291, # Right mouth corner
            'mouth_top_upper': 13,     # Upper lip top
            'mouth_bottom_lower': 14,  # Lower lip bottom
            'mouth_center_top': 12,    # Upper lip center
            'mouth_center_bottom': 15, # Lower lip center
            
            # Cheek regions (AU6, AU12)
            'left_cheek': 116,         # Left cheek
            'right_cheek': 345,        # Right cheek
            
            # Face width for normalization
            'face_left': 234,          # Left face edge
            'face_right': 454,         # Right face edge
            'face_top': 10,            # Forehead top
            'face_bottom': 152,        # Chin bottom
        }
        
        # Temporal windowing for multi-modal fusion (5-10 seconds at 30 FPS)
        self.window_size_frames = 7 * self.fps  # 7 second window
        self.window_step = self.fps  # Slide window by 1 second
        
        # Buffers for temporal data collection
        self.au_buffer = deque(maxlen=self.window_size_frames)  # Action Units over time
        self.bpm_history = deque(maxlen=self.window_size_frames)  # BPM over time
        self.temporal_features = []  # Combined features for LSTM
        
        # Baseline calibration for FACS (establish neutral face measurements)
        self.baseline_calibrated = False
        self.baseline_aus = {}  # Store baseline AU values
        self.baseline_samples = []  # Collect samples for baseline
        self.calibration_samples_needed = 90  # 3 seconds at 30 FPS
        
        # Temporal smoothing for AUs (reduce noise from head movement)
        self.au_smoothing_window = 10  # Smooth over last 10 frames
        self.previous_aus = deque(maxlen=self.au_smoothing_window)
        
        # Head pose tracking (to normalize for head rotation)
        self.head_pose_buffer = deque(maxlen=30)  # Track head pose over time
        
        # LSTM Model (initialized as None, will be created if TensorFlow available)
        self.lstm_model = None
        self.stress_probability = 0.0
        self.use_lstm = False
        
        # Initialize LSTM if TensorFlow is available
        if TF_AVAILABLE:
            self._initialize_lstm()

    def _initialize_lstm(self):
        """LSTM initialization - DISABLED (model removed)."""
        self.lstm_model = None
        self.use_lstm = False
        # LSTM model has been completely removed
    
    def calculate_face_normalization_factor(self, landmarks, frame_h, frame_w):
        """Calculate normalization factor based on face size."""
        # Get face width and height from key landmarks
        face_left = landmarks[self.FACS_LANDMARKS['face_left']]
        face_right = landmarks[self.FACS_LANDMARKS['face_right']]
        face_top = landmarks[self.FACS_LANDMARKS['face_top']]
        face_bottom = landmarks[self.FACS_LANDMARKS['face_bottom']]
        
        # Calculate Euclidean distances in pixel space
        face_width = np.sqrt(
            ((face_left.x - face_right.x) * frame_w) ** 2 + 
            ((face_left.y - face_right.y) * frame_h) ** 2
        )
        face_height = np.sqrt(
            ((face_top.x - face_bottom.x) * frame_w) ** 2 + 
            ((face_top.y - face_bottom.y) * frame_h) ** 2
        )
        
        # Use average of width and height as normalization factor
        normalization_factor = (face_width + face_height) / 2.0
        
        # Avoid division by zero
        return max(normalization_factor, 1.0)
    
    def estimate_head_pose(self, landmarks, frame_h, frame_w):
        """
        Estimate head pose (pitch, yaw, roll) to compensate for head movement.
        Returns dictionary with pitch, yaw, roll angles in radians.
        """
        # Use key facial landmarks to estimate head orientation
        nose_tip = landmarks[self.FACS_LANDMARKS['nose_tip']]
        chin = landmarks[self.FACS_LANDMARKS['face_bottom']]
        left_eye = landmarks[self.FACS_LANDMARKS['left_eye_inner']]
        right_eye = landmarks[self.FACS_LANDMARKS['right_eye_inner']]
        
        # Convert to pixel coordinates
        nose_pt = np.array([nose_tip.x * frame_w, nose_tip.y * frame_h])
        chin_pt = np.array([chin.x * frame_w, chin.y * frame_h])
        left_eye_pt = np.array([left_eye.x * frame_w, left_eye.y * frame_h])
        right_eye_pt = np.array([right_eye.x * frame_w, right_eye.y * frame_h])
        
        # Calculate vectors
        eye_center = (left_eye_pt + right_eye_pt) / 2
        face_vertical = chin_pt - nose_pt
        face_horizontal = right_eye_pt - left_eye_pt
        
        # Estimate roll (head rotation around z-axis)
        roll = np.arctan2(face_horizontal[1], face_horizontal[0])
        
        # Estimate pitch (head nod up/down) - using vertical vector
        face_vertical_norm = np.linalg.norm(face_vertical)
        pitch = np.arcsin(np.clip(face_vertical[1] / (face_vertical_norm + 1e-6), -1, 1))
        
        # Estimate yaw (head turn left/right) - simplified
        nose_offset = nose_pt - eye_center
        yaw = np.arctan2(nose_offset[0], np.linalg.norm(nose_offset) + 1e-6)
        
        return {'pitch': pitch, 'yaw': yaw, 'roll': roll}
    
    def calculate_euclidean_distance(self, p1, p2, frame_h, frame_w):
        """Calculate Euclidean distance between two landmarks in pixel space."""
        dx = (p1.x - p2.x) * frame_w
        dy = (p1.y - p2.y) * frame_h
        return np.sqrt(dx ** 2 + dy ** 2)
    
    def _calculate_raw_au_measurements(self, landmarks, frame_h, frame_w):
        """Calculate raw measurements for AUs (used for baseline and current frame)."""
        if not landmarks or len(landmarks) < 468:
            return None
        
        norm_factor = self.calculate_face_normalization_factor(landmarks, frame_h, frame_w)
        measurements = {}
        
        try:
            # Get head pose to compensate for head movement
            head_pose = self.estimate_head_pose(landmarks, frame_h, frame_w)
            self.head_pose_buffer.append(head_pose)
            
            # Calculate average head pose over recent frames (for stability)
            if len(self.head_pose_buffer) > 5:
                avg_pose = {
                    'pitch': np.mean([p['pitch'] for p in self.head_pose_buffer]),
                    'yaw': np.mean([p['yaw'] for p in self.head_pose_buffer]),
                    'roll': np.mean([p['roll'] for p in self.head_pose_buffer])
                }
            else:
                avg_pose = head_pose
            
            # Use inter-brow distance as stable reference for brow AUs
            left_inner = landmarks[self.FACS_LANDMARKS['left_inner_brow']]
            right_inner = landmarks[self.FACS_LANDMARKS['right_inner_brow']]
            brow_separation = self.calculate_euclidean_distance(left_inner, right_inner, frame_h, frame_w) / norm_factor
            
            # AU1: Inner Brow Raiser - use vertical distance relative to eye, but compensate for head pitch
            left_eye_top = landmarks[self.FACS_LANDMARKS['left_eye_top']]
            brow_eye_dist = self.calculate_euclidean_distance(left_inner, left_eye_top, frame_h, frame_w) / norm_factor
            
            # Compensate for head pitch (when head tilts up, brows appear higher)
            pitch_compensation = abs(avg_pose['pitch']) * 0.3  # Reduce sensitivity to pitch
            measurements['AU1_raw'] = brow_eye_dist - pitch_compensation
            
            # AU2: Outer Brow Raiser - use distance from outer brow to brow center
            left_outer = landmarks[self.FACS_LANDMARKS['left_outer_brow']]
            left_brow_center = landmarks[self.FACS_LANDMARKS['left_brow_center']]
            outer_brow_dist = self.calculate_euclidean_distance(left_outer, left_brow_center, frame_h, frame_w) / norm_factor
            measurements['AU2_raw'] = outer_brow_dist - pitch_compensation * 0.8
            
            # AU4: Brow Lowerer - use normalized brow separation (less sensitive to head movement)
            # Normalize by baseline brow separation to reduce head movement sensitivity
            measurements['AU4_raw'] = brow_separation
            
            # AU5: Upper Lid Raiser - eye opening height
            measurements['AU5_raw'] = self.calculate_euclidean_distance(
                landmarks[self.FACS_LANDMARKS['left_eye_top']],
                landmarks[self.FACS_LANDMARKS['left_eye_bottom']],
                frame_h, frame_w
            ) / norm_factor
            
            # AU6: Cheek Raiser - distance from cheek to eye corner
            left_cheek = landmarks[self.FACS_LANDMARKS['left_cheek']]
            left_eye_outer = landmarks[self.FACS_LANDMARKS['left_eye_outer']]
            measurements['AU6_raw'] = self.calculate_euclidean_distance(left_cheek, left_eye_outer, frame_h, frame_w) / norm_factor
            
            # AU7: Lid Tightener - eye width
            measurements['AU7_raw'] = self.calculate_euclidean_distance(
                landmarks[self.FACS_LANDMARKS['left_eye_inner']],
                landmarks[self.FACS_LANDMARKS['left_eye_outer']],
                frame_h, frame_w
            ) / norm_factor
            
            # AU9: Nose Wrinkler
            measurements['AU9_raw'] = self.calculate_euclidean_distance(
                landmarks[self.FACS_LANDMARKS['nose_tip']],
                landmarks[self.FACS_LANDMARKS['nose_base']],
                frame_h, frame_w
            ) / norm_factor
            
            # AU10: Upper Lip Raiser
            measurements['AU10_raw'] = self.calculate_euclidean_distance(
                landmarks[self.FACS_LANDMARKS['mouth_top_upper']],
                landmarks[self.FACS_LANDMARKS['nose_base']],
                frame_h, frame_w
            ) / norm_factor
            
            # AU12: Lip Corner Puller (smile) - mouth width
            measurements['AU12_raw'] = self.calculate_euclidean_distance(
                landmarks[self.FACS_LANDMARKS['mouth_left_corner']],
                landmarks[self.FACS_LANDMARKS['mouth_right_corner']],
                frame_h, frame_w
            ) / norm_factor
            
            # AU15: Lip Corner Depressor - mouth height
            measurements['AU15_raw'] = self.calculate_euclidean_distance(
                landmarks[self.FACS_LANDMARKS['mouth_center_top']],
                landmarks[self.FACS_LANDMARKS['mouth_center_bottom']],
                frame_h, frame_w
            ) / norm_factor
            
            # AU25: Lips Part (same as AU15)
            measurements['AU25_raw'] = measurements['AU15_raw']
            
            # AU26: Jaw Drop
            measurements['AU26_raw'] = self.calculate_euclidean_distance(
                landmarks[self.FACS_LANDMARKS['face_bottom']],
                landmarks[self.FACS_LANDMARKS['mouth_center_bottom']],
                frame_h, frame_w
            ) / norm_factor
            
        except (IndexError, KeyError) as e:
            # Landmark index might be out of range
            return None
        
        return measurements
    
    def _update_baseline(self, measurements):
        """Update baseline measurements for calibration."""
        if measurements is None:
            return
        
        self.baseline_samples.append(measurements.copy())
        
        if len(self.baseline_samples) >= self.calibration_samples_needed:
            # Calculate mean baseline
            baseline_keys = [k for k in measurements.keys()]
            self.baseline_aus = {}
            for key in baseline_keys:
                values = [s[key] for s in self.baseline_samples if key in s]
                if values:
                    self.baseline_aus[key] = np.mean(values)
            
            self.baseline_calibrated = True
            print(f"Baseline calibrated with {len(self.baseline_samples)} samples")
    
    def calculate_action_units(self, landmarks, frame_h, frame_w):
        """
        Calculate Facial Action Units (AUs) based on FACS using relative measurements.
        Returns a dictionary of AU intensities (0-1 scale).
        """
        # Get raw measurements
        measurements = self._calculate_raw_au_measurements(landmarks, frame_h, frame_w)
        if measurements is None:
            return None
        
        # Update baseline if not calibrated
        if not self.baseline_calibrated:
            self._update_baseline(measurements)
            # Return zeros during calibration
            return {f'AU{i}': 0.0 for i in [1, 2, 4, 5, 6, 7, 9, 10, 12, 15, 20, 25]}
        
        aus = {}
        
        # Calculate relative changes from baseline with reduced sensitivity
        # AU1: Inner Brow Raiser (increase = brow raise)
        baseline = self.baseline_aus.get('AU1_raw', measurements['AU1_raw'])
        change = (measurements['AU1_raw'] - baseline) / (baseline + 0.01)
        # Reduce sensitivity: only activate if change is significant (>5%)
        change_threshold = 0.05
        if abs(change) > change_threshold:
            aus['AU1'] = np.clip(0.5 + (change - np.sign(change) * change_threshold) * 3, 0.0, 1.0)
        else:
            aus['AU1'] = 0.5  # Neutral
        
        # AU2: Outer Brow Raiser
        baseline = self.baseline_aus.get('AU2_raw', measurements['AU2_raw'])
        change = (measurements['AU2_raw'] - baseline) / (baseline + 0.01)
        if abs(change) > change_threshold:
            aus['AU2'] = np.clip(0.5 + (change - np.sign(change) * change_threshold) * 3, 0.0, 1.0)
        else:
            aus['AU2'] = 0.5  # Neutral
        
        # AU4: Brow Lowerer (decrease in brow separation = frown)
        baseline = self.baseline_aus.get('AU4_raw', measurements['AU4_raw'])
        # Use larger threshold for AU4 since it's more stable
        change_au4 = (baseline - measurements['AU4_raw']) / (baseline + 0.01)
        change_threshold_au4 = 0.08  # Higher threshold for AU4
        if change_au4 > change_threshold_au4:
            aus['AU4'] = np.clip((change_au4 - change_threshold_au4) * 4, 0.0, 1.0)
        else:
            aus['AU4'] = 0.0  # No frown detected
        
        # AU5: Upper Lid Raiser (increase = eyes wider)
        baseline = self.baseline_aus.get('AU5_raw', measurements['AU5_raw'])
        change = (measurements['AU5_raw'] - baseline) / (baseline + 0.01)
        aus['AU5'] = np.clip(0.5 + change * 5, 0.0, 1.0)
        
        # AU6: Cheek Raiser (increase = smile/squint)
        baseline = self.baseline_aus.get('AU6_raw', measurements['AU6_raw'])
        change = (measurements['AU6_raw'] - baseline) / (baseline + 0.01)
        aus['AU6'] = np.clip(0.5 + change * 3, 0.0, 1.0)
        
        # AU7: Lid Tightener
        baseline = self.baseline_aus.get('AU7_raw', measurements['AU7_raw'])
        change = (measurements['AU7_raw'] - baseline) / (baseline + 0.01)
        aus['AU7'] = np.clip(0.5 + change * 3, 0.0, 1.0)
        
        # AU9: Nose Wrinkler
        baseline = self.baseline_aus.get('AU9_raw', measurements['AU9_raw'])
        change = (baseline - measurements['AU9_raw']) / (baseline + 0.01)
        aus['AU9'] = np.clip(change * 5, 0.0, 1.0)
        
        # AU10: Upper Lip Raiser
        baseline = self.baseline_aus.get('AU10_raw', measurements['AU10_raw'])
        change = (measurements['AU10_raw'] - baseline) / (baseline + 0.01)
        aus['AU10'] = np.clip(0.5 + change * 5, 0.0, 1.0)
        
        # AU12: Lip Corner Puller (increase = smile)
        baseline = self.baseline_aus.get('AU12_raw', measurements['AU12_raw'])
        change = (measurements['AU12_raw'] - baseline) / (baseline + 0.01)
        aus['AU12'] = np.clip(0.5 + change * 3, 0.0, 1.0)
        
        # AU15: Lip Corner Depressor (decrease = frown)
        baseline = self.baseline_aus.get('AU15_raw', measurements['AU15_raw'])
        change = (baseline - measurements['AU15_raw']) / (baseline + 0.01)
        aus['AU15'] = np.clip(change * 5, 0.0, 1.0)
        
        # AU20: Lip Stretcher (similar to AU12)
        aus['AU20'] = aus['AU12']
        
        # AU25: Lips Part
        baseline = self.baseline_aus.get('AU25_raw', measurements['AU25_raw'])
        change = (measurements['AU25_raw'] - baseline) / (baseline + 0.01)
        aus['AU25'] = np.clip(0.5 + change * 3, 0.0, 1.0)
        
        # AU26: Jaw Drop
        baseline = self.baseline_aus.get('AU26_raw', measurements['AU26_raw'])
        change = (measurements['AU26_raw'] - baseline) / (baseline + 0.01)
        aus['AU26'] = np.clip(0.5 + change * 3, 0.0, 1.0)
        
        # AU27: Mouth Stretch
        aus['AU27'] = min(aus['AU26'] * 1.2, 1.0)
        
        # Apply temporal smoothing to reduce jitter from head movement
        self.previous_aus.append(aus.copy())
        
        if len(self.previous_aus) >= 3:
            # Smooth by taking median of recent values (more robust than mean)
            smoothed_aus = {}
            for au_name in aus.keys():
                values = [prev_aus.get(au_name, 0.0) for prev_aus in self.previous_aus]
                # Use weighted average: recent frames more important
                weights = np.exp(np.linspace(-2, 0, len(values)))
                weights = weights / weights.sum()
                smoothed_aus[au_name] = np.average(values, weights=weights)
            return smoothed_aus
        
        return aus
    
    def create_temporal_window(self):
        """
        Create a temporal window of multi-modal features for LSTM input.
        Returns a numpy array of shape (window_size, features) or None if insufficient data.
        """
        if len(self.au_buffer) < self.window_size_frames:
            return None
        
        # Extract the most recent window_size_frames
        recent_aus = list(self.au_buffer)[-self.window_size_frames:]
        recent_bpm = list(self.bpm_history)[-self.window_size_frames:]
        
        # Normalize BPM to 0-1 range (assuming 50-150 BPM range)
        normalized_bpm = [(bpm - 50) / 100.0 for bpm in recent_bpm] if recent_bpm else [0.5] * self.window_size_frames
        
        # Combine features: [AU1, AU2, AU4, AU5, AU6, AU7, AU9, AU10, AU12, AU15, AU20, AU25, BPM]
        features = []
        for aus, bpm_norm in zip(recent_aus, normalized_bpm):
            if aus is None:
                # Fill with zeros if no AU data
                feature_vector = [0.0] * 12 + [bpm_norm]
            else:
                feature_vector = [
                    aus.get('AU1', 0.0),
                    aus.get('AU2', 0.0),
                    aus.get('AU4', 0.0),  # Key stress indicator
                    aus.get('AU5', 0.0),
                    aus.get('AU6', 0.0),
                    aus.get('AU7', 0.0),
                    aus.get('AU9', 0.0),
                    aus.get('AU10', 0.0),
                    aus.get('AU12', 0.0),
                    aus.get('AU15', 0.0),
                    aus.get('AU20', 0.0),
                    aus.get('AU25', 0.0),
                    bpm_norm
                ]
            features.append(feature_vector)
        
        return np.array(features, dtype=np.float32)
    
    def predict_stress_with_lstm(self, temporal_window):
        """
        LSTM prediction - DISABLED (model removed).
        Returns 0.0 as LSTM model has been removed.
        """
        # LSTM model has been completely removed
        return 0.0
    
    def apply_robust_normalization(self, frame):
        """Stabilizes lighting using CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def calculate_heart_rate(self):
        """Calculate heart rate from signal buffer using FFT with smoothing."""
        if len(self.signal_buffer) < 60:  # Need at least 2 seconds of data
            return 0
        
        # Get signal data
        sig = np.array(self.signal_buffer)
        
        # Detrend the signal (remove DC component and slow drift)
        sig = signal.detrend(sig)
        
        # Apply bandpass filter (0.7-3 Hz, which is 48-180 BPM) - tighter range
        nyquist = self.fps / 2
        low = 0.7 / nyquist  # More restrictive lower bound
        high = 3.0 / nyquist  # More restrictive upper bound
        b, a = signal.butter(5, [low, high], btype='band')
        sig_filtered = signal.filtfilt(b, a, sig)
        
        # FFT to find dominant frequency
        fft = np.fft.rfft(sig_filtered)
        freqs = np.fft.rfftfreq(len(sig_filtered), 1.0 / self.fps)
        
        # Find peak in frequency range corresponding to 50-150 BPM
        bpm_range = (freqs >= 50/60) & (freqs <= 150/60)
        if not np.any(bpm_range):
            return 0
        
        power = np.abs(fft[bpm_range])
        peak_freq_idx = np.argmax(power)
        peak_freq = freqs[bpm_range][peak_freq_idx]
        
        # Convert frequency to BPM
        raw_bpm = peak_freq * 60
        raw_bpm = np.clip(raw_bpm, 50, 150)  # Clip to reasonable range
        
        # Outlier rejection: if we have previous readings, reject if change is too large
        if len(self.bpm_buffer) > 0:
            recent_bpm = np.median(list(self.bpm_buffer))
            # Reject if change is more than 20 BPM (likely noise)
            if abs(raw_bpm - recent_bpm) > 20:
                # Use the last stable reading instead
                return int(recent_bpm)
        
        # Add to buffer for smoothing
        self.bpm_buffer.append(raw_bpm)
        
        # Return median of recent readings for stability
        if len(self.bpm_buffer) >= 3:
            smoothed_bpm = np.median(list(self.bpm_buffer))
            return int(smoothed_bpm)
        else:
            return int(raw_bpm)
    
    def draw_graph(self, frame):
        """Draws a real-time scrolling signal graph at the bottom of the screen."""
        if len(self.signal_buffer) < 2: return frame
        
        h, w, _ = frame.shape
        graph_h = 100
        margin = 10
        
        # Draw background for graph
        cv2.rectangle(frame, (margin, h - graph_h - margin), (w - margin, h - margin), (50, 50, 50), -1)
        
        # Normalize signal for display
        sig = np.array(self.signal_buffer)
        sig = (sig - np.min(sig)) / (np.max(sig) - np.min(sig) + 1e-6)
        
        # Draw the line
        for i in range(len(sig) - 1):
            x1 = int(margin + (i / self.buffer_size) * (w - 2 * margin))
            y1 = int((h - margin) - sig[i] * graph_h)
            x2 = int(margin + ((i + 1) / self.buffer_size) * (w - 2 * margin))
            y2 = int((h - margin) - sig[i+1] * graph_h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            # 1. Lighting Normalization
            norm_frame = self.apply_robust_normalization(frame)
            rgb_frame = cv2.cvtColor(norm_frame, cv2.COLOR_BGR2RGB)
            
            # Convert to MediaPipe Image format
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Process frame with FaceLandmarker
            detection_result = self.face_landmarker.detect_for_video(mp_image, self.frame_timestamp_ms)
            self.frame_timestamp_ms += 33  # Approximate 30 FPS

            if detection_result.face_landmarks:
                landmarks = detection_result.face_landmarks[0]
                h, w, _ = frame.shape
                
                # 2. ROI Masking & Signal Extraction
                mask = np.zeros((h, w), dtype=np.uint8)
                points = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in self.ROI_IDXS])
                cv2.fillPoly(mask, [points], 255)
                
                # Extract mean green channel (Signal)
                mean_val = cv2.mean(frame, mask=mask)[1]
                self.signal_buffer.append(mean_val)

                # 3. Calculate Facial Action Units (FACS)
                action_units = self.calculate_action_units(landmarks, h, w)
                
                # 4. Calculate and display heart rate
                self.current_bpm = self.calculate_heart_rate()
                
                # 5. Collect temporal data for multi-modal fusion
                self.au_buffer.append(action_units)
                if self.current_bpm > 0:
                    self.bpm_history.append(self.current_bpm)
                else:
                    self.bpm_history.append(0)
                
                # 6. LSTM prediction disabled (model removed)
                temporal_window = self.create_temporal_window()
                self.stress_probability = 0.0  # LSTM disabled
                
                # 7. Visual Feedback: Draw the ROI box/points
                for pt in points:
                    cv2.circle(frame, tuple(pt), 1, (0, 255, 255), -1)
                
                # 8. Draw the Live Graph
                frame = self.draw_graph(frame)
                
                # 9. Display status and heart rate
                if not self.baseline_calibrated:
                    calibration_progress = len(self.baseline_samples) / self.calibration_samples_needed
                    status_text = f"STATUS: Calibrating baseline... {calibration_progress:.0%}"
                    cv2.putText(frame, status_text, (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    status_text = "STATUS: Multi-Modal Tracking Active"
                    cv2.putText(frame, status_text, (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display heart rate with larger, prominent text
                if self.current_bpm > 0:
                    # Check stability (lower standard deviation = more stable)
                    if len(self.bpm_buffer) >= 5:
                        bpm_std = np.std(list(self.bpm_buffer))
                        stability_color = (0, 255, 0) if bpm_std < 5 else (0, 255, 255) if bpm_std < 10 else (0, 165, 255)
                        stability_text = "Stable" if bpm_std < 5 else "Stabilizing" if bpm_std < 10 else "Unstable"
                    else:
                        stability_color = (0, 255, 255)
                        stability_text = "Stabilizing"
                    
                    bpm_text = f"HEART RATE: {self.current_bpm} BPM"
                    text_size = cv2.getTextSize(bpm_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                    text_x = (w - text_size[0]) // 2
                    text_y = 100
                    
                    # Draw background rectangle for better visibility
                    cv2.rectangle(frame, (text_x - 10, text_y - 40), (text_x + text_size[0] + 10, text_y + 10), 
                                 (0, 0, 0), -1)
                    cv2.putText(frame, bpm_text, (text_x, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, stability_color, 3)
                    
                    # Display stability indicator
                    if len(self.bpm_buffer) >= 5:
                        stability_size = cv2.getTextSize(stability_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        stability_x = (w - stability_size[0]) // 2
                        cv2.putText(frame, stability_text, (stability_x, text_y + 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, stability_color, 2)
                    
                    # Display Action Units and Stress Probability
                    if action_units and self.baseline_calibrated:
                        au_y = text_y + 70
                        # Display key AUs
                        au1_text = f"AU1 (Brow Raise): {action_units['AU1']:.2f}"
                        au2_text = f"AU2 (Outer Brow): {action_units['AU2']:.2f}"
                        au4_text = f"AU4 (Brow Lowerer): {action_units['AU4']:.2f}"
                        au12_text = f"AU12 (Smile): {action_units['AU12']:.2f}"
                        
                        # Display with color coding
                        cv2.putText(frame, au1_text, (20, au_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if action_units['AU1'] > 0.6 else (255, 255, 255), 2)
                        cv2.putText(frame, au2_text, (20, au_y + 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if action_units['AU2'] > 0.6 else (255, 255, 255), 2)
                        cv2.putText(frame, au4_text, (20, au_y + 40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255) if action_units['AU4'] > 0.3 else (255, 255, 255), 2)
                        cv2.putText(frame, au12_text, (20, au_y + 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if action_units['AU12'] > 0.6 else (255, 255, 255), 2)
                        
                        # LSTM stress prediction disabled (model removed)
                        cv2.putText(frame, "LSTM: Disabled (model removed)", (20, au_y + 25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
                else:
                    cv2.putText(frame, "Calculating heart rate...", (20, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow('rPPG Real-Time Tracker', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()
    
    def export_training_data(self, output_path, label=None):
        """
        Export collected temporal windows for training.
        
        Args:
            output_path: Path to save the data (numpy format)
            label: Optional label (0=normal, 1=stress/deception) for supervised learning
        """
        temporal_window = self.create_temporal_window()
        if temporal_window is not None:
            data = {
                'features': temporal_window,
                'label': label,
                'timestamp': self.frame_timestamp_ms
            }
            np.save(output_path, data)
            print(f"Training data exported to {output_path}")
            return True
        return False
    
    def prepare_dataset_for_training(self, data_dir, output_dir):
        """
        Prepare dataset from CASME II / SAMM / SWELL format for training.
        This is a placeholder - implement according to your dataset structure.
        
        For CASME II / SAMM:
        - Load video files
        - Extract frames
        - Calculate AUs and BPM for each frame
        - Create temporal windows
        - Label based on ground truth annotations
        
        For SWELL:
        - Load video + ECG/PPG data
        - Sync video frames with physiological data
        - Extract features
        - Use stress labels from dataset
        """
        print("Dataset preparation function - implement based on your dataset format")
        print("Expected datasets:")
        print("  - CASME II: High-speed video of spontaneous micro-expressions")
        print("  - SAMM: Spontaneous Actions and Micro-Movements dataset")
        print("  - SWELL: Video + physiological sensor data with stress labels")
        pass


class LSTMTrainer:
    """
    Helper class for training the LSTM model on collected or dataset data.
    """
    def __init__(self, model=None):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required for LSTM training")
        self.model = model
    
    def train_from_windows(self, X_train, y_train, X_val=None, y_val=None, 
                          epochs=50, batch_size=32):
        """
        Train LSTM model on temporal windows.
        
        Args:
            X_train: Training features, shape (samples, timesteps, features)
            y_train: Training labels, shape (samples,)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        if not TF_AVAILABLE:
            print("TensorFlow not available for training")
            return None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        return history
    
    def save_model(self, model_path):
        """Save trained model to file."""
        if self.model:
            self.model.save(model_path)
            print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """Load pre-trained model - DISABLED (LSTM removed)."""
        print("⚠ LSTM model loading disabled - model has been removed")
        return None


if __name__ == "__main__":
    visualizer = RPPGVisualizer()
    print("\n" + "="*60)
    print("Multi-Modal Stress/Deception Detection System")
    print("="*60)
    print("\nFeatures:")
    print("  ✓ rPPG Heart Rate Detection")
    print("  ✓ FACS Action Units (AU) Calculation")
    print("  ✓ Normalized Distance Vectors")
    print("  ✓ Temporal Windowing (7-second windows)")
    print("  ✗ LSTM Multi-Modal Fusion (DISABLED - model removed)")
    print("\nControls:")
    print("  - Press 'q' to quit")
    print("  - Keep face visible and still for best results")
    print("\n" + "="*60 + "\n")
    visualizer.run()