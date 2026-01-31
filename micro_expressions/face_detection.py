import mediapipe as mp
import cv2
import numpy as np
import os
import urllib.request
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class FaceDetectionModule:
    """
    UPGRADED: Uses MediaPipe's 'FaceLandmarker' with pre-trained Blendshapes.
    Visualization: Draws simple dots using OpenCV, matching the reference style.
    """
    def __init__(self):
        model_path = 'face_landmarker.task'
        
        # Auto-download model if missing
        if not os.path.exists(model_path):
            print("Downloading MediaPipe Face Landmarker model...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print("Download complete!")

        # Configure for Blendshapes
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.VIDEO)
            
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.timestamp_ms = 0

    def initialize(self, shared_state):
        return True

    def process(self, shared_state):
        frame = shared_state['frame']
        if frame is None: return

        # MediaPipe needs RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Increment timestamp
        self.timestamp_ms += 33
        
        # Detect
        result = self.detector.detect_for_video(mp_image, self.timestamp_ms)
        
        if result.face_landmarks:
            shared_state['face_detected'] = True
            shared_state['landmarks'] = result.face_landmarks[0]
            if result.face_blendshapes:
                shared_state['blendshapes'] = result.face_blendshapes[0]
        else:
            shared_state['face_detected'] = False
            shared_state['landmarks'] = None
            shared_state['blendshapes'] = None

    def render(self, frame, shared_state):
        # Check if minimal mode is enabled
        minimal_mode = shared_state.get('minimal_mode', False)
        if minimal_mode:
            # In minimal mode, don't render face landmarks (clean view)
            return frame
        
        if not shared_state.get('face_detected'):
            cv2.putText(frame, "NO FACE DETECTED", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return frame

        landmarks = shared_state.get('landmarks')
        if landmarks:
            h, w = frame.shape[:2]
            
            # --- VISUALIZATION STYLE MATCHING YOUR REFERENCE ---
            # Instead of using mp_drawing, we manually iterate and draw circles.
            # This avoids import errors and gives the specific look you requested.
            
            # Convert all landmarks to a numpy array of points (x, y)
            points = np.array([
                [int(lm.x * w), int(lm.y * h)] 
                for lm in landmarks
            ])

            # Draw the points
            for pt in points:
                # Color: (0, 255, 255) is Yellow/Cyan (BGR)
                # Radius: 1
                # Thickness: -1 (Filled)
                cv2.circle(frame, tuple(pt), 1, (0, 255, 255), -1)

            cv2.putText(frame, "TRACKING ACTIVE", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                       
        return frame

    def cleanup(self):
        self.detector.close()