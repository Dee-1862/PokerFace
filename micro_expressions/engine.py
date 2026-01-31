import cv2
import time
import numpy as np

class StressDetectionEngine:
    """
    Central coordinator for the detection pipeline.
    Manages the camera feed and sequential execution of modules.
    """
    def __init__(self, config):
        self.camera_id = config.get('camera_id', 0)
        self.debug_mode = config.get('debug_mode', False)
        self.is_video_file = config.get('is_video_file', False)
        self.modules = {}
        self.shared_state = {
            'frame': None,
            'frame_dimensions': (0, 0),
            'face_detected': False,
            'landmarks': None,
            'timestamp': 0,
            'is_video_file': self.is_video_file,
            'frame_number': 0,
            'video_fps': 0,
            'video_duration': 0
        }

    def attach_module(self, name, module_instance):
        """Register a processing module to the pipeline."""
        if hasattr(module_instance, 'initialize'):
            module_instance.initialize(self.shared_state)
        self.modules[name] = module_instance

    def run(self):
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            raise IOError(f"Cannot open camera {self.camera_id}")

        print("Engine started. Press 'q' to exit.")
        
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Update Shared State
            h, w = frame.shape[:2]
            self.shared_state.update({
                'frame': frame,
                'frame_dimensions': (h, w),
                'timestamp': time.time()
            })

            # Pipeline Execution
            # 1. Process Phase (Analysis)
            for name, module in self.modules.items():
                module.process(self.shared_state)

            # 2. Render Phase (Visualization)
            display_frame = frame.copy()
            for name, module in self.modules.items():
                display_frame = module.render(display_frame, self.shared_state)

            # Display
            cv2.imshow('Multi-Modal Stress Detection', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        
        # Cleanup
        for module in self.modules.values():
            if hasattr(module, 'cleanup'):
                module.cleanup()