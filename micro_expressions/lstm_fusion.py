"""
LSTM Fusion Module - DEPRECATED
This module has been disabled as the LSTM model was unreliable.
The module is kept for compatibility but does nothing.
"""

import cv2


class LSTMFusionModule:
    """Deprecated module - LSTM model has been removed."""
    
    def __init__(self, model_path=None):
        print("⚠ LSTM module is disabled - model has been removed")
        self.score = 0.0

    def initialize(self, shared_state):
        pass

    def process(self, shared_state):
        # Module disabled - no processing
        shared_state['stress_probability'] = 0.0
        pass

    def render(self, frame, shared_state):
        # Module disabled - no rendering
        return frame
        
    def cleanup(self):
        pass