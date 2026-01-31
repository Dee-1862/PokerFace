import cv2

class FACSModule:
    """
    Maps MediaPipe's pre-trained blendshapes to FACS AUs.
    Provides human-readable descriptions for expressions.
    """
    def __init__(self):
        # Map MP Blendshape names to standard AU names
        self.MAPPING = {
            'AU1': 'browInnerUp',
            'AU2': 'browOuterUpLeft', 
            'AU4': 'browDownLeft',
            'AU5': 'eyeWideLeft',       # Eye Widening (surprise/fear)
            'AU6': 'cheekSquintLeft',   # Cheek Raise
            'AU12': 'mouthSmileLeft',
            'AU15': 'mouthFrownLeft',   # Lip Corner Depressor
            'AU17': 'mouthLowerDownLeft', # Chin Raise
            'AU20': 'mouthStretchLeft', # Lip Stretch
            'AU23': 'mouthPressLeft',   # Lip Tightener (stress indicator)
            'AU26': 'jawOpen',
            'AU45': 'eyeBlinkLeft'
        }
        
        # Human-Readable Descriptions
        self.DISPLAY_NAMES = {
            'AU1': 'Inner Brow Raise',
            'AU2': 'Outer Brow Raise',
            'AU4': 'Frown / Anger',
            'AU5': 'Eye Widening',
            'AU6': 'Cheek Raise',
            'AU12': 'Smile / Happiness',
            'AU15': 'Lip Corner Down',
            'AU17': 'Chin Raise',
            'AU20': 'Lip Stretch',
            'AU23': 'Lip Tightener',
            'AU26': 'Jaw Drop / Shock',
            'AU45': 'Blink'
        }
        
        # Stress-related AUs (for quick reference)
        self.STRESS_AUS = ['AU1', 'AU2', 'AU4', 'AU5', 'AU15', 'AU17', 'AU20', 'AU23']


    def initialize(self, shared_state):
        pass

    def process(self, shared_state):
        if not shared_state.get('blendshapes'):
            shared_state['action_units'] = {}
            return

        blendshapes = shared_state['blendshapes']
        # Convert MP's list to dictionary
        bs_dict = {b.category_name: b.score for b in blendshapes}
        
        current_aus = {}
        for au_name, mp_name in self.MAPPING.items():
            val = bs_dict.get(mp_name, 0.0)
            current_aus[au_name] = val

        shared_state['action_units'] = current_aus

    def render(self, frame, shared_state):
        # Check if minimal mode is enabled
        minimal_mode = shared_state.get('minimal_mode', False)
        if minimal_mode:
            # In minimal mode, don't render anything (AR controller handles it)
            return frame
        
        aus = shared_state.get('action_units', {})
        if not aus:
            return frame

        y = 110
        cv2.putText(frame, "FACIAL ACTIONS:", (20, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        
        for au_code, val in aus.items():
            # Get the readable name
            desc = self.DISPLAY_NAMES.get(au_code, au_code)
            
            # Visual feedback: Turn Green if activated (> 0.4)
            color = (0, 255, 0) if val > 0.4 else (180, 180, 180)
            
            # Format: "Inner Brow Raise: 0.85"
            text = f"{desc}: {val:.2f}"
            
            cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y += 20
            
        return frame

    def cleanup(self):
        pass