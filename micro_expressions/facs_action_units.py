import cv2

class FACSModule:
    """
    Maps MediaPipe's pre-trained blendshapes to FACS AUs.
    Provides human-readable descriptions for expressions.
    """
    def __init__(self):
        # Each AU maps to (left_blendshape, right_blendshape_or_None, gain)
        # We take max(left, right) * gain so bilateral expressions register
        # regardless of which side the person favours.
        # Gains compensate for AUs where MediaPipe's raw score tops out low.
        self.MAPPING = {
            'AU1':  ('browInnerUp',           None,                    1.0),
            'AU2':  ('browOuterUpLeft',        'browOuterUpRight',      1.0),
            'AU4':  ('browDownLeft',           'browDownRight',         1.3),  # understated by MP
            'AU5':  ('eyeWideLeft',            'eyeWideRight',          1.0),
            'AU6':  ('cheekSquintLeft',        'cheekSquintRight',      1.1),
            'AU12': ('mouthSmileLeft',         'mouthSmileRight',       1.0),
            'AU15': ('mouthFrownLeft',         'mouthFrownRight',       1.4),  # often very weak
            'AU17': ('mouthLowerDownLeft',     'mouthLowerDownRight',   1.1),
            'AU20': ('mouthStretchLeft',       'mouthStretchRight',     1.0),
            'AU23': ('mouthPressLeft',         'mouthPressRight',       1.0),
            'AU26': ('jawOpen',                None,                    1.5),  # MP caps low
            'AU45': ('eyeBlinkLeft',           'eyeBlinkRight',         1.0),
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
        bs_dict = {b.category_name: b.score for b in blendshapes}

        current_aus = {}
        for au_name, (left_bs, right_bs, gain) in self.MAPPING.items():
            val = bs_dict.get(left_bs, 0.0)
            if right_bs:
                val = max(val, bs_dict.get(right_bs, 0.0))
            current_aus[au_name] = min(1.0, val * gain)

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