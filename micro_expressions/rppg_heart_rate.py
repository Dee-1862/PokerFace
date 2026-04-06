import cv2
import numpy as np
from collections import deque
import time

try:
    from vitallens import VitalLens, Method
    _VITALLENS_OK = True
except ImportError:
    _VITALLENS_OK = False


class RPPGModule:
    """
    Heart rate estimation via VitalLens (POS algorithm, runs locally).

    Uses the vitallens library's streaming API to push frames and receive
    rolling BPM and PPG waveform. HRV (RMSSD) is computed from the returned
    waveform's peak intervals.

    Falls back to a simple green-channel FFT if vitallens is not installed.

    Outputs added to shared_state:
        heart_rate_bpm  - smoothed BPM (int)
        hrv_rmssd       - RMSSD in milliseconds (float or None until stable)
    """

    def __init__(self, buffer_size=300, fps=30, enable_validation=False):
        self.buffer_size = buffer_size
        self.nominal_fps = fps
        self.current_bpm = 0
        self.hrv_rmssd = None
        self.bpm_buffer = deque(maxlen=10)

        # VitalLens streaming session
        self._vl = None
        self._session = None
        self._vl_ready = False
        if _VITALLENS_OK:
            try:
                self._vl = VitalLens(
                    method=Method.POS,
                    detect_faces=False,
                    estimate_rolling_vitals=True,
                    export_to_json=False,
                )
                print("[OK] VitalLens POS engine loaded")
            except Exception as e:
                print(f"[WARN] VitalLens init failed: {e}")
                self._vl = None

        # Green-channel fallback buffers (used only if VitalLens unavailable)
        self._green_buffer = deque(maxlen=buffer_size)
        self._ts_buffer = deque(maxlen=buffer_size)
        self._actual_fps = float(fps)

        # Face ROI landmark indices
        self.ROI_IDXS = [
            10, 338, 297, 332, 284, 251, 389, 356, 454,
            323, 361, 288, 397, 365, 379, 378, 400, 377,
            152, 148, 176, 149, 150, 136,
        ]

        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        # Validation support
        self.enable_validation = enable_validation
        self.validator = None
        if enable_validation:
            try:
                from heart_rate_validator import HeartRateValidator
                self.validator = HeartRateValidator()
                print("[OK] Heart rate validation enabled")
            except ImportError:
                print("[WARN] Validation module not available")
                self.enable_validation = False

    def initialize(self, shared_state):
        if self._vl and not self._session:
            try:
                self._session = self._vl.stream().__enter__()
                self._vl_ready = True
                print("[OK] VitalLens stream session started")
            except Exception as e:
                print(f"[WARN] VitalLens stream failed: {e}")
                self._vl_ready = False

    def _normalize_lighting(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = self.clahe.apply(l)
        return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    def _get_face_box(self, landmarks, h, w):
        """Convert face landmarks to [x0, y0, x1, y1] bounding box."""
        xs = [landmarks[i].x * w for i in self.ROI_IDXS]
        ys = [landmarks[i].y * h for i in self.ROI_IDXS]
        pad_x = (max(xs) - min(xs)) * 0.3
        pad_y = (max(ys) - min(ys)) * 0.3
        x0 = max(0, int(min(xs) - pad_x))
        y0 = max(0, int(min(ys) - pad_y))
        x1 = min(w, int(max(xs) + pad_x))
        y1 = min(h, int(max(ys) + pad_y))
        return [x0, y0, x1, y1]

    def process(self, shared_state):
        if not shared_state.get('face_detected'):
            return

        frame = shared_state['frame']
        landmarks = shared_state['landmarks']
        h, w = shared_state['frame_dimensions']
        now = time.monotonic()

        if self._vl_ready and self._session:
            self._process_vitallens(frame, landmarks, h, w, now, shared_state)
        else:
            self._process_fallback(frame, landmarks, h, w, now, shared_state)

    def _process_vitallens(self, frame, landmarks, h, w, now, shared_state):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_box = self._get_face_box(landmarks, h, w)

        try:
            self._session.push(rgb_frame, now, face=np.array(face_box))
        except Exception:
            self._process_fallback(frame, landmarks, h, w, now, shared_state)
            return

        result = None
        try:
            result = self._session.get_result(block=False)
        except Exception:
            pass

        if result and len(result) > 0:
            face_result = result[0]
            vitals = face_result.get('vitals', {})

            hr_info = vitals.get('heart_rate', {})
            bpm = hr_info.get('value', 0)
            if bpm and bpm > 0:
                if len(self.bpm_buffer) >= self.bpm_buffer.maxlen:
                    recent_med = float(np.median(list(self.bpm_buffer)))
                    if abs(bpm - recent_med) > 25:
                        bpm = recent_med
                self.bpm_buffer.append(bpm)
                self.current_bpm = int(np.median(list(self.bpm_buffer))) if len(self.bpm_buffer) >= 3 else int(bpm)

            # Extract HRV from PPG waveform if available
            waveforms = face_result.get('waveforms', {})
            ppg = waveforms.get('ppg_waveform', {})
            ppg_data = ppg.get('data', [])
            if len(ppg_data) > 60:
                self.hrv_rmssd = self._compute_hrv_from_waveform(np.array(ppg_data), self.nominal_fps)

        shared_state['heart_rate_bpm'] = self.current_bpm
        shared_state['hrv_rmssd'] = self.hrv_rmssd

    def _process_fallback(self, frame, landmarks, h, w, now, shared_state):
        """Green-channel FFT fallback when VitalLens is not available."""
        frame_norm = self._normalize_lighting(frame)
        mask = np.zeros((h, w), dtype=np.uint8)
        points = np.array(
            [[int(landmarks[i].x * w), int(landmarks[i].y * h)]
             for i in self.ROI_IDXS]
        )
        cv2.fillPoly(mask, [points], 255)
        means = cv2.mean(frame_norm, mask=mask)
        g_mean = means[1]

        self._green_buffer.append(g_mean)
        self._ts_buffer.append(now)

        if len(self._ts_buffer) >= 2:
            elapsed = self._ts_buffer[-1] - self._ts_buffer[0]
            if elapsed > 0:
                self._actual_fps = (len(self._ts_buffer) - 1) / elapsed

        if len(self._green_buffer) >= 150:
            self.current_bpm, self.hrv_rmssd = self._green_fft()

        shared_state['heart_rate_bpm'] = self.current_bpm
        shared_state['hrv_rmssd'] = self.hrv_rmssd

    def _green_fft(self):
        from scipy import signal as sig
        fps = max(1.0, self._actual_fps)
        raw = np.array(list(self._green_buffer))
        raw = sig.detrend(raw)
        nyq = fps / 2.0
        lo = 0.7 / nyq
        hi = min(3.0 / nyq, 0.98)
        try:
            b, a = sig.butter(5, [lo, hi], btype='band')
            filtered = sig.filtfilt(b, a, raw)
        except ValueError:
            return self.current_bpm, self.hrv_rmssd

        fft = np.fft.rfft(filtered)
        freqs = np.fft.rfftfreq(len(filtered), 1.0 / fps)
        mask = (freqs >= 50 / 60) & (freqs <= 180 / 60)
        if not np.any(mask):
            return 0, None
        power = np.abs(fft[mask])
        peak_f = freqs[mask][np.argmax(power)]
        bpm = float(np.clip(peak_f * 60, 50, 180))

        self.bpm_buffer.append(bpm)
        smoothed = int(np.median(list(self.bpm_buffer))) if len(self.bpm_buffer) >= 3 else int(bpm)
        hrv = self._compute_hrv_from_waveform(filtered, fps)
        return smoothed, hrv

    def _compute_hrv_from_waveform(self, waveform, fps):
        """Detect peaks in a pulse waveform and compute RMSSD (ms)."""
        from scipy import signal as sig
        if len(waveform) < int(fps * 4):
            return self.hrv_rmssd

        min_dist = max(int(fps * 60 / 180), 5)
        peaks, _ = sig.find_peaks(
            waveform,
            distance=min_dist,
            prominence=np.std(waveform) * 0.3,
        )

        if len(peaks) < 4:
            return self.hrv_rmssd

        ibi = np.diff(peaks) / fps
        ibi = ibi[(ibi >= 0.33) & (ibi <= 2.0)]
        if len(ibi) < 3:
            return self.hrv_rmssd

        diff_ibi = np.diff(ibi) * 1000
        rmssd = float(np.sqrt(np.mean(diff_ibi ** 2)))
        return round(rmssd, 1)

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self, frame, shared_state):
        if shared_state.get('minimal_mode', False):
            return frame

        bpm = self.current_bpm
        hrv = self.hrv_rmssd

        if len(self.bpm_buffer) >= 5:
            std_dev = np.std(list(self.bpm_buffer))
            color = (0, 255, 0) if std_dev < 5 else (0, 255, 255)
            status = "Stable" if std_dev < 5 else "Stabilizing"
        else:
            color, status = (0, 255, 255), "Initializing"

        engine_tag = "VitalLens" if self._vl_ready else "Fallback"
        cv2.putText(frame, f"HEART RATE: {bpm} BPM",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"{status}  [{engine_tag}]",
                    (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if hrv is not None:
            hrv_color = (
                (0, 255, 0) if hrv > 30 else
                (0, 255, 255) if hrv > 15 else
                (0, 120, 255)
            )
            cv2.putText(frame, f"HRV: {hrv:.0f} ms",
                        (20, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hrv_color, 1)

        if self.enable_validation and self.validator:
            if self.validator.is_recording:
                val_text = f"VALIDATING: {len(self.validator.rppg_readings)} readings"
                cv2.putText(frame, val_text, (20, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                if len(self.validator.rppg_readings) >= 10:
                    summary = self.validator.get_summary_string()
                    cv2.putText(frame, summary, (20, 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        h_f, w_f = frame.shape[:2]
        cv2.putText(frame, "Research Tool - Not Medical Device",
                    (20, h_f - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)

        self._draw_graph(frame)
        return frame

    def _draw_graph(self, frame):
        if len(self._green_buffer) < 2 and not self._vl_ready:
            return
        h, w = frame.shape[:2]
        graph_h, margin = 60, 20

        cv2.rectangle(frame,
                      (margin, h - graph_h - margin),
                      (w - margin, h - margin),
                      (50, 50, 50), -1)

        if len(self._green_buffer) >= 2:
            sig = np.array(list(self._green_buffer))
            rng = np.max(sig) - np.min(sig)
            if rng == 0:
                return
            sig = (sig - np.min(sig)) / rng
            buf_len = len(sig)
            for i in range(buf_len - 1):
                x1 = int(margin + (i / self.buffer_size) * (w - 2 * margin))
                y1 = int((h - margin) - sig[i] * graph_h)
                x2 = int(margin + ((i + 1) / self.buffer_size) * (w - 2 * margin))
                y2 = int((h - margin) - sig[i + 1] * graph_h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # ------------------------------------------------------------------
    # Validation (API unchanged)
    # ------------------------------------------------------------------

    def start_validation(self):
        if self.validator:
            self.validator.start_recording()
            return True
        return False

    def stop_validation(self):
        if self.validator:
            self.validator.stop_recording()
            self.validator.print_metrics()
            return True
        return False

    def add_validation_reading(self, apple_watch_bpm):
        if self.validator and self.validator.is_recording:
            return self.validator.add_reading(apple_watch_bpm, self.current_bpm)
        return False

    def get_validation_metrics(self):
        if self.validator:
            return self.validator.calculate_metrics()
        return None

    def save_validation_results(self, filepath=None):
        if self.validator:
            return self.validator.save_results(filepath)
        return None

    def cleanup(self):
        if self._session:
            try:
                self._session.__exit__(None, None, None)
            except Exception:
                pass
            self._session = None
