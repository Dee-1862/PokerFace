import cv2
import numpy as np
from collections import deque
from scipy import signal
import time


class RPPGModule:
    """
    POS (Plane-Orthogonal-to-Skin) rPPG with HRV (RMSSD) extraction.

    Replaces single-channel Green method with POS algorithm (Wang et al. 2017).
    POS cancels motion/lighting artifacts by projecting RGB onto the plane
    orthogonal to the current skin-tone vector — computed fresh each window so
    it adapts to different skin tones and varying illumination.

    Also extracts RMSSD (root mean square of successive IBI differences) from
    detected heartbeat peaks.  RMSSD drops within 2–3 seconds of stress onset,
    making it a far faster indicator than raw BPM change (~30 s lag).

    Outputs added to shared_state:
        heart_rate_bpm  — smoothed BPM (int)
        hrv_rmssd       — RMSSD in milliseconds (float or None until stable)
    """

    def __init__(self, buffer_size=300, fps=30, enable_validation=False):
        self.buffer_size  = buffer_size
        self.nominal_fps  = fps
        self.actual_fps   = float(fps)   # updated each frame from wall-clock

        # Store (R, G, B) means per frame instead of Green only
        self.rgb_buffer       = deque(maxlen=buffer_size)
        self.timestamp_buffer = deque(maxlen=buffer_size)

        # BPM output
        self.bpm_buffer  = deque(maxlen=10)
        self.current_bpm = 0

        # HRV output
        self.hrv_rmssd   = None   # ms; None until at least 4 peaks detected
        self._last_peaks = np.array([], dtype=int)

        # Face ROI: forehead + cheek landmarks
        self.ROI_IDXS = [
            10, 338, 297, 332, 284, 251, 389, 356, 454,
            323, 361, 288, 397, 365, 379, 378, 400, 377,
            152, 148, 176, 149, 150, 136,
        ]

        # CLAHE on luminance channel only — stabilises lighting without
        # distorting colour ratios that POS depends on
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        # Validation support (unchanged API)
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

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def initialize(self, shared_state):
        pass

    def _normalize_lighting(self, frame):
        """CLAHE on L channel — preserves colour ratios needed by POS."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = self.clahe.apply(l)
        return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    def process(self, shared_state):
        if not shared_state.get('face_detected'):
            return

        frame     = self._normalize_lighting(shared_state['frame'])
        landmarks = shared_state['landmarks']
        h, w      = shared_state['frame_dimensions']
        now       = time.monotonic()

        # Build ROI mask from face landmarks
        mask   = np.zeros((h, w), dtype=np.uint8)
        points = np.array(
            [[int(landmarks[i].x * w), int(landmarks[i].y * h)]
             for i in self.ROI_IDXS]
        )
        cv2.fillPoly(mask, [points], 255)

        # Extract mean B, G, R (OpenCV is BGR)
        means  = cv2.mean(frame, mask=mask)          # (B, G, R, _)
        r_mean, g_mean, b_mean = means[2], means[1], means[0]

        self.rgb_buffer.append((r_mean, g_mean, b_mean))
        self.timestamp_buffer.append(now)

        # Measure actual FPS from wall-clock timestamps — fixes BPM drift
        # when camera drops below nominal rate
        if len(self.timestamp_buffer) >= 2:
            elapsed = self.timestamp_buffer[-1] - self.timestamp_buffer[0]
            if elapsed > 0:
                self.actual_fps = (len(self.timestamp_buffer) - 1) / elapsed

        self.current_bpm, self.hrv_rmssd = self._calculate_heart_rate()
        shared_state['heart_rate_bpm'] = self.current_bpm
        shared_state['hrv_rmssd']      = self.hrv_rmssd

    # ------------------------------------------------------------------
    # POS algorithm
    # ------------------------------------------------------------------

    def _pos_algorithm(self, rgb_list):
        """
        Wang et al. 2017 — "Algorithmic Principles of Remote PPG"

        1. Normalise each channel by its temporal mean (removes DC skin tone).
        2. Project onto two orthogonal skin-plane axes S1, S2.
        3. Scale S2 so its variance matches S1 (cancels motion on both axes).
        4. Pulse signal h = S1 + alpha * S2.

        Motion artifacts lie along the skin-tone vector; projecting onto
        the orthogonal plane removes them regardless of skin colour.
        """
        C = np.array(rgb_list, dtype=np.float64)   # shape (N, 3): R G B

        # Normalise by channel mean — adapts to current skin tone per window
        mean_C = np.mean(C, axis=0)
        mean_C = np.where(mean_C < 1.0, 1.0, mean_C)   # guard divide-by-zero
        Cn = C / mean_C

        S1 = Cn[:, 0] - Cn[:, 1]                         # R - G
        S2 = Cn[:, 0] + Cn[:, 1] - 2.0 * Cn[:, 2]       # R + G - 2B

        std_S2 = np.std(S2)
        alpha  = np.std(S1) / (std_S2 + 1e-8)

        h = S1 + alpha * S2
        h -= np.mean(h)     # remove DC offset
        return h

    # ------------------------------------------------------------------
    # BPM + HRV
    # ------------------------------------------------------------------

    def _calculate_heart_rate(self):
        # Wait for 5 seconds of data — FFT on <150 frames has 0.2 Hz bin width
        # (~12 BPM resolution) which produces unreliable first estimates that then
        # get locked in by outlier rejection.
        if len(self.rgb_buffer) < 150:
            return 0, None

        fps = max(1.0, self.actual_fps)

        # 1. POS signal from RGB buffer
        pos_signal = self._pos_algorithm(list(self.rgb_buffer))

        # 2. Detrend (remove slow drift)
        sig = signal.detrend(pos_signal)

        # 3. Bandpass 0.7–3.0 Hz (42–180 BPM)
        nyquist = fps / 2.0
        low  = 0.7 / nyquist
        high = min(3.0 / nyquist, 0.98)    # must be < 1
        try:
            b_coef, a_coef = signal.butter(5, [low, high], btype='band')
            sig_filtered   = signal.filtfilt(b_coef, a_coef, sig)
        except ValueError:
            return self.current_bpm, self.hrv_rmssd

        # 4. FFT — search 50–180 BPM (raised upper limit from 150 for stressed players)
        fft   = np.fft.rfft(sig_filtered)
        freqs = np.fft.rfftfreq(len(sig_filtered), 1.0 / fps)
        bpm_mask = (freqs >= 50 / 60) & (freqs <= 180 / 60)

        if not np.any(bpm_mask):
            return 0, None

        power    = np.abs(fft[bpm_mask])
        peak_f   = freqs[bpm_mask][np.argmax(power)]
        raw_bpm  = float(np.clip(peak_f * 60, 50, 180))

        # 5. Outlier rejection — only active once bpm_buffer is full (10 readings).
        #    Before that, accept all readings so the buffer establishes the true
        #    baseline rather than locking onto a bad early estimate.
        if len(self.bpm_buffer) >= self.bpm_buffer.maxlen:
            recent_med = float(np.median(list(self.bpm_buffer)))
            recent_std = float(np.std(list(self.bpm_buffer)))
            # If the buffer has frozen (std < 2 BPM for many readings) and the
            # new reading differs by > 35 BPM, the buffer is stuck — clear it
            # so it can re-settle at the correct value.
            if recent_std < 2.0 and abs(raw_bpm - recent_med) > 35:
                self.bpm_buffer.clear()
            elif abs(raw_bpm - recent_med) > 25:
                raw_bpm = recent_med   # reject large jump, keep last known

        self.bpm_buffer.append(raw_bpm)
        smoothed_bpm = (
            int(np.median(list(self.bpm_buffer)))
            if len(self.bpm_buffer) >= 3 else int(raw_bpm)
        )

        # 6. HRV from the same filtered signal
        hrv = self._compute_hrv(sig_filtered, fps)

        return smoothed_bpm, hrv

    def _compute_hrv(self, filtered_signal, fps):
        """
        Detect heartbeat peaks → inter-beat intervals → RMSSD (ms).

        RMSSD reflects parasympathetic (vagal) activity.  When the sympathetic
        nervous system activates (stress, bluffing anxiety) it suppresses vagal
        tone and RMSSD drops within 2–3 seconds — far faster than raw BPM rise.

        Returns None until at least 4 peaks are found; returns last known value
        if too few peaks are detected in the current window.
        """
        # Need ≥ 4 seconds of signal for reliable peak detection
        if len(filtered_signal) < int(fps * 4):
            return None

        # Minimum inter-peak distance at 180 BPM max
        min_dist = max(int(fps * 60 / 180), 5)

        peaks, _ = signal.find_peaks(
            filtered_signal,
            distance=min_dist,
            prominence=np.std(filtered_signal) * 0.3,   # adaptive threshold
        )

        if len(peaks) < 4:          # need ≥ 3 IBI values for RMSSD
            return self.hrv_rmssd   # keep last valid reading

        self._last_peaks = peaks

        ibi = np.diff(peaks) / fps   # inter-beat intervals in seconds

        # Reject physiologically impossible values
        ibi = ibi[(ibi >= 0.33) & (ibi <= 2.0)]

        if len(ibi) < 3:
            return self.hrv_rmssd

        # RMSSD in milliseconds
        diff_ibi = np.diff(ibi) * 1000
        rmssd    = float(np.sqrt(np.mean(diff_ibi ** 2)))
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
            color  = (0, 255, 0)   if std_dev < 5 else (0, 255, 255)
            status = "Stable"      if std_dev < 5 else "Stabilizing"
        else:
            color, status = (0, 255, 255), "Initializing"

        cv2.putText(frame, f"HEART RATE: {bpm} BPM",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, status,
                    (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if hrv is not None:
            # Green > 30 ms (relaxed), yellow 15–30, orange < 15 (stressed)
            hrv_color = (
                (0, 255, 0)   if hrv > 30 else
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
        if len(self.rgb_buffer) < 2:
            return
        h, w = frame.shape[:2]
        graph_h, margin = 60, 20

        cv2.rectangle(frame,
                      (margin, h - graph_h - margin),
                      (w - margin, h - margin),
                      (50, 50, 50), -1)

        # Display G channel for visual continuity (POS signal is internal)
        sig = np.array([v[1] for v in self.rgb_buffer])
        rng = np.max(sig) - np.min(sig)
        if rng == 0:
            return
        sig = (sig - np.min(sig)) / rng

        for i in range(len(sig) - 1):
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
        pass
