import cv2
import numpy as np
import logging
import os

# Suppress MediaPipe C++ underlying logging (e.g. NORM_RECT warnings)
os.environ["GLOG_minloglevel"] = "2"

try:
    import mediapipe as mp
except ImportError:
    mp = None

logger = logging.getLogger("VideoAnalyzer")

class VideoAnalyzer:
    """
    targeted perioral LoG analysis for compression artifacts.
    """
    def __init__(self):
        self.mp_face_mesh = None
        self.face_mesh = None
        self.aperture_history = []
        if mp:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )

    def analyze(self, frame_window: np.ndarray, audio_rms_envelope: np.ndarray = None) -> dict:
        if frame_window is None or len(frame_window) < 2:
            return {}

        # 1. Global Motion Jitter (Detects rapid finger movement/occlusion)
        global_jitter = 0.0
        is_small_roi = False
        if len(frame_window) >= 2:
            frame = frame_window[-1]
            diff = cv2.absdiff(frame, frame_window[-2])
            if diff.size > 0:
                h, w, _ = frame.shape
                is_small_roi = (w < 600 or h < 600)
                
                # If the user drew a tight ROI square, the camera logic changes.
                # A tight square has very little "background" to absorb movement, so any
                # natural head movement causes massive frame diffs. We must desensitize it.
                # Lower sensitivity of global jitter to prevent MSS screen artifacts
                divisor = 400.0 if is_small_roi else 250.0
                global_jitter = float(np.nan_to_num(np.mean(diff))) / divisor
            
        # 2. Targeted Perioral (Mouth) LoG Analysis
        perioral_jitter = 0.0
        mouth_aperture = 0.0
        face_confidence = 0.0
        
        if self.face_mesh:
             latest_frame = frame_window[-1]
             results = self.face_mesh.process(cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB))
             if results.multi_face_landmarks:
                 face_confidence = 1.0
                 lms = results.multi_face_landmarks[0].landmark
                 h, w, _ = latest_frame.shape
                 
                 # Mouth aperture (biometric)
                 mouth_aperture = float(np.nan_to_num(abs(lms[13].y - lms[14].y)))
                 
                 # Perioral ROI for 8x8 block-artifact detection
                 mouth_pts = np.array([[lms[i].x * w, lms[i].y * h] for i in [61, 291, 0, 17]], dtype=np.int32)
                 x, y, mw, mh = cv2.boundingRect(mouth_pts)
                 # Safety clip
                 y1, y2 = max(0, y), min(h, y+mh)
                 x1, x2 = max(0, x), min(w, x+mw)
                 
                 if y2 > y1 and x2 > x1:
                     roi = cv2.cvtColor(latest_frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                     
                     if roi.size >= 64:
                         h_r, w_r = roi.shape
                         h_r = (h_r // 8) * 8
                         w_r = (w_r // 8) * 8
                         
                         if h_r >= 8 and w_r >= 8:
                             grid_roi = roi[:h_r, :w_r]
                             blocks = grid_roi.reshape(h_r // 8, 8, -1, 8).swapaxes(1, 2).reshape(-1, 8, 8)
                             block_vars = np.var(blocks, axis=(1, 2))
                             perioral_jitter = float(np.nan_to_num(np.std(block_vars) / 100.0)) if block_vars.size > 0 else 0.0
             else:
                 face_confidence = 0.0
                 perioral_jitter = 0.0 # Remove artificial penalty for no face to let score breathe
                 mouth_aperture = 0.0

        # Aperture History for cross-modal sync
        self.aperture_history.append(mouth_aperture)
        if len(self.aperture_history) > len(frame_window): self.aperture_history.pop(0)
        
        # 3. Mouth-Sync Correlation
        mouth_sync_latency = 0.0
        is_speaking = True
        if audio_rms_envelope is not None and len(audio_rms_envelope) > 0:
             sig_a = np.array(self.aperture_history)
             sig_b = audio_rms_envelope[-len(sig_a):]
             if len(sig_b) == len(sig_a) and len(sig_a) > 2:
                 aperture_std = float(np.std(sig_a))
                 # If the mouth is not moving (variance is mathematically near zero)
                 if aperture_std < 0.002:
                     is_speaking = False
                     mouth_sync_latency = 0.0
                 else:
                     corr = np.correlate(sig_a - np.mean(sig_a), sig_b - np.mean(sig_b), mode='full')
                     delay = np.argmax(corr) - (len(sig_a) - 1)
                     mouth_sync_latency = float(np.nan_to_num(abs(delay) * (1000/15)))

        return {
            "pixel_jitter_score": float(np.clip(np.nan_to_num(perioral_jitter + global_jitter), 0, 1)),
            "mouth_sync_latency_ms": mouth_sync_latency,
            "mouth_aperture": mouth_aperture,
            "face_confidence": face_confidence,
            "is_speaking": is_speaking,
            "lighting_gradient_inconsistency": float(np.clip(np.nan_to_num(global_jitter * (0.0 if is_small_roi else 0.5)), 0, 1))
        }
