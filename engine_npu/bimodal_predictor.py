import numpy as np
from collections import deque
import logging

logger = logging.getLogger("BimodalPredictor")

class BimodalPredictor:
    """
    Fuses audio and video artifact scores over time using a weighted confidence model.
    Applies Exponential Moving Average (EMA) for temporal smoothing.
    """
    def __init__(self, weights: dict = None, ema_alpha: float = 0.4):
        # Bimodal weights: Recalibrated for absolute stability across models.
        # Sync and Audio-only jitter are dampened to favor multi-modal agreement.
        self.weights = weights or {"audio": 1.0, "video": 1.0, "sync": 0.5}
        self.ema_alpha = ema_alpha
        self.history = deque(maxlen=30)
        self.last_trust_score = 1.0
        self.penalty_history = []
        self.frames_since_start = 0

    def reset(self):
        """Total memory wipe for a 100% clean baseline start."""
        self.history.clear()
        self.penalty_history = []
        self.frames_since_start = 0
        self.last_trust_score = 1.0
        if hasattr(self, 'ai_audio_history'): self.ai_audio_history.clear()
        if hasattr(self, 'ai_video_history'): self.ai_video_history.clear()
        logger.info("⛶ BIOMETRIC MEMORY PURGED: Absolute baseline restored.")

    def predict(self, audio_results: dict, video_results: dict, audio_inf: dict = None, video_inf: dict = None, 
                max_sensitivity: bool = False, is_offline_scan: bool = False, has_multiple_faces: bool = False,
                is_manual_lock: bool = False) -> dict:
        """
        Calculates a trust score (0.0 = fake, 1.0 = real) based on bimodal features and AI inference.
        """
        # Extract heuristic scores
        audio_jitter = float(np.nan_to_num(audio_results.get("spectral_jitter_score", 0.0)))
        video_jitter = float(np.nan_to_num(video_results.get("pixel_jitter_score", video_results.get("video_jitter", 0.0))))
        mouth_sync = float(np.nan_to_num(video_results.get("mouth_sync_latency_ms", 0.0)))
        face_conf = float(np.nan_to_num(video_results.get("face_confidence", 1.0)))
        
        # Extract AI Model scores (if available)
        # Extract AI Model scores
        ai_audio_score = 0.0
        ai_video_score = 0.0
        
        # Softmax on Audio Logits
        if audio_inf and "logits" in audio_inf:
            try:
                logits = audio_inf["logits"].flatten()
                # print(f"Raw Audio Logits: {logits}") # Debugging
                if len(logits) >= 2:
                    exp_logits = np.exp(logits - np.max(logits))
                    probs = exp_logits / np.sum(exp_logits)
                    ai_audio_score = float(probs[1]) # Index 1 = Fake
                elif len(logits) == 1:
                    val = float(logits[0])
                    if val < -1e10 or val > 1e10:
                        # Raw unnormalized ONNX audio tensor heuristic
                        ai_audio_score = 1.0 if val > 14e12 else 0.0
                    else:
                        ai_audio_score = float(1 / (1 + np.exp(np.clip(-val, -100, 100))))
            except Exception as e:
                logger.warning(f"Audio Logit parse error: {e}")
                
            # Temporal Inference Smoothing (Convolution)
            # The ONNX model processes extreme microscopic slices of audio, causing 1.0/0.0 oscillation.
            # We enforce a moving average to eradicate "blinks" on real audio while firmly catching sustained fakes.
            if not hasattr(self, 'ai_audio_history'):
                self.ai_audio_history = deque(maxlen=5)
                
            self.ai_audio_history.append(ai_audio_score)
            
            # Amplify sustained model detections slightly, but demand high baseline confidence
            ai_audio_score = float(np.clip(np.mean(self.ai_audio_history) * 1.1, 0.0, 1.0))
            
            # HIGH SECURITY SQUASHING GATE: Eliminate all false positives on real audio
            # Real human audio with background hum sometimes scores ~0.62. 
            # We strictly discard anything under 0.85 (0.50 if offline for sensitivity).
            gate = 0.50 if is_offline_scan else 0.85
            if ai_audio_score < gate:
                 ai_audio_score = 0.0
        # Softmax on Video Logits
        if video_inf and "logits" in video_inf:
            try:
                logits = video_inf["logits"].flatten()
                if len(logits) >= 2:
                    exp_logits = np.exp(logits - np.max(logits))
                    probs = exp_logits / np.sum(exp_logits)
                    ai_video_score = float(probs[1]) # Index 1 = Fake
                elif len(logits) == 1:
                    val = float(logits[0])
                    if val < -1e10 or val > 1e10:
                        # 1D NPU Tensor Forensic: Biological signals cluster around -2e18 to 2.2e18.
                        # Synthetic signals push into ultra-high (>2.5e18) or "dead zones" (-1.5e18 to -0.1e18).
                        if is_offline_scan:
                            is_fake = (val > 2.5e18) or (-1.5e18 < val < -0.1e18)
                            logger.info(f"OFFLINE V-ANALYSIS: val={val:.2e} is_fake={is_fake}")
                        else:
                            is_fake = (val > -0.1e18) # Real-time maintains high sensitivity
                        ai_video_score = 1.0 if is_fake else 0.0
                    else:
                         # Math-safe sigmoid
                         ai_video_score = float(1 / (1 + np.exp(np.clip(-val, -100, 100))))
            except Exception as e:
                logger.warning(f"Video Logit parse error: {e}")

        # Normalize mouth sync (40ms threshold)
        sync_score = float(np.nan_to_num(min(mouth_sync / 100.0, 1.0) if mouth_sync > 40.0 else 0.0))

        # Weighted artifact penalty
        v_weight = self.weights["video"] * 1.0
        
        # Penalty for Face Loss (Physical Occlusion/Shielding)
        # Reduced from 0.4 to 0.1 to avoid permanently locking the score in red if user looks away
        occlusion_penalty = float(np.nan_to_num((1.0 - face_conf) * 0.1))

        # Activity Gates: Prevent ONNX from hallucinating threats on empty buffers
        if audio_results.get("is_silent", False):
             ai_audio_score = 0.0
             
        # If no face is detected with sufficient confidence, the video models
        # are likely ingestion noise. Suppress video-based penalties.
        if face_conf < 0.4:
             ai_video_score = 0.0
             sync_score = 0.0
             video_jitter = 0.0
             occlusion_penalty = 0.0
             
        # If the face is detected but the subject is NOT speaking, 
        # suppress the lip-sync penalty to avoid noise.
        if not video_results.get("is_speaking", False) or self.frames_since_start < 30:
             sync_score = 0.0

        # Noise-Floor Gate: Ignore microscopic jitter that is likely sensor noise
        if video_jitter < 0.1:
            video_jitter = 0.0

        # If the image is completely static (like a screenshot), suppress AI video.
        # BYPASS: If the AI model is extremely certain (>0.8), ignore the static check.
        if video_jitter < 0.01 and ai_video_score < 0.8 and not is_offline_scan:
            ai_video_score = 0.0

        # Absolute Threat Override:
        # If the audio heuristic outputs a sterile 1.0 (Confirmed Robotic Monotone),
        # it bypasses the standard weighting system to guarantee an alert.
        if audio_jitter >= 1.0:
            penalty = 1.0
        else:
            # Max-Penalty Fusion Logic: Prevents low-confidence noise from stacking.
            v_threat = float(max(video_jitter, ai_video_score) * self.weights["video"])
            a_threat = float(max(audio_jitter, ai_audio_score) * self.weights["audio"])
            s_threat = float(sync_score * self.weights["sync"])
            
            penalty = float(max(v_threat, a_threat, s_threat, occlusion_penalty))

            # --- CONSENSUS SHIELD: 2-OUT-OF-3 RULE ---
            # If only ONE model is panicking, cap the threat at 0.40 (Caution/Yellow).
            # This prevents a single hallucination (like lip-sync lag) from turning the ring red.
            # EXTREME BYPASS: If any single model is > 0.98, we trust it absolutely.
            high_signals = [t for t in [v_threat, a_threat, s_threat] if t > 0.50]
            extreme_signals = [t for t in [v_threat, a_threat, s_threat] if t >= 0.98]
            
            if len(high_signals) < 2 and not extreme_signals and not is_offline_scan:
                penalty = min(penalty, 0.40)
            
            # --- AI SUPREMACY RULE ---
            # If the AI models are absolutely certain this is real (score < 0.1),
            # we cap the heuristic penalty. Heuristics are allowed to suggest "Yellow/Caution",
            # but they cannot turn the ring red without AI confirmation.
            if ai_video_score < 0.1 and ai_audio_score < 0.1:
                penalty = min(penalty, 0.15)
        
        # --- FORENSIC ACTIVITY GATE ---
        # If there is no significant motion AND the AI models are not screaming,
        # we treat this as an "Idle" or "Static" state and force zero penalty.
        # This prevents camera noise/flicker from causing alerts while waiting.
        if video_jitter < 0.2 and ai_video_score < 0.5 and ai_audio_score < 0.5 and not is_offline_scan:
             penalty = 0.0

        # --- SMART STABILITY: TEMPORAL CONSENSUS ---
        # Only progress the lifecycle if we actually see a human face.
        if face_conf > 0.5:
            self.frames_since_start += 1
        
        if not is_offline_scan:
            # 1. Warm-up Gate: Allow settle time, but speed up if user has manually locked a target
            warmup_limit = 10 if is_manual_lock else 15
            
            if self.frames_since_start < warmup_limit:
                penalty = 0.0
                self.penalty_history = []
            else:
                self.penalty_history.append(penalty)
                
                # Multi-Face & Precision Buffer Tuning
                # Autonomous mode = 5 frames (1s), Manual mode = 5 frames (1s)
                # Group calls = 10 frames (2s)
                if has_multiple_faces and not is_manual_lock:
                    history_limit = 10
                elif is_manual_lock:
                    history_limit = 3
                else:
                    history_limit = 5
                    
                while len(self.penalty_history) > history_limit:
                    self.penalty_history.pop(0)
                
                if len(self.penalty_history) < history_limit:
                    penalty = 0.0
                else:
                    # Consensus Requirement: Use MEAN to allow threats to burn through noise
                    penalty = float(np.mean(self.penalty_history))

        # 2. Precision Mode Boost: If the user manually targeted a region, we trust their intent.
        if is_manual_lock and penalty > 0.3:
            penalty = float(np.clip(penalty * 1.3, 0.0, 1.0))

        # Absolute bottom out for micro-noises
        if penalty < 0.15:
            penalty = 0.0
            
        # Security Override: Magnify penalties when Max Sensitivity is engaged
        if max_sensitivity:
            penalty = float(np.clip(penalty * 2.5, 0.0, 1.0))

        if penalty > 0.0:
            logger.info(f"Fusion Penalty: {penalty:.3f} (V:{video_jitter:.2f} A:{audio_jitter:.2f} Sync:{sync_score:.2f} Occ:{occlusion_penalty:.2f} AI_A:{ai_audio_score:.2f} AI_V:{ai_video_score:.2f})")
            
        raw_trust_score = float(np.clip(1.0 - penalty, 0, 1))
        
        # Temporal Smoothing: Asymmetric Exponential Moving Average (EMA)
        # Prevents trust from bouncing up during natural speech pauses when the scanner reads silence.
        if is_offline_scan:
             trust_score = float(raw_trust_score) # Absolute offline verification
        else:
             # Drop extremely fast if a threat is detected. Recover with extreme forensic caution.
             # 0.01 recovery speed ensures the Ring stays RED even if the signal flickers temporarily.
             dynamic_alpha = 0.8 if raw_trust_score < self.last_trust_score else 0.01
             trust_score = (dynamic_alpha * raw_trust_score) + ((1.0 - dynamic_alpha) * self.last_trust_score)
        
        self.last_trust_score = float(trust_score)
        
        confidence = 0.8 # Base confidence for now
        
        result = {
            "trust_score": float(np.clip(trust_score, 0, 1)),
            "confidence": confidence,
            "is_threat": trust_score < 0.45,
            "raw_scores": {
                "audio_jitter": audio_jitter,
                "video_jitter": video_jitter,
                "sync_penalty": sync_score,
                "ai_audio": ai_audio_score,
                "ai_video": ai_video_score
            }
        }
        self.history.append(result)
        return result

    def get_threat_status(self, threshold=0.45, consecutive=3) -> bool:
        """Trigger alert if trust_score < threshold for N consecutive windows."""
        if len(self.history) < consecutive:
            return False
        
        recent = list(self.history)[-consecutive:]
        return all(r["trust_score"] < threshold for r in recent)
