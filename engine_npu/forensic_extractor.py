import logging

logger = logging.getLogger("ForensicExtractor")

class ForensicExtractor:
    """
    Converts raw analyzer scores into human-readable explainable evidence strings.
    """
    def __init__(self, thresholds: dict = None):
        self.thresholds = thresholds or {
            "sync": 40.0,
            "audio_jitter": 0.3,
            "video_jitter": 0.5
        }

    def extract_evidence(self, audio_results: dict, video_results: dict, predictor_results: dict) -> list:
        evidence = []
        
        # 1. GAN Vocoder Artifacts (Audio)
        gan_score = audio_results.get("gan_vocoder_presence", 0.0)
        is_speaking = video_results.get("is_speaking", True)
        
        if gan_score > 0.4:
            if not is_speaking:
                msg = "[OFF-SCREEN THREAT] The tracked face is NOT speaking, but deepfake audio is detected on the global call."
                msg += "\n▶ ACTION: Click 'TARGET FACE' and draw a box over the other participants to isolate the threat actor."
                evidence.append({
                    "source": "Global Audio",
                    "message": msg,
                    "severity": "CRITICAL"
                })
            else:
                msg = "AI-Synthesized Voice: High-frequency spectral jitter confirms GAN-vocoder signatures."
                if gan_score > 0.7:
                     msg += "\n▶ ACTION: Ask the caller an unexpected personal question or request they speak with varied emotional intonation. Neural modulators struggle to adapt instantly."
                evidence.append({
                    "source": "Targeted Audio",
                    "message": msg,
                    "severity": "CRITICAL" if gan_score > 0.7 else "WARNING"
                })

        # 2. Perioral Edge Jitter (Video)
        vj = video_results.get("perioral_jitter", 0.0)
        lighting = video_results.get("lighting_gradient_inconsistency", 0.0)
        if vj > 0.5 or lighting > 0.6:
            msg = "Frame-Injection Artifacts: Non-biometric edge-jitter and lighting gradients detected in perioral region."
            if vj > 0.8:
                 msg += "\n▶ ACTION: Request the caller to turn their head 90 degrees left or right; deepfake face-swaps typically fail at extreme side profiles, breaking the illusion."
            evidence.append({
                "source": "Video",
                "message": msg,
                "severity": "CRITICAL" if vj > 0.8 else "WARNING"
            })

        # 3. Mouth Sync (Cross-Modal)
        ms = video_results.get("mouth_sync_latency_ms", 0.0)
        # The `is_speaking` variable is already defined above.
        if ms > self.thresholds["sync"]:
            if not is_speaking:
                # If the tracked face is not speaking, but there's a sync issue, it implies an off-screen audio source.
                msg = "[OFF-SCREEN THREAT] The tracked face is NOT speaking, but a temporal mismatch (Frame-Delay attack) is detected on the global call."
                msg += "\n▶ ACTION: Click 'TARGET FACE' and draw a box over the other participants to isolate the threat actor."
                evidence.append({
                    "source": "Global Audio",
                    "message": msg,
                    "severity": "CRITICAL"
                })
            else:
                evidence.append({
                    "source": "Bimodal",
                    "message": f"Temporal Mismatch: {ms:.1f}ms latency exceeds biometric sync threshold (Frame-Delay attack).",
                    "severity": "WARNING"
                })

        # 4. PLL Sync Deviation
        pll = audio_results.get("pll_sync_deviation", 0.0)
        if pll > 0.5:
             if not is_speaking:
                 msg = "[OFF-SCREEN THREAT] The tracked face is NOT speaking, but electronic phase jitter is detected on the global call."
                 msg += "\n▶ ACTION: Click 'TARGET FACE' and draw a box over the other participants to isolate the threat actor."
                 evidence.append({
                     "source": "Global Audio",
                     "message": msg,
                     "severity": "WARNING"
                 })
             else:
                 evidence.append({
                    "source": "Targeted Audio",
                    "message": "Electronic Phase Jitter: Non-deterministic phase-locks detected (Signature of neural synthesis).",
                    "severity": "WARNING"
                })

        # 5. Summary
        if not evidence and predictor_results.get("trust_score", 1.0) > 0.8:
            evidence.append({
                "source": "System",
                "message": "Biometric Integrity Verified — No synthetic signatures detected in local hardware trace.",
                "severity": "INFO"
            })

        return evidence
