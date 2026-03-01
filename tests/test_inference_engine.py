import pytest
import numpy as np
from engine_npu.bimodal_predictor import BimodalPredictor

def test_bimodal_predictor_fusion():
    predictor = BimodalPredictor(ema_alpha=1.0) # Disable EMA for testing
    
    # Case 1: Pure real
    audio_res = {"spectral_jitter_score": 0.0}
    video_res = {"pixel_jitter_score": 0.0, "mouth_sync_latency_ms": 10.0}
    res = predictor.predict(audio_res, video_res, is_offline_scan=True)
    assert res["trust_score"] == 1.0
    assert not res["is_threat"]

    # Case 2: Suspicious audio
    audio_res = {"spectral_jitter_score": 0.5}
    res = predictor.predict(audio_res, video_res, is_offline_scan=True)
    assert res["trust_score"] <= 1.0
    assert not res["is_threat"]
    
    # Case 3: Deepfake (multiple artifacts)
    audio_res = {"spectral_jitter_score": 1.0}
    video_res = {"pixel_jitter_score": 1.0, "mouth_sync_latency_ms": 80.0}
    res = predictor.predict(audio_res, video_res, is_offline_scan=True)
    
    assert res["trust_score"] < 0.45
    assert res["is_threat"]
