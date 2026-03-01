import argparse
import sys
import os
import json
import logging
import cv2
import librosa
import numpy as np
from engine_npu.bimodal_predictor import BimodalPredictor
from engine_npu.forensic_extractor import ForensicExtractor
from engine_npu.audio_analyzer import AudioAnalyzer
from engine_npu.video_analyzer import VideoAnalyzer

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("VERO-FORENSICS")

def analyze_media(file_path):
    logger.info(f"VERO-CORE TACTICAL FORENSICS MODULE")
    logger.info(f"Target: {file_path}")
    logger.info("-" * 40)
    
    if not os.path.exists(file_path):
        logger.error("File not found.")
        sys.exit(1)
        
    audio_analyzer = AudioAnalyzer()
    video_analyzer = VideoAnalyzer()
    predictor = BimodalPredictor({"audio": 0.65, "video": 0.65, "sync": 0.25})
    forensics = ForensicExtractor()
    
    # 1. Analyze Audio
    logger.info("Extracting Spectral Jitter & PLL Phase Variance...")
    try:
        y, sr = librosa.load(file_path, sr=16000)
        audio_flat = y.flatten().astype(np.float32)
        
        # Process in 1-second chunks (16000 samples)
        chunk_size = 16000
        max_audio_score = 0.0
        audio_res = {}
        for i in range(0, len(audio_flat), chunk_size):
            chunk = audio_flat[i:i+chunk_size]
            if len(chunk) == chunk_size:
                res = audio_analyzer.analyze(chunk)
                if res.get('spectral_jitter_score', 0) > max_audio_score:
                    max_audio_score = res.get('spectral_jitter_score', 0)
                    audio_res = res
                    
        logger.info(f"  [+] Max Audio Jitter Score: {max_audio_score:.4f}")
    except Exception as e:
        logger.warning(f"  [-] Audio Extraction Failed: {e}")
        audio_res = {}
        
    # 2. Analyze Video
    logger.info("Extracting Pixel Jitter & 3D Lighting Anomalies...")
    try:
        cap = cv2.VideoCapture(file_path)
        frames = []
        for _ in range(10):  # Sample 10 frames
            ret, frame = cap.read()
            if not ret: break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()
        
        if len(frames) > 0:
            video_window = np.stack(frames)
            video_res = video_analyzer.analyze(video_window)
            logger.info(f"  [+] Video Jitter Score: {video_res.get('pixel_jitter_score', 0):.4f}")
        else:
            video_res = {}
    except Exception as e:
        logger.warning(f"  [-] Video Extraction Failed: {e}")
        video_res = {}
        
    # 3. Fuse and Predict
    logger.info("Analyzing Multi-Modal Synchronization...")
    prediction = predictor.predict(audio_res, video_res, None, None)
    evidence = forensics.extract_evidence(audio_res, video_res, prediction)
    
    logger.info("=" * 40)
    logger.info("VERDICT COMPUTED")
    logger.info(f"Risk: {'CRITICAL DEEPFAKE THREAT' if prediction['is_threat'] else 'AUTHENTIC MEDIA'}")
    logger.info(f"Trust Score: {prediction['trust_score']*100:.1f}%")
    logger.info(f"Primary Evidence: {evidence.get('primary_flag', 'None')}")
    logger.info("=" * 40)
    
    with open(f"report_{os.path.basename(file_path)}.json", "w") as f:
        json.dump({"prediction": prediction, "evidence": evidence}, f, indent=4)
    logger.info(f"Exported JSON chain-of-custody report.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline Media Forensics Tool")
    parser.add_argument("file", help="Path to the audio/video file to analyze")
    args = parser.parse_args()
    analyze_media(args.file)
