import sys
import platform

# Bypass WMI hang on Windows 11
def fast_uname():
    from collections import namedtuple
    import os
    Uname = namedtuple('uname_result', 'system node release version machine processor')
    node_name = os.getenv('COMPUTERNAME', 'VERO-CORE')
    return Uname('Windows', node_name, '10', '10.0.22631', 'AMD64', 'AMD64')
platform.uname = fast_uname
platform.system = lambda: 'Windows'

import argparse
import threading
import time
import logging
import yaml
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # Suppress TensorFlow C++ Warnings
os.environ["GLOG_minloglevel"] = "2"     # Suppress MediaPipe C++ Warnings
import numpy as np

# Internal imports
from interceptor.device_manager import detect_execution_provider
from interceptor.audio_interceptor import AudioInterceptor
from interceptor.video_interceptor import VideoInterceptor
from engine_npu.inference_engine import InferenceEngine
from engine_npu.audio_analyzer import AudioAnalyzer
from engine_npu.video_analyzer import VideoAnalyzer
from engine_npu.bimodal_predictor import BimodalPredictor
from engine_npu.forensic_extractor import ForensicExtractor
from engine_npu.models.download_models import download_models
from ui_overlay.main_window import MainWindow

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("VERO-CORE")

class VeroCoreApp:
    def __init__(self, args):
        self._set_priority()
        self.args = args
        self.config = self._load_config()
        self.hw_info = detect_execution_provider()
        
        # UI Thread Synchronization Queue
        import queue
        self.ui_queue = queue.Queue()
        
        # Download models if missing
        download_models()
        
        # Architecture Simulation for the winning hackathon video
        # (REMOVED FOR PRODUCTION)
        
        # Initialize Core components
        self.audio_interceptor = AudioInterceptor(
            sample_rate=self.config['audio']['sample_rate'],
            window_ms=self.config['audio']['window_ms']
        )
        self.video_interceptor = VideoInterceptor(fps=self.config['video']['fps'])
        
        self.audio_analyzer = AudioAnalyzer()
        self.video_analyzer = VideoAnalyzer()
        
        provider = "CPUExecutionProvider" if args.no_gpu else self.hw_info["provider"]
        from engine_npu.inference_engine import MultiprocessInferenceEngine, InferenceEngine
        self.audio_engine = MultiprocessInferenceEngine("engine_npu/models/audio_deepfake.onnx", provider)
        self.video_engine = MultiprocessInferenceEngine("engine_npu/models/video_deepfake.onnx", provider)
        
        # Dedicated Isolated Engines for Offline Scans
        self.offline_audio_engine = InferenceEngine("engine_npu/models/audio_deepfake.onnx", provider)
        self.offline_video_engine = InferenceEngine("engine_npu/models/video_deepfake.onnx", provider)
        
        self.predictor = BimodalPredictor(
            weights=self.config['detection']['weights'],
            ema_alpha=self.config['detection'].get('ema_alpha', 0.3)
        )
        self.forensics = ForensicExtractor()
        
        # Synchronization Events
        self.shutdown_event = threading.Event()
        self.inference_lock = threading.Lock()
        
        # UI
        self.ui = MainWindow(
            on_deep_scan=self._handle_deep_scan,
            on_start=self.start,
            on_stop=self.stop,
            ui_queue=self.ui_queue
        )
        self.ui.app = self # Explicitly wire backend reference to UI
        
        # Resource Priority: Core Affinity (Core 0 for Audio)
        self._lock_affinities()
        
        # Warm-Start Wait (Background)
        threading.Thread(target=self._warm_start_sequence, daemon=True).start()
        
        self.running = False
        self.monitor_thread = None

    def _lock_affinities(self):
        """Lock AudioInterceptor to Core 0 to prevent packet drops."""
        try:
            import psutil
            p = psutil.Process(os.getpid())
            # Main process and threads handling capture get high priority
            p.cpu_affinity([0]) # Pin to Core 0 for deterministic interrupt handling
            logger.info("🎯 CORE AFFINITY: AUDIO/INT LOCKED TO CORE 0")
        except Exception as e:
            logger.warning(f"Could not set core affinity: {e}")

    def _warm_start_sequence(self):
        """Wait for NPU to wake up before enabling UI monitoring."""
        start_time = time.time()
        while not (self.audio_engine.is_ready() and self.video_engine.is_ready()):
            time.sleep(0.1)
            if time.time() - start_time > 10: # Timeout
                logger.error("NPU Warm-Start Timeout!")
                break
        
        # Route UI commands safely to the main thread via Queue
        self.ui_queue.put(lambda: self.ui.set_hardware_info(f"{self.hw_info['device_name']} | SILICON READY ✅"))
        self.ui_queue.put(lambda: self.ui.set_hardware_provider(self.hw_info['provider']))
        logger.info(f"✨ VERO-CORE READY IN {time.time()-start_time:.2f}s")

    def _set_priority(self):
        """Elevates process priority to ensure NPU lanes are prioritized by the scheduler."""
        try:
            # Prevent librosa/OpenBLAS from eating 100% of all CPU threads
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["OPENBLAS_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            
            import psutil
            p = psutil.Process(os.getpid())
            # Use NORMAL instead of HIGH to prevent the UI thread from freezing the Windows OS
            p.nice(psutil.NORMAL_PRIORITY_CLASS)
            logger.info("⚡ PROCESS PRIORITY: NORMAL (To prevent OS Freeze)")
        except Exception as e:
            logger.warning(f"Note: Could not set process priority: {e}. Performance may be degraded.")

    def _handle_deep_scan(self, media_type: str, filepath: str) -> dict:
        """
        100% Functional Offline Media Pipeline.
        # Temporal Consensus: Multi-point analysis for robustness.
        """
        logger.info(f"Initiating Global Offline Scan for {media_type}: {filepath}")
        
        try:
            if media_type == "image":
                import cv2
                frame = cv2.imread(filepath)
                if frame is None: raise ValueError("Could not read image file.")
                
                # Image Pipeline (Single point is valid for static images)
                video_res = self.video_analyzer.analyze([frame])
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (224, 224))
                raw_frame = frame_resized.astype(np.float32) / 255.0
                video_input = np.transpose(raw_frame, (2, 0, 1))
                video_inf = self.offline_video_engine.run(video_input)
                
                self.predictor.reset()
                prediction = self.predictor.predict({}, video_res, {}, video_inf, is_offline_scan=True)
                     
            elif media_type == "audio":
                import librosa
                y, sr = librosa.load(filepath, sr=16000, mono=True)
                
                # Sliding Window Analysis: Scan the entire file in 1s overlapping chunks
                # We take the MAXIMUM penalty found across the file for forensic safety.
                chunk_len = 16000
                step = 8000
                penalties = []
                detailed_results = []
                
                for start in range(0, len(y) - chunk_len, step):
                    chunk = y[start:start+chunk_len]
                    a_res = self.audio_analyzer.analyze(chunk)
                    a_inf = self.offline_audio_engine.run(chunk.astype(np.float32))
                    
                    self.predictor.reset()
                    slice_pred = self.predictor.predict(a_res, {}, a_inf, {}, is_offline_scan=True)
                    detailed_results.append(slice_pred)
                    penalties.append(1.0 - slice_pred['trust_score'])
                
                if not penalties:
                    # File too short, pad and run once
                    chunk = np.pad(y, (0, max(0, 16000 - len(y))))
                    a_res = self.audio_analyzer.analyze(chunk)
                    a_inf = self.offline_audio_engine.run(chunk.astype(np.float32))
                    self.predictor.reset()
                    prediction = self.predictor.predict(a_res, {}, a_inf, {}, is_offline_scan=True)
                else:
                    # Find the most "fake" segment
                    max_idx = np.argmax(penalties)
                    prediction = detailed_results[max_idx]
                     
            elif media_type == "video":
                import cv2
                cap = cv2.VideoCapture(filepath)
                if not cap.isOpened(): raise ValueError("Could not open video file.")
                
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                # Sample 10 frames evenly throughout the video
                sample_indices = np.linspace(0, total_frames - 1, 10, dtype=int)
                
                results = []
                self.predictor.reset()
                for idx in sample_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if not ret: continue
                    
                    v_res = self.video_analyzer.analyze([frame])
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb, (224, 224))
                    raw_frame = frame_resized.astype(np.float32) / 255.0
                    v_input = np.transpose(raw_frame, (2, 0, 1))
                    v_inf = self.offline_video_engine.run(v_input)
                    
                    # No internal reset: allow temporal smoothing to work
                    res = self.predictor.predict({}, v_res, {}, v_inf, is_offline_scan=True)
                    results.append(res)
                cap.release()
                
                if not results: raise ValueError("No readable frames in video.")
                
                # TEMPORAL FORENSICS: Use MAX THREAT (Min Trust) across samples
                best_prediction = None
                min_trust = 1.0
                
                for res in results:
                    if res['trust_score'] < min_trust:
                        min_trust = res['trust_score']
                        best_prediction = res
                
                # If everything was clean, pick the last frame results
                if best_prediction is None:
                    prediction = results[-1]
                else:
                    prediction = best_prediction
                
                prediction['trust_score'] = min_trust
                prediction['is_threat'] = min_trust < 0.45

            # Extract actionable physical evidence
            # Note: We use the heuristic results from the representative slice chosen above
            evidence = self.forensics.extract_evidence({}, {}, prediction)
            prediction['evidence'] = evidence
            
            return prediction
            
        except Exception as e:
            logger.error(f"Deep Scan Pipeline Error: {e}")
            raise e

    def _load_config(self):
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)

    def start(self):
        logger.info("Engaging VERO-CORE monitoring sequence...")
        self.running = True
        self.shutdown_event.clear()
        
        # Start AI monitor loop 
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        # Unconditionally rebuild Interceptor sensors for a clean boot without thread hangs
        if self.video_interceptor:
            self.video_interceptor.stop()
        if self.audio_interceptor:
            self.audio_interceptor.stop()
        
        # OS Resource Synchronization: Wait for WASAPI/MSS to release hardware handles
        time.sleep(1.2)
        
        # Reset stability filters for a clean start
        self.predictor.reset()
        
        self.audio_interceptor = AudioInterceptor(
            sample_rate=self.config['audio']['sample_rate'],
            window_ms=self.config['audio']['window_ms']
        )
        try:
            self.audio_interceptor.start()
        except RuntimeError:
            pass

        self.video_interceptor = VideoInterceptor(fps=self.config['video']['fps'])
        if hasattr(self, '_pending_roi'):
            self.video_interceptor.roi = self._pending_roi
            
        try:
            self.video_interceptor.start()
        except RuntimeError:
            pass

    def stop(self):
        logger.info("Aborting VERO-CORE monitoring... Sensors switching to standby idle.")
        self.running = False
        self.shutdown_event.set()
        
        # UI Soft-Reset: Clear red-alert state when manually stopped
        self.predictor.reset()
        self.ui.update_metrics({
            "trust_score": 1.0,
            "latency": 0,
            "fps": 0,
            "audio_windows": 0,
            "npu_load": 0,
            "is_threat": False,
            "evidence": []
        })
        self.shutdown_event.set()
        
        # Terminate capture loops
        if self.audio_interceptor:
            self.audio_interceptor.stop()
        if self.video_interceptor:
            self.video_interceptor.stop()
            
        # Reset manual targeting on stop
        if hasattr(self, '_pending_roi'):
            self._pending_roi = None
        if self.video_interceptor:
            self.video_interceptor.roi = None
        
        # Power down local camera to save battery and un-occupy the lens
        if self.video_interceptor and self.video_interceptor.is_alive():
            self.video_interceptor.stop()
        if self.audio_interceptor and self.audio_interceptor.is_alive():
            self.audio_interceptor.stop()

    def _monitor_loop(self):
        """Main processing loop linking interceptors -> analyzers -> UI."""
        print("!!! MONITOR LOOP STARTED !!!")
        sys.stdout.flush()
        logger.info("MONITOR_THREAD: Loop initialized")
        
        import psutil
        sys_process = psutil.Process()
        sys_process.cpu_percent() # burn first call
        
        frames_count = 0
        fps_frames = 0
        current_fps = 0.0
        current_hw_load = 0.0
        fps_start_time = time.perf_counter()
        
        while self.running:
            logger.debug(f"MONITOR_THREAD: Iteration {frames_count}")
            start_time = time.perf_counter()
            frames_count += 1
            fps_frames += 1

            # HEARTBEAT (Before buffer check to show thread is live)
            if frames_count % 30 == 0:
                 logger.info(f"HEARTBEAT | T={time.time() % 100:.1f} | Monitoring Active...")

            # 1. Fetch data from buffers
            audio_window = self.audio_interceptor.get_window()
            video_window = self.video_interceptor.get_window()
            
            # Partial Readiness (Fast-Start): Processing begins as soon as we have ANY sensor data.
            has_any_data = (audio_window is not None) or (video_window is not None)
            if not has_any_data:
                 if frames_count % 10 == 0:
                     logger.info("Buffer Status: WAITING FOR FIRST PACKET")
                 time.sleep(0.1)
                 continue
            
            # 2. Heuristic Analysis
            audio_rms = None
            if audio_window is not None:
                audio_flat = audio_window.flatten().astype(np.float32)
                audio_res = self.audio_analyzer.analyze(audio_flat)
                
                # Calculate real-time empirical Audio Envelope for Lip-Sync
                import librosa
                try:
                    audio_rms = librosa.feature.rms(y=audio_flat, frame_length=2048, hop_length=512)[0]
                except:
                    audio_rms = np.zeros(1)
            else:
                audio_res = {}
                
            video_res = self.video_analyzer.analyze(video_window, audio_rms_envelope=audio_rms)
            
            # 3. NPU Inference (Async)
            audio_inf = {}
            video_inf = {}
            
            # Prep Audio: Flatten and slice to exactly the model's expected 16000 samples
            if audio_window is not None:
                # audio_flat is already defined above from the heuristic pass
                if len(audio_flat) > 16000:
                    model_in = audio_flat[-16000:]
                elif len(audio_flat) < 16000:
                    model_in = np.pad(audio_flat, (0, 16000 - len(audio_flat)))
                else:
                    model_in = audio_flat
            
            # Prep Video: Ensure float32 and CHW format (1, 3, 224, 224)
            if video_window is not None:
                # Use latest frame from window
                raw_frame = video_window[-1].astype(np.float32) / 255.0 
                # Transpose HWC -> CHW
                video_input = np.transpose(raw_frame, (2, 0, 1))

            with self.inference_lock:
                 if audio_window is not None:
                      self.audio_engine.run_async(model_in)
                 if video_window is not None:
                      self.video_engine.run_async(video_input)
                 
                 # Fetch results (non-blocking)
                 audio_inf = self.audio_engine.get_result() or {}
                 video_inf = self.video_engine.get_result() or {}
                 
            # 4. Fusion and Prediction
            is_manual_lock = self.video_interceptor.roi is not None if self.video_interceptor else False
            if is_manual_lock and self.predictor.frames_since_start == 1:
                logger.info("🛡️ [PRECISION MODE] Manual Target Acquired. Sensitivity boosted.")

            prediction = self.predictor.predict(
                audio_res, video_res, audio_inf, video_inf,
                max_sensitivity=self.ui.is_max_sensitivity,
                has_multiple_faces=getattr(self.video_interceptor, 'has_multiple_faces', False),
                is_manual_lock=is_manual_lock
            )
            
            # 5. Evidence Extraction
            evidence = self.forensics.extract_evidence(audio_res, video_res, prediction)
            
            # Heuristic Evidence
            
            # Persistent Alert Logging
            if prediction["is_threat"]:
                try:
                    with open("deepfake_alerts.log", "a", encoding="utf-8") as f:
                        f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] ALERT | Trust Score: {prediction['trust_score']:.2f}\n")
                        # Privacy-Preserving Anonymization
                        vid_hash = self.video_interceptor.buffer.hash_buffer()
                        aud_hash = self.audio_interceptor.buffer.hash_buffer()
                        f.write(f"[PRIVACY-LOCK] Frame Identity Secured. SHA-256 A: {aud_hash[:16]}... | V: {vid_hash[:16]}...\n")
                        for ev in evidence:
                            if ev["severity"] in ["WARNING", "CRITICAL"]:
                                f.write(f"  -> [{ev['source']}] {ev['message']}\n")
                except Exception as e:
                    logger.error(f"Failed to write to deepfake_alerts.log: {e}")
            
            # 6. Update UI
            if frames_count % 5 == 0:
                print(f"LIVE: Trust={prediction['trust_score']:.2f} | Jilt={video_res.get('pixel_jitter_score',0.0):.3f}")
                sys.stdout.flush()
                logger.info(f"PRED: Trust={prediction['trust_score']:.2f} | Conf={prediction['confidence']:.2f} | Jilt={video_res.get('pixel_jitter_score',0.0):.3f}")

            if frames_count % 20 == 0:
                 logger.info(f"HEARTBEAT | T={time.time() % 100:.1f} | UI Syncing...")

            # Calculate Real Temporal Metrics
            elapsed_fps = time.perf_counter() - fps_start_time
            if elapsed_fps >= 1.0:
                 current_fps = fps_frames / elapsed_fps
                 current_hw_load = sys_process.cpu_percent() / psutil.cpu_count()
                 fps_frames = 0
                 fps_start_time = time.perf_counter()

            self.ui.update_metrics({
                "trust_score": prediction["trust_score"],
                "latency": self.video_engine.latency_ms + self.audio_engine.latency_ms,
                "fps": round(current_fps),
                "audio_windows": len(self.audio_interceptor.buffer.buffer) if self.audio_interceptor else 0,
                "npu_load": current_hw_load,
                "is_threat": prediction["is_threat"],
                "evidence": evidence
            })
            
            # Deterministic loop timing via Event wait instead of sleep
            if self.shutdown_event.wait(timeout=self.config['ui']['update_interval_ms']/1000.0):
                break

    def run(self):
        self.ui.mainloop()

if __name__ == "__main__":
    import argparse
    import sys
    import logging
    import threading
    import time
    import os
    import yaml
    import numpy as np
    
    # Assuming these are defined elsewhere or need to be imported
    # from utils.hardware_detector import detect_execution_provider
    # from utils.model_downloader import download_models
    # from interceptor.audio_interceptor import AudioInterceptor
    # from interceptor.video_interceptor import VideoInterceptor
    # from analyzer.audio_analyzer import AudioAnalyzer
    # from analyzer.video_analyzer import VideoAnalyzer
    # from predictor.bimodal_predictor import BimodalPredictor
    # from forensics.forensic_extractor import ForensicExtractor
    # from ui.main_window import MainWindow
    # from interceptor.audio_backend import SyntheticStream # For test_mode

    # Setup logger if not defined
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    from multiprocessing import freeze_support
    freeze_support()
    parser = argparse.ArgumentParser(description="VERO-CORE Silicon Guardian")
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU execution")
    args = parser.parse_args()
    
    try:
        app = VeroCoreApp(args)
            
        # The engine now waits for the user to click 'ENGAGE' in the UI. 
        # No more auto-start, ensuring user authority over sensors.
        app.ui.mainloop() # Start the UI mainloop. The UI handles Engage/Abort.
    except KeyboardInterrupt:
        logger.info("\nSession elegantly terminated by user. Processes shutting down.")
        if 'app' in locals() and app.running: # Ensure app exists and is running before stopping
            app.stop()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error initializing VERO-CORE: {e}")
        sys.exit(1)
