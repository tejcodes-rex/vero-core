import threading
import time
import logging
import numpy as np
from interceptor.audio_backend import AudioBackend
from interceptor.buffer_manager import BufferManager

logger = logging.getLogger("AudioInterceptor")

class AudioInterceptor(threading.Thread):
    """
    VERO-CORE Audio Interceptor — WASAPI Shared Mode
    Captures system audio output via AudioBackend (3-tier fallback).
    Pushes data into BufferManager for NPU analysis.
    """
    def __init__(self, sample_rate=16000, chunk_size=800, window_ms=1000):
        super().__init__(daemon=True)
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        
        # Safely compute how many chunks to hold based on window_ms
        num_chunks = int((window_ms / 1000.0 * sample_rate) / chunk_size)
        window_size = max(1, num_chunks)
        
        self.buffer = BufferManager(max_size=window_size)
        self.backend = AudioBackend.get_best_available(sample_rate, chunk_size)
        self.running = False
        self.sync_event = threading.Event()

    def run(self):
        logger.info(f"Starting audio capture using {self.backend.get_backend_name()}")
        self.backend.start()
        self.running = True
        while self.running:
            try:
                try:
                    data = self.backend.read()
                    if data is not None and len(data) > 0:
                        self.buffer.push(data)
                        if self.buffer.buffer.__len__() % 10 == 0:
                            logger.debug(f"Audio Push: {len(data)} samples (Total: {len(self.buffer.buffer)}/{self.buffer.max_size})")
                    
                    # Yield for NPU-satiation
                    if self.sync_event.wait(timeout=max(0, self.chunk_size / self.sample_rate)):
                        break
                except (RuntimeError, AttributeError) as e:
                    logger.error(f"DeviceUnavailable (Audio): {e}")
                    # Re-init as Synthetic
                    from interceptor.audio_backend import SyntheticStream, ResamplingWrapper
                    self.backend.stop() # Stop the current backend before switching
                    self.backend = ResamplingWrapper(SyntheticStream(self.sample_rate, self.chunk_size))
                    self.backend.start() # Start the new synthetic backend
                    if self.sync_event.wait(timeout=0.5): break
                except Exception as e:
                    logger.warning(f"Audio read error: {e}")
                    if self.sync_event.wait(timeout=0.1): break
            except Exception as e:
                logger.error(f"Fatal Audio Loop Error: {e}")
                break
        if self.backend: self.backend.stop()


    def stop(self):
        self.running = False
        self.sync_event.set()
        # DELEGATE TO WORKER THREAD:
        # Do not call self.backend.stop() here from the UI thread because
        # PyAudio will deadlock if the worker thread is inside a blocking read.

    def get_window(self) -> np.ndarray:
        window = self.buffer.get_window()
        if window is None:
            return None
            
        flat_audio = window.flatten().astype(np.float32)
        
        # Safe High-Fidelity Resampling Pipeline
        # We mathematically resample the entire 1-second block at once. 
        # This completely eradicates phase-distortion "clicks" at chunk boundaries.
        native_rate = getattr(self.backend, 'device_rate', self.sample_rate)
        if hasattr(self.backend, 'source_rate'):
            native_rate = self.backend.source_rate
            
        if native_rate != self.sample_rate:
            import librosa
            flat_audio = librosa.resample(flat_audio, orig_sr=native_rate, target_sr=self.sample_rate)
            
        return flat_audio

    def is_ready(self) -> bool:
        return self.buffer.is_ready()
