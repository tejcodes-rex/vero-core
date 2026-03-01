import onnxruntime as ort
import numpy as np
import time
import logging

logger = logging.getLogger("InferenceEngine")

import multiprocessing as mp
from multiprocessing.connection import Connection

class InferenceProcess(mp.Process):
    """Isolated process for NPU/GPU inference with Background Warm-up."""
    def __init__(self, model_path: str, provider: str, input_queue: mp.Queue, output_queue: mp.Queue, ready_event: mp.Event):
        super().__init__(daemon=True)
        self.model_path = model_path
        self.provider = provider
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.ready_event = ready_event

    def run(self):
        # 1. Warm-Start: Pre-load and Wake up NPU
        engine = InferenceEngine(self.model_path, self.provider)
        engine.warmup(n=3)
        self.ready_event.set()
        logger.info(f"NPU Warm-Start Complete: {self.model_path} Ready")
        
        while True:
            try:
                input_data = self.input_queue.get()
                if input_data is None: break
                result = engine.run(input_data)
                self.output_queue.put(result)
            except Exception as e:
                logger.error(f"Inference Process Error: {e}")

class MultiprocessInferenceEngine:
    """Wrapper that manages an InferenceProcess with Warm-Start signaling."""
    def __init__(self, model_path: str, provider: str):
        self.input_queue = mp.Queue(maxsize=2)
        self.output_queue = mp.Queue(maxsize=2)
        self.ready_event = mp.Event()
        self.process = InferenceProcess(model_path, provider, self.input_queue, self.output_queue, self.ready_event)
        self.process.start()
        self.latency_ms = 0.0

    def is_ready(self) -> bool:
        return self.ready_event.is_set()

    def run_async(self, input_array: np.ndarray):
        """Non-blocking push to the inference process."""
        if not self.input_queue.full():
            self.input_queue.put(input_array)

    def get_result(self) -> dict:
        """Fetch result from the process if available."""
        if not self.output_queue.empty():
            res = self.output_queue.get()
            self.latency_ms = res.get("latency_ms", 0.0)
            return res
        return None

    def stop(self):
        self.input_queue.put(None)
        self.process.join()

class InferenceEngine:
    """
    ONNX Runtime wrapper for VERO-CORE.
    Supports VitisAI (AMD NPU), CUDA, and CPU execution providers.
    """
    def __init__(self, model_path: str, provider: str = "CPUExecutionProvider"):
        self.model_path = model_path
        self.provider = provider
        self.session = None
        self.input_name = None
        self.latency_ms = 0.0
        
        provider_options = {}
        if provider == "VitisAIExecutionProvider":
            # Link to Vitis AI config if it exists
            config_file = "engine_npu/vitis_ai_config.json"
            provider_options = {"config_file": config_file}
        elif provider == "CUDAExecutionProvider":
            provider_options = {"device_id": 0}

        try:
            self.session = ort.InferenceSession(
                model_path, 
                providers=[provider],
                provider_options=[provider_options] if provider_options else None
            )
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            logger.info(f"Loaded model {model_path} on {provider} | Input Shape: {self.input_shape}")
        except Exception as e:
            logger.error(f"Failed to load model on {provider}: {e}")
            # Fallback to CPU
            self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            logger.info(f"Fallback: Loaded model {model_path} on CPUExecutionProvider | Input Shape: {self.input_shape}")

    def warmup(self, n=3):
        """Pre-load model onto device."""
        # Handle dynamic shapes (None/-1)
        warmup_shape = [s if isinstance(s, int) and s > 0 else 1 for s in self.input_shape]
        dummy_input = np.random.randn(*warmup_shape).astype(np.float32)
        for _ in range(n):
            self.run(dummy_input)

    def run(self, input_array: np.ndarray) -> dict:
        """Run inference and return logits + latency."""
        start_time = time.perf_counter()
        if not hasattr(self, 'frames_count_inf'):
            self.frames_count_inf = 0
        self.frames_count_inf += 1
        
        # 1. Ensure type
        input_array = np.array(input_array).astype(np.float32)
        
        # 2. Automated Rank Matching
        target_rank = len(self.input_shape)
        while len(input_array.shape) < target_rank:
            input_array = np.expand_dims(input_array, axis=0)
            
        # 3. Shape validation (Optional but helpful for debugging)
        if hasattr(self, 'input_shape'):
             # Replace dynamic dims with current input size for comparison
             expected = [self.input_shape[i] if isinstance(self.input_shape[i], int) and self.input_shape[i] > 0 
                         else input_array.shape[i] for i in range(len(self.input_shape))]
             if list(input_array.shape) != expected:
                 # Attempt a reshape if the total elements match
                 if input_array.size == np.prod(expected):
                     input_array = input_array.reshape(expected)
            
        if self.frames_count_inf % 50 == 0:
            import os
            logger.info(f"Inference Run | Model: {os.path.basename(self.model_path)} | Shape: {input_array.shape}")
            
        outputs = self.session.run(None, {self.input_name: input_array})
        
        self.latency_ms = (time.perf_counter() - start_time) * 1000
        return {
            "logits": outputs[0],
            "latency_ms": self.latency_ms
        }
