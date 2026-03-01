import numpy as np
from collections import deque
from threading import Lock

class BufferManager:
    """
    Thread-safe, fixed-size circular buffer using collections.deque.
    Generic: works for both numpy audio arrays and video frame arrays.
    Zero disk I/O — all data lives in RAM.
    """
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = Lock()

    def push(self, data: np.ndarray):
        with self.lock:
            self.buffer.append(data)

    def get_window(self) -> np.ndarray:
        """
        Returns the full sliding window using a resilient stack operation.
        """
        with self.lock:
            if not self.buffer:
                return None
            try:
                # Optimized fast path for uniform shapes
                return np.stack(self.buffer)
            except ValueError:
                # Fallback path: inconsistent shapes detected
                # Filter for most common shape if needed, or simply return the latest frame as a window of 1
                latest = self.buffer[-1]
                return np.expand_dims(latest, axis=0)
            except Exception:
                return None

    def is_ready(self) -> bool:
        """Returns True if the buffer is full."""
        with self.lock:
            return len(self.buffer) == self.max_size

    def clear(self):
        with self.lock:
            self.buffer.clear()

    def get_latest(self) -> np.ndarray:
        """Returns the most recent item added to the buffer."""
        with self.lock:
            if not self.buffer:
                return np.array([])
            return self.buffer[-1]

    def hash_buffer(self) -> str:
        """
        [THEME 6 PRIVACY COMPLIANCE]
        Generates a SHA-256 cryptographic hash of the current Volatile RAM buffer.
        Proves exactly what mathematical signature triggered a threat without EVER 
        writing raw video frames or voice signatures to disk or returning them to cloud APIs.
        """
        import hashlib
        with self.lock:
            if not self.buffer:
                return "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" # Empty Hash
            
            try:
                # Combine bytes of recent frames in RAM
                window = np.stack(self.buffer)
                return hashlib.sha256(window.tobytes()).hexdigest()
            except Exception:
                # Fallback if stack fails
                return hashlib.sha256(self.buffer[-1].tobytes()).hexdigest()
