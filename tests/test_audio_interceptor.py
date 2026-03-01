import pytest
import numpy as np
from interceptor.buffer_manager import BufferManager
from interceptor.audio_backend import SyntheticStream

def test_buffer_manager():
    buf = BufferManager(max_size=5)
    for i in range(10):
        buf.push(np.array([i]))
    
    window = buf.get_window()
    assert len(window) == 5
    assert window[-1][0] == 9
    assert buf.is_ready()

def test_synthetic_stream():
    stream = SyntheticStream(sample_rate=16000, chunk_size=800)
    stream.start()
    data = stream.read()
    assert len(data) == 800
    assert data.dtype == np.float32
    # Ensure it's not all zeros
    assert np.any(data != 0)
    stream.stop()
