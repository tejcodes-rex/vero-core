import time
import numpy as np
import onnxruntime as ort
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Benchmark")

def run_benchmark(model_path, iterations=1000):
    providers = ort.get_available_providers()
    
    results = {}
    
    for provider in providers:
        if provider == 'TensorrtExecutionProvider': continue # Skip heavy ones for now
        
        logger.info(f"Benching {provider}...")
        try:
            session = ort.InferenceSession(model_path, providers=[provider])
            input_name = session.get_inputs()[0].name
            input_shape = session.get_inputs()[0].shape
            input_shape = [s if isinstance(s, int) else 1 for s in input_shape]
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
            
            # Warmup
            for _ in range(10):
                session.run(None, {input_name: dummy_input})
                
            latencies = []
            for _ in range(iterations):
                start = time.perf_counter()
                session.run(None, {input_name: dummy_input})
                latencies.append((time.perf_counter() - start) * 1000)
            
            results[provider] = {
                "p50": np.percentile(latencies, 50),
                "p95": np.percentile(latencies, 95),
                "p99": np.percentile(latencies, 99),
                "avg": np.mean(latencies)
            }
        except Exception as e:
            logger.warning(f"Failed to bench {provider}: {e}")

    print("\n" + "="*60)
    print(f"{'Execution Provider':<30} | {'Avg (ms)':<10} | {'P99 (ms)':<10}")
    print("-"*60)
    for p, stats in results.items():
        print(f"{p:<30} | {stats['avg']:<10.2f} | {stats['p99']:<10.2f}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="engine_npu/models/video_deepfake.onnx")
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    
    if os.path.exists(args.model):
        run_benchmark(args.model, args.iters)
    else:
        print(f"Model {args.model} not found. Run download_models.py first.")
