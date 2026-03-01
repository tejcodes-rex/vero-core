import subprocess
import onnxruntime as ort
import logging
import os

logger = logging.getLogger("DeviceManager")

def detect_execution_provider() -> dict:
    """
    Full hardware detection for AMD Ryzen AI platform.
    Detection order: AMD NPU (XDNA/XDNA2) → NVIDIA CUDA → CPU
    """
    result = {
        "provider": "CPUExecutionProvider",
        "device_name": "Generic CPU",
        "npu_detected": False,
        "xdna_gen": None,
        "tops_rating": "N/A",
        "bf16_supported": False
    }

    available_providers = ort.get_available_providers()

    # Layer 1 — AMD NPU Detection
    npu_found = False
    try:
        # Method A: VitisAI EP check
        if 'VitisAIExecutionProvider' in available_providers:
            result["provider"] = "VitisAIExecutionProvider"
            npu_found = True
        
        # Method B: PCI Enumeration via pnputil (Physical check)
        pnp_output = subprocess.run(['pnputil', '/enum-devices', '/class', 'System'], 
                                   capture_output=True, text=True, check=False).stdout
        
        if "NPU" in pnp_output or "AMD IPU" in pnp_output or "XDNA" in pnp_output:
            npu_found = True
            result["npu_detected"] = True
            
            # Determine XDNA generation
            if "XDNA 2" in pnp_output or "Ryzen AI 300" in pnp_output:
                result["xdna_gen"] = "XDNA2"
                result["tops_rating"] = "50+"
                result["bf16_supported"] = True
            else:
                result["xdna_gen"] = "XDNA"
                result["tops_rating"] = "16"
                result["bf16_supported"] = False
                
            if result["provider"] == "CPUExecutionProvider":
                 # If NPU is there but EP is missing, we still note it
                 result["device_name"] = f"AMD {result['xdna_gen']} (EP Missing)"
            else:
                 result["device_name"] = f"AMD {result['xdna_gen']} NPU"

    except Exception as e:
        logger.warning(f"Silicon Probe Failed: {e}")

    # Layer 2 — CUDA Detection (if NPU not used)
    if not result["npu_detected"] or result["provider"] == "CPUExecutionProvider":
        if 'CUDAExecutionProvider' in available_providers:
            result["provider"] = "CUDAExecutionProvider"
            result["device_name"] = "NVIDIA GPU"
            try:
                import torch
                if torch.cuda.is_available():
                    result["device_name"] = torch.cuda.get_device_name(0)
            except Exception as e:
                logger.debug(f"Torch CUDA name retrieval failed: {e}")

    _log_silicon_attestation(result)
    return result

def _log_silicon_attestation(res: dict):
    header = "VERO-CORE Silicon Attestation"
    logger.info(f"{header:-^50}")
    logger.info(f"PROVIDER : {res['provider']}")
    logger.info(f"NPU      : {res['npu_detected']}")
    logger.info(f"TOPS     : {res['tops_rating']}")
    logger.info(f"BF16     : {res['bf16_supported']}")
    logger.info("-" * 50)

if __name__ == "__main__":
    detect_execution_provider()
