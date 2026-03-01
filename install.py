import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Installer")

PACKAGES_CORE = [
    "numpy>=1.24.0", "opencv-python>=4.8.0", "customtkinter>=5.2.0",
    "mediapipe>=0.10.0", "librosa>=0.10.0", "mss>=9.0.1", "pyyaml>=6.0",
    "tqdm>=4.65.0", "huggingface-hub>=0.20.0", "onnx>=1.14.0",
    "onnxruntime>=1.17.0", "pywin32>=306", "pillow>=10.0.0",
    "colorama>=0.4.6", "onnxconverter-common>=1.13.0",
    "tabulate>=0.9.0",
]

AUDIO_TIER1 = "pyaudiowpatch>=0.2.12.6"
AUDIO_TIER2 = "sounddevice>=0.4.6"

def install():
    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║       VERO-CORE Technical Installer      ║")
    logger.info("╚══════════════════════════════════════════╝")
    
    # 1. Core Packages
    logger.info("\n[1/3] Installing core dependencies...")
    for pkg in PACKAGES_CORE:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            logger.info(f"  ✅ {pkg}")
        except Exception:
            logger.error(f"  ❌ Failed to install {pkg}")

    # 2. Audio Backend Tiers
    logger.info("\n[2/3] Configuring audio backend...")
    tier1_installed = False
    try:
        subprocess.check_output([sys.executable, "-m", "pip", "install", AUDIO_TIER1], stderr=subprocess.STDOUT)
        logger.info(f"  ✅ Audio Tier 1: {AUDIO_TIER1} installed")
        tier1_installed = True
    except subprocess.CalledProcessError:
        logger.warning(f"  ⚠️  Tier 1 fail (PortAudio build likely) — falling back to Tier 2")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", AUDIO_TIER2])
            logger.info(f"  ✅ Audio Tier 2: {AUDIO_TIER2} installed")
        except Exception:
            logger.error("  ❌ Tier 2 also failed. System will use Tier 3 (Synthetic)")

    # 3. PyTorch (CUDA check)
    logger.info("\n[3/3] Optimizing for hardware acceleration...")
    try:
        # Try installing CUDA version if it fails, fallback to CPU
        logger.info("  Attempting CUDA-optimized Torch install...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cu121"])
        logger.info("  ✅ PyTorch CUDA 12.1 installed")
    except Exception:
        logger.warning("  ⚠️  CUDA install failed — installing CPU-only Torch")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])
        logger.info("  ✅ PyTorch CPU installed")

    logger.info("\n" + "═"*44)
    logger.info("  VERO-CORE Installation Complete")
    logger.info("  Next step: python check_env.py")
    logger.info("═"*44)

if __name__ == "__main__":
    install()
