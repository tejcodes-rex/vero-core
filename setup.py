from setuptools import setup, find_packages

setup(
    name="vero-core",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "customtkinter>=5.2.0",
        "mediapipe>=0.10.0",
        "librosa>=0.10.0",
        "mss>=9.0.1",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "huggingface-hub>=0.20.0",
        "onnx>=1.14.0",
        "onnxruntime>=1.17.0",
        "pywin32>=306",
        "pillow>=10.0.0",
        "torch>=2.1.0",
        "colorama>=0.4.6",
        "onnxconverter-common>=1.13.0",
    ],
)
