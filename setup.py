import os
from setuptools import setup
import torch

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 8], "Requires PyTorch >= 1.8"

setup(
    name="zs-cosod",
    author="Haoke Xiao",
    description="ZERO-SHOT CO-SALIENT OBJECT DETECTION FRAMEWORK",
    python_requires=">=3.8",
    py_modules=[],
    install_requires=[
        "loguru>=0.5.3",
        "faiss-cpu>=1.7.1",
        "matplotlib>=3.4.2",
        "Pillow==9.5.0",
        "tqdm>=4.61.2",
        "numpy==1.26.3",
        "gdown>=3.13.0",
        f"mask2former @ file://localhost/{os.getcwd()}/sd-dino/third_party/Mask2Former/",
        f"odise @ file://localhost/{os.getcwd()}/sd-dino/third_party/ODISE/"
    ],
    include_package_data=True,
)