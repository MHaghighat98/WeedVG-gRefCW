# coding=utf-8
"""
Weed-VG: Agricultural Visual Grounding
Setup script — builds the Weed-VG package together with the GroundingDINO
CUDA extension (groundingdino._C / MultiScaleDeformableAttention).

Install with:  pip install -e .
"""

import glob
import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension


def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    with open(req_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return lines


def read_long_description():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, encoding="utf-8") as f:
            return f.read()
    return ""


def get_extensions():
    """Build the groundingdino._C CUDA extension (MultiScaleDeformableAttention)."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "Weed-VG", "groundingdino", "models", "GroundingDINO", "csrc")

    main_source = os.path.join(extensions_dir, "vision.cpp")
    sources = glob.glob(os.path.join(extensions_dir, "**", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "**", "*.cu")) + glob.glob(os.path.join(extensions_dir, "*.cu"))

    sources = [main_source] + sources
    extension = CppExtension
    extra_compile_args = {"cxx": []}
    define_macros = []

    # Blackwell GPU PTX setting.
    if "TORCH_CUDA_ARCH_LIST" not in os.environ and torch.cuda.is_available():
        try:
            for i in range(torch.cuda.device_count()):
                if torch.cuda.get_device_capability(i) == (12, 0):
                    print("Detected Blackwell GPU — forcing TORCH_CUDA_ARCH_LIST='9.0+PTX'.")
                    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0+PTX"
                    break
        except Exception:
            pass

    if CUDA_HOME is not None and (torch.cuda.is_available() or "TORCH_CUDA_ARCH_LIST" in os.environ):
        print("Compiling groundingdino._C with CUDA")
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
    else:
        print("Compiling groundingdino._C without CUDA (CPU only)")
        define_macros += [("WITH_HIP", None)]
        extra_compile_args["nvcc"] = []
        return None

    include_dirs = [extensions_dir]
    return [
        extension(
            "groundingdino._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]


setup(
    name="weedvg",
    version="1.0.0",
    author="Mohammadreza Haghighat",
    description=("Weed-VG: Agricultural Visual Grounding"),
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    package_dir={"": os.path.join("Weed-VG")},
    packages=find_packages(where=os.path.join("Weed-VG"), exclude=["assets", "data", "weights", "checkpoints"]),
    python_requires=">=3.10",
    install_requires=read_requirements(),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    entry_points={
        "console_scripts": [
            "weedvg-demo=demo:main",
        ],
    },
)
