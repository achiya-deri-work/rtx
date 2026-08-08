from pathlib import Path
import re

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
VERSION = re.search(
    r'^__version__ = "([^"]+)"$',
    (ROOT / "rtx" / "_version.py").read_text(encoding="utf-8"),
    re.MULTILINE,
).group(1)


setup(
    name="rtx-mxfp8",
    version=VERSION,
    description="Trainable MXFP8 linear kernels and cross-device autotuning for RTX Blackwell",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    license_files=["LICENSE"],
    python_requires=">=3.11",
    packages=find_packages(include=("rtx", "rtx.*")),
    package_data={"rtx": ["py.typed"]},
    include_package_data=True,
    install_requires=[
        "numpy>=2.0,<3",
        "torch>=2.9",
        "cuda-python>=13.0,<14",
        "nvidia-cutlass-dsl>=4.7,<4.8",
    ],
    extras_require={
        "parquet": ["pyarrow>=16"],
        "dev": ["pytest>=8", "pyarrow>=16"],
    },
    entry_points={
        "console_scripts": [
            "rtx-autotune=rtx.autotune.dataset:main",
            "rtx-autotune-legacy=rtx.autotune.legacy:_cli",
            "rtx-prequant-experiment=rtx.prequant_experiments:_cli",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
