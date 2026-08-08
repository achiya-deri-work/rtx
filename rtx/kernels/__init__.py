"""CuTe DSL kernels and their autotuning configuration spaces."""

from .mxfp8 import MXFP8FwdConfig, MXFP8Problem


def __getattr__(name: str):
    if name in ("MXFP8GemmConfig", "MXFP8QuantConfig"):
        from .. import configs

        value = getattr(configs, name)
        globals()[name] = value
        return value
    raise AttributeError(name)

__all__ = [
    "MXFP8FwdConfig",
    "MXFP8GemmConfig",
    "MXFP8Problem",
    "MXFP8QuantConfig",
]
