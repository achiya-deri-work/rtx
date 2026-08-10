"""RTX low-precision linear layers and architecture-aware autotuning."""

from ._version import __version__


_LAZY_EXPORTS = {
    # Stable public format and frontend API.
    "MXTensor": ("formats", "MXTensor"),
    "MXFP8Tensor": ("formats", "MXFP8Tensor"),
    "NVFP4Tensor": ("formats", "NVFP4Tensor"),
    "LinearOperandState": ("formats", "LinearOperandState"),
    "MXFP8Linear": ("fp8", "MXFP8Linear"),
    "NVFP4Linear": ("fp4", "NVFP4Linear"),
    "mxfp8_linear": ("fp8", "mxfp8_linear"),
    "nvfp4_linear": ("fp4", "nvfp4_linear"),
    "quantize_mxfp8": ("fp8", "quantize_mxfp8"),
    "quantize_nvfp4": ("fp4", "quantize_nvfp4"),
    "DEFAULT_MXFP8_INFERENCE_CONFIG": ("fp8", "DEFAULT_MXFP8_INFERENCE_CONFIG"),
    "DEFAULT_MXFP8_PREQUANT_CONFIG": ("fp8", "DEFAULT_MXFP8_PREQUANT_CONFIG"),
    "MXFP8PrequantConfig": ("fp8", "MXFP8PrequantConfig"),
    "MXFP8QuantConfig": ("configs", "MXFP8QuantConfig"),
    "MXFP8GemmConfig": ("configs", "MXFP8GemmConfig"),
    "MXFP8WeightPrequantConfig": ("configs", "MXFP8WeightPrequantConfig"),
    "MXFP8FullyPrequantConfig": ("configs", "MXFP8FullyPrequantConfig"),
    "MXFP8FwdConfig": ("kernels.mxfp8", "MXFP8FwdConfig"),
    "MXFP8Problem": ("kernels.mxfp8", "MXFP8Problem"),
    "NVFP4FwdConfig": ("configs", "NVFP4FwdConfig"),
    "NVFP4GemmConfig": ("configs", "NVFP4GemmConfig"),
    "NVFP4QuantConfig": ("configs", "NVFP4QuantConfig"),
    "NVFP4Problem": ("configs", "NVFP4Problem"),
    "DEFAULT_MXFP8_BWD_CONFIG": ("fp8_bwd", "DEFAULT_MXFP8_BWD_CONFIG"),
    "DEFAULT_FUSED_MXFP8_BWD_CONFIG": (
        "fp8_bwd",
        "DEFAULT_FUSED_MXFP8_BWD_CONFIG",
    ),
    "MXFP8BwdConfig": ("fp8_bwd", "MXFP8BwdConfig"),
    "MXFP8BwdMatmulConfig": ("fp8_bwd", "MXFP8BwdMatmulConfig"),
    "mxfp8_linear_backward": ("fp8_bwd", "mxfp8_linear_backward"),
    "clear_runtime_caches": ("runtime", "clear_runtime_caches"),
    # Compatibility exports for the legacy production tuners.
    "CoordinateDescentPolicy": ("autotune", "CoordinateDescentPolicy"),
    "load_cached_mxfp8_fwd_config": ("autotune", "load_cached_mxfp8_fwd_config"),
    "tune_mxfp8_fwd": ("autotune", "tune_mxfp8_fwd"),
    "BWD_COORDINATE_ORDER": ("bwd_autotune", "BWD_COORDINATE_ORDER"),
    "BWD_SEARCH_SPACE": ("bwd_autotune", "BWD_SEARCH_SPACE"),
    "BwdTuningResult": ("bwd_autotune", "BwdTuningResult"),
    "load_cached_mxfp8_bwd_config": ("bwd_autotune", "load_cached_mxfp8_bwd_config"),
    "load_mxfp8_bwd_config": ("bwd_autotune", "load_mxfp8_bwd_config"),
    "tune_mxfp8_backward": ("bwd_autotune", "tune_mxfp8_backward"),
    "PREQUANT_COORDINATE_ORDER": ("prequant_autotune", "PREQUANT_COORDINATE_ORDER"),
    "PREQUANT_SEARCH_SPACE": ("prequant_autotune", "PREQUANT_SEARCH_SPACE"),
    "PrequantTuningResult": ("prequant_autotune", "PrequantTuningResult"),
    "load_cached_mxfp8_prequant_config": (
        "prequant_autotune",
        "load_cached_mxfp8_prequant_config",
    ),
    "tune_mxfp8_prequant": ("prequant_autotune", "tune_mxfp8_prequant"),
    "tune_mxfp8_inference_state": (
        "inference_autotune",
        "tune_mxfp8_inference_state",
    ),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    from importlib import import_module

    value = getattr(import_module(f".{module_name}", __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = ["__version__", *_LAZY_EXPORTS]
