"""RTX low-precision linear layers and architecture-aware autotuning."""

from ._version import __version__


_LAZY_EXPORTS = {
    # Stable public format and frontend API.
    "MXTensor": ("formats", "MXTensor"),
    "MXFP8Tensor": ("formats", "MXFP8Tensor"),
    "NVFP4Tensor": ("formats", "NVFP4Tensor"),
    "LinearOperandState": ("formats", "LinearOperandState"),
    "LinearExecutionDecision": ("selection", "LinearExecutionDecision"),
    "AutotuneMode": ("types", "AutotuneMode"),
    "LinearBackend": ("types", "LinearBackend"),
    "MXFP8Backend": ("types", "MXFP8Backend"),
    "NVFP4Backend": ("types", "NVFP4Backend"),
    "NVFP4ScalingMode": ("types", "NVFP4ScalingMode"),
    "MXFP8Linear": ("fp8", "MXFP8Linear"),
    "NVFP4Linear": ("fp4", "NVFP4Linear"),
    "DEFAULT_NVFP4_X_SCALE_REGION_ROWS": (
        "fp4",
        "DEFAULT_NVFP4_X_SCALE_REGION_ROWS",
    ),
    "DEFAULT_NVFP4_WEIGHT_SCALE_REGION_ROWS": (
        "fp4",
        "DEFAULT_NVFP4_WEIGHT_SCALE_REGION_ROWS",
    ),
    "mxfp8_linear": ("fp8", "mxfp8_linear"),
    "nvfp4_linear": ("fp4", "nvfp4_linear"),
    "quantize_mxfp8": ("fp8", "quantize_mxfp8"),
    "quantize_nvfp4": ("fp4", "quantize_nvfp4"),
    "quantize_": ("ptq", "quantize_"),
    "MXFP8WeightOnlyConfig": ("ptq", "MXFP8WeightOnlyConfig"),
    "NVFP4WeightOnlyConfig": ("ptq", "NVFP4WeightOnlyConfig"),
    "MXFP8TrainingConfig": ("ptq", "MXFP8TrainingConfig"),
    "NVFP4TrainingConfig": ("ptq", "NVFP4TrainingConfig"),
    "convert_to_mxfp8_training": ("ptq", "convert_to_mxfp8_training"),
    "convert_to_nvfp4_training": ("ptq", "convert_to_nvfp4_training"),
    "DEFAULT_MXFP8_INFERENCE_CONFIG": ("fp8", "DEFAULT_MXFP8_INFERENCE_CONFIG"),
    "DEFAULT_MXFP8_PREQUANT_CONFIG": ("fp8", "DEFAULT_MXFP8_PREQUANT_CONFIG"),
    "MXFP8PrequantConfig": ("fp8", "MXFP8PrequantConfig"),
    "MXFP8QuantConfig": ("configs", "MXFP8QuantConfig"),
    "MXFP8GemmConfig": ("configs", "MXFP8GemmConfig"),
    "MXFP8WeightPrequantConfig": ("configs", "MXFP8WeightPrequantConfig"),
    "MXFP8FullyPrequantConfig": ("configs", "MXFP8FullyPrequantConfig"),
    "MXFP8FwdConfig": ("kernels.mxfp8", "MXFP8FwdConfig"),
    "MXFP8Problem": ("kernels.mxfp8", "MXFP8Problem"),
    "NVFP4ScaleConfig": ("configs", "NVFP4ScaleConfig"),
    "NVFP4DynamicConfig": ("configs", "NVFP4DynamicConfig"),
    "NVFP4GemmConfig": ("configs", "NVFP4GemmConfig"),
    "NVFP4FullyPrequantConfig": ("configs", "NVFP4FullyPrequantConfig"),
    "NVFP4QuantConfig": ("configs", "NVFP4QuantConfig"),
    "NVFP4WeightPrequantConfig": ("configs", "NVFP4WeightPrequantConfig"),
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
    "validate_runtime_environment": ("runtime", "validate_runtime_environment"),
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
    "tune_nvfp4_inference_state": (
        "nvfp4_inference_autotune",
        "tune_nvfp4_inference_state",
    ),
}

# ``from rtx import *`` is intentionally limited to the stable runtime surface.
# Historical tuner symbols remain directly addressable for compatibility, but
# new integrations should import them from ``rtx.autotune``.
_STABLE_EXPORTS = (
    "MXTensor", "MXFP8Tensor", "NVFP4Tensor", "LinearOperandState",
    "LinearExecutionDecision", "AutotuneMode", "LinearBackend",
    "MXFP8Backend", "NVFP4Backend", "NVFP4ScalingMode", "MXFP8Linear", "NVFP4Linear",
    "mxfp8_linear", "nvfp4_linear", "quantize_mxfp8", "quantize_nvfp4",
    "quantize_", "MXFP8WeightOnlyConfig",
    "NVFP4WeightOnlyConfig", "MXFP8TrainingConfig", "NVFP4TrainingConfig",
    "convert_to_mxfp8_training", "convert_to_nvfp4_training",
    "DEFAULT_NVFP4_X_SCALE_REGION_ROWS",
    "DEFAULT_NVFP4_WEIGHT_SCALE_REGION_ROWS",
    "DEFAULT_MXFP8_INFERENCE_CONFIG", "DEFAULT_MXFP8_PREQUANT_CONFIG",
    "MXFP8PrequantConfig", "MXFP8QuantConfig", "MXFP8GemmConfig",
    "MXFP8WeightPrequantConfig", "MXFP8FullyPrequantConfig",
    "MXFP8FwdConfig", "MXFP8Problem", "NVFP4ScaleConfig",
    "NVFP4DynamicConfig", "NVFP4GemmConfig", "NVFP4FullyPrequantConfig",
    "NVFP4QuantConfig", "NVFP4WeightPrequantConfig", "NVFP4Problem",
    "DEFAULT_MXFP8_BWD_CONFIG", "DEFAULT_FUSED_MXFP8_BWD_CONFIG",
    "MXFP8BwdConfig", "MXFP8BwdMatmulConfig", "mxfp8_linear_backward",
    "clear_runtime_caches", "validate_runtime_environment",
)


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
    return sorted(set(globals()) | set(_STABLE_EXPORTS))


__all__ = ["__version__", *_STABLE_EXPORTS]
