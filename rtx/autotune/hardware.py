"""Hardware discovery and normalized resource features for CUDA autotuning.

The search space remains kernel-owned.  This module supplies facts and soft
resource estimates to legality checks and learned search without baking one
GPU's preferred schedule into the tuner.
"""

from __future__ import annotations

import ctypes
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Mapping

import torch


@dataclass(frozen=True, slots=True)
class ArchitectureProfile:
    key: str
    execution_model: str
    tensor_accumulator: str
    supports_tma: bool
    supports_mxfp8: bool
    supports_nvfp4: bool
    cuda_cores_per_sm: int | None = None
    tensor_cores_per_sm: int | None = None
    register_allocation_unit_per_warp: int | None = None
    shared_memory_allocation_unit: int | None = None


_ARCHITECTURES: dict[tuple[int, int], ArchitectureProfile] = {
    (12, 0): ArchitectureProfile(
        "sm120",
        "warp_mma_sync",
        "rmem",
        True,
        True,
        True,
        cuda_cores_per_sm=128,
        tensor_cores_per_sm=4,
        register_allocation_unit_per_warp=256,
        shared_memory_allocation_unit=256,
    ),
    (12, 1): ArchitectureProfile(
        "sm121",
        "warp_mma_sync",
        "rmem",
        True,
        True,
        True,
        cuda_cores_per_sm=128,
        tensor_cores_per_sm=4,
        register_allocation_unit_per_warp=256,
        shared_memory_allocation_unit=256,
    ),
}


# Official reference/module specifications. Board vendors and laptop power or
# memory configurations may differ, so runtime/NVML and calibration values win.
_SKU_SPECS: tuple[tuple[re.Pattern[str], dict[str, object]], ...] = (
    (
        re.compile(r"\bRTX 5090 Laptop\b", re.IGNORECASE),
        {
            "sku_family": "geforce_rtx_5090_laptop",
            "memory_type": "GDDR7",
            "memory_bus_width_bits": 256,
            "theoretical_memory_bandwidth_gbps": 896.0,
            "cuda_core_count": 10_496,
            "spec_source": "nvidia_reference",
        },
    ),
    (
        re.compile(r"\bRTX 5080 Laptop\b", re.IGNORECASE),
        {
            "sku_family": "geforce_rtx_5080_laptop",
            "memory_type": "GDDR7",
            "memory_bus_width_bits": 256,
            "theoretical_memory_bandwidth_gbps": 896.0,
            "cuda_core_count": 7_680,
            "spec_source": "nvidia_reference",
        },
    ),
    (
        re.compile(r"\bRTX 5070 Ti Laptop\b", re.IGNORECASE),
        {
            "sku_family": "geforce_rtx_5070_ti_laptop",
            "memory_type": "GDDR7",
            "memory_bus_width_bits": 192,
            "theoretical_memory_bandwidth_gbps": 672.0,
            "cuda_core_count": 5_888,
            "spec_source": "nvidia_reference",
        },
    ),
    (
        re.compile(r"\bRTX 5070 Laptop\b", re.IGNORECASE),
        {
            "sku_family": "geforce_rtx_5070_laptop",
            "memory_type": "GDDR7",
            "memory_bus_width_bits": 128,
            "theoretical_memory_bandwidth_gbps": 384.0,
            "cuda_core_count": 4_608,
            "spec_source": "nvidia_reference",
        },
    ),
    (
        re.compile(r"\bRTX 5090\b(?!.*Laptop)", re.IGNORECASE),
        {
            "sku_family": "geforce_rtx_5090",
            "memory_type": "GDDR7",
            "memory_bus_width_bits": 512,
            "theoretical_memory_bandwidth_gbps": 1792.0,
            "cuda_core_count": 21_760,
            "spec_source": "nvidia_reference",
        },
    ),
    (
        re.compile(r"\bRTX 5070 Ti\b(?!.*Laptop)", re.IGNORECASE),
        {
            "sku_family": "geforce_rtx_5070_ti",
            "memory_type": "GDDR7",
            "memory_bus_width_bits": 256,
            "theoretical_memory_bandwidth_gbps": 896.0,
            "cuda_core_count": 8_960,
            "spec_source": "nvidia_reference",
        },
    ),
)


def architecture_profile(capability: tuple[int, int]) -> ArchitectureProfile:
    return _ARCHITECTURES.get(
        capability,
        ArchitectureProfile(
            f"sm{capability[0]}{capability[1]}",
            "unknown",
            "unknown",
            False,
            False,
            False,
        ),
    )


def sku_spec(name: str) -> dict[str, object]:
    for pattern, values in _SKU_SPECS:
        if pattern.search(name):
            return dict(values)
    return {"sku_family": "unknown", "spec_source": "unavailable"}


def _file_identity(path: Path) -> dict[str, object]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


@lru_cache(maxsize=1)
def compiler_profile() -> dict[str, object]:
    """Identify the toolkit and libNVVM that can affect generated-code legality."""

    environment = {
        name: os.environ.get(name)
        for name in (
            "CUDA_TOOLKIT_PATH",
            "CUDA_HOME",
            "CUTE_DSL_ARCH",
            "QUACK_ARCH",
        )
        if os.environ.get(name) is not None
    }
    candidates: list[Path] = []
    for root_name in ("CUDA_TOOLKIT_PATH", "CUDA_HOME"):
        root = os.environ.get(root_name)
        if root:
            candidates.extend(
                (
                    Path(root) / "nvvm" / "lib64" / "libnvvm.so",
                    Path(root) / "lib64" / "libnvvm.so",
                )
            )
    try:
        site_packages = Path(torch.__file__).resolve().parent.parent
        candidates.extend(site_packages.glob("nvidia/**/libnvvm.so*"))
    except Exception:
        pass
    seen: set[Path] = set()
    libraries: list[dict[str, object]] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
            if resolved in seen or not resolved.is_file():
                continue
            seen.add(resolved)
            libraries.append(_file_identity(resolved))
        except OSError:
            continue
    nvcc = shutil.which("nvcc")
    nvcc_version = None
    if nvcc is not None:
        try:
            completed = subprocess.run(
                [nvcc, "--version"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            nvcc_version = completed.stdout.strip()
        except subprocess.SubprocessError:
            pass
    return {
        "environment": environment,
        "nvcc_path": nvcc,
        "nvcc_version": nvcc_version,
        "libnvvm_candidates": libraries,
    }


def device_properties(device: torch.device | str = "cuda") -> dict[str, object]:
    """Return every stable CUDA resource limit exposed by PyTorch."""

    props = torch.cuda.get_device_properties(torch.device(device))

    def integer(*names: str) -> int | None:
        for name in names:
            value = getattr(props, name, None)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass
        return None

    values: dict[str, object] = {
        "name": props.name,
        "major": int(props.major),
        "minor": int(props.minor),
        "multiprocessor_count": int(props.multi_processor_count),
        "total_memory": int(props.total_memory),
    }
    optional = {
        "l2_cache_size": integer("L2_cache_size", "l2_cache_size"),
        "shared_memory_per_block": integer("shared_memory_per_block"),
        "shared_memory_per_block_optin": integer("shared_memory_per_block_optin"),
        "shared_memory_per_multiprocessor": integer("shared_memory_per_multiprocessor"),
        "regs_per_block": integer("regs_per_block"),
        "regs_per_multiprocessor": integer("regs_per_multiprocessor"),
        "max_threads_per_block": integer("max_threads_per_block"),
        "max_threads_per_multiprocessor": integer("max_threads_per_multi_processor"),
        "max_blocks_per_multiprocessor": integer("max_blocks_per_multi_processor"),
        "warp_size": integer("warp_size"),
        "clock_rate_khz": integer("clock_rate"),
        "memory_clock_rate_khz": integer("memory_clock_rate"),
        "memory_bus_width_bits": integer("memory_bus_width"),
    }
    values.update({key: value for key, value in optional.items() if value is not None})
    return values


def _nvml_uint(function_name: str, device_index: int) -> int | None:
    """Read one optional NVML unsigned device property without a dependency."""

    try:
        library = ctypes.CDLL("libnvidia-ml.so.1")
        init = getattr(library, "nvmlInit_v2")
        get_handle = getattr(library, "nvmlDeviceGetHandleByIndex_v2")
        query = getattr(library, function_name)
    except (OSError, AttributeError):
        return None
    handle = ctypes.c_void_p()
    value = ctypes.c_uint()
    init.restype = ctypes.c_int
    get_handle.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
    get_handle.restype = ctypes.c_int
    query.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
    query.restype = ctypes.c_int
    if init() != 0 or get_handle(device_index, ctypes.byref(handle)) != 0:
        return None
    if query(handle, ctypes.byref(value)) != 0:
        return None
    return int(value.value)


def _nvidia_smi_text(field: str, device_index: int) -> str | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(device_index),
                f"--query-gpu={field}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        value = completed.stdout.strip()
        return value or None
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def _nvidia_smi_uint(field: str, device_index: int) -> int | None:
    value = _nvidia_smi_text(field, device_index)
    match = None if value is None else re.search(r"\d+", value)
    return None if match is None else int(match.group())


def static_device_profile(
    device: torch.device | str = "cuda",
    *,
    calibration: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the architecture/SKU profile used directly by model features."""

    resolved = torch.device(device)
    index = torch.cuda.current_device() if resolved.index is None else resolved.index
    properties = device_properties(resolved)
    capability = (int(properties["major"]), int(properties["minor"]))
    architecture = architecture_profile(capability)
    sku = sku_spec(str(properties["name"]))
    calibration_bus = (
        calibration.get("memory_bus_width_bits") if calibration is not None else None
    )
    nvml_bus = _nvml_uint("nvmlDeviceGetMemoryBusWidth", index)
    smi_bus = None if nvml_bus else _nvidia_smi_uint("memory.bus_width", index)
    driver_version = _nvidia_smi_text("driver_version", index)
    if isinstance(calibration_bus, (int, float)) and calibration_bus > 0:
        sku["memory_bus_width_bits"] = int(calibration_bus)
        sku["memory_bus_width_source"] = "calibration_override"
    elif nvml_bus:
        sku["memory_bus_width_bits"] = nvml_bus
        sku["memory_bus_width_source"] = "nvml"
    elif smi_bus:
        sku["memory_bus_width_bits"] = smi_bus
        sku["memory_bus_width_source"] = "nvidia_smi"
    elif properties.get("memory_bus_width_bits"):
        sku["memory_bus_width_bits"] = properties["memory_bus_width_bits"]
        sku["memory_bus_width_source"] = "cuda_runtime"
    elif sku.get("memory_bus_width_bits"):
        sku["memory_bus_width_source"] = "nvidia_spec_database"

    sm_count = int(properties["multiprocessor_count"])
    if architecture.cuda_cores_per_sm is not None:
        sku.setdefault("inferred_cuda_core_count", sm_count * architecture.cuda_cores_per_sm)
    if architecture.tensor_cores_per_sm is not None:
        sku.setdefault("inferred_tensor_core_count", sm_count * architecture.tensor_cores_per_sm)
    calibration_values = (
        {}
        if calibration is None
        else {key: value for key, value in calibration.items() if key != "hardware_profile"}
    )
    return {
        "schema_version": 1,
        # Frequently used values remain flat for old adapters/readers.
        "name": properties["name"],
        "capability": list(capability),
        "multiprocessor_count": sm_count,
        "total_memory": properties["total_memory"],
        "properties": properties,
        "architecture": asdict(architecture),
        "sku": sku,
        "driver_version": driver_version,
        "compiler": compiler_profile(),
        "calibration": calibration_values,
    }


def profile_value(
    profile: Mapping[str, object] | None,
    key: str,
    default: float | int | None = None,
) -> float | int | None:
    if not profile:
        return default
    candidates: list[Mapping[str, object]] = [profile]
    for section in ("properties", "sku", "calibration"):
        nested = profile.get(section)
        if isinstance(nested, Mapping):
            candidates.append(nested)
    for candidate in candidates:
        value = candidate.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    return default


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def geometry_features(
    *,
    m: int,
    n: int,
    k: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    profile: Mapping[str, object] | None,
    grid_ctas: int | None = None,
) -> dict[str, float]:
    sm_count = max(1, int(profile_value(profile, "multiprocessor_count", 1) or 1))
    m_tiles = _ceil_div(m, tile_m)
    n_tiles = _ceil_div(n, tile_n)
    natural_ctas = m_tiles * n_tiles
    actual_ctas = natural_ctas if grid_ctas is None else max(1, grid_ctas)
    return {
        "m_tiles": float(m_tiles),
        "n_tiles": float(n_tiles),
        "k_tiles": float(_ceil_div(k, tile_k)),
        "natural_ctas": float(natural_ctas),
        "grid_ctas": float(actual_ctas),
        "cta_waves_single_residency": actual_ctas / sm_count,
        "complete_waves_single_residency": float(actual_ctas // sm_count),
        "last_wave_fraction_single_residency": (actual_ctas % sm_count) / sm_count,
        "m_tail_fraction": (m % tile_m) / tile_m,
        "n_tail_fraction": (n % tile_n) / tile_n,
        "k_tail_fraction": (k % tile_k) / tile_k,
        "aspect_mn": m / max(1, n),
        "aspect_km": k / max(1, m),
        "aspect_kn": k / max(1, n),
    }


def launch_resource_features(
    *,
    profile: Mapping[str, object] | None,
    grid_ctas: int,
    threads_per_cta: int,
    smem_bytes_per_cta: int,
    register_budget_per_cta: int,
    register_limit_per_thread: int,
) -> dict[str, float]:
    """Estimate occupancy from declared budgets; compiled values remain authoritative."""

    sm_count = max(1, int(profile_value(profile, "multiprocessor_count", 1) or 1))
    threads_sm = int(profile_value(profile, "max_threads_per_multiprocessor", 0) or 0)
    smem_sm = int(profile_value(profile, "shared_memory_per_multiprocessor", 0) or 0)
    regs_sm = int(profile_value(profile, "regs_per_multiprocessor", 0) or 0)
    blocks_sm = int(profile_value(profile, "max_blocks_per_multiprocessor", 0) or 0)
    limits: list[int] = []
    by_threads = threads_sm // max(1, threads_per_cta) if threads_sm else 0
    by_smem = smem_sm // smem_bytes_per_cta if smem_sm and smem_bytes_per_cta else 0
    by_regs = regs_sm // register_budget_per_cta if regs_sm and register_budget_per_cta else 0
    if threads_sm:
        limits.append(by_threads)
    if smem_sm and smem_bytes_per_cta:
        limits.append(by_smem)
    if regs_sm and register_budget_per_cta:
        limits.append(by_regs)
    if blocks_sm:
        limits.append(blocks_sm)
    resident = max(0, min(limits) if limits else 1)
    scheduling_resident = max(1, resident)
    capacity_ctas = sm_count * scheduling_resident
    active_warps = resident * _ceil_div(threads_per_cta, 32)
    max_warps = threads_sm // 32 if threads_sm else 0
    return {
        "threads_per_cta": float(threads_per_cta),
        "warps_per_cta": float(_ceil_div(threads_per_cta, 32)),
        "smem_bytes_per_cta": float(smem_bytes_per_cta),
        "register_budget_per_cta": float(register_budget_per_cta),
        "register_limit_per_thread": float(register_limit_per_thread),
        "resident_ctas_by_threads": float(by_threads),
        "resident_ctas_by_smem": float(by_smem),
        "resident_ctas_by_register_budget": float(by_regs),
        "estimated_resident_ctas_per_sm": float(resident),
        "estimated_resident_warps_per_sm": float(active_warps),
        "estimated_warp_occupancy": active_warps / max_warps if max_warps else 0.0,
        "effective_cta_waves": grid_ctas / max(1, capacity_ctas),
        "effective_last_wave_fraction": (grid_ctas % capacity_ctas) / capacity_ctas,
        "smem_fraction_per_cta": smem_bytes_per_cta / smem_sm if smem_sm else 0.0,
        "register_fraction_per_cta": register_budget_per_cta / regs_sm if regs_sm else 0.0,
        "thread_fraction_per_cta": threads_per_cta / threads_sm if threads_sm else 0.0,
    }


def traffic_features(
    *,
    m: int,
    n: int,
    k: int,
    tile_m: int,
    tile_n: int,
    input_element_bytes: int,
    output_element_bytes: int,
    profile: Mapping[str, object] | None,
    materialized_quant: bool,
) -> dict[str, float]:
    m_tiles = _ceil_div(m, tile_m)
    n_tiles = _ceil_div(n, tile_n)
    x_bytes = m * k * input_element_bytes
    w_bytes = n * k * input_element_bytes
    out_bytes = m * n * output_element_bytes
    if materialized_quant:
        operand_read_bytes = m * k + n * k
        quant_write_bytes = m * k + n * k + (m + n) * _ceil_div(k, 32)
        estimated_dram_bytes = x_bytes + w_bytes + quant_write_bytes + operand_read_bytes + out_bytes
    else:
        quant_write_bytes = 0
        operand_read_bytes = x_bytes * n_tiles + w_bytes * m_tiles
        estimated_dram_bytes = operand_read_bytes + out_bytes
    flops = 2.0 * m * n * k
    l2_bytes = float(profile_value(profile, "l2_cache_size", 0) or 0)
    bandwidth = float(
        profile_value(profile, "measured_dram_bandwidth_gbps", 0)
        or profile_value(profile, "theoretical_memory_bandwidth_gbps", 0)
        or 0
    )
    return {
        "nominal_flops": flops,
        "x_bytes": float(x_bytes),
        "weight_bytes": float(w_bytes),
        "output_bytes": float(out_bytes),
        "quantized_materialization_bytes": float(quant_write_bytes),
        "estimated_operand_read_bytes": float(operand_read_bytes),
        "estimated_total_memory_bytes": float(estimated_dram_bytes),
        "arithmetic_intensity_flops_per_byte": flops / max(1, estimated_dram_bytes),
        "x_reuse_ctas": float(n_tiles),
        "weight_reuse_ctas": float(m_tiles),
        "x_l2_ratio": x_bytes / l2_bytes if l2_bytes else 0.0,
        "weight_l2_ratio": w_bytes / l2_bytes if l2_bytes else 0.0,
        "working_set_l2_ratio": (x_bytes + w_bytes + out_bytes) / l2_bytes if l2_bytes else 0.0,
        "memory_roofline_ms": estimated_dram_bytes / (bandwidth * 1.0e6) if bandwidth else 0.0,
    }


_RESOURCE_ATTRIBUTE_NAMES = (
    "num_regs",
    "registers",
    "register_count",
    "shared_memory",
    "shared_memory_bytes",
    "smem_size",
    "dynamic_smem_size",
    "local_memory",
    "local_size_bytes",
    "spill_store_bytes",
    "spill_load_bytes",
    "max_threads_per_block",
    "binary_version",
    "ptx_version",
)


def compiled_resource_metadata(value: object) -> dict[str, object]:
    """Best-effort resource extraction across CuTe/TVM-FFI wrapper versions."""

    result: dict[str, object] = {}
    seen: set[int] = set()

    def visit(item: object, prefix: str, depth: int) -> None:
        if item is None or depth > 2 or id(item) in seen:
            return
        seen.add(id(item))
        for name in _RESOURCE_ATTRIBUTE_NAMES:
            try:
                field = getattr(item, name)
            except Exception:
                continue
            if isinstance(field, (bool, int, float, str)):
                result[f"{prefix}{name}"] = field
        for name in (
            "kernel",
            "function",
            "compiled",
            "launcher",
            "partial",
            "reducer",
            "converter",
            "module",
            "quant_x",
            "quant_w",
            "quant_a",
            "quant_b",
            "gemm",
            "dx",
            "dw",
        ):
            try:
                child = getattr(item, name)
            except Exception:
                continue
            if child is not item:
                visit(child, f"{prefix}{name}.", depth + 1)

    visit(value, "", 0)
    return {"available": bool(result), **result}


__all__ = [
    "ArchitectureProfile",
    "architecture_profile",
    "compiled_resource_metadata",
    "compiler_profile",
    "device_properties",
    "geometry_features",
    "launch_resource_features",
    "profile_value",
    "sku_spec",
    "static_device_profile",
    "traffic_features",
]
