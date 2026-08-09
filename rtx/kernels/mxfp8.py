"""Configuration and legality model for the fused MXFP8 linear kernels.

The configuration is intentionally broader than the first implemented kernel.  A
candidate is only sent to CuTe after :meth:`implementation_rejection` returns
``None``; this keeps every tuning dimension explicit without making no-op knobs
look like real benchmark results.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from itertools import product
from typing import Iterable, Iterator, Literal

LoadEngine = Literal["scalar", "cpasync", "tma"]
Schedule = Literal[
    "cooperative",
    "warp_specialized",
    "three_role",
    "pingpong",
]
MmaIssue = Literal["sync"]
Reduction = Literal["redux", "shuffle"]
QuantMath = Literal["fp32", "bf16x2"]
QuantAmax = Literal["fp32", "bf16_bits"]
Swizzle = Literal["none", "32b", "64b", "128b"]
Raster = Literal["m", "n"]
Reuse = Literal["none", "x", "weight"]
Epilogue = Literal["direct", "smem", "tma"]
SM120_SMEM_CAPACITY_BYTES = 101_376
# CuTe's prequantized GEMM wrapper adds pipeline barriers/descriptors outside
# the explicitly modeled operand storage. This reserve is part of legality,
# not an autotuner heuristic: a 101,376-byte raw tile launches as 102,400 B.
SM120_GEMM_RUNTIME_SMEM_RESERVE_BYTES = 1_024


@dataclass(frozen=True, slots=True)
class MXFP8Problem:
    """A flattened ``[M, K] @ [N, K].T -> [M, N]`` linear problem."""

    m: int
    n: int
    k: int

    def validate(self) -> None:
        if min(self.m, self.n, self.k) <= 0:
            raise ValueError(f"M, N and K must be positive, got {self}")
        if self.k % 32:
            raise ValueError(
                f"MXFP8 scale vectors contain 32 values, so K must be divisible "
                f"by 32; got K={self.k}"
            )


@dataclass(frozen=True, slots=True)
class MXFP8FwdConfig:
    """One independently compilable forward schedule.

    Values are part of the cache/tuning key even when a backend has not been
    implemented yet.  Such candidates are rejected with a named reason.
    """

    # Tensor-core tile and warp arrangement.
    tile_m: int = 128
    tile_n: int = 128
    tile_k: int = 128
    atom_layout_m: int = 4
    atom_layout_n: int = 2
    num_mma_warps: int = 8

    # Load/quantization pipeline.
    load_engine: LoadEngine = "scalar"
    schedule: Schedule = "cooperative"
    # BF16 transport tile is independent of the 128-wide MXFP8 MMA macro tile.
    # A 32/64-wide load stage permits deeper TMA buffering within RTX SMEM.
    bf16_tile_k: int = 128
    bf16_swizzle: Swizzle = "128b"
    bf16_stages: int = 1
    mxfp8_stages: int = 1
    producer_warps: int = 0
    quantizer_warps: int = 8
    # SM120 rejects floating-point redux.sync, but unsigned integer redux is
    # legal.  Since abs(FP32) bit patterns have the same ordering as their
    # numerical values, the redux variant lowers amax to redux.sync.max.u32.
    reduction: Reduction = "shuffle"
    # Quantization arithmetic/conversion backend.  The packed path uses native
    # two-lane BF16 arithmetic and the SM120 BF16x2 -> E4M3x2 converter.
    quant_math: QuantMath = "fp32"
    quant_amax: QuantAmax = "fp32"
    # Per-thread BF16 quantizer load width. Wider choices map to explicit
    # vector loads and are independently tuned from quant_vec/math.
    quant_load_bits: int = 16
    quant_vec: int = 1
    k_unroll: int = 1

    # Hardware issue and resource controls.
    mma_issue: MmaIssue = "sync"
    num_threads: int = 256
    maxrregcount: int = 255
    producer_registers: int = 40
    quantizer_registers: int = 96
    consumer_registers: int = 232

    # Shared-memory and output policies.
    a_swizzle: Swizzle = "128b"
    b_swizzle: Swizzle = "128b"
    # Native E4M3 SMEM -> RMEM issue widths.  A and B are independent because
    # their fragment shapes and bank-conflict behavior can prefer different
    # ldmatrix transaction sizes.
    a_ldmatrix_matrices: int = 4
    b_ldmatrix_matrices: int = 4
    # E8M0 scale-fragment loads use universal S2R atoms; zero leaves CuTe's
    # auto-vectorizer in control, while 8 requests an explicit scalar byte.
    sfa_s2r_bits: int = 0
    sfb_s2r_bits: int = 0
    epilogue: Epilogue = "direct"
    epilogue_stages: int = 1
    store_vec: int = 1

    # Tile scheduling/reuse.
    persistent: bool = False
    persistent_waves: int = 1
    raster: Raster = "n"
    grid_swizzle: int = 1
    reuse: Reuse = "none"

    @property
    def smem_rmem_tile(self) -> tuple[int, int, int, int, int]:
        """CTA SMEM tile followed by the per-warp RMEM M/N tile.

        The RMEM dimensions are the spatial share owned by one MMA atom.  They
        are derived from the CTA tile and atom replication so the compound
        autotuning coordinate cannot silently disagree with generated code.
        """

        return (
            self.tile_m,
            self.tile_n,
            self.tile_k,
            self.tile_m // self.atom_layout_m,
            self.tile_n // self.atom_layout_n,
        )

    @property
    def bf16_pipeline(self) -> tuple[str, str, int, str, int]:
        """Coupled load/schedule/transport-tile/stage tuning coordinate."""

        return (
            self.load_engine,
            self.schedule,
            self.bf16_tile_k,
            self.bf16_swizzle,
            self.bf16_stages,
        )

    def cache_key(self) -> tuple[object, ...]:
        return tuple(getattr(self, field.name) for field in fields(self))

    def architecture_rejection(self, problem: MXFP8Problem) -> str | None:
        """Reject combinations that can never encode a legal SM120 MXFP8 op."""

        try:
            problem.validate()
        except ValueError as exc:
            return str(exc)
        if (self.tile_m != 64 and self.tile_m % 128) or self.tile_n % 128:
            return "SM120 block-scale tiles require M=64 or M/N divisible by 128"
        if self.tile_k % 128:
            return "MXFP8/E8M0 SM120 scale layouts require tile K divisible by 128"
        if self.atom_layout_m * self.atom_layout_n != self.num_mma_warps:
            return "atom layout product must equal num_mma_warps"
        if self.num_threads > 1024:
            return "CUDA limits one CTA to at most 1024 threads"
        # The SM120 warp-level block-scale fragment mapping has a fixed N
        # quantum.  M=128 supports 2/4/8-way warp layouts, while the 256 tile
        # requires eight M atoms so every SFA fragment has a matching copy
        # partition.  These constraints are verified by compilation and exact
        # device tests below rather than inferred from nominal tile divisibility.
        if self.tile_n != 64 * self.atom_layout_n:
            return "SM120 SFB fragments require tile_n == 64 * atom_layout_n"
        if self.tile_m == 256 and self.atom_layout_m != 8:
            return "the 256-row SFA fragment requires atom_layout_m == 8"
        if self.tile_m == 64 and self.atom_layout_m != 2:
            return "the 64-row SFA fragment requires atom_layout_m == 2"
        if self.quantizer_warps not in (1, 2, 4, 8, 16):
            return "quantizer_warps must be one of 1, 2, 4, 8, 16"
        expected_threads = 32 * (
            self.num_mma_warps
            + self.producer_warps
            + (self.quantizer_warps if self.schedule == "three_role" else 0)
        )
        if self.num_threads != expected_threads:
            return "num_threads does not match the selected pipeline roles"
        if self.bf16_stages not in (1, 2, 3, 4):
            return "bf16_stages must be one of 1, 2, 3, 4"
        if self.bf16_tile_k not in (32, 64, 128, 256):
            return "bf16_tile_k must be one of 32, 64, 128, 256"
        if self.tile_k % self.bf16_tile_k:
            return "MXFP8 tile K must be divisible by the BF16 transport tile K"
        swizzle_bytes = {"none": 0, "32b": 32, "64b": 64, "128b": 128}
        if swizzle_bytes[self.bf16_swizzle] > self.bf16_tile_k * 2:
            return "BF16 SMEM swizzle exceeds the transport tile's contiguous bytes"
        if self.mxfp8_stages not in (1, 2, 3, 4):
            return "mxfp8_stages must be one of 1, 2, 3, 4"
        if self.quant_load_bits not in (16, 32, 64, 128):
            return "quantizer BF16 load width must be 16, 32, 64, or 128 bits"
        if self.quant_load_bits > self.quant_vec * 16:
            return "quantizer BF16 load width exceeds quant_vec"
        if (self.quant_vec * 16) % self.quant_load_bits:
            return "quant_vec must contain an integer number of BF16 vector loads"
        if self.a_ldmatrix_matrices not in (1, 2, 4):
            return "A ldmatrix width must be one of x1, x2, x4"
        if self.b_ldmatrix_matrices not in (1, 2, 4):
            return "B ldmatrix width must be one of x1, x2, x4"
        if self.sfa_s2r_bits not in (0, 8):
            return "SFA S2R width must be auto or 8 bits"
        if self.sfb_s2r_bits not in (0, 8):
            return "SFB S2R width must be auto or 8 bits"
        if self.epilogue_stages not in (1, 2, 3, 4):
            return "epilogue_stages must be one of 1, 2, 3, 4"
        staged_operand_bytes = (
            self.mxfp8_stages
            * (self.tile_m + self.tile_n)
            * self.tile_k
        )
        # SM120 scale blocks are physically padded to 128 rows/columns even
        # when the logical M tile is 64.
        staged_scale_bytes = (
            self.mxfp8_stages
            * (
                ((self.tile_m + 127) // 128) * 128
                + ((self.tile_n + 127) // 128) * 128
            )
            * (self.tile_k // 32)
        )
        bf16_stage_bytes = 0
        if self.load_engine in ("cpasync", "tma"):
            bf16_stage_bytes = (
                self.bf16_stages
                * (self.tile_m + self.tile_n)
                * self.bf16_tile_k
                * 2
            )
        epilogue_bytes = 0
        if self.epilogue != "direct":
            epilogue_bytes = (
                self.epilogue_stages * self.tile_m * self.tile_n * 2
            )
        if (
            staged_operand_bytes
            + staged_scale_bytes
            + bf16_stage_bytes
            + epilogue_bytes
            > SM120_SMEM_CAPACITY_BYTES
        ):
            return "MXFP8 operand stages exceed the SM120 shared-memory capacity"
        if self.grid_swizzle not in (1, 2, 4, 8):
            return "grid_swizzle must be one of 1, 2, 4, 8"
        if self.persistent_waves not in (1, 2, 3, 4):
            return "persistent_waves must be one of 1, 2, 3, 4"
        return None

    def implementation_rejection(self, problem: MXFP8Problem) -> str | None:
        """Return why this revision cannot execute a candidate, or ``None``."""

        reason = self.architecture_rejection(problem)
        if reason is not None:
            return reason
        if self.load_engine == "cpasync" and (
            problem.m % self.tile_m
            or problem.n % self.tile_n
            or problem.k % self.bf16_tile_k
        ):
            return "cp.async staging currently requires full M/N/K tiles"
        if self.quant_load_bits > 16 and (
            problem.m % self.tile_m
            or problem.n % self.tile_n
            or problem.k % self.bf16_tile_k
        ):
            return "vector quantizer loads currently require full M/N/K tiles"
        if (
            self.quant_load_bits > 16
            and self.load_engine != "scalar"
            and self.bf16_swizzle != "none"
        ):
            return "vector quantizer loads currently require unswizzled BF16 SMEM"
        if self.schedule == "pingpong":
            return "ping-pong warp specialization is not implemented"
        if self.schedule == "warp_specialized" and self.load_engine != "tma":
            return "warp-specialized forward requires the TMA load engine"
        if self.schedule == "three_role" and self.load_engine != "tma":
            return "three-role forward requires the TMA load engine"
        if self.load_engine == "cpasync" and self.schedule != "cooperative":
            return "cp.async staging currently uses the cooperative schedule"
        if self.schedule == "cooperative" and self.producer_warps != 0:
            return "cooperative schedule cannot reserve producer warps"
        if self.schedule in ("warp_specialized", "three_role") and self.producer_warps != 1:
            return "specialized schedules require one TMA producer warp"
        if self.schedule != "three_role" and self.quantizer_warps != self.num_mma_warps:
            return "non-three-role schedules quantize with every MMA warp"
        if self.load_engine == "scalar" and self.bf16_stages != 1:
            return "BF16 stages only apply to staged load engines"
        if self.load_engine == "scalar" and self.bf16_tile_k != self.tile_k:
            return "BF16 transport tiles only apply to staged load engines"
        if self.load_engine == "scalar" and self.bf16_swizzle != "128b":
            return "BF16 SMEM swizzle only applies to staged load engines"
        if self.epilogue == "smem":
            return "synchronous SMEM epilogue is not implemented"
        if self.epilogue == "direct" and (
            self.epilogue_stages != 1 or self.store_vec != 1
        ):
            return "epilogue stages/store width only apply to the TMA epilogue"
        if self.epilogue == "tma" and self.store_vec not in (1, 2, 4):
            return "TMA epilogue stmatrix width must be x1, x2, or x4"
        if self.epilogue == "tma" and (
            problem.m % self.tile_m or problem.n % self.tile_n
        ):
            return "TMA epilogue currently requires full M/N output tiles"
        if self.persistent and (
            self.load_engine != "scalar"
            or self.schedule != "cooperative"
            or self.epilogue != "direct"
        ):
            return "persistent revision currently supports scalar/cooperative/direct"
        if self.reuse != "none" and not self.persistent:
            return "operand-locality scheduling requires a persistent CTA"
        if not self.persistent and self.persistent_waves != 1:
            return "persistent_waves only applies to a persistent CTA"
        baseline = MXFP8FwdConfig()
        if self.schedule == "cooperative" and (
            self.producer_registers != baseline.producer_registers
            or self.quantizer_registers != baseline.quantizer_registers
            or self.consumer_registers != baseline.consumer_registers
        ):
            return "role-specific register limits require a specialized schedule"
        if (
            self.schedule == "warp_specialized"
            and self.quantizer_registers != baseline.quantizer_registers
        ):
            return "quantizer register limit only applies to the three-role schedule"
        # These fields all select distinct generated code in the cooperative
        # kernel.  Dependent launch fields are included because changing the
        # atom layout recomputes them in ``normalize_fwd_config``.
        implemented = {
            "tile_m",
            "tile_n",
            "tile_k",
            "atom_layout_m",
            "atom_layout_n",
            "num_mma_warps",
            "quantizer_warps",
            "reduction",
            "quant_math",
            "quant_amax",
            "quant_load_bits",
            "load_engine",
            "schedule",
            "bf16_tile_k",
            "bf16_swizzle",
            "bf16_stages",
            "producer_warps",
            "quant_vec",
            "mxfp8_stages",
            "k_unroll",
            "num_threads",
            "maxrregcount",
            "producer_registers",
            "quantizer_registers",
            "consumer_registers",
            "a_swizzle",
            "b_swizzle",
            "a_ldmatrix_matrices",
            "b_ldmatrix_matrices",
            "sfa_s2r_bits",
            "sfb_s2r_bits",
            "epilogue",
            "epilogue_stages",
            "store_vec",
            "persistent",
            "persistent_waves",
            "raster",
            "grid_swizzle",
            "reuse",
        }
        unsupported = [
            field.name
            for field in fields(self)
            if field.name not in implemented
            and getattr(self, field.name) != getattr(baseline, field.name)
        ]
        if unsupported:
            return "not implemented by cooperative kernel: " + ", ".join(unsupported)
        return None


DEFAULT_MXFP8_FWD_CONFIG = MXFP8FwdConfig()
MXFP8_FWD_KERNEL_REVISION = 13


# Block coordinates supplement, rather than replace, their primitive fields.
# They let coordinate descent cross a slower intermediate configuration and
# directly enter small-SMEM/deep-pipeline or small-RMEM/many-warp basins.
SMEM_RMEM_TILE_COORDINATES: tuple[tuple[int, int, int, int, int], ...] = tuple(
    (smem_m, smem_n, smem_k, rmem_m, 64)
    for smem_m, smem_n, smem_k in product(
        (64, 128, 256), (128, 256), (128, 256)
    )
    for rmem_m in (16, 32, 64)
    if smem_m % rmem_m == 0
)

BF16_PIPELINE_COORDINATES: tuple[tuple[str, str, int, str, int], ...] = (
    # Scalar anchors retain a route back from staged basins.
    ("scalar", "cooperative", 128, "128b", 1),
    ("scalar", "cooperative", 256, "128b", 1),
    # Complete staged neighborhoods.  Legality still accounts for the current
    # MXFP8 macro tile and other SMEM allocations before compilation.
    *tuple(
        (engine, schedule, tile_k, swizzle, stages)
        for engine, schedule in (
            ("cpasync", "cooperative"),
            ("tma", "cooperative"),
            ("tma", "warp_specialized"),
            ("tma", "three_role"),
        )
        for tile_k, swizzles, stage_values in (
            (32, ("none", "32b", "64b"), (1, 2, 3, 4)),
            (64, ("none", "32b", "64b", "128b"), (1, 2)),
            (128, ("none", "32b", "64b", "128b"), (1,)),
        )
        for swizzle in swizzles
        for stages in stage_values
    ),
)


# These axes define the intended search space.  ``iter_fwd_candidates`` is lazy:
# the cross product is deliberately large enough for long, shape-local tuning.
FWD_SEARCH_SPACE: dict[str, tuple[object, ...]] = {
    "smem_rmem_tile": SMEM_RMEM_TILE_COORDINATES,
    "bf16_pipeline": BF16_PIPELINE_COORDINATES,
    "tile_m": (64, 128, 256),
    "tile_n": (128, 256),
    "tile_k": (128, 256),
    "atom_layout_m": (2, 4, 8),
    "atom_layout_n": (1, 2, 4),
    "load_engine": ("scalar", "cpasync", "tma"),
    "schedule": ("cooperative", "warp_specialized", "three_role", "pingpong"),
    "bf16_tile_k": (32, 64, 128, 256),
    "bf16_swizzle": ("none", "32b", "64b", "128b"),
    "bf16_stages": (1, 2, 3, 4),
    "mxfp8_stages": (1, 2, 3, 4),
    "quantizer_warps": (1, 2, 4, 8, 16),
    "reduction": ("redux", "shuffle"),
    "quant_math": ("fp32", "bf16x2"),
    "quant_amax": ("fp32", "bf16_bits"),
    "quant_load_bits": (16, 32, 64, 128),
    "quant_vec": (1, 2, 4, 8),
    "k_unroll": (1, 2, 4),
    "maxrregcount": (128, 160, 192, 224, 255),
    "producer_registers": (32, 40, 48, 56),
    "quantizer_registers": (64, 80, 96, 112, 128),
    "consumer_registers": (160, 192, 224, 232),
    "a_swizzle": ("none", "32b", "64b", "128b"),
    "b_swizzle": ("none", "32b", "64b", "128b"),
    "a_ldmatrix_matrices": (1, 2, 4),
    "b_ldmatrix_matrices": (1, 2, 4),
    "sfa_s2r_bits": (0, 8),
    "sfb_s2r_bits": (0, 8),
    "epilogue": ("direct", "smem", "tma"),
    "epilogue_stages": (1, 2, 3, 4),
    "store_vec": (1, 2, 4, 8),
    "persistent": (False, True),
    "persistent_waves": (1, 2, 3, 4),
    "raster": ("m", "n"),
    "grid_swizzle": (1, 2, 4, 8),
    "reuse": ("none", "x", "weight"),
}

# Start with coordinates that usually dominate the schedule, then refine data
# movement and resource limits.  The order is persisted in each tuning record.
FWD_COORDINATE_ORDER: tuple[str, ...] = (
    # Block moves come first so the search can enter coupled basins without
    # requiring every slower one-field intermediate to win.
    "smem_rmem_tile",
    "bf16_pipeline",
    # Probe the data path before growing the CTA.  A 128x128 TMA tile fits in
    # SMEM, while a 256-row/column scalar winner can make every neighboring
    # TMA candidate illegal and strand ordinary coordinate descent.
    "load_engine",
    "quant_vec",
    "quant_math",
    "quant_amax",
    "quant_load_bits",
    "tile_m",
    "tile_n",
    "tile_k",
    "schedule",
    "atom_layout_m",
    "atom_layout_n",
    "bf16_tile_k",
    "bf16_swizzle",
    "bf16_stages",
    "mxfp8_stages",
    "quantizer_warps",
    "reduction",
    "k_unroll",
    "a_swizzle",
    "b_swizzle",
    "a_ldmatrix_matrices",
    "b_ldmatrix_matrices",
    "sfa_s2r_bits",
    "sfb_s2r_bits",
    "epilogue",
    "epilogue_stages",
    "store_vec",
    "maxrregcount",
    "producer_registers",
    "quantizer_registers",
    "consumer_registers",
    "persistent",
    "persistent_waves",
    "raster",
    "grid_swizzle",
    "reuse",
)


def normalize_fwd_config(
    base: MXFP8FwdConfig | None = None,
    /,
    **updates: object,
) -> MXFP8FwdConfig:
    """Apply independent coordinates and recompute dependent launch fields."""

    values = asdict(base or DEFAULT_MXFP8_FWD_CONFIG)
    updates = dict(updates)
    compound_names = {"smem_rmem_tile", "bf16_pipeline"}
    unknown = set(updates).difference(values).difference(compound_names)
    if unknown:
        raise ValueError(f"unknown MXFP8 forward coordinates: {sorted(unknown)}")

    smem_rmem_tile = updates.pop("smem_rmem_tile", None)
    if smem_rmem_tile is not None:
        smem_m, smem_n, smem_k, rmem_m, rmem_n = smem_rmem_tile
        if smem_m % rmem_m or smem_n % rmem_n:
            raise ValueError(
                "SMEM M/N tile must be divisible by its RMEM M/N tile"
            )
        updates.update(
            tile_m=smem_m,
            tile_n=smem_n,
            tile_k=smem_k,
            atom_layout_m=smem_m // rmem_m,
            atom_layout_n=smem_n // rmem_n,
        )

    bf16_pipeline = updates.pop("bf16_pipeline", None)
    if bf16_pipeline is not None:
        load_engine, schedule, bf16_tile_k, bf16_swizzle, bf16_stages = (
            bf16_pipeline
        )
        updates.update(
            load_engine=load_engine,
            schedule=schedule,
            bf16_tile_k=bf16_tile_k,
            bf16_swizzle=bf16_swizzle,
            bf16_stages=bf16_stages,
        )

    values.update(updates)
    if "bf16_tile_k" in updates and "bf16_swizzle" not in updates:
        tile_bytes = int(values["bf16_tile_k"]) * 2
        values["bf16_swizzle"] = (
            "128b" if tile_bytes >= 128 else "64b" if tile_bytes >= 64 else "32b"
        )
    if (
        ("tile_k" in updates or updates.get("load_engine") == "scalar")
        and "bf16_tile_k" not in updates
        and values["load_engine"] == "scalar"
    ):
        values["bf16_tile_k"] = values["tile_k"]
        values["bf16_swizzle"] = "128b"
    # Tile coordinates are compound schedule moves: the scale-fragment layout
    # requires a matching warp arrangement.  Explicit atom-layout updates still
    # win, which lets callers test/reject arbitrary combinations deliberately.
    if "tile_m" in updates and "atom_layout_m" not in updates:
        values["atom_layout_m"] = {
            64: 2,
            128: 4,
            256: 8,
        }[int(values["tile_m"])]
    if "tile_n" in updates and "atom_layout_n" not in updates:
        values["atom_layout_n"] = int(values["tile_n"]) // 64
    mma_warps = int(values["atom_layout_m"]) * int(values["atom_layout_n"])
    schedule = values["schedule"]
    producer_warps = 0 if schedule == "cooperative" else 1
    if schedule != "three_role":
        values["quantizer_warps"] = mma_warps
    elif "quantizer_warps" not in updates:
        values["quantizer_warps"] = min(4, mma_warps)
    role_quantizer_warps = (
        int(values["quantizer_warps"]) if schedule == "three_role" else 0
    )
    values.update(
        num_mma_warps=mma_warps,
        producer_warps=producer_warps,
        quantizer_warps=int(values["quantizer_warps"]),
        num_threads=32 * (mma_warps + producer_warps + role_quantizer_warps),
    )
    return MXFP8FwdConfig(**values)


def fwd_config_to_dict(config: MXFP8FwdConfig) -> dict[str, object]:
    return asdict(config)


def fwd_config_from_dict(values: dict[str, object]) -> MXFP8FwdConfig:
    return normalize_fwd_config(**values)


def fwd_config_id(config: MXFP8FwdConfig) -> str:
    payload = json.dumps(
        fwd_config_to_dict(config), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:20]


def fwd_search_space_digest(
    axes: dict[str, Iterable[object]] | None = None,
) -> str:
    search = FWD_SEARCH_SPACE if axes is None else axes
    payload = {
        name: list(values)
        for name, values in search.items()
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def iter_fwd_candidates(
    axes: dict[str, Iterable[object]] | None = None,
) -> Iterator[MXFP8FwdConfig]:
    """Lazily generate schedule candidates, deriving dependent warp fields."""

    search = FWD_SEARCH_SPACE if axes is None else axes
    names = tuple(search)
    for values in product(*(search[name] for name in names)):
        kwargs = dict(zip(names, values, strict=True))
        yield normalize_fwd_config(**kwargs)
