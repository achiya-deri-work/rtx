"""Composable autotuning framework plus compatibility with the original tuner."""

from . import legacy as _legacy
from .adapters import (
    make_mxfp8_bwd_adapter,
    make_mxfp8_fully_prequant_adapter,
    make_mxfp8_fwd_adapter,
    make_mxfp8_prequant_adapter,
    make_mxfp8_weight_prequant_adapter,
)
from .calibration import calibrate_device
from .core import (
    ComposableTuningResult,
    DiscreteKernelAdapter,
    KernelAdapter,
    KernelContext,
    Observation,
    Proposal,
    SearchHistory,
    TuningBudget,
)
from .cost_model import (
    GradientBoostedCostModel,
    GradientBoostedFeasibilityModel,
    SparseFeatureVectorizer,
)
from .hardware import (
    ArchitectureProfile,
    architecture_profile,
    compiled_resource_metadata,
    compiler_profile,
    device_properties,
    static_device_profile,
)
from .evaluators import CalibratedBwdEvaluator, CalibratedPrequantEvaluator
from .legacy import (
    CoordinateDescentPolicy,
    CoordinateDescentTuner,
    DeviceFingerprint,
    JsonTuningDatabase,
    MXFP8ForwardEvaluator,
    TrialOutcome,
    TuningResult,
    default_cache_dir,
    load_cached_mxfp8_fwd_config,
    tune_mxfp8_fwd,
)
from .orchestrator import (
    ArmStatistics,
    AutotuneOrchestrator,
    ConfirmationPolicy,
    SequentialScheduler,
    UCB1Scheduler,
)
from .migration import import_legacy_json_database
from .recipes import HybridTuningPolicy, make_hybrid_autotuner
from .store import InMemoryTuningStore, JsonlTuningStore, TuningStore
from .winners import (
    RuntimeWinnerKey,
    load_runtime_winner,
    runtime_winner_key,
    save_runtime_winner,
)
from .strategies import (
    CoordinateLocalSearch,
    CostModelGuidedSearch,
    CostModelLocalSearch,
    RandomSearch,
    SearchStrategy,
    StrategyPipeline,
)


def __getattr__(name: str):
    if name in {
        "AnytimeRunPolicy",
        "DatasetCampaign",
        "DatasetBackend",
        "DatasetJob",
        "DatasetManifest",
        "export_bundle",
        "normalized_rows",
        "register_dataset_backend",
    }:
        from . import dataset

        return getattr(dataset, name)
    # Some existing internal users import private forward-reference helpers.
    if hasattr(_legacy, name):
        return getattr(_legacy, name)
    raise AttributeError(name)


__all__ = [
    "ArmStatistics",
    "ArchitectureProfile",
    "AnytimeRunPolicy",
    "AutotuneOrchestrator",
    "CalibratedBwdEvaluator",
    "CalibratedPrequantEvaluator",
    "ComposableTuningResult",
    "ConfirmationPolicy",
    "CoordinateDescentPolicy",
    "CoordinateDescentTuner",
    "CoordinateLocalSearch",
    "CostModelGuidedSearch",
    "CostModelLocalSearch",
    "DeviceFingerprint",
    "DatasetCampaign",
    "DatasetBackend",
    "DatasetJob",
    "DatasetManifest",
    "DiscreteKernelAdapter",
    "GradientBoostedCostModel",
    "GradientBoostedFeasibilityModel",
    "HybridTuningPolicy",
    "InMemoryTuningStore",
    "JsonTuningDatabase",
    "JsonlTuningStore",
    "KernelAdapter",
    "KernelContext",
    "MXFP8ForwardEvaluator",
    "Observation",
    "Proposal",
    "RandomSearch",
    "RuntimeWinnerKey",
    "SearchHistory",
    "SearchStrategy",
    "SequentialScheduler",
    "SparseFeatureVectorizer",
    "StrategyPipeline",
    "TrialOutcome",
    "TuningBudget",
    "TuningResult",
    "TuningStore",
    "UCB1Scheduler",
    "default_cache_dir",
    "architecture_profile",
    "calibrate_device",
    "compiled_resource_metadata",
    "compiler_profile",
    "device_properties",
    "export_bundle",
    "load_cached_mxfp8_fwd_config",
    "load_runtime_winner",
    "import_legacy_json_database",
    "make_mxfp8_bwd_adapter",
    "make_mxfp8_fully_prequant_adapter",
    "make_mxfp8_fwd_adapter",
    "make_mxfp8_prequant_adapter",
    "make_mxfp8_weight_prequant_adapter",
    "make_hybrid_autotuner",
    "normalized_rows",
    "register_dataset_backend",
    "runtime_winner_key",
    "save_runtime_winner",
    "static_device_profile",
    "tune_mxfp8_fwd",
]
