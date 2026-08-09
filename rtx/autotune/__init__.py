"""Composable autotuning framework plus compatibility with the original tuner."""

from . import legacy as _legacy
from .adapters import (
    make_mxfp8_bwd_adapter,
    make_mxfp8_fully_prequant_adapter,
    make_mxfp8_fwd_adapter,
    make_mxfp8_prequant_adapter,
    make_mxfp8_weight_prequant_adapter,
)
from .bandit import (
    AdaptiveBanditScheduler,
    ArmStatistics,
    DiscountedArmStatistics,
    UCB1Scheduler,
    contextual_ucb_scores,
)
from .ask_tell import AskTellSession, LocalTrialWorker, TrialRequest, TrialResponse
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
    TuningResult,
    default_cache_dir,
    load_cached_mxfp8_fwd_config,
    tune_mxfp8_fwd,
)
from .outcomes import TrialOutcome, TrialStatus
from .orchestrator import (
    AutotuneOrchestrator,
    ConfirmationPolicy,
    SequentialScheduler,
)
from .migration import import_legacy_json_database
from .recipes import HybridTuningPolicy, make_hybrid_autotuner
from .pretrained import (
    ConditionalEffectRule,
    ConditionalRuleSet,
    ContextRankingModel,
    NormalizedCostModel,
    load_offline_observations,
    load_pretrained_family,
    train_pretrained_bundle,
)
from .store import InMemoryTuningStore, JsonlTuningStore, TuningStore
from .space import (
    Condition,
    ConditionalSearchSpace,
    DiscreteParameter,
    SearchSpace,
    SpaceConstraint,
)
from .task import (
    AdapterKernelTask,
    AdapterSearchSpace,
    EvaluationPlan,
    EvaluationStage,
    FunctionKernelTask,
    PortableKernelTask,
    StageKind,
    StageResult,
    StagedTaskAdapter,
)
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
        "export_bundle",
        "normalized_rows",
    }:
        from . import dataset_export

        return getattr(dataset_export, name)
    if name in {
        "AnytimeRunPolicy",
        "DatasetCampaign",
        "DatasetBackend",
        "DatasetJob",
        "DatasetManifest",
        "register_dataset_backend",
    }:
        from . import dataset

        return getattr(dataset, name)
    # Some existing internal users import private forward-reference helpers.
    if hasattr(_legacy, name):
        return getattr(_legacy, name)
    raise AttributeError(name)


__all__ = [
    "AdaptiveBanditScheduler",
    "AdapterKernelTask",
    "AdapterSearchSpace",
    "ArmStatistics",
    "ArchitectureProfile",
    "AskTellSession",
    "AnytimeRunPolicy",
    "AutotuneOrchestrator",
    "CalibratedBwdEvaluator",
    "CalibratedPrequantEvaluator",
    "ComposableTuningResult",
    "ConfirmationPolicy",
    "ConditionalEffectRule",
    "ConditionalRuleSet",
    "ConditionalSearchSpace",
    "Condition",
    "ContextRankingModel",
    "CoordinateDescentPolicy",
    "CoordinateDescentTuner",
    "CoordinateLocalSearch",
    "CostModelGuidedSearch",
    "CostModelLocalSearch",
    "DeviceFingerprint",
    "DiscountedArmStatistics",
    "DatasetCampaign",
    "DatasetBackend",
    "DatasetJob",
    "DatasetManifest",
    "DiscreteKernelAdapter",
    "DiscreteParameter",
    "EvaluationPlan",
    "EvaluationStage",
    "FunctionKernelTask",
    "GradientBoostedCostModel",
    "GradientBoostedFeasibilityModel",
    "HybridTuningPolicy",
    "InMemoryTuningStore",
    "JsonTuningDatabase",
    "JsonlTuningStore",
    "KernelAdapter",
    "KernelContext",
    "LocalTrialWorker",
    "MXFP8ForwardEvaluator",
    "NormalizedCostModel",
    "Observation",
    "Proposal",
    "PortableKernelTask",
    "RandomSearch",
    "RuntimeWinnerKey",
    "SearchHistory",
    "SearchSpace",
    "SearchStrategy",
    "SequentialScheduler",
    "SparseFeatureVectorizer",
    "SpaceConstraint",
    "StageKind",
    "StageResult",
    "StagedTaskAdapter",
    "StrategyPipeline",
    "TrialOutcome",
    "TrialStatus",
    "TrialRequest",
    "TrialResponse",
    "TuningBudget",
    "TuningResult",
    "TuningStore",
    "UCB1Scheduler",
    "default_cache_dir",
    "architecture_profile",
    "calibrate_device",
    "compiled_resource_metadata",
    "contextual_ucb_scores",
    "compiler_profile",
    "device_properties",
    "export_bundle",
    "load_cached_mxfp8_fwd_config",
    "load_offline_observations",
    "load_pretrained_family",
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
    "train_pretrained_bundle",
]
