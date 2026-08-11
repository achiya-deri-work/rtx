"""Composable autotuning framework plus compatibility with the original tuner."""

from . import legacy as _legacy
from .adapters import (
    make_mxfp8_bwd_adapter,
    make_mxfp8_fully_prequant_adapter,
    make_mxfp8_fwd_adapter,
    make_mxfp8_prequant_adapter,
    make_mxfp8_weight_prequant_adapter,
    make_nvfp4_fully_prequant_adapter,
    make_nvfp4_dynamic_adapter,
    make_nvfp4_fwd_adapter,
    make_nvfp4_weight_prequant_adapter,
)
from .bandit import (
    AdaptiveBanditScheduler,
    ArmStatistics,
    DiscountedArmStatistics,
    UCB1Scheduler,
    contextual_ucb_scores,
)
from .ask_tell import (
    AskTellSession,
    DurableLocalAskTellRunner,
    LocalTrialWorker,
    TrialRequest,
    TrialResponse,
)
from .audit import audit_bundles
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
from .outcomes import (
    FatalDeviceContextError,
    TrialOutcome,
    TrialStatus,
    is_fatal_device_context_error,
)
from .orchestrator import (
    AutotuneOrchestrator,
    ConfirmationPolicy,
    SequentialScheduler,
)
from .supervisor import STALL_EXIT_CODE, WATCHDOG_CHILD_ENV, supervise_command
from .migration import import_legacy_json_database
from .recipes import (
    HybridTuningPolicy,
    make_hybrid_ask_tell_runner,
    make_hybrid_autotuner,
)
from .pretrained import (
    ConditionalEffectRule,
    ConditionalRuleSet,
    ContextRankingModel,
    NormalizedCostModel,
    evaluate_pretrained_bundle,
    load_offline_observations,
    load_pretrained_family,
    train_pretrained_bundle,
)
from .promotion import install_verified_winners
from .store import (
    InMemoryTuningStore,
    JsonlTuningStore,
    ResidualTuningStore,
    TuningStore,
)
from .space import (
    Condition,
    ConditionalSearchSpace,
    DiscreteParameter,
    SearchSpace,
    SpaceConstraint,
)
from .safety import JsonlFailureLedger, SafetyAwareAdapter, failure_scope
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
        "DatasetTreatment",
        "register_dataset_backend",
    }:
        from . import dataset

        return getattr(dataset, name)
    if name in {"optimizer_study_rows", "summarize_optimizer_study"}:
        from . import optimizer_benchmark

        return getattr(optimizer_benchmark, name)
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
    "DatasetTreatment",
    "DiscreteKernelAdapter",
    "DiscreteParameter",
    "DurableLocalAskTellRunner",
    "EvaluationPlan",
    "EvaluationStage",
    "FunctionKernelTask",
    "FatalDeviceContextError",
    "GradientBoostedCostModel",
    "GradientBoostedFeasibilityModel",
    "HybridTuningPolicy",
    "InMemoryTuningStore",
    "JsonTuningDatabase",
    "JsonlTuningStore",
    "JsonlFailureLedger",
    "KernelAdapter",
    "KernelContext",
    "LocalTrialWorker",
    "MXFP8ForwardEvaluator",
    "NormalizedCostModel",
    "Observation",
    "Proposal",
    "PortableKernelTask",
    "RandomSearch",
    "ResidualTuningStore",
    "RuntimeWinnerKey",
    "SearchHistory",
    "SafetyAwareAdapter",
    "SearchSpace",
    "SearchStrategy",
    "SequentialScheduler",
    "SparseFeatureVectorizer",
    "SpaceConstraint",
    "StageKind",
    "StageResult",
    "STALL_EXIT_CODE",
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
    "WATCHDOG_CHILD_ENV",
    "default_cache_dir",
    "architecture_profile",
    "audit_bundles",
    "calibrate_device",
    "compiled_resource_metadata",
    "contextual_ucb_scores",
    "compiler_profile",
    "device_properties",
    "export_bundle",
    "failure_scope",
    "evaluate_pretrained_bundle",
    "load_cached_mxfp8_fwd_config",
    "load_offline_observations",
    "load_pretrained_family",
    "load_runtime_winner",
    "import_legacy_json_database",
    "install_verified_winners",
    "is_fatal_device_context_error",
    "make_mxfp8_bwd_adapter",
    "make_mxfp8_fully_prequant_adapter",
    "make_mxfp8_fwd_adapter",
    "make_nvfp4_fwd_adapter",
    "make_nvfp4_dynamic_adapter",
    "make_nvfp4_fully_prequant_adapter",
    "make_nvfp4_weight_prequant_adapter",
    "make_mxfp8_prequant_adapter",
    "make_mxfp8_weight_prequant_adapter",
    "make_hybrid_autotuner",
    "make_hybrid_ask_tell_runner",
    "normalized_rows",
    "optimizer_study_rows",
    "register_dataset_backend",
    "runtime_winner_key",
    "save_runtime_winner",
    "static_device_profile",
    "summarize_optimizer_study",
    "supervise_command",
    "tune_mxfp8_fwd",
    "train_pretrained_bundle",
]
