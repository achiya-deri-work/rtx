"""Retrospective shape-held-out preparation prior study, never auto-promoted."""
import argparse
import json
import math
import statistics
from pathlib import Path
from types import SimpleNamespace

from rtx.autotune.cost_model import GradientBoostedCostModel


def portable_features(row, feature_set="work"):
    """Explicit pre-execution features; no identities or measured outcomes."""
    source = row["features"]
    sm = max(1., source.get("context.device.multiprocessor_count", 1.))
    features = {}
    for leg in ("dx", "dw"):
        prefix = "derived." + leg + "_"
        for key in ("preparation_prepared_elements", "nominal_flops",
                    "estimated_operand_read_bytes"):
            features[leg + "_" + key + "_per_sm"] = math.log1p(source.get(prefix + key, 0.) / sm)
        for key in ("preparation_stage_a", "preparation_stage_b",
                    "preparation_workspace_l2_ratio", "preparation_cta_waves"):
            features[leg + "_" + key] = source.get(prefix + key, 0.)
    features["dual_stream"] = source.get("config.config.stream_schedule=dual_stream", 0.)
    features["quad_quant"] = source.get("config.config.quant_schedule=quad", 0.)
    features["shared_g_quad"] = source.get("config.config.quant_schedule=shared_g_quad", 0.)
    if feature_set == "schedule":
        allowed = {"tile_m","tile_n","tile_k","stages","bf16_stages","mxfp8_stages",
            "bf16_tile_k","num_warps","num_threads","num_mma_warps","quantizer_warps",
            "producer_warps","maxrregcount","producer_registers","consumer_registers",
            "quantizer_registers","atom_layout_m","atom_layout_n","load_engine","schedule",
            "quant_vec","load_bits","quant_store_bits","quant_load_bits","quant_schedule",
            "quant_launches","persistent","persistent_waves","grid_swizzle","raster",
            "tile_locality","tiles_per_cta","a_swizzle","b_swizzle","bf16_swizzle",
            "a_ldmatrix_matrices","b_ldmatrix_matrices","sfa_s2r_bits","sfb_s2r_bits",
            "scale_schedule","scale_load_vec","scale_role","scale_layout","scale_cache",
            "scale_smem_store","scale_recycle","epilogue","epilogue_stages","store_vec",
            "transposed_load_engine","transposed_tile_rows","transposed_tile_k",
            "transposed_smem_padding","native_scale_store","mma_schedule","k_unroll",
            "execution_order","stream_schedule","backend","reduction_tile","reduction_threads"}
        for key,value in source.items():
            if not key.startswith("config.config."):
                continue
            path = key.split("=",1)[0]
            if path.rsplit(".",1)[-1] not in allowed:
                continue
            inactive = False
            for leg in ("dx","dw"):
                staged = source.get(f"config.config.{leg}.backend=decomposed",0.) == 1.
                if key.startswith(f"config.config.{leg}.fused.") and staged:
                    inactive = True
                if not staged and any(key.startswith(f"config.config.{leg}.{part}.")
                                      for part in ("gemm","quant_a","quant_b")):
                    inactive = True
            if not inactive:
                features[key] = value
    elif feature_set != "work":
        raise ValueError("unknown feature set")
    return features


def training_scores(rows, target):
    """Normalize within training contexts only; held-out latency never enters fit."""
    if target not in ("absolute", "relative"):
        raise ValueError("unknown cost-model target")
    best = {}
    for row in rows:
        key = row["context_id"]
        best[key] = min(best.get(key, math.inf), row["outcome"]["median_ms"])
    return [r["outcome"]["median_ms"] / (best[r["context_id"]] if target == "relative" else 1.)
            for r in rows]


def consolidate_schedules(rows):
    """One median observation per active schedule per exact context.

    Do not select the fastest repeat or collapse devices, shapes, revisions,
    or cache regimes. Source observations remain untouched on disk.
    """
    from rtx.autotune.core import stable_id
    from rtx.autotune.preparation import PreparationCandidate
    from rtx.autotune.preparation_mxfp8 import execution_signature
    from rtx.bwd_autotune import bwd_config_from_dict
    groups = {}
    for row in rows:
        candidate = PreparationCandidate(row["config"]["implementation"],
            bwd_config_from_dict(row["config"]["config"]))
        key = (row["context_id"], stable_id(execution_signature(candidate)))
        groups.setdefault(key, []).append(row)
    result = []
    for (context_id, signature), group in groups.items():
        first = min(group, key=lambda row: row["sequence"])
        result.append({**first, "config_id": signature,
            "outcome": {**first["outcome"], "median_ms": statistics.median(
                row["outcome"]["median_ms"] for row in group)},
            "repeat_count": len(group)})
    return result


def analyze(rows, target="absolute", feature_set="work"):
    usable = [r for r in rows if r.get("outcome",{}).get("status") == "ok"
              and "derived.combined_nominal_flops" in r.get("features",{})]
    unstable_rows = sum(r["outcome"].get("metadata",{}).get("sampling",{}).get("collection",{}).get("stationary") is False for r in usable)
    usable = [r for r in usable if r["outcome"].get("metadata",{}).get("sampling",{}).get("collection",{}).get("stationary") is not False]
    # Keep full paired-gradient observations; partial gradients are distinct tasks.
    usable = [r for r in usable if not any(".tags.phase=" in key and not key.endswith("=backward")
              for key in r["features"])]
    for row in usable:
        latency = row["outcome"]["median_ms"]
        if not math.isfinite(latency) or latency <= 0:
            raise ValueError("successful observations must have finite positive latency")
    signatures = {tuple(sorted((key,value) for key,value in row["features"].items()
                  if key.startswith("context.tags.numerical_policy.")
                  or key.startswith("context.regime=")
                  or key.startswith("context.environment.")
                  or any(key.startswith("context.device."+field+"=") for field in
                         ("torch_version","torch_cuda_version","cutlass_dsl_version"))
                  or key.startswith("context.tags.components.") and
                  (key.endswith(".kernel_revision") or ".tags.measurement." in key)))
                  for row in usable}
    if len(signatures) != 1:
        raise ValueError("do not pool incompatible numerical/kernel/measurement contracts")
    raw_rows = len(usable)
    usable = consolidate_schedules(usable)
    def shape(r):
        return tuple(int(r["features"]["context.workload." + axis]) for axis in ("m","n","k"))
    shapes = sorted({shape(r) for r in usable})
    if len(shapes) < 2:
        raise ValueError("at least two paired-backward shapes are required")
    folds = []
    for held, context_id in sorted({(shape(r),r["context_id"]) for r in usable}):
        train = [r for r in usable if shape(r) != held]
        test = [r for r in usable if r["context_id"] == context_id]
        model = GradientBoostedCostModel(n_estimators=32, max_depth=2, min_leaf=2,
                                         ensembles=4, seed=20261023)
        scores = training_scores(train, target)
        model.fit([SimpleNamespace(successful=True, score=score,
                                   features=portable_features(r,feature_set)) for r,score in zip(train,scores)])
        if not model.fitted:
            raise ValueError("insufficient training observations in held-out fold")
        predicted,_ = model.predict([portable_features(r,feature_set) for r in test])
        ranked = sorted(range(len(test)), key=lambda i: predicted[i])
        actual = [r["outcome"]["median_ms"] for r in test]
        optimum = min(actual)
        baseline = min(test, key=lambda r: r["sequence"])["outcome"]["median_ms"]
        heuristic = sorted(range(len(test)), key=lambda i: sum(
            test[i]["config"]["config"][leg]["backend"] == "fused" for leg in ("dx","dw")))
        from rtx.bwd_autotune import bwd_config_from_dict
        from rtx.autotune.preparation import PreparationCandidate
        from rtx.autotune.preparation_mxfp8 import reuse_prior
        def reuse_score(index):
            row = test[index]
            candidate = PreparationCandidate(row["config"]["implementation"],
                bwd_config_from_dict(row["config"]["config"]))
            return reuse_prior(candidate, *held, phase="backward",
                sm_count=int(row["features"].get("context.device.multiprocessor_count",0)),
                l2_bytes=int(row["features"].get("context.device.l2_cache_size",0)))
        reuse_order = sorted(range(len(test)), key=reuse_score)
        ordered_actual = sorted(actual)
        def expected_random_regret(budget):
            count = min(budget,len(actual))
            denominator = math.comb(len(actual),count)
            expected = sum(value * math.comb(len(actual)-index-1,count-1) / denominator
                           for index,value in enumerate(ordered_actual[:len(actual)-count+1]))
            return expected/optimum-1
        folds.append(dict(shape=held, context_id=context_id, train_rows=len(train), test_rows=len(test),
            observed_best_ms=optimum, initial_ms=baseline,
            initial_regret=baseline/optimum-1,
            ranked_regret={str(k): min(actual[i] for i in ranked[:k])/optimum-1
                           for k in (1,2,4,8)},
            staged_first_regret={str(k): min(actual[i] for i in heuristic[:k])/optimum-1
                                 for k in (1,2,4,8)},
            reuse_prior_regret={str(k): min(actual[i] for i in reuse_order[:k])/optimum-1
                               for k in (1,2,4,8)},
            random_expected_regret={str(k): expected_random_regret(k) for k in (1,2,4,8)},
            predicted_first_config=test[ranked[0]]["config_id"]))
    return dict(schema_version=2, deployment_enabled=False, target=target, feature_set=feature_set,
        raw_rows=raw_rows, rows=len(usable), excluded_unstable_rows=unstable_rows,
        collapsed_repeats=raw_rows-len(usable), folds=folds,
        caveats=["Retrospective ranking among observed candidates, not prospective tuning.",
                 "Only RTX5090 evidence; per-SM normalization is an assumption, not SKU transfer validation.",
                 "Kernel phase, arithmetic policy, and measurement protocol must remain matched.",
                 "Small uneven dataset; no hyperparameter selection or production promotion."])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root")
    p.add_argument("--output",required=True)
    p.add_argument("--target",choices=("absolute","relative"),default="absolute")
    p.add_argument("--features",choices=("work","schedule"),default="work")
    args = p.parse_args()
    paths = sorted(Path(args.root).glob("*/residuals/observations.jsonl"))
    rows = [json.loads(line) for path in paths for line in path.read_text().splitlines() if line]
    result = analyze(rows, args.target, args.features)
    result["sources"] = [str(path) for path in paths]
    out = Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("x") as handle:
        json.dump(result,handle,indent=2)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
