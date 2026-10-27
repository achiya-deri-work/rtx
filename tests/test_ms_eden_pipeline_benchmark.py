import json
import pytest

from benchmarks import compare_ms_eden_pipeline as benchmark


def test_baseline_exact_policy_and_per_phase_order():
    pair = benchmark.schedule_pair()
    for phase, value in pair.items():
        schedule = json.loads(value)
        assert schedule["producer_threads"] == 128
        assert schedule["producer_blocks"] == 256
        assert schedule["producer_load"] == "coalesced_wide"
        assert schedule["producer_signs"] == "register"
        assert schedule["rng_mode"] == "explicit"
        assert schedule["amax_reduction"] == "inline"
        assert schedule["finalizer_threads"] == 128
        assert schedule["finalizer_blocks"] == 224
        assert schedule["producer_order"] == ("row_major" if phase == "dx" else "reduction_major")


def test_phase_overrides_and_public_builder():
    pair = benchmark.schedule_pair({"amax_reduction": "staged"}, {"producer_blocks": 32})
    assert json.loads(pair["dx"])["producer_blocks"] == 32
    assert json.loads(pair["dw"])["producer_blocks"] == 256
    assert all(json.loads(value)["amax_reduction"] == "staged" for value in pair.values())
    args = benchmark.options((512, 512, 512), pair, pipeline="paired")
    assert args.policy == "fp4_fp4" and args.backward_pipeline == "paired"
    assert args.dx_schedule_json == pair["dx"]
    overridden = benchmark.schedule_pair({"producer_order": "reduction_major"})
    assert all(json.loads(value)["producer_order"] == "reduction_major" for value in overridden.values())


@pytest.mark.parametrize("changes", [{"reader_mode": "staged"}])
def test_numerical_policy_changes_rejected(changes):
    with pytest.raises(ValueError, match="preserves original BF16"):
        benchmark.schedule_pair(changes)


def test_fused_rng_is_explicitly_recorded_and_not_default():
    pair = benchmark.schedule_pair({'rng_mode': 'philox', 'philox_packet_size': 4})
    assert all(json.loads(value)['rng_mode'] == 'philox' for value in pair.values())
    assert all(json.loads(value)['philox_packet_size'] == 4 for value in pair.values())
    assert all(json.loads(value)['rng_mode'] == 'explicit'
               for value in benchmark.schedule_pair().values())


def test_fixed_uniform_probe_does_not_claim_seeded_qualification():
    with pytest.raises(ValueError, match='requires explicit RNG'):
        benchmark.validate_exact({'rng_mode': 'philox'})


def test_shape_coverage_and_invalid_shape():
    assert len(benchmark.SHAPES) == 7
    assert benchmark.shape_value("long") == (32768, 512, 512)
    assert benchmark.shape_value("wide") == (4096, 2048, 2048)
    assert benchmark.shape_value("128,256,384") == (128, 256, 384)
    with pytest.raises(ValueError):
        benchmark.shape_value("129,256,384")


def test_randomized_rounds_have_complete_ab_ba_clusters():
    rounds = list(benchmark.round_orders(10, 941))
    assert rounds == list(benchmark.round_orders(10, 941))
    assert len({tuple(orders[0]) for _, orders in rounds}) == 2
    for _, orders in rounds:
        assert set(orders) == {("baseline", "candidate"), ("candidate", "baseline")}


def test_cli_propagates_schedule_pipeline_scope(monkeypatch, tmp_path):
    seen = []
    def fake(*args, **kwargs):
        seen.append((args, kwargs))
        return {"timings": {}}
    monkeypatch.setattr(benchmark, "compare", fake)
    benchmark.main(["--shape", "tiny", "--output", str(tmp_path/"out.json"),
                    "--candidate-pipeline", "paired", "--candidate-dw-json", '{"producer_blocks":32}',
                    "--scope", "backward"])
    args, kwargs = seen[0]
    assert args[0] == (512, 512, 512)
    assert args[2] == {}
    assert kwargs["pipeline"] == "paired" and kwargs["scopes"] == ("backward",)
    assert kwargs["dw_overrides"] == {"producer_blocks": 32}
    assert kwargs["min_batch_ms"] >= 2


def test_exact_cli_does_not_run_timing(monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark, "validate_exact", lambda changes, seed: dict(changes=changes, seed=seed))
    output = tmp_path/"exact.json"
    benchmark.main(["--validate-exact", "--output", str(output)])
    assert json.loads(output.read_text())["changes"] == {}
    with pytest.raises(FileExistsError):
        benchmark.main(["--validate-exact", "--output", str(output)])


def test_default_candidate_is_exact_identity_for_pipeline_only_probes():
    args = benchmark.parser().parse_args(["--output", "unused.json", "--candidate-pipeline", "paired"])
    assert json.loads(args.candidate_json) == {}
    assert benchmark.schedule_pair(json.loads(args.candidate_json)) == benchmark.schedule_pair()


def test_stable_baseline_options_preserve_full_public_worker_policy():
    config = '{"dx":{},"dw":{}}'
    args = benchmark.options((512, 512, 512), benchmark.schedule_pair(),
        policy="stable_mxfp8", mxfp8_config_json=config)
    assert args.policy == "stable_mxfp8"
    assert args.backward_pipeline == "serial"
    assert args.mxfp8_config_json == config
    assert benchmark.parser().parse_args(["--output", "unused.json"]).baseline_policy == "fp4_fp4"


def test_cli_stable_baseline_config_forwarding(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(benchmark, "compare", lambda *a, **kw: seen.append(kw) or {"timings": {}})
    benchmark.main(["--output", str(tmp_path / "out.json"), "--baseline-policy", "stable_mxfp8",
                    "--mxfp8-config-json", "{}", "--candidate-pipeline", "paired"])
    assert seen[0]["baseline_policy"] == "stable_mxfp8"
    assert seen[0]["mxfp8_config_json"] == "{}"


def target_timings(speedup=1.21):
    from rtx.autotune.measurement import runtime_timing_metadata
    quality = {name: runtime_timing_metadata([1.] * 10) for name in ("baseline", "candidate")}
    paired = {"round_cluster_bootstrap95": [speedup, speedup]}
    return {"backward": dict(quality=quality, paired=paired,
        by_order={kind: dict(paired) for kind in ("AB", "BA")},
        order_quality={kind: {name: runtime_timing_metadata([1.] * 10)
            for name in ("baseline", "candidate")} for kind in ("AB", "BA")})}


def test_performance_target_requires_stable_baseline_and_complete_backward():
    assert benchmark.performance_target(target_timings(), baseline_policy="stable_mxfp8")["met"]
    assert not benchmark.performance_target(target_timings(), baseline_policy="fp4_fp4")["met"]
    assert not benchmark.performance_target({"total": {}}, baseline_policy="stable_mxfp8")["met"]
    assert not benchmark.performance_target(target_timings(1.19), baseline_policy="stable_mxfp8")["met"]


def test_performance_target_rejects_order_instability_and_order_regression():
    timings = target_timings()
    timings["backward"]["order_quality"]["BA"]["candidate"]["sampling"]["collection"]["stationary"] = False
    result = benchmark.performance_target(timings, baseline_policy="stable_mxfp8")
    assert not result["met"] and "nonstationary_BA_candidate" in result["reasons"]
    timings = target_timings()
    timings["backward"]["by_order"]["BA"]["round_cluster_bootstrap95"] = [1.19, 1.3]
    assert not benchmark.performance_target(timings, baseline_policy="stable_mxfp8")["met"]


def test_stable_gradient_gate_is_not_fp4_gate():
    import torch
    reference = torch.ones(8)
    actual = reference * 1.10
    benchmark.gradient_metrics(actual, reference)
    with pytest.raises(RuntimeError, match="exceeds 0.06"):
        benchmark.gradient_metrics(actual, reference, gate=.06)
