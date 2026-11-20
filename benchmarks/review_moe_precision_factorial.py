"""Diagnostic ONLY: cross routed/shared precision, fixed router and checkpointed BF16.

Use qualified benchmark_moe_varlen first-hit JSON. Never selects runtime schedules.
"""
import argparse
import gc
import hashlib
import json
import os
from itertools import accumulate
from pathlib import Path
import statistics
import subprocess
import sys
import time


def run(source_result, output):
    import torch
    from torch.utils.checkpoint import checkpoint
    from benchmarks.benchmark_moe_checkpoint import BF16FFNReference
    from benchmarks.benchmark_moe_composition import compare
    from benchmarks.benchmark_moe_execution import _python_source_digest, _tensor_digest, write
    from benchmarks.benchmark_moe_joint_recompute import _capture
    from benchmarks.benchmark_moe_varlen import compile_options, statistics_ms
    from rtx.autotune.measurement import measure_cuda_batches
    from rtx.autotune.preparation_runtime import _warm_measurement_graph
    from rtx.mlp import MXFP8MLPConfig, mxfp8_mlp
    from rtx.moe import BalancedMoE, MXFP8MoEConfig, mxfp8_routed_experts, torch_routed_experts
    from rtx.routing import BalancedRouting, RoutingKernelConfig

    if output.exists():
        raise FileExistsError(output)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
    uuid = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=uuid",
                                    "--format=csv,noheader"], text=True).strip()
    assert uuid == "GPU-09db6472-cc08-c4cb-90db-560de232e5aa" and torch.cuda.device_count() == 1
    src = json.loads(source_result.read_text())
    case, lengths = src["case"], src["lengths"]
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    report = dict(status="started", uuid=uuid, argv=sys.argv, torch=str(torch.__version__),
                  cuda=torch.version.cuda, source_result=str(source_result), first_hit_sha256=digest(source_result),
                  driver_sha256=digest(Path(__file__)), runtime_sha256=_python_source_digest(Path(__file__).resolve().parents[1]),
                  lengths=lengths, case=case, checks={}, visits={}, timings={}, warm_compile_capture_s={})
    write(output, report)
    torch.manual_seed(src["seed"])
    def model():
        return BalancedMoE(case["d_model"], case["num_experts"], case["expert_hidden_dim"],
            top_k=case["top_k"], shared_hidden_features=case["shared_hidden_dim"],
            num_shared_experts=case["num_shared_experts"], device="cuda", autotune="off",
            config=MXFP8MoEConfig.from_dict(src["expert_config"]), shared_config=MXFP8MLPConfig.from_dict(src["shared_config"]),
            router_options=dict(backend="cute", config=RoutingKernelConfig.from_dict(src["router_config"]),
                                affinity_layout=src["router_affinity_layout"]))
    base = model()
    with torch.no_grad():
        base.router.projection_weight.normal_(std=.05)
    x = (torch.randn(sum(lengths), case["d_model"], device="cuda", dtype=torch.bfloat16)*.3).requires_grad_()
    dy = torch.randn_like(x)
    cu = torch.tensor([0, *accumulate(lengths)], device="cuda", dtype=torch.int32)
    with torch.no_grad():
        route = base.router(x, cu_seqlens=cu)
    identities = dict(input=_tensor_digest(x), upstream=_tensor_digest(dy),
        parameters={name:_tensor_digest(p) for name,p in base.named_parameters()},
        routing={name:_tensor_digest(getattr(route,name)) for name in ("expert_ids","inverse")})
    assert identities == src["identities"], "Source fixture mismatch"
    report["identities"] = identities

    class Mixed(torch.nn.Module):
        def __init__(self, layer, native_experts):
            super().__init__()
            self.layer, self.native_experts = layer, native_experts
        def forward(self, value):
            layer = self.layer
            r = layer.router(value, cu_seqlens=cu)
            gu, down = layer.gate_up.weight, layer.down.weight
            if self.native_experts:
                out = mxfp8_routed_experts(value, gu, down, r, config=layer.config)
            else:
                def routed(v, g, o, w):
                    return torch_routed_experts(v, g, o, BalancedRouting(r.expert_ids, w, r.inverse))
                out = checkpoint(routed, value, gu, down, r.route_weights, use_reentrant=True)
            for gate, proj in zip(layer.shared_gate_up, layer.shared_down):
                if self.native_experts:
                    shared = checkpoint(BF16FFNReference.shared, value, gate.weight, proj.weight, use_reentrant=True)
                else:
                    shared = mxfp8_mlp(value, gate.weight, proj.weight, activation="silu", config=layer.shared_config)
                out = out + shared.float()
            return out.bfloat16()

    graphs = {}
    stream = torch.cuda.current_stream()
    order = ("MM", "MB", "BB", "BM")
    for label in order:
        print("QUALIFY", label, flush=True)
        layer = model()
        layer.load_state_dict(base.state_dict())
        with torch.no_grad():
            selected = layer.router(x, cu_seqlens=cu)
            assert all(torch.equal(a,b) for a,b in zip(route, selected))
        fn = layer if label == "MM" else (BF16FFNReference(layer, checkpoint_ffns=True)
            if label == "BB" else Mixed(layer, label == "MB"))
        compiled = torch.compile(fn, **compile_options(True))
        leaves = (x,*layer.parameters())
        def step(compiled=compiled, leaves=leaves, label=label):
            for leaf in leaves:
                leaf.grad = None
            out = compiled(x, cu_seqlens=cu) if label in ("MM","BB") else compiled(x)
            out.backward(dy)
            return (out, *(p.grad for p in leaves))
        start = time.monotonic()
        graph, values = _capture(step, stream)
        graph.replay()
        stream.synchronize()
        report["warm_compile_capture_s"][label] = time.monotonic()-start
        old = dy.clone()
        replay_errors = []
        for factor in (-.75, 1.):
            dy.copy_(old*factor)
            graph.replay()
            stream.synchronize()
            replay_errors.append(max(r["nrms"] for r in compare(values, step())))
        assert _tensor_digest(dy) == identities["upstream"]
        live = []
        for _ in range(7):
            fresh = step()
            del fresh
            stream.synchronize()
            gc.collect()
            live.append(torch.cuda.memory_allocated())
        report["checks"][label] = dict(replay_nrms=replay_errors, live_allocations=live, all_leaves=len(values))
        write(output, report)
        assert len(set(live[-5:])) == 1, (label, live)
        graphs[label] = graph, values, step, layer
    # Balanced Latin square: every label appears at every visit position.
    for i in range(4):
        for ix in (0,1,3,2):
            label = order[(i+ix)%4]
            graph = graphs[label][0]
            _warm_measurement_graph(graph)
            samples, meta = measure_cuda_batches(graph.replay, samples=7, calls_per_sample=8,
                                                 device=x.device, min_batch_ms=5.)
            row = dict(**statistics_ms(samples), **meta)
            report["visits"].setdefault(label, []).append(row)
            print("TIME", i, label, row["median_ms"], flush=True)
    report["timings"] = {k:statistics.median(r["median_ms"] for r in v) for k,v in report["visits"].items()}
    report["status"] = "qualified"
    write(output, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    if args.output.exists():
        raise FileExistsError(args.output)
    try:
        with torch.cuda.stream(torch.cuda.Stream()):
            run(args.source_result,args.output)
    except Exception:
        import traceback
        from benchmarks.benchmark_moe_execution import write
        failure = json.loads(args.output.read_text()) if args.output.exists() else {}
        write(args.output, {**failure, "status":"failed", "error":traceback.format_exc()})
        raise
