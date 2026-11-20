"""Matched-source MoE execution benchmark: host latency versus outer graph replay.

The controller launches sequential fresh worker processes in ABBA order. It
never runs two GPU workers concurrently. Compilation, correctness and graph
capture are outside timing. Both source snapshots need the frozen
``benchmarks/profile_moe.py`` helper and the same seed-study files.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def summary(samples):
    median = statistics.median(samples)
    return dict(median_ms=median, min_ms=min(samples), max_ms=max(samples),
                relative_mad=statistics.median(abs(v-median) for v in samples)/max(median, 1e-12),
                samples_ms=samples)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("worker", "compare"):
        s = sub.add_parser(name)
        s.add_argument("--output", type=Path, required=True)
        s.add_argument("--case-id", default="d768_e48_k6_h256")
        s.add_argument("--tokens", type=int, default=2000)
        s.add_argument("--seed-studies", nargs="+", type=Path, default=[])
        s.add_argument("--config", type=Path)
        s.add_argument("--router-config",type=Path,
                       help="Explicit frozen router config, applied equally to native and BF16 whole layers")
        s.add_argument("--shared-config",type=Path,
                       help="Explicit native shared-MLP schedule; BF16 shared remains compiler-managed")
        s.add_argument("--whole-layer", action="store_true")
        s.add_argument("--decompose", action="store_true",
                       help="Whole-layer router/shared/parameterized expert attribution; phases are not additive")
        s.add_argument("--router-details", action="store_true",
                       help="Also time real-affinity/Sinkhorn/QB/repair/metadata component boundaries")
        s.add_argument("--include-bf16", action="store_true")
        s.add_argument("--samples", type=int, default=9)
        s.add_argument("--rounds", type=int, default=2)
        s.add_argument("--device", default="cuda:0")
        s.add_argument("--phases", nargs="+", default=["expert_step", "public_expert_step"])
    worker = sub.choices["worker"]
    worker.add_argument("--source", type=Path, required=True)
    worker.add_argument("--label", default="candidate")
    worker.add_argument("--fixture", type=Path)
    worker.add_argument("--save-fixture", type=Path)
    worker.add_argument("--tune-trials", type=int, default=0)
    worker.add_argument("--tune-seconds", type=float, default=120.)
    worker.add_argument("--tune-router-trials",type=int,default=0)
    worker.add_argument("--tune-router-seconds",type=float,default=120.)
    worker.add_argument("--save-router-config",type=Path)
    worker.add_argument("--tune-shared-trials",type=int,default=0)
    worker.add_argument("--tune-shared-seconds",type=float,default=180.)
    worker.add_argument("--save-shared-config",type=Path)
    worker.add_argument("--whole-reference", type=Path)
    worker.add_argument("--save-whole-reference", type=Path)
    compare = sub.choices["compare"]
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--baseline-config", type=Path)
    compare.add_argument("--candidate-config", type=Path)
    compare.add_argument("--baseline-router-config",type=Path)
    compare.add_argument("--candidate-router-config",type=Path)
    compare.add_argument("--baseline-shared-config",type=Path)
    compare.add_argument("--candidate-shared-config",type=Path)
    compare.add_argument("--python", type=Path, default=Path(sys.executable))
    return p


def _use_source(source):
    source = source.resolve()
    current = Path(__file__).resolve().parents[1]
    # This script can reside outside either snapshot. Do not accidentally use
    # its project's editable installation or namespace benchmark package.
    sys.path[:] = [str(source)] + [p for p in sys.path
        if p and Path(p).resolve() not in (current, current / "benchmarks", source)]
    os.chdir(source)
    rtx = importlib.import_module("rtx")
    helper = importlib.import_module("benchmarks.profile_moe")
    assert Path(rtx.__file__).resolve().is_relative_to(source), rtx.__file__
    assert Path(helper.__file__).resolve().is_relative_to(source), helper.__file__
    return helper


def _tensor_digest(tensor):
    value = tensor.detach().contiguous().cpu().reshape(-1).view(__import__("torch").uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _python_source_digest(source):
    """Whole-package identity includes routing changes absent from MoE key."""
    digest=hashlib.sha256()
    for path in sorted((source/"rtx").rglob("*.py")):
        digest.update(str(path.relative_to(source)).encode());digest.update(path.read_bytes())
    return digest.hexdigest()


def _host(call, samples):
    import torch
    values = []
    # Clock ramp is excluded. Keep no saved training graph from a prior call.
    until = time.monotonic() + .25
    while time.monotonic() < until:
        output = call()
        del output
    torch.cuda.synchronize()
    for _ in range(samples):
        start = time.perf_counter()
        output = call()
        torch.cuda.synchronize()
        values.append((time.perf_counter()-start)*1000)
        del output
    return summary(values)


def _whole_validation(output, gradients, names, *, reference=None, save_reference=None):
    """Source-parity check for every real optimizer-visible parameter.

    This is separate from the independent FP64 expert oracle. It detects lost
    gradient plumbing across a wrapper refactor, not BF16 convergence parity.
    """
    import torch
    tensors=(output,*gradients)
    names=("output","input",*names)
    if len(tensors)!=len(names):raise ValueError("Whole validation gradient/name arity mismatch")
    if len(set(names))!=len(names):raise ValueError("Whole validation names must be unique")
    for name,value in zip(names,tensors):
        if not bool(value.isfinite().all()):raise AssertionError(("nonfinite",name))
    record=dict(oracle="matched_source_all_parameter_parity",count=len(tensors),
                names=names,all_finite=True,
                digests={name:_tensor_digest(value) for name,value in zip(names,tensors)})
    if reference:
        saved=torch.load(reference,map_location="cpu",weights_only=True)
        if list(saved["names"])!=list(names):raise ValueError("Whole reference parameter names/order differ")
        errors={}
        for name,value,expected in zip(names,tensors,saved["values"]):
            if value.shape!=expected.shape or value.dtype!=expected.dtype:
                raise ValueError(("Whole reference tensor contract differs",name))
            delta=value.detach().float().cpu()-expected.float()
            maximum=float(delta.abs().max());nrms=float(delta.norm()/expected.float().norm().clamp_min(1e-12))
            if nrms>.006 and maximum>2e-7:raise AssertionError((name,nrms,maximum))
            errors[name]=dict(nrms=nrms,max_abs=maximum)
        record["errors"]=errors
    if save_reference:
        save_reference.parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(names=names,values=[v.detach().cpu() for v in tensors]),save_reference)
    return record


def _router_component_calls(model,x,route,route_gradient,options):
    """Use exact model data/upstreams, not easier synthetic router matrices."""
    import math
    import torch
    from rtx import _routing_reference as reference
    from rtx.routing import _sinkhorn_forward,_sinkhorn_backward
    from rtx._routing_hard import HardRoutingPlan,hard_schedule,_plans
    router=model.router;c=router.config
    n=x.shape[0];k=router.top_k;iterations=router.sinkhorn_iterations
    scheduled=reference.padded_token_count(n,router.num_experts,k)
    affinity=torch.compile(router.log_affinities,**options)
    def get_affinity():
        return torch.nn.functional.pad(affinity(x),(0,0,0,scheduled-n),value=-math.log(2.))
    values=get_affinity().detach()
    f=c.forward;b=c.backward
    def sinkhorn_forward():
        return _sinkhorn_forward(values,k,iterations,f.threads,f.row_groups,f.column_groups)
    logp,history=sinkhorn_forward()
    scores=logp.detach().exp();owner=id(router)
    def hard():
        return hard_schedule(scores,k,router.dual_iterations,router.repair_rounds,
            c.segmented_quotas,owner,c.quota_threshold,c.qb_backend,c.qb_threads)
    assignments=hard()
    identity=(scores.device.index,torch.cuda.current_stream(scores.device).cuda_stream,
              owner,*scores.shape,k,router.dual_iterations,router.repair_rounds,
              c.segmented_quotas,c.quota_threshold,c.qb_backend,c.qb_threads)
    executed_plan=_plans.get(identity)
    repair_rounds_taken=None if executed_plan is None else int(executed_plan.phase.item())
    p=logp.detach().requires_grad_()
    metadata=torch.compile(lambda v:reference.routing_metadata(v,assignments,num_real_tokens=n),**options)
    actual=metadata(p)
    if not torch.equal(actual[0],route.expert_ids) or not torch.equal(actual[2],route.inverse):
        raise AssertionError("Router attribution reconstructed different assignments")
    torch.testing.assert_close(actual[1],route.route_weights,rtol=2e-5,atol=2e-7)
    logp_gradient=torch.autograd.grad(actual[1],p,route_gradient)[0].detach()
    def sinkhorn_backward():
        return _sinkhorn_backward(logp_gradient,history,k,iterations,b.threads,b.row_groups,b.column_groups)
    affinity_gradient=sinkhorn_backward().detach()
    # QB initialization is stateless compiled arithmetic; the enclosing hard
    # plan is kept alive but this callable itself is safe on the timing stream.
    plan=HardRoutingPlan(scores,k,router.dual_iterations,router.repair_rounds,
        c.segmented_quotas,c.quota_threshold,c.qb_backend,c.qb_threads)
    parameters=tuple(router.parameters())
    calls=dict(router_affinities_forward=get_affinity,
        router_affinities_step=lambda:torch.autograd.grad(get_affinity(),(x,*parameters),affinity_gradient),
        router_sinkhorn_forward=sinkhorn_forward,router_sinkhorn_backward=sinkhorn_backward,
        router_qb=lambda:plan.initialize_op(scores),router_hard_total=hard,
        router_metadata_forward=lambda:metadata(p),
        router_metadata_step=lambda:torch.autograd.grad(metadata(p)[1],p,route_gradient))
    return calls,dict(scheduled_tokens=scheduled,config=c.to_dict(),
        repair_rounds_taken=repair_rounds_taken,
        affinity_digest=_tensor_digest(values),assignments_digest=_tensor_digest(assignments),
        logp_gradient_digest=_tensor_digest(logp_gradient),
        affinity_gradient_digest=_tensor_digest(affinity_gradient),
        scope="QB initialization excludes extraction; hard_total includes QB+conditional repair+copies")


def _component_calls(model, x, route, dy, *, router_details=False,include_bf16=False):
    """Production component boundaries, preserving individual expert Parameters.

    Independent component timings include their own autograd/compile boundary;
    summing them is not a decomposition of the compiled whole-layer latency.
    """
    import torch
    from rtx.moe import mxfp8_routed_experts
    from rtx.mlp import mxfp8_mlp
    from rtx.routing import BalancedRouting
    options=dict(fullgraph=True,dynamic=False,options={"triton.cudagraphs":False})
    router=torch.compile(model.router,**options)
    router_parameters=tuple(model.router.parameters())
    expert_parameters=tuple(model.gate_up.parameters())+tuple(model.down.parameters())
    shared_parameters=tuple(model.shared_gate_up.parameters())+tuple(model.shared_down.parameters())
    fixed_route=BalancedRouting(route.expert_ids,route.route_weights.detach().requires_grad_(),route.inverse)
    def routed(value,weights):
        return mxfp8_routed_experts(value,model.gate_up.weight,model.down.weight,
            BalancedRouting(fixed_route.expert_ids,weights,fixed_route.inverse),config=model.config)
    expert=torch.compile(routed,**options)
    # This upstream really reaches the router's normalized mixture weights in
    # the complete layer, unlike an unrelated random routing benchmark input.
    route_gradient=torch.autograd.grad(expert(x,fixed_route.route_weights),fixed_route.route_weights,dy.float())[0].detach()
    def shared(value):
        result=None
        for gate,down in zip(model.shared_gate_up,model.shared_down):
            if model._shared_spec is None:
                output=mxfp8_mlp(value,gate.weight,down.weight,activation="silu",config=model.shared_config)
            else:
                from rtx._mlp_autotune import selected_mlp
                output=selected_mlp(value,gate.weight,down.weight,"silu",model._shared_spec,True)[0]
            result=output.float() if result is None else result+output.float()
        return result.to(torch.bfloat16)
    result={"router_forward":lambda:router(x),
        "router_step":lambda:torch.autograd.grad(router(x).route_weights,(x,*router_parameters),route_gradient),
        "parameterized_expert_step":lambda:torch.autograd.grad(expert(x,fixed_route.route_weights),
            (x,*expert_parameters,fixed_route.route_weights),dy.float())}
    if shared_parameters:
        shared_compiled=torch.compile(shared,**options)
        result.update(shared_forward=lambda:shared_compiled(x),
            shared_step=lambda:torch.autograd.grad(shared_compiled(x),(x,*shared_parameters),dy))
        if include_bf16:
            def shared_bf16(value):
                total=None
                for gate,down in zip(model.shared_gate_up,model.shared_down):
                    width=down.weight.shape[1]
                    projected=torch.nn.functional.linear(value,gate.weight).float()
                    hidden=(torch.nn.functional.silu(projected[:,:width])*projected[:,width:]).bfloat16()
                    output=torch.nn.functional.linear(hidden,down.weight).float()
                    total=output if total is None else total+output
                return total.bfloat16()
            shared_baseline=torch.compile(shared_bf16,**options)
            result["bf16_shared_step"]=lambda:torch.autograd.grad(shared_baseline(x),(x,*shared_parameters),dy)
    contract=dict(router_parameter_count=len(router_parameters),expert_parameter_count=len(expert_parameters),
        shared_parameter_count=len(shared_parameters),route_gradient_digest=_tensor_digest(route_gradient),
        bf16_shared_contract="same BF16 Parameters/activation; ordinary autograd saves intermediates; native recomputes",
        semantics="standalone production components; timings non-additive; subtraction is not pure plumbing cost")
    if router_details:
        details,detail_contract=_router_component_calls(model,x,route,route_gradient,options)
        result.update(details);contract["router_details"]=detail_contract
    return result,contract


def _apply_router_config(calls,config):
    """A common-router optimization must be common to both precision paths."""
    models=[inspect.getclosurevars(calls["whole_step"]).nonlocals["native"]._orig_mod]
    if "bf16_whole_step" in calls:
        models.append(inspect.getclosurevars(calls["bf16_whole_step"]).nonlocals["bfn"]._orig_mod)
    for model in models:
        model.router.config=config;model.router.sinkhorn_config=config.forward
        model.router.autotune="off";model.router._refresh_selection_spec()
    return models[0]


def _apply_shared_config(calls,config):
    # BF16 has no MXFP8 schedule. Do not replace or weaken that baseline.
    model=inspect.getclosurevars(calls["whole_step"]).nonlocals["native"]._orig_mod
    model.shared_config=config;model._shared_spec=None
    return model


def worker(args):
    args.source = args.source.resolve()
    args.output = args.output.resolve()
    args.config = args.config.resolve() if args.config else None
    args.router_config=args.router_config.resolve() if args.router_config else None
    args.shared_config=args.shared_config.resolve() if args.shared_config else None
    args.seed_studies = [p.resolve() for p in args.seed_studies]
    args.fixture = args.fixture.resolve() if args.fixture else None
    args.save_fixture = args.save_fixture.resolve() if args.save_fixture else None
    helper = _use_source(args.source)
    import torch
    from rtx.autotune.moe import source_revision
    from rtx.moe import mxfp8_routed_experts
    from rtx.routing import BalancedRouting

    torch.cuda.set_device(args.device)
    requested_trials = args.tune_trials
    args.tune_trials = 0  # Bound the search explicitly below, outside timing.
    args.skip_baseline = not (args.include_bf16 and args.whole_layer)
    if args.decompose and not args.whole_layer:raise ValueError("--decompose requires --whole-layer")
    if args.router_details and not args.decompose:raise ValueError("--router-details requires --decompose")
    if args.whole_layer and args.fixture:
        raise ValueError("Whole-layer source comparison uses matched state/input digests, not an expert-only fixture")
    began = time.monotonic()
    case, shape, config, origin, operands, dy, calls = helper.build(args)
    x, gu, down, ids, weights, inverse = operands
    router_override=None
    if args.router_config:
        if not args.whole_layer:raise ValueError("--router-config requires --whole-layer")
        from rtx.routing import RoutingKernelConfig
        payload=json.loads(args.router_config.read_text())
        router_override=RoutingKernelConfig.from_dict(payload.get("config",payload))
        _apply_router_config(calls,router_override)
    router_tuning=None
    if args.tune_router_trials:
        if not args.whole_layer:raise ValueError("Router tuning requires --whole-layer")
        from rtx.autotune.effort import AutotunePolicy
        from rtx.autotune.orchestrator import NoEligibleTimingError
        model=inspect.getclosurevars(calls["whole_step"]).nonlocals["native"]._orig_mod
        initial=model.router.config
        policy=AutotunePolicy.preset("quick",max_trials=args.tune_router_trials,
            max_seconds=args.tune_router_seconds,search_schedule="progressive",candidate_mix="mixed",confirmation_repeats=0)
        began_tuning=time.monotonic()
        try:
            tuned=model.router.autotune_(x,tuning_policy=policy,phase="train",
                output_dir=args.output.parent/"router-tuning",progress=lambda message:print(message,flush=True))
            router_override=tuned.config
            router_tuning=dict(status="selected",median_ms=tuned.median_ms)
        except NoEligibleTimingError as error:
            if error.fallback_config is None:raise
            router_override=error.fallback_config
            router_tuning=dict(status="timing_ineligible_correctness_checked_fallback",median_ms=None)
        router_tuning.update(cold_seconds=time.monotonic()-began_tuning,
            requested_trials=args.tune_router_trials,initial_config=initial.to_dict(),
            selected_config=router_override.to_dict())
        _apply_router_config(calls,router_override)
        if args.save_router_config:write(args.save_router_config,dict(config=router_override.to_dict(),tuning=router_tuning))
    shared_override=None;shared_tuning=None
    if args.shared_config:
        if not args.whole_layer:raise ValueError("--shared-config requires --whole-layer")
        from rtx.mlp import MXFP8MLPConfig
        payload=json.loads(args.shared_config.read_text())
        shared_override=MXFP8MLPConfig.from_dict(payload.get("config",payload))
        _apply_shared_config(calls,shared_override)
    if args.tune_shared_trials:
        if not args.whole_layer:raise ValueError("Shared tuning requires --whole-layer")
        from rtx.autotune.effort import AutotunePolicy
        from rtx.autotune.mlp import tune_mxfp8_mlp
        from rtx.autotune.orchestrator import NoEligibleTimingError
        model=inspect.getclosurevars(calls["whole_step"]).nonlocals["native"]._orig_mod
        if len(model.shared_down)!=1:raise ValueError("This bounded shared study requires exactly one shared MLP")
        initial=model.shared_config
        policy=AutotunePolicy.preset("quick",max_trials=args.tune_shared_trials,
            max_seconds=args.tune_shared_seconds,search_schedule="progressive",candidate_mix="mixed",confirmation_repeats=0)
        began_tuning=time.monotonic()
        try:
            tuned=tune_mxfp8_mlp(x.detach().requires_grad_(),model.shared_gate_up[0].weight.detach().requires_grad_(),
                model.shared_down[0].weight.detach().requires_grad_(),activation="silu",phase="train",initial=initial,
                tuning_policy=policy,output_dir=args.output.parent/"shared-tuning",qualification_rounds=4,
                progress=lambda message:print(message,flush=True))
            shared_override=tuned.config
            shared_tuning=dict(status="selected",median_ms=tuned.median_ms)
        except NoEligibleTimingError as error:
            if error.fallback_config is None:raise
            shared_override=error.fallback_config
            shared_tuning=dict(status="timing_ineligible_correctness_checked_fallback",median_ms=None)
        shared_tuning.update(cold_seconds=time.monotonic()-began_tuning,requested_trials=args.tune_shared_trials,
            initial_config=initial.to_dict(),selected_config=shared_override.to_dict())
        _apply_shared_config(calls,shared_override)
        if args.save_shared_config:write(args.save_shared_config,dict(config=shared_override.to_dict(),tuning=shared_tuning))
    tuning = None
    if requested_trials:
        from rtx.autotune.effort import AutotunePolicy
        from rtx.autotune.moe import tune_moe
        from rtx.moe import _selection_spec, _parse_spec, _run_forward, _run_backward
        calls["frozen_expert_step"] = calls["expert_step"]
        policy = AutotunePolicy(max_trials=requested_trials, max_seconds=args.tune_seconds,
            search_schedule="progressive", candidate_mix="mixed", confirmation_repeats=0)
        cold = time.monotonic()
        result = tune_moe(operands, _parse_spec(_selection_spec(config, "balanced", policy,
            args.output.parent/"tuning")), training=True, progress=lambda message: print(message, flush=True))
        tuning = dict(cold_seconds=time.monotonic()-cold, requested_trials=requested_trials,
            max_seconds=args.tune_seconds, initial_config=config.to_dict(), selected_config=result.config.to_dict(),
            median_ms=result.median_ms, evidence=getattr(result, "tuning_evidence", None))
        config = result.config
        calls["expert_forward"] = lambda: _run_forward(*operands, config)
        calls["expert_backward"] = lambda: _run_backward(*operands, dy, config)
        calls["expert_step"] = lambda: (calls["expert_forward"](), *calls["expert_backward"]())
        if args.whole_layer:
            model = inspect.getclosurevars(calls["whole_step"]).nonlocals["native"]._orig_mod
            model.config = config
            model._expert_spec = _selection_spec(config, "off", None, None)
    public = torch.compile(lambda x, gu, down, weights: mxfp8_routed_experts(
        x, gu, down, BalancedRouting(ids, weights, inverse), config=config),
        fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
    calls["public_expert_forward"] = lambda: public(x, gu, down, weights)
    calls["public_expert_step"] = lambda: torch.autograd.grad(public(x, gu, down, weights),
                                                             (x, gu, down, weights), dy)
    if args.include_bf16:
        from rtx.moe import torch_routed_experts
        baseline = torch.compile(lambda x, gu, down, weights: torch_routed_experts(
            x, gu, down, BalancedRouting(ids, weights, inverse)), fullgraph=True, dynamic=False,
            options={"triton.cudagraphs": False})
        calls["bf16_expert_step"] = lambda: torch.autograd.grad(baseline(x, gu, down, weights),
                                                               (x, gu, down, weights), dy)
    phases = list(args.phases)
    if args.whole_layer and "whole_step" not in phases:
        phases.append("whole_step")
    component_contract=None
    if args.decompose:
        model=inspect.getclosurevars(calls["whole_step"]).nonlocals["native"]._orig_mod
        components,component_contract=_component_calls(model,x,BalancedRouting(ids,weights,inverse),dy.bfloat16(),
                                                       router_details=args.router_details,include_bf16=args.include_bf16)
        calls.update(components)
        phases.extend(name for name in components if name not in phases)
        if args.include_bf16 and "bf16_whole_step" not in phases:phases.append("bf16_whole_step")
    for phase in phases:
        if phase not in calls:
            raise ValueError(f"Unknown/unavailable phase {phase!r}; available={sorted(calls)}")
    record = dict(status="running", label=args.label, case=case, shape=shape,
        source=str(args.source.resolve()), source_revision=source_revision(),
        whole_source_revision=_python_source_digest(args.source),config=config.to_dict(),
        seed_origin=origin, commands=sys.argv, torch=torch.__version__, cuda=torch.version.cuda,
        device=str(torch.cuda.get_device_properties(torch.cuda.current_device())),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256(Path(helper.__file__).read_bytes()).hexdigest(),
        compile_options=dict(fullgraph=True, dynamic=False, options={"triton.cudagraphs": False}),
        timing_contract="boundary-synchronized host and externally captured CUDAgraph; no automatic per-op graphs",
        tuning=tuning,
        component_contract=component_contract,
        router_override=None if router_override is None else router_override.to_dict(),
        router_tuning=router_tuning,
        shared_override=None if shared_override is None else shared_override.to_dict(),shared_tuning=shared_tuning,
        input_digests={name:_tensor_digest(t) for name,t in zip(
            ("x","gate_up","down","ids","weights","inverse","dy"),(*operands,dy))})
    if args.whole_layer:
        # The established frozen helper closes over all real model parameters.
        parameters = inspect.getclosurevars(calls["whole_step"]).nonlocals["params"]
        record["whole_parameter_digests"] = [_tensor_digest(t) for t in parameters]
    for phase in phases:
        output = calls[phase]()
        del output
    torch.cuda.synchronize()
    record["construction_compile_seconds"] = time.monotonic()-began
    validation_start = time.monotonic()
    actual_public = lambda: (calls["public_expert_forward"](), *calls["public_expert_step"]())
    # The helper oracle is independent FP64 quantized Torch. For large N it
    # checks both full-C weight gradients on first/last experts, all finite.
    record["validation_public"] = helper.correctness(operands, dy, {"expert_step":actual_public}, args.tokens)
    record["validation_seconds"] = time.monotonic()-validation_start
    if args.whole_layer:
        names=tuple(name for name,_ in model.named_parameters()) if args.decompose else None
        if names is None:
            model=inspect.getclosurevars(calls["whole_step"]).nonlocals["native"]._orig_mod
            names=tuple(name for name,_ in model.named_parameters())
        record["validation_whole"]=_whole_validation(calls["whole_forward"](),calls["whole_step"](),names,
            reference=args.whole_reference,save_reference=args.save_whole_reference)
    record["timings"] = {phase:dict(host=[], graph=[]) for phase in phases}
    write(args.output, record)
    stream = torch.cuda.Stream()
    for round_id in range(args.rounds):
        order = phases if round_id % 2 == 0 else list(reversed(phases))
        for phase in order:
            call = calls[phase]
            host = _host(call, args.samples)
            graph = helper.measure(call, stream, args.samples)
            record["timings"][phase]["host"].append(host)
            record["timings"][phase]["graph"].append(graph)
            print("TIME", args.label, round_id, phase,
                  f"host={host['median_ms']:.6f}ms graph={graph['median_ms']:.6f}ms", flush=True)
            write(args.output, record)
    live = []
    live_phase="whole_step" if args.whole_layer else "public_expert_step"
    torch.cuda.reset_peak_memory_stats()
    live_before=torch.cuda.memory_allocated()
    for _ in range(8):
        output = calls[live_phase]()
        del output
        torch.cuda.synchronize()
        live.append(torch.cuda.memory_allocated())
    record.update(status="complete", live_allocated_samples=live, live_growth=max(live)-min(live),
                  live_scope=live_phase,steady_peak_allocated=torch.cuda.max_memory_allocated(),
                  steady_temporary_peak=torch.cuda.max_memory_allocated()-live_before)
    write(args.output, record)


def aggregate(records):
    rows = []
    if not records:
        return rows
    expected = records[0]["input_digests"]
    for record in records:
        if record["input_digests"] != expected:
            raise ValueError("Input/route/parameter digests differ across source workers")
        if record.get("whole_parameter_digests") != records[0].get("whole_parameter_digests"):
            raise ValueError("Whole-model parameter digests differ across source workers")
        if record["status"] != "complete":
            raise ValueError("Cannot aggregate an incomplete worker")
    phases = list(records[0]["timings"])
    for phase in phases:
        for mode in ("host", "graph"):
            values = {label:[statistics.median(r["median_ms"] for r in record["timings"][phase][mode])
                             for record in records if record["label"] == label]
                      for label in ("baseline", "candidate")}
            if not all(values.values()):
                raise ValueError("Both source labels are required")
            medians = {key:statistics.median(value) for key,value in values.items()}
            rows.append(dict(phase=phase, execution=mode, baseline_ms=medians["baseline"],
                candidate_ms=medians["candidate"], speedup=medians["baseline"]/medians["candidate"],
                worker_medians=values,
                worker_relative_range={k:max(v)/min(v)-1 for k,v in values.items()}))
    return rows


def compare(args):
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    fixture = args.output / "matched-expert-inputs.pt"
    records = []
    for index, label in enumerate(("baseline", "candidate", "candidate", "baseline")):
        source = getattr(args, label).resolve()
        destination = args.output / f"visit-{index:02d}-{label}.json"
        # Resolving a venv's python symlink jumps to the system interpreter and
        # silently drops its site-packages. Keep the launcher path intact.
        command = [str(args.python.absolute()), str(script), "worker", "--source", str(source),
            "--label", label, "--output", str(destination), "--case-id", args.case_id,
            "--tokens", str(args.tokens), "--device", args.device, "--samples", str(args.samples),
            "--rounds", str(args.rounds), "--phases", *args.phases]
        config = getattr(args, label+"_config") or args.config
        if config:
            command += ["--config", str(config.resolve())]
        router_config=getattr(args,label+"_router_config") or args.router_config
        if router_config:command += ["--router-config",str(router_config.resolve())]
        shared_config=getattr(args,label+"_shared_config") or args.shared_config
        if shared_config:command += ["--shared-config",str(shared_config.resolve())]
        if args.seed_studies:
            command += ["--seed-studies", *[str(p.resolve()) for p in args.seed_studies]]
        if args.whole_layer:
            command.append("--whole-layer")
            if args.decompose:
                command.append("--decompose")
                if args.router_details:command.append("--router-details")
                command += ["--save-whole-reference" if index==0 else "--whole-reference",
                            str(args.output/"matched-whole-reference.pt")]
        elif index == 0:
            command += ["--save-fixture", str(fixture)]
        else:
            command += ["--fixture", str(fixture)]
        if args.include_bf16:
            command.append("--include-bf16")
        environment = dict(os.environ, TORCHINDUCTOR_COMPILE_THREADS="2",
            TORCHINDUCTOR_CACHE_DIR=str(args.output / ("inductor-"+label)), PYTHONPATH=str(source))
        print("START", index, label, flush=True)
        with destination.with_suffix(".log").open("w") as log:
            result = subprocess.run(command, cwd=source, env=environment, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{label} worker failed ({result.returncode}); inspect {destination.with_suffix('.log')}")
        records.append(json.loads(destination.read_text()))
        if index >= 1:
            write(args.output / "comparison.json", dict(status="running", order=[r["label"] for r in records],
                                                       results=aggregate(records)))
    write(args.output / "comparison.json", dict(status="complete", order=[r["label"] for r in records],
        scope="Matched precision/config policies unless explicit per-source config override; source ABBA, not BF16 baseline",
        results=aggregate(records), workers=[str(args.output/f"visit-{i:02d}-{r['label']}.json") for i,r in enumerate(records)]))
    print("DONE", args.output / "comparison.json", flush=True)


def main():
    args = parser().parse_args()
    if args.samples < 3 or args.rounds < 1:
        raise ValueError("At least three samples and one round are required")
    worker(args) if args.command == "worker" else compare(args)


if __name__ == "__main__":
    main()
