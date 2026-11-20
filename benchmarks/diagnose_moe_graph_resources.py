"""Force an RTX graph owner drop across CUDA capture boundaries.

Run each scenario in a new process. Instrumentation is opt-in: wrappers record
the CUDA operation already performed by the tested implementation, not a
second synchronizing operation.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import traceback
import weakref

import torch

import rtx
from rtx import _resources
from rtx.runtime import tensor_storage_bytes


def run(args):
    from rtx import fp8_bwd
    from rtx.kernels.mxfp8_bwd import DEFAULT_SEPARATE_DECOMPOSED_MXFP8_BWD_CONFIG as config

    torch.manual_seed(7606)
    trace = []
    state = {"phase": "setup", "probe_stream": None}

    def status():
        # cuStreamIsCapturing returns Active / Invalidated / None. Unlike a
        # synchronization, asking this of a non-default stream doesn't invalidate
        # capture. The stream is explicit even after the context has unwound.
        from cuda.bindings import driver
        error, value = driver.cuStreamIsCapturing(state["probe_stream"].cuda_stream)
        return value.name if not error else error.name

    def record(event, **fields):
        trace.append(dict(event=event, phase=state["phase"], **fields))

    retained_for_native_control = []

    def install_instrumentation():
        from rtx._routing_graph import ConditionalRoutingGraph
        from rtx._routing_hard import HardRoutingPlan

        release = _resources._release_graph
        synchronize = torch.cuda.synchronize

        def release_wrapper(owners, device):
            record("release_enter", count=len(owners), device=device, capture=status())
            try:
                if args.native_only:
                    # A diagnostic arm, not a candidate fix: keep every owner
                    # until this process reaches its post-capture recovery sync.
                    retained_for_native_control.append(owners)
                    return None
                return release(owners, device)
            finally:
                record("release_exit", count=len(owners), capture=status())

        def sync_wrapper(device=None):
            if state["probe_stream"] is not None:
                record("device_synchronize_enter", device=str(device), capture=status())
            try:
                return synchronize(device)
            except Exception as e:
                record("device_synchronize_error", error_type=type(e).__name__, message=str(e),
                       capture=status())
                raise

        def record_destructor(original, name):
            def wrapper(self):
                if state["probe_stream"] is not None:
                    record(name, capture=status())
                return original(self)
            return wrapper

        _resources._release_graph = release_wrapper
        torch.cuda.synchronize = sync_wrapper
        HardRoutingPlan.__del__ = record_destructor(HardRoutingPlan.__del__, "hard_plan_del")
        ConditionalRoutingGraph.__del__ = record_destructor(
            ConditionalRoutingGraph.__del__, "conditional_routing_graph_del"
        )

    if args.instrument or args.native_only:
        install_instrumentation()

    a_stream, b_stream = torch.cuda.Stream(), torch.cuda.Stream()
    state["probe_stream"] = b_stream
    values = tuple(torch.randn((128, 128), device="cuda", dtype=torch.bfloat16) * .1 for _ in range(3))
    x = torch.randn((64, 128), device="cuda", dtype=torch.bfloat16)
    a_stream.wait_stream(torch.cuda.current_stream())
    b_stream.wait_stream(torch.cuda.current_stream())

    def step():
        return fp8_bwd.mxfp8_linear_backward(*values, config=config, autotune=False)

    with torch.cuda.stream(a_stream):
        expected_a = tuple(out.clone() for out in step())
        graph_a = torch.cuda.CUDAGraph()
        if args.unpinned:
            with torch.cuda.graph(graph_a, stream=a_stream):
                actual_a = (x.square(),)
            expected_a = (x.square(),)
        else:
            with torch.cuda.graph(graph_a, stream=a_stream):
                actual_a = step()
        graph_a.replay()
    a_stream.synchronize()
    for actual, expected in zip(actual_a, expected_a):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    pins = graph_a._rtx_resource_owners
    count, nbytes = len(pins), tensor_storage_bytes(pins)
    assert (not count and not nbytes) if args.unpinned else (count > 0 and nbytes > 0)
    del pins
    graph_ref = weakref.ref(graph_a)
    detached_pins = dict(graph_a._rtx_resource_owners) if args.action == "pins" else None
    holder = [graph_a]
    del graph_a
    record("graph_a_ready", count=count, storage_bytes=nbytes, exact_grads=True)

    def drop():
        if args.action == "weak":
            holder.clear()
        elif args.action == "cycle":
            cycle = [holder.pop()]
            cycle.append(cycle)
            cycle[0]._diagnostic_cycle = cycle
            del cycle
            gc.collect()
        elif args.action == "reset":
            assert graph_ref() is not None
            try:
                holder[0].reset()
            except RuntimeError as e:
                if "reset CUDA graphs outside active CUDA graph capture" not in str(e):
                    raise
                record("reset_rejected", message=str(e))
                assert hasattr(holder[0], "_rtx_resource_owners")
        elif args.action == "pins":
            class Trigger:
                pass
            trigger = Trigger()
            weakref.finalize(trigger, _resources._release_graph, detached_pins, a_stream.device.index)
            del trigger
            record("standalone_pin_count", count=len(detached_pins))
        elif args.action != "none":
            raise AssertionError(args.action)
        record("drop_return", target_gone=graph_ref() is None)

    result = {"scenario": vars(args).copy(), "events": trace, "first_outer_error": None,
              "torch": torch.__version__, "torch_git": torch.version.git_version,
              "force_cudagraph_gc": torch.compiler.config.force_cudagraph_gc,
              "gc_enabled": gc.isenabled(), "gc_threshold": gc.get_threshold(),
              "device": torch.cuda.get_device_name(0),
              "resources_path": str(Path(_resources.__file__).resolve()),
              "resources_sha256": hashlib.sha256(Path(_resources.__file__).read_bytes()).hexdigest()}
    if args.where == "outside":
        state["phase"] = "before_capture"
        drop()
    graph_b = torch.cuda.CUDAGraph()
    b_replay_pass = False
    with torch.cuda.stream(b_stream):
        try:
            state["phase"] = "capture_begin"
            graph_b.capture_begin()
            try:
                state["phase"] = "capture_body"
                actual_b = x.square()
                if args.where == "inside":
                    drop()
            finally:
                state["phase"] = "capture_end"
                graph_b.capture_end()
            state["phase"] = "verify_replay"
            graph_b.replay()
            b_stream.synchronize()
            torch.testing.assert_close(actual_b, x.square(), rtol=0, atol=0)
            b_replay_pass = True
        except Exception as e:
            result["first_outer_error"] = dict(phase=state["phase"], error_type=type(e).__name__,
                                               message=str(e), traceback=traceback.format_exc())
    # Small separate post-error operation checks nonsticky recovery. Do not
    # attempt a driver reset on any failure.
    state["phase"] = "recovery"
    try:
        if args.action == "cycle":
            gc.collect()
        if args.action == "reset" and hasattr(holder[0], "_rtx_resource_owners"):
            holder[0].reset()
        recovery = x + 1
        torch.cuda.synchronize()
        result["device_recovered"] = bool(torch.isfinite(recovery).all().item())
    except Exception as e:
        result["device_recovered"] = False
        result["recovery_error"] = str(e)
    result["graph_b_replay_pass"] = b_replay_pass
    result["graph_a_gone_after_boundary"] = graph_ref() is None
    result["standalone_pins_after_boundary"] = len(detached_pins) if detached_pins is not None else None
    if result["device_recovered"]:
        retained_for_native_control.clear()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("weak", "cycle", "reset", "pins", "none"), required=True)
    parser.add_argument("--where", choices=("inside", "outside"), required=True)
    parser.add_argument("--instrument", action="store_true")
    parser.add_argument("--native-only", action="store_true",
                        help="retain RTX pins without invoking the RTX CUDA callback")
    parser.add_argument("--unpinned", action="store_true",
                        help="warm the RTX runner but capture A with only a Torch square")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(dict(scenario=result["scenario"], recovered=result["device_recovered"],
                          replay=result["graph_b_replay_pass"], error=result["first_outer_error"]),
                     default=str))
    return 0 if result["device_recovered"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
