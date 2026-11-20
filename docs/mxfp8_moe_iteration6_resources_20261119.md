# Iteration 6: CUDA graph retirement

Two independent teardown operations can invalidate a capture on HOME GPU 0:
RTX's weak pin finalizer calling device synchronization, and PyTorch's native
graph executable destruction. The fix delays both for **tracked public
PyTorch graphs**, preserves scratch until previous replays finish, and leaves
replay untouched. This proves the forced mechanisms, not which one caused the
intermittent iteration-5 packed factorial failure.

## What failed, with controls

In a fresh process on RTX 5090 rank 0 (Torch 2.13.0+cu132), graph A ran a
warmed real RTX MXFP8 backward and matched eager dX/dW exactly. It owned one
runner with 67,584 explicit scratch bytes. Graph B captured on a different
stream on the same device. The diagnostic observed B's driver capture status
without querying completion or adding synchronization inside capture.

| Action on A or its pins | Original runtime | Fixed runtime |
| --- | --- | --- |
| Drop last caller reference inside B | Native `cuGraphExecDestroy` invalidates B | B replays; A is released after capture |
| Collect a graph-owning cycle inside B | RTX synchronization enters at ACTIVE, fails and invalidates B | B replays; cycle and pins retire |
| Invoke a separate weak pin finalizer inside B | Sync fails; pins cleared despite unknown completion | Pins stay owned until the safe boundary; B replays |
| Explicit A reset inside B | Native destruction invalidates B | Rejected before native reset; B replays, A resets outside |
| Drop/reset outside B; no drop inside B | B replays | B replays |

With RTX's callback replaced by **host-only pin retention**, dropping or
resetting A still failed. The first failing teardown API in the driver log was
`cuGraphExecDestroy` (CUDA error 900, capture unsupported); B's
`cuStreamEndCapture` returned 901 (capture invalidated). This also happened
for a Torch-only square graph with zero RTX pins. PyTorch's
[native reset](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/cuda/CUDAGraph.cpp#L317-L339)
destroys the native executable independently of the RTX callback. Neither
RTX routing destructor ran in these controls. The original failed scenarios
recovered on a separate CUDA operation; this was not a sticky device fault.
No routing destructor was changed.

Torch graph-entry GC was **false** in this environment
(`torch.compiler.config.force_cudagraph_gc`); automatic cyclic GC remained
enabled. The earlier successful benchmark-only GC-isolation rerun is a
diagnostic observation, not part of this implementation.

## What changed

- [Graph resource tracking](../rtx/_resources.py) takes strong leases on
  already-live registered peer graphs only across a tracked capture window.
  The registry is still weak outside that window. This protects native
  teardown even when a peer has no RTX pins. Explicit reset during tracked
  capture raises before touching the native graph.
- Weak finalizers queue owners. After the last tracked capture has really
  ended, peer leases are released and pending pins retire behind a device
  completion barrier, covering replay on streams other than the capture
  stream. A failed barrier retains pins for retry. A wrong-stream
  `capture_end` leaves leases intact until the real boundary. The host mutex
  protects ownership transitions only; native destruction, synchronization
  and collection occur without holding it.
- If Python collected while peer leases were held, an additional safe-boundary
  collection can reclaim cycles that the lease kept reachable. Nothing
  disables GC, changes its thresholds, or hooks graph replay. The extra
  collection is conditional on a collection actually occurring in the lease
  window, and honors application-disabled GC.
- The expanded default-GC stress initially exposed a separate native
  allocator abort in `MemPool` destruction. A CPU sentinel isolated an RTX
  source of delayed ownership: recursive scratch-tagging closed over its
  `MemPool` even when it tagged no tensors. [Scratch tagging](../rtx/_workspace.py)
  now breaks that transient function cycle in a `finally`. It changes neither
  real tensor-owned pool lifetime nor stream cache eviction.

The scope is warm RTX runners and the existing public PyTorch capture
integration, including its use by Inductor. This does not make arbitrary
external CUDA teardown safe or add support for abandoned raw CUDA captures.
There is no caller-owned cleanup requirement, replay synchronization, new
search axis, cache enlargement, or precision or kernel change.

## Qualification and artifacts

Artifact root: `/tmp/rtx-moe-iteration6-20261119/resources/`.

- Baseline runtime: `759c030b5867e029ebdd40def2bac8d8ebee82154075d06a2a79c31b324a79b9`.
  The `baseline-*-instrumented.json`, `baseline-*-native.{json,log}` and
  `baseline-weak-inside-plain-*.json` files include both failure mechanisms,
  controls and plain repetition. The intermediate allocator abort is
  preserved as `candidate3-focused.log`.
- Final owner runtime: `1fe29c84cb28bf0f5e654097424d1b6eb030e07aa93126d2b0645e9880d09ab2`
  (`candidate4/`). Its [focused test suite](../tests/test_moe_graph_resource_lifetime.py)
  plus existing resource/cleanup tests passed **33/33** in the CUDA-enabled
  run; CPU-only **16 passed, 17 CUDA skipped**. In three plain fresh
  processes the original weak drop failed 3/3 and final passed 3/3. All
  seven final instrumented controls passed (`final-candidate4-*.json`).
  In all 24 repeated reciprocal graph
  cycles, automatic GC ran inside capture and synchronized live memory
  returned to **105,472 bytes**, with **zero** retired graphs still reachable
  and **zero** pending owner groups at every boundary. The loop did not call
  `gc.collect()` or alter GC thresholds. Raw samples and JUnit:
  `candidate4-focused.{log,xml}`.
- Two real live tiny MXFP8 forward graphs passed exact replay. For
  the normal peer drop, the native destructor had already completed its late
  replay when RTX reached its barrier. The independent pin-only test kept
  the native graph live, moved its sole scratch ownership out of it, and
  confirmed the event on a **different** replay stream was still pending
  when the noncapturing barrier ran; pins were alive then and gone afterward.
  Wrong-stream end, normal body exception, reset rejection/retirement and
  retry after unproved completion also passed.
- One fresh packed **B8×1088, D384/E128/K6/H192/S384** run used the existing
  progressive eight-trial expert search (active width `6*192+384=4*384`).
  All eight journal rows were successful and stationary. Search invocation
  took **38.255 s**, including setup; context
  `79e75148632f3531eafcbf1f`, policy path
  `0c99f34ac4fa0cddbd3daa6a`, selected configuration
  `43b81af494c74d065a663eaa` (kernel revision 15). Compiled whole-step fresh
  and captured outputs/all optimizer-visible gradients agreed exactly on
  two replays, and the FP64 operational expert reference's worst NRMS was
  `3.3043e-5`. Live bytes over five warmed repetitions were
  **329,286,656** throughout. `cold-b8x1088/` retains the raw journal and
  selected runtime-winner record. The cold search ran on the graph fix
  before the scratch closure correction (runtime
  `71b0beaf556b213da70e574778df73c6dd4a85b91854ad19998a5afc1945320a`);
  `final-b8x1088/result.json` rechecks the exact selected config on the
  final version, with no new trials. All three configurations, seed/fixture
  initialization, hard IDs/inverse hashes, the validation metrics and the
  live-byte plateau matched.

The original packed four-native + three-BF16 composition passed with
default GC, matched inputs/hard routes, all 393 output/input/parameter
fields and two changed-dY replays per cell on the final runtime
(`../integration/packed-default-gc.json` relative to the artifact root).
The combined control was **3.758230 ms** vs its prior **3.760056 ms**,
with compact-checkpoint BF16 **3.736676 ms** and saved BF16 **4.325201 ms**.
The bounded 2×3 integration visits are a stability check, not a speedup
claim or proof of iteration 5's cause.

To rerun a forced control, use the [diagnostic](../benchmarks/diagnose_moe_graph_resources.py)
with the relevant frozen source (flags: `--action weak|cycle|pins|reset|none`,
`--where inside|outside`). Omit `--instrument` for the plain confirmation;
`--native-only` retains pins in the diagnostic rather than calling RTX's
finalizer, and `--unpinned` uses the Torch-only square for graph A.

```bash
base=/tmp/rtx-moe-iteration6-20261119
repo=/home/achiya/vs_code_main/rtx
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$base/baseline" \
  flock /tmp/rtx-moe-backward-20261119-gpu0.lock timeout 60 \
  "$repo/.venv/bin/python" "$repo/benchmarks/diagnose_moe_graph_resources.py" \
  --where inside --action cycle --instrument --output "$base/resources/repro-cycle.json"
cd "$base/resources/candidate4"
CUDA_VISIBLE_DEVICES=0 flock /tmp/rtx-moe-backward-20261119-gpu0.lock timeout 120 \
  "$repo/.venv/bin/python" -m pytest -q -s tests/test_moe_graph_resource_lifetime.py \
  tests/test_execution_resources.py tests/test_memory_cleanup_protocol.py
```
