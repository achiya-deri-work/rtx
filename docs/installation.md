# Installation

These instructions describe this checkout's supported stack, not arbitrary
newer dependency versions. The distribution name is `rtx-blackwell`; the Python
namespace is `rtx`. Install from the repository for the feature-branch APIs.

## Requirements

| Component | Contract |
|---|---|
| OS / GPU | Linux; RTX Blackwell SM120/SM121 |
| Python | Metadata requires >=3.11; documented qualification targets 3.11–3.13 |
| Torch | 2.13.0+cu132 or 2.12.1+cu132 |
| TorchAO | 0.18.0 |
| CuTe DSL | `nvidia-cutlass-dsl[cu13]==4.7.0` |
| FFI | `apache-tvm-ffi==0.1.13.post2` |
| Other runtime dependencies | NumPy 2.x, CUDA Python 13.x, PyArrow >=16, Einops >=0.8,<0.9 |

A compatible NVIDIA driver must expose CUDA to the process. Runtime validation
checks the pinned Torch/CUDA, TorchAO, and DSL versions; importing `rtx` alone
does not prove the native execution environment is usable.

## Preferred environment

From a repository checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m pip check
```

For Torch 2.12.1, use a separate environment and substitute
`requirements-torch212-cu132.txt` for `requirements.txt`. Do not install the two
Torch stacks into the same environment.

The requirements files select the CUDA-local wheel from the CUDA 13.2 PyTorch
index. Package metadata alone cannot make a bare `pip install rtx-blackwell`
select the correct CUDA wheel. Install the supported Torch stack first.
`setup.py` is a compatibility shim; `pyproject.toml` owns packaging metadata.

For development, install `requirements-dev.txt`. Optional torchvision test
dependencies are listed under the `vision` extra in `pyproject.toml`; they are
not required for linear execution. Benchmarks, docs, tests and manifests live
in the checkout and are not installed as Python runtime packages in the wheel.

## Verify before tuning

```bash
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available())'
python -c 'import rtx; print(rtx.__version__); print(rtx.validate_runtime_environment())'
rtx-autotune probe
python docs/examples/basic_workflows.py --mode training
```

The last command runs actual CUDA work and may compile kernels on first use.
The examples use `autotune="cache"` to avoid starting an unexpected search.
For toolkit diagnostics, `CUDA_TOOLKIT_PATH` should identify the intended CUDA
toolkit. Do not force a datacenter architecture to bypass an RTX rejection.

## Updating

Use a cleanly recorded source revision on every measurement machine. Reinstall
dependencies if the requirement files changed, then run validation again.
Kernel/runtime-cache revisions can invalidate old winners; this is intentional.
Keep original BF16 checkpoints for repacking. Do not delete old dataset journals
merely because a new kernel revision cannot use their winners.

See [troubleshooting](troubleshooting.md) for NVVM failures and
[autotuning](autotuning.md) for safe dataset resume and winner deployment.
