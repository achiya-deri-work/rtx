# Optional CUDA/CUB reference

`routing_cub.cu` and `loader.py` preserve the historical CUDA/CUB implementation
for diagnostics. They are not imported by `rtx` and are not included in its
wheel. Only this optional reference needs a CUDA 13.2 NVCC/CCCL toolkit.

Production order statistics, quota selection and conditional predicates are
CuTe DSL kernels in `rtx/kernels/routing_selection.py`. Graph construction and
submission use `cuda-python` in `rtx/_routing_graph.py`.

The historical reference exports `rtx_routing_quantile` and
`bimoe_quota_apply`; `loader.library()` declares their ctypes signatures. Its
on-chip CUB quantile sorter is limited to 8,192 elements per vector, unlike the
CuTe streaming path. Never silently deploy this reference as a fallback.
