# Changelog

All notable public API and dataset-schema changes are recorded here. This
project follows semantic versioning while it is in active alpha development.

## 0.3.0

- Rename the distribution and public package identity to `rtx`.
- Add the public `rtx.MXFP8Linear` and `rtx.NVFP4Linear` frontends.
- Add versioned `MXFP8Tensor` and `NVFP4Tensor` packed-operand contracts.
- Support dynamic BF16/BF16, dynamic-X/prequantized-weight, and fully
  prequantized MXFP8 forward states.
- Keep NVFP4 execution explicitly unavailable until its RTX kernel exists,
  while registering its forward, fake, context, and MXFP8-backward boundary.
- Make frontend and autotuner imports architecture neutral; SM120 CuTe modules
  are selected lazily at kernel compilation.
- Preserve v1/v2 autotuning family names and resume identities.
