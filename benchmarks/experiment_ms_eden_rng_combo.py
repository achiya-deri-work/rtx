"""Complete-backward experiment with scoped Inductor RNG combo fusion.

No runtime defaults are changed. Run probe_ms_eden_rng_combo first to qualify
identical draws and RNG consumption on the installed compiler version.
"""


def main():
    import torch._inductor.config as config
    from benchmarks.compare_ms_eden_pipeline import main as compare
    with config.patch(combo_kernels=True, combo_kernels_pointwise_only=True):
        compare()


if __name__ == "__main__":
    main()
