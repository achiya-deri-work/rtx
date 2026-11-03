"""Research-only complete backward comparison for scale finalizer variants.

Requires test_ms_eden_paired_finalizer GPU qualification before timing. Stable
runtime defaults are unchanged; explicit RNG and original BF16 only.
"""
import gc
import json


def main():
    import torch._inductor.config as compiler_config
    from benchmarks import compare_ms_eden_pipeline as comparison
    from benchmarks.experiment_ms_eden_operand_order import exact_smoke
    from rtx.kernels import nvfp4_posthoc as kernels
    from rtx.kernels.ms_eden_paired_finalizer import compile_paired_scales
    parser = comparison.parser()
    parser.add_argument('--finalizer',choices=('paired','cta'),default='paired')
    args = parser.parse_args()
    if args.candidate_pipeline != 'operand_paired' or args.baseline_policy != 'stable_mxfp8':
        parser.error('requires operand_paired candidate and stable_mxfp8 baseline')
    if args.validate_exact:
        parser.error('full-shape exact smoke always precedes this experiment')
    if args.output.exists():
        raise FileExistsError(args.output)
    shape = comparison.shape_value(args.shape)
    common, dx, dw = [json.loads(value) for value in
        (args.candidate_json,args.candidate_dx_json,args.candidate_dw_json)]
    schedules = comparison.schedule_pair(common, dx, dw)
    if args.finalizer == 'cta' and any(json.loads(value)['amax_reduction'] != 'inline' for value in schedules.values()):
        parser.error('CTA reduction experiment requires explicit inline amax in both schedules')
    previous = kernels.compile_posthoc_scales
    try:
        if args.finalizer == 'cta':
            from rtx.kernels.ms_eden_cta_finalizer import compile_cta_scales
            kernels.compile_posthoc_scales = compile_cta_scales
        else:
            kernels.compile_posthoc_scales = compile_paired_scales
        with compiler_config.patch(combo_kernels=True,combo_kernels_pointwise_only=True):
            checks = exact_smoke(shape,schedules,args.seed)
            gc.collect()
            result = comparison.compare(shape,args.output,common,dx_overrides=dx,dw_overrides=dw,
                pipeline=args.candidate_pipeline,rounds=args.rounds,replays=args.replays,
                min_batch_ms=args.min_batch_ms,warmup_ms=args.warmup_ms,seed=args.seed,
                baseline_policy=args.baseline_policy,mxfp8_config_json=args.mxfp8_config_json,
                scopes=('backward','total') if args.scope=='both' else (args.scope,))
    finally:
        kernels.compile_posthoc_scales = previous
    # Companion metadata does not overwrite the immutable comparison result.
    with args.output.with_suffix('.qualification.json').open('x') as handle:
        json.dump(dict(same_draw_exact=checks,finalizer=args.finalizer,
            compiler_options=dict(combo_kernels=True,combo_kernels_pointwise_only=True)),handle,indent=2)
    print(json.dumps(result['timings'],indent=2))


if __name__ == '__main__':
    main()
