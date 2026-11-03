"""Matched-forward conditional gradient estimator study; not unbiasedness proof."""
import argparse
import json
from pathlib import Path

import torch


class Moments:
    """FP64 online moments with explicit finite Monte Carlo uncertainty."""
    def __init__(self, reference):
        self.reference = reference.double()
        self.mean = torch.zeros_like(self.reference)
        self.m2 = torch.zeros_like(self.reference)
        self.count = 0
        self.first_error_squared = None

    def add(self, sample):
        sample = sample.double()
        if sample.shape != self.reference.shape or not torch.isfinite(sample).all():
            raise ValueError("invalid gradient sample")
        self.count += 1
        if self.count == 1:
            self.first_error_squared = (sample-self.reference).square().sum()
        delta = sample-self.mean
        self.mean.add_(delta/self.count)
        self.m2.add_(delta*(sample-self.mean))

    def report(self):
        if self.count < 2:
            raise ValueError("at least two repetitions required")
        error = self.mean-self.reference
        denominator = self.reference.square().sum().clamp_min(1e-24)
        variance = self.m2.sum()/(self.count-1)
        mean_error = error.square().sum()
        return dict(samples=self.count, finite=True,
            nonfinite_count=0,
            reference_rms=self.reference.square().mean().sqrt().item(),
            mean_error_rms=error.square().mean().sqrt().item(),
            mean_error_max_abs=error.abs().max().item(),
            mean_error_normalized_max=(error.abs().max()/self.reference.abs().max().clamp_min(1e-12)).item(),
            worst_flat_index=error.abs().flatten().argmax().item(),
            mean_error_nrms=(mean_error/denominator).sqrt().item(),
            first_draw_nrms=(self.first_error_squared/denominator).sqrt().item(),
            per_draw_nrms_rms=((mean_error+self.m2.sum()/self.count)/denominator).sqrt().item(),
            sample_std_nrms=(variance/denominator).sqrt().item(),
            mean_standard_error_nrms=(variance/self.count/denominator).sqrt().item(),
            bias_squared_noise_corrected_nrms2=((mean_error-variance/self.count)/denominator).item(),
            signed_gain_error=((error*self.reference).sum()/denominator).item(),
            floor=1e-12,
            interpretation="Conditional sample mean error, not proven bias; noise-corrected squared bias may be negative.")


def run(shape, distribution, samples=32, seed=807):
    import rtx
    from benchmarks.study_ms_eden_backward import study_operand
    from rtx.experimental_ms_eden import ms_eden_linear
    if samples < 2 or any(size <= 0 or size % 128 for size in shape):
        raise ValueError("samples >=2 and positive dimensions divisible by128 required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(seed)
    m,n,k = shape
    x = study_operand(m,k,.25,distribution).requires_grad_()
    w = study_operand(n,k,.02,distribution).requires_grad_()
    dy = study_operand(m,n,.25,distribution)
    # FP64 semantic GEMMs independent of preparation, rotation, and FP4 GEMM.
    references = (dy.double()@w.double(), dy.double().T@x.double())
    results = {}
    expected = None
    for dx_policy,dw_policy in (("mxfp8","mxfp8"),("ms_eden","ms_eden"),
                                ("mxfp8","ms_eden"),("ms_eden","mxfp8")):
        torch.manual_seed(seed+1000)
        moments = [Moments(ref) for ref in references]
        checkpoints = []
        for index in range(samples):
            y = ms_eden_linear(x,w,source_policy="original_bf16",
                forward_scaling="jit_row_region",forward_outer_mode="exact",
                mxfp8_backward_config=rtx.DEFAULT_MXFP8_BWD_CONFIG if "mxfp8" in (dx_policy,dw_policy) else None,
                dx_policy=dx_policy,dw_policy=dw_policy)
            if not torch.isfinite(y).all():
                raise RuntimeError("nonfinite forward")
            if expected is None:
                expected = y.detach().clone()
            else:
                torch.testing.assert_close(y,expected,rtol=0,atol=0)
            grads = torch.autograd.grad(y,(x,w),dy)
            for moment,gradient in zip(moments,grads):
                moment.add(gradient)
            count = index+1
            if count >= 2 and (count & (count-1) == 0 or count == samples):
                checkpoints.append(dict(samples=count,dx=moments[0].report(),dw=moments[1].report()))
        results[f"dx_{dx_policy}__dw_{dw_policy}"] = checkpoints
    return dict(shape=list(shape),distribution=distribution,seed=seed,samples=samples,
        source_policy="original_bf16",forward="identical NVFP4 jit_row_region exact FP32 outer scaling; bitwise checked",
        oracle="FP64 eager semantic GEMM of fixed BF16 original operands and upstream gradient",
        results=results,torch=torch.__version__,cuda=torch.version.cuda,
        device=str(torch.cuda.get_device_properties(0)),
        caveat="One fixed operand draw; repetitions resample signs/scales, not data. No unbiasedness or training qualification claim.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shape",type=int,nargs=3,default=(128,128,128))
    p.add_argument("--distribution",choices=("gaussian","sparse","outliers"),default="gaussian")
    p.add_argument("--samples",type=int,default=32)
    p.add_argument("--seed",type=int,default=807)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    result=run(tuple(args.shape),args.distribution,args.samples,args.seed)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result,handle,indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
