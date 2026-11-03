"""Research GEMM: finalize corrected BF16 scales at the point of use.

Explicit caller uniforms preserve the same draw at every repeated CTA load.
Global maxima must already be reduced. No RNG or outer-factor approximation.
"""
from functools import lru_cache
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float4E2M1FN, Float8E4M3FN, Float32, Int16, Int32, Uint8
from cutlass.experimental.primitives import nvvm_wrapper as nvvm
from .nvfp4_gemm import NVFP4GemmKernel
from .nvfp4_posthoc import PosthocScaleKernel, PosthocScaleConfig
from ._layout_contracts import preserve_tma_singletons


class InlineScaleGemm(NVFP4GemmKernel):
    def __init__(self, problem, config, probability_math='divide'):
        if config.scale_role not in ('producer','consumers') or config.scale_layout!='row_major' or config.scale_load_vec!=1:
            raise ValueError('inline scales require scalar row-major producer/consumer staging')
        super().__init__(problem,config,outer_factor_mode='pair')
        self.finalizer=PosthocScaleKernel(problem.m,problem.k//16,
            PosthocScaleConfig(probability_math=probability_math))

    @cute.jit
    def _outer(self, maximum):
        value=Float32(1.)
        if maximum[0]>Float32(0.):value=maximum[0]*Float32(1./448.)
        return value

    @cute.jit
    def _load_global_output_scale(self, output_scale):
        return nvvm.inline_ptx_hl('mul.rn.f32 {$w0}, {$r0}, {$r1};',
            write_only_types=[Float32],read_only_args=[self._outer(output_scale[0]),self._outer(output_scale[1])])

    @cute.jit
    def _load_scale_element(self,source,row,group,is_a:cutlass.Constexpr,output_scale):
        maximum=output_scale[0] if cutlass.const_expr(is_a) else output_scale[1]
        uniforms=output_scale[2] if cutlass.const_expr(is_a) else output_scale[3]
        scale=self._outer(maximum)
        value=cute.arch.fmin(source[row,group].to(Float32)/scale,Float32(448.))
        packed=nvvm.inline_ptx_hl('cvt.rn.satfinite.e4m3x2.f32 {$w0}, {$r0}, {$r0};',
            write_only_types=[Int16],read_only_args=[value])
        code=Int32(packed&Int16(255))
        nearest=Uint8(code).bitcast(Float8E4M3FN).to(Float32)
        lo,hi=code,code
        if nearest>value:lo=code-1
        if nearest<value:hi=code+1
        low=Uint8(lo).bitcast(Float8E4M3FN).to(Float32)
        high=Uint8(hi).bitcast(Float8E4M3FN).to(Float32)
        probability=self.finalizer.rounding_probability(value,low,high)
        selected=lo
        if uniforms[row*(self.problem.k//16)+group]<probability:selected=hi
        return Uint8(selected).bitcast(Float8E4M3FN)


@lru_cache(maxsize=64)
def compile_inline_scale_gemm(problem,config,probability_math='divide'):
    kernel=InlineScaleGemm(problem,config,probability_math)
    def operand(rows):
        return cute.runtime.make_fake_tensor(Float4E2M1FN,
            preserve_tma_singletons((rows,problem.k)),stride=(problem.k,1),assumed_align=16)
    def fake(dtype,shape,stride,align=16):
        return cute.runtime.make_fake_tensor(dtype,shape,stride=stride,assumed_align=align)
    groups=problem.k//16
    sx=fake(BFloat16,(problem.m,groups),(groups,1))
    sw=fake(BFloat16,(problem.n,groups),(groups,1))
    out=fake(BFloat16,(problem.m,problem.n),(problem.n,1))
    metadata=(fake(Float32,(1,),(1,),4),fake(Float32,(1,),(1,),4),
        fake(Float32,(problem.m*groups,),(1,)),fake(Float32,(problem.n*groups,),(1,)))
    return cute.compile(kernel,operand(problem.m),operand(problem.n),sx,sw,out,metadata,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options='--enable-tvm-ffi --opt-level 3 '+f"--ptxas-options '-O3 -v --maxrregcount={config.maxrregcount}'")
