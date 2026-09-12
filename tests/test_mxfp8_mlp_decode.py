"""Exhaustive finite E4M3 / E8M0 coverage for the packed hidden decoder."""
import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class PackedHiddenDecodeTests(unittest.TestCase):
    def test_all_finite_values_and_scale_exponents(self):
        import cutlass
        import cutlass.cute as cute
        from cutlass import BFloat16, Int16, Int32, Uint16, Uint32
        from cutlass.experimental.primitives import nvvm_wrapper as nvvm

        class Decode:
            @cute.jit
            def __call__(self, data, scales, out, stream):
                self.kernel(data, scales, out).launch(
                    grid=(cute.ceil_div(cute.size(scales),256),1,1),
                    block=(256,1,1),stream=stream)

            @cute.kernel
            def kernel(self, data, scales, out):
                i=cute.arch.block_idx()[0]*256+cute.arch.thread_idx()[0]
                if i<cute.size(scales):
                    packed=Int16(Int32(data[i,0].bitcast(cutlass.Uint8))|(Int32(data[i,1].bitcast(cutlass.Uint8))<<8))
                    decoded=nvvm.inline_ptx_hl("cvt.rn.bf16x2.e4m3x2 {$w0}, {$r0};",
                                              write_only_types=[Int32],read_only_args=[packed])
                    bits=Uint32(BFloat16(scales[i]).bitcast(Uint16))
                    result=Uint32(nvvm.mul_bf16x2(decoded,Int32(bits|(bits<<16))))
                    out[i,0]=Uint16(result).bitcast(BFloat16)
                    out[i,1]=Uint16(result>>16).bitcast(BFloat16)

        raw=torch.arange(256,device="cuda",dtype=torch.int32).to(torch.uint8)
        raw=raw[raw.view(torch.float8_e4m3fn).float().isfinite()]
        pairs=torch.stack((raw,raw.flip(0)),dim=1).repeat(255,1).contiguous()
        scales=torch.exp2(torch.arange(-127,128,device="cuda",dtype=torch.float32)).repeat_interleave(raw.numel())
        data=pairs.view(torch.float8_e4m3fn)
        expected=(data.float()*scales[:,None]).bfloat16()
        out=torch.empty_like(expected)
        fn=cute.compile(Decode(),cute.runtime.from_dlpack(data),cute.runtime.from_dlpack(scales),
                        cute.runtime.from_dlpack(out),
                        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                        options="--enable-tvm-ffi --opt-level 3")
        fn(data,scales,out)
        torch.testing.assert_close(out,expected,rtol=0,atol=0)


if __name__ == "__main__":
    unittest.main()
