"""Runnable CUDA examples: --mode training|inference|mlp|all.

Run from the checkout after installing rtx-blackwell. No online search is
started. These examples test API usage, not performance or convergence.
"""
import argparse

import torch
import rtx


def training():
    for convert,policy in (
        (rtx.convert_to_mxfp8_training,rtx.MXFP8TrainingConfig(autotune="cache")),
        (rtx.convert_to_nvfp4_training,rtx.NVFP4TrainingConfig(autotune="cache")),
    ):
        model=torch.nn.Sequential(
            torch.nn.Linear(128,256,bias=False),torch.nn.SiLU(),
            torch.nn.Linear(256,128,bias=False),
        ).to(device="cuda",dtype=torch.bfloat16)
        original=list(model.parameters())
        model=convert(model,config=policy)
        assert all(a is b for a,b in zip(original,model.parameters()))
        fn=torch.compile(model,fullgraph=True,dynamic=False)
        x=torch.randn(32,128,device="cuda",dtype=torch.bfloat16,requires_grad=True)
        fn(x).float().square().mean().backward()
        for tensor in (x,*model.parameters()):
            assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
        print(type(model[0]).__name__,"compiled backward passed",flush=True)


def inference():
    for cls,quantize in ((rtx.MXFP8Linear,rtx.quantize_mxfp8),
                         (rtx.NVFP4Linear,rtx.quantize_nvfp4)):
        layer=cls(128,256,device="cuda",autotune="cache").eval()
        x=torch.randn(32,128,device="cuda",dtype=torch.bfloat16)
        with torch.inference_mode():
            dynamic_y=layer(x)
            packed=layer.to_quantized_weight()
            packed_y=torch.compile(packed,fullgraph=True,dynamic=False)(x)
            fully_y=packed(quantize(x))
            for y in (dynamic_y,packed_y,fully_y):
                assert y.shape==(32,256) and y.dtype==torch.bfloat16
                assert torch.isfinite(y).all()
            restored=layer.to_quantized_weight()
            restored.load_state_dict(packed.state_dict())
            torch.testing.assert_close(restored(x),packed(x),rtol=0,atol=0)
        print(cls.__name__,"three operand states and checkpoint passed",flush=True)


def mlp():
    projections=[torch.nn.Linear(i,o,bias=False,device="cuda",dtype=torch.bfloat16)
                 for i,o in ((128,256),(128,256),(256,128))]
    layer=rtx.MXFP8MLP.from_linears(*projections,activation="silu")
    policy=rtx.MXFP8MLPConfig.from_json(layer.config.to_json())
    assert policy==layer.config
    x=torch.randn(32,128,device="cuda",dtype=torch.bfloat16,requires_grad=True)
    y=torch.compile(layer,fullgraph=True,dynamic=False)(x)
    y.float().square().mean().backward()
    for tensor in (x,*layer.parameters()):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
    print("MXFP8MLP conversion, policy and compiled backward passed",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("training","inference","mlp","all"),default="all")
    args=parser.parse_args()
    rtx.validate_runtime_environment()
    torch.manual_seed(12)
    for name,function in (("training",training),("inference",inference),("mlp",mlp)):
        if args.mode in (name,"all"):
            function()


if __name__=="__main__":
    main()
