"""Short compiled optimizer lifecycles, not transformer convergence claims."""
import pytest
import torch
import torch.nn.functional as F
import rtx


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", ["mxfp8", "current", "delayed", "jit_row_region", "block", "mlp_silu", "mlp_gelu"])
def test_compiled_bf16_adamw_training(mode):
    torch.manual_seed(3421)
    torch._dynamo.reset()
    if mode.startswith("mlp_"):
        layer = rtx.MXFP8MLP(128, 256, device="cuda", activation=mode[4:], autotune="off")
    elif mode == "mxfp8":
        layer = rtx.MXFP8Linear(128, 128, device="cuda", autotune="off")
    else:
        layer = rtx.NVFP4Linear(128, 128, device="cuda", scaling=mode, autotune="off")
    x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)*.5
    teacher = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)*.03
    target = F.linear(x, teacher).float()
    parameters = list(layer.parameters())
    initial = [p.detach().clone() for p in parameters]
    optimizer = torch.optim.AdamW(parameters, lr=.003, weight_decay=.01, foreach=False)
    forward = torch.compile(layer, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
    losses = []
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        out = forward(x)
        loss = F.mse_loss(out.float(), target)
        assert bool(loss.isfinite())
        loss.backward()
        assert all(p.grad is not None and bool(p.grad.isfinite().all()) for p in parameters)
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0], (mode, losses)
    assert all(not torch.equal(a, p) for a, p in zip(initial, parameters))
    for state in optimizer.state.values():
        assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.bfloat16
    del out, loss, forward, optimizer, layer
    rtx.clear_runtime_caches(device="cuda")
