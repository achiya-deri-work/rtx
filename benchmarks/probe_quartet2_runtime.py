"""Optional upstream runtime probe; run alone in a disposable CUDA process."""
import json
import argparse

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quartet", "rtx"), default="quartet")
    parser.add_argument("--rng-probe", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(20261023)
    if args.mode == "quartet":
        from quartet2.linear import Quartet_II_linear
        layer = Quartet_II_linear(256, 256, bias=False, device="cuda", dtype=torch.bfloat16)
    else:
        import rtx
        layer = rtx.NVFP4Linear(256, 256, device="cuda", autotune="off",
                               backward_config=rtx.DEFAULT_MXFP8_BWD_CONFIG)
    print(json.dumps({"kind": "mode", "mode": args.mode}), flush=True)
    x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn_like(x)
    if args.rng_probe:
        from quartet2.quant import rht128_quant_eden
        # Fixed rotation isolates micro-scale RNG from Hadamard sign RNG.
        def quant():
            return rht128_quant_eden(x=x.detach(), h=layer.had[:16, :])
        for _ in range(3):
            quant()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            packed = quant()
        scales = []
        for _ in range(16):
            graph.replay()
            scales.append(packed.micro_scales.view(torch.uint8).clone())
        torch.cuda.synchronize()
        print(json.dumps({"kind": "fixed_rotation_graph_rng",
                          "distinct_scale_samples": len({v.cpu().numpy().tobytes() for v in scales}),
                          "replays": 16}), flush=True)
        torch.cuda.manual_seed(123456)
        graph.replay()
        seeded = packed.micro_scales.view(torch.uint8).clone()
        torch.cuda.manual_seed(123456)
        graph.replay()
        torch.cuda.synchronize()
        print(json.dumps({"kind": "graph_rng_reseed",
                          "reproducible": torch.equal(seeded, packed.micro_scales.view(torch.uint8))}), flush=True)
        return

    def step():
        y = layer(x)
        dx, dw = torch.autograd.grad(y, (x, layer.weight), dy)
        return y, dx, dw

    for _ in range(3):
        reference = step()[0].detach().clone()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = step()[0]
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    print(json.dumps({"kind": "side_stream_smoke",
                      "forward_equal": torch.equal(actual, reference),
                      "note": "smoke only; not a race-freedom proof"}), flush=True)
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = step()
        graph.replay()
        torch.cuda.synchronize()
        first = [v.clone() for v in outputs]
        graph.replay()
        torch.cuda.synchronize()
        print(json.dumps({"kind": "graph_replay",
                          "finite": [bool(v.isfinite().all()) for v in outputs],
                          "changed": [not torch.equal(a, b) for a, b in zip(first, outputs)],
                          "note": "changed gradients alone do not validate all RNG streams"}), flush=True)
        with torch.no_grad():
            x.mul_(2)
        graph.replay()
        torch.cuda.synchronize()
        replay_y = outputs[0].clone()
        expected_y = layer(x).detach()
        torch.cuda.synchronize()
        print(json.dumps({"kind": "graph_changed_input",
                          "forward_changed": not torch.equal(replay_y, first[0]),
                          "matches_eager": torch.equal(replay_y, expected_y),
                          "normalized_rms": float((replay_y.float()-expected_y.float()).square().mean().sqrt()
                                                   / expected_y.float().square().mean().sqrt())}), flush=True)
    except Exception as exc:
        print(json.dumps({"kind": "graph_error", "error": f"{type(exc).__name__}: {exc}"}), flush=True)
        raise


if __name__ == "__main__":
    main()
