"""Replay archived packed-weight failures and change only the reciprocal."""
import argparse
from dataclasses import replace
import json
import zipfile
import torch
from rtx.autotune.dataset import _digest
from rtx.autotune.first_hit_study import atomic_json
from rtx.nvfp4_inference_autotune import weight_prequant_config_from_dict
from rtx.nvfp4_inference_experiments import NVFP4WeightPrequantBenchmarkHarness
from rtx.prequant_experiments import ShapeSpec,BenchmarkProtocol


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive');p.add_argument('--output',required=True)
    a=p.parse_args();records=[];seen=set()
    with zipfile.ZipFile(a.archive) as z:
        manifest=json.loads(z.read(next(n for n in z.namelist() if n.endswith('/manifest.json'))))
        for name in z.namelist():
            if not name.endswith('/observations.jsonl') or '/nvfp4_weight_prequant_fwd/' not in name:continue
            for line in z.read(name).splitlines():
                r=json.loads(line)
                if r['outcome']['status']!='correctness_error':continue
                dims=tuple(int(r['features']['context.workload.'+k]) for k in ('m','n','k'))
                if (dims[0],dims[2]) not in ((127,1536),(1009,257),(513,145)):continue
                config=weight_prequant_config_from_dict(r['config'])
                regime='rotate' if '/rotate/' in name else 'hot'
                key=(dims,regime,config.quant_x.scale_reciprocal)
                if key in seen:continue
                seen.add(key);shape=ShapeSpec(*dims)
                seed=manifest['seed'] ^ int(_digest((r['family'],shape.key,regime),8),16)
                # Same first X/W draw as archived harness; no cache-rotation
                # allocation needed for this non-timing first-pair comparison.
                h=NVFP4WeightPrequantBenchmarkHarness(shape,'hot',BenchmarkProtocol(),device='cuda',seed=seed)
                result=dict(shape=dims,seed=seed,archived_regime=regime,config_id=r['config_id'],
                            original_mode=config.quant_x.scale_reciprocal,measurements=[])
                for label,c in [('archived',config),('direct_only',replace(config,quant_x=replace(config.quant_x,scale_reciprocal='direct')))]:
                    runner=h._build_runner(c);out=torch.empty_like(h._expected)
                    runner(h.x,h.weight,out);torch.cuda.synchronize()
                    delta=out.float()-h._expected.float();ref=h._expected.float()
                    result['measurements'].append(dict(variant=label,finite=bool(out.isfinite().all()),
                        max_abs=float(delta.abs().max()),rms_error=float(delta.square().mean().sqrt()),
                        reference_rms=float(ref.square().mean().sqrt()),
                        relative_l2=float(delta.norm()/ref.norm().clamp_min(1e-20)),
                        passes_existing_gate=bool(torch.allclose(out,h._expected,rtol=h.protocol.correctness_rtol,atol=h.protocol.correctness_atol))))
                    del runner,out
                records.append(result);print(result,flush=True);del h
    atomic_json(a.output,dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,records=records,
        oracle='Original baseline schedule output; isolates reciprocal only, not independent correctness proof'))


if __name__=='__main__':main()
