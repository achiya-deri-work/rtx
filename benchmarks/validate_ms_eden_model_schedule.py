"""Fixed-draw and independent-FP32 checks for the model scheduling change."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--designs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    import torch
    from rtx.autotune.ms_eden_design import MSEdenDesign
    from benchmarks.ms_eden_design_dispatch import dispatch
    from rtx.experimental_ms_eden import ms_eden_paired_gradient
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(21941)
    records=[]
    for key,raw in json.loads(args.designs.read_text()).items():
        m,n,k=map(int,key.split(','))
        dy=torch.randn(m,n,device='cuda',dtype=torch.bfloat16)*.25
        x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)*.25
        w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*.02
        signs=[(torch.randint(2,(128,),device='cuda')*2-1).float() for _ in range(2)]
        uniforms=[torch.rand(count//16,device='cuda') for count in (dy.numel(),w.numel(),dy.numel(),x.numel())]
        values=(dy,x,w,signs[0],uniforms[0],uniforms[1],signs[1],uniforms[2],uniforms[3])
        design=MSEdenDesign(**raw)
        kwargs={f'{phase}_schedule_json':json.dumps(asdict(getattr(design.schedule,phase))) for phase in ('dx','dw')}
        with dispatch(design):
            def run(pipeline):return ms_eden_paired_gradient(*values,pipeline=pipeline,**kwargs)
            expected=run('serial')
            actual=run('paired_dw_first')
            for a,b in zip(expected,actual):torch.testing.assert_close(a,b,atol=0,rtol=0)
            references=(dy.float()@w.float(),dy.float().T@x.float())
            metrics=[]
            for a,r in zip(actual,references):
                error=a.float()-r
                nrms=float(error.norm()/r.norm().clamp_min(1e-12))
                assert bool(a.isfinite().all()) and nrms<.25
                metrics.append(dict(nrms=nrms,max_abs=float(error.abs().max())))
            del references,error
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):run('paired_dw_first')
            stream.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):captured=run('paired_dw_first')
            graph.replay()
            for a,b in zip(expected,captured):torch.testing.assert_close(a,b,atol=0,rtol=0)
            dy.mul_(.5);x.add_(.01);w.mul_(.875)
            for value in uniforms:value.uniform_()
            graph.replay()
            changed=run('serial')
            for a,b in zip(changed,captured):torch.testing.assert_close(a,b,atol=0,rtol=0)
            dy.zero_();graph.replay()
            for value in captured:assert not bool(value.count_nonzero())
            torch.cuda.synchronize()
            before=torch.cuda.memory_allocated()
            for _ in range(10):graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated()==before
            records.append(dict(shape=key,byte_exact=True,changed_input_uniform_replay=True,
                zero_gradients=True,replay_live_growth_bytes=0,fp32_metrics=metrics))
        del expected,actual,changed,captured,graph,values,dy,x,w,uniforms,signs
        gc.collect();torch.cuda.synchronize()
        print(records[-1],flush=True)
    with args.output.open('x') as f:json.dump(dict(records=records),f,indent=2)


if __name__=='__main__':main()
