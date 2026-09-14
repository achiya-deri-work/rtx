"""Optimizer trajectories through compiled first-hit selection and cache reuse."""
import json
from pathlib import Path
import tempfile
import unittest
import torch
import rtx


class TrainingLoss(torch.nn.Module):
    def __init__(self,linear):
        super().__init__();self.linear=linear

    def forward(self,x,target):
        y=self.linear(x)
        return y,.5*(y.float()-target.float()).square().sum(dim=-1).mean()


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class FirstHitTrainingCudaTests(unittest.TestCase):
    def test_optimizer_trajectories_and_warm_cache(self):
        for family,scaling in (('mxfp8',None),('nvfp4','block'),('nvfp4','current'),
                               ('nvfp4','jit_row_region'),('nvfp4','delayed')):
            with self.subTest(family=family,scaling=scaling):
                torch.compiler.reset()
                self._case(family,scaling)

    def _case(self,family,scaling):
        from torch._inductor.compile_fx import compile_fx
        torch.manual_seed(4139)
        m,k,n=256,128,128
        policy=rtx.AutotunePolicy(max_trials=8,max_seconds=90,search_schedule='progressive',
            cache_behavior='refine',confirmation_repeats=0,finalist_rounds=2,
            warmup=1,samples=3,calls_per_sample=2)
        with tempfile.TemporaryDirectory() as root:
            cls=rtx.MXFP8Linear if family=='mxfp8' else rtx.NVFP4Linear
            kwargs={} if scaling is None else dict(scaling=scaling)
            layer=cls(k,n,device='cuda',backend='materialized',tuning_policy=policy,
                      autotune_cache_dir=root,**kwargs)
            control=torch.nn.Linear(k,n,bias=False,device='cuda',dtype=torch.bfloat16)
            with torch.no_grad():control.weight.copy_(layer.weight)
            x=(torch.randn(m,k,device='cuda',dtype=torch.bfloat16)*.1).requires_grad_()
            teacher=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*.02
            target=(x.detach().float()@teacher.float().T).bfloat16()
            graphs=[]
            def backend(graph,inputs):
                graphs.append(graph);return compile_fx(graph,inputs)
            compiled=torch.compile(TrainingLoss(layer),backend=backend,fullgraph=True,dynamic=False)
            reference=TrainingLoss(control)
            optimizer=torch.optim.SGD(layer.parameters(),lr=1.)
            control_optimizer=torch.optim.SGD(control.parameters(),lr=1.)
            records=None;graph_count=None;losses=[];control_losses=[];metrics=[]
            for step in range(16):
                optimizer.zero_grad(set_to_none=True);control_optimizer.zero_grad(set_to_none=True);x.grad=None
                y,loss=compiled(x,target)
                _,control_loss=reference(x.detach(),target)
                loss.backward();control_loss.backward()
                with torch.no_grad():
                    dy=((y.float()-target.float())/m).bfloat16()
                    refs=((x.float()@layer.weight.float().T).bfloat16(),
                          (dy.float()@layer.weight.float()).bfloat16(),
                          (dy.float().T@x.float()).bfloat16())
                    for label,actual,expected in zip(('y','dx','dw'),(y,x.grad,layer.weight.grad),refs):
                        self.assertEqual(actual.dtype,torch.bfloat16)
                        self.assertTrue(bool(actual.isfinite().all()),(step,label))
                        delta=actual.float()-expected.float()
                        rms=float(delta.square().mean().sqrt());ref_rms=float(expected.float().square().mean().sqrt())
                        error=rms/max(ref_rms,1e-12)
                        gate=.20 if family=='nvfp4' and label=='y' else .07
                        metrics.append(dict(step=step,output=label,rms=rms,reference_rms=ref_rms,
                                            normalized_rms=error,max_abs=float(delta.abs().max()),gate=gate))
                        self.assertLess(error,gate,(step,label))
                losses.append(float(loss.detach()));control_losses.append(float(control_loss.detach()))
                optimizer.step();control_optimizer.step()
                self.assertEqual(layer.weight.dtype,torch.bfloat16)
                current={str(p):p.read_bytes() for p in Path(root).rglob('observations.jsonl')}
                if records is None:
                    self.assertEqual(len(current),2)
                    self.assertTrue(all(len(v.splitlines())==8 for v in current.values()))
                    records=current;graph_count=len(graphs)
                else:
                    self.assertEqual(current,records,'optimizer update retuned')
                    self.assertEqual(len(graphs),graph_count,'optimizer update recompiled')
            common_evaluation={}
            with torch.no_grad():
                generator=torch.Generator(device='cuda').manual_seed(9217)
                heldout=torch.randn(m,k,device='cuda',dtype=torch.bfloat16,generator=generator)*.1
                for label,inputs in (('training',x),('heldout',heldout)):
                    labels=(inputs.float()@teacher.float().T).bfloat16()
                    values=[]
                    for weights in (layer.weight,control.weight):
                        predicted=(inputs.float()@weights.float().T).bfloat16()
                        values.append(float(.5*(predicted.float()-labels.float()).square().sum(-1).mean()))
                    common_evaluation[label]=dict(candidate_loss=values[0],bf16_control_loss=values[1])
                    self.assertLess(values[0],1.1*values[1],label)
            print('TRAINING',json.dumps(dict(family=family,scaling=scaling,shape=[m,n,k],seed=4139,
                steps=16,optimizer='SGD',lr=1.,losses=losses,bf16_losses=control_losses,metrics=metrics,
                common_evaluation=common_evaluation,heldout_seed=9217,
                graphs=graph_count,torch=str(torch.__version__),cuda=torch.version.cuda,
                device=torch.cuda.get_device_name())),flush=True)
            self.assertLess(losses[-1],.9*losses[0])
            self.assertLess(control_losses[-1],.9*control_losses[0])
            self.assertLess(losses[-1],1.1*control_losses[-1])


if __name__=='__main__':unittest.main()
