"""Matched ~21M-parameter TinyStories precision study, one policy per process.

Only block projections change precision. CPU initialization is shared, norms
and residual additions are FP32, and attention backend is explicitly pinned.
Training wall includes data movement/forward/backward/clipping/AdamW, excludes
compile, validation, checks, logging and checkpoints. This is not certification.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from torch import nn

from rtx.models import DecoderConfig, DecoderOnlyTransformer, LinearSpec, causal_lm_loss
from rtx.models.decoder import FP32RMSNorm, QKRMSNorm

MODES=("mxfp8","nvfp4_mxfp8","nvfp4_full")


class FP32ResidualBlock(nn.Module):
    def __init__(self, original):
        super().__init__()
        for name in ("input_layernorm","self_attn","post_attention_layernorm",
                     "pre_feedforward_layernorm","mlp","post_feedforward_layernorm"):
            setattr(self,name,getattr(original,name))

    def forward(self,x):
        branch=self.post_attention_layernorm(self.self_attn(self.input_layernorm(x)))
        x=(x.float()+branch.float()).to(torch.bfloat16)
        branch=self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(x)))
        return (x.float()+branch.float()).to(torch.bfloat16)


class PrecisionProjection(nn.Module):
    def __init__(self,original,mode,m,sm_count,producer_load=None,producer_blocks=None,
                 rng_mode=None,producer_signs=None,backward_pipeline="serial"):
        super().__init__()
        import rtx
        from rtx.autotune.posthoc import preparation_resource_seeds
        from rtx.nvfp4_inference_autotune import preferred_jit_row_region_config
        if original.bias is not None or mode not in MODES:
            raise ValueError("bias-free projection and explicit supported mode required")
        self.weight=original.weight
        self.mode=mode
        self.backward_pipeline=backward_pipeline
        self.shape=(m,original.out_features,original.in_features)
        self.scale_config=rtx.NVFP4ScaleConfig(tensor_scale_mode="exact")
        self.forward_config=replace(preferred_jit_row_region_config(rtx.NVFP4Problem(*self.shape)),
                                    tensor_scale_mode="exact")
        # Explicit qualified wide-loader seed; these are recorded, not tuned here.
        self.dx_schedule=next(c for c in preparation_resource_seeds(self.shape,"dx",sm_count)
                              if c.producer_load=="coalesced_wide")
        self.dw_schedule=next(c for c in preparation_resource_seeds(self.shape,"dw",sm_count)
                              if c.producer_load=="coalesced_wide")
        if producer_load is not None:
            self.dx_schedule=replace(self.dx_schedule,producer_load=producer_load)
            self.dw_schedule=replace(self.dw_schedule,producer_load=producer_load)
        if producer_blocks is not None:
            self.dx_schedule=replace(self.dx_schedule,producer_blocks=producer_blocks)
            self.dw_schedule=replace(self.dw_schedule,producer_blocks=producer_blocks)
        overrides={key:value for key,value in (("rng_mode",rng_mode),("producer_signs",producer_signs))
                   if value is not None}
        self.dx_schedule=replace(self.dx_schedule,**overrides)
        self.dw_schedule=replace(self.dw_schedule,**overrides)
        self.dx_json=json.dumps(asdict(self.dx_schedule),sort_keys=True)
        self.dw_json=json.dumps(asdict(self.dw_schedule),sort_keys=True)

    def forward(self,x):
        import rtx
        shape=x.shape
        flat=x.reshape(-1,shape[-1]).contiguous()
        if self.mode=="mxfp8":
            out=rtx.mxfp8_linear(flat,self.weight,autotune="off",
                                backward_config=rtx.DEFAULT_MXFP8_BWD_CONFIG)
        elif self.mode=="nvfp4_mxfp8":
            out=rtx.nvfp4_linear(flat,self.weight,scaling="jit_row_region",autotune="off",
                dynamic_config=self.forward_config,
                scale_config=self.scale_config,
                backward_config=rtx.DEFAULT_MXFP8_BWD_CONFIG)
        else:
            from rtx.experimental_ms_eden import ms_eden_linear
            out=ms_eden_linear(flat,self.weight,source_policy="original_bf16",
                forward_scaling="jit_row_region",forward_outer_mode="exact",
                forward_config=self.forward_config,dx_schedule_json=self.dx_json,
                dw_schedule_json=self.dw_json,backward_pipeline=self.backward_pipeline)
        return out.reshape(*shape[:-1],self.weight.shape[0])


def model_config(args):
    return DecoderConfig(vocab_size=257,max_seq_len=args.sequence,num_layers=args.layers,
        hidden_size=args.width,intermediate_size=args.intermediate,
        num_attention_heads=args.heads,dtype=torch.bfloat16,
        linear=LinearSpec(precision="bf16"),tie_word_embeddings=True,
        initializer_range=.02,residual_init_multiplier=2.,attention_dropout=0.)


def build_model(config,mode,batch,sm_count=1,producer_load=None,producer_blocks=None,
                rng_mode=None,producer_signs=None,backward_pipeline="serial",schedule_map=None):
    """Always initialize on CPU first; replacement never resets any weight."""
    model=DecoderOnlyTransformer(config,device="cpu")
    model.layers=nn.ModuleList(FP32ResidualBlock(layer) for layer in model.layers)
    for module in model.modules():
        if isinstance(module,(FP32RMSNorm,QKRMSNorm)):
            module.float()
    for name,module in list(model.named_modules()):
        if name.startswith("layers.") and isinstance(module,nn.Linear):
            parent,leaf=name.rsplit(".",1)
            setattr(model.get_submodule(parent),leaf,
                    PrecisionProjection(module,mode,batch*config.max_seq_len,sm_count,producer_load,
                                        producer_blocks,rng_mode,producer_signs,backward_pipeline))
    if schedule_map is not None:
        if mode != "nvfp4_full":
            raise ValueError("schedule map requires nvfp4_full")
        projections=[module for module in model.modules() if isinstance(module,PrecisionProjection)]
        required={",".join(map(str,module.shape)) for module in projections}
        if set(schedule_map)!=required:
            raise ValueError(f"schedule map must exactly cover model shapes; missing={sorted(required-set(schedule_map))}, extra={sorted(set(schedule_map)-required)}")
        from rtx.autotune.ms_eden_pipeline import MSEdenPipelineSchedule
        for module in projections:
            selected=MSEdenPipelineSchedule(**schedule_map[",".join(map(str,module.shape))])
            module.dx_schedule, module.dw_schedule = selected.dx, selected.dw
            module.backward_pipeline=selected.backward_pipeline
            module.dx_json=json.dumps(asdict(selected.dx),sort_keys=True)
            module.dw_json=json.dumps(asdict(selected.dw),sort_keys=True)
    return model


def _load_unique_map(path):
    def unique_pairs(pairs):
        result={}
        for key,value in pairs:
            if key in result:
                raise ValueError(f"duplicate schedule key: {key}")
            result[key]=value
        return result
    raw=json.loads(Path(path).read_text(),object_pairs_hook=unique_pairs)
    if not isinstance(raw,dict) or not raw:
        raise ValueError("schedule map must be a nonempty object")
    return raw


def load_schedule_map(path):
    """Explicit per-shape research schedules, never implicit runtime winners."""
    from rtx.autotune.ms_eden_pipeline import MSEdenPipelineSchedule
    raw=_load_unique_map(path)
    result={}
    for key,value in raw.items():
        try:
            shape=tuple(int(part) for part in key.split(","))
        except ValueError as exc:
            raise ValueError("schedule keys must be canonical M,N,K") from exc
        if len(shape)!=3 or any(v<=0 or v%128 for v in shape) or key!=",".join(map(str,shape)):
            raise ValueError("schedule keys must be canonical positive 128-aligned M,N,K")
        if not isinstance(value,dict) or set(value)!={"dx","dw","backward_pipeline"}:
            raise ValueError("each schedule requires explicit dx, dw, backward_pipeline")
        result[key]=asdict(MSEdenPipelineSchedule(**value))
    return result


def load_design_map(path):
    """Exact designs; scoped compile patches require one shared implementation."""
    from rtx.autotune.ms_eden_design import MSEdenDesign, make_ms_eden_design_adapter
    result={}
    for key,value in _load_unique_map(path).items():
        try:
            shape=tuple(int(part) for part in key.split(','))
        except ValueError as exc:
            raise ValueError('design keys must be canonical M,N,K') from exc
        if len(shape)!=3 or key!=','.join(map(str,shape)) or any(v<=0 or v%128 for v in shape):
            raise ValueError('design keys must be canonical positive 128-aligned M,N,K')
        if not isinstance(value,dict) or set(value)!={'schedule','transpose_loader','finalizer'}:
            raise ValueError('each design requires explicit schedule, transpose_loader and finalizer')
        if not isinstance(value['schedule'],dict) or set(value['schedule'])!={'dx','dw','backward_pipeline'}:
            raise ValueError('each design schedule requires explicit dx, dw, backward_pipeline')
        design=MSEdenDesign(**value)
        adapter=make_ms_eden_design_adapter(shape,lambda _:None,allow_row_major=True)
        rejection=adapter.rejection(design)
        if rejection:
            raise ValueError(f'{key}: {rejection[1]}')
        result[key]=asdict(design)
    implementations={(v['transpose_loader'],v['finalizer']) for v in result.values()}
    if len(implementations)!=1:
        raise ValueError('training dispatch requires one shared loader/finalizer implementation')
    return result


def verify_design_dispatch(args):
    designs=getattr(args,'ms_eden_design_map',None)
    if designs is None:
        return
    loader=next(iter(designs.values()))['transpose_loader']
    records=getattr(args,'ms_eden_dispatch_records',[])
    transposes=[record for record in records if record.get('transpose') is True]
    if loader!='native' and (not transposes or any(record.get('selected')!=loader for record in transposes)):
        raise RuntimeError('training warmup did not establish the requested TMA implementation')


def parameter_hash(model):
    digest=hashlib.sha256()
    for name,param in model.named_parameters():
        digest.update(name.encode())
        digest.update(str(param.dtype).encode())
        digest.update(param.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def source_hashes():
    root=Path(__file__).resolve().parents[1]
    names=("benchmarks/train_precision20m.py","rtx/models/decoder.py","rtx/fp8.py",
        "rtx/fp8_bwd.py","rtx/fp4.py","rtx/experimental_ms_eden.py",
        "rtx/kernels/nvfp4_posthoc.py","rtx/kernels/nvfp4_rng.py",
        "rtx/kernels/nvfp4_gemm.py","rtx/kernels/mxfp8_gemm.py",
        "rtx/autotune/posthoc.py")
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}


def configure_attention(mode):
    if mode not in ("flash","math"):
        raise ValueError("attention must be explicitly flash or math")
    torch.backends.cuda.enable_flash_sdp(mode=="flash")
    torch.backends.cuda.enable_math_sdp(mode=="math")
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False


def gradient_norm_and_clip(grads,max_norm=1.):
    # dtype=FP32 performs cast+reduction, not BF16 reduction followed by a cast.
    norm=torch.linalg.vector_norm(torch.stack([
        torch.linalg.vector_norm(g,2,dtype=torch.float32) for g in grads]),2)
    coefficient=(max_norm/(norm+1e-6)).clamp(max=1.)
    torch._foreach_mul_(grads,coefficient)
    return norm


def learning_rate(step,total,warmup,maximum,minimum_ratio):
    if warmup and step<warmup:
        return maximum*(step+1)/warmup
    progress=(step-warmup)/max(total-warmup-1,1)
    return maximum*(minimum_ratio+(1-minimum_ratio)*.5*(1+math.cos(math.pi*progress)))


class Corpus:
    def __init__(self,path,limit,seed,batch,sequence,validation_batches):
        self.tokens=np.load(path,mmap_mode="r")[:limit]
        if self.tokens.ndim!=1 or not np.issubdtype(self.tokens.dtype,np.integer):
            raise ValueError("expected one-dimensional integer byte-token corpus")
        if self.tokens.size==0 or self.tokens.min()<0 or self.tokens.max()>256:
            raise ValueError("byte tokens must be in [0,256]")
        self.seed,self.batch,self.sequence=seed,batch,sequence
        self.split=int(len(self.tokens)*.9)
        if min(self.split,len(self.tokens)-self.split)<=sequence+1:
            raise ValueError("corpus is too small for disjoint train/validation ranges")
        rng=np.random.default_rng(seed+100_000_000)
        self.validation_offsets=rng.integers(self.split,len(self.tokens)-sequence,
            size=(validation_batches,batch),dtype=np.int64)
        self.sha256=hashlib.sha256(self.tokens.tobytes()).hexdigest()
        self.validation_sha256=hashlib.sha256(self.validation_offsets.tobytes()).hexdigest()

    def offsets(self,step):
        return np.random.default_rng(self.seed+step*0x9E3779B1).integers(
            0,self.split-self.sequence,size=self.batch,dtype=np.int64)

    def batch_at(self,offsets,device="cuda"):
        host=np.stack([self.tokens[i:i+self.sequence+1] for i in offsets]).astype(np.int64)
        data=torch.from_numpy(host).to(device)
        return data[:,:-1].contiguous(),data[:,1:].contiguous()


def rng_state():
    return dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),
                numpy=np.random.get_state(),python=random.getstate())


def restore_rng(state):
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state(state["cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--mode",choices=MODES,required=True)
    p.add_argument("--sku",default="")
    p.add_argument("--attention",choices=("flash","math"),default="flash")
    p.add_argument("--cpu-threads",type=int,default=4)
    for name,default in (("steps",2048),("batch",16),("sequence",256),("layers",8),
        ("width",512),("intermediate",1024),("heads",8),("seed",941),
        ("validation-interval",128),("validation-batches",32),("log-interval",128),
        ("checkpoint-interval",512),("corpus-tokens",10_000_000),("warmup-steps",128)):
        p.add_argument("--"+name,type=int,default=default)
    p.add_argument("--eval-every",type=int,dest="validation_interval",default=argparse.SUPPRESS)
    p.add_argument("--eval-batches",type=int,dest="validation_batches",default=argparse.SUPPRESS)
    p.add_argument("--learning-rate",type=float,default=3e-4)
    p.add_argument("--minimum-lr-ratio",type=float,default=.1)
    p.add_argument("--resume",type=Path)
    p.add_argument("--cudagraphs",action="store_true",help="opt-in; qualify replay/RNG before use")
    p.add_argument("--combo-kernels",action="store_true",
                   help="opt-in compiler pointwise/RNG fusion; recorded in checkpoint identity")
    p.add_argument("--profile-step",type=int,default=-1,
                   help="zero-based update to capture with cudaProfilerApi; diagnostic timing only")
    p.add_argument("--ms-eden-producer-load",choices=("direct","coalesced","coalesced_wide","coalesced_x4","coalesced_x8"),
                   help="explicit experimental preparation geometry; arithmetic policy unchanged")
    p.add_argument("--ms-eden-producer-blocks",type=int,
                   help="explicit producer CTA cap; recorded in the run identity")
    p.add_argument("--ms-eden-rng-mode",choices=("explicit","philox"),
                   help="optional fused uniform RNG; changes random sequence, not precision policy")
    p.add_argument("--ms-eden-producer-signs",choices=("reload","register"),
                   help="optional exact register reuse of the fixed Hadamard signs")
    p.add_argument("--ms-eden-backward-pipeline",choices=("serial","paired","operand_paired"),default="serial",
                   help="experimental FP4 backward overlap: paired gradients or four operand preparations; default serial")
    p.add_argument("--ms-eden-schedules",type=Path,
                   help="JSON map of canonical M,N,K to dx/dw/backward_pipeline; exact model coverage required")
    p.add_argument("--ms-eden-designs",type=Path,
                   help="exact per-shape MSEdenDesign map; shared loader/finalizer and exact model coverage required")
    p.add_argument("--ms-eden-research-variant",choices=("retained","tma_dual"),default="retained",
                   help="opt-in TMA XOR5 plus dual finalization; requires an explicit schedule map")
    return p


def validate_args(args):
    for name in ("steps","batch","sequence","layers","width","intermediate","heads",
                 "validation_interval","validation_batches","log_interval","checkpoint_interval","corpus_tokens","cpu_threads"):
        if getattr(args,name)<=0:
            raise ValueError(f"{name} must be positive")
    if args.width%128 or args.intermediate%128 or args.batch*args.sequence%128:
        raise ValueError("projection M/N/K must be divisible by128")
    if not 0<=args.warmup_steps<args.steps or args.learning_rate<=0 or not 0<=args.minimum_lr_ratio<=1:
        raise ValueError("invalid learning-rate schedule")
    model_config(args)
    if args.profile_step != -1 and not 0 <= args.profile_step < args.steps:
        raise ValueError("profile-step must be -1 or an executed update")
    if args.ms_eden_producer_blocks is not None and args.ms_eden_producer_blocks<1:
        raise ValueError("MS-EDEN producer blocks must be positive")
    if any(getattr(args,key) is not None for key in ("ms_eden_producer_load","ms_eden_producer_blocks",
                                                   "ms_eden_rng_mode","ms_eden_producer_signs")) and args.mode!="nvfp4_full":
        raise ValueError("MS-EDEN schedule override requires nvfp4_full")
    if args.ms_eden_backward_pipeline!="serial" and args.mode!="nvfp4_full":
        raise ValueError("MS-EDEN backward pipeline override requires nvfp4_full")
    args.ms_eden_schedule_map=None
    args.ms_eden_design_map=None
    if args.ms_eden_designs is not None:
        if (args.mode!='nvfp4_full' or args.ms_eden_schedules is not None
                or args.ms_eden_research_variant!='retained'
                or args.ms_eden_backward_pipeline!='serial'
                or any(getattr(args,key) is not None for key in (
                    'ms_eden_producer_load','ms_eden_producer_blocks','ms_eden_rng_mode','ms_eden_producer_signs'))):
            raise ValueError('design map requires nvfp4_full and cannot be combined with other schedule overrides')
        args.ms_eden_design_map=load_design_map(args.ms_eden_designs)
        args.ms_eden_schedule_map={key:value['schedule'] for key,value in args.ms_eden_design_map.items()}
    if args.ms_eden_research_variant != "retained" and (args.mode != "nvfp4_full" or args.ms_eden_schedules is None):
        raise ValueError("research variant requires nvfp4_full and an explicit schedule map")
    if args.ms_eden_schedules is not None:
        if args.mode!="nvfp4_full":
            raise ValueError("schedule map requires nvfp4_full")
        if args.ms_eden_backward_pipeline!="serial" or any(getattr(args,key) is not None for key in (
                "ms_eden_producer_load","ms_eden_producer_blocks","ms_eden_rng_mode","ms_eden_producer_signs")):
            raise ValueError("schedule map cannot be combined with individual overrides")
        args.ms_eden_schedule_map=load_schedule_map(args.ms_eden_schedules)
        if args.ms_eden_research_variant == "tma_dual":
            for entry in args.ms_eden_schedule_map.values():
                if entry['backward_pipeline'] != 'operand_paired' or any(
                        entry[phase]['amax_reduction'] != 'staged' or entry[phase]['rng_mode'] != 'explicit'
                        for phase in ('dx','dw')):
                    raise ValueError("tma_dual requires operand_paired, staged maxima and explicit uniforms")


def ms_eden_identity_overrides(args):
    """Keep legacy serial checkpoint identity; pin every explicit experiment."""
    values={key:getattr(args,key) for key in (
        "ms_eden_producer_load","ms_eden_producer_blocks","ms_eden_rng_mode","ms_eden_producer_signs")
        if getattr(args,key) is not None}
    if args.ms_eden_backward_pipeline!="serial":
        values["ms_eden_backward_pipeline"]=args.ms_eden_backward_pipeline
    if getattr(args,"ms_eden_schedule_map",None) is not None:
        values["ms_eden_schedule_map"]=args.ms_eden_schedule_map
    if getattr(args,'ms_eden_design_map',None) is not None:
        values['ms_eden_design_map']=args.ms_eden_design_map
        root=Path(__file__).resolve().parents[1]
        values['design_dispatch_source_sha256']={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
            for name in ('benchmarks/ms_eden_design_dispatch.py',
                'benchmarks/experiment_ms_eden_bank_layout.py','benchmarks/experiment_ms_eden_dual_finalizer.py',
                'rtx/kernels/ms_eden_tma_preparation.py','rtx/kernels/ms_eden_dual_finalizer.py')}
    if getattr(args,"combo_kernels",False):
        values["compiler_options"]=training_compiler_options(args)
    if getattr(args,"ms_eden_research_variant","retained") != "retained":
        root=Path(__file__).resolve().parents[1]
        values["research_variant"]=args.ms_eden_research_variant
        values["research_source_sha256"]={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
            for name in ("rtx/kernels/ms_eden_tma_preparation.py","rtx/kernels/ms_eden_dual_finalizer.py",
                         "benchmarks/experiment_ms_eden_bank_layout.py","benchmarks/experiment_ms_eden_dual_finalizer.py")}
    return values


def training_compiler_options(args):
    options={"triton.cudagraphs":args.cudagraphs}
    if getattr(args,"combo_kernels",False):
        options.update(combo_kernels=True,combo_kernels_pointwise_only=True)
    return options


def run(args):
    validate_args(args)
    from contextlib import ExitStack
    with ExitStack() as contexts:
        if getattr(args,'ms_eden_design_map',None) is not None:
            from benchmarks.ms_eden_design_dispatch import dispatch
            from rtx.autotune.ms_eden_design import MSEdenDesign
            design=MSEdenDesign(**next(iter(args.ms_eden_design_map.values())))
            args.ms_eden_dispatch_records=contexts.enter_context(dispatch(design))
        if args.ms_eden_research_variant == "tma_dual":
            from benchmarks.experiment_ms_eden_bank_layout import patched
            from benchmarks.experiment_ms_eden_dual_finalizer import patched as dual_patch
            contexts.enter_context(patched('tma_xor5_1'))
            contexts.enter_context(dual_patch('both'))
        return _run_validated(args)


def _run_validated(args):
    torch.set_num_threads(args.cpu_threads)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    started=time.perf_counter()
    props=torch.cuda.get_device_properties(0)
    if props.major!=12:
        raise RuntimeError("SM120/SM121 required")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    checkpoint_dir=args.output.with_suffix(".checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    interrupted=[False]
    def request_stop(signum,frame):
        interrupted[0]=True
    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig,request_stop)
    with args.output.open("x") as stream:
        def emit(event,**values):
            stream.write(json.dumps(dict(event=event,**values),allow_nan=False)+"\n")
            stream.flush()
        configure_attention(args.attention)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        corpus=Corpus(args.tokens,args.corpus_tokens,args.seed,args.batch,args.sequence,args.validation_batches)
        config=model_config(args)
        model=build_model(config,args.mode,args.batch,props.multi_processor_count,
                          args.ms_eden_producer_load,args.ms_eden_producer_blocks,
                          args.ms_eden_rng_mode,args.ms_eden_producer_signs,args.ms_eden_backward_pipeline,
                          args.ms_eden_schedule_map)
        initial_hash=parameter_hash(model)
        model.cuda()
        if model.embed_tokens.weight is not model.lm_head.weight:
            raise RuntimeError("CUDA conversion broke tied embeddings")
        norm_params=[p for p in model.parameters() if p.dtype==torch.float32]
        matrix_params=[p for p in model.parameters() if p.dtype==torch.bfloat16]
        optimizer=torch.optim.AdamW([dict(params=matrix_params,weight_decay=.1),
            dict(params=norm_params,weight_decay=0.)],lr=args.learning_rate,betas=(.9,.95),
            eps=1e-8,fused=True)
        projection_configs={str(module.shape):dict(forward=asdict(module.forward_config),
            dx=asdict(module.dx_schedule),dw=asdict(module.dw_schedule),
            backward_pipeline=module.backward_pipeline)
            for module in model.modules() if isinstance(module,PrecisionProjection)}
        config_doc=asdict(config)
        config_doc["dtype"]=str(config.dtype)
        optimizer_doc=dict(name="AdamW",betas=[.9,.95],eps=1e-8,learning_rate=args.learning_rate,
            weight_decay=.1,norm_weight_decay=0.,warmup_steps=args.warmup_steps,
            minimum_lr_ratio=args.minimum_lr_ratio,fused=True,
            states="BF16 matrix/embed parameters; FP32 normalization parameters")
        identity=dict(mode=args.mode,seed=args.seed,params=model.num_parameters(),batch=args.batch,
            sequence=args.sequence,steps=args.steps,model_config=config_doc,optimizer=optimizer_doc,
            initial_weights_sha256=initial_hash,corpus_sha256=corpus.sha256,
            validation_offsets_sha256=corpus.validation_sha256)
        identity.update(ms_eden_identity_overrides(args))
        step=0
        training_wall=0.
        prior_end_to_end=0.
        if args.resume:
            state=torch.load(args.resume,map_location="cpu",weights_only=False)
            if state["identity"]!=identity:
                raise ValueError("resume mode/architecture/data/seed/schedule identity mismatch")
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            step=state["step"]
            training_wall=state["training_wall_s"]
            prior_end_to_end=state["end_to_end_wall_s"]
            restore_rng(state["rng"])
        def elapsed():
            return prior_end_to_end+time.perf_counter()-started
        emit("environment",**identity,sku=args.sku or props.name,device=str(props),
            torch=torch.__version__,cuda=torch.version.cuda,attention=args.attention,
            source_sha256=source_hashes(),projection_configs=projection_configs,
            total_train_tokens=args.steps*args.batch*args.sequence,
            resume=str(args.resume) if args.resume else None,
            precision=dict(matrix_parameters="BF16",normalization_parameters="FP32",
                norm_reductions="FP32",rope="FP32",residual_add="FP32 then BF16",
                swiglu="BF16",loss="FP32",grad_norm="FP32",attention_qkv="BF16",
                attention_softmax="FP32 statistics/exponentiation; BF16 probability cast for FLASH PV",
                cudagraphs=args.cudagraphs),
            wall_contract="Synchronized training blocks include data movement and optimizer; exclude compilation, validation, checkpoint, logging and outside-block checks.")
        def objective(x,y):
            return causal_lm_loss(model(x),y)
        options=training_compiler_options(args)
        compiled=torch.compile(objective,fullgraph=True,dynamic=False,options=options)
        clip=torch.compile(gradient_norm_and_clip,fullgraph=True,dynamic=False,options=options)
        before_warmup=rng_state()
        torch.cuda.synchronize()
        warmup_start=time.perf_counter()
        torch.compiler.cudagraph_mark_step_begin()
        loss=compiled(*corpus.batch_at(corpus.offsets(step)))
        loss.backward()
        clip([p.grad for p in model.parameters() if p.requires_grad])
        optimizer.zero_grad(set_to_none=True)
        del loss
        with torch.no_grad():
            torch.compiler.cudagraph_mark_step_begin()
            compiled(*corpus.batch_at(corpus.validation_offsets[0]))
        torch.cuda.synchronize()
        verify_design_dispatch(args)
        if getattr(args,'ms_eden_design_map',None) is not None:
            emit('design_dispatch',records=args.ms_eden_dispatch_records)
        restore_rng(before_warmup)
        emit("compile_warmup",compile_warmup_wall_s=time.perf_counter()-warmup_start,
            end_to_end_wall_s=elapsed(),optimizer_steps=0,
            scope="First train loss/backward/FP32clip and no-grad validation compile; RNG restored, no optimizer update")
        last_validation=None
        def validation():
            nonlocal last_validation
            preserved=rng_state()
            torch.cuda.synchronize()
            begin=time.perf_counter()
            try:
                with torch.no_grad():
                    values=torch.empty(len(corpus.validation_offsets),device="cuda",dtype=torch.float32)
                    for index,offsets in enumerate(corpus.validation_offsets):
                        torch.compiler.cudagraph_mark_step_begin()
                        value=compiled(*corpus.batch_at(offsets))
                        values[index].copy_(value)
                        del value
                    last_validation=float(values.float().mean())
                if not math.isfinite(last_validation):
                    raise RuntimeError("nonfinite validation loss")
            finally:
                restore_rng(preserved)
            torch.cuda.synchronize()
            emit("validation",step=step,tokens=step*args.batch*args.sequence,
                validation_loss=last_validation,training_wall_s=training_wall,
                end_to_end_wall_s=elapsed(),validation_wall_s=time.perf_counter()-begin)
        def checkpoint():
            torch.cuda.synchronize()
            begin=time.perf_counter()
            path=checkpoint_dir/f"step-{step:06d}.pt"
            if path.exists():
                raise FileExistsError(path)
            temporary=path.with_suffix(".pt.partial")
            torch.save(dict(identity=identity,model=model.state_dict(),optimizer=optimizer.state_dict(),
                rng=rng_state(),step=step,training_wall_s=training_wall,
                end_to_end_wall_s=elapsed()),temporary)
            temporary.replace(path)
            emit("checkpoint",step=step,training_wall_s=training_wall,end_to_end_wall_s=elapsed(),
                checkpoint_wall_s=time.perf_counter()-begin,path=str(path))
        try:
            validation()
            while step<args.steps and not interrupted[0]:
                count=min(args.log_interval,args.steps-step,
                    args.validation_interval-step%args.validation_interval,
                    args.checkpoint_interval-step%args.checkpoint_interval)
                losses=torch.empty(count,device="cuda",dtype=torch.float32)
                norms=torch.empty_like(losses)
                torch.cuda.synchronize()
                block_start=time.perf_counter()
                first_step=step
                for index in range(count):
                    profiling=step==args.profile_step
                    if profiling:
                        torch.cuda.synchronize()
                        torch.cuda.cudart().cudaProfilerStart()
                    torch.compiler.cudagraph_mark_step_begin()
                    optimizer.zero_grad(set_to_none=True)
                    lr=learning_rate(step,args.steps,args.warmup_steps,args.learning_rate,args.minimum_lr_ratio)
                    for group in optimizer.param_groups:
                        group["lr"]=lr
                    loss=compiled(*corpus.batch_at(corpus.offsets(step)))
                    loss.backward()
                    norm=clip([p.grad for p in model.parameters() if p.requires_grad])
                    optimizer.step()
                    losses[index].copy_(loss.detach())
                    norms[index].copy_(norm.detach())
                    del loss,norm
                    if profiling:
                        torch.cuda.synchronize()
                        torch.cuda.cudart().cudaProfilerStop()
                    step+=1
                    if interrupted[0]:
                        break
                torch.cuda.synchronize()
                block_wall=time.perf_counter()-block_start
                training_wall+=block_wall
                actual=step-first_step
                history=torch.stack((losses[:actual],norms[:actual])).cpu().tolist()
                if not all(math.isfinite(v) for row in history for v in row):
                    raise RuntimeError("nonfinite block loss or gradient norm")
                for offset,(loss_value,norm_value) in enumerate(zip(*history),start=1):
                    emit("train_step",step=first_step+offset,loss=loss_value,grad_norm=norm_value,
                         tokens=(first_step+offset)*args.batch*args.sequence)
                block_tokens=actual*args.batch*args.sequence
                emit("train",step=step,tokens=step*args.batch*args.sequence,loss=history[0][-1],
                    grad_norm=history[1][-1],loss_mean=float(np.mean(history[0])),lr=lr,
                    training_wall_s=training_wall,end_to_end_wall_s=elapsed(),block_steps=actual,
                    block_tokens=block_tokens,block_training_wall_s=block_wall,
                    block_tokens_per_s=block_tokens/block_wall)
                # Outside measured blocks, audit all required gradients/state dtypes.
                for name,param in model.named_parameters():
                    if param.grad is None or not bool(param.grad.isfinite().all()):
                        raise RuntimeError(f"missing/nonfinite gradient: {name}")
                    for key in ("exp_avg","exp_avg_sq"):
                        if optimizer.state[param][key].dtype!=param.dtype:
                            raise RuntimeError(f"optimizer state dtype mismatch: {name} {key}")
                if step%args.validation_interval==0 or step==args.steps:
                    validation()
                if step%args.checkpoint_interval==0 or step==args.steps or interrupted[0]:
                    checkpoint()
            if interrupted[0]:
                # Stop can also arrive between blocks.
                if not (checkpoint_dir/f"step-{step:06d}.pt").exists():
                    checkpoint()
            torch.cuda.synchronize()
            finite=all(bool(p.isfinite().all()) for p in model.parameters())
            emit("result",step=step,tokens=step*args.batch*args.sequence,training_wall_s=training_wall,
                end_to_end_wall_s=elapsed(),validation_loss=last_validation,finite_parameters=finite,
                interrupted=interrupted[0])
            if not finite:
                raise RuntimeError("nonfinite final parameters")
        except Exception as exc:
            emit("error",step=step,error=f"{type(exc).__name__}: {exc}",
                 training_wall_s=training_wall,end_to_end_wall_s=elapsed())
            raise


def main():
    run(parser().parse_args())


if __name__=="__main__":
    main()
