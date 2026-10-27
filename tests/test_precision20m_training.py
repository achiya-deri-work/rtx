from dataclasses import asdict
import math

import numpy as np
import pytest
import torch

from benchmarks import train_precision20m as study
from rtx.models import DecoderConfig, LinearSpec
from rtx.models.decoder import FP32RMSNorm, QKRMSNorm


def test_profile_step_is_bounded_and_opt_in():
    args=study.parser().parse_args(["--tokens","unused.npy","--output","unused.jsonl","--mode","mxfp8"])
    assert args.profile_step==-1
    study.validate_args(args)
    for value in (-2,args.steps):
        args.profile_step=value
        with pytest.raises(ValueError,match="profile-step"):
            study.validate_args(args)


def test_backward_pipeline_is_opt_in_and_checkpoint_identity_is_preserved():
    base=["--tokens","unused.npy","--output","unused.jsonl","--mode","nvfp4_full"]
    args=study.parser().parse_args(base)
    assert args.ms_eden_backward_pipeline=="serial"
    study.validate_args(args)
    assert study.ms_eden_identity_overrides(args)=={}
    explicit=study.parser().parse_args(base+["--ms-eden-backward-pipeline","serial"])
    assert study.ms_eden_identity_overrides(explicit)=={}
    args.ms_eden_backward_pipeline="paired"
    study.validate_args(args)
    assert study.ms_eden_identity_overrides(args)=={"ms_eden_backward_pipeline":"paired"}
    args.ms_eden_rng_mode="philox"
    assert study.ms_eden_identity_overrides(args)=={
        "ms_eden_backward_pipeline":"paired","ms_eden_rng_mode":"philox"}
    for mode in ("mxfp8","nvfp4_mxfp8"):
        args.mode=mode
        args.ms_eden_rng_mode=None
        with pytest.raises(ValueError,match="pipeline override requires nvfp4_full"):
            study.validate_args(args)
    with pytest.raises(SystemExit):
        study.parser().parse_args(base+["--ms-eden-backward-pipeline","auto"])


def test_backward_pipeline_preserves_initial_weights_and_projection_recipe():
    records=[]
    for pipeline in ("serial","paired"):
        torch.manual_seed(123)
        model=study.build_model(small_config(),"nvfp4_full",16,26,backward_pipeline=pipeline)
        projections=[module for module in model.modules() if isinstance(module,study.PrecisionProjection)]
        assert all(module.backward_pipeline==pipeline for module in projections)
        records.append((study.parameter_hash(model),[(module.dx_json,module.dw_json,
                                                   asdict(module.forward_config)) for module in projections]))
    assert records[0]==records[1]


@pytest.mark.parametrize("mode",study.MODES)
def test_backward_pipeline_is_forwarded_only_to_ms_eden(monkeypatch,mode):
    import rtx
    import rtx.experimental_ms_eden as experimental
    calls=[]
    def fake(x,w,**kwargs):
        calls.append(kwargs)
        return x@w.T
    monkeypatch.setattr(rtx,"mxfp8_linear",fake)
    monkeypatch.setattr(rtx,"nvfp4_linear",fake)
    monkeypatch.setattr(experimental,"ms_eden_linear",fake)
    projection=study.PrecisionProjection(torch.nn.Linear(128,256,bias=False),mode,128,26,
                                         backward_pipeline="paired")
    x=torch.randn(2,64,128)
    torch.testing.assert_close(projection(x),x@projection.weight.T)
    if mode=="nvfp4_full":
        assert calls[0]["backward_pipeline"]=="paired"
    else:
        assert "backward_pipeline" not in calls[0]


def test_source_hashes_cover_training_and_paired_implementation():
    import hashlib
    from pathlib import Path
    root=Path(study.__file__).resolve().parents[1]
    hashes=study.source_hashes()
    for name in ("benchmarks/train_precision20m.py","rtx/experimental_ms_eden.py",
                 "rtx/autotune/posthoc.py","rtx/kernels/nvfp4_posthoc.py",
                 "rtx/kernels/nvfp4_gemm.py","rtx/kernels/mxfp8_gemm.py"):
        assert hashes[name]==hashlib.sha256((root/name).read_bytes()).hexdigest()


def test_preparation_override_preserves_weights_and_is_explicit():
    hashes=[]
    for mode in (None,"coalesced_x4","coalesced_x8"):
        torch.manual_seed(123)
        model=study.build_model(small_config(),"nvfp4_full",16,26,producer_load=mode,producer_blocks=256)
        hashes.append(study.parameter_hash(model))
        for module in model.modules():
            if isinstance(module,study.PrecisionProjection):
                assert module.dx_schedule.producer_load==(mode or "coalesced_wide")
                assert module.dw_schedule.producer_load==(mode or "coalesced_wide")
                assert module.dx_schedule.producer_blocks==module.dw_schedule.producer_blocks==256
    assert len(set(hashes))==1
    args=study.parser().parse_args(["--tokens","unused.npy","--output","unused.jsonl",
        "--mode","mxfp8","--ms-eden-producer-load","coalesced_x8"])
    with pytest.raises(ValueError,match="requires nvfp4_full"):
        study.validate_args(args)
    args.mode="nvfp4_full"
    args.ms_eden_producer_blocks=0
    with pytest.raises(ValueError,match="must be positive"):
        study.validate_args(args)


def small_config():
    return DecoderConfig(vocab_size=257,max_seq_len=8,num_layers=1,hidden_size=128,
        intermediate_size=128,num_attention_heads=2,dtype=torch.bfloat16,
        linear=LinearSpec(precision='bf16'))


@pytest.mark.parametrize("rng_mode",["explicit","philox"])
def test_amortization_schedule_overrides_are_recorded_without_reinitializing(rng_mode):
    torch.manual_seed(125)
    model=study.build_model(small_config(),"nvfp4_full",16,26,producer_blocks=256,
                            rng_mode=rng_mode,producer_signs="register")
    for module in model.modules():
        if isinstance(module,study.PrecisionProjection):
            for schedule in (module.dx_schedule,module.dw_schedule):
                assert schedule.rng_mode==rng_mode
                assert schedule.producer_signs=="register"
    digest=study.parameter_hash(model)
    torch.manual_seed(125)
    assert study.parameter_hash(study.build_model(small_config(),"nvfp4_full",16,26))==digest


def test_cpu_initialization_identical_and_norm_precision():
    torch.set_num_threads(2)
    hashes=[]
    for mode in study.MODES:
        torch.manual_seed(123)
        model=study.build_model(small_config(),mode,batch=16,sm_count=26)
        hashes.append(study.parameter_hash(model))
        assert model.embed_tokens.weight is model.lm_head.weight
        assert model.embed_tokens.weight.dtype==torch.bfloat16
        for module in model.modules():
            if isinstance(module,(FP32RMSNorm,QKRMSNorm)):
                assert all(p.dtype==torch.float32 for p in module.parameters())
        projections=[m for m in model.modules() if isinstance(m,study.PrecisionProjection)]
        assert len(projections)==4
        assert all(m.forward_config.tensor_scale_mode=='exact' for m in projections)
        assert all(m.dx_schedule.producer_load==m.dw_schedule.producer_load=='coalesced_wide' for m in projections)
    assert len(set(hashes))==1


@pytest.mark.parametrize('mode',study.MODES)
def test_only_projection_arithmetic_policy_changes(monkeypatch,mode):
    import rtx
    import rtx.experimental_ms_eden as experimental
    seen=[]
    def fake(x,w,**kwargs):
        seen.append(kwargs)
        return x@w.T
    monkeypatch.setattr(rtx,'mxfp8_linear',fake)
    monkeypatch.setattr(rtx,'nvfp4_linear',fake)
    monkeypatch.setattr(experimental,'ms_eden_linear',fake)
    projection=study.PrecisionProjection(torch.nn.Linear(128,256,bias=False),mode,128,26)
    x=torch.randn(2,64,128)
    torch.testing.assert_close(projection(x),x@projection.weight.T)
    if mode=='nvfp4_full':
        assert seen[0]['source_policy']=='original_bf16'
        assert seen[0]['forward_outer_mode']=='exact'
        assert 'backward_config' not in seen[0]
    else:
        assert seen[0]['backward_config'] is rtx.DEFAULT_MXFP8_BWD_CONFIG
    if mode=='nvfp4_mxfp8':
        assert seen[0]['dynamic_config'].tensor_scale_mode=='exact'


def test_fp32_gradnorm_with_bf16_and_fp32_storage():
    grads=[torch.tensor([3.,4.],dtype=torch.bfloat16),torch.tensor([12.],dtype=torch.float32)]
    result=study.gradient_norm_and_clip(grads,max_norm=13.)
    assert result.dtype==torch.float32
    assert result.item()==13.
    assert [g.dtype for g in grads]==[torch.bfloat16,torch.float32]


@pytest.mark.parametrize('mode',study.MODES)
def test_cpu_model_lifecycle_with_mocked_projection_kernels(monkeypatch,mode):
    import rtx
    import rtx.experimental_ms_eden as experimental
    def fake(x,w,**kwargs):
        return x@w.T
    monkeypatch.setattr(rtx,'mxfp8_linear',fake)
    monkeypatch.setattr(rtx,'nvfp4_linear',fake)
    monkeypatch.setattr(experimental,'ms_eden_linear',fake)
    torch.manual_seed(124)
    model=study.build_model(small_config(),mode,batch=16,sm_count=26)
    tokens=torch.randint(257,(16,8))
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        loss=study.causal_lm_loss(model(tokens),tokens)
        loss.backward()
    assert loss.dtype==torch.float32 and loss.isfinite()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert parameter.grad.dtype==parameter.dtype
        assert parameter.grad.isfinite().all()


def test_schedule_endpoints_and_warmup():
    assert study.learning_rate(0,2048,128,3e-4,.1)==3e-4/128
    assert study.learning_rate(127,2048,128,3e-4,.1)==3e-4
    assert study.learning_rate(2047,2048,128,3e-4,.1)==pytest.approx(3e-5)


def test_corpus_disjoint_deterministic_and_no_rng_consumption(tmp_path):
    path=tmp_path/'corpus.npy'
    np.save(path,np.arange(10000,dtype=np.uint16)%257)
    corpus=study.Corpus(path,10000,941,2,16,4)
    first=corpus.offsets(12)
    np.random.seed(5)
    before=np.random.get_state()
    np.testing.assert_array_equal(first,corpus.offsets(12))
    np.testing.assert_array_equal(before[1],np.random.get_state()[1])
    assert np.all(first+16<corpus.split)
    assert np.all(corpus.validation_offsets>=corpus.split)
    x,y=corpus.batch_at(first,device='cpu')
    assert x.shape==y.shape==(2,16)
    torch.testing.assert_close(x[:,1:],y[:,:-1])


def test_default_architecture_and_cli_aliases():
    args=study.parser().parse_args(['--tokens','unused.npy','--output','unused.jsonl',
        '--mode','mxfp8','--eval-every','64','--eval-batches','8'])
    study.validate_args(args)
    config=study.model_config(args)
    assert config.num_layers==8 and config.hidden_size==512 and config.intermediate_size==1024
    assert args.validation_interval==64 and args.validation_batches==8
    assert args.attention=='flash' and args.cudagraphs is False


@pytest.mark.parametrize('extra',[
    ['--steps','0'],['--width','129'],['--batch','0'],['--warmup-steps','2048'],
    ['--validation-batches','0'],['--minimum-lr-ratio','1.1'],['--cpu-threads','0']])
def test_invalid_protocol_rejected(extra):
    args=study.parser().parse_args(['--tokens','unused.npy','--output','unused.jsonl',
                                  '--mode','mxfp8',*extra])
    with pytest.raises(ValueError):
        study.validate_args(args)
