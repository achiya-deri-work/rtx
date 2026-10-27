"""CPU contracts for transferring measured schedules into model training."""
from dataclasses import asdict
import json

import pytest
import torch

from benchmarks import train_precision20m as study
from rtx.autotune.ms_eden_pipeline import MSEdenPipelineSchedule
from rtx.models import DecoderConfig, LinearSpec


def config():
    return DecoderConfig(vocab_size=257,max_seq_len=8,num_layers=1,hidden_size=128,
        intermediate_size=128,num_attention_heads=2,dtype=torch.bfloat16,
        linear=LinearSpec(precision="bf16"))


def test_compiler_combo_is_opt_in_and_enters_identity():
    args=study.parser().parse_args(['--tokens','unused.npy','--output','unused.jsonl',
        '--mode','nvfp4_mxfp8','--cudagraphs'])
    study.validate_args(args)
    assert study.training_compiler_options(args)=={'triton.cudagraphs':True}
    assert 'compiler_options' not in study.ms_eden_identity_overrides(args)
    args.combo_kernels=True
    expected={'triton.cudagraphs':True,'combo_kernels':True,'combo_kernels_pointwise_only':True}
    assert study.training_compiler_options(args)==expected
    assert study.ms_eden_identity_overrides(args)['compiler_options']==expected


def schedules(model):
    return {",".join(map(str,module.shape)):asdict(MSEdenPipelineSchedule(
        backward_pipeline="paired",dx=module.dx_schedule,dw=module.dw_schedule))
        for module in model.modules() if isinstance(module,study.PrecisionProjection)}


def test_shape_schedules_preserve_initial_weights_and_forward_recipe():
    torch.manual_seed(311)
    baseline=study.build_model(config(),"nvfp4_full",16,26)
    selected=schedules(baseline)
    for entry in selected.values():
        entry["dx"]["producer_blocks"]=32
        entry["dw"]["producer_blocks"]=64
    torch.manual_seed(311)
    candidate=study.build_model(config(),"nvfp4_full",16,26,schedule_map=selected)
    assert study.parameter_hash(candidate)==study.parameter_hash(baseline)
    for base,actual in zip(baseline.modules(),candidate.modules()):
        if isinstance(actual,study.PrecisionProjection):
            assert actual.forward_config==base.forward_config
            assert actual.backward_pipeline=="paired"
            assert actual.dx_schedule.producer_blocks==32
            assert actual.dw_schedule.producer_blocks==64
            assert json.loads(actual.dw_json)==asdict(actual.dw_schedule)
    with pytest.raises(ValueError,match="exactly cover"):
        study.build_model(config(),"nvfp4_full",16,26,schedule_map={})
    with pytest.raises(ValueError,match="requires nvfp4_full"):
        study.build_model(config(),"nvfp4_mxfp8",16,26,schedule_map=selected)


def test_file_content_not_path_enters_checkpoint_identity(tmp_path):
    model=study.build_model(config(),"nvfp4_full",16,26)
    selected=schedules(model)
    path=tmp_path/"schedules.json"
    path.write_text(json.dumps(selected))
    args=study.parser().parse_args(["--tokens","unused.npy","--output","unused.jsonl",
        "--mode","nvfp4_full","--ms-eden-schedules",str(path)])
    study.validate_args(args)
    assert study.ms_eden_identity_overrides(args)=={"ms_eden_schedule_map":selected}
    # The loaded content is frozen for this run even if the source is edited.
    path.write_text("{}")
    assert study.ms_eden_identity_overrides(args)["ms_eden_schedule_map"]==selected
    args.ms_eden_producer_blocks=32
    with pytest.raises(ValueError,match="cannot be combined"):
        study.validate_args(args)


def test_research_variant_requires_map_and_records_code_identity(tmp_path):
    args=study.parser().parse_args(['--tokens','unused.npy','--output','unused.jsonl',
        '--mode','nvfp4_full','--ms-eden-research-variant','tma_dual'])
    with pytest.raises(ValueError,match='explicit schedule map'):
        study.validate_args(args)
    model=study.build_model(config(),'nvfp4_full',16,26)
    selected=schedules(model)
    for entry in selected.values():
        entry['backward_pipeline']='operand_paired'
        for phase in ('dx','dw'):
            entry[phase]['amax_reduction']='staged'
            entry[phase]['rng_mode']='explicit'
    path=tmp_path/'schedules.json';path.write_text(json.dumps(selected))
    args.ms_eden_schedules=path
    study.validate_args(args)
    identity=study.ms_eden_identity_overrides(args)
    assert identity['research_variant']=='tma_dual'
    assert len(identity['research_source_sha256'])==4
    assert all(len(value)==64 for value in identity['research_source_sha256'].values())
    next(iter(selected.values()))['backward_pipeline']='paired'
    path.write_text(json.dumps(selected))
    with pytest.raises(ValueError,match='operand_paired'):
        study.validate_args(args)


def test_research_context_restores_patches_on_training_failure(monkeypatch):
    from contextlib import contextmanager
    from benchmarks import experiment_ms_eden_bank_layout as preparation
    from benchmarks import experiment_ms_eden_dual_finalizer as finalizer
    args=study.parser().parse_args(['--tokens','unused.npy','--output','unused.jsonl','--mode','nvfp4_full'])
    args.ms_eden_research_variant='tma_dual'
    events=[]
    @contextmanager
    def patch(name):
        events.append(('enter',name))
        try:yield
        finally:events.append(('exit',name))
    monkeypatch.setattr(study,'validate_args',lambda _:None)
    monkeypatch.setattr(preparation,'patched',patch)
    monkeypatch.setattr(finalizer,'patched',patch)
    def fail(_):raise RuntimeError('training failure')
    monkeypatch.setattr(study,'_run_validated',fail)
    with pytest.raises(RuntimeError,match='training failure'):study.run(args)
    assert events==[('enter','tma_xor5_1'),('enter','both'),('exit','both'),('exit','tma_xor5_1')]


@pytest.mark.parametrize("text",[
    '{}','[]','{"128,128,128":{},"128,128,128":{}}',
    '{"0128,128,128":{"dx":{},"dw":{},"backward_pipeline":"paired"}}',
    '{"128,128,128":{"dx":{},"dw":{}}}',
    '{"129,128,128":{"dx":{},"dw":{},"backward_pipeline":"paired"}}',
])
def test_malformed_or_ambiguous_maps_fail(tmp_path,text):
    path=tmp_path/"invalid.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        study.load_schedule_map(path)


def test_exact_design_map_identity_and_model_transfer(tmp_path):
    from rtx.autotune.ms_eden_design import MSEdenDesign
    model=study.build_model(config(),'nvfp4_full',16,26)
    selected={key:asdict(MSEdenDesign(schedule=value)) for key,value in schedules(model).items()}
    path=tmp_path/'designs.json'; path.write_text(json.dumps(selected))
    args=study.parser().parse_args(['--tokens','unused.npy','--output','unused.jsonl',
        '--mode','nvfp4_full','--ms-eden-designs',str(path)])
    study.validate_args(args)
    assert args.ms_eden_schedule_map=={k:v['schedule'] for k,v in selected.items()}
    identity=study.ms_eden_identity_overrides(args)
    assert identity['ms_eden_design_map']==selected
    assert len(identity['design_dispatch_source_sha256'])==7
    assert 'rtx/kernels/ms_eden_bank_layout.py' in identity['design_dispatch_source_sha256']
    candidate=study.build_model(config(),'nvfp4_full',16,26,schedule_map=args.ms_eden_schedule_map)
    assert schedules(candidate)==schedules(model)
    args.ms_eden_research_variant='tma_dual'
    with pytest.raises(ValueError,match='cannot be combined'):study.validate_args(args)


def test_row_fma_training_requires_actual_dispatch():
    from types import SimpleNamespace
    args=SimpleNamespace(ms_eden_design_map={'128,128,128':dict(
        transpose_loader='native',row_transform='fma')},ms_eden_dispatch_records=[])
    with pytest.raises(RuntimeError,match='row FMA'):
        study.verify_design_dispatch(args)
    args.ms_eden_dispatch_records=[dict(transpose=False,selected='native')]
    with pytest.raises(RuntimeError,match='row FMA'):
        study.verify_design_dispatch(args)
    args.ms_eden_dispatch_records=[dict(transpose=False,selected='row_fma')]
    study.verify_design_dispatch(args)


def test_design_map_rejects_mixed_implementations_and_illegal_layout(tmp_path):
    from dataclasses import replace
    from rtx.autotune.ms_eden_design import MSEdenDesign
    base=MSEdenPipelineSchedule(backward_pipeline='operand_paired')
    base=replace(base,dx=replace(base.dx,amax_reduction='staged'),
                 dw=replace(base.dw,amax_reduction='staged'))
    raw={'128,128,128':asdict(MSEdenDesign(schedule=base)),
         '256,128,128':asdict(MSEdenDesign(schedule=base,finalizer='dual'))}
    path=tmp_path/'designs.json';path.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='one shared'):study.load_design_map(path)
    bad=replace(base,dx=replace(base.dx,gemm=replace(base.dx.gemm,scale_role='producer')))
    path.write_text(json.dumps({'128,128,128':asdict(MSEdenDesign(schedule=bad))}))
    with pytest.raises(ValueError,match='mma128'):study.load_design_map(path)


def test_design_dispatch_evidence_required_and_context_restored(monkeypatch):
    from contextlib import contextmanager
    from benchmarks import ms_eden_design_dispatch
    from rtx.autotune.ms_eden_design import MSEdenDesign
    args=study.parser().parse_args(['--tokens','unused.npy','--output','unused.jsonl','--mode','nvfp4_full'])
    args.ms_eden_design_map={'128,128,128':asdict(MSEdenDesign(transpose_loader='tma_xor4_1'))}
    events=[]
    @contextmanager
    def dispatch(design):
        events.append(design.transpose_loader)
        try:yield [{'transpose':True,'selected':design.transpose_loader}]
        finally:events.append('restored')
    monkeypatch.setattr(study,'validate_args',lambda _:None)
    monkeypatch.setattr(ms_eden_design_dispatch,'dispatch',dispatch)
    def fail(args):
        study.verify_design_dispatch(args)
        raise RuntimeError('intentional training failure')
    monkeypatch.setattr(study,'_run_validated',fail)
    with pytest.raises(RuntimeError,match='intentional'):study.run(args)
    assert events==['tma_xor4_1','restored']
    args.ms_eden_dispatch_records=[]
    with pytest.raises(RuntimeError,match='did not establish'):study.verify_design_dispatch(args)
