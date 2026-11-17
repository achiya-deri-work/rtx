"""CPU invariants for compiler-visible MoE buffer ownership and liveness.

These checks do not stand in for CUDA numerical/initialization validation.
They independently account for DAG uses, alias roots, typed arenas and public
output ownership before the allocation-free launch sequence is constructed.
"""
from dataclasses import replace
import ctypes
import json
import math

import pytest
import torch

from rtx import _moe_execution as execution
from rtx.moe import DEFAULT_MOE_CONFIG, _config_key


def _cases():
    base=DEFAULT_MOE_CONFIG;p=base.prepared
    joint=replace(p,backward_preparation='joint',recompute=replace(p.recompute,glu_grad_store='exchange'))
    native=p.with_scale_layout('mma128')
    linear_stages=('down','down_dx','down_dw','gate_dx','gate_dw')
    tma=replace(p,capacity_multiple=128,**{name:replace(getattr(p,name),epilogue='tma') for name in linear_stages})
    from rtx.autotune.moe import structural_seeds
    onchip_shape=(128,128,8,32,2,128)
    onchip=next(cfg for cfg in structural_seeds(onchip_shape) if cfg.family=='grouped_onchip')
    return [
        ('aligned',(32,128,4,16,2,64),base),
        ('ragged',(13,35,5,6,2,49),base),
        ('native',(13,128,8,4,2,128),replace(base,prepared=native)),
        ('joint',(13,64,8,4,2,64),replace(base,prepared=joint)),
        ('global',(32,128,4,16,2,64),replace(base,prepared=replace(p,input_preparation='quantize_once'))),
        ('global_native',(13,128,8,4,2,128),replace(base,prepared=replace(native,input_preparation='quantize_once'))),
        ('global_joint',(13,64,8,4,2,64),replace(base,prepared=replace(joint,input_preparation='quantize_once'))),
        ('tma_outputs',(128,128,4,64,2,128),replace(base,prepared=tma)),
        ('staged_aligned',(32,128,4,16,2,64),replace(base,family='grouped_staged')),
        ('staged_ragged',(13,35,5,6,2,49),replace(base,family='grouped_staged')),
        # Prepared fields remain in the immutable config after a family
        # switch, but must not inflate legacy movement/storage capacity.
        ('staged_inactive_capacity',(32,128,4,16,2,64),
         replace(base,family='grouped_staged',prepared=tma)),
        ('onchip',onchip_shape,onchip),
        ('onchip_inactive_capacity',onchip_shape,replace(onchip,prepared=tma)),
    ]


_CASES=_cases()


@torch.library.custom_op('rtx_moe_plan_test::execute',mutates_args=('buffers',),device_types='cpu')
def _mock_execute(inputs:list[torch.Tensor],buffers:list[torch.Tensor],key:str)->None:
    # Compiler-contract probe only. No unsupported CPU float8 arithmetic and
    # no numerical claim about the real CUDA MoE operation.
    for buffer in buffers:
        ctypes.memset(buffer.data_ptr(),0,buffer.numel()*buffer.element_size())
    buffers[-1].fill_(2)


@_mock_execute.register_fake
def _mock_execute_fake(inputs,buffers,key):
    return None


def _key(shape,config,phase):
    return json.dumps((shape,_config_key(config),phase),sort_keys=True)


def _root(ref,aliases):
    seen=set()
    while ref in aliases:
        assert ref not in seen,'cyclic view ownership'
        seen.add(ref);ref=aliases[ref]
    return ref


def _uses(value):
    if isinstance(value,int):
        yield value
    else:
        for element in value:
            yield from _uses(element)


def _lifetimes(builder):
    aliases=getattr(builder,'aliases',{})
    last={i:buffer.birth for i,buffer in enumerate(builder.buffers) if i not in aliases}
    for step,(_,_,_,arguments) in enumerate(builder.steps):
        for ref in _uses(arguments):
            if ref>=0:
                root=_root(ref,aliases);last[root]=max(last[root],step)
    return last


def _check_live_intervals(builder,views):
    aliases=getattr(builder,'aliases',{});last=_lifetimes(builder)
    roots=[i for i in range(len(builder.buffers)) if i not in aliases and not builder.buffers[i].output]
    for pos,left in enumerate(roots):
        a=builder.buffers[left];_,_,a_group,a_start=views[left];a_end=a_start+math.prod(a.shape)
        for right in roots[pos+1:]:
            b=builder.buffers[right];_,_,b_group,b_start=views[right];b_end=b_start+math.prod(b.shape)
            if a_group!=b_group:continue
            live_overlap=max(a.birth,b.birth)<=min(last[left],last[right])
            storage_overlap=max(a_start,b_start)<min(a_end,b_end)
            assert not(live_overlap and storage_overlap),(
                'live scratch overlap',left,right,(a.birth,last[left]),(b.birth,last[right]),
                (a_start,a_end),(b_start,b_end))


@pytest.mark.parametrize('case,shape,config',_CASES,ids=[c[0] for c in _CASES])
@pytest.mark.parametrize('phase',('forward','backward'))
def test_typed_plan_live_intervals_and_capacity(case,shape,config,phase):
    builder,dtypes,sizes,views,outputs=execution.plan(_key(shape,config,phase))
    aliases=getattr(builder,'aliases',{})
    assert len(views)==len(builder.buffers)
    assert len(dtypes)==len(set(dtypes))==len(sizes)
    assert all(size>0 for size in sizes)
    for ref,(buffer,view) in enumerate(zip(builder.buffers,views)):
        extent,dtype,group,offset=view
        assert tuple(extent)==buffer.shape and dtype==buffer.dtype
        assert all(type(size)is int and size>0 for size in extent)
        assert 0<=buffer.birth<=len(builder.steps)
        if buffer.output:
            assert group<0 and ref in outputs
        else:
            assert 0<=group<len(dtypes) and dtype==dtypes[group]
            assert offset>=0 and offset+math.prod(extent)<=sizes[group]
            assert offset*dtype.itemsize%16==0
        if ref in aliases:
            root=_root(ref,aliases)
            assert views[ref][2:]==views[root][2:]
            assert buffer.dtype==builder.buffers[root].dtype
            assert math.prod(buffer.shape)==math.prod(builder.buffers[root].shape)
    _check_live_intervals(builder,views)


@pytest.mark.parametrize('case,shape,config',_CASES,ids=[c[0] for c in _CASES])
@pytest.mark.parametrize('phase',('forward','backward'))
def test_allocated_outputs_own_storage_and_scratch_aliases_keep_one_dtype(case,shape,config,phase):
    builder,dtypes,sizes,views,outputs=execution.plan(_key(shape,config,phase))
    buffers=execution.allocate(torch.empty(1),dtypes,sizes,views)
    storage_dtypes={}
    for value in buffers:
        storage_dtypes.setdefault(value.untyped_storage()._cdata,set()).add(value.dtype)
    assert all(len(dtypes)==1 for dtypes in storage_dtypes.values())
    scratch={value.untyped_storage()._cdata for i,value in enumerate(buffers) if i not in outputs}
    result_storages=[]
    for ref in outputs:
        result=buffers[ref];storage=result.untyped_storage()._cdata
        assert result._base is None
        assert storage not in scratch and storage not in result_storages
        result_storages.append(storage)
    n,d,e,c,k,h=shape
    if phase=='forward':
        assert [(tuple(buffers[i].shape),buffers[i].dtype) for i in outputs]==[((n,d),torch.float32)]
    else:
        expected=[((n,d),torch.bfloat16),((e*2*h,d),torch.bfloat16),
                  ((e*d,h),torch.bfloat16),((e,c),torch.float32)]
        assert [(tuple(buffers[i].shape),buffers[i].dtype) for i in outputs]==expected


def _read_write(symbol,params,args):
    """Independent interpretation of each existing launch ABI."""
    if symbol=='compile_movement':
        kind=params[6]
        read=[args[0],args[1],args[2],args[3]]
        write=[args[5]]
        if kind in ('slot_grad','route_grad'):
            read.append(args[4]);write.append(args[6])
        return read,write
    if symbol=='compile_expert_quant':return [args[0],args[1]],[args[2],args[3]]
    if symbol=='compile_expert_packed_gemm':
        read=list(args[:4]);write=[args[4]]
        if params[3]=='glu':write.append(args[5])
        elif params[3]=='derivative':read.append(args[5])
        return read,write
    if symbol=='compile_dequant_hidden':return list(args[:2]),[args[2]]
    if symbol=='compile_native_hidden_scales':return [args[0]],[args[1]]
    if symbol=='compile_packed_input_gather':
        return list(args[:5]),list(args[5:7])+([args[7]] if params[-1] else [])
    if symbol=='compile_expert_joint_recompute':
        return list(args[:5])+[args[5][0]],list(args[5][1:])
    if symbol=='compile_grouped_linear':
        return list(args[:2])+([args[3]] if params[-1] else []),[args[2]]
    if symbol=='compile_grouped_glu':
        if len(params)>3 and params[3]:return list(args[:2])+[args[3]],[args[2]]
        return list(args[:2]),list(args[2:4])
    if symbol=='compile_onchip_experts':return [args[0],args[1],args[3]],[args[2]]
    raise AssertionError('unreviewed kernel ABI: '+symbol)


@pytest.mark.parametrize('case,shape,config',_CASES,ids=[c[0] for c in _CASES])
@pytest.mark.parametrize('phase',('forward','backward'))
def test_every_scratch_read_has_a_preceding_producer(case,shape,config,phase):
    builder,*_=execution.plan(_key(shape,config,phase));aliases=getattr(builder,'aliases',{})
    initialized=set()
    for step,(_module,symbol,params,args) in enumerate(builder.steps):
        read,write=_read_write(symbol,params,args)
        for ref in _uses(read):
            if ref<0:
                assert -ref<=(6 if phase=='forward' else 7)
            else:
                assert _root(ref,aliases) in initialized,(case,phase,step,symbol,'uninitialized read',ref)
        initialized.update(_root(ref,aliases) for ref in _uses(write) if ref>=0)
    assert all(_root(ref,aliases) in initialized for ref in builder.outputs)


def test_cold_frontend_metadata_dtypes_match_hot_output_contract():
    import rtx.moe as moe
    n,d,e,c,k,h=13,35,5,6,2,49
    def tensor(shape,dtype):return torch.empty(shape,device='meta',dtype=dtype)
    operands=(tensor((n,d),torch.bfloat16),tensor((e,2*h,d),torch.bfloat16),
              tensor((e,d,h),torch.bfloat16),tensor((e,c),torch.int64),
              tensor((e,c),torch.float32),tensor((n,k),torch.int64))
    y,token=moe._experts_fake(*operands,'unused',True)
    assert y.shape==(n,d) and y.dtype==torch.float32
    assert token.device.type=='cpu' and token.dtype==torch.uint8
    fake_gradients=moe._experts_backward_fake(*operands,tensor((n,d),torch.float32),token)
    _,_,_,views,outputs=execution.plan(_key((n,d,e,c,k,h),DEFAULT_MOE_CONFIG,'backward'))
    for gradient,ref in zip(fake_gradients,outputs):
        assert gradient.dtype==views[ref][1]
        assert gradient.numel()==math.prod(views[ref][0])


@pytest.mark.parametrize('family',('grouped_prepared','grouped_staged','grouped_onchip'))
@pytest.mark.parametrize('phase',('forward','backward'))
def test_canonical_byte_scale_plan_survives_actual_inductor_functionalization(family,phase):
    # E8M0 arenas caused Inductor's unsupported-dtype matcher to skip the
    # mandatory HOP decomposition, although aot_eager succeeded. Run actual
    # Inductor against the exact multi-dtype/aliased arena plan to catch this.
    case=next(case for case in _CASES if case[2].family==family)
    shape=case[1];policy=_config_key(case[2])
    def run(x):
        key,dtypes,sizes,views,outputs=execution.layout(shape,policy,phase)
        buffers=execution.allocate(x,dtypes,sizes,views)
        _mock_execute([x],buffers,key)
        return buffers[outputs[0]]
    compiled=torch.compile(run,fullgraph=True,dynamic=False)
    result=compiled(torch.randn(shape[:2],dtype=torch.bfloat16))
    assert result.dtype==(torch.float32 if phase=='forward' else torch.bfloat16)
    torch.testing.assert_close(result,torch.full_like(result,2))


@pytest.mark.parametrize('case,shape,config',[case for case in _CASES if case[2].family!='grouped_prepared'],
                         ids=[case[0] for case in _CASES if case[2].family!='grouped_prepared'])
@pytest.mark.parametrize('phase',('forward','backward'))
def test_legacy_movement_capacity_does_not_inherit_prepared_padding(case,shape,config,phase):
    builder,*_=execution.plan(_key(shape,config,phase))
    for _,symbol,params,_ in builder.steps:
        if symbol=='compile_movement':assert params[-1]==shape[3]


@pytest.mark.parametrize('family',('grouped_prepared','grouped_staged','grouped_onchip'))
def test_all_selected_families_public_fullgraph_fake_cuda(family):
    from torch._subclasses.fake_tensor import FakeTensor,FakeTensorMode
    from rtx.moe import mxfp8_routed_experts
    from rtx.routing import BalancedRouting
    _,(n,d,e,c,k,h),config=next(case for case in _CASES if case[2].family==family)
    def call(x,gu,down,ids,weights,inverse):
        return mxfp8_routed_experts(x,gu,down,BalancedRouting(ids,weights,inverse),config=config)
    with FakeTensorMode() as mode:
        def fake(shape,dtype):
            return FakeTensor(mode,torch.empty(shape,device='meta',dtype=dtype),torch.device('cuda:0'))
        operands=(fake((n,d),torch.bfloat16),fake((e,2*h,d),torch.bfloat16),
                  fake((e,d,h),torch.bfloat16),fake((e,c),torch.int64),
                  fake((e,c),torch.float32),fake((n,k),torch.int64))
        result=torch.compile(call,backend='eager',fullgraph=True,dynamic=False)(*operands)
        assert result.shape==(n,d) and result.dtype==torch.float32 and result.device.type=='cuda'


def test_nested_alias_extends_original_owners_lifetime(monkeypatch):
    # Regression: resolving only one alias hop frees original storage at the
    # first alias's last use, despite a nested view being used later.
    builder=execution.Builder((1,1,1,1,1,1),DEFAULT_MOE_CONFIG,'nested-regression')
    root=builder.tensor((16,),torch.float32)
    builder.step('stub','write',(),(root,))
    first=builder.alias(root,(4,4));builder.step('stub','first-use',(),(first,))
    nested=builder.alias(first,(16,))
    other=builder.tensor((16,),torch.float32);builder.step('stub','other-write',(),(other,))
    builder.step('stub','late-nested-read',(),(nested,))
    monkeypatch.setattr(execution.Builder,'build',lambda self:builder)
    key=_key((1,1,1,1,1,1),DEFAULT_MOE_CONFIG,'nested-regression')
    execution.plan.cache_clear()
    try:
        planned,_,_,views,_=execution.plan(key)
        _check_live_intervals(planned,views)
    finally:execution.plan.cache_clear()
