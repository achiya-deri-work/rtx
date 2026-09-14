"""Deferred GLU selection using shared evidence, locking and winner storage."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .autotune.effort import AutotunePolicy


@dataclass(frozen=True)
class DeferredGLUConfig:
    key: str


def deferred_config(config, mode, policy, cache_dir):
    from .glu import _config_key
    if mode not in ('off','cache','balanced'):
        raise ValueError('GLU autotune must be off, cache, or balanced')
    if policy is not None and not isinstance(policy,AutotunePolicy):
        raise TypeError('tuning_policy must be AutotunePolicy')
    if mode=='off':return None
    policy=(policy or AutotunePolicy.preset('balanced')).for_pass()
    return DeferredGLUConfig(json.dumps(dict(_glu_autotune=1,initial=json.loads(_config_key(config)),
        mode=mode,policy=asdict(policy),cache_dir=None if cache_dir is None else str(Path(cache_dir).expanduser()),
        _glu_frontend_revision=4),sort_keys=True))


def resolve(x,weight,activation,output_format,spec):
    import torch
    from .configs.glu import glu_config_from_dict,prequant_glu_config_from_dict,glu_rejection
    from .kernels.mxfp8 import MXFP8Problem,fwd_config_to_dict
    from .autotune.glu import GLU_KERNEL_REVISION,tune_mxfp8_glu
    from .autotune.legacy import default_cache_dir
    from .autotune.core import stable_id
    from .autotune.winners import runtime_winner_key,load_runtime_winner,save_runtime_winner,runtime_tuning_lock
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('warm GLU first-hit selection before CUDA graph capture')
    values=dict(spec['policy']);values['predictor_heads']=tuple(values['predictor_heads'])
    policy=AutotunePolicy(**values)
    backend='prequant' if spec['initial'].get('backend')=='prequant' else 'cta'
    decode=prequant_glu_config_from_dict if backend=='prequant' else glu_config_from_dict
    initial=decode(spec['initial'])
    root=default_cache_dir() if spec['cache_dir'] is None else Path(spec['cache_dir'])
    problem=MXFP8Problem(x.shape[0],weight.shape[0],x.shape[1])
    key=runtime_winner_key('mxfp8_glu_prequant_fwd' if backend=='prequant' else 'mxfp8_glu_fwd',
        problem,device=x.device,kernel_revision=GLU_KERNEL_REVISION,
        variant=stable_id(dict(activation=activation,output_format=output_format,initial=spec['initial'])))
    def rejection(c):
        return c.rejection(problem,output_format) if backend=='prequant' else glu_rejection(problem,c,output_format)
    def cached():
        return load_runtime_winner(key,decode,root=root,rejection=rejection,
            tuning_policy=policy if spec['mode']=='balanced' else None)
    found=cached()
    if found is not None:return found
    if spec['mode']=='cache':return initial
    with runtime_tuning_lock(key,root=root):
        found=cached()
        if found is not None:return found
        result=tune_mxfp8_glu(x,weight,activation=activation,output_format=output_format,backend=backend,
            output_dir=root/'glu_effort',tuning_policy=policy,progress=None,initial=initial)
        save_runtime_winner(key,fwd_config_to_dict(result.config),config_id=stable_id(fwd_config_to_dict(result.config)),
            root=root,median_ms=result.median_ms,tuning_evidence=result.tuning_evidence)
        return result.config
