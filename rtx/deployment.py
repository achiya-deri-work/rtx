"""CPU-side, proposal-only handoff into disposable MLP qualification.

Importing evidence never installs a winner or mutates a model. This deliberately
does not equate an RTX-to-RTX timing result with a BF16 replacement decision.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys


def bind_qualified_mlp_plans_(module, bindings):
    """Pin current-source preflight winners by exact input shape and grad phase.

    ``bindings`` contains ``(request_directory, input_shape)`` pairs. Separate
    requests may use different configs, caches and objectives. Checks complete
    before mutation. Parameters and disk winners are unchanged; unknown shapes,
    phases, devices and replay mode fail closed. Call before compile/capture.
    Synthetic 2D preflight does NOT qualify 3D application memory/performance.
    The returned JSON-compatible manifest records that boundary explicitly.
    """
    import math
    import torch
    from .mlp import MXFP8MLP
    from .autotune.replacement_benchmark import source_manifest
    if torch.compiler.is_compiling():
        raise RuntimeError('bind MLP plans before compilation')
    if not isinstance(module, MXFP8MLP) or module.precision != 'mxfp8':
        raise TypeError('requires an existing MXFP8MLP with mxfp8 precision')
    if module._shape_configs:
        raise ValueError('remove manual shape overrides before binding phase-specific plans')
    device = module.gate_up_weight.device
    if (device.type != 'cuda' or module.down_weight.device != device
            or any(p.dtype != torch.bfloat16 for p in (module.gate_up_weight, module.down_weight))
            or tuple(module.gate_up_weight.shape) != (2*module.hidden_features, module.in_features)
            or tuple(module.down_weight.shape) != (module.out_features, module.hidden_features)):
        raise ValueError('requires matching BF16 CUDA parameters')
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('bind MLP plans before capture')
    plans, evidence, seen = [], [], set()
    for request, input_shape in bindings:
        path = Path(request)
        shape = tuple(input_shape)
        if len(shape) < 2 or any(type(v) is not int or v <= 0 for v in shape):
            raise ValueError('input_shape must contain positive integer dimensions, rank >= 2')
        candidate = MLPCandidate.load(path / 'candidate.json')
        if (candidate.shape != (math.prod(shape[:-1]), shape[-1], module.hidden_features, module.out_features)
                or shape[-1] != module.in_features or candidate.activation != module.activation):
            raise ValueError('binding shape/activation differs from qualified context')
        identity = (shape, candidate.phase)
        if identity in seen:
            raise ValueError(f'duplicate bound context: {identity}')
        seen.add(identity)
        coverage = candidate.check_cache(path, device=device)
        if coverage['status'] != 'covered':
            raise RuntimeError(f'qualification cache gap: {coverage["status"]}')
        plans.append(dict(input_shape=list(shape), training=candidate.phase == 'train',
                          config=coverage['qualified_config']))
        evidence.append(dict(request=str(path.resolve()), input_shape=list(shape), phase=candidate.phase,
                             coverage=coverage))
    if not plans:
        raise ValueError('at least one binding is required')
    spec = dict(mode='bound', device=str(device), activation=module.activation,
                gate_shape=list(module.gate_up_weight.shape), down_shape=list(module.down_weight.shape),
                plans=plans)
    manifest = dict(schema=1, status='preflight_plans_pinned', measurement_mode='selected_direct',
        rtx_source_sha256=source_manifest(), bindings=evidence, parameters_replaced=False,
        unknown_context='error', runtime_disk_lookups=False, runtime_tuning=False,
        application_qualified=False, actual_input_layout_qualified=False,
        remaining_requirements=['validate actual input ranks, gradients and caller compilation',
            'measure full-application peak memory and ordinary steady update time',
            'compare BF16 numerics separately from quantized-recipe correctness',
            'rebind after source/environment/device changes; recompile after binding'])
    module._first_hit_spec = json.dumps(spec, sort_keys=True)
    module._bound_plans = True
    return manifest


def require_qualified_mlp_cache_(module, requests):
    """Validate declared contexts, then make an existing MXFP8MLP cache-only.

    Startup-only: call before compile/capture and without concurrent execution.
    All checks precede module mutation. Parameters/configuration/precision and
    cache files are unchanged. This is not BF16 conversion or a locked freeze;
    undeclared contexts and subsequent cache changes remain caller concerns.
    Returns a detached coverage report for logging.
    """
    import torch
    from .mlp import MXFP8MLP
    if not isinstance(module, MXFP8MLP) or module.precision != 'mxfp8':
        raise TypeError('requires an existing MXFP8MLP with mxfp8 precision')
    if torch.compiler.is_compiling():
        raise RuntimeError('qualify cached deployment before compilation')
    if isinstance(requests, (str, Path)):
        raise TypeError('requests must be an iterable of request directories')
    paths = tuple(Path(path) for path in requests)
    if not paths:
        raise ValueError('at least one declared qualification request is required')
    device = module.gate_up_weight.device
    if device.type != 'cuda' or module.down_weight.device != device:
        raise ValueError('module weights must be on the qualified CUDA device')
    if (module.gate_up_weight.dtype != torch.bfloat16 or module.down_weight.dtype != torch.bfloat16
            or tuple(module.gate_up_weight.shape) != (2*module.hidden_features, module.in_features)
            or tuple(module.down_weight.shape) != (module.out_features, module.hidden_features)):
        raise ValueError('module parameter dtype/shapes differ from qualification contract')
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('qualify cached deployment before graph capture')
    def identity(spec):
        # Strict mode changes miss handling, not winner/objective identity.
        return {key: value for key, value in spec.items() if key not in ('mode', 'cache_miss')}
    module_spec = module.export_tuning_spec()
    coverage = []
    contexts = set()
    for path in paths:
        candidate = MLPCandidate.load(path / 'candidate.json')
        if (candidate.shape[1:] != (module.in_features, module.hidden_features, module.out_features)
                or candidate.activation != module.activation):
            raise ValueError(f'qualification dimensions/activation differ: {path}')
        context = (candidate.shape, candidate.phase, candidate.activation)
        if context in contexts:
            raise ValueError(f'duplicate declared context: {context}')
        contexts.add(context)
        if identity(json.loads((path / 'spec.json').read_text())) != identity(module_spec):
            raise ValueError(f'qualification spec differs from consuming module: {path}')
        result = candidate.check_cache(path, device=device)
        if result['status'] != 'covered':
            raise RuntimeError(f'qualification cache gap at {path}: {result["status"]}')
        coverage.append(result)
    module.require_cached_()
    return dict(schema=1, status='declared_contexts_checked_cache_only', contexts=coverage,
                parameters_replaced=False, precision_changed=False,
                remaining_requirements=['compile or recapture the caller after this startup mutation',
                    'undeclared contexts are not qualified; strict runtime misses cannot tune',
                    'cache entries are not locked or pinned by this operation',
                    'application/BF16 replacement acceptance remains separate'])


@dataclass(frozen=True)
class MLPCandidate:
    """Exact contiguous-2D selected-direct MXFP8 MLP proposal.

    ``shape`` is (M,K,H,O), where H is post-gating width. ``phase`` is train or
    forward (no-grad inference). Configuration and provenance are transported,
    not weights. Preparation performs no CUDA work or model mutation. Neither
    import nor successful synthetic preflight authorizes application installation.
    """

    shape: tuple[int, int, int, int]
    config: object
    phase: str = 'train'
    activation: str = 'silu'
    evidence_json: str = '{}'

    def __post_init__(self):
        from .mlp import MXFP8MLPConfig
        if type(self.shape) is not tuple or len(self.shape)!=4 or any(type(v) is not int or v<=0 for v in self.shape):
            raise ValueError('shape must be a tuple of positive M,K,H,O integers')
        if not isinstance(self.config,MXFP8MLPConfig):
            raise TypeError('config must be MXFP8MLPConfig')
        if self.phase not in ('train','forward') or self.activation not in ('silu','gelu','relu'):
            raise ValueError('unsupported phase or activation')
        evidence=json.loads(self.evidence_json)
        if not isinstance(evidence,dict):
            raise ValueError('evidence must encode a JSON object')
        json.dumps(evidence,allow_nan=False)

    def to_dict(self):
        return dict(schema=1,status='proposal',family='mxfp8_recompute_mlp',
            shape=list(self.shape),phase=self.phase,activation=self.activation,
            precision='mxfp8',layout='contiguous_2d',measurement_mode='selected_direct',
            config=self.config.to_dict(),evidence=json.loads(self.evidence_json))

    def save(self,path):
        """Write a new JSON bundle; refuse to overwrite evidence."""
        with Path(path).open('x') as stream:
            json.dump(self.to_dict(),stream,indent=2,sort_keys=True,allow_nan=False)

    @classmethod
    def load(cls,path):
        from .mlp import MXFP8MLPConfig
        data=json.loads(Path(path).read_text())
        constants=dict(schema=1,status='proposal',family='mxfp8_recompute_mlp',
            precision='mxfp8',layout='contiguous_2d',measurement_mode='selected_direct')
        required={*constants,'shape','phase','activation','config','evidence'}
        if type(data.get('schema')) is not int or set(data)!=required or any(data[k]!=v for k,v in constants.items()):
            raise ValueError('unsupported or malformed MLP candidate bundle')
        return cls(tuple(data['shape']),MXFP8MLPConfig.from_dict(data['config']),
                   data['phase'],data['activation'],json.dumps(data['evidence'],sort_keys=True,allow_nan=False))

    @classmethod
    def from_qualification(cls,path,*,field='challenger'):
        """Import a ``tools/qualify_mlp_deployment.py`` report as a proposal.

        That tool's contract is SiLU/MXFP8; other precisions or measurement modes
        are not inferred. Retain exact shape, source manifest and artifact hash.
        Even a reported promotion remains research evidence requiring preflight
        and BF16/application qualification, not a production approval.
        """
        from .mlp import MXFP8MLPConfig
        source=Path(path)
        raw=source.read_bytes()
        data=json.loads(raw)
        if field not in ('incumbent','challenger'):
            raise ValueError('field must explicitly name incumbent or challenger')
        if (data.get('mode')!='selected_direct' or not data.get('rtx_source_sha256')
                or not isinstance(data.get('records'),list) or not data['records']
                or data.get('arguments',{}).get('mode')!='selected_direct'
                or data.get('activation','silu')!='silu'
                or data.get('precision','mxfp8')!='mxfp8'):
            raise ValueError('expected a source-pinned selected-direct MLP qualification report')
        evidence=dict(artifact=str(source.resolve()),sha256=hashlib.sha256(raw).hexdigest(),
            selected_field=field,source_hashes=data['rtx_source_sha256'],
            research_decision=data.get('decision'),torch=data.get('torch'),
            cuda=data.get('cuda'),gpu=data.get('gpu'),
            qualified_for_application=False)
        return cls(tuple(data['shape']),MXFP8MLPConfig.from_dict(data[field]),
                   data['phase'],'silu',json.dumps(evidence,sort_keys=True,allow_nan=False))

    def prepare(self,output,*,policy=None,cache_dir=None,execution_mix=None,reserve_mib=0):
        """Export a NEW preflight request and argv; do not launch or install.

        Eight trials by default. User effort is preserved verbatim when supplied.
        The returned command runs cold tuning, strict fresh-process reuse and
        separate independent numerics. Run before allocating the application.
        Changed sources are disclosed, never treated as compatible qualification.
        """
        from .autotune.effort import AutotunePolicy
        from ._mlp_autotune import deferred_spec
        from .autotune.replacement_benchmark import source_manifest
        if type(reserve_mib) is not int or reserve_mib<0:
            raise ValueError('reserve_mib must be a nonnegative integer')
        if policy is None:
            policy=AutotunePolicy(max_trials=8,max_seconds=120,warmup=1,samples=3,
                calls_per_sample=1,confirmation_repeats=0,finalist_rounds=6)
        root=Path(output).resolve()
        cache=(root/'cache') if cache_dir is None else Path(cache_dir).expanduser().resolve()
        mixes={} if execution_mix is None else {
            'training_mix' if self.phase=='train' else 'inference_mix':execution_mix}
        spec=deferred_spec(self.config,'mxfp8','balanced',policy,cache,**mixes)
        sources=source_manifest()
        evidence=json.loads(self.evidence_json)
        command=[sys.executable,'-m','rtx.autotune.mlp_preflight',
            '--shape',','.join(map(str,self.shape)),'--phase',self.phase,
            '--activation',self.activation,'--spec',str(root/'spec.json'),
            '--output',str(root/'qualification'),'--reserve-mib',str(reserve_mib),
            '--check-numerics']
        request=dict(schema=1,status='prepared_request',candidate=self.to_dict(),
            source_matches_evidence=bool(evidence.get('source_hashes')) and evidence['source_hashes']==sources,
            prepared_source_hashes=sources,spec_sha256=hashlib.sha256(spec.encode()).hexdigest(),
            requires_qualification=True,installs_runtime_winner=False,
            qualification_cache_dir=str(cache),cache_is_request_local=cache==root/'cache',
            command=command)
        root.mkdir(parents=True,exist_ok=False)
        self.save(root/'candidate.json')
        with (root/'spec.json').open('x') as stream: stream.write(spec)
        with (root/'request.json').open('x') as stream:
            json.dump(request,stream,indent=2,sort_keys=True,allow_nan=False)
        return request

    def inspect(self, output):
        """Verify a completed request's recorded gates without CUDA or writes.

        Raises on missing, inconsistent or failed evidence. A returned report
        describes historical exact-context qualification, NOT current cache
        coverage, application acceptance or permission to install. Source drift
        is disclosed separately so archived evidence remains inspectable.
        """
        from .autotune.mlp_preflight import (
            require_numerical_gate, require_preflight_gates, require_selected_plan)
        from .autotune.replacement_benchmark import source_manifest

        root = Path(output)
        request = json.loads((root / 'request.json').read_text())
        if (request.get('schema') != 1 or request.get('status') != 'prepared_request'
                or request.get('candidate') != self.to_dict()
                or self.load(root / 'candidate.json') != self):
            raise ValueError('candidate/request identity mismatch')
        spec = (root / 'spec.json').read_bytes()
        digest = hashlib.sha256(spec).hexdigest()
        if digest != request.get('spec_sha256'):
            raise ValueError('prepared spec changed')
        payload = json.loads(spec)
        if payload.get('initial') != self.config.to_dict():
            raise ValueError('prepared initial configuration differs from candidate')
        reports = {stage: json.loads((root / 'qualification' / stage / 'report.json').read_text())
                   for stage in ('cold', 'reuse', 'numerics')}
        sources = reports['cold'].get('source_hashes_before')
        if not sources:
            raise ValueError('missing qualification source identity')
        environment = {key: reports['cold'].get(key) for key in ('torch', 'cuda', 'gpu')}
        if not all(environment.values()):
            raise ValueError('missing qualification environment')
        contract = dict(shape=list(self.shape), phase=self.phase,
                        activation=self.activation, measurement_mode='selected_direct',
                        spec_sha256=digest)
        for stage, report in reports.items():
            if any(report.get(k) != v for k, v in {**contract, **environment}.items()):
                raise ValueError(f'{stage} qualification contract mismatch')
            if (report.get('source_hashes_before') != sources
                    or report.get('source_hashes_after') != sources):
                raise ValueError(f'{stage} qualification source mismatch')
            if report.get('verify_cache') is not (stage != 'cold'):
                raise ValueError(f'{stage} strict-cache contract mismatch')
            if stage == 'numerics':
                require_numerical_gate(report)
            else:
                if report.get('spec') != payload:
                    raise ValueError(f'{stage} recorded spec mismatch')
                require_preflight_gates(report)
        # Check the requested occupancy, not merely consistency between workers.
        command = request['command']
        reserve = int(command[command.index('--reserve-mib') + 1])
        for stage in ('cold', 'reuse'):
            report = reports[stage]
            if (reserve < 0 or report.get('reserve_mib') != reserve
                    or report.get('synthetic_reserve_bytes') != reserve * 1024**2):
                raise ValueError(f'{stage} reserve contract mismatch')
        current_matches = source_manifest() == sources
        selected_plan = None
        if any('selected_plan' in report for report in reports.values()):
            for report in reports.values():
                selected_plan = require_selected_plan(report, selected_plan)
        return dict(schema=1, status='preflight_passed', contract=contract,
            selected_plan=selected_plan,
            matches_initial_serialization=None if selected_plan is None else
                selected_plan['config'] == self.config.to_dict(),
            environment=environment, current_sources_match=current_matches,
            launch_sources_match_preparation=sources == request.get('prepared_source_hashes'),
            requires_requalification=not current_matches,
            capacity={stage: {key: reports[stage][key] for key in
                ('reserve_mib', 'first_iteration_peak_allocated',
                 'first_iteration_peak_reserved', 'capacity_gate', 'live_memory_gate')}
                for stage in ('cold', 'reuse')},
            numerics=reports['numerics']['cases'],
            historical_fresh_process_reuse=True, installs_runtime_winner=False,
            gaps=(['historical reports do not identify the executed selection']
                  if selected_plan is None else []) + ['current durable-cache contents not checked',
                  'only this exact contiguous-2D phase/activation is covered',
                  'BF16 replacement decision and application acceptance still required'])

    def check_cache(self, output, *, device='cuda:0'):
        """Read current strict-cache eligibility for this qualified context.

        May initialize CUDA to fingerprint the requested device, but launches
        no kernels, tuning or compilation. Requires current source/environment
        qualification and compares the eligible winner with the executed plan.
        This is a point-in-time observation, not a lock, installation or freeze.
        """
        import torch
        from ._mlp_autotune import _runtime_variant
        from .autotune.execution_mix import MLPExecutionMix
        from .autotune.mlp import MLP_AUTOTUNE_REVISION
        from .autotune.winners import runtime_winner_key, load_runtime_winner
        from .kernels.mxfp8 import MXFP8Problem
        from .mlp import MXFP8MLPConfig

        inspection = self.inspect(output)
        if not inspection['current_sources_match'] or inspection['selected_plan'] is None:
            raise RuntimeError('current-source executed-plan qualification required before cache coverage check')
        actual_environment = dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                                  gpu=torch.cuda.get_device_name(device))
        if actual_environment != inspection['environment']:
            raise RuntimeError('qualification environment differs from requested runtime environment')
        spec = json.loads((Path(output) / 'spec.json').read_text())
        mix_values = spec.get('execution_mixes', {}).get(self.phase)
        mix = None if mix_values is None else MLPExecutionMix(**mix_values)
        m, k, h, o = self.shape
        key = runtime_winner_key('mxfp8_recompute_mlp', MXFP8Problem(m, 2*h, k),
            device=device, kernel_revision=MLP_AUTOTUNE_REVISION,
            variant=_runtime_variant(o, self.activation, self.phase == 'train',
                                     spec['initial'], False, mix))
        # Same eligibility as require_cached_(): never increase effort or fall
        # back to the initial config. A missing/invalid winner is a coverage gap.
        selected = load_runtime_winner(key, MXFP8MLPConfig.from_dict,
                                       root=spec['cache_dir'])
        config = None if selected is None else selected.to_dict()
        qualified = inspection['selected_plan']['config']
        status = ('missing_or_ineligible' if config is None else
                  'covered' if config == qualified else 'winner_changed')
        return dict(schema=1, status=status, contract=inspection['contract'],
            winner_path=str(Path(spec['cache_dir']).expanduser() / key.relative_path),
            selected_config=config, qualified_config=qualified,
            requires_requalification=status != 'covered',
            installs_runtime_winner=False, point_in_time_only=True,
            remaining_requirements=['caller must prohibit live tuning before compilation',
                'cache files can change after this check',
                'other shapes/phases/layouts are not covered',
                'explicit BF16 replacement and application acceptance required'])
