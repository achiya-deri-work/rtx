"""Matched packed-router opening A/B with frozen whole-MoE FFNs on HOME GPU0.

Run from the repository with isolated baseline and candidate source bundles.
Each 4→8 arm uses a fresh cache, then continues the same residual journal in a
second fresh process. Hold the common GPU0 lock through each supervised visit.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

from rtx.autotune.supervisor import supervise_command


PROFILES={
    'evidence': ('packed_uneven','d384_e32_k4_h128',(127,513,2001,767,4095,689),'row_broadcast'),
    'heldout_skew': ('packed_uneven','d384_e32_k4_h128',(33,95,257,769,1703,5335),'scalar'),
    'heldout_balanced': ('original_384_top6','d384_e24_k6_h128',(256,)*8,'scalar'),
    'short_control': ('original_384_top6','d384_e24_k6_h128',(128,)*16,'scalar'),
    'single_control': ('original_384_top6','d384_e24_k6_h128',(2048,),'scalar'),
}
# The inherited E24 FFN uses a full-tile TMA epilogue from a single-pool
# context. Keep that plan and pad its physical expert capacity equally in
# both arms when the new pack has a partial M tile.
EXPERT_OVERRIDES={name:{'prepared':{'capacity_multiple':128}}
                  for name in ('heldout_balanced','short_control')}
LOCK=Path('/tmp/rtx-moe-backward-20261119-gpu0.lock')


def digest(source):
    value=hashlib.sha256()
    for path in sorted((source/'rtx').rglob('*.py')):
        value.update(str(path.relative_to(source)).encode())
        value.update(path.read_bytes())
    return value.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def observations(root):
    paths=list((root/'varlen_routing_effort').glob('*/*/observations.jsonl'))
    if len(paths)!=1:raise AssertionError(paths)
    return [json.loads(line) for line in paths[0].read_text().splitlines()]


def qualify(args, profile, sources, fingerprints):
    frozen,case,lengths,affinity=PROFILES[profile]
    root=args.output/profile
    root.mkdir(parents=True,exist_ok=False)
    ref=root/'baseline-4.pt'
    fixture=args.evidence/frozen
    source_result=read(fixture/'result.json')
    spec={key:source_result['case'][key] for key in (
        'd_model','num_experts','top_k','expert_hidden_dim','shared_hidden_dim')}
    records={}
    baseline_identity=None
    for arm,budget in (('baseline',4),('candidate',4),('baseline',8),('candidate',8)):
        name=f'{arm}-{budget}'
        source=sources[arm]
        out=root/name/'result.json'
        out.parent.mkdir()
        tuning=root/f'tuning-{arm}'
        command=[sys.executable,'-m','benchmarks.benchmark_moe_varlen','--source',str(source),
            '--output',str(out),'--case-id',case,'--case-spec',json.dumps(spec),'--lengths',*map(str,lengths),
            '--hidden-alignment','32','--seed','17391','--input-profile','normal','--affinity-layout',affinity,
            '--expert-config',str(fixture/'result.expert.json'),'--shared-config',str(fixture/'result.shared.json'),
            '--expert-overrides',json.dumps(EXPERT_OVERRIDES.get(profile,{})),
            '--oracle','--max-autotune','--search-schedule','progressive','--candidate-mix','mixed',
            '--tuning-dir',str(tuning),'--tune-router-trials',str(budget),'--tune-seconds','90',
            '--rounds','2','--samples','7','--phases','mxfp8_step','bf16_checkpoint_step','router_step',
            *(['--save-reference',str(ref)] if name=='baseline-4' else ['--reference',str(ref)])]
        if digest(source)!=fingerprints[arm]:raise AssertionError('Runtime changed before visit')
        print('VISIT',profile,name,flush=True)
        write(out.parent/'command.json',command)
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','CUDA_DEVICE_ORDER':'PCI_BUS_ID'}
        with LOCK.open('r') as lock, (out.parent/'run.log').open('w') as log:
            fcntl.flock(lock,fcntl.LOCK_EX)
            status=supervise_command(command,stall_timeout_s=120,active_stall_timeout_s=360,
                wall_timeout_s=600,cwd=source,environment=env,output=log)
        if status:raise RuntimeError(f'{profile}/{name} exited {status}; see {out.parent / "run.log"}')
        if digest(source)!=fingerprints[arm]:raise AssertionError('Runtime changed during visit')
        result=read(out)
        if result['status']!='complete' or result['source_digest']!=fingerprints[arm]:raise AssertionError(out)
        identity={key:result[key] for key in ('identities','expert_config','shared_config','router_affinity_layout')}
        if baseline_identity is None:baseline_identity=identity
        elif baseline_identity!=identity:raise AssertionError(f'Operative identity changed at {profile}/{name}')
        if profile=='evidence' and source_result['identities']!=result['identities']:
            raise AssertionError('Known evidence operands/routes did not reproduce')
        rows=observations(tuning)
        if len(rows)!=budget or len({r['config_id'] for r in rows})!=budget:
            raise AssertionError(f'{profile}/{name} ended with {len(rows)} unique trial records; see journal')
        records[name]=dict(timing_ms={phase:statistics.median(
            sample for round_ in rounds for sample in round_['samples_ms']) for phase,rounds in result['timings'].items()},
            trials=[{key:r[key] for key in ('strategy','config_id','config','outcome')} for r in rows],
            router_config=result['router_config'],tuning_counts=result['tuning_counts'],validation=result['validation'],
            memory=result['memory'],source_digest=result['source_digest'])
        if budget==8 and records[name]['trials'][:4]!=records[arm+'-4']['trials']:
            raise AssertionError('4→8 continuation changed admitted prefix')
        print('PASS',profile,name,records[name]['timing_ms'],flush=True)
        write(root/'summary.json',records)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','candidate','evidence','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--profiles',nargs='+',choices=tuple(PROFILES),default=list(PROFILES)[:3])
    args=p.parse_args()
    for name in ('baseline','candidate','evidence','output'):setattr(args,name,getattr(args,name).resolve())
    args.output.mkdir(parents=True,exist_ok=True)
    sources={key:getattr(args,key) for key in ('baseline','candidate')}
    fingerprints={key:digest(source) for key,source in sources.items()}
    for path in (Path('benchmarks/benchmark_moe_varlen.py'),Path('benchmarks/benchmark_moe_checkpoint.py')):
        if (args.baseline/path).read_bytes()!=(args.candidate/path).read_bytes():
            raise AssertionError(f'Benchmark driver/reference differed: {path}')
    for profile in args.profiles:qualify(args,profile,sources,fingerprints)


if __name__=='__main__':main()
