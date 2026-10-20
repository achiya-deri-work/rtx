"""Explicit replacement recommendations; never mutate a model or its cache."""
from dataclasses import asdict, dataclass
import math
import statistics
from itertools import permutations


BACKENDS = ('bf16', 'bf16_recompute', 'mxfp8')


def comparison_orders(rounds):
    """All six permutations balance position and pairwise precedence."""
    if type(rounds) is not int or rounds < 6 or rounds % 6:
        raise ValueError('balanced comparison requires a positive multiple of six rounds')
    orders = tuple(permutations(BACKENDS))
    return [orders[i % 6] for i in range(rounds)]


def summarize_repeated_comparison(rounds, *, policy=None):
    """Preserve within-round pairing; do not manufacture qualification or CIs."""
    if not rounds:
        raise ValueError('no comparison rounds')
    comparison_orders(len(rounds))
    contract = rounds[0][0]['contract']
    sources = rounds[0][0].get('source_hashes')
    provenance = {}
    for batch, order in zip(rounds, comparison_orders(len(rounds))):
        if tuple(r['backend'] for r in batch) != order:
            raise ValueError('worker order is not balanced')
        if sorted(r['backend'] for r in batch) != sorted(BACKENDS):
            raise ValueError('each round requires all three unique backends')
        for r in batch:
            if r['contract'] != contract or r['status'] != 'ok':
                raise ValueError('failed or mismatched worker contract')
            if r.get('source_hashes') != sources:
                raise ValueError('source/config changed across comparison workers')
            identity = (r.get('source_hashes'), r.get('config'), r.get('config_source'))
            if r['backend'] in provenance and identity != provenance[r['backend']]:
                raise ValueError('source/config changed across rounds')
            provenance[r['backend']] = identity
            if any(not math.isfinite(v['median_ms']) or v['median_ms'] <= 0
                   for v in r['timings'].values()):
                raise ValueError('invalid timing')
    summary = []
    paired = {}
    for backend in BACKENDS:
        rows = [next(r for r in batch if r['backend'] == backend) for batch in rounds]
        phases = rows[0]['timings'].keys()
        if any(r['timings'].keys() != phases for r in rows):
            raise ValueError('timing phases differ')
        timing = {phase: dict(median_ms=statistics.median(r['timings'][phase]['median_ms'] for r in rows),
                             process_medians_ms=[r['timings'][phase]['median_ms'] for r in rows])
                  for phase in phases}
        ratios = [r['timings']['workload']['median_ms'] /
                  next(b for b in batch if b['backend']=='bf16')['timings']['workload']['median_ms']
                  for r, batch in zip(rows, rounds)]
        paired[backend] = dict(ratios=ratios,median=statistics.median(ratios),
                               minimum=min(ratios),maximum=max(ratios))
        peaks = [r['memory']['warm_peak_allocated'] for r in rows]
        if any(type(p) is not int or p < 0 for p in peaks):
            raise ValueError('invalid memory evidence')
        summary.append(dict(backend=backend,status='ok',contract=contract,
            environment_qualified=False, qualification_note='order balanced, contention not qualified',
            timings=timing,memory=dict(warm_peak_allocated=max(peaks),process_peaks=peaks)))
    return dict(reports=summary,paired_workload_ratios=paired,
                inference='descriptive paired ratios, not confidence intervals',
                decision=recommend_replacement(summary, policy or ReplacementPolicy()))


@dataclass(frozen=True)
class ReplacementPolicy:
    prefer_memory: bool = False
    max_slowdown: float = 0.0
    minimum_speedup: float = .01

    def __post_init__(self):
        if type(self.prefer_memory) is not bool:
            raise TypeError('prefer_memory must be bool')
        if not math.isfinite(self.max_slowdown) or self.max_slowdown < 0:
            raise ValueError('max_slowdown must be finite and nonnegative')
        if not math.isfinite(self.minimum_speedup) or not 0 <= self.minimum_speedup < 1:
            raise ValueError('minimum_speedup must be in [0,1)')
        if self.max_slowdown and not self.prefer_memory:
            raise ValueError('slower replacements require explicit prefer_memory=True')


def recommend_replacement(reports, policy=ReplacementPolicy()):
    """Compare qualified, same-contract reports against ordinary compiled BF16.

    Isolated peaks are not full-model capacity estimates. This recommendation
    grants no permission to change a caller's numerical policy automatically.
    """
    reports=list(reports)
    names=[r['backend'] for r in reports]
    if len(set(names))!=len(names) or 'bf16' not in names:
        raise ValueError('unique backends and an ordinary BF16 baseline are required')
    baseline=next(r for r in reports if r['backend']=='bf16')
    if any(r['contract']!=baseline['contract'] for r in reports):
        raise ValueError('replacement contracts differ')
    for r in reports:
        value=r['timings']['workload']['median_ms']
        peak=r['memory']['warm_peak_allocated']
        if not math.isfinite(value) or value<=0 or type(peak) is not int or peak<0:
            raise ValueError('invalid timing or memory evidence')
    bt=baseline['timings']['workload']['median_ms']
    bp=baseline['memory']['warm_peak_allocated']
    comparisons={r['backend']: dict(
        workload_ms=r['timings']['workload']['median_ms'],
        time_ratio=r['timings']['workload']['median_ms']/bt,
        warm_peak_allocated=r['memory']['warm_peak_allocated'],
        peak_saved_bytes=bp-r['memory']['warm_peak_allocated'],
        environment_qualified=r['environment_qualified'],status=r['status'])
        for r in reports}
    evidence=dict(policy=asdict(policy),comparisons=comparisons,
                  requires_explicit_application_acceptance=True)
    if any(r['status']!='ok' or not r['environment_qualified'] for r in reports):
        return dict(status='unqualified',selected=None,baseline='bf16',**evidence)
    eligible=[baseline]
    for r in reports:
        t=r['timings']['workload']['median_ms']
        p=r['memory']['warm_peak_allocated']
        if t < bt*(1-policy.minimum_speedup) or (policy.prefer_memory and p<bp and t<=bt*(1+policy.max_slowdown)):
            eligible.append(r)
    chosen=min(eligible,key=lambda r:(r['memory']['warm_peak_allocated'],r['timings']['workload']['median_ms'])
               if policy.prefer_memory else (r['timings']['workload']['median_ms'],r['memory']['warm_peak_allocated']))
    return dict(status='recommendation_only',selected=chosen['backend'],baseline='bf16',
        time_ratio=chosen['timings']['workload']['median_ms']/bt,
        peak_saved_bytes=bp-chosen['memory']['warm_peak_allocated'],
        slower_memory_tradeoff=chosen['timings']['workload']['median_ms']>bt,
        **evidence)
