"""Research-only post-opening phase-balanced local proposals.

No kernel, evaluator, winner gate or public default changes. Run from a frozen
source snapshot, with a separate output root for this search contract:
  python -m benchmarks.astra32_phase_search --module benchmarks.benchmark_moe_varlen -- <driver arguments>
"""
from collections import Counter
from dataclasses import replace


REVISION = 2


def phase_group(family, coordinate):
    """Map executed coordinate ownership, not SKU or shape winner labels."""
    name=coordinate or ''
    if family=='mxfp8_recompute_mlp':
        for leg in ('gate_backward.dx','gate_backward.dw','down_backward.dx','down_backward.dw'):
            if name==leg or name.startswith((leg+'.',leg+'_')):return leg
        if name.startswith(('gate_backward.','down_backward.','backward_')):
            return 'backward_schedule'
        if name.startswith(('gate_up.','cooperative_','up_weight_quant.','up_quant_','gate_backend')):
            return 'gate_forward'
        if name.startswith(('down.','down_gemm.','down_quant.','down_backend')):
            return 'down_forward'
        if name.startswith('recompute'):return 'recompute'
        return 'cross_stage'
    if family=='mxfp8_balanced_experts':
        for stage in ('gate_up','down_dx','down_dw','gate_dx','gate_dw','recompute','down'):
            if name.startswith(('prepared.'+stage+'.','prepared.'+stage+'_')):return stage
        if name.startswith(('combine_','dx_','movement_','gather_')):return 'movement'
        if name.endswith('_recipe'):return 'coupled_recipe'
        if name.startswith(('prepared.quant.','prepared.transpose','prepared.weight_')):return 'preparation'
        return 'execution_layout'
    return name.split('.')[0]


def counts(adapter, history):
    phases=Counter();coordinates=Counter()
    # Failures count as attempted coverage too; unstable successes do not get
    # a free repeat, and they never change the inner search's winner rule.
    for row in history.current:
        if getattr(row,'coordinate',None):
            coordinate=row.coordinate
            phases[phase_group(adapter.context.family,coordinate)]+=1
            coordinates[coordinate]+=1
    return phases,coordinates


class PhaseBalancedSearch:
    """Retain the inner strategy's anchors, rebalance only proposal ownership."""
    def __init__(self,inner,*,after=8):
        self.inner=inner;self.name=inner.name;self.after=after

    def propose(self,adapter,history,rng,limit):
        if len(history.current)<self.after:return self.inner.propose(adapter,history,rng,limit)
        phases,coordinates=counts(adapter,history)
        def priority(coordinate):
            return phases[phase_group(adapter.context.family,coordinate)],coordinates[coordinate]
        if self.name=='coordinate_local':
            # Reorder coordinates before generation, so a broad phase late in
            # the domain is not truncated out of a finite proposal batch.
            axes=dict(sorted(adapter.axes.items(),key=lambda item:priority(item[0])))
            return self.inner.propose(replace(adapter,axes=axes),history,rng,limit)
        # The prepared component arm owns a separate compound recipe list.
        # Preserve all of its legal candidates/anchors, but do not always take
        # the first ownership recipe while seven GEMM stages remain untouched.
        proposals=self.inner.propose(adapter,history,rng,max(limit,4096))
        proposals.sort(key=lambda p:priority(p.coordinate or ''))
        return proposals[:limit]

    def observe(self,observation):return self.inner.observe(observation)


def balance_tuner(tuner):
    from rtx.autotune.core import stable_id
    for name in ('coordinate_local','prepared_components'):
        strategy=tuner.strategies.get(name)
        if strategy is None:continue
        if type(strategy).__name__=='MoEFamilySearch':
            # Keep the existing on-chip family admission outside this wrapper.
            strategy.inner=PhaseBalancedSearch(strategy.inner)
        else:tuner.strategies[name]=PhaseBalancedSearch(strategy)
    if tuner.checkpoint_key is not None:
        tuner.checkpoint_key=stable_id(dict(base=tuner.checkpoint_key,phase_coverage_revision=REVISION))
    return tuner


def install():
    """Only explicit offline research drivers use this monkeypatch."""
    from rtx.autotune import mlp,moe
    def wrap(original):
        def make(adapter,*args,**kwargs):
            context=replace(adapter.context,tags={**adapter.context.tags,
                'research_phase_coverage_revision':REVISION})
            return balance_tuner(original(replace(adapter,context=context),*args,**kwargs))
        return make
    mlp.make_mlp_tuner=wrap(mlp.make_mlp_tuner)
    moe.make_moe_tuner=wrap(moe.make_moe_tuner)


def main():
    import argparse
    import runpy
    import sys
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module',required=True)
    parser.add_argument('arguments',nargs=argparse.REMAINDER)
    args=parser.parse_args();install()
    arguments=args.arguments[1:] if args.arguments[:1]==['--'] else args.arguments
    sys.argv=[args.module,*arguments]
    runpy.run_module(args.module,run_name='__main__')


if __name__=='__main__':main()
