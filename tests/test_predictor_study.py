import json
from pathlib import Path
import tempfile
import unittest

from tools.predictor_study import make_tasks, observed_geometry, task_policy, summarize, atomic


class PredictorStudyTests(unittest.TestCase):
    def test_mixed_treatments_preserve_baseline_and_effort(self):
        document = dict(task_seconds=30, artifact='frozen.json',
                        timing=dict(warmup=1, samples=3, calls_per_sample=2))
        treatments = ('existing', 'mixed', 'residual_mlp', 'mixed_residual_mlp')
        tasks = make_tasks([dict(m=384,k=640,n=1280)], (8,16,32,64), 1,
                           families=('mxfp8_bwd',), treatments=treatments)
        self.assertEqual(len(tasks), 16)
        for task in tasks:
            p = task_policy(task, document)
            self.assertEqual(p.max_trials, task['budget'])
            self.assertEqual(p.candidate_mix, 'mixed' if task['treatment'].startswith('mixed') else 'legacy')
            self.assertEqual(p.predictor_artifact is None, task['treatment'] in ('existing', 'mixed'))

    def test_balanced_tasks_and_policies(self):
        shapes = [dict(m=384,k=640,n=1280), dict(m=3072,k=1280,n=2560)]
        tasks = make_tasks(shapes, (8,16,32), 2)
        self.assertEqual(len(tasks), 384)
        self.assertEqual(len({t['id'] for t in tasks}), len(tasks))
        self.assertTrue(all(t['budget']==8 for t in tasks[:128]))
        for i in range(0,len(tasks),4):
            self.assertEqual(len({t['treatment'] for t in tasks[i:i+4]}),4)
            self.assertEqual(len({t['family'] for t in tasks[i:i+4]}),1)
        document = dict(task_seconds=120, artifact='frozen.json',
                        timing=dict(warmup=3,samples=7,calls_per_sample=5))
        for t in tasks[:4]:
            policy = task_policy(t,document)
            self.assertEqual(policy.max_trials,8)
            self.assertEqual(policy.exploration_trials,int(t['treatment']!='existing'))
            if t['treatment']=='retrieval':
                self.assertEqual(policy.predictor_heads,())

    def test_strict_unseen_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'observations.jsonl'
            path.write_text(json.dumps(dict(features={'context.workload.m':384,'context.workload.k':640}))+ '\n')
            seen,count=observed_geometry([tmp])
            self.assertEqual((seen,count),({(384,640)},1))
            path.write_text('{broken\n')
            with self.assertRaises(json.JSONDecodeError): observed_geometry([tmp])

    def test_paired_summary_preserves_failed_cells(self):
        tasks=make_tasks([dict(m=384,k=640,n=1280)],(8,),1,families=('mxfp8_bwd',))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            atomic(root/'plan.json',dict(tasks=tasks))
            for i,t in enumerate(tasks):
                atomic(root/'tasks'/t['id']/'result.json',dict(task=t,status='ok',attempts=8,verified_ms=1+i))
            summarize(root)
            rows=json.loads((root/'summary.json').read_text())['rows']
            self.assertTrue(all(r['matched_complete'] for r in rows))
            atomic(root/'tasks'/tasks[0]['id']/'result.json',dict(task=tasks[0],status='worker_failed'))
            summarize(root)
            rows=json.loads((root/'summary.json').read_text())['rows']
            self.assertTrue(all(r['observed_regret_pct'] is None for r in rows))
            self.assertEqual(len(rows),4)

    def test_analysis_excludes_unknown_contracts_and_pairs_control(self):
        from tools.analyze_predictor_study import comparisons
        rows=[]
        for shape in ('small','large'):
            for treatment, ms in [('existing',2.),('retrieval',1.5),('residual',1.),('residual_mlp',1.25)]:
                rows.append(dict(family='test',shape=shape,replicate=0,budget=8,
                    treatment=treatment,matched_complete=True,contract_compatible=True,
                    verified_ms=ms,tuning_wall_s=3.))
        output=comparisons(rows)
        residual=next(r for r in output if r['control']=='existing' and r['treatment']=='residual')
        self.assertEqual(residual['geometric_mean_speedup'],2.)
        self.assertEqual(residual['pairs'],2)
        self.assertEqual(residual['exploratory_cluster_ci95'],[2.,2.])
        for row in rows:
            row['contract_compatible']=False
        self.assertEqual(comparisons(rows),[])


if __name__=='__main__': unittest.main()
