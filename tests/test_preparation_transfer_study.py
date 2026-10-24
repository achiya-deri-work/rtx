from benchmarks.compare_preparation_transfer import compare


def row(context, config_id, latency, **features):
    return {"context_id":context,"config_id":config_id,
        "features":{"context.workload.m":128.,"context.regime=hot":1.,
                    "context.device.multiprocessor_count":40. if context=="a" else 170.,**features},
        "config":{"config":{"dx":{"backend":"decomposed"},"dw":{"backend":"decomposed"}}},
        "outcome":{"status":"ok","median_ms":latency}}


def test_transfer_is_matched_and_uses_target_timings():
    rows=[row("a","x",1.),row("a","y",2.),row("b","x",4.),row("b","y",2.),
          row("b","unmatched",.01)]
    result=compare(rows)
    ab=next(c for c in result["comparisons"] if c["source_context"]=="a" and c["scope"]=="all")
    assert ab["matched_configs"]==2
    assert ab["transfer_regret"]["1"]==1.
    assert ab["transfer_regret"]["2"]==0.
    assert ab["discordant_pairs"]==1


def test_transfer_rejects_different_contract_and_marks_near_ties():
    assert not compare([row("a","x",1.),row("b","x",2.,**{"context.tags.policy=changed":1.})])["comparisons"]
    result=compare([row("a","x",1.),row("a","y",1.01),row("b","x",2.),row("b","y",3.)])
    assert all(c["near_tied_pairs"]==1 for c in result["comparisons"])


def test_transfer_excludes_explicitly_unstable_successes():
    a,b=row("a","x",1.),row("b","x",2.)
    b["outcome"]["metadata"]={"sampling":{"collection":{"stationary":False}}}
    assert not compare([a,b])["comparisons"]
