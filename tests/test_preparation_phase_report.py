import json

import pytest

from benchmarks.profile_preparation_phases import completed_winner


@pytest.mark.parametrize("fault", [None, "unfinished", "shape", "unstable", "fused"])
def test_completed_winner_gates(tmp_path, fault):
    root = tmp_path / "residuals"
    root.mkdir()
    start = dict(event="start", session_id="s", context_id="c",
                 context={"workload": dict(m=128, n=256, k=384)})
    finish = dict(event="finish", session_id="s", best_config_id="winner")
    sessions = [start] if fault == "unfinished" else [start, finish]
    (root / "sessions.jsonl").write_text("\n".join(map(json.dumps, sessions)))
    row = dict(context_id="c", config_id="winner",
               config=dict(implementation="fused" if fault == "fused" else "materialized",
                           config={"sentinel": 123}),
               outcome=dict(status="ok", median_ms=.1, metadata={"sampling": {
                   "collection": {"stationary": fault != "unstable"}}}))
    (root / "observations.jsonl").write_text(json.dumps(row))
    shape = (1, 2, 3) if fault == "shape" else (128, 256, 384)
    if fault:
        with pytest.raises(ValueError):
            completed_winner(tmp_path, shape, "materialized")
    else:
        assert completed_winner(tmp_path, shape, "materialized") == {"sentinel": 123}
