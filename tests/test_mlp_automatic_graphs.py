"""Fresh-process regression for the mixed-device automatic partition crash."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")
@pytest.mark.parametrize("selection",["explicit","first_hit"])
def test_automatic_graph_mode_with_changing_operands(tmp_path,selection):
    root=Path(__file__).resolve().parents[1]
    output=tmp_path/f"{selection}.json"
    command=[sys.executable,str(root/"tools/probe_mlp_automatic_graphs.py"),
             "--selection",selection,"--output",str(output)]
    with (tmp_path/f"{selection}.log").open("w") as log:
        result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=120)
    assert result.returncode==0,(tmp_path/f"{selection}.log").read_text()
    report=json.loads(output.read_text())
    assert len(report["steps"])==8
    assert report["data_update_steps"]==[3,6]
    assert max(error for step in report["steps"] for error in step["relative_l2"])<.025
    if selection=="explicit":
        assert all(step["forward_launcher_calls"]==0 for step in report["steps"][2:])
    else:
        for step in report["steps"][3:]:
            assert step["forward_launcher_calls"]==0,report
            assert step["backward_launcher_calls"]==0,report


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")
@pytest.mark.parametrize("shape,activation,phase,mask",[
    ("17,35,97,41","gelu","train","all"),
    ("64,128,256,64","silu","train","x"),
    ("64,128,256,64","gelu","train","gu"),
    ("64,128,256,64","silu","train","down"),
    ("512,768,1536,768","silu","forward","all"),
])
def test_nested_replay_matrix(tmp_path,shape,activation,phase,mask):
    root=Path(__file__).resolve().parents[1]
    output=tmp_path/"nested.json"
    command=[sys.executable,str(root/"tools/probe_mlp_automatic_graphs.py"),
        "--selection","first_hit","--nested-replay","--output",str(output),
        "--shape",shape,"--activation",activation,"--phase",phase,"--mask",mask]
    env=dict(os.environ,TORCHINDUCTOR_CACHE_DIR=str(tmp_path/"inductor"))
    with (tmp_path/"nested.log").open("w") as log:
        result=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
    assert result.returncode==0,(tmp_path/"nested.log").read_text()
    report=json.loads(output.read_text())
    assert report["passed"] and len(report["steps"])==8
    assert max(error for step in report["steps"] for error in step["relative_l2"])<.025
    for step in report["steps"][3:]:
        assert step["forward_launcher_calls"]==0,report
        assert step["backward_launcher_calls"]==0,report


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")
@pytest.mark.parametrize("case",["outstanding","external"])
def test_nested_replay_lifetime(tmp_path,case):
    root=Path(__file__).resolve().parents[1]
    output=tmp_path/f"{case}.json"
    env=dict(os.environ,TORCHINDUCTOR_CACHE_DIR=str(tmp_path/"inductor"))
    with (tmp_path/"probe.log").open("w") as log:
        result=subprocess.run([sys.executable,str(root/"tools/probe_mlp_replay_lifetime.py"),
            "--case",case,"--output",str(output)],cwd=root,env=env,
            stdout=log,stderr=subprocess.STDOUT,timeout=180)
    assert result.returncode==0,(tmp_path/"probe.log").read_text()
    report=json.loads(output.read_text())
    assert report["passed"]
    assert len(report["errors"])==8 and max(report["errors"])<.025
    if case=="outstanding":
        assert len(report["live_bytes"])==4
        assert len(set(report["live_bytes"]))==1
