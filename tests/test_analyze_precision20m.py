import json
import xml.etree.ElementTree as ET

import pytest

from benchmarks import analyze_precision20m as analysis


def make_run(tmp_path,mode="mxfp8",sku="5070",*,points=None,env_changes=None,blocks=None,allow_partial=False):
    environment=dict(event="environment",sku=sku,mode=mode,seed=941,params=20_000_000,
        batch=2,sequence=128,steps=20,total_train_tokens=5120,
        model_config={"width":512},optimizer={"name":"AdamW","lr":.001},
        attention="flash",precision={"matrix_parameters":"BF16","cudagraphs":False},
        initial_weights_sha256="weights",corpus_sha256="corpus",validation_offsets_sha256="offsets")
    environment.update(env_changes or {})
    points=points if points is not None else [(0,0,4.,0.,2.),(10,2560,3.,10.,14.),(20,5120,2.,20.,27.)]
    events=[environment,dict(event="compile_warmup",compile_warmup_wall_s=2.)]
    for step,tokens,loss,train,e2e in points:
        events.append(dict(event="validation",step=step,tokens=tokens,validation_loss=loss,
                           training_wall_s=train,end_to_end_wall_s=e2e,validation_wall_s=1.))
    for tokens,seconds in (blocks if blocks is not None else [(100,1.),(100,2.),(100,4.)]):
        events.append(dict(event="train",block_tokens=tokens,block_training_wall_s=seconds,
                           block_tokens_per_s=999999.))
    events.append(dict(event="result",step=points[-1][0],tokens=points[-1][1],
                       training_wall_s=points[-1][3],end_to_end_wall_s=points[-1][4],finite_parameters=True))
    path=tmp_path/f"{sku}_{mode}.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in events)+"\n")
    return analysis.read_run(path,allow_partial=allow_partial)


def test_equal_work_and_conservative_actual_wall_checkpoints(tmp_path):
    baseline=make_run(tmp_path)
    candidate=make_run(tmp_path,"nvfp4_full",points=[(0,0,4.,0.,2.),(10,2560,3.2,8.,15.),(20,5120,2.2,16.,30.)])
    result=analysis.analyze([baseline,candidate])["skus"]["5070"]
    match=result["equal_step_tokens"]["nvfp4_full"]["last_actual_match"]
    assert match["step"]==20 and match["validation_loss_delta_vs_mxfp8"]==pytest.approx(.2)
    wall=result["equal_training_wall"]
    assert wall["shared_budget_s"]==16.
    assert wall["results"]["mxfp8"]["step"]==10
    assert wall["results"]["nvfp4_full"]["step"]==20
    assert wall["results"]["mxfp8"]["checkpoint_age_s"]==6.
    assert wall["results"]["nvfp4_full"]["validation_loss_delta_vs_mxfp8"]==pytest.approx(-.8)
    # Inclusive wall has a different conservative checkpoint selection.
    inclusive=result["equal_end_to_end_wall"]
    assert inclusive["shared_budget_s"]==27.
    assert inclusive["results"]["mxfp8"]["step"]==20
    assert inclusive["results"]["nvfp4_full"]["step"]==10


def test_hybrid_baseline_validates_forward_and_labels_deltas_correctly(tmp_path):
    extra=dict(projection_configs={"(128,128,128)":dict(forward={"scale":"exact"})})
    baseline=make_run(tmp_path,"nvfp4_mxfp8",env_changes=extra)
    candidate=make_run(tmp_path,"nvfp4_full",env_changes=extra,
        points=[(0,0,4.,0.,2.),(10,2560,3.1,11.,15.),(20,5120,2.1,22.,29.)])
    result=analysis.analyze([baseline,candidate],baseline_mode="nvfp4_mxfp8")
    assert result["baseline_mode"]=="nvfp4_mxfp8"
    point=result["skus"]["5070"]["equal_step_tokens"]["nvfp4_full"]["last_actual_match"]
    assert point["validation_loss_delta_vs_baseline"]==pytest.approx(.1)
    assert "validation_loss_delta_vs_mxfp8" not in point
    candidate["environment"]["projection_configs"]={"(128,128,128)":dict(forward={"scale":"different"})}
    with pytest.raises(ValueError,match="mismatched forward"):
        analysis.analyze([baseline,candidate],baseline_mode="nvfp4_mxfp8")
    del baseline["environment"]["projection_configs"]
    with pytest.raises(ValueError,match="forward provenance"):
        analysis.analyze([baseline,candidate],baseline_mode="nvfp4_mxfp8")


def test_step_alone_is_not_equal_work(tmp_path):
    baseline=make_run(tmp_path)
    candidate=make_run(tmp_path,"nvfp4_full",points=[(10,3000,3.1,8.,12.),(20,5120,2.1,16.,24.)])
    result=analysis.analyze([baseline,candidate])["skus"]["5070"]
    assert [row["step"] for row in result["equal_step_tokens"]["nvfp4_full"]["actual_matching_checkpoints"]]==[20]


def test_threshold_first_observation_no_interpolation_or_extrapolation(tmp_path):
    run=make_run(tmp_path)
    results=analysis.observed_thresholds({"mxfp8":run},[2.5,1.0])
    observed=results[0]["results"]["mxfp8"]
    assert observed["first_observed_checkpoint"]["step"]==20
    assert observed["first_observed_checkpoint"]["training_wall_s"]==20.
    assert results[1]["results"]["mxfp8"]["status"]=="not_observed"
    assert results[1]["results"]["mxfp8"]["first_observed_checkpoint"] is None


def test_default_threshold_is_observed_baseline_final_loss(tmp_path):
    result=analysis.analyze([make_run(tmp_path)])["skus"]["5070"]
    assert result["time_to_loss_observed"][0]["validation_loss_threshold"]==2.
    assert "actually observed" in result["threshold_origin"]


def test_throughput_uses_raw_block_timer_median(tmp_path):
    run=make_run(tmp_path)
    assert run["throughput"]["samples_tokens_per_s"]==[100.,50.,25.]
    assert run["throughput"]["median_tokens_per_s"]==50.
    assert run["compile_warmup_wall_s"]==2.
    assert run["final_training_wall_s"]==20.
    assert run["final_end_to_end_wall_s"]==27.


def test_distinct_skus_never_pool_trajectories(tmp_path):
    result=analysis.analyze([make_run(tmp_path,sku="5060"),make_run(tmp_path,sku="5070")])
    assert set(result["skus"])=={"5060","5070"}


@pytest.mark.parametrize("field",analysis.MATCH_FIELDS)
def test_mismatched_training_contract_rejected(tmp_path,field):
    baseline=make_run(tmp_path)
    candidate=make_run(tmp_path,"nvfp4_full",env_changes={field:"changed"},allow_partial=True)
    with pytest.raises(ValueError,match="mismatched"):
        analysis.analyze([baseline,candidate],allow_partial=True)


def test_duplicate_sku_mode_and_missing_baseline_rejected(tmp_path):
    baseline=make_run(tmp_path)
    with pytest.raises(ValueError,match="duplicate"):
        analysis.analyze([baseline,baseline])
    with pytest.raises(ValueError,match="missing mxfp8"):
        analysis.analyze([make_run(tmp_path,"nvfp4_full")])


@pytest.mark.parametrize("bad",[0.,-1.,float("nan"),float("inf")])
def test_invalid_block_timers_rejected(tmp_path,bad):
    with pytest.raises(ValueError):
        make_run(tmp_path,blocks=[(100,bad)])


def test_nonmonotonic_validation_timeline_rejected(tmp_path):
    with pytest.raises(ValueError,match="monotone"):
        make_run(tmp_path,points=[(10,2560,3.,10.,15.),(20,5120,2.,9.,20.)])


def test_partial_line_only_optional_and_only_final(tmp_path):
    run=make_run(tmp_path)
    path=tmp_path/"partial.jsonl"
    path.write_text((tmp_path/"5070_mxfp8.jsonl").read_text()+'{"event":')
    with pytest.raises(ValueError,match="invalid JSONL"):
        analysis.read_run(path)
    replay=analysis.read_run(path,allow_partial=True)
    assert len(replay["warnings"])==1 and replay["validations"]==run["validations"]


def test_no_checkpoint_before_shared_budget_is_not_fabricated(tmp_path):
    early=make_run(tmp_path,points=[(10,2560,3.,10.,15.)],allow_partial=True)
    late=make_run(tmp_path,"nvfp4_full",points=[(10,2560,3.,20.,25.)],allow_partial=True)
    result=analysis.analyze([early,late],allow_partial=True)["skus"]["5070"]["equal_training_wall"]
    assert result["results"]["nvfp4_full"] is None


def test_cli_output_and_refuse_overwrite(tmp_path):
    make_run(tmp_path)
    source=tmp_path/"5070_mxfp8.jsonl"
    output=tmp_path/"summary.json"
    analysis.main([str(source),"--output",str(output),"--loss-threshold","2.5"])
    assert "5070" in json.loads(output.read_text())["skus"]
    with pytest.raises(FileExistsError):
        analysis.main([str(source),"--output",str(output)])


@pytest.mark.parametrize("updates,expected",[
    ({"interrupted":True},"interrupted"),
    ({"step":19},"incomplete"),
    ({"tokens":5000},"incomplete"),
    ({"finite_parameters":False},"nonfinite_parameters"),
    ({"finite_parameters":None},"incomplete"),
])
def test_incomplete_terminal_result_requires_explicit_partial(tmp_path,updates,expected):
    make_run(tmp_path)
    path=tmp_path/"5070_mxfp8.jsonl"
    events=[json.loads(line) for line in path.read_text().splitlines()]
    events[-1].update(updates)
    path.write_text("\n".join(json.dumps(row) for row in events)+"\n")
    with pytest.raises(ValueError,match=expected):
        analysis.read_run(path)
    partial=analysis.read_run(path,allow_partial=True)
    assert partial["status"]==expected and partial["partial_analysis"]
    with pytest.raises(ValueError,match="allow_partial"):
        analysis.analyze([partial])
    assert analysis.analyze([partial],allow_partial=True)["skus"]["5070"]["analysis_status"]=="partial_observations"


def test_error_and_missing_result_are_not_complete(tmp_path):
    make_run(tmp_path)
    path=tmp_path/"5070_mxfp8.jsonl"
    events=[json.loads(line) for line in path.read_text().splitlines()]
    events.pop()
    path.write_text("\n".join(json.dumps(row) for row in events)+"\n")
    with pytest.raises(ValueError,match="incomplete"):
        analysis.read_run(path)
    events.append(dict(event="error",step=20,error="failure",training_wall_s=20.,end_to_end_wall_s=28.))
    path.write_text("\n".join(json.dumps(row) for row in events)+"\n")
    with pytest.raises(ValueError,match="error"):
        analysis.read_run(path)
    assert analysis.read_run(path,allow_partial=True)["status"]=="error"


def test_partial_shared_budget_uses_last_actual_validation_not_later_terminal_wall(tmp_path):
    baseline=make_run(tmp_path)
    make_run(tmp_path,"nvfp4_full",points=[(0,0,4.,0.,2.),(10,2560,3.2,8.,15.)],allow_partial=True)
    path=tmp_path/"5070_nvfp4_full.jsonl"
    events=[json.loads(line) for line in path.read_text().splitlines()]
    events[-1].update(step=12,tokens=3072,training_wall_s=12.,end_to_end_wall_s=22.,interrupted=True)
    path.write_text("\n".join(json.dumps(row) for row in events)+"\n")
    partial=analysis.read_run(path,allow_partial=True)
    result=analysis.analyze([baseline,partial],allow_partial=True)["skus"]["5070"]
    assert result["analysis_status"]=="partial_observations"
    assert result["runs"]["nvfp4_full"]["final_training_wall_s"]==12.
    assert result["equal_training_wall"]["shared_budget_s"]==8.
    assert result["equal_training_wall"]["results"]["mxfp8"]["step"]==0
    assert result["equal_end_to_end_wall"]["shared_budget_s"]==15.
    assert result["equal_end_to_end_wall"]["results"]["mxfp8"]["step"]==10


def test_common_cudagraph_recipe_must_match(tmp_path):
    baseline=make_run(tmp_path)
    candidate=make_run(tmp_path,"nvfp4_full",env_changes={"precision":{"matrix_parameters":"BF16","cudagraphs":True}})
    with pytest.raises(ValueError,match="precision"):
        analysis.analyze([baseline,candidate])


def test_svg_actual_markers_exclude_step_zero_and_fit_bounds(tmp_path):
    result=analysis.analyze([make_run(tmp_path),make_run(tmp_path,sku="5060")])
    svg=analysis.validation_svg(result)
    root=ET.fromstring(svg)
    ns={"svg":"http://www.w3.org/2000/svg"}
    panels=root.findall("svg:g",ns)
    assert len(panels)==4  # two columns per device
    for panel in panels:
        points=[circle for circle in panel.findall("svg:circle",ns) if "data-step" in circle.attrib]
        assert [int(point.attrib["data-step"]) for point in points]==[10,20]
        assert all(float(panel.attrib["data-y-min"])<float(point.attrib["data-validation-loss"])<float(panel.attrib["data-y-max"]) for point in points)
        assert len(panel.findall("svg:polyline",ns))==1
    assert "Step 0 excluded" in svg and "visual guides" in svg and "Full raw observations" in svg
    assert result["skus"]["5070"]["runs"]["mxfp8"]["validation_checkpoints"][0]["step"]==0


def test_svg_escapes_titles(tmp_path):
    sku='GPU <script>& "quoted"'
    svg=analysis.validation_svg(analysis.analyze([make_run(tmp_path,sku=sku)]))
    root=ET.fromstring(svg)
    assert "<script>" not in svg and "&lt;script&gt;" in svg
    assert any(sku in (node.text or "") for node in root.iter())


def test_svg_empty_zoom_is_explicit(tmp_path):
    run=make_run(tmp_path,points=[(0,0,4.,0.,2.)],allow_partial=True)
    result=analysis.analyze([run],allow_partial=True)
    svg=analysis.validation_svg(result)
    assert "No post-step-0 validation checkpoints observed." in svg
    assert "data-step=" not in svg


@pytest.mark.parametrize("values",[[],[-1.],[float("nan")],[float("inf")]])
def test_plot_bounds_reject_invalid_values(values):
    with pytest.raises(ValueError):
        analysis.plot_bounds(values)


def test_plot_bounds_pad_flat_and_variable_observations():
    lo,hi=analysis.plot_bounds([2.,2.])
    assert 0<=lo<2.<hi
    lo,hi=analysis.plot_bounds([1.,100.])
    assert 0<=lo<1. and hi>100.


def test_svg_exclusive_write_and_cli_preflight(tmp_path):
    run=make_run(tmp_path)
    result=analysis.analyze([run])
    svg=tmp_path/"plot.svg"
    analysis.write_svg(result,svg)
    retained=svg.read_text()
    with pytest.raises(FileExistsError):
        analysis.write_svg(result,svg)
    assert svg.read_text()==retained
    output=tmp_path/"not_created.json"
    with pytest.raises(FileExistsError):
        analysis.main([run["path"],"--output",str(output),"--svg",str(svg)])
    assert not output.exists()


def test_cli_concise_default_and_explicit_verbose_json(tmp_path,capsys):
    run=make_run(tmp_path)
    output=tmp_path/"short.json"
    svg=tmp_path/"plot.svg"
    analysis.main([run["path"],"--output",str(output),"--svg",str(svg)])
    text=capsys.readouterr().out
    assert text.startswith("Analysis:") and "val=" in text and len(text)<500
    assert "training_blocks" not in text
    assert svg.exists() and output.exists()
    analysis.main([run["path"],"--output",str(tmp_path/"verbose.json"),"--verbose-json"])
    assert "skus" in json.loads(capsys.readouterr().out)


def test_cli_json_and_svg_paths_must_differ(tmp_path):
    run=make_run(tmp_path)
    path=tmp_path/"shared"
    with pytest.raises(ValueError,match="distinct"):
        analysis.main([run["path"],"--output",str(path),"--svg",str(path)])
    assert not path.exists()
