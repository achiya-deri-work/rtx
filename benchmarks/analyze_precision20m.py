"""Observed-checkpoint analysis for train_precision20m JSONL runs.

Comparisons are within SKU only. Training wall excludes compile/evaluation;
end-to-end wall is reported separately. No interpolated or extrapolated loss.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
from html import escape

BASELINE="mxfp8"
MATCH_FIELDS=("seed","params","batch","sequence","steps","total_train_tokens","model_config","optimizer",
              "initial_weights_sha256","corpus_sha256","validation_offsets_sha256","attention","precision")


def positive_finite(value,label,*,allow_zero=False):
    if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    if value<0 or (value==0 and not allow_zero):
        raise ValueError(f"{label} must be {'nonnegative' if allow_zero else 'positive'}")
    return value


def read_run(path,*,allow_partial=False):
    path=Path(path)
    lines=path.read_text().splitlines()
    events=[]
    warnings=[]
    for index,line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row=json.loads(line)
        except json.JSONDecodeError:
            if allow_partial and index==len(lines)-1:
                warnings.append(f"ignored incomplete final JSONL line {index+1}")
                continue
            raise ValueError(f"{path}:{index+1}: invalid JSONL") from None
        if not isinstance(row,dict) or "event" not in row:
            raise ValueError(f"{path}:{index+1}: expected event object")
        events.append(row)
    environments=[row for row in events if row["event"]=="environment"]
    if len(environments)!=1:
        raise ValueError(f"{path}: exactly one environment event required")
    env=environments[0]
    if not env.get("sku") or not env.get("mode"):
        raise ValueError(f"{path}: environment requires sku and mode")
    validations=[row for row in events if row["event"]=="validation"]
    previous=None
    for row in validations:
        for key in ("step","tokens","training_wall_s","end_to_end_wall_s"):
            positive_finite(row.get(key),f"validation.{key}",allow_zero=True)
        positive_finite(row.get("validation_loss"),"validation.validation_loss",allow_zero=True)
        if row["end_to_end_wall_s"]<row["training_wall_s"]:
            raise ValueError("end-to-end wall cannot be smaller than training wall")
        current=tuple(row[k] for k in ("step","tokens","training_wall_s","end_to_end_wall_s"))
        if previous is not None and any(a<b for a,b in zip(current,previous)):
            raise ValueError(f"{path}: validation timeline must be monotone")
        previous=current
    keys=[(row["step"],row["tokens"]) for row in validations]
    if len(set(keys))!=len(keys):
        raise ValueError(f"{path}: duplicate validation step/token checkpoint")
    blocks=[row for row in events if row["event"]=="train"]
    rates=[]
    for row in blocks:
        tokens=positive_finite(row.get("block_tokens"),"train.block_tokens")
        elapsed=positive_finite(row.get("block_training_wall_s"),"train.block_training_wall_s")
        rates.append(tokens/elapsed)
    results=[row for row in events if row["event"]=="result"]
    errors=[row for row in events if row["event"]=="error"]
    latest=next((row for row in reversed(events) if "training_wall_s" in row),{})
    terminal=results[-1] if results else None
    if errors:
        status="error"
    elif terminal is not None and terminal.get("interrupted",False):
        status="interrupted"
    elif terminal is not None and terminal.get("finite_parameters") is False:
        status="nonfinite_parameters"
    elif (terminal is not None and terminal.get("finite_parameters") is True
          and isinstance(env.get("steps"),int) and env["steps"]>0
          and isinstance(env.get("total_train_tokens"),int) and env["total_train_tokens"]>0
          and terminal.get("step")==env.get("steps")
          and terminal.get("tokens")==env.get("total_train_tokens")):
        status="complete"
    else:
        status="incomplete"
    if status!="complete":
        if not allow_partial:
            raise ValueError(f"{path}: {status} run; use --allow-partial to analyze flagged observations")
        warnings.append(f"partial analysis permitted: run status={status}; comparisons cover observed checkpoints only")
    return dict(path=str(path),environment=env,validations=validations,events=events,warnings=warnings,
        validation_checkpoints=[checkpoint(row) for row in validations],training_blocks=blocks,
        throughput=dict(block_samples=len(rates),median_tokens_per_s=statistics.median(rates) if rates else None,
            min_tokens_per_s=min(rates) if rates else None,max_tokens_per_s=max(rates) if rates else None,
            samples_tokens_per_s=rates,scope="tokens / synchronized measured training block wall; not compile/warmup/evaluation"),
        status=status,partial_analysis=status!="complete",terminal_result=terminal,
        errors=errors,final_training_wall_s=latest.get("training_wall_s"),
        final_end_to_end_wall_s=latest.get("end_to_end_wall_s"),
        compile_warmup_wall_s=sum(row.get("compile_warmup_wall_s",0.) for row in events if row["event"]=="compile_warmup"),
        validation_wall_s=sum(row.get("validation_wall_s",0.) for row in validations),
        checkpoint_wall_s=sum(row.get("checkpoint_wall_s",0.) for row in events if row["event"]=="checkpoint"))


def checkpoint(row):
    return {key:row[key] for key in ("step","tokens","validation_loss","training_wall_s","end_to_end_wall_s")}


def compare_equal_work(runs,baseline,*,baseline_mode=BASELINE):
    reference={(row["step"],row["tokens"]):row for row in baseline["validations"]}
    output={}
    for mode,run in runs.items():
        matches=[]
        for row in run["validations"]:
            ref=reference.get((row["step"],row["tokens"]))
            if ref is not None:
                matches.append(dict(**checkpoint(row),baseline_validation_loss=ref["validation_loss"],
                    **{("validation_loss_delta_vs_mxfp8" if baseline_mode==BASELINE else "validation_loss_delta_vs_baseline"):row["validation_loss"]-ref["validation_loss"]},
                    baseline_training_wall_s=ref["training_wall_s"],
                    baseline_end_to_end_wall_s=ref["end_to_end_wall_s"]))
        output[mode]=dict(actual_matching_checkpoints=matches,last_actual_match=matches[-1] if matches else None)
    return output


def compare_equal_wall(runs,*,clock,baseline_mode=BASELINE):
    if any(not run["validations"] for run in runs.values()):
        return dict(status="insufficient_validation_checkpoints",clock=clock)
    budget=min(run["validations"][-1][clock] for run in runs.values())
    selected={mode:next((row for row in reversed(run["validations"]) if row[clock]<=budget),None)
              for mode,run in runs.items()}
    baseline=selected.get(baseline_mode)
    rows={}
    for mode,row in selected.items():
        rows[mode]=(None if row is None else dict(**checkpoint(row),checkpoint_age_s=budget-row[clock],
            **{("validation_loss_delta_vs_mxfp8" if baseline_mode==BASELINE else "validation_loss_delta_vs_baseline"):None if baseline is None else row["validation_loss"]-baseline["validation_loss"]}))
    return dict(status="observed_checkpoints_only",clock=clock,shared_budget_s=budget,results=rows,
        caveat="last actual validation checkpoint at or before shared budget; unequal checkpoint ages retained; no interpolation/extrapolation")


def observed_thresholds(runs,thresholds):
    output=[]
    for threshold in thresholds:
        positive_finite(threshold,"loss threshold",allow_zero=True)
        modes={}
        for mode,run in runs.items():
            crossing=next((row for row in run["validations"] if row["validation_loss"]<=threshold),None)
            modes[mode]=dict(status="observed_at_or_below" if crossing else "not_observed",
                first_observed_checkpoint=checkpoint(crossing) if crossing else None,
                last_observed_checkpoint=checkpoint(run["validations"][-1]) if run["validations"] else None)
        output.append(dict(validation_loss_threshold=threshold,results=modes,
            caveat="first observed checkpoint at or below threshold, not interpolated crossing time; unreached thresholds are not extrapolated"))
    return output


def analyze(runs,*,thresholds=(),allow_partial=False,baseline_mode=BASELINE):
    if baseline_mode not in ("mxfp8","nvfp4_mxfp8"):
        raise ValueError("unsupported baseline mode")
    grouped={}
    for run in runs:
        if run["status"]!="complete" and not allow_partial:
            raise ValueError("incomplete/error/interrupted run requires allow_partial=True")
        env=run["environment"]
        sku=env["sku"]
        mode=env["mode"]
        modes=grouped.setdefault(sku,{})
        if mode in modes:
            raise ValueError(f"duplicate run for SKU {sku}, mode {mode}; analyze seeds separately")
        modes[mode]=run
    output={}
    for sku,modes in grouped.items():
        if baseline_mode not in modes:
            raise ValueError(f"SKU {sku} missing {baseline_mode} baseline")
        baseline=modes[baseline_mode]
        provenance={}
        for field in MATCH_FIELDS:
            supplied={mode:run["environment"].get(field) for mode,run in modes.items()}
            if any(value is None for value in supplied.values()):
                raise ValueError(f"SKU {sku}: missing matched provenance {field}")
            if any(value!=supplied[baseline_mode] for value in supplied.values()):
                raise ValueError(f"SKU {sku}: mismatched {field}")
            provenance[field]=supplied[baseline_mode]
        if baseline_mode=="nvfp4_mxfp8":
            reference=baseline["environment"].get("projection_configs")
            if not reference or any(mode not in ("nvfp4_mxfp8","nvfp4_full") for mode in modes):
                raise ValueError("matched NVFP4 forward provenance required")
            for run in modes.values():
                projections=run["environment"].get("projection_configs",{})
                if set(projections)!=set(reference) or any(
                    "forward" not in projections[key] or "forward" not in reference[key]
                    or projections[key]["forward"]!=reference[key]["forward"] for key in reference):
                    raise ValueError("mismatched forward projection configurations")
        selected_thresholds=list(thresholds)
        threshold_origin="explicit CLI thresholds"
        if not selected_thresholds and baseline["validations"]:
            selected_thresholds=[baseline["validations"][-1]["validation_loss"]]
            threshold_origin=f"last actually observed {baseline_mode} baseline validation loss"
        summaries={mode:{key:value for key,value in run.items() if key not in ("events","validations")}
                   for mode,run in modes.items()}
        output[sku]=dict(matched_provenance=provenance,runs=summaries,
            analysis_status="partial_observations" if any(run["status"]!="complete" for run in modes.values()) else "completed_runs",
            equal_step_tokens=compare_equal_work(modes,baseline,baseline_mode=baseline_mode),
            equal_training_wall=compare_equal_wall(modes,clock="training_wall_s",baseline_mode=baseline_mode),
            equal_end_to_end_wall=compare_equal_wall(modes,clock="end_to_end_wall_s",baseline_mode=baseline_mode),
            threshold_origin=threshold_origin,time_to_loss_observed=observed_thresholds(modes,selected_thresholds))
    return dict(skus=output,baseline_mode=baseline_mode,comparison_scope="within SKU only; independent hardware timing and trajectories are not pooled",
        loss_semantics=f"actual fixed-validation checkpoints only; lower validation loss is better; delta=candidate minus {baseline_mode}",
        clock_semantics="training_wall_s excludes compile/evaluation/checkpoints; end_to_end_wall_s includes them and is separately reported",
        throughput_semantics="median raw block token/time rates, never single warmup timing")


def plot_bounds(values):
    if not values:
        raise ValueError("plot bounds require observed values")
    for value in values:
        positive_finite(value,"plot value",allow_zero=True)
    lo,hi=min(values),max(values)
    padding=max((hi-lo)*.05,abs(hi)*.005,1e-6)
    return max(0.,lo-padding),hi+padding


def validation_svg(result):
    """Actual checkpoint markers; connecting segments are visual guides only."""
    width,row_height,top=1120,330,110
    height=top+row_height*len(result["skus"])+55
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<title>Observed validation loss by step and training wall time</title>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:system-ui,sans-serif;fill:#17202a} .tick{font-size:11px} .heading{font-size:15px;font-weight:600}</style>',
        '<text x="24" y="26" font-size="19" font-weight="600">Actual validation checkpoints — post-initialization zoom</text>',
        '<text x="24" y="48" font-size="12">Step 0 excluded. Markers are measured checkpoints; straight segments are visual guides, not interpolated measurements.</text>',
        '<text x="24" y="67" font-size="12">Training wall excludes compilation, evaluation and checkpoints. Full raw observations, including step 0, remain in JSON.</text>']
    colors={"mxfp8":"#1764a0","nvfp4_mxfp8":"#c7680c","nvfp4_full":"#25874b"}
    fallback=("#9046aa","#ae3558","#58636b")
    modes=sorted({mode for sku in result["skus"].values() for mode in sku["runs"]})
    for index,mode in enumerate(modes):
        color=colors.get(mode,fallback[index%len(fallback)])
        x=24+index*255
        parts.append(f'<circle cx="{x}" cy="89" r="4" fill="{color}"/><text x="{x+10}" y="93" font-size="12">{escape(mode)}</text>')
    for row_index,(sku,data) in enumerate(result["skus"].items()):
        series={mode:[point for point in run["validation_checkpoints"] if point["step"]>0]
                for mode,run in data["runs"].items()}
        all_points=[point for points in series.values() for point in points]
        y0=top+row_index*row_height
        for column,(clock,label) in enumerate((("step","Training step"),("training_wall_s","Training wall (s)"))):
            panel_x=column*560
            left,right=panel_x+70,panel_x+535
            chart_top,chart_bottom=y0+35,y0+275
            title=f"{sku} — validation loss vs {label.lower()}"
            parts.append(f'<text x="{panel_x+20}" y="{y0+18}" class="heading">{escape(title)}</text>')
            if not all_points:
                parts.append(f'<text x="{left}" y="{chart_top+40}" font-size="13">No post-step-0 validation checkpoints observed.</text>')
                continue
            xmin,xmax=plot_bounds([point[clock] for point in all_points])
            ymin,ymax=plot_bounds([point["validation_loss"] for point in all_points])
            def xpos(value):
                return left+(value-xmin)/(xmax-xmin)*(right-left)
            def ypos(value):
                return chart_bottom-(value-ymin)/(ymax-ymin)*(chart_bottom-chart_top)
            parts.append(f'<g data-sku="{escape(str(sku),quote=True)}" data-clock="{clock}" data-x-min="{xmin}" data-x-max="{xmax}" data-y-min="{ymin}" data-y-max="{ymax}">')
            for tick in range(5):
                frac=tick/4
                xv=xmin+(xmax-xmin)*frac
                yv=ymin+(ymax-ymin)*frac
                px,py=xpos(xv),ypos(yv)
                parts.append(f'<path d="M {left} {py:.3f} H {right}" stroke="#e6e9ed"/><text x="{left-8}" y="{py+4:.3f}" text-anchor="end" class="tick">{yv:.4g}</text>')
                parts.append(f'<text x="{px:.3f}" y="{chart_bottom+19}" text-anchor="middle" class="tick">{xv:.4g}</text>')
            parts.append(f'<path d="M {left} {chart_top} V {chart_bottom} H {right}" fill="none" stroke="#66717c"/>')
            parts.append(f'<text x="{(left+right)/2}" y="{chart_bottom+42}" text-anchor="middle" font-size="12">{label}</text>')
            parts.append(f'<text transform="translate({panel_x+16},{(chart_top+chart_bottom)/2}) rotate(-90)" text-anchor="middle" font-size="12">Validation cross entropy / token</text>')
            for index,(mode,points) in enumerate(series.items()):
                color=colors.get(mode,fallback[index%len(fallback)])
                coords=" ".join(f'{xpos(point[clock]):.3f},{ypos(point["validation_loss"]):.3f}' for point in points)
                if points:
                    parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="1.7"/>')
                for point in points:
                    title=f'{mode}: step {point["step"]}, loss {point["validation_loss"]:.6g}, training wall {point["training_wall_s"]:.6g}s'
                    parts.append(f'<circle cx="{xpos(point[clock]):.3f}" cy="{ypos(point["validation_loss"]):.3f}" r="3" fill="{color}" data-step="{point["step"]}" data-validation-loss="{point["validation_loss"]}"><title>{escape(title)}</title></circle>')
            parts.append('</g>')
    parts.append(f'<text x="24" y="{height-18}" font-size="12">Within-device comparisons only. Unequal checkpoint spacing is retained; no endpoint extrapolation.</text></svg>')
    return "\n".join(parts)


def write_svg(result,path):
    path=Path(path)
    content=validation_svg(result)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x") as handle:
        handle.write(content)


def console_summary(result,output,svg=None):
    lines=[f"Analysis: {output}"+(f"; plot: {svg}" if svg else "")]
    for sku,data in result["skus"].items():
        values=[]
        for mode,run in data["runs"].items():
            points=run["validation_checkpoints"]
            values.append(f'{mode}: val={points[-1]["validation_loss"]:.5f}@step{points[-1]["step"]}' if points else f"{mode}: no validation")
        lines.append(f'{sku} ({data["analysis_status"]}): '+"; ".join(values))
    return "\n".join(lines)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs",nargs="+",type=Path)
    p.add_argument("--loss-threshold",action="append",type=float,default=[])
    p.add_argument("--baseline-mode",choices=("mxfp8","nvfp4_mxfp8"),default=BASELINE,
                   help="nvfp4_mxfp8 requires matching NVFP4 forward projection configs")
    p.add_argument("--allow-partial",action="store_true",help="permit explicitly flagged incomplete/error/interrupted runs and ignore only an incomplete final JSONL line")
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--svg",type=Path,help="optional actual-checkpoint SVG plot; step 0 excluded from zoom")
    p.add_argument("--verbose-json",action="store_true",help="print full analysis JSON instead of concise summary")
    args=p.parse_args(argv)
    result=analyze([read_run(path,allow_partial=args.allow_partial) for path in args.runs],
                   thresholds=args.loss_threshold,allow_partial=args.allow_partial,baseline_mode=args.baseline_mode)
    targets=[args.output]+([args.svg] if args.svg else [])
    if len({path.resolve() for path in targets})!=len(targets):
        raise ValueError("JSON and SVG output paths must be distinct")
    for target in targets:
        if target.exists():
            raise FileExistsError(target)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
    if args.svg:
        write_svg(result,args.svg)
    print(json.dumps(result,allow_nan=False) if args.verbose_json else console_summary(result,args.output,args.svg))


if __name__=="__main__":
    main()
