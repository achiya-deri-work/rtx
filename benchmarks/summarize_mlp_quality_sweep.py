"""Summarize isolated MLP quality trials without pooling numerical policies."""
import argparse
import csv
import json
from pathlib import Path


def summarize(root,extra_roots=()):
    groups={}; rows=[]; failures=[]
    reports=[]
    for path in sorted(root.glob("*/outcome.json")):
        outcome=json.loads(path.read_text())
        if outcome["status"] not in ("ok","recovered"):
            failures.append({"path":str(path),"status":outcome["status"]})
            continue
        reports.append((path.with_name("result.json"),outcome.get("wave","recovered")))
    for extra in extra_roots:
        reports.extend((path,"followup") for path in sorted(extra.glob("*.json")))
    for result,wave in reports:
        r=json.loads(result.read_text()); a=r["args"]
        row={"shape":a["shape"],"activation":a["activation"],"policy":a["policy"],
             "distribution":a["distribution"],"seed":a["seed"],
             "wave":wave,"train_us":r["train"]["median_us"],
             "forward_us":r["forward"]["median_us"],"result":str(result)}
        for name in ("y","dx","dgu","ddown"):
            row[f"{name}_error_percent"]=100*r["bf16_error"][name]["relative_l2"]
        rows.append(row)
        key=(a["shape"],a["activation"])
        if a["distribution"]=="init_002" and a["seed"]==17:
            group=groups.setdefault(key,{})
            if a["policy"] not in group or row["train_us"]<group[a["policy"]]["train_us"]:
                group[a["policy"]]=row
    contexts=[]
    for (shape,activation),policies in sorted(groups.items()):
        context={"shape":shape,"activation":activation,"best_search_observations":policies}
        hybrid=policies.get("bf16_gate_fwd")
        if hybrid:
            for baseline in ("bf16","fused"):
                if baseline in policies:
                    context[f"hybrid_speedup_vs_{baseline}"]=policies[baseline]["train_us"]/hybrid["train_us"]
        contexts.append(context)
    return {"successful_trials":len(rows),"failures":failures,"contexts":contexts},rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("--extra-root",type=Path,action="append",default=[])
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); summary,rows=summarize(a.root,a.extra_root)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(summary,indent=2))
    if rows:
        with a.output.with_suffix(".csv").open("w") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    print(json.dumps({"success":len(rows),"failures":len(summary["failures"]),"contexts":len(summary["contexts"])}))


if __name__=="__main__":
    main()
