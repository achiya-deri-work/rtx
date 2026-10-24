"""Matched-candidate cross-context ranking; no transferred timing cache hits."""
import argparse
import itertools
import json
import math
from pathlib import Path
import statistics


def contract_features(features):
    return {k:v for k,v in features.items() if k.startswith("context.tags.")
            or k.startswith("context.regime=") or k.startswith("context.environment.")
            or k in ("context.kernel_revision",)
            or k.startswith("context.family=")
            or k.startswith("context.workload.")
            or any(k.startswith("context.device."+field+"=") for field in
                   ("torch_version","torch_cuda_version","cutlass_dsl_version"))}


def compare(rows, tie_fraction=.02):
    """Compare only matched shapes/contracts and exact active configuration IDs.

    Repeated measurements are reduced by median within each context/config.
    Near-ties use a fixed 2% threshold, not a statistical confidence claim.
    """
    if not 0 <= tie_fraction < 1:
        raise ValueError("invalid tie fraction")
    groups = {}
    for row in rows:
        if row.get("outcome",{}).get("status") != "ok":
            continue
        if row["outcome"].get("metadata",{}).get("sampling",{}).get("collection",{}).get("stationary") is False:
            continue
        value = row["outcome"]["median_ms"]
        if not math.isfinite(value) or value <= 0:
            raise ValueError("invalid successful timing")
        group = groups.setdefault(row["context_id"], {"features":row["features"],"configs":{}})
        if group["features"] != row["features"]:
            # Config and derived features legitimately differ within a context.
            old = {k:v for k,v in group["features"].items() if k.startswith("context.")}
            new = {k:v for k,v in row["features"].items() if k.startswith("context.")}
            if old != new:
                raise ValueError("inconsistent context identity")
        entry = group["configs"].setdefault(row["config_id"], {"times":[],"config":row["config"]})
        entry["times"].append(value)
    results = []
    for source_id, target_id in itertools.permutations(groups,2):
        source, target = groups[source_id],groups[target_id]
        if contract_features(source["features"]) != contract_features(target["features"]):
            continue
        shared = sorted(source["configs"].keys() & target["configs"].keys())
        for scope in ("all", "staged"):
            ids = shared
            if scope == "staged":
                ids = [cid for cid in ids if all(
                    source["configs"][cid]["config"]["config"][leg]["backend"] == "decomposed"
                    for leg in ("dx","dw"))]
            if not ids:
                continue
            a = {cid:statistics.median(source["configs"][cid]["times"]) for cid in ids}
            b = {cid:statistics.median(target["configs"][cid]["times"]) for cid in ids}
            order = sorted(ids,key=lambda cid:(a[cid],cid))
            best = min(b.values())
            agree, disagree, tied = 0,0,0
            for i,j in itertools.combinations(ids,2):
                if abs(a[i]-a[j]) <= tie_fraction*min(a[i],a[j]) or abs(b[i]-b[j]) <= tie_fraction*min(b[i],b[j]):
                    tied += 1
                elif (a[i]-a[j])*(b[i]-b[j]) > 0:
                    agree += 1
                else:
                    disagree += 1
            results.append({"source_context":source_id,"target_context":target_id,
                "scope":scope,"matched_configs":len(ids),"target_matched_best_ms":best,
                "source_first_config":order[0],
                "transfer_regret":{str(k):min(b[cid] for cid in order[:k])/best-1 for k in (1,2,4,8)},
                "concordant_pairs":agree,"discordant_pairs":disagree,"near_tied_pairs":tied})
    return {"schema_version":1,"deployment_enabled":False,"tie_fraction":tie_fraction,
        "contexts":{cid:{k:v for k,v in group["features"].items() if k.startswith("context.")}
                    for cid,group in groups.items()},"comparisons":results,
        "caveat":"Retrospective matched-candidate ranking, not first-hit replay or confidence intervals. No source latency is reused as target evidence."}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations",nargs="+")
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    rows=[json.loads(line) for path in args.observations for line in Path(path).read_text().splitlines() if line]
    result=compare(rows)
    result["sources"]=args.observations
    with Path(args.output).open("x") as handle:
        json.dump(result,handle,indent=2)
    print(json.dumps(result["comparisons"],indent=2))


if __name__ == "__main__":
    main()
