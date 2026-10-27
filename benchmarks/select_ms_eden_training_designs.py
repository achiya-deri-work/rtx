"""Build a compatible, device-local training map from validated search trials.

Selection is not confirmation or runtime promotion. Every selected design must
still pass an independent full-backward confirmation before model timing.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from benchmarks.compare_ms_eden_pipeline import SHAPES
from rtx.autotune.ms_eden_design import MSEdenDesign, make_ms_eden_design_adapter

MODEL_SHAPES = ('decoder1536', 'decoder512', 'decoder2048', 'decoder1024')


def implementation_key(design):
    return (design.transpose_loader, design.finalizer, design.tma_tile_rows, design.row_transform)


def choose_common(candidates, required=MODEL_SHAPES):
    """Minimize one model layer's sum of four projection backward latencies."""
    families = {}
    for record in candidates:
        label = record['shape']
        if label not in required:
            continue
        family = implementation_key(MSEdenDesign(**record['design']))
        best = families.setdefault(family, {})
        previous = best.get(label)
        if previous is None or (record['median_ms'], record['path']) < (previous['median_ms'], previous['path']):
            best[label] = record
    complete = [(sum(rows[label]['median_ms'] for label in required), family, rows)
                for family, rows in families.items() if all(label in rows for label in required)]
    if not complete:
        raise ValueError('no timing-eligible shared implementation covers all model projections')
    cost, family, rows = min(complete, key=lambda value: (value[0], value[1]))
    return dict(estimated_layer_backward_ms=cost, implementation=family,
                selected={label: rows[label] for label in required},
                requires_independent_confirmation=True, automatic_promotion=False)


def collect(root):
    candidates, identities, excluded = [], set(), []
    for label in MODEL_SHAPES:
        shape = SHAPES[label]
        for path in sorted((root / f'search-{label}').glob('*/result.json')):
            result = json.loads(path.read_text())
            design = MSEdenDesign(**result['design'])
            adapter = make_ms_eden_design_adapter(shape, lambda _: result,
                                                  initial=design, allow_row_major=True)
            outcome = adapter.evaluate(design)
            stationary = outcome.metadata.get('sampling', {}).get('collection', {}).get('stationary')
            if not outcome.timing_eligible or stationary is not True:
                excluded.append(dict(path=str(path), status=outcome.status, error=outcome.error))
                continue
            # UUID-bearing device description and exact benchmark/kernel hashes:
            # never silently assemble a map from different machines/revisions.
            identities.add((result['device'], json.dumps(result['source_sha256'], sort_keys=True)))
            candidates.append(dict(shape=label, design=asdict(design), path=str(path.resolve()),
                                   median_ms=outcome.median_ms))
    if len(identities) > 1:
        raise ValueError('mixed device or source identities in training selection')
    return candidates, excluded, sorted(identities)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records, excluded, identities = collect(args.campaign)
    selection = choose_common(records)
    selection.update(excluded=excluded, identities=identities)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'selection.json').write_text(json.dumps(selection, indent=2))
    mapping = {','.join(map(str, SHAPES[label])): row['design']
               for label, row in selection['selected'].items()}
    (args.output / 'designs.json').write_text(json.dumps(mapping, indent=2))
    for label, row in selection['selected'].items():
        (args.output / f'{label}.json').write_text(json.dumps({'candidate': row['design']}, indent=2))
    print(json.dumps(dict(output=str(args.output), implementation=selection['implementation'],
                         estimated_layer_backward_ms=selection['estimated_layer_backward_ms'])))


if __name__ == '__main__':
    main()
