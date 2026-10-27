"""Export an immutable byte-token prefix for matched cross-device training."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def prepare(source, destination, count=10_000_000):
    if type(count) is not int or count < 1:
        raise ValueError('count must be a positive integer')
    tokens=np.load(source,mmap_mode='r',allow_pickle=False)
    if tokens.ndim != 1 or not np.issubdtype(tokens.dtype,np.integer):
        raise ValueError('expected a one-dimensional integer token corpus')
    if len(tokens)<count:
        raise ValueError('source has fewer than the requested tokens')
    prefix=tokens[:count]
    if prefix.min()<0 or prefix.max()>256:
        raise ValueError('byte tokens must be in [0,256]')
    # Preserve dtype and exact byte stream used by Corpus.sha256 in training.
    raw_hash=hashlib.sha256(prefix.tobytes()).hexdigest()
    destination=Path(destination)
    with destination.open('xb') as stream:
        np.save(stream,prefix,allow_pickle=False)
    return dict(tokens=count,dtype=str(prefix.dtype),token_sha256=raw_hash,
                file_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                source=str(source),destination=str(destination))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tokens',type=int,default=10_000_000)
    args=p.parse_args()
    print(json.dumps(prepare(args.source,args.output,args.tokens),sort_keys=True))


if __name__=='__main__':
    main()
