"""Create an isolated first-hit cache/spec, then launch disposable preflight."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from rtx import AutotunePolicy, MXFP8MLPConfig
from rtx._mlp_autotune import deferred_spec


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--field',default='challenger')
    p.add_argument('--shape',default='11712,256,768,256')
    p.add_argument('--reserve-mib',type=int,default=12288)
    p.add_argument('--trials',type=int,default=8)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.reserve_mib<0 or args.trials<1:
        p.error('nonnegative reserve and positive trials required')
    config=MXFP8MLPConfig.from_dict(json.loads(args.config.read_text())[args.field])
    policy=AutotunePolicy(max_trials=args.trials,max_seconds=120,warmup=1,
                         samples=3,calls_per_sample=1,confirmation_repeats=0,
                         finalist_rounds=6)
    root=args.output.resolve()
    root.mkdir(parents=True,exist_ok=False)
    spec=deferred_spec(config,'mxfp8','balanced',policy,root/'cache')
    with (root/'spec.json').open('x') as stream:
        stream.write(spec)
    command=[sys.executable,'-m','rtx.autotune.mlp_preflight','--shape',args.shape,
             '--spec',str(root/'spec.json'),'--output',str(root/'preflight'),
             '--reserve-mib',str(args.reserve_mib)]
    with (root/'command.json').open('x') as stream:
        json.dump(command,stream,indent=2)
    subprocess.run(command,check=True)


if __name__=='__main__': main()
