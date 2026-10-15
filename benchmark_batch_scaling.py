#!/usr/bin/env python3
"""Run NexQus v0.3 batch-size scaling with matched workloads and validation.

Save this file in the SDK root and run:
    python benchmark_batch_scaling.py
Or run from anywhere:
    python benchmark_batch_scaling.py --sdk-dir ~/nexqus-sdk-v0.3.0

Uses the current Python environment. Each batch runs in a fresh subprocess.
Results include per-batch JSON, logs, summary JSON/CSV, and best observed sizes.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk-dir',type=Path,default=Path.cwd())
    parser.add_argument('--batches',nargs='+',type=int,default=[32,64,128,256])
    parser.add_argument('--qubits',type=int,default=10)
    parser.add_argument('--noise',type=float,default=.01)
    parser.add_argument('--trajectories',type=int,default=512)
    parser.add_argument('--repeat',type=int,default=5)
    parser.add_argument('--cpu-threads',type=int,default=1)
    parser.add_argument('--precision',choices=['single','double'],default='double')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    if any(x<1 for x in args.batches+[args.qubits,args.trajectories,args.repeat,args.cpu_threads]):
        parser.error('Batch sizes, qubits, trajectories, repeat and cpu-threads must be positive')
    if len(set(args.batches))!=len(args.batches): parser.error('Batch sizes must be unique')
    if max(args.batches)>args.trajectories:
        parser.error('Batch sizes must not exceed trajectories; otherwise effective sizes repeat')
    if not 0<=args.noise<=1: parser.error('noise must lie in [0,1]')
    root=args.sdk_dir.expanduser().resolve()
    runner=root/'examples'/'benchmark_v3.py'
    if not runner.is_file(): parser.error(f'Cannot find {runner}; pass --sdk-dir pointing to v0.3.0')
    directory=(args.output_dir.expanduser().resolve() if args.output_dir else
               root/('batch_scaling_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')))
    directory.mkdir(parents=True,exist_ok=False)
    configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    (directory/'configuration.json').write_text(json.dumps(configuration,indent=2)+'\n',encoding='utf-8')
    rows=[]; skipped={}; expected_backends=None
    for batch in args.batches:
        report_path=directory/f'batch_{batch}.json'
        command=[sys.executable,str(runner),'--compare','--method','trajectory',
                 '--qubits',str(args.qubits),'--noise',str(args.noise),
                 '--trajectories',str(args.trajectories),'--batch-size',str(batch),
                 '--cpu-threads',str(args.cpu_threads),'--random-engine','native',
                 '--precision',args.precision,'--seed',str(args.seed),'--repeat',str(args.repeat),
                 '--output',str(report_path)]
        print(f'Running batch {batch} ({args.trajectories} trajectories, {args.repeat} repeats)...',flush=True)
        completed=subprocess.run(command,cwd=root,capture_output=True,text=True)
        (directory/f'batch_{batch}.log').write_text(completed.stdout+'\n'+completed.stderr,encoding='utf-8')
        if completed.returncode:
            print(f'Batch {batch} failed. See {directory/f"batch_{batch}.log"}',file=sys.stderr)
            print(completed.stderr[-4000:],file=sys.stderr)
            return completed.returncode
        report=json.loads(report_path.read_text(encoding='utf-8'))
        if report['sdk_version']!='0.3.0':
            raise RuntimeError('Activate/install NexQus 0.3.0 in this Python environment')
        backends={(r['backend'],r['device']) for r in report['results']}
        if expected_backends is None: expected_backends=backends
        if backends!=expected_backends: raise RuntimeError('Available backends changed across runs')
        skipped[str(batch)]=report.get('skipped',[])
        for r in report['results']:
            if r.get('validation',{}).get('check')!='passed':
                raise AssertionError('Strict portable validation did not pass')
            row=dict(batch_size=batch,backend=r['backend'],device=r['device'],
                     median_seconds=r['median_run_wall_seconds'],
                     trajectories_per_second=args.trajectories/r['median_run_wall_seconds'],
                     estimated_working_mib=r['estimated_working_bytes']/1024**2,
                     cuda_peak_mib=(r['cuda_peak_allocated_bytes']/1024**2
                                    if 'cuda_peak_allocated_bytes' in r else None),
                     validation_max_error=r['validation']['max_probability_error'],
                     validation_tvd=r['validation']['total_variation_distance'])
            rows.append(row)
            print(f"  {row['backend']:5} {row['device']:7} {row['median_seconds']:.6f} s"
                  f"  {row['trajectories_per_second']:.1f} trajectories/s",flush=True)
        # Partial summary survives interruption of a later batch.
        (directory/'partial_results.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
    best=[]
    for backend,device in sorted(expected_backends):
        group=[r for r in rows if (r['backend'],r['device'])==(backend,device)]
        winner=min(group,key=lambda r:r['median_seconds'])
        baseline=next(r for r in group if r['batch_size']==args.batches[0])
        best.append(dict(winner,speedup_vs_first_batch=baseline['median_seconds']/winner['median_seconds']))
    summary=dict(configuration=configuration,results=rows,best_observed=best,skipped=skipped,
                 interpretation='Best observed for this workload/hardware; not a universal optimal batch size. '
                 'Timings use native RNG; strict numerical validation uses a separate portable RNG run. '
                 'Changing batch layout changes random streams. Repeat close comparisons to assess variability.')
    (directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    with (directory/'summary.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print('\nBest observed batch sizes:')
    for r in best:
        print(f"  {r['backend']} {r['device']}: batch {r['batch_size']}, "
              f"{r['median_seconds']:.6f} s, {r['speedup_vs_first_batch']:.2f}x versus first batch")
    if any(skipped.values()): print('Some backends were skipped; inspect summary.json for details.')
    print(f'\nSaved summary and raw results to: {directory}')
    return 0


if __name__=='__main__': sys.exit(main())
