"""Matched CPU/GPU benchmarks and reference probability checks.

python examples/benchmark_v2.py --compare --qubits 12 --output benchmark.json
"""
import argparse
import datetime
import json
import os
import platform
from pathlib import Path
import statistics
import time
import numpy as np
from nexqus import __version__, Circuit, Simulator, Noise, NoiseModel, algorithms


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qubits',type=int,default=12)
    p.add_argument('--circuit',choices=['ghz','layered','qft'],default='layered')
    p.add_argument('--depth',type=int,default=4)
    p.add_argument('--backend',choices=['numpy','torch'],default='numpy')
    p.add_argument('--device',default='cpu')
    p.add_argument('--compare',action='store_true',help='Run NumPy CPU, Torch CPU and CUDA when available')
    p.add_argument('--method',choices=['auto','statevector','density_matrix','trajectory'],default='auto')
    p.add_argument('--noise',type=float,default=0)
    p.add_argument('--precision',choices=['single','double'],default='double')
    p.add_argument('--repeat',type=int,default=5)
    p.add_argument('--trajectories',type=int,default=256)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--max-memory-mb',type=float,default=1024)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--uncompiled',action='store_true')
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.repeat<1 or args.depth<1 or args.qubits<1: p.error('repeat, depth and qubits must be positive')
    rng=np.random.default_rng(123)
    if args.circuit=='ghz': q=algorithms.ghz(args.qubits)
    elif args.circuit=='qft': q=algorithms.qft(args.qubits)
    else:
        q=Circuit(args.qubits)
        for _ in range(args.depth):
            for j in range(args.qubits): q.ry(float(rng.uniform(-np.pi,np.pi)),j).rz(.31,j)
            for j in range(args.qubits-1): q.cx(j,j+1)
    noise=NoiseModel(Noise.depolarizing(args.noise)) if args.noise else None
    kw=dict(shots=0,seed=args.seed,noise=noise,trajectories=args.trajectories,
            batch_size=args.batch_size,retain_state=False)
    configs=[(args.backend,args.device)]
    skipped=[]
    torch=None
    try:
        import torch
    except ImportError:
        pass
    if args.compare:
        configs=[('numpy','cpu')]
        if torch is None:
            skipped.append('Torch CPU and CUDA: PyTorch not installed')
        else:
            configs.append(('torch','cpu'))
            if torch.cuda.is_available(): configs.append(('torch','cuda:0'))
            else: skipped.append('CUDA: unavailable')
    reference=Simulator(method=args.method,precision=args.precision,max_memory_mb=args.max_memory_mb).run(q,**kw)
    records=[]
    for backend,device in configs:
        sim=Simulator(backend=backend,device=device,method=args.method,precision=args.precision,
                      max_memory_mb=args.max_memory_mb)
        workload=q if args.uncompiled else sim.compile(q,noise)
        options=dict(kw)
        if not args.uncompiled: options.pop('noise')
        sim.run(workload,**options) # Warmup before every measured series.
        if backend=='torch' and device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(device)
        rows=[]; wall=[]
        for _ in range(args.repeat):
            start=time.perf_counter()
            result=sim.run(workload,**options)
            wall.append(time.perf_counter()-start)
            rows.append(result.metadata)
        difference=float(np.abs(reference.probabilities-result.probabilities).max())
        # Same seed and batch layout should select matching trajectories. A
        # tolerance leaves room for backend arithmetic and rare branch-boundary
        # differences in stochastic mode; it is not proof of statevector parity.
        stochastic=result.metadata['method']=='trajectory'
        tolerance=max(2e-5,5/np.sqrt(args.trajectories)) if stochastic else (2e-5 if args.precision=='single' else 1e-9)
        if difference>tolerance:
            raise AssertionError(f'{backend}/{device}: reference probability mismatch {difference}')
        record=dict(result.metadata)
        record.update(median_simulation_seconds=statistics.median(r['simulation_seconds'] for r in rows),
                      median_run_wall_seconds=statistics.median(wall),
                      median_process_cpu_seconds=statistics.median(r['process_cpu_seconds'] for r in rows),
                      repeat=args.repeat,max_probability_error_vs_numpy=difference,
                      comparison_tolerance=tolerance,reference_probability_check='passed',cache=sim.cache_info())
        if backend=='torch' and device.startswith('cuda'):
            record['cuda_device_name']=torch.cuda.get_device_name(device)
            record['cuda_peak_allocated_bytes']=torch.cuda.max_memory_allocated(device)
        records.append(record)
    reference_time=records[0]['median_run_wall_seconds']
    for record in records:
        record['run_wall_speedup_vs_first']=reference_time/record['median_run_wall_seconds']
    report=dict(sdk_version=__version__,utc_time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                environment=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
                                 logical_cpus=os.cpu_count(),torch=None if torch is None else torch.__version__,
                                 cuda_runtime=None if torch is None else torch.version.cuda,
                                 thread_environment={k:os.environ.get(k) for k in
                                                     ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}),
                settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                results=records,skipped=skipped)
    payload=json.dumps(report,indent=2)
    if args.output: args.output.write_text(payload+'\n',encoding='utf-8')
    print(payload)


if __name__=='__main__': main()
