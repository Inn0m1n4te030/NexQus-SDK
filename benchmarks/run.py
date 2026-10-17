"""Independent CPU correctness/performance comparisons. No SDK is a winner by design.

Install: python -m pip install -e '.[benchmark-compare]'
Run: python benchmarks/run.py --suite standard --output benchmarks/results/local
Each engine/case executes in a fresh subprocess, sequentially, on this host.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import numpy as np
from cases import suite
from reference import accuracy, dense_reference

ROOT = Path(__file__).resolve().parents[1]


def current_rss():
    # /proc/self also works in containers where the exposed /proc PID numbers
    # differ from os.getpid(). Other platforms use psutil's native API.
    if sys.platform.startswith('linux'):
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'): return int(line.split()[1])*1024
    import psutil
    return psutil.Process().memory_info().rss


class PeakRSS:
    """Sample native + Python resident memory; not tracemalloc allocation counts."""
    def __init__(self):
        self.baseline = current_rss()
        self.peak = self.baseline
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.sample,daemon=True)

    def sample(self):
        while not self.stop_event.wait(.001): self.peak = max(self.peak,current_rss())

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.peak = max(self.peak,current_rss())
        self.stop_event.set(); self.thread.join()


def process_peak_rss():
    # OS high-water RSS includes imports/preparation. Linux/macOS units differ.
    if sys.platform.startswith('linux'):
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmHWM:'): return int(line.split()[1])*1024
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value * (1 if sys.platform == 'darwin' else 1024))
    except ImportError:
        import psutil
        return getattr(psutil.Process().memory_info(),'peak_wset',None)


def worker(request):
    from adapters import prepare, Unsupported
    from threadpoolctl import threadpool_limits, threadpool_info
    try:
        started = time.perf_counter()
        run, settings = prepare(request['engine'],request['case'],request['precision'],request['threads'])
        preparation_seconds = time.perf_counter()-started
        with threadpool_limits(limits=request['threads']):
            pools = threadpool_info()
            samples = []
            with PeakRSS() as memory:
                for _ in range(request['warmup']):
                    state, probabilities = run()
                    del state, probabilities
                for _ in range(request['repeats']):
                    # Previous results are released before the next timed run.
                    started = time.perf_counter()
                    state, probabilities = run()
                    samples.append(time.perf_counter()-started)
                    # Force both full outputs to be materialized on this CPU.
                    if not np.isfinite(state).all() or not np.isfinite(probabilities).all():
                        raise ValueError('Nonfinite output')
                    del state, probabilities
                state, probabilities = run()  # untimed accuracy output
            np.savez(request['state_path'],state=state,probabilities=probabilities)
        return dict(status='ok',engine=request['engine'],case=request['case']['name'],
                    precision=request['precision'],settings=settings,threadpools=pools,
                    preparation_seconds=preparation_seconds,seconds=samples,
                    median_seconds=statistics.median(samples),min_seconds=min(samples),max_seconds=max(samples),
                    stdev_seconds=statistics.stdev(samples) if len(samples)>1 else 0,
                    output_dtype=str(state.dtype),state_bytes=state.nbytes,probabilities_bytes=probabilities.nbytes,
                    baseline_rss_bytes=memory.baseline,execution_peak_rss_bytes=memory.peak,
                    sampled_incremental_rss_bytes=max(0,memory.peak-memory.baseline),
                    process_peak_rss_bytes=process_peak_rss())
    except Unsupported as exc:
        return dict(status='unsupported',engine=request['engine'],case=request['case']['name'],reason=str(exc))
    except Exception:
        return dict(status='error',engine=request['engine'],case=request['case']['name'],error=traceback.format_exc())


def provenance():
    import psutil
    import nexqus
    versions = {}
    for package in ('nexqus-sdk','numpy','qiskit','qiskit-aer','pennylane','psutil','threadpoolctl'):
        try: versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package] = None
    def git(*args):
        result = subprocess.run(['git',*args],cwd=ROOT,capture_output=True,text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    cpu = platform.processor()
    if Path('/proc/cpuinfo').exists():
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                cpu = line.split(':',1)[1].strip(); break
    digest = hashlib.sha256()
    for path in sorted(Path(nexqus.__file__).parent.glob('*.py')):
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    benchmark_digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        benchmark_digest.update(path.name.encode()); benchmark_digest.update(path.read_bytes())
    try:
        affinity = sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else psutil.Process().cpu_affinity()
    except (AttributeError, OSError, psutil.Error): affinity = None
    return dict(utc_time=datetime.now(timezone.utc).isoformat(),python=platform.python_version(),
                platform=platform.platform(),cpu=cpu,logical_cpus=psutil.cpu_count(),
                physical_cpus=psutil.cpu_count(logical=False),
                cpu_affinity=affinity,
                total_memory_bytes=psutil.virtual_memory().total,versions=versions,
                git_commit=git('rev-parse','HEAD'),git_dirty=bool(git('status','--porcelain')),
                runtime_source_sha256=digest.hexdigest(),benchmark_source_sha256=benchmark_digest.hexdigest(),
                runtime_path=str(Path(nexqus.__file__).resolve()),device='CPU')


def write_reports(output, report):
    output.mkdir(parents=True,exist_ok=True)
    (output/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    columns = ['case','engine','status','precision','median_seconds','min_seconds','max_seconds',
               'stdev_seconds','preparation_seconds','execution_peak_rss_bytes','baseline_rss_bytes',
               'sampled_incremental_rss_bytes','process_peak_rss_bytes','max_state_error','probability_tvd',
               'accuracy_passed','reference']
    with (output/'results.csv').open('w',newline='',encoding='utf-8') as handle:
        writer = csv.DictWriter(handle,columns,extrasaction='ignore'); writer.writeheader()
        for row in report['results']:
            check = row.get('accuracy',{})
            writer.writerow(dict(row, max_state_error=check.get('max_state_error'),
                                 probability_tvd=check.get('probability_tvd'),accuracy_passed=check.get('passed')))
    p = report['environment']
    lines = ['# Independent simulator comparison', '',
             f"Run: {p['utc_time']}. CPU: {p['cpu']}. Python {p['python']}.", '',
             f"Suite `{report['configuration']['suite']}`; precision `{report['configuration']['precision']}`; "
             f"{report['configuration']['threads']} CPU thread(s); {report['configuration']['warmup']} warmups and "
             f"{report['configuration']['repeats']} measured repetitions per engine/case.", '',
             'Full specifications, raw timings, versions, settings, hardware and errors: [JSON](results.json). '
             'Tabular data: [CSV](results.csv).', '',
             'Timing covers prepared execution and full state/probability output. Imports, circuit construction, '
             'and compilation are excluded and preparation is reported separately. Gate fusion and idle-wire '
             'truncation are disabled. Every engine uses the same gates, noise, precision, zero shots and CPU. '
             'PennyLane uses default.qubit/default.mixed; this is not a benchmark of lightning.qubit.', '',
             'RSS is process resident memory, including native allocations. Execution peak is sampled every 1 ms '
             'and includes warmups plus the untimed validation run; short peaks may be missed. OS process peak '
             'also includes imports and preparation. Baselines differ between SDKs; incremental RSS can be zero '
             'when an allocator reuses memory. Sampling is active during timing, so its overhead is included. '
             'No GPU memory or hardware execution is measured.', '',
             '| Circuit | Engine | Median ms | Execution peak MiB | Process peak MiB | Max state error | Accuracy |',
             '|---|---|---:|---:|---:|---:|---|']
    for row in report['results']:
        if row['status'] == 'ok':
            check = row['accuracy']; peak = row.get('process_peak_rss_bytes')
            lines.append(f"| {row['case']} | {row['engine']} | {row['median_seconds']*1000:.3f} | "
                         f"{row['execution_peak_rss_bytes']/2**20:.1f} | "
                         f"{format(peak/2**20,'.1f') if peak is not None else 'n/a'} | "
                         f"{format(check['max_state_error'],'.2e') if 'max_state_error' in check else 'n/a'} | "
                         f"{'pass' if check['passed'] else 'FAIL'} |")
        else: lines.append(f"| {row['case']} | {row['engine']} | — | — | — | — | {row['status']} |")
    lines += ['', 'Small circuits (up to 5 qubits) use an independent full-matrix NumPy oracle. Larger circuits '
              'use Qiskit Aer as the cross-simulator reference when available; the JSON labels every reference. '
              'Statevector errors are aligned for global phase. Density errors compare full matrix entries. '
              'Probability TVD, normalization, Hermiticity, and small-density eigenvalues are also checked.', '',
              'These measurements describe this host and these workloads. Keep slower NexQus results; rerun on '
              'your own hardware before making performance claims. Prepared API overhead is included, so tiny '
              'circuits are often dominated by framework overhead.', '']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',choices=['smoke','correctness','standard'],default='smoke')
    parser.add_argument('--engines',nargs='+',choices=['nexqus','qiskit','pennylane'],default=['nexqus','qiskit','pennylane'])
    parser.add_argument('--precision',choices=['single','double'],default='double')
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--repeats',type=int,default=7)
    parser.add_argument('--timeout',type=float,default=120)
    parser.add_argument('--output',type=Path,default=ROOT/'.release-artifacts'/'independent-benchmarks')
    parser.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(json.loads(args.worker.read_text())),allow_nan=False)); return 0
    if args.threads < 1 or args.repeats < 1 or args.warmup < 0 or args.timeout <= 0:
        parser.error('threads/repeats/timeout must be positive; warmup must be nonnegative')
    if len(set(args.engines)) != len(args.engines): parser.error('engines must be distinct')
    if args.suite == 'standard' and len(args.engines) < 2:
        parser.error('standard suite requires at least two engines for independent large-circuit comparisons')
    specs, rows = suite(args.suite), []
    report = dict(environment=provenance(),configuration=dict(suite=args.suite,precision=args.precision,
                  threads=args.threads,warmup=args.warmup,repeats=args.repeats,engines=args.engines,
                  timeout_seconds=args.timeout,shots=0,execution='sequential fresh subprocesses'),
                  cases=specs,results=rows,all_passed=False)
    environment = dict(os.environ)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        environment[name] = str(args.threads)
    environment['PYTHONUTF8'] = '1'
    with tempfile.TemporaryDirectory(prefix='nexqus-independent-') as directory:
        temp = Path(directory)
        for spec in specs:
            states, probabilities, case_rows = {}, {}, []
            # Fixed randomized order avoids always measuring NexQus first.
            order = list(np.random.default_rng(spec['seed']).permutation(args.engines))
            for engine in order:
                request = dict(engine=str(engine),case=spec,precision=args.precision,threads=args.threads,
                               warmup=args.warmup,repeats=args.repeats,state_path=str(temp/(engine+'.npz')))
                request_path = temp/'request.json'; request_path.write_text(json.dumps(request),encoding='utf-8')
                try:
                    completed = subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(request_path)],
                                               env=environment,capture_output=True,text=True,timeout=args.timeout)
                    if completed.returncode: raise RuntimeError(completed.stderr[-4000:])
                    try: row = json.loads(completed.stdout)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError('Invalid worker output: '+completed.stdout[-2000:]+'\n'+completed.stderr[-2000:]) from exc
                    if row['status'] == 'ok':
                        with np.load(request['state_path'],allow_pickle=False) as data:
                            states[engine] = data['state']; probabilities[engine] = data['probabilities']
                    elif completed.stderr: row['stderr'] = completed.stderr[-4000:]
                except Exception as exc:
                    row = dict(status='error',case=spec['name'],engine=str(engine),error=str(exc))
                case_rows.append(row)
                print(f"{spec['name']}: {engine}: {row['status']}",flush=True)
            if spec['n_qubits'] <= 5:
                reference, reference_name = dense_reference(spec), 'independent dense oracle (complex128)'
            else:
                reference_name = 'qiskit' if 'qiskit' in states else next(iter(states),None)
                reference = states.get(reference_name)
            for row in case_rows:
                if row['status'] == 'ok':
                    row['reference'] = reference_name
                    row['accuracy'] = accuracy(reference,states[row['engine']],spec['method'],args.precision,
                                               probabilities[row['engine']])
            rows.extend(sorted(case_rows,key=lambda r: args.engines.index(r['engine'])))
            write_reports(args.output,report)  # checkpoint even on partial failure
    report['all_passed'] = bool(rows) and all(row['status']=='ok' and row['accuracy']['passed'] for row in rows)
    write_reports(args.output,report)
    print('Reports:',args.output.resolve(), 'all_passed:',report['all_passed'])
    return 0 if report['all_passed'] else 1


if __name__ == '__main__': sys.exit(main())
