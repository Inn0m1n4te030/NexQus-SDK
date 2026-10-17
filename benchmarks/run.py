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
        measurement = request['measurement']
        started = time.perf_counter()
        if measurement != 'correctness':
            # Equal shared initialization for every engine. Untimed warmups
            # below also exercise each SDK's own lazy execution paths.
            np.random.default_rng(request['case']['seed'])
        run, settings = prepare(request['engine'],request['case'],request['precision'],request['threads'])
        preparation_seconds = time.perf_counter()-started
        with threadpool_limits(limits=request['threads']):
            pools = threadpool_info()
            if measurement == 'memory':
                with PeakRSS() as memory:
                    for _ in range(request['warmup']):
                        state, probabilities = run()
                        del state, probabilities
                    state, probabilities = run()
                return dict(status='ok',baseline_rss_bytes=memory.baseline,
                            execution_peak_rss_bytes=memory.peak,
                            sampled_incremental_rss_bytes=max(0,memory.peak-memory.baseline),
                            process_peak_rss_bytes=process_peak_rss(),
                            state_bytes=state.nbytes,probabilities_bytes=probabilities.nbytes,
                            memory_measurement='separate fresh subprocess')
            timing = {}
            if measurement == 'performance':
                for _ in range(request['warmup']):
                    state, probabilities = run()
                    del state, probabilities
                samples = []
                for _ in range(request['repeats']):
                    # Previous results are released before the next timed run.
                    started = time.perf_counter()
                    state, probabilities = run()
                    samples.append(time.perf_counter()-started)
                    # Force both full outputs to be materialized on this CPU.
                    if not np.isfinite(state).all() or not np.isfinite(probabilities).all():
                        raise ValueError('Nonfinite output')
                    del state, probabilities
                timing = dict(preparation_seconds=preparation_seconds,seconds=samples,
                              median_seconds=statistics.median(samples),min_seconds=min(samples),
                              max_seconds=max(samples),stdev_seconds=statistics.stdev(samples))
            state, probabilities = run()  # correctness is always outside timing
            np.savez(request['state_path'],state=state,probabilities=probabilities)
        return dict(status='ok',engine=request['engine'],case=request['case']['name'],
                    precision=request['precision'],settings=settings,threadpools=pools,
                    output_dtype=str(state.dtype),**timing)
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
    performance = report['configuration']['measurement'] == 'performance'
    columns = ['case','engine','status','precision']
    if performance:
        columns += ['median_seconds','min_seconds','max_seconds','stdev_seconds','preparation_seconds',
                    'execution_peak_rss_bytes','baseline_rss_bytes','sampled_incremental_rss_bytes','process_peak_rss_bytes']
    columns += ['max_state_error','probability_tvd','normalization_error',
               'accuracy_passed','reference']
    with (output/'results.csv').open('w',newline='',encoding='utf-8') as handle:
        writer = csv.DictWriter(handle,columns,extrasaction='ignore',lineterminator='\n'); writer.writeheader()
        for row in report['results']:
            check = row.get('accuracy',{})
            writer.writerow(dict(row, max_state_error=check.get('max_state_error'),
                                 probability_tvd=check.get('probability_tvd'),
                                 normalization_error=check.get('normalization_error'),accuracy_passed=check.get('passed')))
    p = report['environment']
    lines = ['# '+('CPU performance comparison' if performance else 'Independent correctness checks'), '',
             f"Run: {p['utc_time']}. CPU: {p['cpu']}. Python {p['python']}.", '',
             f"Suite `{report['configuration']['suite']}`; precision `{report['configuration']['precision']}`; "
             f"{report['configuration']['threads']} CPU thread(s).", '',
             'Specifications, versions, settings, hardware and numerical diagnostics: [JSON](results.json). '
             'Tabular data: [CSV](results.csv).', '']
    if performance:
        lines += [f"Every engine receives {report['configuration']['warmup']} untimed warmups and "
                  f"{report['configuration']['repeats']} measured executions. NumPy random initialization is "
                  'performed equally before preparation for all engines. Timing covers prepared execution '
                  'and full state/probability materialization; setup and correctness checks are excluded. '
                  'No memory sampler runs during timing. Raw repetitions are in the JSON.', '',
                  'Memory is measured in a separate fresh subprocess with the same warmup count. RSS '
                  'includes native/Python allocations; sampled peaks can miss allocations shorter than 1 ms. '
                  'OS process peak also includes imports/preparation. Baselines and allocator reuse differ '
                  'between SDKs. No GPU memory is measured.', '',
                  '| Circuit | Engine | Median ms | Execution peak MiB | Process peak MiB | Max state error | Accuracy |',
                  '|---|---|---:|---:|---:|---:|---|']
    else:
        lines += ['This report checks numerical correctness only. It contains no execution timing or '
                  'performance ranking. See the separately warmed CPU performance report for speed comparisons.', '',
                  '| Circuit | Engine | Max state error | Probability TVD | Normalization error | Accuracy |',
                  '|---|---|---:|---:|---:|---|']
    for row in report['results']:
        if row['status'] == 'ok':
            check = row['accuracy']; peak = row.get('process_peak_rss_bytes')
            prefix = f"| {row['case']} | {row['engine']} | "
            if performance:
                prefix += (f"{row['median_seconds']*1000:.3f} | {row['execution_peak_rss_bytes']/2**20:.1f} | "
                           f"{format(peak/2**20,'.1f') if peak is not None else 'n/a'} | ")
            diagnostics = [check.get('max_state_error')]
            if not performance: diagnostics += [check.get('probability_tvd'),check.get('normalization_error')]
            lines.append(prefix+' | '.join(format(v,'.2e') if v is not None else 'n/a' for v in diagnostics)+
                         f" | {'pass' if check['passed'] else 'FAIL'} |")
        else:
            lines.append(f"| {row['case']} | {row['engine']} | "+'— | '*(4 if performance else 3)+row['status']+' |')
    lines += ['', 'Small circuits (up to 5 qubits) use an independent full-matrix NumPy oracle. Larger circuits '
              'use Qiskit Aer as the cross-simulator reference when available; the JSON labels every reference. '
              'Statevector errors are aligned for global phase. Density errors compare full matrix entries. '
              'Probability TVD, normalization, Hermiticity, and small-density eigenvalues are also checked.', '',
              'All engines use identical gate/noise specifications, precision, zero shots and CPU settings. '
              'Fusion and idle-wire truncation are disabled. PennyLane uses default.qubit/default.mixed, '
              'not Lightning. Keep slower NexQus results; performance describes this host and these workloads.', '']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',choices=['smoke','correctness','standard'],default='smoke')
    parser.add_argument('--engines',nargs='+',choices=['nexqus','qiskit','pennylane'],default=['nexqus','qiskit','pennylane'])
    parser.add_argument('--precision',choices=['single','double'],default='double')
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--warmup',type=int,default=3,help='Untimed warmups for the standard performance suite')
    parser.add_argument('--repeats',type=int,default=9,help='Timed repetitions for the standard performance suite')
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
    if args.suite == 'standard' and (args.warmup < 2 or args.repeats < 3):
        parser.error('performance comparisons require at least 2 equal warmups and 3 repetitions per engine')
    measurement = 'performance' if args.suite == 'standard' else 'correctness'
    specs, rows = suite(args.suite), []
    configuration = dict(suite=args.suite,measurement=measurement,precision=args.precision,
                  threads=args.threads,engines=args.engines,
                  timeout_seconds=args.timeout,shots=0,execution='sequential fresh subprocesses')
    if measurement == 'performance': configuration.update(warmup=args.warmup,repeats=args.repeats,
                      memory_instrumented_timing=False,memory_measurement='separate fresh subprocess')
    report = dict(environment=provenance(),configuration=configuration,cases=specs,results=rows,all_passed=False)
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
                request = dict(engine=str(engine),case=spec,precision=args.precision,threads=args.threads,measurement=measurement,
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
                        if measurement == 'performance':
                            request['measurement'] = 'memory'
                            request_path.write_text(json.dumps(request),encoding='utf-8')
                            memory_run = subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(request_path)],
                                                        env=environment,capture_output=True,text=True,timeout=args.timeout)
                            if memory_run.returncode: raise RuntimeError(memory_run.stderr[-4000:])
                            memory = json.loads(memory_run.stdout)
                            if memory.pop('status') != 'ok': raise RuntimeError(str(memory))
                            row.update(memory)
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
