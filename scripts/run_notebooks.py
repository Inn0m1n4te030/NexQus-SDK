"""Execute the tutorial series, recording the engine and optional skips.

auto uses nbclient/Jupyter if available; otherwise it runs these plain Python
cells in fresh subprocess namespaces. The fallback has no magics/IPython or
interactive-widget support. It captures stdout and Matplotlib PNGs into .ipynb.
Use --engine jupyter --kernel nexqus for a real registered Jupyter kernel.
"""
import argparse
import ast
import base64
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]


def source_version():
    tree=ast.parse((ROOT/'src'/'nexqus'/'__init__.py').read_text(encoding='utf-8'))
    for statement in tree.body:
        if isinstance(statement,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='__version__' for t in statement.targets):
            return ast.literal_eval(statement.value)
    raise ValueError('Missing source version')


def reported_sdk_version(notebook):
    """Read complete stdout lines; Jupyter may split a single print() message.

    Join each cell's stdout chunks before parsing. Do not mix stderr or rich
    display data into the marker, or join incomplete lines across cells.
    """
    versions=[]
    for cell in notebook['cells']:
        if cell['cell_type']!='code': continue
        chunks=[]
        for output in cell.get('outputs',[]):
            if output.get('output_type')!='stream' or output.get('name')!='stdout': continue
            value=output.get('text','')
            chunks.append(value if isinstance(value,str) else ''.join(value))
        for line in ''.join(chunks).splitlines():
            if line.startswith('NexQus SDK: '):
                versions.append(line.removeprefix('NexQus SDK: ').strip())
    return versions[0] if len(versions)==1 and versions[0] else None


def python_worker(path):
    os.environ['MPLBACKEND']='Agg'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    notebook=json.loads(path.read_text(encoding='utf-8'))
    namespace={'__name__':'__main__'}
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            cell['execution_count']=None; cell['outputs']=[]
    execution_count=0; status='passed'; code_cells=0; optional_skips=0
    started=time.perf_counter()
    for cell in notebook['cells']:
        if cell['cell_type']!='code': continue
        execution_count+=1; code_cells+=1; outputs=[]
        cell['execution_count']=execution_count; cell['outputs']=outputs
        stdout=io.StringIO(); stderr=io.StringIO()
        def flush():
            text=stdout.getvalue()
            if text: outputs.append(dict(output_type='stream',name='stdout',text=text))
            stdout.seek(0); stdout.truncate(0)
        def show(*args,**kwargs):
            flush()
            for number in plt.get_fignums():
                fig=plt.figure(number); image=io.BytesIO()
                fig.savefig(image,format='png',dpi=130,bbox_inches='tight')
                outputs.append(dict(output_type='display_data',metadata={},data={
                    'image/png':base64.b64encode(image.getvalue()).decode(),
                    'text/plain':'Matplotlib figure (rendered by standalone Python runner)'}))
            plt.close('all')
        plt.show=show
        source=cell['source']
        if isinstance(source,list): source=''.join(source)
        try:
            with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
                exec(compile(source,f'{path.name}:cell{execution_count}','exec'),namespace)
            flush()
        except Exception as exc:
            flush(); status='failed'
            outputs.append(dict(output_type='error',ename=type(exc).__name__,evalue=str(exc),
                                traceback=traceback.format_exc().splitlines()))
        if stderr.getvalue(): outputs.append(dict(output_type='stream',name='stderr',text=stderr.getvalue()))
        optional_skips+=sum(output.get('text','').count('SKIPPED optional CUDA') for output in outputs)
        if status=='failed': break
    notebook['metadata']['language_info']['version']=platform.python_version()
    notebook['metadata']['nexqus']['execution']=dict(engine='standalone_python',status=status,
                          code_cells=code_cells,optional_skips=optional_skips,
                          note='Plain Python cells executed; not a Jupyter/IPython kernel execution')
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    record=dict(notebook=path.name,engine='standalone_python',status=status,code_cells=code_cells,
                optional_skips=optional_skips,seconds=time.perf_counter()-started,
                sdk_version=getattr(namespace.get('nexqus'),'__version__',None))
    print(json.dumps(record))
    return 0 if status=='passed' else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine',choices=['auto','python','jupyter'],default='auto')
    p.add_argument('--kernel',default='nexqus')
    p.add_argument('--transport',choices=['tcp','ipc'],default='tcp',
                   help='Jupyter transport; IPC is useful on Unix hosts without local TCP sockets')
    p.add_argument('--timeout',type=int,default=120,help='Jupyter cell timeout / standalone notebook process timeout')
    p.add_argument('--require-gpu',action='store_true')
    p.add_argument('--report',type=Path,default=ROOT/'.release-artifacts'/'notebook_execution.json')
    p.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.worker: return python_worker(args.worker.resolve())
    if args.timeout<1: p.error('timeout must be positive')
    if args.require_gpu:
        try:
            import torch
        except ImportError: p.error('--require-gpu needs compatible PyTorch/CUDA')
        if not torch.cuda.is_available(): p.error('--require-gpu requested but CUDA unavailable')
    available=all(importlib.util.find_spec(name) is not None for name in ['nbformat','nbclient','ipykernel'])
    engine=('jupyter' if available else 'python') if args.engine=='auto' else args.engine
    if engine=='jupyter' and not available: p.error('Install the tutorial extra before --engine jupyter')
    paths=sorted((ROOT/'notebooks').glob('*.ipynb'))
    if not paths: p.error('No notebooks found; run scripts/build_notebooks.py')
    records=[]
    expected_version=source_version()
    for path in paths:
        print(f'Executing {path.name} with {engine}...',flush=True)
        if engine=='python':
            try:
                result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(path)],
                                      cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=args.timeout)
                try: record=json.loads(result.stdout)
                except json.JSONDecodeError:
                    record=dict(notebook=path.name,engine='standalone_python',status='failed',error=result.stderr[-4000:])
                if result.returncode:
                    record['status']='failed'
                    print(result.stderr or record,flush=True)
            except subprocess.TimeoutExpired:
                record=dict(notebook=path.name,engine='standalone_python',status='failed',
                            error=f'Notebook exceeded {args.timeout} seconds')
        else:
            import nbformat
            from nbclient import NotebookClient
            from traitlets.config import Config
            notebook=nbformat.read(path,as_version=4); started=time.perf_counter()
            execution_error=None
            for cell in notebook.cells:
                if cell.cell_type=='code': cell.execution_count=None; cell.outputs=[]
            try:
                NotebookClient(notebook,timeout=args.timeout,kernel_name=args.kernel,
                               config=Config({'KernelManager':{'transport':args.transport}}),
                               resources={'metadata':{'path':str(ROOT)}}).execute()
                status='passed'
            except Exception as exc:
                status='failed'; execution_error=str(exc); print(execution_error,flush=True)
            skips=sum(output.get('text','').count('SKIPPED optional CUDA') for cell in notebook.cells
                      if cell.cell_type=='code' for output in cell.get('outputs',[]))
            record=dict(notebook=path.name,engine='jupyter',status=status,transport=args.transport,
                        sdk_version=reported_sdk_version(notebook),
                        code_cells=sum(c.cell_type=='code' for c in notebook.cells),
                        optional_skips=skips,seconds=time.perf_counter()-started)
            if execution_error is not None: record['error']=execution_error
        record['expected_sdk_version']=expected_version
        if record.get('sdk_version')!=expected_version and record['status']=='passed':
            record['status']='failed'
            record['error']='Notebook kernel did not report the expected SDK version'
        if engine=='jupyter':
            notebook.metadata.nexqus.execution=record
            nbformat.write(notebook,path)
        else:
            notebook=json.loads(path.read_text(encoding='utf-8'))
            notebook['metadata']['nexqus']['execution']=record
            path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
        records.append(record)
        print(f"  {record['status']} ({record.get('code_cells',0)} code cells)",flush=True)
    report=dict(engine=engine,python=platform.python_version(),sdk_version=expected_version,notebooks=records,
                all_passed=all(r['status']=='passed' for r in records) and
                           not (args.require_gpu and any(r.get('optional_skips',0)>0 for r in records)),
                gpu_required=args.require_gpu,
                execution_note=('Jupyter kernel execution; see per-notebook status and SDK version' if engine=='jupyter' else
                                'Standalone engine tests plain Python cells, not a Jupyter kernel'))
    destination=args.report.resolve()
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f'Saved execution report: {destination}')
    return 0 if report['all_passed'] else 1


if __name__=='__main__': sys.exit(main())
