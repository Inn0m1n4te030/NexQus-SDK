"""Check exactly one wheel/sdist pair, install each in a fresh environment,
and run the full suite against the installed package outside the source tree.

Run after `python -m build`: python scripts/check_distribution.py.
Builds a wheel from the sdist too. Never uploads. Network is needed to install
NumPy and isolated build dependencies unless your pip cache supplies them.
"""
import argparse
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def run(command,cwd,env):
    subprocess.run([str(part) for part in command],cwd=cwd,env=env,check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dist',type=Path,default=ROOT/'dist')
    parser.add_argument('--report',type=Path,default=ROOT/'.release-artifacts'/'distribution_validation.json')
    args=parser.parse_args()
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(dict(all_passed=False,status='running'))+'\n',encoding='utf-8')
    directory=args.dist.resolve()
    wheels=sorted(directory.glob('*.whl')); sdists=sorted(directory.glob('*.tar.gz'))
    if len(wheels)!=1 or len(sdists)!=1:
        parser.error('Use a clean dist folder containing exactly one wheel and one sdist')
    records=[]
    with zipfile.ZipFile(wheels[0]) as archive:
        names=archive.namelist()
        required={'nexqus/__init__.py','nexqus/dynamic.py','nexqus/simulator.py','nexqus/validation.py'}
        if not required.issubset(names): raise ValueError('Wheel is missing runtime files')
        if any(name.startswith(('tests/','examples/','notebooks/')) for name in names):
            raise ValueError('Tutorials/tests belong in the source release, not the runtime wheel')
        metadata_name=next(name for name in names if name.endswith('.dist-info/METADATA'))
        metadata=BytesParser().parsebytes(archive.read(metadata_name))
        version=metadata['Version']
        if metadata['Name']!='nexqus-sdk' or metadata['Description-Content-Type']!='text/markdown':
            raise ValueError('Invalid wheel metadata')
        if len([name for name in names if name.endswith('.dist-info/licenses/LICENSE')])!=1:
            raise ValueError('Wheel is missing the license notice')
        sources={name:archive.read(name) for name in names if name.startswith('nexqus/') and name.endswith('.py')}
        for name,data in sources.items():
            if data!=(ROOT/'src'/name).read_bytes(): raise ValueError(f'Wheel/source mismatch: {name}')
    with tarfile.open(sdists[0],'r:gz') as archive:
        names=archive.getnames(); base=names[0].split('/')[0]
        expected=['pyproject.toml','LICENSE','README.md','scripts/check_release.py',
                  'tests/test_release.py','notebooks/00_start_here.ipynb',
                  'notebooks/07_inspection_and_qml_templates.ipynb','benchmarks/run.py',
                  '.github/workflows/ci.yml']
        if not all(f'{base}/{path}' in names for path in expected):
            raise ValueError('Source distribution lacks release, test or tutorial files')
        public_docs={'API.md','API_STABILITY.md','BURMESE_QUICKSTART.md',
                     'PERFORMANCE.md','LEARNING.md','DYNAMIC_CIRCUITS.md',
                     'RELEASE_NOTES.md','RELEASING.md','QML_TEMPLATES.md','BRAKET.md','EXPLAINABILITY.md','QGAN_CHEMISTRY.md','MOLECULES.md','IMAGE_GANS.md','GPU.md','QUANTUM_MAZE.md',
                     'NLP.md','NLP_SIMILARITY.md','images/teleportation.svg','images/custom_8_qubits.svg',
                     'images/sine_comparison.svg','images/sine_comparison.png','images/methane.svg','images/benzene.svg'}
        packaged_docs={name.removeprefix(f'{base}/docs/') for name in names
                       if name.startswith(f'{base}/docs/') and archive.getmember(name).isfile()}
        if packaged_docs!=public_docs:
            raise ValueError(f'Public documentation contents differ: {packaged_docs ^ public_docs}')
        for name in public_docs:
            if archive.extractfile(f'{base}/docs/{name}').read()!=(ROOT/'docs'/name).read_bytes():
                raise ValueError(f'Source documentation mismatch: {name}')
        if any('/.release-artifacts/' in name for name in names):
            raise ValueError('Internal release reports must not be packaged')
    environment=dict(os.environ)
    environment.pop('PYTHONPATH',None)
    environment['PYTHONUTF8']='1'
    # Keep source checkout/editable installs out of both test environments.
    with tempfile.TemporaryDirectory(prefix='nexqus-dist-') as temp:
        scratch=Path(temp)
        rebuilt=scratch/'rebuilt'; rebuilt.mkdir()
        source=scratch/'source'; source.mkdir()
        with tarfile.open(sdists[0],'r:gz') as archive:
            for member in archive.getmembers():
                destination=(source/member.name).resolve()
                if not destination.is_relative_to(source) or not (member.isdir() or member.isfile()):
                    raise ValueError('Unsafe source archive member')
                if member.isdir(): destination.mkdir(parents=True,exist_ok=True)
                else:
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    destination.write_bytes(archive.extractfile(member).read())
        run([sys.executable,'-m','build','--wheel','--outdir',rebuilt,source/base],scratch,environment)
        rebuilt_wheel=next(rebuilt.glob('*.whl'))
        with zipfile.ZipFile(rebuilt_wheel) as archive:
            rebuilt_sources={name:archive.read(name) for name in archive.namelist() if name.startswith('nexqus/') and name.endswith('.py')}
            if rebuilt_sources!=sources: raise ValueError('Source-built runtime differs from wheel')
        for kind,wheel in [('wheel',wheels[0]),('source_rebuilt_wheel',rebuilt_wheel)]:
            target=scratch/kind
            venv.EnvBuilder(with_pip=True).create(target)
            python=target/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
            run([python,'-m','pip','install','--disable-pip-version-check',wheel],scratch,environment)
            code=('import importlib.metadata as m, pathlib, nexqus; '
                  'assert nexqus.__version__ == '+repr(version)+' == m.version("nexqus-sdk"); '
                  'assert not pathlib.Path(nexqus.__file__).is_relative_to('+repr(str(ROOT))+'); '
                  'from nexqus import Simulator, algorithms; '
                  'r=Simulator().run(algorithms.bell(), shots=0); '
                  'assert abs(r.expectation("ZZ")-1)<1e-10; '
                  'from nexqus import Circuit, Parameter, QuantumDistribution, QGAN, Molecule; '
                  'QGAN(QuantumDistribution(Circuit(1).ry(Parameter("theta"),0)), {"theta":1.3}); '
                  'from nexqus.maze import Maze, PRESETS; '
                  'from nexqus.maze_walk import compare_walks; '
                  'from nexqus.glued_trees import experiment; '
                  'assert compare_walks(Maze(PRESETS["Two corridors"]),2)["steps"] == 2; '
                  'assert experiment(2,17)["columns"] == 6; '
                  'print("Installed distribution:", nexqus.__version__, nexqus.__file__)')
            run([python,'-c',code],scratch,environment)
            run([python,'-m','unittest','discover','-s',ROOT/'tests'],scratch,environment)
            records.append(dict(kind=kind,status='passed',version=version))
    report=dict(all_passed=True,version=version,checks=records,
                artifacts=[dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
                           for path in [wheels[0],sdists[0]]])
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('Distribution checks passed:',args.report)


if __name__=='__main__': main()
