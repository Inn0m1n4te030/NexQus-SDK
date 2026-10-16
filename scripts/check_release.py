"""Run release regressions and write machine-readable validation evidence.

--require-torch fails if a CPU Torch test is skipped. --require-gpu fails for
any skipped test. --public also checks public-distribution metadata and license.
This script performs no uploads and does not claim PyPI name ownership.
"""
import argparse
import ast
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]


def source_version():
    tree=ast.parse((ROOT/'src'/'nexqus'/'__init__.py').read_text(encoding='utf-8'))
    for statement in tree.body:
        if isinstance(statement,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='__version__' for t in statement.targets):
            return ast.literal_eval(statement.value)
    raise ValueError('Missing source version')


def public_metadata_errors():
    # Check the installed public metadata as well as the source license.
    metadata=importlib.metadata.metadata('nexqus-sdk')
    errors=[]
    license_expression=metadata.get('License-Expression','')
    if not license_expression or license_expression.startswith('LicenseRef-'):
        errors.append('Select public distribution terms: replace LICENSE and update project.license')
    if 'No public distribution license has been selected' in (ROOT/'LICENSE').read_text(encoding='utf-8'):
        errors.append('LICENSE is still the private review notice')
    if not metadata.get_payload().strip() or metadata.get('Description-Content-Type')!='text/markdown':
        errors.append('Missing Markdown project description')
    urls=metadata.get_all('Project-URL',[])
    for label in ('Source','Issues'):
        if not any(value.startswith(label+', https://') for value in urls):
            errors.append(f'Add a verified {label} URL to project.urls')
    return errors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-torch',action='store_true')
    parser.add_argument('--require-gpu',action='store_true')
    parser.add_argument('--public',action='store_true')
    parser.add_argument('--report',type=Path,default=ROOT/'.release-artifacts'/'release_validation.json')
    args=parser.parse_args()
    path=args.report.resolve(); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(dict(all_passed=False,status='running'))+'\n',encoding='utf-8')
    import nexqus
    expected=source_version()
    errors=[]
    if nexqus.__version__!=expected or importlib.metadata.version('nexqus-sdk')!=expected:
        errors.append('Installed distribution/source version mismatch; install this source release first')
    torch_info=None
    try:
        import torch
        torch_info=dict(version=torch.__version__,cuda_available=torch.cuda.is_available(),
                        cuda_runtime=torch.version.cuda)
    except ImportError:
        if args.require_torch or args.require_gpu: errors.append('Required PyTorch is not installed')
    if args.require_gpu and (not torch_info or not torch_info['cuda_available']):
        errors.append('CUDA is required but unavailable')
    if args.public: errors.extend(public_metadata_errors())
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if result.testsRun==0: errors.append('No tests were discovered')
    skipped=[dict(test=test.id(),reason=reason) for test,reason in result.skipped]
    if args.require_gpu and skipped: errors.append('GPU release validation requires zero skipped tests')
    if args.require_torch and any(item['reason']!='CUDA GPU unavailable' for item in skipped):
        errors.append('Torch CPU validation has unexpected skipped tests')
    passed=result.wasSuccessful() and not errors
    report=dict(sdk_version=nexqus.__version__,utc_time=datetime.now(timezone.utc).isoformat(),
                python=platform.python_version(),numpy=importlib.metadata.version('numpy'),
                platform=platform.platform(),package_path=str(Path(nexqus.__file__).resolve()),
                source_version=expected,torch=torch_info,require_gpu=args.require_gpu,
                require_torch=args.require_torch,public_metadata_required=args.public,
                tests_run=result.testsRun,tests_passed=result.testsRun-len(result.skipped)-len(result.failures)-len(result.errors),
                skips=skipped,failures=[test.id() for test,_ in result.failures],
                errors=[test.id() for test,_ in result.errors],release_gate_errors=errors,
                all_passed=passed,git_commit=os.environ.get('GITHUB_SHA'))
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('Saved release evidence:',path)
    for error in errors: print('RELEASE GATE:',error,file=sys.stderr)
    return 0 if passed else 1


if __name__=='__main__': sys.exit(main())
