"""Strict real-CUDA validation. CPU-only machines fail preflight, never pass.

python scripts/check_gpu_workflows.py --device cuda:0
Add --include-scf on Linux/WSL with compatible PySCF/GPU4PySCF/CuPy installed.
No cloud access, QPU submission or GPU provisioning is performed.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import unittest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default=os.environ.get('NEXQUS_TEST_CUDA_DEVICE','cuda:0')); p.add_argument('--include-scf',action='store_true')
    p.add_argument('--report',type=Path,default=Path('.release-artifacts/cuda_workflows.json'))
    args=p.parse_args(); args.report.parent.mkdir(parents=True,exist_ok=True)
    report=dict(all_passed=False,requested_device=args.device,hardware_validated=False)
    try:
        import torch
        report.update(torch_version=str(torch.__version__),cuda_runtime=torch.version.cuda)
        from nexqus.torch_circuit import tensor_device
        device=tensor_device(args.device)
        if device.type!='cuda': raise RuntimeError('This validator requires a real CUDA device')
        report.update(torch_version=str(torch.__version__),cuda_runtime=torch.version.cuda,
                      device=str(device),gpu_name=torch.cuda.get_device_name(device))
        os.environ['NEXQUS_TEST_CUDA_DEVICE']=str(device)
        report['configured_default']=os.environ.pop('NEXQUS_DEVICE',None)
        root=Path(__file__).resolve().parents[1]
        suite=unittest.TestSuite([
            unittest.TestLoader().discover(str(root/'tests')),
            unittest.TestLoader().discover(str(root/'tests'/'image_gan'))])
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        report.update(tests_run=result.testsRun,skips=[dict(test=t.id(),reason=r) for t,r in result.skipped],
                      failures=[t.id() for t,_ in result.failures],errors=[t.id() for t,_ in result.errors])
        report['all_passed']=result.wasSuccessful() and not result.skipped
        report['hardware_validated']=report['all_passed']
        if args.include_scf:
            from nexqus.chemistry import molecular_hamiltonian
            import numpy as np
            cpu=molecular_hamiltonian(['H','H'],[[0,0,0],[0,0,.735]])
            gpu=molecular_hamiltonian(['H','H'],[[0,0,0],[0,0,.735]],scf_device=str(device))
            np.testing.assert_allclose(gpu.hartree_fock_energy,cpu.hartree_fock_energy,atol=1e-8,rtol=0)
            np.testing.assert_allclose(gpu.hamiltonian.dense(device='cpu'),cpu.hamiltonian.dense(device='cpu'),atol=1e-8,rtol=0)
            report['gpu4pyscf_scf']='passed; integral generation and mapping remain CPU'
        else: report['gpu4pyscf_scf']='not requested; not hardware validated'
    except Exception as exc:
        report.update(all_passed=False,hardware_validated=False,error=f'{type(exc).__name__}: {exc}')
    args.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    return 0 if report['all_passed'] else 1


if __name__=='__main__': sys.exit(main())
