"""Fail, rather than skip, if v0.2 CUDA validation cannot run.

Run from the extracted SDK root after installation:
python examples/validate_gpu.py
"""
from pathlib import Path
import sys
import unittest
import numpy as np
from nexqus import Simulator, Noise, NoiseModel, Parameter, Circuit, algorithms, __version__


def main():
    import torch  # Missing PyTorch is a hard failure for this entry point.
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: GPU validation was NOT performed')
    print('NexQus',__version__,'CUDA device:',torch.cuda.get_device_name(0))
    sim=Simulator(backend='torch',device='cuda:0',method='trajectory')
    noise=NoiseModel(Noise.depolarizing(.1))
    result=sim.compile(algorithms.ghz(3),noise).run(trajectories=128,batch_size=16,shots=0,seed=42)
    assert result.metadata['device']=='cuda:0'
    assert result.metadata['batch_size']==16
    assert np.isclose(result.probabilities.sum(),1)
    template=Circuit(1).ry(Parameter('theta'),0)
    results=Simulator(backend='torch',device='cuda:0').run_sweep(template,[{'theta':0},{'theta':np.pi}],shots=0)
    np.testing.assert_allclose(results[1].probabilities,[0,1],atol=1e-10)
    suite=unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'))
    tested=unittest.TextTestRunner(verbosity=2).run(suite)
    if not tested.wasSuccessful() or tested.skipped:
        print('GPU validation incomplete: failures or skipped tests remain')
        sys.exit(1)
    print('GPU validation complete: all tests passed, none skipped')


if __name__=='__main__': main()
