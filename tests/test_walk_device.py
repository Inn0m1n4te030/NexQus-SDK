"""Device parity for walk circuits, with strict CUDA execution when available."""
import unittest
from unittest.mock import patch
import numpy as np
from nexqus.glued_trees import ColumnQuantumWalk
from nexqus.maze import Maze, PRESETS
from nexqus.maze_walk import compare_walks
from nexqus.walk_device import walk_simulator

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Optional PyTorch not installed')
class WalkDeviceTests(unittest.TestCase):
    def check_device(self, device):
        reference=ColumnQuantumWalk(4).states([0,1,5,20])
        model=ColumnQuantumWalk(4,device=device,backend='torch')
        self.assertEqual(model.simulator.backend.device,device)
        np.testing.assert_allclose(model.states([0,1,5,20]),reference,atol=1e-12)
        maze=Maze(PRESETS['Two corridors'])
        baseline=compare_walks(maze,8)
        actual=compare_walks(maze,8,device=device,backend='torch')
        self.assertEqual(actual['timings']['device'],device)
        for a,b in zip(actual['quantum'],baseline['quantum']):
            np.testing.assert_allclose(a['probabilities'],b['probabilities'],atol=1e-12)
            self.assertAlmostEqual(a['cumulative_hit'],b['cumulative_hit'])

    def test_torch_cpu_parity(self): self.check_device('cpu')

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'CUDA GPU unavailable')
    def test_cuda_parity(self): self.check_device('cuda')

    def test_cuda_unavailable_is_an_error(self):
        with patch('torch.cuda.is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'unavailable'): walk_simulator('cuda')


if __name__=='__main__': unittest.main()
