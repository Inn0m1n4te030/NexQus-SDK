"""Permutation kernels compared to independently constructed dense operators."""
import importlib.util
import unittest
import numpy as np
from nexqus import Circuit, Simulator, Noise, NoiseModel, Parameter, QuantumFunction


class PermutationTests(unittest.TestCase):
    def pair(self):
        mapping = [3, 7, 1, 6, 0, 2, 5, 4]
        matrix = np.zeros((8, 8), complex)
        matrix[mapping, np.arange(8)] = 1
        sparse = Circuit(4).h(1).ry(.37, 3).permutation(mapping, (3, 0, 2), 'P')
        dense = Circuit(4).h(1).ry(.37, 3).unitary(matrix, (3, 0, 2), 'P')
        return sparse, dense

    def check_backend(self, backend='numpy', device='cpu', precision='double'):
        sparse, dense = self.pair()
        rng = np.random.default_rng(123)
        psi = rng.normal(size=16) + 1j * rng.normal(size=16)
        psi /= np.linalg.norm(psi)
        tol = 2e-6 if precision == 'single' else 1e-12
        for local in (True, False):
            for method in ('statevector', 'density_matrix', 'trajectory'):
                sim = Simulator(backend=backend, device=device, precision=precision, method=method, local_gates=local)
                opts = dict(initial_state=psi, shots=0, trajectories=7, batch_size=3)
                a = sim.run(sparse, **opts); b = sim.run(dense, **opts)
                np.testing.assert_allclose(a.probabilities, b.probabilities, atol=tol)
                if method == 'statevector':
                    np.testing.assert_allclose(a.statevector, b.statevector, atol=tol)
                elif method == 'density_matrix':
                    np.testing.assert_allclose(a.density_matrix, b.density_matrix, atol=tol)
                compiled = sim.compile(sparse)
                np.testing.assert_allclose(compiled.run(**opts).probabilities, b.probabilities, atol=tol)
                self.assertTrue(any(v.dtype == (sim.backend.torch.int64 if sim.backend.torch else np.int64)
                                    for v in compiled._arrays.values()))

    def test_numpy_all_modes(self):
        self.check_backend()
        self.check_backend(precision='single')

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional PyTorch not installed')
    def test_torch_cpu_all_modes(self):
        self.check_backend('torch')
        self.check_backend('torch', precision='single')

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional PyTorch not installed')
    def test_torch_cuda_all_modes(self):
        import torch
        if not torch.cuda.is_available(): self.skipTest('CUDA unavailable')
        self.check_backend('torch', 'cuda')

    def test_inverse_compose_and_snapshot(self):
        sparse, _ = self.pair()
        circuit = Circuit(5).h(4).compose(sparse, (4, 1, 2, 0)).compose(sparse.inverse(), (4, 1, 2, 0))
        expected = Simulator().run(Circuit(5).h(4), shots=0)
        np.testing.assert_allclose(Simulator().run(circuit, shots=0).statevector, expected.statevector, atol=1e-12)
        mapping = np.array([1, 2, 3, 0])
        q = Circuit(2).permutation(mapping, (1, 0))
        compiled = Simulator().compile(q)
        mapping[:] = 0
        self.assertFalse(q.operations[0].data.flags.writeable)
        q.x(1)
        self.assertEqual(compiled.run(shots=0).most_likely, '10')

    def test_dynamic_and_noise(self):
        a, b = self.pair()
        for q in (a, b):
            q.measure(1, 'flag')
        a.permutation([1, 2, 3, 0], (2, 0), 'P').c_if('flag', 1)
        matrix = np.zeros((4, 4)); matrix[[1, 2, 3, 0], np.arange(4)] = 1
        b.unitary(matrix, (2, 0), 'P').c_if('flag', 1)
        for method in ('auto', 'trajectory'):
            sim = Simulator(method=method)
            opts = dict(shots=0, seed=5, trajectories=31, batch_size=8)
            left = sim.compile(a).run(**opts); right = sim.run(b, **opts)
            np.testing.assert_allclose(left.probabilities, right.probabilities, atol=1e-12)
            self.assertEqual(left.classical_probabilities, right.classical_probabilities)
        a, b = self.pair()
        for q in (a, b): q.noise(Noise.bit_flip(.2), 0)
        for method in ('auto', 'trajectory'):
            sim = Simulator(method=method)
            opts = dict(shots=0, seed=5, trajectories=31, batch_size=8)
            np.testing.assert_allclose(sim.run(a, **opts).probabilities, sim.run(b, **opts).probabilities, atol=1e-12)

    def test_gradient_with_permutation(self):
        q = Circuit(1).rx(Parameter('theta'), 0).permutation([1, 0], 0)
        f = QuantumFunction(q, 'Z')
        self.assertAlmostEqual(f({'theta': .3}), -np.cos(.3))
        self.assertAlmostEqual(f.gradient({'theta': .3})['theta'], np.sin(.3))

    def test_validation(self):
        for mapping in ([0, 0], [0, 2], [-1, 0], [0., 1.], [False, True], [[0, 1]], [0]):
            with self.assertRaises(ValueError): Circuit(1).permutation(mapping, 0)
        for wires in [(0, 0), (0, 2), ()]:
            with self.assertRaises(ValueError): Circuit(2).permutation([0, 1, 2, 3], wires)


if __name__ == '__main__':
    unittest.main()
