"""Independent order-finding distributions and factoring regressions."""
import unittest
from unittest.mock import Mock
import numpy as np
from nexqus import Simulator, algorithms


class ShorTests(unittest.TestCase):
    def fake_simulator(self):
        simulator = Mock(spec=Simulator)
        simulator.precision = 'double'
        simulator.method = 'auto'
        simulator.max_memory_mb = 1024
        return simulator

    def test_order_four_exact_peaks(self):
        result = Simulator().run(algorithms.order_finding(2, 15), shots=0)
        expected = np.zeros(256)
        expected[[0, 64, 128, 192]] = .25
        np.testing.assert_allclose(result.marginal(range(8)), expected, atol=1e-12)

    def test_order_six_against_fourier_sum(self):
        # Independent analytical distribution: trace out equal modular residues.
        size, order = 1024, 6
        y = np.arange(size)
        expected = np.zeros(size)
        for residue in range(order):
            x = np.arange(residue, size, order)
            expected += abs(np.exp(2j * np.pi * np.outer(y, x) / size).sum(axis=1))**2 / size**2
        result = Simulator().run(algorithms.order_finding(2, 21), shots=0)
        np.testing.assert_allclose(result.marginal(range(10)), expected, atol=1e-12)

    def test_modular_gates_on_every_basis_state(self):
        circuit = algorithms.order_finding(2, 21, counting_qubits=3)
        gates = [op for op in circuit.operations if op.name == 'controlled_modular_multiply']
        for j, op in enumerate(gates):
            multiplier = pow(2, 2**(2-j), 21)
            for column in range(64):
                control, value = divmod(column, 32)
                target = control * 32 + (multiplier * value % 21 if control and value < 21 else value)
                self.assertEqual(op.kind, "permutation")
                self.assertEqual(op.data[column], target)

    def test_quantum_factorization(self):
        for modulus, factors, order in [(15, (3, 5), 4), (21, (3, 7), 6)]:
            result = algorithms.shor(modulus, base=2, seed=42)
            self.assertEqual(result.factors, factors)
            self.assertEqual(result.method, 'quantum')
            self.assertEqual(result.order, order)

    def test_seed_reproducibility(self):
        self.assertEqual(algorithms.shor(15, seed=7), algorithms.shor(15, seed=7))

    def test_classical_shortcuts(self):
        simulator = self.fake_simulator()
        for modulus, factors, method in [(12, (2, 6), 'even'), (9, (3, 3), 'perfect_power'),
                                          (27, (3, 9), 'perfect_power'), (31, None, 'prime'),
                                          (15, (3, 5), 'gcd')]:
            result = algorithms.shor(modulus, base=3, simulator=simulator)
            self.assertEqual((result.factors, result.method), (factors, method))
        simulator.run.assert_not_called()

    def test_zero_phases_fail_without_classical_factoring(self):
        simulator = self.fake_simulator()
        simulator.run.return_value.counts = {'0' * 12: 32}
        result = algorithms.shor(15, base=2, simulator=simulator, max_attempts=2)
        self.assertIsNone(result.factors)
        self.assertEqual((result.method, result.attempts), ('failed', 2))
        self.assertEqual(simulator.run.call_count, 2)

    def test_combines_reduced_phase_denominators(self):
        simulator = self.fake_simulator()
        # 1/2 and approximately 1/3 together recover r=6 for 2 modulo 21.
        simulator.run.return_value.counts = {format(512, '010b') + '00001': 1,
                                            format(341, '010b') + '00001': 1}
        result = algorithms.shor(21, base=2, simulator=simulator, max_attempts=1)
        self.assertEqual((result.factors, result.order), ((3, 7), 6))

    def test_inaccurate_phase_does_not_poison_later_samples(self):
        simulator = self.fake_simulator()
        simulator.run.return_value.counts = {format(205, '010b') + '00001': 1,
                                            format(512, '010b') + '00001': 1,
                                            format(341, '010b') + '00001': 1}
        result = algorithms.shor(21, base=2, simulator=simulator, max_attempts=1)
        self.assertEqual((result.factors, result.order), ((3, 7), 6))

    def test_unhelpful_order(self):
        result = algorithms.shor(15, base=14, seed=42, max_attempts=1)
        self.assertEqual(result.method, 'failed')
        self.assertIsNone(result.factors)

    def test_larger_quantum_factorization(self):
        for modulus, factors in [(35, (5, 7)), (77, (7, 11))]:
            result = algorithms.shor(modulus, base=2, seed=42)
            self.assertEqual((result.factors, result.method), (factors, 'quantum'))

    def test_resources_and_preallocation_guards(self):
        resources = algorithms.shor_resources(77)
        self.assertEqual((resources.target_qubits, resources.counting_qubits,
                          resources.total_qubits), (7, 14, 21))
        self.assertEqual(resources.statevector_bytes, 32 * 1024**2)
        q = algorithms.order_finding(2, 77)
        self.assertEqual(sum(op.data.nbytes for op in q.operations if op.kind == 'permutation'),
                         resources.permutation_bytes)
        with self.assertRaises(MemoryError):
            algorithms.shor(77, simulator=Simulator(max_memory_mb=1))
        with self.assertRaises(MemoryError):
            algorithms.order_finding(2, 77, max_memory_mb=.001)
        with self.assertRaises(MemoryError):
            algorithms.shor_resources(2**100 + 1)
        with self.assertRaises(ValueError):
            algorithms.shor(14, counting_qubits=0)
        # Circuit construction is distinct from state allocation.
        self.assertEqual(algorithms.order_finding(2, 143).n_qubits, 24)
        self.assertEqual(algorithms.shor_resources(143, 8).total_qubits, 16)
        self.assertEqual(algorithms.shor_resources(77, precision='single').statevector_bytes,
                         resources.statevector_bytes // 2)

    def test_validation(self):
        for modulus in [True, 2, 15.0, -15]:
            with self.assertRaises(ValueError): algorithms.shor(modulus)
            with self.assertRaises(ValueError): algorithms.order_finding(2, modulus)
        for base in [True, 1, 15, 2.5, -1]:
            with self.assertRaises(ValueError): algorithms.shor(15, base=base)
            with self.assertRaises(ValueError): algorithms.order_finding(base, 15)
        with self.assertRaises(ValueError): algorithms.order_finding(3, 15)
        for count in [0, True, 2.5]:
            with self.assertRaises(ValueError): algorithms.order_finding(2, 15, count)
        for name in ['shots', 'max_attempts']:
            for value in [0, True, -1, 1.5]:
                with self.assertRaises(ValueError): algorithms.shor(15, **{name: value})


if __name__ == '__main__':
    unittest.main()
