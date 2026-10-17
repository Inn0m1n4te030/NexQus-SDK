import unittest
import numpy as np
from nexqus import Circuit, Parameter, Simulator, PauliHamiltonian
from nexqus.chemistry import jordan_wigner, h2_pair_ansatz, molecular_hamiltonian


class HamiltonianTests(unittest.TestCase):
    def test_weighted_expectation_and_gradient(self):
        h = PauliHamiltonian(2, [(1.2, 'II'), (-.7, 'ZZ'), (.3, 'XI'), (.2, 'XI')])
        t = Parameter('t'); c = Circuit(2).ry(t, 0).rx(t*.5, 1).cx(0, 1)
        f = h.function(c); b = {'t': .7}
        result = Simulator().run(c.bind(b), shots=0, retain_state=True)
        expected = np.vdot(result.statevector, h.dense()@result.statevector).real
        self.assertAlmostEqual(h.expectation(result), expected)
        value, g = f.value_and_grad(b)
        self.assertAlmostEqual(value, expected)
        self.assertAlmostEqual(g['t'], (f({'t': .700001})-f({'t': .699999}))/2e-6, places=7)
        self.assertAlmostEqual(f(b), value)

    def test_zero_constant_validation_and_limits(self):
        f = PauliHamiltonian(1, []).function(Circuit(1).ry(Parameter('t'), 0))
        self.assertEqual(f.value_and_grad({'t': 1}), (0., {'t': 0.}))
        with self.assertRaises(ValueError): f({})
        self.assertEqual(PauliHamiltonian(1, [(3, 'I')]).function(Circuit(1))({}), 3)
        for terms in ([(1j, 'X')], [(np.inf, 'X')], [(1, 'XX')], [(1, 'A')]):
            with self.assertRaises(ValueError): PauliHamiltonian(1, terms)
        with self.assertRaises(ValueError): PauliHamiltonian(11, []).dense()
        with self.assertRaises(ValueError): PauliHamiltonian(1, []).function(Circuit(2))
        with self.assertRaises(ValueError): PauliHamiltonian(1, []).function(Circuit(1).measure(0, 'm'))

    def test_h2_ansatz(self):
        for angle in (0., .7, -1., np.pi):
            r = Simulator().run(h2_pair_ansatz(angle), shots=0, retain_state=True)
            expected = np.zeros(16); expected[12] = np.cos(angle/2); expected[3] = np.sin(angle/2)
            np.testing.assert_allclose(r.statevector, expected, atol=1e-12)

    def test_jordan_wigner_independent_fock_matrix(self):
        # Independent occupation-basis ladder signs, without Pauli multiplication.
        n = 4; size = 2**n; annihilation = []
        for wire in range(n):
            a = np.zeros((size, size))
            for source in range(size):
                bits = [(source >> (n-1-q)) & 1 for q in range(n)]
                if bits[wire]:
                    a[source ^ (1 << (n-1-wire)), source] = (-1)**sum(bits[:wire])
            annihilation.append(a)
        h = np.array([[-1., .2], [.2, -.3]])
        b = np.array([[.8, .15], [.15, .5]]); eri = np.einsum('pq,rs->pqrs', b, b)
        dense = .4*np.eye(size)
        for p in range(2):
            for q in range(2):
                for spin in range(2):
                    dense += h[p, q]*annihilation[2*p+spin].T@annihilation[2*q+spin]
                for r in range(2):
                    for s in range(2):
                        for alpha in range(2):
                            for beta in range(2):
                                dense += (.5*eri[p, q, r, s]*annihilation[2*p+alpha].T
                                          @annihilation[2*r+beta].T@annihilation[2*s+beta]
                                          @annihilation[2*q+alpha])
        np.testing.assert_allclose(jordan_wigner(h, eri, .4).dense(), dense, atol=1e-12)

    def test_integral_validation(self):
        for h, eri in (([[1j]], np.zeros((1,)*4)), ([[1]], np.zeros((2,)*4)),
                       ([[np.nan]], np.zeros((1,)*4))):
            with self.assertRaises(ValueError): jordan_wigner(h, eri)
        with self.assertRaises(ValueError): jordan_wigner(np.eye(7), np.zeros((7,)*4))
        with self.assertRaises(ValueError): molecular_hamiltonian(['H'], [[0, 0, 0]], spin=1)
        with self.assertRaises(ValueError): molecular_hamiltonian(['H'], [[0, np.nan, 0]])


if __name__ == '__main__': unittest.main()
