"""Optional chemistry integration suite; CI installs the chemistry extra."""
import unittest
import numpy as np
from pyscf import gto, scf, fci
from nexqus import Simulator, Parameter, Adam
from nexqus.chemistry import molecular_hamiltonian, h2_pair_ansatz


class PySCFTests(unittest.TestCase):
    def test_h2_curve_against_fci_and_hf(self):
        for distance in (.5, .735, 1.5):
            with self.subTest(distance=distance):
                xyz = [[0, 0, 0], [0, 0, distance]]
                problem = molecular_hamiltonian(['H', 'H'], xyz)
                mol = gto.M(atom=list(zip(['H', 'H'], xyz)), basis='sto-3g', unit='Angstrom', verbose=0)
                hf = scf.RHF(mol).run(conv_tol=1e-11)
                reference = fci.FCI(hf).kernel()[0]
                self.assertAlmostEqual(problem.exact_energy(), reference, places=8)
                r = Simulator().run(problem.hartree_fock_circuit(), shots=0, retain_state=True)
                self.assertAlmostEqual(problem.hamiltonian.expectation(r), hf.e_tot, places=8)
                f = problem.hamiltonian.function(h2_pair_ansatz(Parameter('theta')))
                params = {'theta': 0.}; opt = Adam(lr=.04)
                for _ in range(180):
                    _, g = f.value_and_grad(params); params = opt.step(params, g)
                self.assertLess(abs(f(params)-reference), 2e-7)

    def test_preflight(self):
        with self.assertRaisesRegex(ValueError, 'max_qubits'):
            molecular_hamiltonian(['O', 'H', 'H'], [[0,0,0], [0,0,1], [0,1,0]], max_qubits=4)

    def test_helium_closed_shell(self):
        problem = molecular_hamiltonian(['He'], [[0, 0, 0]])
        mol = gto.M(atom='He 0 0 0', basis='sto-3g', verbose=0)
        hf = scf.RHF(mol).run()
        reference = fci.FCI(hf).kernel()[0]
        self.assertEqual(problem.hamiltonian.n_qubits, 2)
        self.assertAlmostEqual(problem.exact_energy(), reference, places=9)


if __name__ == '__main__': unittest.main()
