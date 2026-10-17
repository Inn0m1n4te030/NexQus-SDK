import tempfile
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
import numpy as np
from nexqus import Molecule, MolecularView, molecule, molecular_gallery, Simulator
from nexqus.chemistry import h2_reference_problem


class MoleculeTests(unittest.TestCase):
    def test_methane_and_water_geometry(self):
        m = molecule('CH4')
        self.assertEqual((m.formula, m.n_electrons, len(m.symbols), len(m.bonds)), ('CH4', 10, 5, 4))
        for i in range(1, 5):
            self.assertAlmostEqual(m.distance(0, i), 1.09)
            for j in range(1, i): self.assertAlmostEqual(m.angle(i, 0, j), 109.47122063449069)
        self.assertEqual(m.to_dict()['sto3g_qubits_estimate'], 18)
        water = molecule('H2O')
        self.assertAlmostEqual(water.distance(0, 1), .958)
        self.assertAlmostEqual(water.angle(1, 0, 2), 104.5)

    def test_organic_connectivity_and_basis_sizes(self):
        ethene = molecule('C2H4'); benzene = molecule('C6H6')
        self.assertEqual(sum(o == 2 for _, _, o in ethene.bonds), 1)
        self.assertEqual((benzene.formula, benzene.n_electrons, len(benzene.bonds)), ('C6H6', 42, 12))
        self.assertEqual(benzene.to_dict()['sto3g_qubits_estimate'], 72)
        for a, b, order in benzene.bonds:
            self.assertAlmostEqual(benzene.distance(a, b), 1.397 if order == 1.5 else 1.09)
        for m in molecular_gallery().to_dict()['molecules']:
            self.assertNotIn('energies_hartree', m)

    def test_snapshot_inference_and_charge(self):
        xyz = np.array([[0, 0, 0], [0, 0, .735]])
        m = Molecule(['H', 'H'], xyz, charge=1)
        xyz[1, 2] = 99
        self.assertAlmostEqual(m.distance(0, 1), .735)
        self.assertEqual(m.n_electrons, 1)
        self.assertEqual(len(m.bonds), 1)
        self.assertIn('heuristic', m.bond_source)
        self.assertEqual(Molecule(['H', 'H'], [[0,0,0],[0,0,.735]], []).bonds, ())
        report = m.view(); data = report.to_dict(); data['molecules'][0]['atoms'][0]['position'][0] = 99
        self.assertEqual(report.to_dict()['molecules'][0]['atoms'][0]['position'][0], 0)

    def test_invalid_inputs_and_limits(self):
        for symbols, xyz, bonds in [([], [], []), (['Xx'], [[0,0,0]], []),
            (['H'], [[np.nan,0,0]], []), (['H'], [[1j,0,0]], []),
            (['H','H'], [[0,0,0],[0,0,0]], []),
            (['H','H'], [[0,0,0],[0,0,1]], [(0,2,1)]),
            (['H','H'], [[0,0,0],[0,0,1]], [(0,1,1),(1,0,1)]),
            (['H','H'], [[0,0,0],[0,0,1]], [(0,1,4)])]:
            with self.assertRaises(ValueError): Molecule(symbols, xyz, bonds)
        with self.assertRaises(ValueError): Molecule(['H']*257, np.zeros((257,3)))
        with self.assertRaises(ValueError): molecule('CH4').angle(0, 0, 1)
        with self.assertRaises(ValueError): molecule('H2').distance(-1, 0)
        with self.assertRaises(ValueError): MolecularView([])
        with self.assertRaises(ValueError): molecule('H2').view(energies_hartree={'VQE': -1.})
        with self.assertRaises(ValueError): molecule('H2').view(energies_hartree={'VQE': float('nan')}, calculation_note='test')

    def test_exports_and_injection(self):
        m = Molecule(['H'], [[0,0,0]], [], name='</script><img src=x onerror=alert(1)>')
        report = m.view().to_html()
        self.assertNotIn('</script><img', report)
        self.assertIn(r'\u003c/script', report)
        self.assertIn('sandbox="allow-scripts allow-downloads"', report._repr_html_())
        svg = m.draw(); ET.fromstring(svg)
        self.assertNotIn('<img', svg)
        with tempfile.TemporaryDirectory() as path:
            path = Path(path)
            self.assertEqual(m.to_xyz(path/'test.xyz'), (path/'test.xyz').read_text(encoding='utf-8'))
            self.assertEqual(m.draw(path/'test.svg'), (path/'test.svg').read_text(encoding='utf-8'))
            self.assertEqual(m.view().to_html(path/'test.html'), (path/'test.html').read_text(encoding='utf-8'))

    def test_h2_reference_geometry_and_energy(self):
        problem = h2_reference_problem(); m = problem.molecule()
        self.assertEqual((m.formula, m.n_electrons), ('H2', 2))
        self.assertAlmostEqual(m.distance(0, 1), .735)
        self.assertAlmostEqual(problem.exact_energy(), -1.1373060357534004)
        state = Simulator().run(problem.hartree_fock_circuit(), shots=0, retain_state=True)
        self.assertAlmostEqual(problem.hamiltonian.expectation(state), problem.hartree_fock_energy)
        view = m.view(energies_hartree={'Classical HF': problem.hartree_fock_energy},
                      calculation_note='H2 / STO-3G / fixed 0.735 Angstrom')
        self.assertEqual(view.to_dict()['molecules'][0]['energies_hartree']['Classical HF'], problem.hartree_fock_energy)


if __name__ == '__main__': unittest.main()
