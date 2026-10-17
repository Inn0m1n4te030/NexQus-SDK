"""Circuit drawer behavior, semantic labels, and safe SVG output."""
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np
from nexqus import Circuit, Parameter, Simulator, Noise, algorithms


class DrawingTests(unittest.TestCase):
    def test_eight_qubit_custom_circuit(self):
        q = Circuit(8).rx(.5, 0).cx(0, 1).ry(.8, 3).cx(3, 7)
        rows = q.draw().splitlines()
        self.assertEqual(len(rows), 8)
        self.assertIn('[RX(0.5)]', rows[0])
        self.assertIn('[RY(0.8)]', rows[3])
        self.assertIn('⊕', rows[7])
        root = ET.fromstring(q.draw('svg'))
        labels = [node.text for node in root.findall('.//{*}text')]
        for i in range(8): self.assertIn(f'q{i}', labels)

    def test_bell_symbols_and_notebook(self):
        q = algorithms.bell()
        text = q.draw()
        self.assertIn('[H]', text)
        self.assertIn('●', text)
        self.assertIn('⊕', text)
        self.assertIn('q0', text)
        svg = q.draw('svg')
        self.assertEqual(svg._repr_svg_(), q._repr_svg_())
        root = ET.fromstring(svg)
        self.assertEqual(root.tag, '{http://www.w3.org/2000/svg}svg')
        self.assertEqual(len(root.findall('.//{*}circle')), 2)

    def test_reversed_nonadjacent_operands(self):
        q = Circuit(4).unitary(np.eye(4), (3, 0), 'custom')
        rows = q.draw().splitlines()
        self.assertIn('[custom:0]', rows[3])
        self.assertIn('[custom:1]', rows[0])
        self.assertIn('│', rows[1])
        self.assertIn('│', rows[2])

    def test_custom_builtin_name_is_not_a_control(self):
        q = Circuit(2).unitary(np.eye(4), (0, 1), 'CX')
        self.assertNotIn('●', q.draw())
        self.assertIn('[CX:0]', q.draw())
        self.assertEqual(len(ET.fromstring(q.draw('svg')).findall('.//{*}circle')), 0)

    def test_dynamic_conditions_noise_and_parameters(self):
        q = Circuit(2).rx(Parameter('theta') * 2, 0).measure(0, 'flag')
        q.x(1).c_if('flag', 0).reset(0).noise(Noise.bit_flip(.1), 1)
        text = q.draw()
        for label in ['RX((theta*2))', 'c:flag', '=0', '|0>', 'noise:', '↓']:
            self.assertIn(label, text)
        ET.fromstring(q.draw('svg'))

    def test_svg_escapes_labels_and_keys(self):
        q = Circuit(1).unitary(np.eye(2), 0, '<script>&"').measure(0, '<key>\n')
        svg = q.draw('svg')
        root = ET.fromstring(svg)
        self.assertEqual(root.findall('.//{*}script'), [])
        self.assertIn('&lt;script&gt;', svg)
        self.assertIn('&lt;key&gt;', svg)
        self.assertEqual(len(q.draw().splitlines()), 2)

    def test_wrap_and_empty_circuit(self):
        q = Circuit(2)
        self.assertIn('q1', q.draw())
        ET.fromstring(q.draw('svg'))
        for _ in range(5): q.h(0)
        self.assertIn('Panel 3/3', q.draw(fold=2))
        self.assertIn('panel 3/3', q.draw('svg', fold=2))
        self.assertNotIn('Panel', q.draw(fold=None))

    def test_files_and_validation(self):
        q = algorithms.bell()
        with tempfile.TemporaryDirectory() as directory:
            for kind in ('svg', 'text'):
                path = Path(directory) / ('circuit.' + kind)
                self.assertEqual(q.draw(kind, filename=path), path.read_text(encoding='utf-8'))
        for kind in ['html', None]:
            with self.assertRaises(ValueError): q.draw(kind)
        for fold in [True, 0, -1, 1.5]:
            with self.assertRaises(ValueError): q.draw(fold=fold)

    def test_drawing_preserves_simulation(self):
        q = algorithms.teleportation()
        before = Simulator().run(q, shots=0).probabilities.copy()
        operations = list(q.operations)
        q.draw(); q.draw('svg')
        self.assertEqual(q.operations, operations)
        np.testing.assert_array_equal(Simulator().run(q, shots=0).probabilities, before)

    def test_swap_cz_and_inverse(self):
        q = Circuit(2).swap(1, 0).cz(0, 1).cx(1, 0).inverse()
        text = q.draw()
        self.assertEqual(text.count('×'), 2)
        self.assertEqual(text.count('●'), 3)
        ET.fromstring(q.draw('svg'))


if __name__ == '__main__':
    unittest.main()
