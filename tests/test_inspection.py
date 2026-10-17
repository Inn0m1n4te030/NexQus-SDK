"""Metadata and dependency-aware circuit inspection regressions."""
import builtins
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from nexqus import Circuit, Noise, Parameter, Simulator


class InspectionTests(unittest.TestCase):
    def test_numeric_angles_and_precision(self):
        q = Circuit(2).rx(.5, 0).ry(-.25, 1).rz(.123456789, 0).phase(1.2, 0).cp(.7, 1, 0)
        for text in ('RX(0.5)', 'RY(-0.25)', 'RZ(0.1235)', 'P(1.2)', 'CP(0.7)'):
            self.assertIn(text, q.draw())
            self.assertIn(text, q.draw('svg'))
        self.assertIn('RZ(0.12345679)', q.draw(precision=8))
        self.assertIn('[RX]', Circuit(1).unitary(np.eye(2), 0, 'RX').draw())
        for precision in (0, 17, True, 1.5):
            with self.assertRaises(ValueError): q.draw(precision=precision)

    def test_metadata_binding_composition_condition_and_inverse(self):
        theta = Parameter('theta')
        block = Circuit(2).rx(theta, 0).cp(.5, 1, 0)
        q = Circuit(3).compose(block, (2, 0)).bind({'theta': .7})
        self.assertEqual(q.operations[0].params, (.7,))
        self.assertIn('RX(0.7)', q.draw())
        q.measure(2, 'm').ry(.2, 1).c_if('m')
        self.assertEqual(q.operations[-1].params, (.2,))
        inverse = block.inverse().bind({'theta': .7})
        self.assertIn('RX†(0.7)', inverse.draw())
        direct = block.bind({'theta': .7})
        identity = Circuit(2).compose(direct).compose(inverse)
        np.testing.assert_allclose(Simulator().run(identity, shots=0).statevector, [1,0,0,0], atol=1e-14)
        self.assertEqual(Simulator().compile(direct)._circuit.operations[0].params, (.7,))

    def test_counts_and_parallel_depth(self):
        q = Circuit(8)
        self.assertEqual(q.depth(), 0)
        self.assertEqual(q.count_ops(), {})
        q.rx(.5, 0).h(7).cx(0, 1).x(2).cz(2, 7)
        self.assertEqual(q.depth(), 2)
        self.assertEqual(q.count_ops(), {'RX':1, 'H':1, 'CX':1, 'X':1, 'CZ':1})
        q.noise(Noise.bit_flip(.1), 1).measure(1, 'm').reset(1)
        self.assertEqual(q.depth(), 5)
        self.assertNotIn('MEASURE', q.count_ops(gates_only=True))
        self.assertEqual(sum(q.count_ops(gates_only=True).values()), 5)
        with self.assertRaises(ValueError): q.count_ops(gates_only=1)

    def test_classical_read_write_dependencies(self):
        q = Circuit(4).measure(0, 'm').x(1).c_if('m').x(2).c_if('m')
        # Both readers can occupy layer 2, despite sharing a classical input.
        self.assertEqual(q.depth(), 2)
        q.measure(3, 'm')  # overwrite must wait for both earlier readers
        self.assertEqual(q.depth(), 3)
        q.x(0).c_if('m')
        self.assertEqual(q.depth(), 4)

    def test_export_dependency_error_is_actionable(self):
        original = builtins.__import__
        def without_cairo(name, *args, **kwargs):
            if name == 'cairosvg': raise ImportError('not installed')
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=without_cairo):
            for kind in ('png', 'pdf'):
                with self.assertRaisesRegex(ImportError, r'nexqus-sdk\[visualization\]'):
                    Circuit(1).rx(.5, 0).draw(kind)

    def test_optional_binary_exports(self):
        # Core-only installs are covered by the missing-dependency regression;
        # CI's inspection job installs visualization and exercises real exports.
        if importlib.util.find_spec('cairosvg') is None: return
        q = Circuit(8).rx(.5, 0).cx(0, 7).measure(0, 'm').x(3).c_if('m')
        with tempfile.TemporaryDirectory() as directory:
            for kind, signature in [('png', b'\x89PNG\r\n\x1a\n'), ('pdf', b'%PDF-')]:
                path = Path(directory) / ('circuit.'+kind)
                result = q.draw(kind, filename=path, fold=2)
                self.assertTrue(result.startswith(signature))
                self.assertEqual(result, path.read_bytes())
                if kind == 'png': self.assertEqual(result._repr_png_(), bytes(result))
