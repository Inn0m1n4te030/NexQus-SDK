"""Template equivalence, input validation and repeated-data gradients."""
import unittest
import numpy as np
from nexqus import Circuit, Parameter, QuantumFunction, Simulator, templates as t


class TemplateTests(unittest.TestCase):
    def assert_same_state(self, a, b):
        sim = Simulator()
        np.testing.assert_allclose(sim.run(a, shots=0).statevector, sim.run(b, shots=0).statevector, atol=1e-13)

    def test_basis_and_angle_encoding_order(self):
        q = Circuit(4)
        self.assertIs(t.basis_encoding(q, [1,0,1], wires=[3,1,0]), q)
        self.assert_same_state(q, Circuit(4).x(3).x(0))
        for rotation in 'XYZ':
            q = t.angle_encoding(Circuit(4), [.2,-.4], [3,0,2], rotation=rotation, scale=2)
            expected = Circuit(4)
            getattr(expected, 'r'+rotation.lower())(.4, 3)
            getattr(expected, 'r'+rotation.lower())(-.8, 0)
            self.assert_same_state(q, expected)

    def test_connectivity_and_two_wire_ring(self):
        for pattern, pairs in [('linear',[(3,0),(0,2)]), ('ring',[(3,0),(0,2),(2,3)]),
                               ('full',[(3,0),(3,2),(0,2)]), ([(2,3)],[(2,3)])]:
            for gate in ('CX','CZ'):
                q = t.entangling_layer(Circuit(4), [3,0,2], entanglement=pattern, entangler=gate)
                self.assertEqual([op.wires for op in q.operations], pairs)
                expected = Circuit(4)
                for a,b in pairs: getattr(expected, gate.lower())(a,b)
                self.assert_same_state(Circuit(4).h(3).compose(q), Circuit(4).h(3).compose(expected))
        self.assertEqual(len(t.entangling_layer(Circuit(2), entanglement='ring').operations), 1)
        self.assertEqual(t.entangling_layer(Circuit(1)).operations, [])

    def test_layers_equal_handwritten_circuit(self):
        weights = np.arange(12).reshape(t.layer_shape(2,2,rotations='XYZ')) / 10
        q = t.variational_layers(Circuit(3), weights, [2,0], rotations='XYZ', entangler='CZ')
        expected = Circuit(3)
        for layer in weights:
            for wire, angles in zip([2,0], layer):
                for axis, angle in zip('xyz', angles): getattr(expected, 'r'+axis)(angle, wire)
            expected.cz(2,0)
        self.assert_same_state(q, expected)
        self.assertEqual(q.depth(), 8)

    def test_reuploading_shared_parameters_and_gradient(self):
        x, w, b = Parameter('x'), Parameter('w'), Parameter('b')
        weights = np.array([[[w,b]], [[w*.5,b]]], dtype=object)
        q = t.data_reuploading(Circuit(1), [x], weights, rotations='YZ', encoding_rotation='X')
        expected = Circuit(1).rx(x,0).ry(w,0).rz(b,0).rx(x,0).ry(w*.5,0).rz(b,0)
        bindings = {'x':.3, 'w':.8, 'b':-.2}
        self.assert_same_state(q.bind(bindings), expected.bind(bindings))
        function = QuantumFunction(q, 'Z')
        _, gradient = function.value_and_grad(bindings)
        for name in bindings:
            plus, minus = dict(bindings), dict(bindings)
            plus[name] += 1e-6; minus[name] -= 1e-6
            numerical = (function(plus)-function(minus))/2e-6
            self.assertAlmostEqual(gradient[name], numerical, places=8)
        self.assertIn('RX(x)', q.draw())

    def test_invalid_inputs_do_not_partly_modify_circuit(self):
        q = Circuit(3).h(0)
        invalid = [lambda: t.angle_encoding(q, [1,float('nan')]),
                   lambda: t.angle_encoding(q, [[1,2]]),
                   lambda: t.angle_encoding(q, [1,2,3,4]),
                   lambda: t.angle_encoding(q, [1], rotation='XY'),
                   lambda: t.angle_encoding(q, [1], wires=[1,1]),
                   lambda: t.basis_encoding(q, [1,0,2]),
                   lambda: t.basis_encoding(q, [.0,1,0]),
                   lambda: t.entangling_layer(q, entanglement=[(0,1),(1,8)]),
                   lambda: t.entangling_layer(q, entanglement=[(0,1),(0,1)]),
                   lambda: t.entangling_layer(q, entanglement='bad'),
                   lambda: t.variational_layers(q, np.zeros((2,3))),
                   lambda: t.data_reuploading(q, [], np.zeros((2,3,1))),
                   lambda: t.data_reuploading(q, [1,2], np.zeros((2,3,1)), entangler='SWAP')]
        for call in invalid:
            with self.subTest(call=call):
                with self.assertRaises((TypeError, ValueError)): call()
                self.assertEqual(q.count_ops(), {'H':1})
        with self.assertRaises(ValueError): t.layer_shape(0, 3)
