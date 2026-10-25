"""Independent overlap checks, parser contracts and train-only preprocessing."""
from types import SimpleNamespace
import unittest
import numpy as np
from nexqus import Circuit, Parameter, QuantumFunction, Simulator, nlp


class NLPTests(unittest.TestCase):
    def test_pooling_ignores_padding_and_rejects_invalid_inputs(self):
        np.testing.assert_allclose(nlp.pool_token_features([[1,2],[3,4],[99,99]],
            mask=[True,True,False]), [2,3])
        for values in ([], [[1j]], [[float('nan')]], [1,2], [[True]]):
            with self.assertRaises(ValueError): nlp.pool_token_features(values)
        for mask in ([False,False], [1,0], [True]):
            with self.assertRaises(ValueError): nlp.pool_token_features([[1],[2]],mask=mask)

    def test_scaler_training_range_constants_and_heldout_clipping(self):
        scaler=nlp.FeatureAngleScaler()
        with self.assertRaises(ValueError): scaler.transform([[1]])
        train=np.array([[2.,7.],[4.,7.]])
        np.testing.assert_allclose(scaler.fit_transform(train),[[0,0],[np.pi,0]])
        train[:]=99
        np.testing.assert_allclose(scaler.transform([[3.,99.],[100.,0.],[-9.,7.]]),
                                   [[np.pi/2,0],[np.pi,0],[0,0]])
        with self.assertRaises(ValueError): scaler.transform([[1]])

    def test_pair_matches_independent_complex_statevectors_and_symmetry(self):
        a=Circuit(3).h(0).ry(.7,1).cx(1,2).rz(.4,0)
        b=Circuit(3).rx(.3,0).h(2).cx(2,1).phase(.8,2)
        sa=Simulator().run(a,shots=0).statevector
        sb=Simulator().run(b,shots=0).statevector
        expected=abs(np.vdot(sb,sa))**2
        self.assertAlmostEqual(nlp.pair_fidelity(a,b),expected)
        self.assertAlmostEqual(nlp.pair_fidelity(b,a),expected)
        self.assertAlmostEqual(nlp.pair_fidelity(a,a),1)
        self.assertEqual((len(a.operations),len(b.operations)),(4,4))

    def test_pair_analytic_orthogonal_and_symbolic_bindings(self):
        self.assertAlmostEqual(nlp.pair_fidelity(Circuit(1),Circuit(1).x(0)),0)
        a=nlp.sentence_encoder([Parameter('a')])
        b=nlp.sentence_encoder([Parameter('b')])
        self.assertAlmostEqual(nlp.pair_fidelity(a,b,bindings={'a':.2,'b':.8}),np.cos(.3)**2)
        pair=nlp.sentence_pair_circuit(a,b)
        self.assertEqual(set(pair.parameters),{'a','b'})
        with self.assertRaises(ValueError): nlp.pair_fidelity(a,b)

    def test_pair_rejects_width_noise_and_dynamics(self):
        from nexqus import Noise
        for right in (Circuit(2),Circuit(1).measure(0,'m'),Circuit(1).noise(Noise.bit_flip(.1),0)):
            with self.assertRaises(ValueError): nlp.sentence_pair_circuit(Circuit(1),right)
        with self.assertRaises(TypeError): nlp.sentence_pair_circuit([],Circuit(1))

    def test_encoder_reuploading_and_gradients(self):
        theta=Parameter('theta')
        weights=np.array([[[theta]],[[.3]]],dtype=object)
        a=nlp.sentence_encoder([.2],weights=weights)
        b=nlp.sentence_encoder([.8],weights=weights)
        pair=nlp.sentence_pair_circuit(a,b)
        # P(0)=(1+<Z>)/2 for this one-qubit overlap circuit.
        f=QuantumFunction(pair,'Z',Simulator())
        value,grad=f.value_and_grad({'theta':.4})
        h=1e-5
        self.assertAlmostEqual(grad['theta'],(f({'theta':.4+h})-f({'theta':.4-h}))/(2*h),places=7)
        self.assertAlmostEqual((1+value)/2,nlp.pair_fidelity(a,b,bindings={'theta':.4}))
        self.assertEqual(len(a.operations),4)
        two_weights=np.array([[[theta],[.3]],[[.4],[.1]]],dtype=object)
        two_a=nlp.sentence_encoder([.2,.6],weights=two_weights)
        two_b=nlp.sentence_encoder([.8,.1],weights=two_weights)
        self.assertNotAlmostEqual(nlp.pair_fidelity(two_a,two_b,bindings={'theta':.4}),
                                  nlp.pair_fidelity(two_a,two_b,bindings={'theta':.8}),places=5)
        for features in ([],[[.2]],[float('inf')]):
            with self.assertRaises(ValueError): nlp.sentence_encoder(features)
        with self.assertRaises(ValueError): nlp.sentence_encoder([.1,.2],max_qubits=1)

    def test_dependency_edges_follow_heads_and_annotations(self):
        parse=nlp.DependencyParse(['Alice','likes','Bob'],[1,1,1])
        circuit=parse.circuit([.2,.3,.4])
        self.assertEqual([op.wires for op in circuit.operations[-2:]],[(1,0),(1,2)])
        reference=Circuit(3).ry(.2,0).ry(.3,1).ry(.4,2).cx(1,0).cx(1,2)
        np.testing.assert_allclose(Simulator().run(circuit,shots=0).statevector,
                                   Simulator().run(reference,shots=0).statevector)
        self.assertEqual(parse.annotations()[2],dict(wire=2,token='Bob',head=1))
        self.assertEqual(parse.heads,(1,1,1))
        with self.assertRaises(ValueError): parse.circuit([.2,.3])
        with self.assertRaises(ValueError): parse.circuit([.2,.3,.4],max_qubits=2)

    def test_dependency_validation_rejects_cycles_and_multiple_roots(self):
        for tokens,heads in [([],[]),(['a','b'],[0,1]),(['a','b','c'],[0,2,1]),
                            (['a'],[1]),(['a'],[True]),([''],[0]),(['a'],[.0])]:
            with self.assertRaises(ValueError): nlp.DependencyParse(tokens,heads)

    def test_spacy_adapter_remaps_span_offsets_and_rejects_unparsed(self):
        tokens=[SimpleNamespace(i=i,text=t,dep_='dep',head=SimpleNamespace(i=h))
                for i,t,h in [(5,'Alice',6),(6,'likes',6),(7,'Bob',6)]]
        parse=nlp.DependencyParse.from_spacy(tokens)
        self.assertEqual(parse.heads,(1,1,1))
        tokens[0].head.i=4
        with self.assertRaises(ValueError): nlp.DependencyParse.from_spacy(tokens)
        tokens[0].head.i=6;tokens[0].dep_=''
        with self.assertRaises(ValueError): nlp.DependencyParse.from_spacy(tokens)


if __name__=='__main__': unittest.main()
