"""v0.2 regression, parameter binding, batching and prepared execution tests."""
import unittest
import numpy as np
from nexqus import Circuit, Parameter, Simulator, Noise, NoiseModel, circuit, algorithms


class ParameterTests(unittest.TestCase):
    def test_bind_arithmetic(self):
        theta,phi=Parameter('theta'),Parameter('phi')
        q=Circuit(2).ry(2*theta-phi/2,0).rz(1/(phi+2),1).cp(-theta,0,1).phase(theta+phi,1)
        self.assertEqual(q.parameters,('phi','theta'))
        bound=q.bind({theta:.4,phi:.2})
        direct=Circuit(2).ry(.7,0).rz(1/2.2,1).cp(-.4,0,1).phase(.6,1)
        a=Simulator().run(bound,shots=0).statevector
        b=Simulator().run(direct,shots=0).statevector
        np.testing.assert_allclose(a,b,atol=1e-12)
        self.assertEqual(bound.parameters,())
        self.assertEqual(q.parameters,('phi','theta'))

    def test_symbolic_inverse_and_compose(self):
        t=Parameter('t')
        block=Circuit(2).rx(t,0).cp(2*t,1,0).phase(t,1)
        q=Circuit(3).h(0).compose(block,(2,0)).compose(block.inverse(),(2,0))
        np.testing.assert_allclose(Simulator().run(q.bind({'t':.9}),shots=0).statevector,
                                   Simulator().run(Circuit(3).h(0),shots=0).statevector,atol=1e-12)

    def test_decorated_template(self):
        @circuit(1)
        def fn(q,t): q.ry(t,0)
        q=fn(Parameter('angle'))
        self.assertAlmostEqual(Simulator().run(q.bind({'angle':np.pi}),shots=0).probabilities[1],1)

    def test_binding_errors(self):
        t=Parameter('t'); q=Circuit(1).ry(t,0)
        for values in ({},{'t':1,'typo':1},{'t':float('nan')},{'t':1j},{'t':True},{t:1,'t':2}):
            with self.assertRaises(ValueError): q.bind(values)
        with self.assertRaises(ValueError): Parameter('')
        with self.assertRaises(ValueError): Circuit(1).rx(float('inf'),0)
        with self.assertRaises(ValueError): Simulator().run(q)
        with self.assertRaises(ValueError): Simulator().compile(q)

    def test_sweep_order_seeds_and_retention(self):
        t=Parameter('t'); q=Circuit(1).ry(t,0)
        bindings=[{'t':0},{'t':np.pi/2},{'t':np.pi}]
        sim=Simulator()
        a=sim.run_sweep(q,bindings,seed=12,shots=100)
        b=sim.run_sweep(q,iter(bindings),seed=12,shots=100)
        self.assertEqual([r.counts for r in a],[r.counts for r in b])
        self.assertEqual([r.metadata['sweep_index'] for r in a],[0,1,2])
        self.assertEqual(len(set(r.metadata['seed'] for r in a)),3)
        self.assertTrue(all(r.statevector is None for r in a))
        for result,values in zip(a,bindings):
            expected=sim.run(q.bind(values),shots=0)
            np.testing.assert_allclose(result.probabilities,expected.probabilities)
        self.assertEqual(sim.run_sweep(q,[],seed=1),[])
        self.assertIsNotNone(sim.run_sweep(q,[bindings[0]],retain_state=True)[0].statevector)

    def test_sweep_with_noise(self):
        t=Parameter('t'); q=Circuit(1).ry(t,0)
        noise=NoiseModel(Noise.amplitude_damping(.2))
        result=Simulator().run_sweep(q,[{'t':np.pi}],noise=noise,shots=0)[0]
        np.testing.assert_allclose(result.probabilities,[.2,.8],atol=1e-12)


class CompilationTests(unittest.TestCase):
    def test_compiled_all_methods(self):
        noise=NoiseModel(Noise.depolarizing(.13),Noise.amplitude_damping(.07))
        for method in ('statevector','density_matrix','trajectory'):
            sim=Simulator(method=method)
            q=algorithms.ghz(3).cp(.5,2,0)
            model=None if method=='statevector' else noise
            compiled=sim.compile(q,noise=model)
            kw=dict(shots=100,seed=1,trajectories=77,batch_size=8)
            a=sim.run(q,noise=model,**kw); b=compiled.run(**kw)
            np.testing.assert_allclose(a.probabilities,b.probabilities,atol=1e-12)
            self.assertEqual(a.counts,b.counts)
            self.assertTrue(b.metadata['compiled'])

    def test_snapshot_of_source_and_noise(self):
        sim=Simulator(); q=Circuit(1).x(0)
        noise=NoiseModel(Noise.bit_flip(.25))
        compiled=sim.compile(q,noise)
        q.x(0); noise.after(None,Noise.bit_flip(1))
        np.testing.assert_allclose(compiled.run(shots=0).probabilities,[.25,.75])

    def test_snapshot_matrix_mutation(self):
        sim=Simulator(); q=Circuit(1).x(0)
        compiled=sim.compile(q)
        q.operations[0].data.flags.writeable=True
        q.operations[0].data[:]=np.eye(2)
        np.testing.assert_allclose(compiled.run(shots=0).probabilities,[0,1])

    def test_owner_and_frozen_noise_errors(self):
        sim=Simulator(); compiled=sim.compile(Circuit(1))
        with self.assertRaises(ValueError): Simulator().run(compiled)
        with self.assertRaises(ValueError): compiled.run(noise=NoiseModel())

    def test_cache_hits_clear_and_bounds(self):
        sim=Simulator(gate_cache_mb=0.0001)
        q=Circuit(1).h(0)
        sim.run(q); sim.run(q)
        self.assertGreater(sim.cache_info()['hits'],0)
        for i in range(10): sim.run(Circuit(1).ry(i/10,0))
        self.assertLessEqual(sim.cache_info()['bytes'],sim.cache_info()['limit_bytes'])
        compiled=sim.compile(Circuit(1).x(0)); sim.clear_cache()
        self.assertEqual(sim.cache_info()['entries'],0)
        self.assertEqual(compiled.run(shots=0).most_likely,'1')
        with self.assertRaises(ValueError): Simulator(gate_cache_mb=-1)

    def test_cache_disabled(self):
        sim=Simulator(gate_cache_mb=0)
        sim.run(algorithms.bell())
        self.assertEqual(sim.cache_info()['entries'],0)


class BatchTests(unittest.TestCase):
    def test_noiseless_batch_wire_order(self):
        q=Circuit(3).h(2).ry(.7,0).cx(2,0).cp(.4,1,2).swap(0,1)
        expected=Simulator().run(q,shots=0)
        for batch in (1,2,7,20):
            result=Simulator(method='trajectory').run(q,trajectories=17,batch_size=batch,shots=0)
            np.testing.assert_allclose(result.probabilities,expected.probabilities,atol=1e-12)

    def test_batched_against_exact(self):
        q=algorithms.bell().ry(.3,0).noise(Noise.phase_damping(.3),1)
        model=NoiseModel(Noise.amplitude_damping(.2),Noise.depolarizing(.1))
        exact=Simulator().run(q,noise=model,shots=0)
        result=Simulator(method='trajectory').run(q,noise=model,trajectories=4000,batch_size=64,seed=9,shots=0)
        np.testing.assert_allclose(result.probabilities,exact.probabilities,atol=.025)

    def test_batch_determinism_and_partial_batch(self):
        q=algorithms.ghz(3)
        model=NoiseModel(Noise.depolarizing(.1))
        sim=Simulator(method='trajectory')
        kwargs=dict(noise=model,trajectories=19,batch_size=8,seed=18,shots=50)
        a=sim.run(q,**kwargs); b=sim.run(q,**kwargs)
        np.testing.assert_array_equal(a.probabilities,b.probabilities)
        self.assertEqual(a.counts,b.counts)
        self.assertEqual(a.metadata['trajectories'],19)
        self.assertEqual(a.metadata['batch_size'],8)

    def test_batch_noise_endpoints(self):
        for noise in (Noise.amplitude_damping(1),Noise.bit_flip(1)):
            q=Circuit(1).x(0).noise(noise,0)
            r=Simulator(method='trajectory').run(q,trajectories=13,batch_size=5,shots=0,seed=7)
            np.testing.assert_allclose(r.probabilities,[1,0])

    def test_custom_three_kraus(self):
        channel=Noise((np.sqrt(.5)*np.eye(2),np.sqrt(.25)*np.diag([1,-1]),
                       np.sqrt(.25)*np.array([[0,1],[1,0]])))
        q=Circuit(2).h(0).cx(0,1).noise(channel,0)
        exact=Simulator().run(q,shots=0)
        sampled=Simulator(method='trajectory').run(q,trajectories=3000,batch_size=37,seed=21,shots=0)
        np.testing.assert_allclose(sampled.probabilities,exact.probabilities,atol=.025)

    def test_memory_and_input_guards(self):
        sim=Simulator(method='trajectory',max_memory_mb=1)
        with self.assertRaises(MemoryError): sim.run(Circuit(12),batch_size=16)
        for batch in (0,-1,True,1.5):
            with self.assertRaises(ValueError): sim.run(Circuit(1),batch_size=batch)
        with self.assertRaises(ValueError): sim.run(Circuit(1),trajectories=0)

    def test_single_precision_batch(self):
        r=Simulator(method='trajectory',precision='single').run(algorithms.ghz(5),
              trajectories=23,batch_size=8,shots=500,noise=NoiseModel(Noise.depolarizing(.1)),seed=1)
        self.assertAlmostEqual(r.probabilities.sum(),1)


try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'Optional PyTorch not installed')
class TorchV02Tests(unittest.TestCase):
    def compare(self,device):
        q=algorithms.ghz(3).cp(.9,2,0)
        noise=NoiseModel(Noise.amplitude_damping(.17))
        for precision,atol in [('single',2e-6),('double',1e-10)]:
            a=Simulator(method='trajectory',precision=precision)
            b=Simulator(backend='torch',device=device,method='trajectory',precision=precision)
            kwargs=dict(trajectories=19,batch_size=7,seed=18,shots=0)
            expected=a.run(q,noise=noise,**kwargs)
            actual=b.compile(q,noise).run(**kwargs)
            np.testing.assert_allclose(expected.probabilities,actual.probabilities,atol=atol)
        t=Parameter('t'); template=Circuit(2).ry(t,0).cx(0,1)
        sim=Simulator(backend='torch',device=device)
        results=sim.run_sweep(template,[{'t':.2},{'t':.8}],seed=9,shots=0)
        for angle,result in zip([.2,.8],results):
            expected=Simulator().run(template.bind({'t':angle}),shots=0)
            np.testing.assert_allclose(result.probabilities,expected.probabilities,atol=1e-10)

    def test_torch_cpu_v02(self): self.compare('cpu')

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(),'CUDA GPU unavailable')
    def test_torch_cuda_v02(self): self.compare('cuda:0')


if __name__=='__main__': unittest.main()
