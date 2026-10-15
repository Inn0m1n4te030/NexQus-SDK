"""Independent kernel, device sampling, strict validation and RNG regressions."""
import copy
import unittest
import numpy as np
from nexqus import Circuit, Simulator, Noise, NoiseModel, algorithms
from nexqus.backend import Backend
from nexqus.random import DeviceRandom
from nexqus.validation import compare_results,trajectory_confidence_bound


class KernelTests(unittest.TestCase):
    def test_local_one_all_axes_against_matrix_kernel(self):
        rng=np.random.default_rng(1)
        for batch in (None,5):
            shape=(2,)*4 if batch is None else (batch,)+(2,)*4
            state=rng.normal(size=shape)+1j*rng.normal(size=shape)
            matrix=rng.normal(size=(2,2))+1j*rng.normal(size=(2,2))
            fast=Backend(); slow=Backend(local_gates=False)
            for q in range(4):
                fn='apply' if batch is None else 'apply_batch'
                a=getattr(fast,fn)(state,matrix,(q,))
                b=getattr(slow,fn)(state,matrix,(q,))
                np.testing.assert_allclose(a,b,atol=1e-12)

    def test_diagonal_unsorted_axes_and_batches(self):
        rng=np.random.default_rng(5)
        for batch in (None,3):
            shape=(2,)*4 if batch is None else (batch,)+(2,)*4
            state=rng.normal(size=shape)+1j*rng.normal(size=shape)
            for wires in ((3,0),(2,0,1),(0,),(0,1,2,3)):
                d=np.exp(1j*rng.random(2**len(wires)))
                fn='apply' if batch is None else 'apply_batch'
                a=getattr(Backend(),fn)(state,d,wires,True)
                b=getattr(Backend(local_gates=False),fn)(state,d,wires,True)
                np.testing.assert_allclose(a,b,atol=1e-12)

    def test_density_complex_noise_local_vs_general(self):
        q=algorithms.ghz(4).rx(.3,2).cp(.7,3,0)
        noise=NoiseModel(Noise.amplitude_damping(.17),Noise.phase_damping(.25))
        a=Simulator(local_gates=False).run(q,noise=noise,shots=0)
        b=Simulator().run(q,noise=noise,shots=0)
        compare_results(a,b)


class RandomTests(unittest.TestCase):
    def test_counter_stream_independent_of_chunking(self):
        a=DeviceRandom(Backend(),42)
        full=a.uniform(100)
        b=DeviceRandom(Backend(),42)
        parts=np.concatenate([b.uniform(7),b.uniform(31),b.uniform(62)])
        np.testing.assert_array_equal(full,parts)

    def test_integer_hash_independent_scalar_reference(self):
        r=DeviceRandom(Backend(),123)
        expected=[]
        for i in range(20):
            x=(i+r.key)&0xffffffff
            x=((x^(x>>16))*0x45d9f3b)&0xffffffff
            x=((x^(x>>16))*0x45d9f3b)&0xffffffff
            x=x^(x>>16)
            expected.append((x+.5)/4294967296.)
        np.testing.assert_array_equal(r.uniform(20),expected)

    def test_distribution_sanity(self):
        x=DeviceRandom(Backend(),99).uniform(100000)
        self.assertTrue(np.all((x>0)&(x<1)))
        self.assertLess(abs(x.mean()-.5),.005)
        self.assertLess(abs(x.var()-1/12),.003)
        self.assertLess(abs(np.corrcoef(x[:-1],x[1:])[0,1]),.02)

    def test_engine_and_limit_errors(self):
        with self.assertRaises(ValueError): DeviceRandom(Backend(),engine='typo')
        r=DeviceRandom(Backend()); r.position=2**32
        with self.assertRaises(ValueError): r.uniform(1)

    def test_native_seed_reproduction(self):
        np.testing.assert_array_equal(DeviceRandom(Backend(),5,'native').uniform(10),
                                      DeviceRandom(Backend(),5,'native').uniform(10))


class NoiseFastpathTests(unittest.TestCase):
    def test_pauli_channels_paired_with_generic(self):
        q=algorithms.ghz(3).ry(.3,0).h(2)
        for ch in (Noise.bit_flip(.2),Noise.phase_flip(.3),Noise.depolarizing(.4)):
            model=NoiseModel(ch)
            sim=Simulator(method='trajectory')
            kwargs=dict(noise=model,trajectories=151,batch_size=17,seed=11,shots=0)
            a=sim.run(q,noise_fastpath=False,**kwargs)
            b=sim.run(q,noise_fastpath=True,**kwargs)
            np.testing.assert_allclose(a.probabilities,b.probabilities,atol=1e-12)

    def test_damping_channels_paired_with_generic(self):
        q=Circuit(3).ry(.7,1).h(2).cx(2,0).rx(.9,0)
        for ch in (Noise.amplitude_damping(.3),Noise.phase_damping(.4)):
            kwargs=dict(noise=NoiseModel(ch),trajectories=151,batch_size=17,seed=11,shots=0)
            a=Simulator(method='trajectory').run(q,noise_fastpath=False,**kwargs)
            b=Simulator(method='trajectory').run(q,noise_fastpath=True,**kwargs)
            np.testing.assert_allclose(a.probabilities,b.probabilities,atol=1e-12)

    def test_specializations_do_not_trust_name(self):
        original=Noise.amplitude_damping(.2)
        renamed=Noise(original.kraus,'depolarizing')
        self.assertIsNone(renamed.pauli_weights())
        self.assertEqual(renamed.damping_spec()[0],'amplitude')
        ch=Noise((np.array([[0,1],[1,0]],complex),),'amplitude_damping')
        np.testing.assert_allclose(ch.pauli_weights(),[0,1,0,0])

    def test_pauli_y_relative_phases(self):
        y=Noise((np.array([[0,-1j],[1j,0]],complex),))
        q=Circuit(2).ry(.3,0).h(1).noise(y,0).h(0).cx(0,1)
        exact=Simulator().run(q,shots=0)
        sampled=Simulator(method='trajectory').run(q,trajectories=7,batch_size=4,shots=0)
        np.testing.assert_allclose(sampled.probabilities,exact.probabilities,atol=1e-12)

    def test_generic_custom_channel_and_no_host_reads(self):
        # Rotated amplitude damping does not match a built-in specialization.
        rot=np.array([[1,1j],[1j,1]])/np.sqrt(2)
        channel=Noise(tuple(rot@k@rot.conj().T for k in Noise.amplitude_damping(.3).kraus))
        self.assertIsNone(channel.pauli_weights()); self.assertIsNone(channel.damping_spec())
        q=Circuit(2).h(0).cx(0,1).noise(channel,1).h(1)
        sim=Simulator(method='trajectory')
        calls=[]; original=sim.backend.numpy
        def counted(value):
            calls.append(1)
            return original(value)
        sim.backend.numpy=counted
        result=sim.run(q,trajectories=2000,batch_size=32,shots=0,seed=19)
        self.assertEqual(len(calls),1) # Final accumulated probabilities only.
        exact=Simulator().run(q,shots=0)
        np.testing.assert_allclose(result.probabilities,exact.probabilities,atol=.035)

    def test_fast_noise_converges_to_exact(self):
        q=Circuit(2).h(0).cx(0,1).ry(.4,1)
        noise=NoiseModel(Noise.depolarizing(.13),Noise.amplitude_damping(.2),Noise.phase_damping(.1))
        exact=Simulator().run(q,noise=noise,shots=0)
        sampled=Simulator(method='trajectory').run(q,noise=noise,trajectories=4000,batch_size=64,shots=0,seed=17)
        np.testing.assert_allclose(sampled.probabilities,exact.probabilities,atol=.025)


class ValidationTests(unittest.TestCase):
    def test_probability_mismatch_rejected(self):
        a=Simulator().run(algorithms.bell(),shots=0)
        b=copy.deepcopy(a); b.probabilities=np.array([.49,.01,0,.5])
        with self.assertRaises(AssertionError): compare_results(a,b)

    def test_phase_error_rejected_and_global_phase_allowed(self):
        a=Simulator().run(algorithms.bell(),shots=0)
        b=copy.deepcopy(a); b.statevector*=np.exp(.7j)
        compare_results(a,b)
        b.statevector[3]*=-1
        with self.assertRaises(AssertionError): compare_results(a,b)

    def test_unpaired_trajectory_rejected(self):
        sim=Simulator(method='trajectory')
        a=sim.run(algorithms.bell(),trajectories=10,batch_size=2,seed=1,shots=0)
        b=copy.deepcopy(a); b.metadata['seed']=2
        with self.assertRaises(ValueError): compare_results(a,b)
        a.metadata['random_engine']=b.metadata['random_engine']='native'; b.metadata['seed']=1
        with self.assertRaises(ValueError): compare_results(a,b)

    def test_tvd_accumulated_errors_rejected(self):
        a=Simulator().run(algorithms.qft(10),shots=0)
        b=copy.deepcopy(a)
        b.probabilities[:512]+=5e-10; b.probabilities[512:]-=5e-10
        # Each error is under 1e-9, but cumulative distribution error is too big.
        with self.assertRaises(AssertionError): compare_results(a,b)

    def test_hoeffding_bound_decreases(self):
        self.assertLess(trajectory_confidence_bound(1000,8),trajectory_confidence_bound(100,8))


try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'Optional PyTorch not installed')
class TorchV03Tests(unittest.TestCase):
    def compare(self,device):
        for engine in ('portable','native'):
            a=DeviceRandom(Backend('torch',device),42,engine).uniform(100)
            b=DeviceRandom(Backend('torch',device),42,engine).uniform(100)
            np.testing.assert_array_equal(a.cpu().numpy(),b.cpu().numpy())
            if engine=='portable':
                np.testing.assert_array_equal(a.cpu().numpy(),DeviceRandom(Backend(),42).uniform(100))
        q=algorithms.ghz(3).ry(.7,1).cp(.3,2,0)
        rot=np.array([[1,1j],[1j,1]])/np.sqrt(2)
        custom=Noise(tuple(rot@k@rot.conj().T for k in Noise.amplitude_damping(.2).kraus))
        for ch in (Noise.depolarizing(.2),Noise.amplitude_damping(.3),Noise.phase_damping(.2),custom):
            for fast in (True,False):
                options=dict(noise=NoiseModel(ch),trajectories=101,batch_size=13,
                             shots=0,seed=42,noise_fastpath=fast)
                a=Simulator(method='trajectory',local_gates=False).run(q,**options)
                b=Simulator(backend='torch',device=device,method='trajectory').run(q,**options)
                compare_results(a,b)
        # Native RNG is checked statistically against an exact density result.
        exact=Simulator().run(q,noise=NoiseModel(Noise.amplitude_damping(.3)),shots=0)
        sampled=Simulator(backend='torch',device=device,method='trajectory').run(q,
                    noise=NoiseModel(Noise.amplitude_damping(.3)),trajectories=4000,batch_size=128,
                    random_engine='native',seed=8,shots=0)
        np.testing.assert_allclose(sampled.probabilities,exact.probabilities,atol=.03)

    def test_torch_cpu_v03(self): self.compare('cpu')

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(),'CUDA GPU unavailable')
    def test_torch_cuda_v03(self): self.compare('cuda:0')


if __name__=='__main__': unittest.main()
