import unittest
import numpy as np
from nexqus import Circuit, Parameter, QuantumDistribution, QGAN, distribution_metrics, Adam


class GenerativeTests(unittest.TestCase):
    def test_probabilities_and_sampling(self):
        t = Parameter('t'); q = QuantumDistribution(Circuit(2).ry(t, 0).cx(0, 1))
        np.testing.assert_allclose(q({'t': np.pi/2}), [.5, 0, 0, .5], atol=1e-12)
        a = q.sample({'t': np.pi/2}, 2000, seed=9)
        np.testing.assert_array_equal(a, q.sample({'t': np.pi/2}, 2000, seed=9))
        self.assertTrue(set(a).issubset({0, 3}))
        self.assertLess(abs(np.mean(a == 0)-.5), .04)

    def test_shared_parameter_jacobian(self):
        t = Parameter('t'); u = Parameter('u')
        c = Circuit(2).h(0).rx(t*.7, 0).ry(t*t+u, 1).cx(1, 0).cp(t, 0, 1).phase(u, 0).rz(u, 1).h(1)
        q = QuantumDistribution(c); b = dict(t=.6, u=-.2)
        p, g = q.value_and_grad(b)
        for name in b:
            plus = dict(b); minus = dict(b)
            plus[name] += 1e-6; minus[name] -= 1e-6
            np.testing.assert_allclose(g[name], (q(plus)-q(minus))/2e-6, atol=1e-8)
            self.assertAlmostEqual(float(g[name].sum()), 0)
        np.testing.assert_allclose(p, q(b))
        self.assertEqual(set(q.gradient(b, wrt=['t'])), {'t'})

    def test_snapshot_and_limits(self):
        c = Circuit(1).ry(Parameter('t'), 0); q = QuantumDistribution(c); c.x(0)
        np.testing.assert_allclose(q({'t': 0}), [1, 0])
        with self.assertRaises(ValueError): QuantumDistribution(Circuit(13))
        with self.assertRaises(ValueError): QuantumDistribution(Circuit(1).measure(0, 'm'))
        with self.assertRaises(ValueError): q({})
        with self.assertRaises(ValueError): q.sample({'t': 0}, shots=0)

    def test_zero_chain_derivative_shape(self):
        t = Parameter('t')
        q = QuantumDistribution(Circuit(2).ry(t*t, 0))
        np.testing.assert_array_equal(q.gradient({'t': 0})['t'], np.zeros(4))
        model = QGAN(q, {'t': 0})
        model.fit([.25]*4, epochs=1)
        self.assertTrue(np.isfinite(model.history[0]['generator_loss']))

    def test_metrics_zero_support(self):
        m = distribution_metrics([1, 0], [0, 1])
        self.assertAlmostEqual(m['total_variation'], 1)
        self.assertAlmostEqual(m['jensen_shannon'], np.log(2))
        self.assertEqual(distribution_metrics([1, 0], [1, 0])['jensen_shannon'], 0)
        for bad in ([0, 0], [1.1, -.1], [np.nan, 0], [1j, 0]):
            with self.assertRaises(ValueError): distribution_metrics(bad, [.5, .5])

    def test_qgan_reproducible_learning(self):
        q = QuantumDistribution(Circuit(1).ry(Parameter('t'), 0))
        target = [.25, .75]; models = [QGAN(q, {'t': 1.3}) for _ in range(2)]
        before = distribution_metrics(target, models[0].probabilities())['total_variation']
        for model in models:
            model.fit(target, epochs=160, generator_optimizer=Adam(.02), discriminator_optimizer=Adam(.05))
            self.assertLess(distribution_metrics(target, model.probabilities())['total_variation'], before)
            self.assertTrue(np.isfinite([r['generator_loss'] for r in model.history]).all())
            self.assertAlmostEqual(model.history[-1]['total_variation'],
                                   distribution_metrics(target, model.probabilities())['total_variation'])
        np.testing.assert_array_equal(models[0].probabilities(), models[1].probabilities())

    def test_validation(self):
        q = QuantumDistribution(Circuit(1).ry(Parameter('t'), 0))
        with self.assertRaises(ValueError): QGAN(q, {})
        with self.assertRaises(ValueError): QGAN(QuantumDistribution(Circuit(1)), {})
        model = QGAN(q, {'t': .5})
        for target in ([1], [.5, .6], [0, np.inf]):
            with self.assertRaises(ValueError): model.fit(target)
        with self.assertRaises(ValueError): model.fit([.5, .5], epochs=0)
        with self.assertRaises(ValueError): model.fit([.5, .5], discriminator_steps=0)

    def test_adversarial_gradients(self):
        # Capture derivatives without updating either model, and compare the
        # non-saturating loss against independent finite differences.
        class Capture:
            def step(self, parameters, gradients):
                self.gradients = dict(gradients)
                return dict(parameters)
        q = QuantumDistribution(Circuit(1).ry(Parameter('t'), 0))
        model = QGAN(q, {'t': .9}); model.discriminator = {'d0': -.3, 'd1': .7}
        gopt = Capture(); dopt = Capture(); target = np.array([.2, .8])
        model.fit(target, epochs=1, generator_optimizer=gopt, discriminator_optimizer=dopt,
                  discriminator_steps=1)
        logits = np.array([-.3, .7])
        def generator_loss(t): return float(q({'t': t})@np.logaddexp(0, -logits))
        self.assertAlmostEqual(gopt.gradients['t'], (generator_loss(.900001)-generator_loss(.899999))/2e-6)
        p = q({'t': .9})
        def dloss(d): return float(.5*(target@np.logaddexp(0, -d)+p@np.logaddexp(0, d)))
        for i, name in enumerate(model.discriminator):
            delta = np.zeros(2); delta[i] = 1e-6
            self.assertAlmostEqual(dopt.gradients[name], (dloss(logits+delta)-dloss(logits-delta))/2e-6)


if __name__ == '__main__': unittest.main()
