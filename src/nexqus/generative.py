"""Small exact, discrete hybrid QGANs. No cloud calls or ML dependencies."""
import copy
import numpy as np
from .circuit import Circuit, positive_integer
from .gradients import QuantumFunction
from .optimizers import Adam
from .parameters import bindings_dict


def distribution(values, size=None):
    """Validate a probability vector without silently normalizing user data."""
    if np.iscomplexobj(values):
        raise ValueError('Probabilities must be real')
    p = np.asarray(values, dtype=float)
    if (p.ndim != 1 or not len(p) or (size is not None and len(p) != size)
            or not np.isfinite(p).all() or np.any(p < 0)
            or not np.isclose(p.sum(), 1, atol=1e-10, rtol=0)):
        raise ValueError('Provide a finite nonnegative probability vector summing to one')
    return p.copy()


def distribution_metrics(target, generated):
    """Total variation and Jensen–Shannon divergence (natural logarithms)."""
    p = distribution(target); q = distribution(generated, len(p)); m = (p + q) / 2
    def kl(a):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / m[mask])))
    return dict(total_variation=float(np.sum(abs(p-q))/2),
                jensen_shannon=(kl(p)+kl(q))/2)


class QuantumDistribution(QuantumFunction):
    """q0-first Born probabilities and gate-local parameter-shift Jacobians.

    Returns {parameter: derivative probability vector}; shared parameters use
    the gate-local chain rule, not a shift of the whole parameter. Exact local
    pure-state execution only; full distributions are bounded by max_qubits.
    The circuit is snapshotted on construction.
    """
    def __init__(self, template, max_qubits=12):
        if not isinstance(template, Circuit):
            raise TypeError('Expected Circuit')
        if template.n_qubits > positive_integer(max_qubits, 'max_qubits'):
            raise ValueError('Distribution exceeds max_qubits')
        if template.dynamic or any(op.kind not in ('unitary', 'diagonal', 'permutation', 'parameterized')
                                   for op in template.operations):
            raise ValueError('QuantumDistribution requires a static pure-state circuit')
        super().__init__(copy.deepcopy(template), 'I' * template.n_qubits)

    def _evaluate(self, circuit):
        return self.simulator.run(circuit, shots=0).probabilities.copy()

    def _compute(self, bindings, wrt, include_value):
        value, gradients = super()._compute(bindings, wrt, include_value)
        # A zero arithmetic chain derivative can skip every gate-local shift.
        # Even then the Jacobian column is a vector, never a scalar zero.
        size = 2**self.template.n_qubits
        return value, {name: np.broadcast_to(g, (size,)).copy() for name, g in gradients.items()}

    def sample(self, bindings, shots=1000, seed=None):
        """Return integer basis outcomes; format with n_qubits bits for labels."""
        shots = positive_integer(shots, 'shots')
        return np.random.default_rng(seed).choice(2**self.template.n_qubits,
                                                  size=shots, p=self(bindings))


class QGAN:
    """Quantum Born generator + classical categorical-logit discriminator.

    Full-distribution, deterministic adversarial training, not minibatch or
    image/molecule generation. D minimizes balanced binary cross entropy;
    G minimizes non-saturating -E_generated[log D]. All generator parameters
    are trainable. The discriminator has one independent logit per category.
    """
    def __init__(self, generator, initial_parameters):
        if not isinstance(generator, QuantumDistribution):
            raise TypeError('Expected QuantumDistribution')
        self.generator = generator
        self.parameters = bindings_dict(initial_parameters)
        generator.template.bind(self.parameters)
        if not self.parameters:
            raise ValueError('Generator must have trainable parameters')
        self.discriminator = {f'd{i}': 0.0 for i in range(2**generator.template.n_qubits)}
        self.history = []

    def probabilities(self):
        return self.generator(self.parameters)

    def fit(self, target, epochs=200, generator_optimizer=None, discriminator_optimizer=None,
            discriminator_steps=5):
        target = distribution(target, len(self.discriminator))
        epochs = positive_integer(epochs, 'epochs')
        discriminator_steps = positive_integer(discriminator_steps, 'discriminator_steps')
        gopt = generator_optimizer if generator_optimizer is not None else Adam(lr=.02)
        dopt = discriminator_optimizer if discriminator_optimizer is not None else Adam(lr=.05)
        if any(not callable(getattr(opt, 'step', None)) for opt in (gopt, dopt)):
            raise TypeError('Optimizers must provide step()')
        self.history = []
        for epoch in range(epochs):
            p, jac = self.generator.value_and_grad(self.parameters)
            for _ in range(discriminator_steps):
                logits = np.array(list(self.discriminator.values()))
                d = np.exp(-np.logaddexp(0, -logits))
                dg = .5 * ((target+p)*d-target)
                self.discriminator = dopt.step(self.discriminator,
                    {name: float(g) for name, g in zip(self.discriminator, dg)})
            logits = np.array(list(self.discriminator.values()))
            cost = np.logaddexp(0, -logits)
            self.parameters = gopt.step(self.parameters,
                {name: float(np.dot(cost, derivative)) for name, derivative in jac.items()})
            final = self.probabilities()
            self.history.append(dict(epoch=epoch, generator_loss=float(final@cost),
                discriminator_loss=float(.5*(target@cost+final@np.logaddexp(0, logits))),
                **distribution_metrics(target, final)))
        return self
