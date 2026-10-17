"""Discrete QGAN vs classical categorical GAN, exact local NumPy training.

Run: python examples/qgan_distribution.py
Install the tutorial extra for plotting. Outputs are research demonstrations,
not cross-SDK performance benchmarks or molecular generators.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import matplotlib.pyplot as plt
import nexqus
from nexqus import Circuit, Parameter, QuantumDistribution, QGAN, Adam, distribution_metrics
from nexqus.circuit import positive_integer


def softmax(logits):
    p = np.exp(logits-np.max(logits)); return p/p.sum()


def classical_gan(target, initial, epochs):
    """Same categorical discriminator, objective, update count and Adam rates."""
    params = {f'w{i}': float(v) for i, v in enumerate(np.log(initial))}
    discriminator = {f'd{i}': 0. for i in range(len(target))}
    gopt = Adam(.02); dopt = Adam(.05); history = []
    for epoch in range(epochs):
        p = softmax(np.array(list(params.values())))
        for _ in range(5):
            logits = np.array(list(discriminator.values()))
            d = np.exp(-np.logaddexp(0, -logits))
            dg = .5*((target+p)*d-target)
            discriminator = dopt.step(discriminator, dict(zip(discriminator, dg)))
        logits = np.array(list(discriminator.values())); cost = np.logaddexp(0, -logits)
        gg = p*(cost-p@cost)
        params = gopt.step(params, dict(zip(params, gg)))
        final = softmax(np.array(list(params.values())))
        history.append(dict(epoch=epoch, generator_loss=float(final@cost),
            discriminator_loss=float(.5*(target@cost+final@np.logaddexp(0, logits))),
            **distribution_metrics(target, final)))
    return final, history


def generate(output, epochs=300, seeds=(7, 21, 42)):
    epochs = positive_integer(epochs, 'epochs')
    seeds = tuple(seeds)
    if not seeds: raise ValueError('Provide at least one seed')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    target = np.array([.03, .18, .24, .05, .04, .25, .18, .03])
    circuit = Circuit(3)
    for layer in range(3):
        for wire in range(3): circuit.ry(Parameter(f'w{layer}_{wire}'), wire)
        circuit.cx(0, 1).cx(1, 2)
    generator = QuantumDistribution(circuit)
    records = []; examples = None
    for seed in seeds:
        rng = np.random.default_rng(seed)
        initial_parameters = {name: float(rng.uniform(.2, 2.8)) for name in circuit.parameters}
        model = QGAN(generator, initial_parameters); initial = model.probabilities()
        # Both methods start with precisely the same distribution. No timing is
        # reported here: execution costs and parameter counts are not matched.
        model.fit(target, epochs=epochs)
        classical, classical_history = classical_gan(target, initial, epochs)
        for name, p, history in [('Quantum', model.probabilities(), model.history),
                                  ('Classical', classical, classical_history)]:
            records.append(dict(seed=int(seed), model=name,
                initial_probabilities=initial.tolist(), probabilities=p.tolist(), history=history,
                **distribution_metrics(target, p)))
        if examples is None: examples = (model, classical)
    model, classical = examples
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')
    x = np.arange(8)
    for offset, name, p, color in [(-.25, 'Target', target, '#1f2937'),
            (0, 'Quantum', model.probabilities(), '#7c3aed'), (.25, 'Classical', classical, '#0891b2')]:
        axes[0].bar(x+offset, p, width=.24, label=name, color=color)
    axes[0].set_xticks(x, [format(i, '03b') for i in x], rotation=45)
    axes[0].set(title=f'Distributions · seed {seeds[0]}', ylabel='Probability'); axes[0].legend()
    for name, color in [('Quantum', '#7c3aed'), ('Classical', '#0891b2')]:
        selected = [r for r in records if r['model'] == name]
        curves = np.array([[h['total_variation'] for h in r['history']] for r in selected])
        axes[1].plot(np.arange(1, epochs+1), np.median(curves, axis=0), label=name, color=color)
        axes[1].fill_between(np.arange(1, epochs+1), curves.min(axis=0), curves.max(axis=0), color=color, alpha=.13)
    axes[1].set(title='Median and min–max across seeds', xlabel='Adversarial update', ylabel='Total variation ↓')
    axes[1].legend()
    _, jacobian = generator.value_and_grad(model.parameters)
    sensitivity = np.array(list(jacobian.values()))
    im = axes[2].imshow(sensitivity, aspect='auto', cmap='RdBu_r',
                         vmin=-np.max(abs(sensitivity)), vmax=np.max(abs(sensitivity)))
    axes[2].set_yticks(range(len(jacobian)), list(jacobian)); axes[2].set_xticks(x, [format(i, '03b') for i in x], rotation=45)
    axes[2].set(title='Local probability sensitivity', xlabel='Basis outcome')
    fig.colorbar(im, ax=axes[2], label='∂ probability / ∂ angle')
    fig.savefig(output/'comparison.png', dpi=160); fig.savefig(output/'comparison.svg'); plt.close(fig)
    circuit.explain(bindings=model.parameters, title='QGAN generator · learned probability distribution',
        parameter_roles={name: 'weight' for name in model.parameters}).to_html(filename=output/'generator.html')
    payload = dict(sdk_version=nexqus.__version__, numpy=np.__version__, python=platform.python_version(),
        platform=platform.platform(), execution='local NumPy CPU double precision, exact probabilities',
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(Path(nexqus.__file__).parent.glob('*.py'))},
        example_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        target=target.tolist(), epochs=epochs, seeds=list(seeds), records=records,
        generator_lr=.02, discriminator_lr=.05, discriminator_steps=5,
        quantum_generator_parameters=len(model.parameters), classical_generator_parameters=8,
        discriminator_parameters=8, objective='non-saturating generator; balanced BCE discriminator',
        methodology='Exact full distributions; same initial probabilities, D initialization, Adam rates and update counts. '
                     'Different generator parameterizations and execution costs; synthetic fit, no held-out generalization claim.',
        parameters=model.parameters, probability_jacobian={k: v.tolist() for k, v in jacobian.items()})
    (output/'metrics.json').write_text(json.dumps(payload, indent=2)+'\n', encoding='utf-8')
    for name in ('Quantum', 'Classical'):
        selected = [r for r in records if r['model'] == name]
        print(name, 'median TV:', np.median([r['total_variation'] for r in selected]),
              'median JS:', np.median([r['jensen_shannon'] for r in selected]))
    print('Saved:', output.resolve())
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('.release-artifacts/qgan'))
    parser.add_argument('--epochs', type=int, default=300)
    args = parser.parse_args(); generate(args.output, args.epochs)
