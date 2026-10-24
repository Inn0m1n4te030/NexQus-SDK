"""Quantum/classical sine comparison. Generated from scripts/build_notebooks.py.

python examples/compare_sine.py --output-dir results/sine
Requires the tutorial extra for Matplotlib. Edit the notebook generator
to keep this script and lesson 04 synchronized.
"""
import argparse
import json
from pathlib import Path
from nexqus import QuantumFunction, QuantumRegressor, Adam

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path)
parser.add_argument('--device', help='Quantum device: cpu, cuda, cuda:N or auto')
args = parser.parse_args()

import numpy as np
import matplotlib.pyplot as plt
import nexqus
from nexqus import Circuit, Simulator, Parameter, Noise, NoiseModel, algorithms
assert nexqus.__version__ == '1.3.0rc1', 'Install NexQus SDK 1.3.0rc1 in this kernel environment and restart it'
print('NexQus SDK:', nexqus.__version__)

from time import perf_counter
x, weight, bias = Parameter('x'), Parameter('weight'), Parameter('bias')
template = Circuit(1).ry(weight*x + bias, 0)
function = QuantumFunction(template, 'Z', device=args.device)
train_x = np.linspace(-np.pi, np.pi, 20); train_y = np.sin(train_x)
# Midpoints give 201 evaluation inputs distinct from the training inputs.
test_x = np.linspace(-np.pi, np.pi, 201, endpoint=False) + np.pi/201
test_y = np.sin(test_x)
assert np.min(abs(test_x[:, None] - train_x[None, :])) > 1e-12
grid = np.linspace(-np.pi, np.pi, 401)
epochs, learning_rate, width = 80, .05, 8


def classical_prediction_and_jacobian(kind, values, inputs):
    if kind == 'sinusoid':
        angle = values[0]*inputs + values[1]
        prediction = np.cos(angle)
        jacobian = np.column_stack([-np.sin(angle)*inputs, -np.sin(angle)])
    else:
        # A 1 -> 8 -> 1 tanh MLP; normalized inputs and bounded outputs.
        inputs = inputs/np.pi
        hidden = np.tanh(inputs[:, None]*values[:width] + values[width:2*width])
        prediction = np.tanh(hidden @ values[2*width:3*width] + values[-1])
        output_gradient = 1-prediction**2
        back = output_gradient[:, None]*(1-hidden**2)*values[2*width:3*width]
        jacobian = np.column_stack([back*inputs[:, None], back,
                                   output_gradient[:, None]*hidden, output_gradient])
    return prediction, jacobian


def fit_classical(kind, initial):
    values = np.array(initial, dtype=float, copy=True)
    optimizer = Adam(lr=learning_rate)
    history = []
    for _ in range(epochs):
        predicted, jacobian = classical_prediction_and_jacobian(kind, values, train_x)
        errors = predicted-train_y
        history.append(float(np.mean(errors**2)))
        gradients = 2*jacobian.T @ errors/len(train_x)
        updated = optimizer.step({str(i): float(v) for i, v in enumerate(values)},
                                 {str(i): float(v) for i, v in enumerate(gradients)})
        values = np.array([updated[str(i)] for i in range(len(values))])
    return values, history


# Check classical derivatives independently before comparing training results.
probe = np.random.default_rng(0).normal(0, .2, 3*width+1)
for kind, initial in [('sinusoid', np.array([.7, -.8])), ('mlp', probe)]:
    _, jacobian = classical_prediction_and_jacobian(kind, initial, train_x)
    for index in range(len(initial)):
        step = np.zeros_like(initial); step[index] = 1e-6
        plus = classical_prediction_and_jacobian(kind, initial+step, train_x)[0]
        minus = classical_prediction_and_jacobian(kind, initial-step, train_x)[0]
        np.testing.assert_allclose(jacobian[:, index], (plus-minus)/2e-6,
                                   atol=1e-8, rtol=1e-6)

records, examples = [], {}
for seed in (7, 21, 42):
    rng = np.random.default_rng(seed)
    initial = np.array([.7, -.8]) + rng.uniform(-.05, .05, 2)
    model = QuantumRegressor(function, ['x'], dict(zip(['weight', 'bias'], initial)))
    started = perf_counter()
    model.fit(train_x, train_y, epochs=epochs, optimizer=Adam(lr=learning_rate))
    elapsed = perf_counter()-started
    q_test = model.predict(test_x)
    q_grid = model.predict(grid)
    records.append(dict(seed=seed, name='Quantum circuit', parameters=2,
                        train_mse=float(np.mean((model.predict(train_x)-train_y)**2)),
                        test_mse=float(np.mean((q_test-test_y)**2)), seconds=elapsed))
    if seed == 42:
        examples['Quantum circuit'] = (q_grid, [r['loss'] for r in model.history])
    for kind, label, initial_values in [
            ('sinusoid', 'Classical sinusoid', initial),
            ('mlp', 'Classical MLP', np.r_[rng.normal(0, .6, width),
                                        np.zeros(width), rng.normal(0, .3, width), 0.])]:
        started = perf_counter()
        values, history = fit_classical(kind, initial_values)
        elapsed = perf_counter()-started
        predicted = classical_prediction_and_jacobian(kind, values, test_x)[0]
        if kind == 'sinusoid':
            # Same mathematical model and initialization: predictions should agree.
            np.testing.assert_allclose(predicted, q_test, atol=1e-10, rtol=1e-10)
        train_predicted = classical_prediction_and_jacobian(kind, values, train_x)[0]
        records.append(dict(seed=seed, name=label, parameters=len(values),
                            train_mse=float(np.mean((train_predicted-train_y)**2)),
                            test_mse=float(np.mean((predicted-test_y)**2)), seconds=elapsed))
        if seed == 42:
            examples[label] = (classical_prediction_and_jacobian(kind, values, grid)[0], history)

names = ['Quantum circuit', 'Classical MLP', 'Classical sinusoid']
print('Same 20 training samples; 201 disjoint interpolation points; 80 Adam updates, lr=.05.')
print('Three initializations: 7, 21, 42. Times measure training only on this machine.')
print(f'{"Model":22} {"Params":>6} {"Train MSE":>12} {"Test MSE mean":>15} {"Test MSE std":>14} {"Fit sec mean":>13}')
for name in names:
    rows = [r for r in records if r['name'] == name]
    errors = np.array([r['test_mse'] for r in rows])
    print(f'{name:22} {rows[0]["parameters"]:6d} '
          f'{np.mean([r["train_mse"] for r in rows]):12.3e} '
          f'{errors.mean():15.3e} {errors.std():14.3e} '
          f'{np.mean([r["seconds"] for r in rows]):13.4f}')
assert all(np.isfinite(r['test_mse']) for r in records)

colors = {'Quantum circuit': '#2563eb', 'Classical MLP': '#ea580c',
          'Classical sinusoid': '#9333ea'}
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('NexQus 1.3.0rc1: quantum vs classical sine regression', fontsize=15)
axes[0, 0].plot(grid, np.sin(grid), color='#111827', lw=2, alpha=.45, label='Target sin(x)')
axes[0, 0].scatter(train_x, train_y, color='#111827', s=17, label='20 training samples', zorder=5)
for name in names:
    prediction, history = examples[name]
    style = ':' if name == 'Classical sinusoid' else '-'
    axes[0, 0].plot(grid, prediction, style, color=colors[name], label=name, lw=1.7)
    axes[0, 1].semilogy(np.maximum(history, 1e-16), style, color=colors[name], label=name)
    axes[1, 1].semilogy(grid, np.maximum(abs(prediction-np.sin(grid)), 1e-16),
                        style, color=colors[name], label=name)
means = [np.mean([r['test_mse'] for r in records if r['name'] == name]) for name in names]
stds = [np.std([r['test_mse'] for r in records if r['name'] == name]) for name in names]
bars = axes[1, 0].bar(range(3), means, yerr=stds, capsize=5, color=[colors[n] for n in names])
axes[1, 0].set_yscale('log'); axes[1, 0].set_xticks(range(3), ['Quantum', 'Classical MLP', 'Classical sinusoid'])
for bar, value in zip(bars, means):
    axes[1, 0].annotate(f'{value:.2e}', (bar.get_x()+bar.get_width()/2, value),
                        xytext=(0, 8), textcoords='offset points', ha='center', fontsize=9)
axes[1, 0].set_ylim(top=max(means)*4)
axes[0, 0].set(title='Predictions, seed 42', xlabel='x', ylabel='y')
axes[0, 1].set(title='Training loss, seed 42', xlabel='Adam update', ylabel='MSE before update')
axes[1, 0].set(title='Held-out MSE: mean +/- SD over 3 seeds', ylabel='MSE; lower is better')
axes[1, 1].set(title='Absolute prediction error, seed 42', xlabel='x', ylabel='Absolute error')
for axis in (axes[0, 0], axes[0, 1], axes[1, 1]):
    axis.legend(fontsize=8)
for axis in axes.flat: axis.grid(alpha=.15)
plt.tight_layout()
if args.output_dir is not None:
    import csv
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir/'comparison.png', dpi=160, bbox_inches='tight')
    fig.savefig(args.output_dir/'comparison.svg', bbox_inches='tight')
    payload = dict(sdk_version=nexqus.__version__, backend=function.simulator.backend.name, device=function.simulator.backend.device,
                   classical_comparison_device='numpy cpu',
                   epochs=epochs, learning_rate=learning_rate, seeds=[7, 21, 42],
                   training_samples=len(train_x), held_out_samples=len(test_x),
                   evaluation='disjoint in-domain interpolation', records=records)
    (args.output_dir/'metrics.json').write_text(json.dumps(payload, indent=2)+'\n', encoding='utf-8')
    with (args.output_dir/'predictions.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['x', 'target', *names])
        writer.writerows(zip(grid, np.sin(grid), *(examples[n][0] for n in names)))
    print('Saved plots, metrics and predictions to', args.output_dir)
plt.show()
