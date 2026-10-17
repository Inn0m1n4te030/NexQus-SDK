"""Generate bilingual, plain-Python tutorial notebooks without extra dependencies."""
import hashlib
import json
from pathlib import Path
from textwrap import dedent

ROOT=Path(__file__).resolve().parents[1]


def md(text): return ('markdown',dedent(text).strip())
def code(text): return ('code',dedent(text).strip())


SETUP=code('''
import numpy as np
import matplotlib.pyplot as plt
import nexqus
from nexqus import Circuit, Simulator, Parameter, Noise, NoiseModel, algorithms
assert nexqus.__version__ == '1.2.0', 'Install NexQus SDK 1.2.0 in this kernel environment and restart it'
print('NexQus SDK:', nexqus.__version__)
''')


BOOKS=[
('00_start_here',[
md('''
# 00 · Your first NexQus circuit / စတင်အသုံးပြုခြင်း

**Goal:** install NexQus, run a Bell circuit, and distinguish probabilities from sampled counts.
**Time:** about 15 minutes. No physics or GPU background is required. Targets NexQus SDK 1.2.0.

NexQus ဖြင့် မိမိကွန်ပျူတာပေါ်တွင် quantum circuit ကို simulation လုပ်နိုင်ပါသည်။
စတင်ရန် quantum hardware သို့မဟုတ် GPU မလိုအပ်ပါ။

Open a terminal in the extracted SDK folder, activate your environment, and run:

```bash
python -m pip install 'nexqus-sdk[tutorial]==1.2.0'
python -m ipykernel install --user --name nexqus --display-name "Python (NexQus)"
jupyter lab notebooks
```

Choose **Python (NexQus)** in Jupyter. Install commands belong in your terminal;
the code cells below contain plain Python, so they also work with the bundled runner.
'''),SETUP,
md('''
## Circuit → simulator → result

A circuit records operations. A simulator executes them. A result contains probabilities and sampled counts.
`q0` is the **leftmost** bit. `Circuit(2).x(0)` prepares `10`.

Circuit သည် gate အစဉ်ကို သိမ်းထားပြီး `run()` က တွက်ချက်ပေးပါသည်။
Bitstring ကို ဘယ်ဘက်မှ q0, q1, ... အစဉ်ဖြင့် ဖတ်ပါ။
'''),
code('''
q = Circuit(2).x(0)
result = Simulator().run(q, shots=100, seed=42)
print(result.counts)
assert result.counts == {'10': 100}
'''),
code('''
# H creates superposition; CX entangles the two qubits.
# qubit နှစ်ခုကို entangle လုပ်ပါ။
bell = algorithms.bell()
result = Simulator().run(bell, shots=1000, seed=42)
print('Probabilities:', result.probabilities)
print('Counts:', result.counts)
np.testing.assert_allclose(result.probabilities, [.5, 0, 0, .5], atol=1e-12)
assert set(result.counts) <= {'00', '11'}
assert sum(result.counts.values()) == 1000
'''),
md('''
Probabilities describe the simulated distribution. Counts fluctuate because we sample a finite number of shots.
More shots do not add qubits or make an approximate noise ensemble exact.
`seed` makes these draws repeatable under a fixed SDK/environment configuration.

Probability နှင့် count ကို ခွဲခြားကြည့်ပါ။ Count သည် random sampling ကြောင့် အနည်းငယ်ကွာနိုင်ပါသည်။
'''),
code('''
plt.figure(figsize=(5, 3))
plt.bar(['00', '01', '10', '11'], result.probabilities, color='#3b82f6')
plt.ylim(0, 1); plt.ylabel('Probability'); plt.title('Bell-state distribution')
plt.tight_layout(); plt.show()
print('Checks passed. Next: 01_qubits_and_gates.ipynb')
'''),
md('''
## Draw your own circuit / Circuit ပုံဆွဲခြင်း

SDK 1.2.0 draws any ordinary circuit without simulating it. Here RX and RY
are rotations in radians; `cx` is CNOT. All eight wires appear, including idle
wires. `draw()` returns terminal text; `draw('svg')` displays in Jupyter.
To save a vector file, use `custom.draw('svg', filename='custom.svg')`.
Built-in rotations display numerical angles; symbolic gates include their
expressions. Lesson 07 covers inspection, export and QML helpers.

မိမိတည်ဆောက်ထားသော circuit ကို simulation မလုပ်ဘဲ ပုံဆွဲနိုင်ပါသည်။
Jupyter တွင် SVG ပုံကို ကြည့်နိုင်ပြီး file အဖြစ်လည်း သိမ်းနိုင်ပါသည်။
'''),
code('''
custom = Circuit(8).rx(.5, 0).cx(0, 1).ry(.8, 3).cx(3, 7)
text_diagram = custom.draw()
assert all(f'q{i}' in text_diagram for i in range(8))
print(text_diagram)
svg_diagram = custom.draw('svg')
from xml.etree import ElementTree
assert ElementTree.fromstring(svg_diagram).tag.endswith('svg')
svg_diagram  # Jupyter displays SVG; the standalone runner prints the text above.
'''),
md('''
**Try it:** remove CX and inspect the bitstrings. Then change `shots` from 1000 to 100.
Predict which probabilities change and which differences are only sampling variation.
''')]),
('01_qubits_and_gates',[
md('''
# 01 · Qubits, gates and entanglement / Qubit နှင့် gate များ

**Goal:** connect amplitudes, rotations, correlated measurements and algorithms. About 20 minutes.
A pure-state amplitude vector has squared magnitudes summing to one. Measurement probabilities
do not reveal every complex phase in that vector.

Amplitude သည် complex number ဖြစ်နိုင်ပြီး probability သည် ၎င်း၏ absolute value squared ဖြစ်ပါသည်။
'''),SETUP,
code('''
plus = Simulator().run(Circuit(1).h(0), shots=0)
print('Amplitudes:', plus.statevector)
print('Probabilities:', plus.probabilities)
np.testing.assert_allclose(np.abs(plus.statevector)**2, plus.probabilities)
assert np.isclose(np.vdot(plus.statevector, plus.statevector), 1)
'''),
md('''
RY(theta) on |0> gives probability of 1 equal to sin²(theta/2).
Angle များကို radians ဖြင့် သတ်မှတ်ပါ။ pi radians သည် 180 degrees ဖြစ်ပါသည်။
'''),
code('''
angles = np.linspace(0, 2*np.pi, 41)
p_one = np.array([Simulator().run(Circuit(1).ry(float(t), 0), shots=0).probabilities[1] for t in angles])
np.testing.assert_allclose(p_one, np.sin(angles/2)**2, atol=1e-12)
plt.figure(figsize=(6, 3)); plt.plot(angles, p_one)
plt.xlabel('RY angle (radians)'); plt.ylabel('P(1)'); plt.title('A single-qubit rotation')
plt.tight_layout(); plt.show()
'''),
code('''
bell = Simulator().run(algorithms.bell(), shots=0)
print('ZZ correlation:', bell.expectation('ZZ'))
print('XX correlation:', bell.expectation('XX'))
print('State of q0 alone:\\n', bell.reduced_density([0]))
np.testing.assert_allclose(bell.reduced_density([0]), np.eye(2)/2, atol=1e-12)
assert np.isclose(bell.expectation('XX'), 1)
assert np.isclose(bell.expectation('ZZ'), 1)
'''),
md('''
Each Bell qubit alone is mixed even though the pair has a pure state. Correlation in both X and Z
is more informative than seeing matching computational-basis samples alone.

Entangled pair သည် pure state ဖြစ်သော်လည်း qubit တစ်ခုတည်းကို သီးခြားကြည့်လျှင် mixed state ဖြစ်နိုင်ပါသည်။
'''),
code('''
# QFT is implemented from gates. Check inverse QFT on a complex input.
vector = np.array([1, 1j, 2, -1], complex); vector /= np.linalg.norm(vector)
round_trip = algorithms.qft(2).compose(algorithms.qft(2, inverse=True))
recovered = Simulator().run(round_trip, initial_state=vector, shots=0).statevector
np.testing.assert_allclose(recovered, vector, atol=1e-12)
grover = Simulator().run(algorithms.grover(2, '10'), shots=0)
assert grover.most_likely == '10'
assert np.isclose(grover.probabilities[2], 1)
print('QFT inverse and Grover checks passed.')
'''),
md('''
## Shor factoring and resources / Shor algorithm

In SDK 1.2.0, `shor` combines quantum order finding, inverse QFT and classical
factor recovery. Modular operations use compact permutation maps. Inspect
resources before trying larger inputs: storing a statevector still grows
exponentially with the total qubit count, and the memory checks are estimates.

77 is a 7-bit modulus (7 × 11), not an RSA-2048-size example. At default counting
precision it uses 21 simulated qubits: 32 MiB for the double-precision state and
about 192 MiB estimated state working storage, plus gates. The smaller factoring
examples below use the quantum path, not the classical even/perfect-power shortcut.
An unsuccessful bounded attempt returns `factors=None`; it is not proof of primality.
These are logical simulated gates, not a hardware arithmetic decomposition.

Input number ကြီးလာလျှင် memory ပိုလိုအပ်ပါသည်။ မစမ်းမီ resource estimate ကို
စစ်ဆေးပါ။ ဤဥပမာများသည် RSA-2048 ကို factor လုပ်နိုင်သည်ဟု မဆိုလိုပါ။
'''),
code('''
resources = algorithms.shor_resources(77)
assert resources.total_qubits == 21
print('77 resources:', resources)
for modulus, expected in [(15, (3, 5)), (21, (3, 7)), (35, (5, 7))]:
    factors = algorithms.shor(modulus, base=2, seed=42)
    assert factors.factors == expected and factors.method == 'quantum'
    print(modulus, '->', factors.factors, 'method:', factors.method)
'''),
md('''
**Try it:** prepare `algorithms.ghz(3)` and predict its nonzero probabilities.
Simulator access to all amplitudes is not the same as measuring those amplitudes on hardware.
''')]),
('02_noise_and_methods',[
md('''
# 02 · Noise and simulation methods / Noise နှင့် simulation mode များ

**Goal:** choose statevector, density matrix or trajectories. About 15 minutes.
- Statevector: noiseless pure-state evolution.
- Density matrix: deterministic exact channel evolution, subject to floating-point error.
- Trajectory: a sampled noise ensemble; increasing shots does not increase its histories.

Exact noise model နှင့် sampled estimate ကို ခွဲခြားပါ။ Trajectory အရေအတွက် ပိုများလျှင်
Monte Carlo error ကို လျှော့ချနိုင်ပါသည်။
'''),SETUP,
code('''
flipped = Simulator().run(Circuit(1).noise(Noise.bit_flip(.2), 0), shots=0)
np.testing.assert_allclose(flipped.probabilities, [.8, .2], atol=1e-12)
print('Bit-flip p=0.2:', flipped.probabilities)
'''),
code('''
q = algorithms.bell()
model = NoiseModel(Noise.depolarizing(.08), Noise.amplitude_damping(.04))
ideal = Simulator().run(q, shots=0)
exact = Simulator(method='density_matrix').run(q, noise=model, shots=0)
sampled = Simulator(method='trajectory').run(q, noise=model, trajectories=2048,
                                           batch_size=64, shots=0, seed=42)
tvd = float(np.abs(exact.probabilities - sampled.probabilities).sum()/2)
print('Exact:', exact.probabilities)
print('Trajectory estimate:', sampled.probabilities)
print('Total variation distance:', tvd)
assert tvd < .08
assert np.isclose(np.trace(exact.density_matrix), 1)
assert np.linalg.eigvalsh(exact.density_matrix).min() > -1e-12
'''),
code('''
positions = np.arange(4)
plt.figure(figsize=(7, 3))
for offset, (name, probabilities) in enumerate([('Ideal', ideal.probabilities), ('Exact noise', exact.probabilities), ('Trajectories', sampled.probabilities)]):
    plt.bar(positions + (offset-1)*.25, probabilities, width=.25, label=name)
plt.xticks(positions, ['00', '01', '10', '11']); plt.ylabel('Probability')
plt.title('Specified noise model: exact versus sampled'); plt.legend()
plt.tight_layout(); plt.show()
'''),
md('''
Global noise rules act after each gate on every gate-touched wire. Depolarizing uses
E(rho)=(1-p)rho+p I/2 on the selected qubit. Reset and intermediate measurement are ideal
unless explicit noise is inserted. Do not assume every library uses the same p convention.

Global noise သည် gate လုပ်ပြီးတိုင်း သက်ဆိုင်ရာ qubit များတွင် ထည့်ပါသည်။
'''),
code('''
# Storage estimates only: do not allocate these large density matrices.
for n in [10, 16, 20]:
    state_mib = 16 * 2**n / 1024**2
    density_mib = 16 * 4**n / 1024**2
    print(f'{n} qubits: statevector {state_mib:g} MiB; density matrix {density_mib:g} MiB')
try:
    Simulator(method='density_matrix', max_memory_mb=16).run(Circuit(12), shots=0)
except MemoryError:
    print('Expected: the working-memory guard blocked this allocation.')
else:
    raise AssertionError('Memory guard should reject this run')
'''),
md('''
**Try it:** compare 256 and 2048 trajectories with a fixed seed. One trial need not improve monotonically;
sample more seeds to assess variation. Peak process memory exceeds the main-state estimates above.
''')]),
('03_functions_sweeps_and_devices',[
md('''
# 03 · Functions, sweeps and devices / Function နှင့် parameter sweep

**Goal:** write once, bind inputs, reuse blocks, and choose a backend. About 15 minutes.
All required cells use CPU. The optional CUDA cell prints a clear skip when unavailable.

Circuit ကို တစ်ကြိမ်ရေးပြီး parameter များ ပြောင်းလဲစမ်းသပ်ပါ။ GPU မရှိလည်း ဤ lesson ကို လေ့လာနိုင်ပါသည်။
'''),SETUP,
code('''
from nexqus import circuit

@circuit(2)
def entangler(q, theta):
    q.ry(theta, 0).cx(0, 1)

theta = Parameter('theta')
template = entangler(theta)
sim = Simulator()
bindings = [{'theta': float(t)} for t in np.linspace(0, np.pi, 9)]
results = sim.run_sweep(template, bindings, shots=0, seed=42, retain_state=True)
measured = np.array([r.expectation('ZI') for r in results])
np.testing.assert_allclose(measured, np.cos([v['theta'] for v in bindings]), atol=1e-12)
print('Parameters:', template.parameters)
print('Expectations:', measured)
'''),
code('''
prepared = sim.compile(template.bind({'theta': .7}))
a = prepared.run(shots=100, seed=7)
b = prepared.run(shots=100, seed=7)
assert a.counts == b.counts
print('Reusable counts:', a.counts)
print('Gate cache:', sim.cache_info())
larger = Circuit(3).compose(entangler(.7), wires=(2, 0))
assert larger.n_qubits == 3
print('Remapped block:', larger)
'''),
code('''
# Optional CUDA check. Tiny circuits are not a GPU speed benchmark.
import importlib.util
gpu_available = False
if importlib.util.find_spec('torch') is not None:
    import torch
    gpu_available = torch.cuda.is_available()
if gpu_available:
    from nexqus.validation import compare_results
    q = algorithms.ghz(3)
    cpu_result = Simulator().run(q, shots=0)
    gpu_result = Simulator(backend='torch', device='cuda:0').run(q, shots=0)
    print('CUDA parity:', compare_results(cpu_result, gpu_result)['check'])
else:
    print('SKIPPED optional CUDA cell: compatible PyTorch/CUDA is unavailable.')
'''),
md('''
The SDK's tested RTX 2070 experiment favored GPU batch 256 for one 10-qubit noisy workload.
That is a measured example, not a universal default. Memory grows with batch size; CPU can favor smaller batches.
Only use GPU through `backend='torch', device='cuda:0'`; NumPy is CPU-only and CUDA never silently falls back.

GPU ပိုမြန်မည်ဟု အမြဲမယူဆပါနှင့်။ Circuit အရွယ်အစားနှင့် batch size ပေါ် မူတည်ပါသည်။
'''),
code('''
result = Simulator(method='trajectory').run(algorithms.ghz(3),
    noise=NoiseModel(Noise.depolarizing(.02)), trajectories=128, batch_size=32,
    shots=0, seed=42)
assert result.metadata['batch_size'] == 32
print({key: result.metadata[key] for key in ['backend', 'device', 'method', 'batch_size', 'estimated_working_bytes']})
'''),
md('''
**Try it:** stream a large sweep with `sim.iter_sweep(...)` rather than retaining all results.
Keep backend, precision, RNG engine, batch layout and SDK version fixed for reproducibility.
''')]),
('04_gradients_and_learning',[
md('''
# 04 · Gradients and learning / Gradient နှင့် QML အခြေခံ

**Goal:** differentiate an expectation and compare quantum/classical learning. About 25 minutes.
Parameter-shift changes one gate occurrence at a time and applies the arithmetic chain rule.
This SDK API is explicit optimization, not PyTorch autograd.
Run all cells to display the four-panel quantum/classical comparison below.
A saved preview is linked from [the learning guide](../docs/LEARNING.md).

Parameter တစ်ခုကို gate များစွာတွင် ပြန်သုံးထားလျှင် derivative များကို chain rule ဖြင့် ပေါင်းပါသည်။
'''),SETUP,
code('''
from nexqus import QuantumFunction, QuantumRegressor, QuantumBinaryClassifier, Adam
theta = Parameter('theta')
function = QuantumFunction(Circuit(1).ry(theta, 0).ry(2*theta, 0), 'Z')
value, gradients = function.value_and_grad({'theta': .2})
assert np.isclose(value, np.cos(.6))
assert np.isclose(gradients['theta'], -3*np.sin(.6))
h = 1e-6
finite_difference = (function({'theta': .2+h}) - function({'theta': .2-h}))/(2*h)
np.testing.assert_allclose(gradients['theta'], finite_difference, atol=1e-8)
print('Analytic gradient:', gradients['theta'])
print('Finite-difference check:', finite_difference)
'''),
code('''
from time import perf_counter
x, weight, bias = Parameter('x'), Parameter('weight'), Parameter('bias')
template = Circuit(1).ry(weight*x + bias, 0)
function = QuantumFunction(template, 'Z')
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
'''),
code('''
colors = {'Quantum circuit': '#2563eb', 'Classical MLP': '#ea580c',
          'Classical sinusoid': '#9333ea'}
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('NexQus 1.2.0: quantum vs classical sine regression', fontsize=15)
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
plt.tight_layout(); plt.show()
'''),
md('''
## Read the comparison / ရလဒ်များကို နှိုင်းယှဉ်ခြင်း

The blue curve is the trained quantum circuit. Orange is a classical 1-8-1 tanh
neural network (25 parameters). Purple is a classical cosine model (2 parameters),
with exactly the same initialization and function family as the quantum circuit.
The quantum expectation is cos(weight*x+bias); both 2-parameter models can learn
sine at weight=1, bias=-pi/2. Their overlapping curves and matching errors are expected.

The table reports the actual result for every model; no model is required to win.
If quantum MSE is lower than the MLP's, it fits this task better under these stated
settings. The matched sinusoidal baseline shows whether the benefit comes from
the periodic model family. Compare timing separately: quantum simulation executes
on a classical CPU here. An 80-update budget with one MLP configuration is not an
exhaustive hyperparameter search. These held-out inputs test interpolation within
the same interval, not extrapolation or real cybersecurity data.

ယခု quantum model နှင့် classical model နှစ်မျိုးကို data နှင့် training update
အရေအတွက်တူညီစွာ အသုံးပြုပြီး နှိုင်းယှဉ်ထားပါသည်။ Test MSE နည်းလေ prediction
ပိုကောင်းလေ ဖြစ်ပါသည်။ Quantum curve နှင့် classical sinusoid curve ထပ်နေခြင်းသည်
ဤ circuit အတွက် function family တူညီသောကြောင့် ဖြစ်ပါသည်။ မိမိ run ရလဒ်၏
error၊ parameter အရေအတွက်နှင့် training time ကို အတူကြည့်ပါ။
'''),
code('''
classifier = QuantumBinaryClassifier(function, ['x'], {'weight': .7, 'bias': -1.0})
features = np.array([-1.2, -.7, -.3, .3, .7, 1.2]); labels = (features > 0).astype(int)
classifier.fit(features, labels, epochs=35, optimizer=Adam(lr=.06))
np.testing.assert_array_equal(classifier.predict(features), labels)
print('Toy classification labels:', classifier.predict(features))
print('Scores:', classifier.predict_proba(features))
'''),
md('''
**Limits:** binary scores are not guaranteed calibrated probabilities. Trajectory gradients, dynamic-circuit
gradients, multiclass learning and a Torch neural-network layer are not included.
**Try it:** vary the initial parameters or training rate and compare the recorded losses.
''')]),
('05_measurement_reset_teleportation',[
md('''
# 05 · Measurement, reset and teleportation / Measurement နှင့် feed-forward

**Goal:** use measurement outcomes to control later operations. About 20 minutes.
Default dynamic execution tracks exact density branches; trajectory mode samples histories instead.

Measurement ရလဒ်ကို classical bit တွင် သိမ်းပြီး နောက် gate ကို ထိန်းချုပ်နိုင်ပါသည်။
'''),SETUP,
code('''
q = Circuit(1).h(0).measure(0, 'm').x(0).c_if('m', 1)
result = Simulator().run(q, shots=1000, seed=42)
print('Final quantum counts:', result.counts)
print('Intermediate measurement counts:', result.classical_counts)
assert result.counts == {'0': 1000}
np.testing.assert_allclose(list(result.classical_probabilities.values()), [.5, .5], atol=1e-12)
'''),
code('''
result = Simulator().run(Circuit(2).h(0).cx(0, 1).reset(0), shots=0)
print('After resetting q0:', result.probabilities)
np.testing.assert_allclose(result.reduced_density([0]), [[1, 0], [0, 0]], atol=1e-12)
np.testing.assert_allclose(result.reduced_density([1]), np.eye(2)/2, atol=1e-12)
'''),
md('''
Reset discards q0's state and prepares |0>. Its entangled partner remains mixed.
`c_if` supports one named bit equal to 0 or 1; compound conditions are not implemented.
Overwriting a named bit discards its old value. `joint_counts` uses `classical|quantum` strings.

Reset လုပ်ခြင်းသည် entangled partner ကို pure state သို့ ပြန်မထားပါ။
'''),
code('''
theta, phi = .83, -.47
target = np.array([np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)])
q = Circuit(3).ry(theta, 0).phase(phi, 0).compose(algorithms.teleportation())
result = Simulator().run(q, shots=1000, seed=42)
bob = result.reduced_density([2])
fidelity = float(np.vdot(target, bob @ target).real)
print('Alice probabilities:', result.classical_probabilities)
print('Bob state:\\n', bob)
print('Teleportation fidelity:', fidelity)
np.testing.assert_allclose(bob, np.outer(target, target.conj()), atol=1e-12)
assert np.isclose(fidelity, 1)
'''),
code('''
# Remove feed-forward corrections: the receiver alone is now mixed.
without_corrections = Circuit(3).ry(theta, 0).phase(phi, 0)
without_corrections.h(1).cx(1, 2).cx(0, 1).h(0).measure(0, 'z').measure(1, 'x')
uncorrected = Simulator().run(without_corrections, shots=0).reduced_density([2])
np.testing.assert_allclose(uncorrected, np.eye(2)/2, atol=1e-12)
print('Without corrections:\\n', uncorrected)
'''),
code('''
correlated = Circuit(2).h(0).measure(0, 'm').x(1).c_if('m', 1)
sampled = Simulator(method='trajectory').run(correlated, trajectories=512,
                                            batch_size=64, shots=1000, seed=42)
print('Joint sampled counts:', sampled.joint_counts)
assert set(sampled.joint_counts) <= {'0|00', '1|11'}
assert sum(sampled.joint_counts.values()) == 1000
'''),
md('''
**Try it:** change the teleported state's angles, including a nonzero phase. Check the reduced density,
not just P(0)/P(1): probabilities alone cannot verify phase preservation.
Exact branch count can grow exponentially; use `max_branches`/memory guards or sampled trajectories.
''')]),
('06_your_first_experiment',[
md('''
# 06 · Build a reproducible experiment / ကိုယ်ပိုင် experiment တည်ဆောက်ခြင်း

**Goal:** compare ideal, exact-noisy and sampled-noisy expectations with a reusable template.
About 15 minutes. Keep this notebook as a starting point for your next experiment.

မေးခွန်း၊ circuit၊ noise model၊ seed နှင့် အကဲဖြတ်မည့် metric ကို ရှင်းလင်းစွာ သတ်မှတ်ပါ။
'''),SETUP,
md('''
**Question:** how does the specified depolarizing model change the first qubit's Z expectation
as the input angle varies? We compare simulation methods, not quantum versus classical computational advantage.
'''),
code('''
theta = Parameter('theta')
template = Circuit(2).ry(theta, 0).cx(0, 1)
bindings = [{'theta': float(t)} for t in np.linspace(0, np.pi, 9)]
model = NoiseModel(Noise.depolarizing(.05))
sim = Simulator()
ideal = sim.run_sweep(template, bindings, seed=17, shots=0, retain_state=True)
exact = sim.run_sweep(template, bindings, noise=model, seed=17, shots=0, retain_state=True)
ideal_z = np.array([r.expectation('ZI') for r in ideal])
exact_z = np.array([r.expectation('ZI') for r in exact])
# q0 is touched by RY and CX, so local depolarization attenuates Z twice.
np.testing.assert_allclose(exact_z, .95**2 * ideal_z, atol=1e-12)
'''),
code('''
trajectories = Simulator(method='trajectory').run_sweep(template, bindings, noise=model,
    seed=17, shots=0, trajectories=2048, batch_size=64)
# A diagonal observable can be evaluated from probabilities without retaining a density matrix.
sampled_z = np.array([r.marginal([0])[0] - r.marginal([0])[1] for r in trajectories])
max_error = float(np.max(np.abs(exact_z - sampled_z)))
print('Max expectation discrepancy:', max_error)
assert max_error < .1
'''),
code('''
angle_values = [b['theta'] for b in bindings]
plt.figure(figsize=(7, 3))
plt.plot(angle_values, ideal_z, label='Ideal')
plt.plot(angle_values, exact_z, label='Exact specified noise')
plt.scatter(angle_values, sampled_z, label='Trajectory estimate', color='#f59e0b')
plt.xlabel('Input angle (radians)'); plt.ylabel('Expectation ZI')
plt.title('A reproducible circuit experiment'); plt.legend()
plt.tight_layout(); plt.show()
'''),
code('''
import json
report = dict(sdk_version=nexqus.__version__, backend='numpy', precision='double',
              base_seed=17, noise='post-gate local depolarizing p=0.05',
              trajectories=2048, batch_size=64, max_expectation_error=max_error,
              angles=angle_values, ideal=ideal_z.tolist(), exact=exact_z.tolist(),
              sampled=sampled_z.tolist())
print(json.dumps(report, indent=2))
print('All experiment checks passed.')
'''),
md('''
## Next steps / ဆက်လက်လေ့လာရန်

1. Change one variable at a time: noise probability, circuit depth or input angles.
2. Repeat stochastic experiments with several seeds and report variability.
3. Add a new application only after defining a relevant classical baseline and metric.
4. Use `examples/train_sine.py`, `examples/vqe_two_qubits.py` and the API guides as starting points.

တစ်ကြိမ်လျှင် variable တစ်ခုသာ ပြောင်းပြီး မူလ experiment နှင့် နှိုင်းယှဉ်ပါ။
သင်ခန်းစာများ၏ passing assertions သည် ဤ example များအတွက်သာ မှန်ကန်ကြောင်း ပြပါသည်။
''')])]


BOOKS.append(('07_inspection_and_qml_templates',[
md('''
# 07 · Circuit inspection and reusable QML / Circuit စစ်ဆေးခြင်းနှင့် QML

**Goal:** inspect/export a circuit and build a differentiable data-reuploading
model. About 20 minutes. Install NexQus 1.2.0 with
`python -m pip install 'nexqus-sdk[tutorial,visualization]==1.2.0'` and restart
the kernel. PNG/PDF export also requires native Cairo on your system.

Data encoding နှင့် trainable layer များကို helper များဖြင့် ပြန်သုံးနိုင်ပါသည်။
ဤ lesson အတွက် NexQus SDK 1.2.0 ကို install လုပ်ထားရန် လိုပါသည်။
'''),SETUP,
code('''
from nexqus import templates, QuantumFunction
assert hasattr(Circuit, 'depth'), 'Install NexQus SDK 1.2.0 and restart the kernel'
q = Circuit(8).rx(.5, 0).cx(0, 1).ry(.8, 3).cx(3, 7)
assert q.depth() == 2
assert q.count_ops() == {'RX': 1, 'CX': 2, 'RY': 1}
assert 'RX(0.5)' in q.draw()
print(q.draw())
print('Logical depth:', q.depth(), 'gate counts:', q.count_ops(gates_only=True))
q.draw('svg')
'''),
md('''
## Encode data and choose connections

Features follow the selected wire order. Angles use radians; fit any classical
scaler using training data. `layer_shape` includes a final rotation-axis
dimension, even for a single axis. A two-wire ring has one entangling edge.
'''),
code('''
basis = templates.basis_encoding(Circuit(3), [1, 0, 1])
assert Simulator().run(basis, shots=0).most_likely == '101'
encoded = templates.angle_encoding(Circuit(4), [.2, -.4], wires=[3, 0], rotation='X')
shape = templates.layer_shape(2, 2, rotations='YZ')
templates.variational_layers(encoded, np.zeros(shape), wires=[3, 0], rotations='YZ',
                            entanglement='ring', entangler='CZ')
assert encoded.count_ops()['CZ'] == 2
print('Weight shape:', shape)
print(encoded.draw())
'''),
md('''
## Reupload data and check a gradient

Each layer encodes the same features, then applies trainable rotations and
entangling edges. Binding and parameter-shift gradients use the ordinary
Circuit API. `wrt` selects weights while the feature bindings stay fixed.
'''),
code('''
features = [Parameter('x0'), Parameter('x1')]
weights = np.array([Parameter(f'w{i}') for i in range(np.prod(shape))],
                   dtype=object).reshape(shape)
model = templates.data_reuploading(Circuit(2), features, weights,
                                  rotations='YZ', entangler='CZ')
values = {'x0': .2, 'x1': -.4, **{f'w{i}': .1 for i in range(weights.size)}}
function = QuantumFunction(model, 'ZI')
value, gradients = function.value_and_grad(values, wrt=['w0'])
plus, minus = dict(values), dict(values)
plus['w0'] += 1e-6; minus['w0'] -= 1e-6
finite_difference = (function(plus)-function(minus))/2e-6
assert np.isclose(gradients['w0'], finite_difference, atol=1e-8)
print('Expectation:', value, 'weight gradient:', gradients['w0'])
print('Depth:', model.depth(), 'gate counts:', model.count_ops(gates_only=True))
model.bind(values).draw('svg')
'''),
md('''
## Optional PNG/PDF export

PNG/PDF convert the same native SVG using the visualization extra and system
Cairo. This example uses a temporary folder; choose your own filename to keep
an export. Text/SVG work without Cairo. Logical depth counts dependencies,
not drawing columns or hardware duration.
'''),
code('''
import importlib.util
import tempfile
from pathlib import Path
if importlib.util.find_spec('cairosvg') is None:
    print("Optional PNG/PDF export: install this checkout with '.[visualization]'")
else:
    with tempfile.TemporaryDirectory() as folder:
        png = q.draw('png', filename=Path(folder)/'circuit.png')
        pdf = q.draw('pdf', filename=Path(folder)/'circuit.pdf')
        assert png.startswith(b'\\x89PNG') and pdf.startswith(b'%PDF-')
        assert (Path(folder)/'circuit.png').read_bytes() == png
        print('PNG and PDF exports passed:', len(png), len(pdf), 'bytes')
'''),
md('''
**Try it:** choose explicit entangling edges, add a data-reuploading layer, and
compare depth, gate counts and a finite-difference weight gradient. See the
[independent benchmark protocol](../benchmarks/README.md) for SDK comparisons.

Connection များနှင့် layer အရေအတွက်ကို ပြောင်းကာ circuit depth၊ gate count နှင့်
gradient ရလဒ်များကို နှိုင်းယှဉ်ကြည့်ပါ။
''')]))


SOURCE_API_SETUP=code('''
import sys
import nexqus
print('Kernel Python:', sys.executable)
print('Imported NexQus:', nexqus.__file__)
required = ('QuantumDistribution', 'QGAN', 'distribution_metrics', 'PauliHamiltonian')
missing = [name for name in required if not hasattr(nexqus, name)]
if missing:
    raise ImportError(
        'This kernel imported an older NexQus package without: ' + ', '.join(missing) + '. '
        'Install the current source into this notebook kernel with %pip install --upgrade '
        '"nexqus-sdk[tutorial] @ https://github.com/Inn0m1n4te030/NexQus-SDK/archive/refs/heads/main.zip" '
        'or %pip install -e "/path/to/NexQus-SDK[tutorial]", then restart the kernel and Run All. '
        'The published 1.2.0 wheel lacks these additions; version 1.2.0 alone is not a feature check. '
        'QGAN and this fixed-H2 lesson do not require the chemistry extra or PySCF.')
''')


BOOKS.append(('08_qgan_distribution', [
md('''
# 08 · Explainable QGAN / Quantum generator သင်ယူပုံ

Install the current source checkout with `python -m pip install -e ".[tutorial]"`.
The published 1.2.0 wheel does not include these APIs. Restart your notebook
kernel after installing. The first cell shows the Python and package paths
and checks that the new APIs are available.

QGAN works on native Windows without PySCF. Do not add the chemistry extra
for this lesson. PySCF requires Linux, macOS or WSL on Windows.
Everything runs locally without a QPU.

The quantum generator produces a Born probability distribution. A classical
discriminator tries to distinguish its outcomes from the target. We alternate
five discriminator updates with one generator update. This is adversarial
training, not direct minimization of distribution error.

Quantum circuit မှ probability များ ထုတ်ပေးပြီး classical discriminator နှင့်
အပြိုင် သင်ယူပါသည်။ အောက်တွင် classical model နှင့်လည်း နှိုင်းယှဉ်ထားပါသည်။
'''), SOURCE_API_SETUP, SETUP,
code('''
from nexqus import QuantumDistribution, QGAN, Adam, distribution_metrics
theta = Parameter('theta')
generator = QuantumDistribution(Circuit(1).ry(theta, 0))
model = QGAN(generator, {'theta': 1.3})
target = np.array([.25, .75])
initial = model.probabilities()
model.fit(target, epochs=160)
final = model.probabilities()
print('Initial:', initial, 'Final:', final)
print(distribution_metrics(target, final))
assert distribution_metrics(target, final)['total_variation'] < .03
'''),
md('''
## Compare against classical, then inspect the generator

A two-category classical model can represent this distribution directly with
probabilities `[1-p, p]`. The target itself is therefore an exact classical
representability reference, not a trained performance comparison. The full
`examples/qgan_distribution.py` example trains both quantum and classical GANs
with identical initial distributions, discriminator initialization, learning
rates and update counts across three seeds; it reports every result.

`jacobian['theta'][i]` tells us the local change in probability of outcome i
per radian, with the current circuit and all other weights fixed. It is not
a global causal explanation or a feature attribution method.
'''),
code('''
p, jacobian = generator.value_and_grad(model.parameters)
fig, axes = plt.subplots(1, 2, figsize=(9, 3), layout='constrained')
x = np.arange(2)
axes[0].bar(x-.15, target, width=.3, label='Target / exact classical representation')
axes[0].bar(x+.15, p, width=.3, label='Quantum GAN')
axes[0].set_xticks(x, ['0', '1']); axes[0].set_ylabel('Probability'); axes[0].legend(fontsize=7)
axes[1].plot([r['total_variation'] for r in model.history])
axes[1].set(xlabel='Adversarial update', ylabel='Total variation error')
plt.show()
print('Probability derivatives:', jacobian)
assert abs(jacobian['theta'].sum()) < 1e-10
'''),
code('''
from IPython.display import display
display(generator.template.explain(bindings=model.parameters,
    parameter_roles={'theta': 'weight'}, title='QGAN · inspect the trained generator'))
''')]))

BOOKS.append(('09_h2_energy_explained', [
md('''
# 09 · Explain H₂ energy / မော်လီကျူး စွမ်းအင်ကို လေ့လာခြင်း

**Unreleased source-checkout lesson.** Install this checkout with the tutorial
extra: `python -m pip install -e ".[tutorial]"`, then restart the notebook kernel.
This fixed-geometry lesson works on native Windows and does not need PySCF.
To regenerate integrals and compare a bond-length curve against independent
PySCF FCI, use Linux, macOS or WSL on Windows, install the core first, then add
`python -m pip install -e ".[chemistry]"` and run `python examples/h2_vqe.py`.
PySCF does not support native Windows.

The coefficients below were generated with PySCF 2.14.0, RHF/STO-3G, neutral
singlet H₂ at 0.735 Å. They use Jordan–Wigner mapping and interleaved alpha/beta
spin orbitals in q0-first order. Total energies include nuclear repulsion.
The saved classical FCI reference is -1.1373060357534004 Hartree.

H₂ ကိုသာ အခြေခံထားသော ဥပမာဖြစ်ပြီး molecule အကြီးများ သို့မဟုတ် ဆေးဝါး၏
ထိရောက်မှုကို တွက်ချက်ထားခြင်း မဟုတ်ပါ။
'''), SOURCE_API_SETUP, SETUP,
code('''
from nexqus import PauliHamiltonian, Adam
from nexqus.chemistry import h2_pair_ansatz
terms = [
    (-.09057898608834797, 'IIII'), (-.22575349222402394, 'IIIZ'),
    (-.22575349222402394, 'IIZI'), (.17464343068300442, 'IIZZ'),
    (.1721839326191555, 'IZII'), (.1209126326177663, 'IZIZ'),
    (.16614543256382414, 'IZZI'), (-.04523279994605785, 'XXYY'),
    (.04523279994605785, 'XYYX'), (.04523279994605785, 'YXXY'),
    (-.04523279994605785, 'YYXX'), (.1721839326191555, 'ZIII'),
    (.16614543256382414, 'ZIIZ'), (.1209126326177663, 'ZIZI'),
    (.16892753870087907, 'ZZII')]
hamiltonian = PauliHamiltonian(4, terms)
ansatz = h2_pair_ansatz(Parameter('theta'))
energy = hamiltonian.function(ansatz)
parameters = {'theta': 0.}; optimizer = Adam(.04); energies = []
hf_energy = energy(parameters)
for epoch in range(180):
    value, gradient = energy.value_and_grad(parameters)
    energies.append(value)
    parameters = optimizer.step(parameters, gradient)
vqe_energy = energy(parameters)
fci_energy = -1.1373060357534004
print('Classical Hartree–Fock:', hf_energy)
print('Classical FCI reference:', fci_energy)
print('NexQus VQE:', vqe_energy)
assert abs(hf_energy - (-1.116998996754004)) < 1e-8
assert abs(vqe_energy-fci_energy) < 2e-7
'''),
md('''
## What is learned, and where does the energy come from?

The ansatz output is `cos(theta/2)|1100> + sin(theta/2)|0011>`.
It mixes a doubly occupied bonding orbital with a doubly excited determinant.
Every final state contains two electrons. This is a specialized H₂ family,
not a general-purpose chemistry ansatz.

The gate decomposition can temporarily leave the two-electron sector.
Intermediate Bloch states are circuit internals, not a molecular reaction
path. The explorer's `ZZII` readout is a single Pauli observable, not the
whole molecular energy. The plot below explicitly sums all energy terms.
'''),
code('''
result = Simulator().run(ansatz.bind(parameters), shots=0, retain_state=True)
contributions = [c*result.expectation(word) for c, word in hamiltonian.terms]
assert abs(sum(contributions)-vqe_energy) < 1e-10
fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
axes[0].plot(energies, label='NexQus VQE')
axes[0].axhline(fci_energy, linestyle='--', label='Classical FCI')
axes[0].set(xlabel='Update', ylabel='Total energy / Hartree'); axes[0].legend()
axes[1].barh([word for _, word in hamiltonian.terms], contributions)
axes[1].set_xlabel('Energy contribution / Hartree')
plt.show()
from IPython.display import display
display(ansatz.explain(bindings=parameters, observable='ZZII',
    parameter_roles={'theta': 'weight'}, title='H₂ · determinant mixing'))
''')]))


def main():
    directory=ROOT/'notebooks'; directory.mkdir(exist_ok=True)
    for name,cells in BOOKS:
        content=[]
        for index,(kind,text) in enumerate(cells):
            cell=dict(cell_type=kind,id=hashlib.sha256(f'{name}-{index}'.encode()).hexdigest()[:12],
                      metadata={},source=text+'\n')
            if kind=='code': cell.update(execution_count=None,outputs=[])
            content.append(cell)
        notebook=dict(cells=content,metadata=dict(kernelspec=dict(display_name='Python (NexQus)',language='python',name='nexqus'),
                        language_info=dict(name='python',version='3.10'),
                        nexqus=dict(lesson=name,sdk_version='1.2.0')),nbformat=4,nbformat_minor=5)
        (directory/(name+'.ipynb')).write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    # Keep the terminal comparison and lesson 04 on exactly the same code.
    learning_cells = next(cells for name, cells in BOOKS if name == '04_gradients_and_learning')
    training = next(text for kind, text in learning_cells
                    if kind == 'code' and text.startswith('from time import perf_counter'))
    plotting = next(text for kind, text in learning_cells
                    if kind == 'code' and text.startswith('colors ='))
    export = dedent('''
if args.output_dir is not None:
    import csv
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir/'comparison.png', dpi=160, bbox_inches='tight')
    fig.savefig(args.output_dir/'comparison.svg', bbox_inches='tight')
    payload = dict(sdk_version=nexqus.__version__, backend='numpy', device='cpu',
                   epochs=epochs, learning_rate=learning_rate, seeds=[7, 21, 42],
                   training_samples=len(train_x), held_out_samples=len(test_x),
                   evaluation='disjoint in-domain interpolation', records=records)
    (args.output_dir/'metrics.json').write_text(json.dumps(payload, indent=2)+'\\n', encoding='utf-8')
    with (args.output_dir/'predictions.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['x', 'target', *names])
        writer.writerows(zip(grid, np.sin(grid), *(examples[n][0] for n in names)))
    print('Saved plots, metrics and predictions to', args.output_dir)
''').strip()
    plotting = plotting.replace('plt.tight_layout(); plt.show()',
                                'plt.tight_layout()\n'+export+'\nplt.show()')
    header = dedent('''
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
        args = parser.parse_args()
    ''').strip()
    (ROOT/'examples'/'compare_sine.py').write_text(
        '\n\n'.join([header, SETUP[1], training, plotting])+'\n', encoding='utf-8')
    print(f'Generated {len(BOOKS)} notebooks in {directory}')


if __name__=='__main__': main()
