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
assert tuple(map(int, nexqus.__version__.split('.')[:2])) >= (0, 6), 'Install this SDK with: python -m pip install -e .'
print('NexQus SDK:', nexqus.__version__)
''')


BOOKS=[
('00_start_here',[
md('''
# 00 · Your first NexQus circuit / စတင်အသုံးပြုခြင်း

**Goal:** install NexQus, run a Bell circuit, and distinguish probabilities from sampled counts.
**Time:** about 10 minutes. No physics or GPU background is required.

NexQus ဖြင့် မိမိကွန်ပျူတာပေါ်တွင် quantum circuit ကို simulation လုပ်နိုင်ပါသည်။
စတင်ရန် quantum hardware သို့မဟုတ် GPU မလိုအပ်ပါ။

Open a terminal in the extracted SDK folder, activate your environment, and run:

```bash
python -m pip install --upgrade -e '.[tutorial]'
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
**Try it:** remove CX and inspect the bitstrings. Then change `shots` from 1000 to 100.
Predict which probabilities change and which differences are only sampling variation.
''')]),
('01_qubits_and_gates',[
md('''
# 01 · Qubits, gates and entanglement / Qubit နှင့် gate များ

**Goal:** connect amplitudes, rotations and correlated measurements. About 15 minutes.
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

**Goal:** differentiate an expectation and train a small circuit. About 20 minutes.
Parameter-shift changes one gate occurrence at a time and applies the arithmetic chain rule.
This SDK API is explicit optimization, not PyTorch autograd.

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
x, weight, bias = Parameter('x'), Parameter('weight'), Parameter('bias')
template = Circuit(1).ry(weight*x + bias, 0)
function = QuantumFunction(template, 'Z')
model = QuantumRegressor(function, ['x'], {'weight': .7, 'bias': -.8})
train_x = np.linspace(-np.pi, np.pi, 20); train_y = np.sin(train_x)
initial_loss = float(np.mean((model.predict(train_x) - train_y)**2))
# Adam ဖြင့် weight နှင့် bias ကို train ပါ။
model.fit(train_x, train_y, epochs=80, optimizer=Adam(lr=.05))
grid = np.linspace(-np.pi, np.pi, 101); predictions = model.predict(grid)
grid_mse = float(np.mean((predictions-np.sin(grid))**2))
print('Initial training MSE:', initial_loss)
print('Final evaluation-grid MSE:', grid_mse)
print('Learned parameters:', model.parameters)
assert grid_mse < 1e-4
'''),
code('''
fig, axes = plt.subplots(1, 2, figsize=(9, 3))
axes[0].semilogy([r['loss'] for r in model.history]); axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE before update'); axes[0].set_title('Training history')
axes[1].plot(grid, np.sin(grid), label='Target sin(x)')
axes[1].plot(grid, predictions, '--', label='Circuit prediction')
axes[1].set_xlabel('x'); axes[1].set_title('Dense evaluation grid'); axes[1].legend()
plt.tight_layout(); plt.show()
'''),
md('''
The circuit computes cos(weight*x+bias). Sine is representable at weight=1, bias=-pi/2.
This is a learning demonstration, not evidence of quantum advantage or performance on real datasets.
Pauli expectations lie in [-1,1], so the built-in MSE model requires scaled targets.

ဤ example သည် synthetic data ပေါ်တွင် circuit parameter သင်ယူပုံကို ပြသခြင်းဖြစ်ပါသည်။
Classical model ထက် ပိုကောင်းကြောင်း သက်သေမဟုတ်ပါ။
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
Production readiness သို့မဟုတ် quantum advantage ကို အတည်ပြုခြင်း မဟုတ်ပါ။
''')])]


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
                        language_info=dict(name='python',version='3.10'),nexqus=dict(lesson=name)),nbformat=4,nbformat_minor=5)
        (directory/(name+'.ipynb')).write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print(f'Generated {len(BOOKS)} notebooks in {directory}')


if __name__=='__main__': main()
