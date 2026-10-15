# NexQus SDK 0.1.0

A working Python prototype with NexQus's own circuit representation, local
gate application, state evolution, noise channels, and algorithms. NumPy and
optional PyTorch supply numerical tensors; Qiskit, PennyLane, and other quantum
SDKs are not execution dependencies.

## Install

Extract the ZIP, open a terminal in `nexqus-sdk`, then:

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install -e .
python examples/quickstart.py
```

Python 3.10+ and NumPy are required. To run tests without installing the package
on Linux/macOS: `PYTHONPATH=src python -m unittest discover -s tests -v`.
After installation the cross-platform command is:

```bash
python -m unittest discover -s tests -v
```

## Simple functions

```python
from nexqus import circuit, Simulator

@circuit(2)
def my_quantum_function(q, theta):
    # ပထမ qubit ကို rotate လုပ်ပြီး ဒုတိယ qubit နှင့် ချိတ်ဆက်ပါ။
    q.ry(theta, 0)
    q.cx(0, 1)

result = Simulator().run(my_quantum_function(0.8), shots=1000, seed=42)
print(result.counts)
print(result.probabilities)
print(result.expectation("ZZ"))
```

The function builds a circuit; `Simulator.run()` executes it. Ordinary Python
loops, helper functions and parameters work. Gates chain: `q.h(0).cx(0, 1)`.
These parameters are Python/NumPy values; this release does not expose autograd
or a differentiable QNN interface.

**Bit order:** q0 is the LEFTMOST, most significant bit. `Circuit(2).x(0)` returns
`10`, not `01`. This convention differs from some quantum frameworks. Wire
order supplied to custom gates and marginals is respected exactly.

## Simulation modes

| Method | Quantum representation | Noise | Storage scaling |
|---|---|---|---|
| `statevector` | Pure state | Rejects quantum noise | O(2^n) |
| `density_matrix` | Mixed state | Exact channel evolution | O(4^n) |
| `trajectory` | Independent stochastic pure states | Ensemble estimate | O(2^n), sequential trajectories |
| `auto` (default) | Selects statevector / density matrix | Density matrix when noise present | Depends on selected method |

“Exact” means deterministic evolution under the specified model, subject to
floating-point error. Finite-shot counts still fluctuate. Trajectory
probabilities are estimates even if `shots=0`; increase `trajectories` to reduce
Monte Carlo error. They are computed by averaging branch probabilities, not by
using one trajectory for each output shot.

```python
from nexqus import Simulator, Noise, NoiseModel, algorithms

model = NoiseModel(Noise.depolarizing(0.02), Noise.amplitude_damping(0.01))
result = Simulator().run(algorithms.bell(), noise=model, seed=7)

estimate = Simulator(method="trajectory").run(
    algorithms.ghz(10), noise=model, trajectories=1000, shots=1000, seed=7,
)
```

Global rules apply each channel, in insertion order, after each unitary/diagonal
operation, independently to every wire touched by that operation. A CX therefore
receives a channel on each of its two wires. Gate-specific rules:

```python
model = NoiseModel().after(["CX", "CZ"], Noise.depolarizing(0.03))
```

Custom gate labels must match the rule names exactly (e.g. `P`, `oracle`,
`controlled_power`); inverse gates have a `†` suffix. Compose before applying a
noise model if you need to choose rules for the final gate names.

Explicit noise can be inserted at a selected point:

```python
from nexqus import Circuit
q = Circuit(1).x(0).noise(Noise.amplitude_damping(0.2), 0)
```

Supported channels: bit flip, phase flip, depolarizing, amplitude damping, phase
damping, and validated custom one-qubit Kraus operators:

```python
import numpy as np
p = 0.1
custom = Noise((np.sqrt(1-p)*np.eye(2), np.sqrt(p)*np.array([[0,1],[1,0]])))
```

Channel conventions:

- Depolarizing: E(rho) = (1-p)rho + p I/2 on the affected qubit. Its Pauli-error
  probability is 3p/4; other libraries may use a different parameter convention.
- Amplitude damping: excited-state population decays by gamma.
- Phase damping: coherences multiply by sqrt(1-gamma), populations unchanged.
- `readout_error=p` independently flips each classical output bit with
  probability p. It changes reported probabilities/counts, not retained quantum
  states or their expectations.

No gate-duration, thermal-relaxation, correlated multi-qubit noise, pulse-level
physics, or hardware-calibrated model is implied by these simple channels.

## CPU and GPU

```python
cpu = Simulator()                           # NumPy CPU
torch_cpu = Simulator(backend="torch")      # optional PyTorch CPU
gpu = Simulator(backend="torch", device="cuda:0", precision="single")
```

Install a CUDA-compatible PyTorch build using the official selector:
https://pytorch.org/get-started/locally/ . The optional `pip install -e '.[gpu]'`
extra installs PyTorch, but the installed build must support your CUDA system.
Confirm before running:

```bash
python -c "import torch; print(torch.cuda.is_available())"
python examples/gpu.py
```

CUDA execution applies NexQus's tensor operations on the selected GPU. An
unavailable GPU raises an error; it does not silently switch to CPU. Only CPU
and NVIDIA CUDA devices are supported in this release. Result arrays are copied
to CPU NumPy arrays for a consistent API. Trajectory branch selection and final
shot sampling use a CPU random generator, so GPU trajectories incur host
synchronization. Small circuits may be faster on CPU.

The current delivery environment has NumPy but no PyTorch or CUDA hardware.
CPU numerical tests passed; optional Torch CPU and CUDA parity tests are
included but skipped here. Run those tests on your intended machine before
relying on GPU results.

## Algorithms

```python
import numpy as np
from nexqus import algorithms

algorithms.bell()
algorithms.ghz(5)
algorithms.qft(4)
algorithms.qft(4, inverse=True)
algorithms.grover(3, marked="101")
algorithms.grover(4, marked=["0011", "1100"], iterations=2)
algorithms.bernstein_vazirani("1011")
algorithms.deutsch_jozsa([0, 1, 0, 1])
algorithms.phase_estimation(np.diag([1, np.exp(2j*np.pi*0.375)]), counting_qubits=3)
algorithms.qaoa_maxcut(3, edges=[(0,1), (1,2)], gammas=[0.7], betas=[0.3])
```

- Bell/GHZ: state preparation and entanglement demonstrations.
- QFT/iQFT: gate-decomposed transform with output swaps; positive-exponent QFT.
- Grover: explicit marked bitstrings and a simulated phase oracle. Building the
  length-2^n oracle is classical work; this is not a quantum-speedup demonstration.
- Bernstein–Vazirani: recover the hidden bitstring from the first n wires; last
  wire is an ancilla. Use `result.marginal(range(n))`.
- Deutsch–Jozsa: promised constant/balanced phase oracle, no ancilla.
- Phase estimation: single-qubit unitary target, optional prepared target state,
  n-bit counting register, inverse QFT. Last wire is the target. Read the first
  n wires using `marginal()`.
- QAOA: unweighted MaxCut circuit builder. It does not optimize angles for you.
  Graph cost and mixer are documented in its docstring.

Grover and Deutsch–Jozsa oracles are logical diagonal gates. Noise applied once
after these logical operations is not equivalent to noise on a hardware-level
oracle decomposition. QAOA uses decomposed CX/RZ/CX operations instead.

## Custom gates, composition, initial states

```python
from nexqus import Circuit
q = Circuit(3)
q.unitary(np.array([[0,1],[1,0]]), wires=2, name="my_X")
q.compose(algorithms.bell(), wires=(2,0))
inverse = q.inverse()    # unitary circuits only

vector = np.array([1,2,3,4,0,0,0,0], complex)
vector /= np.linalg.norm(vector)
r = Simulator().run(algorithms.qft(3), initial_state=vector, shots=0)
```

Inputs are checked for finite values, shape, normalization, unitarity, and
trace preservation. `initial_state` accepts pure statevectors only; direct
mixed-state initialization is not included. Arbitrary dense custom operators
cost O(4^k) storage for k target wires. Prefer local gates and diagonal operators.

Access to simulated complex amplitudes is different from hardware measurement:
`statevector` is available in noiseless mode when retained, `density_matrix` in
density mode, and neither is returned for trajectory ensembles.

## Results and runtime reporting

- `probabilities`: vector in q0-first basis order, including readout noise.
- `counts`: terminal all-qubit samples; use `shots=0` to avoid sampling.
- `most_likely`: highest-probability bitstring (ties use the lowest index).
- `marginal(wires)`: reduced observed probabilities in the requested wire order.
- `expectation("XYZ...")`: Pauli expectation using the retained quantum state.
- `metadata`: actual device/method, precision, seed, circuit operation count,
  wall-clock evolution time, process CPU time, state-storage bytes and estimated
  working storage. CUDA adds allocated bytes at end when available.

`simulation_seconds` includes evolution, first-use gate tensor transfers, and
probability/state transfer; excludes gate construction, backend initialization,
and final shot sampling. Process CPU
time is not CPU utilization percentage. CUDA allocation at end is not peak GPU
memory or GPU utilization. Use system profilers if those metrics are needed.

## Benchmarks and memory

```bash
python examples/benchmark.py --qubits 16 --repeat 3
python examples/benchmark.py --qubits 6 --noise 0.01 --method density_matrix
python examples/benchmark.py --qubits 10 --noise 0.01 --method trajectory --trajectories 500
python examples/benchmark.py --backend torch --device cuda:0 --qubits 20 --precision single
```

Compare matched qubit count, circuit, method, precision, and trajectory count;
the script performs a warmup and reports median repeated evolution time.

| Qubits | Statevector complex128 | Density matrix complex128 |
|---|---:|---:|
| 10 | 16 KiB | 16 MiB |
| 12 | 64 KiB | 256 MiB |
| 16 | 1 MiB | 64 GiB |
| 20 | 16 MiB | 16 TiB |

This table describes the main state only, not all working memory. Default
`max_memory_mb=1024` checks a conservative six-state working estimate before
simulation. It is not a hard process/GPU allocator limit and excludes circuit
data, custom operator caches, and retained host arrays. Increase it deliberately
or use trajectory mode for noisy circuits that would need large density matrices.

## Current scope and next engineering work

Working prototype: synchronous local simulation, terminal measurement, sequential
trajectories, custom unitary gates, reusable algorithms, and Burmese comments in
selected examples. It has no mid-circuit measurement/feed-forward, reset,
distributed execution, fused CUDA kernels, native compiler, automatic quantum
gradients, cloud hardware connector, or production performance guarantee.

Next development candidates: validate/benchmark GPU parity on NexQus hardware,
add differentiable parameter handling, batched trajectories, mid-circuit
measurement, more algorithms, and a curated Burmese tutorial series. These are
future work, not shipped features.

No distribution license was selected for this private prototype. Choose the
intended open-source or commercial licensing before publishing.

## Numerical references

These official sources informed channel/device conventions; their simulator
implementations are not dependencies of this engine:

- IBM Quantum quantum information states/channels:
  https://quantum.cloud.ibm.com/docs/en/api/qiskit/quantum_info
- PyTorch complex tensors:
  https://docs.pytorch.org/docs/stable/complex_numbers.html
- PyTorch CUDA availability:
  https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_available.html
