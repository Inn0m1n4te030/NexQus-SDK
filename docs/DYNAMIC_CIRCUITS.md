# Measurement, reset and classical feed-forward

Use classical measurement outcomes to control subsequent circuit operations.
NexQus 1.1.0 supports exact and sampled dynamic simulation on CPU and CUDA.

## Install and run

From the NexQus 1.1.0 source checkout:

```bash
python -m pip install --upgrade -e .
python -m unittest discover -s tests -v
python examples/validate_gpu.py
python examples/measurement_reset.py
python examples/teleportation.py
python examples/teleportation.py --backend torch --device cuda:0
python examples/dynamic_trajectories.py --backend torch --device cuda:0
```

## Measure and condition

```python
from nexqus import Circuit, Simulator

q = Circuit(2).h(0).measure(0, 'm')
q.x(1).c_if('m', 1)
r = Simulator().run(q, shots=1000, seed=42)
print(r.classical_keys)           # ('m',)
print(r.classical_probabilities)  # {'0': 0.5, '1': 0.5}
print(r.joint_counts)             # only '0|00' and '1|11'
```

`measure(qubit, key)` is an ideal projective Z measurement of one qubit and
writes a named classical bit. It collapses the quantum state. To measure in a
different basis, rotate first. Reusing the same key overwrites its previous
value. Classical bitstrings use keys in first-measurement order; quantum strings
keep the existing q0-leftmost convention. Only final classical-register values
are returned, not a complete per-instruction measurement history.

`c_if(key, value)` conditions the immediately preceding gate, reset, or explicit
noise operation. The key must already have been measured; value must be integer
0 or 1. One condition per operation is supported. Compound Boolean expressions,
conditional measurements, arbitrary Python callbacks, loops and classical
arithmetic are not included.

Gate-level noise applies only when that conditional gate executes. Measurement
and reset do not automatically receive post-gate noise. Explicit noise can be
inserted before or after them. `readout_error` affects final quantum sampling
only; it does not corrupt intermediate classical measurement records.

## Reset

```python
q = Circuit(2).h(0).cx(0, 1).reset(0)
r = Simulator().run(q, shots=0)
print(r.reduced_density([0]))  # |0><0|
print(r.reduced_density([1]))  # I/2
```

Reset discards the selected qubit's previous state and prepares |0>. It is a
non-unitary channel. Resetting one half of an entangled pair leaves the other
half mixed; it does not restore a coherent product state. Reset does not write a
classical bit. It may itself be conditioned on an earlier measured bit.

## Exact versus sampled dynamic execution

| Requested method | Dynamic behavior |
|---|---|
| auto | Exact classical/quantum density branching |
| density_matrix | Exact classical/quantum density branching |
| trajectory | Sampled batched statevectors with one classical register per trajectory |
| statevector | Rejected; a single deterministic pure state cannot represent all branches |

The result metadata calls the exact method `branch_density`. Each branch holds
an unnormalized density matrix and a complete classical-register value.
Measurement splits branches; branches with the same final register merge by
adding density matrices. The retained density matrix is the unconditional
quantum state, summing all branches. Noisy channels use exact Kraus evolution.

Branch count can grow exponentially with independent measurements. The default
`max_branches=1024` and simulator working-memory estimate guard bound growth.
Raise limits deliberately or switch to trajectory mode. The memory estimate is
not a hard allocator limit and excludes cache/circuit/sampling bookkeeping.

```python
sim = Simulator(method='trajectory')
r = sim.run(q, trajectories=2048, batch_size=128, shots=1000, seed=42)
```

Sampled dynamic execution keeps quantum states, measurement collapse,
classical predicates and Kraus branch decisions on the chosen device. Classical
records and requested output distributions are copied at batch boundaries for
reporting. Dynamic execution uses generic
Kraus paths for noise and reset; `noise_fastpath` does not select specialized
dynamic noise kernels. There is no promise that dynamic throughput matches
static circuits. Inactive conditional rows may consume random draws but their
states are left unchanged.

## Results and correlations

- `probabilities` / `counts`: final quantum distribution and samples.
- `classical_keys`: bit order for the final classical register.
- `classical_probabilities`: exact branch weights or empirical trajectory weights.
- `classical_counts`: classical records paired with requested quantum shots.
- `joint_counts`: keys such as `01|110` mean classical 01 and quantum 110.
- `reduced_density(wires)`: partial trace of a retained state, in supplied wire order.

Exact mode samples the joint branch/quantum distribution. Sampled mode assigns
shots to sampled histories and draws terminal quantum outcomes conditional on
each history, preserving classical/quantum correlations. Increasing shots does
not increase the number of simulated intermediate histories; increase
`trajectories` to reduce that Monte Carlo error. With shots=0, distributions are
still calculated and all count dictionaries are empty.

Trajectory mode does not retain a full ensemble density matrix, so reduced
states/expectations require exact mode. Reset-only circuits have no classical
keys or classical count distributions. Shot sampling and readout errors do not
modify retained quantum states.

Exact dynamic `simulation_seconds` excludes final shot draws. Sampled dynamic
timing includes batch-end reporting and correlated shot sampling because those
occur before the batch is released. These timings are not directly identical
to the static evolution-only timing convention.

## Teleportation

```python
from nexqus import algorithms

q = Circuit(3).ry(0.83, 0).phase(-0.47, 0)
q.compose(algorithms.teleportation())
r = Simulator().run(q, seed=42)
bob = r.reduced_density([2])
```

The protocol assumes an arbitrary input on q0, with q1/q2 initially |00>.
It prepares the Bell resource, measures Alice's qubits, applies Bob's X and Z
corrections, and returns Bob's reduced state. The example checks fidelity against
the input, including a nonzero complex phase. Classical outcomes are uniform
ideally; the receiver's corrected state is independent of them.

`algorithms.teleportation()` uses keys alice_z/alice_x. Reusable protocol blocks
can rename those keys as well as remap quantum wires:

```python
q.compose(algorithms.teleportation(), wires=(0, 2, 3),
          classical_map={'alice_z': 'z2', 'alice_x': 'x2'})  # q must have >=4 qubits
```

Without a mapping, composing a block uses the same classical names; collisions
overwrite those existing bits deliberately. `inverse()` rejects dynamic/noisy
circuits. Bind and compile work with dynamic templates and preserve conditions.
Compiled objects still snapshot the source circuit/noise for one simulator.

## Learning and current limits

QuantumFunction supports static unitary and exact noisy circuits. Dynamic
QuantumFunctions are explicitly rejected; gradients
through measurements/reset/feed-forward have not been implemented.

No pulse simulation, physical measurement latency, calibrated measurement/reset
errors, branching hardware jobs or arbitrary classical programs are implied.
The implementation covers circuit-level measurement/reset and single-bit
feed-forward in local simulation.

## Burmese notes

`measure(0, 'm')` သည် qubit 0 ကို measure လုပ်ပြီး ရလဒ်ကို classical bit `m` တွင်
သိမ်းပါသည်။ `x(1).c_if('m', 1)` သည် `m` တန်ဖိုး 1 ဖြစ်မှသာ qubit 1 ကို X gate
လုပ်ဆောင်ပါသည်။

`reset(0)` သည် qubit 0 ကို |0> သို့ ပြန်ထားပါသည်။ Entangled state ဖြစ်လျှင်
အခြား qubit ၏ state ကိုလည်း မှန်ကန်စွာ mixed state အဖြစ် ထိန်းသိမ်းပါသည်။

သေးငယ်သော protocol များအတွက် default exact branching ကို သုံးနိုင်ပါသည်။
Measurement များစွာကြောင့် memory မလုံလောက်ပါက trajectory mode ကို ရွေးပါ။
