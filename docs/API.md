# NexQus public API reference

Applies to NexQus SDK 1.1.0; additions for 1.1 are marked below. Read API_STABILITY.md for the supported 1.x
contract. Internal tensor adapters and direct operation-list mutation are not
supported public APIs.

Numerical angle labels, inspection, PNG/PDF export and `nexqus.templates` below
are **unreleased source-checkout additions**. Install this checkout with
`python -m pip install -e '.[tutorial,visualization]'` to try them; the existing
PyPI 1.1.0 archives do not contain these additions.

## Circuit construction

`Circuit(n_qubits)` requires a positive integer. Wire 0 is the leftmost/MSB bit;
wire lists must be nonempty, distinct, in range, and are interpreted in the order
supplied. Angles and bindings are finite real scalars in radians. Construction
records operations; it does not execute a quantum state.

| Method | Purpose |
|---|---|
| h(q), x(q), y(q), z(q) | Single-qubit gates |
| s(q), t(q), phase(theta,q) | Phase gates; S and T are P aliases with gate label P |
| rx(theta,q), ry(theta,q), rz(theta,q) | Pauli rotations exp(-i theta P/2) |
| cx(control,target), cz(control,target), cp(theta,control,target) | Controlled gates |
| swap(a,b) | Swap two distinct qubits |
| unitary(matrix,wires,name='U') | Finite 2^k square validated unitary |
| diagonal(values,wires,name='diagonal') | 2^k unit-modulus entries; tolerance atol=rtol=1e-10 |
| permutation(mapping,wires,name='permutation') | 1.1: basis map x → mapping[x], storing 2^k integer indices |
| controlled(matrix,control,target,name='controlled-U') | Controlled single-qubit 2×2 unitary |
| noise(channel,q) | Insert a one-qubit Noise channel at this point |
| measure(q,key) | Ideal Z measurement, overwriting a named classical bit |
| reset(q) | Ideal reset to zero, discarding the prior value |
| c_if(key,value=1) | Condition the immediately preceding gate, reset or noise on an earlier measured bit |
| compose(other,wires=None,classical_map=None) | Append a block with optional wire/key remapping |
| inverse() | New inverse circuit; rejects noise and dynamic operations |
| bind(mapping) | New concrete circuit; rejects missing/unknown names |

Most mutating construction methods return self for chaining. `inverse()` and
`bind()` return new circuits. `compose()` with no wire list uses all destination
wires and therefore requires matching circuit widths. Key names collide and
overwrite intentionally unless renamed; classical_map cannot merge distinct
source keys. Self-composition snapshots the operations being appended.

Properties: n_qubits, parameters (sorted names), classical_keys (first-measurement
order), dynamic (contains measurement/reset/conditions). The operations collection
is inspectable; do not mutate it or construct Operation objects directly.

`@circuit(n_qubits)` decorates `def build(q, ...)` as an ordinary Python circuit
factory. It supports Python helper functions and loops. It is not a runtime
compiler or an automatic differentiation decorator.

## Parameters

`Parameter(name)` supports addition, subtraction, multiplication, division and
unary negation, including constants on either side. Symbolic angles are supported
by rx/ry/rz/phase/cp. Custom unitary matrices are not symbolically parameterized.
Bindings accept name strings or Parameter keys; repeated parameter names share a
value. A mapping must supply exactly the required finite real values.

`Simulator.run_sweep(template,bindings,seed=None,**run_options)` returns ordered
results. `iter_sweep` yields one result at a time. Both are sequential. The default
is retain_state=False to limit accumulated memory; pass True for expectations.
Child seeds are deterministically derived from the base seed. Result metadata
includes sweep_index and the concrete bindings.

## Simulator and execution

```python
Simulator(backend='numpy', device='cpu', method='auto', precision='double',
          max_memory_mb=1024, gate_cache_mb=64, local_gates=True)

sim.run(circuit, shots=1024, seed=None, noise=None, trajectories=256,
        initial_state=None, retain_state=True, readout_error=0.0, batch_size=1,
        random_engine='portable', noise_fastpath=True, max_branches=1024)
```

- backend: numpy or optional torch. Devices: cpu, cuda or cuda:N with Torch.
  NumPy accepts only cpu. A missing device/backend raises instead of falling back.
- precision: double (complex128) or single (complex64).
- method: auto, statevector, density_matrix or trajectory. See README for storage
  scaling. Dynamic auto/density runs report method=branch_density in metadata.
- shots: nonnegative integer; zero skips terminal draws. seed is ordinarily a
  nonnegative integer or None. Report all stochastic settings for reproducibility.
- trajectories and batch_size: positive integers. Effective batch is capped at
  the trajectory count. Final partial batches are supported.
- initial_state: finite normalized pure vector of length 2^n in q0-first order.
  Squared norm must be within 1e-10 of 1; accepted roundoff is rescaled. No mixed
  input density is accepted. The caller's array is not modified.
- retain_state: keep a final pure or density state for exact runs; trajectory
  ensembles return neither representation. Counts do not collapse the retained
  state. Dynamic exact output retains the unconditional final density.
- readout_error: independent probability of flipping each terminal quantum bit.
  Intermediate classical records and control are not corrupted by this option.
- random_engine: portable paired CPU/CUDA stream or native backend stream. Native
  engines need not match across backends. Neither is a cryptographic interface.
- max_branches: bounds exact classical/density branching; the memory estimate may
  reject a smaller case. There is no silent conversion to trajectories.
- local_gates and noise_fastpath: diagnostic optimization switches. Static noise
  fast paths recognize actual Kraus values. Dynamic sampling currently uses
  generic Kraus kernels; noise_fastpath does not enable a dynamic specialization.

max_memory_mb checks an estimated core workspace before allocation and branch
growth. It does not cap all RAM/VRAM: circuit matrices, tensor caches, host results
and output shots are additional. Increasing qubits, density branches, batch size
or retained sweep results can raise memory substantially.

`sim.compile(circuit,noise=None)` snapshots a bound circuit and noise, prepares
device tensors, and returns CompiledCircuit. Use prepared.run(**options) or
sim.run(prepared,**options). Noise is frozen at compilation; recompile to change it.
A prepared object belongs to its creator. This is not gate fusion/native code.

`sim.cache_info()` reports entries/bytes/limit/hits/misses for the bounded LRU.
`sim.clear_cache()` resets that cache; live compiled objects keep their own tensor
references. gate_cache_mb=0 disables persistent caching, not all per-run tensors.

## Noise

`Noise(kraus,name='custom')` validates a finite one-qubit CPTP channel:
sum(K†K)=I within atol=rtol=1e-10. Built-ins accept probabilities in [0,1]:
bit_flip(p), phase_flip(p), depolarizing(p), amplitude_damping(gamma), and
phase_damping(gamma). Depolarizing means (1-p)rho+p I/2 on the affected qubit;
phase-damping coherences scale by sqrt(1-gamma).

`NoiseModel(*channels)` applies channels after each executed gate on each touched
wire, in insertion order. `model.after(gates,channel)` adds a rule: gates may be a
name, iterable of names or None for all gates. Labels match exactly, including
custom names. Inverse gates append †. S/T use P. Explicit noise, reset and
measurement do not automatically trigger post-gate rules. A skipped conditional
gate receives no post-gate noise.

Noise models are logical channel models, not pulse physics or calibrated device
models. Noise after a logical oracle is not equivalent to noise after every gate
in a hardware decomposition of that oracle.

## Results

Result has probabilities, counts, metadata, statevector, density_matrix,
classical_keys, classical_probabilities, classical_counts and joint_counts.

- probabilities is a host NumPy vector in computational-basis order, including
  terminal readout error. Counts sum to shots when shots>0.
- most_likely returns the highest-probability bitstring; ties use the lowest index.
- marginal(wires) returns observed marginal probabilities in requested wire order;
  a scalar wire is accepted. reduced_density(wires) needs a retained quantum state
  and excludes readout noise.
- expectation(pauli) requires a retained state, with one I/X/Y/Z character per
  qubit. The value is a real Pauli expectation, not a finite-shot estimator.
- classical_keys follows first measurement order. Probabilities describe the final
  register, not every overwritten event. Joint samples use classical|quantum;
  classical_counts and counts are consistent marginals of joint_counts.

Core metadata includes n_qubits, method, backend, device, precision, seed, shots,
trajectories, batch_size, probability_kind, readout_error, operations, compiled,
compile_seconds, simulation_seconds, process_cpu_seconds, state_storage_bytes,
estimated_working_bytes, gate_cache_bytes and random_engine. CUDA/dynamic fields
are conditional; tolerate missing optional and newly added fields.

Timing includes quantum evolution and result transfers. Static/exact timings
exclude final draws; dynamic trajectory timing includes correlated sample
reporting. Backend creation, Python circuit construction and compile time are
separate. Process CPU time is not CPU utilization. These metrics are useful for
matched experiments, not universal backend speed claims.

## Gradients and learning

`QuantumFunction(template,observable,simulator=None,noise=None)` returns a scalar
Pauli expectation when called with bindings. `.gradient(bindings,wrt=None)` and
`.value_and_grad(bindings,wrt=None)` return a gradient dict or (value,dict).
Differentiate by gate occurrence, sum shared parameters, and apply arithmetic
chain rules. A value plus gradient costs one circuit plus two executions per
active supported gate. Noise is independent of symbolic angles. Dynamic controls,
trajectory gradients, Torch autograd and neural-network modules are unsupported.

`QuantumRegressor(function,input_names,initial_parameters)` supplies inputs from
X columns and trains all remaining named parameters. `.fit(X,y,epochs=60,
optimizer=None,callback=None)` uses full-batch MSE; default Adam. Targets must be
real, finite and in [-1,1]. `.predict(X)` returns expectations. `history` records
loss/parameters before each update; `parameters` contains the final updated values.

`QuantumBinaryClassifier` accepts labels 0/1, maps them to -1/+1 and trains MSE.
predict returns labels, predict_expectation returns raw expectation values, and
predict_proba maps expectations to two probability-like scores. They are not
guaranteed calibrated probabilities. Only binary classification is supported.

`SGD(lr=.05)` and `Adam(lr=.05,beta1=.9,beta2=.999,epsilon=1e-8)` use
step(parameters,gradients), returning a new parameter dictionary. Values must be
finite real scalars and keys must match. Reuse Adam only for the same parameter
names; call reset() for a new optimization. A rejected Adam step preserves state.

## Algorithms

bell(), ghz(n_qubits), qft(n_qubits,inverse=False),
grover(n_qubits,marked,iterations=None), bernstein_vazirani(secret),
deutsch_jozsa(truth_table), phase_estimation(unitary,counting_qubits=3,
target_state=None), qaoa_maxcut(n_qubits,edges,gammas,betas), teleportation().

QFT uses a positive exponent and output swaps. Grover and Deutsch–Jozsa construct
explicit logical diagonal oracles; oracle construction is classical work. QPE
uses a one-qubit target; inspect the leading counting register with marginal().
QAOA builds an unweighted MaxCut ansatz and leaves optimization to the caller.
Teleportation assumes input on q0 and zero q1/q2, with output on q2 and classical
keys alice_z/alice_x. Rename keys when composing repeated protocol blocks.

### Shor factoring (added in 1.1)

`order_finding(base, modulus, counting_qubits=None, *, max_memory_mb=64)` returns
a Circuit. It requires modulus >= 3 and coprime 1 < base < modulus. The trailing register has
`(modulus-1).bit_length()` qubits initialized to |1>; the leading counting
register defaults to twice that width. The initial 31/10-qubit limits are removed.
Use `result.marginal(range(counting_qubits))` to inspect its phase distribution.
Each controlled modular multiplier leaves unused basis states >= modulus fixed.

`shor(modulus, *, base=None, shots=32, max_attempts=8, seed=None, simulator=None,
counting_qubits=None)`
returns an immutable `ShorResult(modulus, factors, attempts, method, base, order)`.
The factors are a sorted nontrivial pair whose product is modulus, not necessarily
prime. `factors=None` with method `failed` means attempts were exhausted, not that
the input is prime. Methods `even`, `perfect_power`, `prime`, and `gcd` identify
classical preprocessing; method `quantum` means sampled order finding supplied
a verified even order. The prime check uses trial division after resource checks.

A supplied base is reused on retries; otherwise a seeded generator selects bases.
`shots` and `max_attempts` must be positive integers. Zero phases, odd orders,
and trivial square roots may fail; changing the base or seed can help.
Continued fractions recover phase denominators, LCM combines them, and modular
exponentiation verifies candidate orders before GCD factor extraction. There is
no fallback classical factor search after unsuccessful quantum samples.

The optional simulator uses the normal NexQus execution backend; the default is
NumPy statevector. A noiseless trajectory simulator uses one trajectory because
all such histories are identical. Modular gates use integer permutation maps
and gather amplitudes without constructing dense matrices. They are logical
operations, not an elementary-gate arithmetic decomposition for quantum hardware.

`shor_resources(modulus, counting_qubits=None, *, precision='double')` returns
an immutable `ShorResources` with target/counting/total qubits, statevector bytes,
estimated state working bytes, permutation-map bytes, and estimated construction
bytes. It allocates no gates or quantum state. The default m=ceil(log2(modulus))
target qubits and 2m counting qubits require 2^(3m) amplitudes. For 77 this is
21 qubits, 32 MiB for the double-precision state and 192 MiB estimated state
working storage, plus gate storage. Density matrices cost substantially more.

`order_finding.max_memory_mb` checks estimated gate-construction storage only.
`shor` checks state working storage plus construction against the supplied
Simulator's `max_memory_mb` (default 1024 MiB) before allocating odd-input
circuits. Even inputs return their trivial factors without allocating a state.
Estimates are not hard process-RAM caps; increasing the budget does not create
physical RAM. The implementation also rejects totals above 62 qubits for index
capacity; practical statevector limits occur much earlier.

Reduce `counting_qubits` to trade phase resolution and factoring success rate for
lower memory, or explicitly configure a larger simulator budget when sufficient
RAM is available. Quantum factoring is regression-tested for 15, 21, 35 and 77;
larger values are resource-dependent rather than guaranteed to finish quickly.
The existing single-target-qubit `phase_estimation` API is unchanged.

`Circuit.permutation` validates that mapping contains every basis index exactly
once, copies it, and supports compose, inverse, binding, compilation, classical
conditions, noise, statevector/density/trajectory modes and NumPy/PyTorch.
Wire order follows the same MSB-first convention as `unitary`. The stored map
has one int64 per basis state; prepared kernels use its inverse as gather indices.

Algorithm background: [IBM Quantum, Shor’s algorithm](https://quantum.cloud.ibm.com/docs/en/tutorials/shors-algorithm).

### Circuit drawing (added in 1.1)

`Circuit.draw(output='text', *, filename=None, fold=12, precision=4)` returns a diagram without
executing the circuit. Use `print(q.draw())` in a terminal, or `q.draw('svg')` in
Jupyter. Displaying a Circuit as the last notebook expression also renders SVG.
The SVG result is a string subclass with `_repr_svg_()` support; it needs no
plotting or quantum SDK dependencies. Save with `filename='circuit.svg'`.

`fold` sets operations per panel; `None` puts all operations in one panel.
Quantum wires are ordered q0 at the top, and named classical wires appear below.
CX, CZ and SWAP have standard symbols. Measurements write to classical wires;
conditional gates connect to their classical bit with an `=0` or `=1` label.
Noise and resets are labeled explicitly. General multiwire gates use connected
boxes with `:0`, `:1`, etc. to preserve operand order, including nonadjacent wires.
Each operation has its own column; the drawer does not reschedule gates.

Symbolic gates show parameter expressions. In the development checkout, built-in
RX/RY/RZ/P/CP gates retain their supplied numerical angles: `RX(0.5)`.
`precision` is 1–16 significant digits (default 4); it changes labels only.
Binding, composition and conditions preserve labels. `RX†(0.5)` denotes the
adjoint of RX(0.5), including after symbolic binding. Custom matrices are never
assigned inferred angles. Custom names are escaped and flattened to one line.

Development outputs `png` and `pdf` require the optional `visualization` extra
and the Cairo system library. Both convert the same native SVG. PNG returns
bytes with notebook `_repr_png_()` support; PDF returns bytes. `filename` writes
binary data for these formats. No simulator or external quantum SDK is involved.
See [CairoSVG installation](https://cairosvg.org/documentation/) for platform
requirements. Text/SVG continue to work without CairoSVG or Cairo.

```python
q = Circuit(8).rx(.5, 0).cx(0, 1).ry(.8, 3).cx(3, 7)
print(q.depth())                 # 2: the two disjoint gate chains are parallel
print(q.count_ops())             # {'RX': 1, 'CX': 2, 'RY': 1}
q.draw('png', filename='circuit.png')
q.draw('pdf', filename='circuit.pdf')
```

### Logical inspection (unreleased)

`q.count_ops()` counts stored operation names, including noise, measurement and
reset. `q.count_ops(gates_only=True)` excludes those three kinds. Custom
unitaries, diagonal gates and permutations each count as one gate; there is no
decomposition. Adjoint names remain distinct. Results are fresh dictionaries.

`q.depth()` computes critical-path layers across qubit and classical data
dependencies. Each stored operation costs one layer, including noise,
measurement and reset. A conditional operation waits for its bit's last write;
overwriting a bit waits for previous readers. Independent reads can run in
parallel. Empty circuits have depth zero. This is logical depth, without device
connectivity, timing, routing, or branch-probability estimates. Drawing still
uses one column per operation and does not change execution order.

### QML templates (unreleased)

`from nexqus import templates` provides `basis_encoding`, `angle_encoding`,
`entangling_layer`, `layer_shape`, `variational_layers` and `data_reuploading`.
They append ordinary gates, return the supplied circuit, support ordered wire
subsets and validate inputs before changing the caller's circuit. See
[QML_TEMPLATES.md](QML_TEMPLATES.md) for shapes, connectivity, symbolic parameters
and a complete gradient example.

## Numerical validation helpers

`nexqus.validation.compare_results(reference,candidate,precision='double')`
raises on mismatches and otherwise returns diagnostic errors. It checks finite,
normalized probability vectors; matching retained-state presence/shape;
finite normalized vectors or Hermitian unit-trace densities; global-phase-aware
state differences; and classical probabilities/order. Positivity of arbitrary
user-created density matrices is not established by this helper; simulated
channels are CPTP and independent tests check physical examples.

Double max-probability/state tolerance is 1e-9 and probability TVD limit 1e-7;
single tolerances are 2e-5 and 1e-4. Trajectories need the same explicit seed,
portable RNG, method, count, batch size and static fast-path options. A parity
check is not an independent verification of physical correctness.

`trajectory_confidence_bound(trajectories,outcomes,alpha=.01)` returns a
conservative Hoeffding/union-bound max-probability threshold. It assumes
independent trajectories and is a statistical sanity bound, not device parity or
RNG certification. More stringent physical tests may need additional samples.
