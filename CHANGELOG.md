# Changelog

## 1.3.0 — 2026-10-24

Promote the CI-validated 1.3.0rc1 candidate to a production release with no
runtime changes beyond the version identifier. The candidate passed fresh
TestPyPI installation checks. CUDA hardware validation and volunteer feedback
remain ongoing; CPU parity and skipped CUDA tests are not hardware evidence.

- Add seeded maze-run playback with a classical trail, quantum wave/coin-flow
  display, visible exit detection, step controls and explicit horizon failures.
- Add explicit CPU/Torch/CUDA circuit device selection to glued-tree and maze
  walks; disclose CPU analysis stages and synchronize GPU wall timings.
- Reproduce random-cycle glued-tree continuous-time transport with actual SDK
  column circuits, positive classical probabilities, analytic uniform-time
  quantum success, small full-graph validation, a depth sweep and an offline
  explanation. Separate the published oracle theorem from classical simulation.
- Add a monitored coined quantum walk with SDK coin/shift operations, a matched
  dephased classical walk, BFS, exact exit detection, warmed CPU timings and a
  self-contained visual report. Verify against independent matrix/Markov models;
  report finite-instance random-walk benefits without claiming a BFS speedup.
- Explain maze BFS queue visits and Grover encoding, phase marking, diffusion
  arithmetic and measurement; show actual logical operation counts and warmed
  CPU timing including classical oracle construction, separate from trace replay.
- Add a local Quantum Maze Explorer school project with editable mazes, a
  bounded junction-plan oracle, actual Grover stage traces, BFS comparison,
  seeded stage measurement, circuit views and self-contained lesson export.
- Disclose classical oracle compilation and distinguish unreachable exits from
  insufficient plan budgets; verify the algorithm and real-server UI interactions.

- Add opt-in regularized classical/hybrid image GAN training with separate rates,
  lazy R1, symmetric differentiable translations and EMA previews.
- Add fresh DCGAN spatial models and optional hybrid probability readout,
  linear angles and classical latent bypass; preserve legacy model defaults.
- Retune checkpoints without discarding learned weights/Adam state, retain
  numbered checkpoints and reject accidental fresh overwrites. Show generated
  images first with duplicate, latent and nearest-reference diagnostics.
- Test R1 scaling, native gradients, old checkpoints and exact EMA/augmented
  continuation. Pet-image quality remains unvalidated.

- Add optional convolutional classical/quantum-feature image generators and a
  conditional convolutional discriminator; retain MLP checkpoint compatibility.
- Default the image CLI to convolutional models; add epoch budgets, seeded file
  selection, resumable training-only flips, periodic samples/checkpoints and
  explicit separation of software checks from animal-image quality evaluation.

- Extend device selection across probability generators, QGAN, learning,
  Hamiltonian/VQE, bounded exact references and explanations; retain optional
  native result tensors and Pauli reductions on CPU/CUDA.
- Replace image GAN's per-sample NumPy bridge with batched Torch circuit
  autograd, CUDA model/optimizer execution and device-aware checkpoint reload.
- Add optional explicit GPU4PySCF RHF dispatch, labelled CPU integral/mapping
  stages, device diagnostics and strict real-CUDA validation tooling.

- Add conditional quantum patch and quantum-feature/classical-decoder image
  generators, a Torch/native-parameter-shift bridge, classical GAN baseline,
  image loaders, minibatch training, resumable checkpoints and pixel effects.
- Add optional image-GAN dependencies, comparison report/example, lesson 11,
  gradient/data/resume tests and Linux/Windows image-training CI.

- Add an offline native molecular viewer with H2/CH4/H2O/C2H4/C6H6 structures,
  rotation, atom selection, distance/angle measurement and SVG/XYZ/JSON export.
- Attach actual H2 VQE energies to its geometry, centralize the fixed reference
  coefficients and add Windows-compatible molecular examples and lesson 10.
- Reassemble Jupyter stdout fragments before validating the printed SDK version,
  fixing a false Windows notebook failure without weakening the version check.

- Separate native Windows QGAN/tutorial installation from the optional PySCF
  driver (Linux/macOS/WSL), add notebook API/path diagnostics and recovery
  instructions, and execute all lessons on Windows/Python 3.13 in CI.

- Add bounded exact probability Jacobians, a hybrid discrete QGAN, classical
  distribution comparisons and local probability-sensitivity visualization.
- Add weighted Pauli Hamiltonians/gradients, optional PySCF closed-shell
  integrals and Jordan–Wigner mapping, plus H₂ VQE/FCI energy curves and
  energy-term explanations. Add lessons 08/09 and independent physics tests.

- Add a bounded local pure-state execution explorer with self-contained HTML,
  gate/block navigation, probability/phase plots, reduced Bloch states,
  purity/entropy, Z covariance and observable readout traces.
- Preserve symbolic feature/weight occurrence in reports, support explicit
  QNN block annotations, and add Bell/interference/trained-QNN examples.
- Validate analytic states, trace limits, injection safety and report controls
  with core and offline DOM regressions. Noise/dynamic tracing and predictive
  feature attribution are outside this first milestone.

- Add an optional Amazon Braket bridge: offline gate translation/capability and
  cost checks, local simulation, explicit paid QPU submission, job lifecycle and
  sparse q0-first measurement results. Cloud execution is mocked, not hardware-validated.
- Keep Braket optional (Python 3.11+), block cloud requests in integration tests
  and provide a local-only example. No cloud task was submitted during validation.

## 1.2.0 — 2026-10-17

- Add visual quantum-versus-classical sine regression with prediction/loss/error
  plots, held-out evaluation, multiple seeds and a matched sinusoidal baseline.
- Repair malformed notebook JSON and update all eight tutorials for 1.2.0.

- Remove all latency metrics from correctness/smoke reports. Warm every engine
  equally for performance and sample memory in a separate fresh subprocess.
- Optimize NumPy single-wire rotations/diagonals and exact Pauli/damping noise,
  cache compiled channel recognition and avoid RNG setup for exact zero-shot runs.
- Publish a newly measured unoptimized baseline and matched optimization results.

- Retain numeric RX/RY/RZ/P/CP angle labels through binding, composition,
  conditions and inverses; expose logical depth and operation/gate counts.
- Add optional PNG/PDF export from the native SVG renderer.
- Add basis/angle encoding, configurable entangling and variational layers,
  data reuploading, symbolic gradients, examples and a bilingual notebook.
- Add independent dense-oracle correctness and CPU performance comparisons
  against optional Qiskit Aer and PennyLane development dependencies, with
  raw accuracy/timing/RSS reports and a correctness-only CI gate.

## 1.1.0 — 2026-10-16

- Add native text/SVG circuit drawing, Jupyter display, export, wrapping, and
  quantum/classical wire support without additional dependencies.

- Add quantum order finding and Shor factorization with resource estimates,
  explicit result metadata, deterministic seeds, tests, and a runnable example.
- Replace dense modular matrices and the initial 31 limit with permutation-index
  gates and memory preflight checks; validate quantum factoring through 77.
- Add Circuit.permutation across NumPy/PyTorch execution, compilation, inverses,
  density matrices, trajectories, and dynamic conditions.

## 1.0.0 — 2026-10-16

First stable public release under the MIT license.

### Features

- NumPy CPU and optional PyTorch CPU/NVIDIA CUDA simulation.
- Noiseless statevectors, exact density matrices and batched noisy trajectories.
- Circuit functions, symbolic parameters, sweeps, prepared execution and built-in algorithms.
- Parameter-shift gradients, Adam/SGD, small learning models and VQE examples.
- Mid-circuit measurement, reset, classical feed-forward and teleportation.
- Seven English/Burmese notebooks, current feature guides and the public 1.x API policy.

### Correctness and packaging

- Reject non-finite or malformed validation results and invalid state/diagonal inputs.
- Preserve single precision in NumPy damping and bound dynamic shot scheduling memory.
- Reject complex learning data; isolate training callbacks and keep Adam failures atomic.
- Keep validation reports and internal audit notes out of public guides and package archives.
- Validate wheel/source contents, clean installations and notebook execution in CI.

## Development history

## 0.6.0

Seven tested English/Burmese notebooks and a notebook execution runner. Runtime
features unchanged from v0.5 except version identifier.

## 0.5.0

Mid-circuit measurement, reset, single-bit feed-forward, exact classical/density
branching, sampled dynamic histories, joint counts and teleportation.

## 0.4.0

Parameters, gate-local analytic gradients, Adam/SGD, regression, binary
classification and VQE demonstrations.

## 0.3.0

Device-local RNG and noise branch sampling, Pauli/damping specializations, local
gate kernels, CPU threading controls and stricter numerical parity checks.

## 0.2.0

Parameters/sweeps, prepared execution, bounded gate cache and batched trajectories.

## 0.1.0

Own circuit engine, NumPy/optional Torch backends, statevector/density/trajectory
simulation, single-qubit channels, reusable algorithms and examples.
