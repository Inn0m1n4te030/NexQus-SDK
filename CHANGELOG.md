# Unreleased — 1.1.0

- Add bounded quantum order finding and Shor factorization for integers 3..31,
  explicit result metadata, deterministic seeds, tests, and a runnable example.

# Changelog

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
