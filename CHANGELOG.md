# Changelog

## 1.0.0rc1 — maintainer review, 2026-10-16 (Asia/Singapore)

This candidate is not a published stable release. It preserves the v0.6 feature
set and hardens correctness, validation and distribution.

### Fixed

- A numerical parity check could pass NaN retained states, malformed state
  shapes, missing retained states or nonfinite classical probabilities. These
  now fail explicitly. Paired trajectory checks also require a supplied seed.
- Diagonal validation and initial-state normalization used NumPy's loose default
  relative tolerance. Both now use explicit tolerances. Input state norm within
  1e-10 is accepted and rescaled to remove roundoff; larger errors are rejected.
- Dynamic trajectory shot scheduling allocated one host counter per trajectory,
  including runs with zero shots. It now stores only histories selected by the
  requested shots, with O(shots) schedule storage independent of trajectory count.
- NumPy damping fast paths could promote complex64 trajectory batches to
  complex128. Their scalar arithmetic now preserves selected state precision,
  consistent with state-storage reporting; dtype regressions cover the paths.
- Complex learning inputs/targets no longer silently lose imaginary components.
- A learning callback cannot mutate stored training-history parameter snapshots.
- Adam updates are atomic and reject nonfinite moments instead of retaining a
  partially updated or overflowed optimizer state.
- QuantumFunction rechecks unsupported dynamic control if its template changes.
- Result.marginal accepts a scalar wire index consistently with reduced_density.
- Notebook execution reports describe the actual engine. Failed runs clear stale
  later-cell outputs; standalone timeouts produce failed report entries.

### Added

- Independent full-matrix reference checks for 40 random static and 24 random
  dynamic circuits, with complex initial states, channels and feed-forward.
- Targeted regression and additional CPU/CUDA parity tests: 133 total discovered.
- Complete project metadata, a single source of version information, explicit
  source-distribution contents, and a small runtime wheel.
- Fresh-environment wheel and source-rebuilt-wheel checks, release evidence
  reports, a multi-OS/Python CI matrix, and manual Trusted Publishing workflows.
- Public API reference, a proposed 1.x stability policy and release instructions.

### Compatibility notes

Previously accepted invalid diagonal entries and noticeably unnormalized initial
states now raise ValueError. The validator requires matching retained-state
presence/representation. It rejects unseeded trajectory parity, which cannot
demonstrate paired reproducibility. Valid existing circuit APIs are retained.

The maintainer approved MIT licensing and confirmed PyPI ownership on 2026-10-16.
The public repository is Inn0m1n4te030/NexQus-SDK; Source/Issues/Documentation
metadata now point to it. CI execution, fresh CUDA validation and TestPyPI
installation remain release gates. Build and validation do not perform uploads.

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

Historical details and validation limitations are preserved in the versioned
release notes under docs; they are not claims about this release candidate.
