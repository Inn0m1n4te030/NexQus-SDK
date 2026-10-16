# Validation evidence and limits

Audit date: 2026-10-16, Asia/Singapore. Candidate: 1.0.0rc1.

## Current candidate

- 133 tests discovered; 127 pass locally with NumPy and PyTorch CPU. Six CUDA
  tests are explicitly skipped because this host has no CUDA device. This is
  CPU validation, not complete GPU release validation.
- The independent dense oracle covers 40 random static and 24 random dynamic
  circuits, including complex initial states, arbitrary wire order, custom
  operators, channels, overwritten classical bits and conditional operations.
  Static cases also compare single and double precision. These checks do not
  rely on NexQus's tensor kernels to produce the reference result.
- Regression tests cover nonfinite parity data, retained-state shape/presence,
  invalid norms/diagonals, sampled-history memory allocation, learning inputs,
  mutable templates/callbacks and atomic optimizer failure.
- scripts/check_release.py checks the installed/source versions and saves
  docs/release_validation.json. The local report uses Python 3.12.14, NumPy
  2.3.5 and Torch 2.14.1+cpu. It records every skipped test and asserts required
  device capabilities when requested.
- Wheel/source archive validation is recorded separately in
  docs/distribution_validation.json after scripts/check_distribution.py completes.
  The report includes exact artifact hashes. Passing source-tree tests alone is
  insufficient evidence about an installable distribution.
- All seven notebooks and 35 cells pass under the standalone Python engine with
  SDK 1.0.0rc1; one optional CUDA cell is skipped. Notebook content execution is
  recorded in docs/notebook_execution.json.
  The audit host can execute plain Python cells but forbids Jupyter kernel socket
  creation for both TCP and IPC. The failed actual-kernel attempt is retained in
  docs/notebook_jupyter_attempt_rc1.json. A local standalone pass is not a
  substitute for the candidate's required Jupyter/GPU validation on another host.
- Workflow YAML is provided for the declared platform matrix; adding it does not
  establish that GitHub Actions or the matrix passed. No CI run, PyPI project
  TestPyPI upload or public PyPI upload is implied by those files. The maintainer
  has confirmed PyPI ownership; upload authentication remains a separate step.
- A scoped known-vulnerability audit of installed core NumPy was attempted using
  pip-audit, but its PyPI advisory-service request timed out. No clean dependency
  security audit is claimed. CI includes a core-dependency audit; maintainers
  should also audit their resolved optional Torch/tutorial environment.
- All 18 CPU-compatible example scripts pass; gpu.py and validate_gpu.py remain
  for the CUDA host. Details are in docs/example_smoke_validation.json.

## Historical maintainer evidence

The maintainer supplied the following results in the development conversation;
they were not independently rerun on the audit host's hardware:

- v0.5: all 111 tests passed in 7.126 s, with GPU validation reporting no skips.
- v0.5 teleportation on NumPy CPU and CUDA: fidelity 0.9999999999999997 and the
  same classical probabilities and seeded joint counts for the supplied example.
- v0.6: seven notebooks, 35 code cells, engine=jupyter, all_passed=true,
  gpu_required=true and zero optional skips on every notebook.
- v0.3: RTX 2070 statevector benchmark at 20 qubits and 236 operations, median
  wall times 2.2154 s NumPy CPU, 2.2931 s Torch CPU and 0.7612 s CUDA, with strict
  numerical validation. This measured case is not a maximum-capacity claim.
- v0.3 noisy 10-qubit/512-trajectory batch sweep: GPU batch 256 about 0.2365 s;
  batch 512 about 0.2368 s. These are practically tied for that workload, and
  batch 256 used half the main batch-state storage. They are not general defaults
  for every circuit, backend, device or trajectory count.

Older local logs and versioned release notes remain historical records. No new
candidate automatically inherits their GPU, timing or notebook claims.

## Required next evidence

1. Fresh candidate run on the supported CUDA machine: all 133 tests, zero skips.
2. Seven candidate notebooks through a real Jupyter kernel, GPU required,
   correct SDK version and no optional skips.
3. Green cross-platform/version CI on the actual source-release commit.
4. Clean TestPyPI install and smoke test of the exact staged artifacts after the
   owner selects the license and verifies repository/PyPI ownership.

The memory guard estimates workspace. It is not a measured qubit limit or a hard
RAM/VRAM cap. Benchmarks require matched circuit/depth, precision, methods, noise,
trajectories, batch size, threading and repeated synchronized timings. Correctness
under these tests does not establish quantum advantage or production performance.
