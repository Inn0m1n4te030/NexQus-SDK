# NexQus SDK 0.3.0 — 15 October 2026

Focus: performance bottlenecks revealed by the user's RTX 2070 v0.2 benchmarks.

## Shipped

Device-local trajectory sampling, portable/native RNGs, fast Pauli mixtures,
analytic damping channels, NumPy amplitude-pair updates, diagonal broadcasting,
initial-state reuse, device probability accumulation, stricter paired validation,
retained-state checks and explicit CPU thread controls. Existing gate, function,
sweep, compilation and algorithm APIs remain available.

Common CX operations also use amplitude-slice swaps rather than dense matrices.

See V03_GUIDE.md for exact commands and compatibility details. Stochastic
streams change from v0.2. No new QNN/autograd, hardware connectors, distributed
simulation, mid-circuit measurements or native CUDA kernels are included.

## Evidence and limitations

- The user supplied v0.2 RTX 2070 benchmarks: matching CPU/GPU probabilities;
  20-qubit GPU was near Torch CPU, and noisy GPU was near NumPy CPU.
- v0.3 local suite: 69 discovered, 63 passed, 6 optional Torch CPU/CUDA tests
  skipped because PyTorch is not installed in this environment.
- New checks cover local-vs-generic kernels, portable RNG, specialized-vs-generic
  noise, exact-channel convergence, absence of per-channel host reads, phase
  errors, total variation, native generator reproducibility and optional CUDA.
- New CUDA paths require the bundled strict GPU validator on the target machine.
  Earlier-version GPU results do not establish v0.3 GPU correctness or speed.
- CPU examples and benchmark output are included. Performance claims must be
  tied to workload, backend, precision, batch layout and thread settings.
- One local NumPy CPU comparison (6 qubits, layered depth4, 512 noisy
  trajectories, batch32, double precision, one BLAS thread, three repeats)
  measured median full-run time 0.765 s in v0.2 versus 0.108 s in v0.3, about
  7.1x faster. Versions use different trajectory streams with the same channel
  semantics. This does not predict target-GPU performance. Raw JSON is bundled.

Historical release notes, test logs and benchmarks with v01/v02 names refer to
their original versions. New v0.3 results use v03 names. Licensing remains
unselected for this private prototype.
