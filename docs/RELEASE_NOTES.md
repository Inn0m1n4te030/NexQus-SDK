# NexQus SDK 0.1.0 — 14 October 2026

First functional local simulation prototype created for NexQus.

## Delivered

- Own Circuit, decorator API, targeted tensor gate evolution and noise engine.
- NumPy CPU adapter; optional PyTorch CPU/CUDA adapter.
- Noiseless statevectors, exact noisy density matrices, sampled trajectories.
- Five single-qubit noise families, custom Kraus channels and readout bit flips.
- Bell, GHZ, QFT/iQFT, Grover, Bernstein–Vazirani, Deutsch–Jozsa,
  single-target phase estimation and QAOA MaxCut ansatz.
- Python functions/loops, custom unitaries, composition, inverse circuits,
  initial statevectors, Pauli expectations and marginals.
- Six example scripts, English documentation and Burmese quickstart/comments.
- Timing/device/memory metadata, memory estimate guard and benchmark script.

## Validation

- Editable installation completed successfully.
- 27 discovered tests: **25 passed**, **2 skipped**.
- Optional Torch CPU/CUDA parity tests skipped because PyTorch is not installed
  in the build environment. CUDA validation remains outstanding.
- Mathematical checks include dense basis-enumeration references, all basis
  states for QFT through four qubits, QAOA against explicit cost/mixer matrices,
  QPE eigenphases, channel positivity/trace and trajectory convergence.
- All four CPU demonstration scripts ran successfully.
- CPU benchmark ran successfully; raw result saved in `cpu_benchmark.json`.

This is a prototype release. Native GPU optimization, gradients/QNN integration,
mid-circuit measurements, distributed simulation and cloud hardware execution
are not implemented. No license or public distribution decision has been made.
