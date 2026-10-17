# CPU performance comparison

Run: 2026-10-17T09:20:45.227710+00:00. CPU: AMD EPYC 9V74 80-Core Processor. Python 3.12.14.

Suite `standard`; precision `double`; 1 CPU thread(s).

Specifications, versions, settings, hardware and numerical diagnostics: [JSON](results.json). Tabular data: [CSV](results.csv).

Every engine receives 3 untimed warmups and 9 measured executions. NumPy random initialization is performed equally before preparation for all engines. Timing covers prepared execution and full state/probability materialization; setup and correctness checks are excluded. No memory sampler runs during timing. Raw repetitions are in the JSON.

Memory is measured in a separate fresh subprocess with the same warmup count. RSS includes native/Python allocations; sampled peaks can miss allocations shorter than 1 ms. OS process peak also includes imports/preparation. Baselines and allocator reuse differ between SDKs. No GPU memory is measured.

| Circuit | Engine | Median ms | Execution peak MiB | Process peak MiB | Max state error | Accuracy |
|---|---|---:|---:|---:|---:|---|
| ghz_4q | nexqus | 0.043 | 37.8 | 37.9 | 0.00e+00 | pass |
| ghz_4q | qiskit | 0.389 | 106.9 | 107.0 | 1.11e-16 | pass |
| ghz_4q | pennylane | 0.242 | 171.5 | 171.6 | 0.00e+00 | pass |
| rotations_4q | nexqus | 1.147 | 38.0 | 38.0 | 3.19e-16 | pass |
| rotations_4q | qiskit | 1.429 | 107.8 | 107.8 | 3.48e-16 | pass |
| rotations_4q | pennylane | 2.885 | 171.7 | 171.7 | 3.19e-16 | pass |
| qft_4q | nexqus | 0.243 | 37.8 | 37.8 | 5.72e-17 | pass |
| qft_4q | qiskit | 0.884 | 107.6 | 107.6 | 1.75e-16 | pass |
| qft_4q | pennylane | 0.721 | 171.8 | 171.8 | 5.58e-17 | pass |
| diagonal_4q | nexqus | 0.822 | 37.9 | 37.9 | 2.86e-16 | pass |
| diagonal_4q | qiskit | 1.534 | 104.3 | 104.3 | 1.22e-15 | pass |
| diagonal_4q | pennylane | 3.392 | 171.4 | 171.4 | 2.86e-16 | pass |
| ghz_12q | nexqus | 0.227 | 37.5 | 37.5 | 1.11e-16 | pass |
| ghz_12q | qiskit | 0.749 | 107.4 | 107.4 | 0.00e+00 | pass |
| ghz_12q | pennylane | 0.688 | 172.3 | 172.3 | 1.11e-16 | pass |
| rotations_12q | nexqus | 6.500 | 39.7 | 39.7 | 6.14e-17 | pass |
| rotations_12q | qiskit | 6.592 | 105.2 | 105.2 | 0.00e+00 | pass |
| rotations_12q | pennylane | 12.855 | 170.1 | 170.1 | 6.14e-17 | pass |
| qft_12q | nexqus | 1.941 | 39.7 | 39.7 | 1.80e-16 | pass |
| qft_12q | qiskit | 4.366 | 107.4 | 107.4 | 0.00e+00 | pass |
| qft_12q | pennylane | 16.761 | 172.1 | 172.2 | 1.76e-16 | pass |
| diagonal_12q | nexqus | 4.544 | 40.0 | 40.0 | 5.83e-16 | pass |
| diagonal_12q | qiskit | 6.591 | 104.7 | 104.7 | 0.00e+00 | pass |
| diagonal_12q | pennylane | 16.011 | 172.2 | 172.2 | 5.83e-16 | pass |
| ghz_18q | nexqus | 7.079 | 56.7 | 56.7 | 1.11e-16 | pass |
| ghz_18q | qiskit | 18.966 | 110.6 | 110.6 | 0.00e+00 | pass |
| ghz_18q | pennylane | 9.511 | 183.2 | 183.2 | 1.11e-16 | pass |
| rotations_18q | nexqus | 278.928 | 59.1 | 59.1 | 1.17e-17 | pass |
| rotations_18q | qiskit | 295.237 | 113.1 | 113.1 | 0.00e+00 | pass |
| rotations_18q | pennylane | 358.512 | 185.9 | 185.8 | 1.20e-17 | pass |
| qft_18q | nexqus | 119.342 | 59.8 | 59.6 | 9.44e-17 | pass |
| qft_18q | qiskit | 210.719 | 114.5 | 114.5 | 0.00e+00 | pass |
| qft_18q | pennylane | 203.260 | 184.4 | 184.1 | 9.90e-17 | pass |
| diagonal_18q | nexqus | 181.188 | 58.9 | 58.9 | 3.05e-16 | pass |
| diagonal_18q | qiskit | 253.671 | 114.2 | 114.2 | 0.00e+00 | pass |
| diagonal_18q | pennylane | 389.943 | 184.2 | 183.9 | 3.12e-16 | pass |
| amplitude_damping_4q | nexqus | 1.678 | 39.3 | 39.3 | 1.94e-16 | pass |
| amplitude_damping_4q | qiskit | 1.482 | 105.2 | 105.2 | 1.57e-16 | pass |
| amplitude_damping_4q | pennylane | 8.687 | 170.0 | 170.0 | 2.27e-14 | pass |
| depolarizing_4q | nexqus | 1.890 | 38.1 | 38.1 | 1.03e-16 | pass |
| depolarizing_4q | qiskit | 1.325 | 106.0 | 106.0 | 1.14e-16 | pass |
| depolarizing_4q | pennylane | 10.127 | 170.1 | 170.1 | 6.85e-14 | pass |
| amplitude_damping_8q | nexqus | 83.420 | 43.4 | 43.2 | 1.39e-17 | pass |
| amplitude_damping_8q | qiskit | 42.745 | 105.2 | 105.2 | 0.00e+00 | pass |
| amplitude_damping_8q | pennylane | 419.933 | 172.7 | 172.7 | 3.56e-15 | pass |
| depolarizing_8q | nexqus | 65.672 | 44.9 | 44.8 | 1.27e-17 | pass |
| depolarizing_8q | qiskit | 46.040 | 110.4 | 110.4 | 0.00e+00 | pass |
| depolarizing_8q | pennylane | 586.024 | 173.7 | 173.7 | 1.22e-14 | pass |

Small circuits (up to 5 qubits) use an independent full-matrix NumPy oracle. Larger circuits use Qiskit Aer as the cross-simulator reference when available; the JSON labels every reference. Statevector errors are aligned for global phase. Density errors compare full matrix entries. Probability TVD, normalization, Hermiticity, and small-density eigenvalues are also checked.

All engines use identical gate/noise specifications, precision, zero shots and CPU settings. Fusion and idle-wire truncation are disabled. PennyLane uses default.qubit/default.mixed, not Lightning. Keep slower NexQus results; performance describes this host and these workloads.
