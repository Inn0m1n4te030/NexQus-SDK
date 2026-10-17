# CPU performance comparison

Run: 2026-10-17T09:16:10.796865+00:00. CPU: AMD EPYC 9V74 80-Core Processor. Python 3.12.14.

Suite `standard`; precision `double`; 1 CPU thread(s).

Specifications, versions, settings, hardware and numerical diagnostics: [JSON](results.json). Tabular data: [CSV](results.csv).

Every engine receives 3 untimed warmups and 9 measured executions. NumPy random initialization is performed equally before preparation for all engines. Timing covers prepared execution and full state/probability materialization; setup and correctness checks are excluded. No memory sampler runs during timing. Raw repetitions are in the JSON.

Memory is measured in a separate fresh subprocess with the same warmup count. RSS includes native/Python allocations; sampled peaks can miss allocations shorter than 1 ms. OS process peak also includes imports/preparation. Baselines and allocator reuse differ between SDKs. No GPU memory is measured.

| Circuit | Engine | Median ms | Execution peak MiB | Process peak MiB | Max state error | Accuracy |
|---|---|---:|---:|---:|---:|---|
| ghz_4q | nexqus | 0.075 | 37.8 | 37.8 | 0.00e+00 | pass |
| ghz_4q | qiskit | 0.356 | 107.0 | 107.0 | 1.11e-16 | pass |
| ghz_4q | pennylane | 0.215 | 171.2 | 171.3 | 0.00e+00 | pass |
| rotations_4q | nexqus | 1.122 | 37.9 | 38.0 | 2.88e-16 | pass |
| rotations_4q | qiskit | 1.484 | 107.8 | 107.8 | 3.48e-16 | pass |
| rotations_4q | pennylane | 2.695 | 170.8 | 170.8 | 3.19e-16 | pass |
| qft_4q | nexqus | 0.283 | 37.9 | 37.9 | 5.72e-17 | pass |
| qft_4q | qiskit | 0.861 | 107.1 | 107.1 | 1.75e-16 | pass |
| qft_4q | pennylane | 0.716 | 171.3 | 171.3 | 5.58e-17 | pass |
| diagonal_4q | nexqus | 0.887 | 38.3 | 38.3 | 3.08e-16 | pass |
| diagonal_4q | qiskit | 1.658 | 104.0 | 104.0 | 1.22e-15 | pass |
| diagonal_4q | pennylane | 2.455 | 172.4 | 172.4 | 2.86e-16 | pass |
| ghz_12q | nexqus | 0.186 | 39.4 | 39.5 | 1.11e-16 | pass |
| ghz_12q | qiskit | 0.637 | 107.2 | 107.2 | 0.00e+00 | pass |
| ghz_12q | pennylane | 0.776 | 173.0 | 173.0 | 1.11e-16 | pass |
| rotations_12q | nexqus | 7.710 | 40.0 | 40.0 | 5.97e-17 | pass |
| rotations_12q | qiskit | 7.116 | 108.4 | 108.4 | 0.00e+00 | pass |
| rotations_12q | pennylane | 12.521 | 170.8 | 170.8 | 6.14e-17 | pass |
| qft_12q | nexqus | 2.064 | 39.7 | 39.7 | 1.80e-16 | pass |
| qft_12q | qiskit | 4.209 | 102.6 | 102.6 | 0.00e+00 | pass |
| qft_12q | pennylane | 16.512 | 171.6 | 171.7 | 1.76e-16 | pass |
| diagonal_12q | nexqus | 5.414 | 39.4 | 39.4 | 5.83e-16 | pass |
| diagonal_12q | qiskit | 7.007 | 107.9 | 107.9 | 0.00e+00 | pass |
| diagonal_12q | pennylane | 19.191 | 172.5 | 172.5 | 5.83e-16 | pass |
| ghz_18q | nexqus | 8.950 | 54.5 | 54.4 | 1.11e-16 | pass |
| ghz_18q | qiskit | 18.089 | 113.4 | 113.4 | 0.00e+00 | pass |
| ghz_18q | pennylane | 10.255 | 184.3 | 184.3 | 1.11e-16 | pass |
| rotations_18q | nexqus | 555.611 | 58.2 | 58.1 | 1.37e-17 | pass |
| rotations_18q | qiskit | 314.141 | 116.6 | 116.6 | 0.00e+00 | pass |
| rotations_18q | pennylane | 380.668 | 186.0 | 185.8 | 1.20e-17 | pass |
| qft_18q | nexqus | 142.409 | 54.9 | 54.8 | 9.44e-17 | pass |
| qft_18q | qiskit | 212.285 | 113.9 | 113.9 | 0.00e+00 | pass |
| qft_18q | pennylane | 201.320 | 183.8 | 183.8 | 9.90e-17 | pass |
| diagonal_18q | nexqus | 373.121 | 58.0 | 57.8 | 3.06e-16 | pass |
| diagonal_18q | qiskit | 254.527 | 114.3 | 114.3 | 0.00e+00 | pass |
| diagonal_18q | pennylane | 349.999 | 185.0 | 184.8 | 3.12e-16 | pass |
| amplitude_damping_4q | nexqus | 4.454 | 38.9 | 38.9 | 1.13e-16 | pass |
| amplitude_damping_4q | qiskit | 1.450 | 108.7 | 108.7 | 1.57e-16 | pass |
| amplitude_damping_4q | pennylane | 8.648 | 171.2 | 171.2 | 2.27e-14 | pass |
| depolarizing_4q | nexqus | 7.143 | 38.2 | 38.2 | 9.93e-17 | pass |
| depolarizing_4q | qiskit | 1.375 | 109.4 | 109.4 | 1.14e-16 | pass |
| depolarizing_4q | pennylane | 10.351 | 171.2 | 171.2 | 6.85e-14 | pass |
| amplitude_damping_8q | nexqus | 149.594 | 46.1 | 46.0 | 1.74e-17 | pass |
| amplitude_damping_8q | qiskit | 41.316 | 110.0 | 110.0 | 0.00e+00 | pass |
| amplitude_damping_8q | pennylane | 456.845 | 172.9 | 172.9 | 3.56e-15 | pass |
| depolarizing_8q | nexqus | 258.643 | 45.9 | 45.8 | 2.08e-17 | pass |
| depolarizing_8q | qiskit | 43.261 | 110.1 | 110.1 | 0.00e+00 | pass |
| depolarizing_8q | pennylane | 607.606 | 173.5 | 173.5 | 1.22e-14 | pass |

Small circuits (up to 5 qubits) use an independent full-matrix NumPy oracle. Larger circuits use Qiskit Aer as the cross-simulator reference when available; the JSON labels every reference. Statevector errors are aligned for global phase. Density errors compare full matrix entries. Probability TVD, normalization, Hermiticity, and small-density eigenvalues are also checked.

All engines use identical gate/noise specifications, precision, zero shots and CPU settings. Fusion and idle-wire truncation are disabled. PennyLane uses default.qubit/default.mixed, not Lightning. Keep slower NexQus results; performance describes this host and these workloads.
