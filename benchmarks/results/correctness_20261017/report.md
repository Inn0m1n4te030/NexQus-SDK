# Independent correctness checks

Run: 2026-10-17T09:23:01.995253+00:00. CPU: AMD EPYC 9V74 80-Core Processor. Python 3.12.14.

Suite `correctness`; precision `double`; 1 CPU thread(s).

Specifications, versions, settings, hardware and numerical diagnostics: [JSON](results.json). Tabular data: [CSV](results.csv).

This report checks numerical correctness only. It contains no execution timing or performance ranking. See the separately warmed CPU performance report for speed comparisons.

| Circuit | Engine | Max state error | Probability TVD | Normalization error | Accuracy |
|---|---|---:|---:|---:|---|
| rotations_2q_seed7 | nexqus | 1.76e-16 | 2.71e-16 | 0.00e+00 | pass |
| rotations_2q_seed7 | qiskit | 2.62e-16 | 2.78e-16 | 1.11e-16 | pass |
| rotations_2q_seed7 | pennylane | 1.76e-16 | 2.71e-16 | 0.00e+00 | pass |
| qft_2q_seed7 | nexqus | 5.55e-17 | 4.30e-19 | 4.44e-16 | pass |
| qft_2q_seed7 | qiskit | 2.25e-16 | 2.64e-16 | 1.11e-16 | pass |
| qft_2q_seed7 | pennylane | 2.69e-17 | 2.10e-19 | 4.44e-16 | pass |
| diagonal_2q_seed7 | nexqus | 2.78e-16 | 2.12e-16 | 9.99e-16 | pass |
| diagonal_2q_seed7 | qiskit | 2.83e-16 | 3.71e-16 | 0.00e+00 | pass |
| diagonal_2q_seed7 | pennylane | 1.24e-16 | 1.49e-16 | 7.77e-16 | pass |
| amplitude_damping_2q_seed7 | nexqus | 1.39e-16 | 3.47e-17 | 1.11e-16 | pass |
| amplitude_damping_2q_seed7 | qiskit | 2.84e-16 | 1.53e-16 | 3.82e-18 | pass |
| amplitude_damping_2q_seed7 | pennylane | 2.45e-14 | 2.93e-14 | 5.86e-14 | pass |
| depolarizing_2q_seed7 | nexqus | 3.33e-16 | 3.40e-16 | 6.66e-16 | pass |
| depolarizing_2q_seed7 | qiskit | 3.34e-16 | 3.33e-16 | 6.66e-16 | pass |
| depolarizing_2q_seed7 | pennylane | 7.87e-14 | 1.19e-13 | 2.40e-13 | pass |
| rotations_2q_seed73 | nexqus | 1.25e-16 | 0.00e+00 | 5.55e-16 | pass |
| rotations_2q_seed73 | qiskit | 2.78e-16 | 2.84e-16 | 2.22e-16 | pass |
| rotations_2q_seed73 | pennylane | 1.25e-16 | 0.00e+00 | 5.55e-16 | pass |
| qft_2q_seed73 | nexqus | 0.00e+00 | 0.00e+00 | 3.33e-16 | pass |
| qft_2q_seed73 | qiskit | 1.12e-16 | 9.71e-17 | 2.22e-16 | pass |
| qft_2q_seed73 | pennylane | 0.00e+00 | 0.00e+00 | 3.33e-16 | pass |
| diagonal_2q_seed73 | nexqus | 1.53e-16 | 2.10e-16 | 1.22e-15 | pass |
| diagonal_2q_seed73 | qiskit | 2.62e-16 | 1.75e-16 | 4.44e-16 | pass |
| diagonal_2q_seed73 | pennylane | 1.97e-16 | 1.02e-16 | 8.88e-16 | pass |
| amplitude_damping_2q_seed73 | nexqus | 1.49e-16 | 1.39e-16 | 3.33e-16 | pass |
| amplitude_damping_2q_seed73 | qiskit | 1.69e-16 | 9.02e-17 | 3.34e-16 | pass |
| amplitude_damping_2q_seed73 | pennylane | 2.04e-14 | 2.52e-14 | 5.02e-14 | pass |
| depolarizing_2q_seed73 | nexqus | 2.24e-16 | 2.12e-16 | 6.69e-16 | pass |
| depolarizing_2q_seed73 | qiskit | 2.22e-16 | 2.12e-16 | 4.44e-16 | pass |
| depolarizing_2q_seed73 | pennylane | 7.82e-14 | 1.20e-13 | 2.40e-13 | pass |
| rotations_2q_seed419 | nexqus | 2.48e-16 | 2.78e-16 | 4.44e-16 | pass |
| rotations_2q_seed419 | qiskit | 2.24e-16 | 3.33e-16 | 4.44e-16 | pass |
| rotations_2q_seed419 | pennylane | 2.48e-16 | 2.78e-16 | 4.44e-16 | pass |
| qft_2q_seed419 | nexqus | 1.96e-17 | 3.47e-18 | 4.44e-16 | pass |
| qft_2q_seed419 | qiskit | 1.11e-16 | 1.37e-16 | 2.22e-16 | pass |
| qft_2q_seed419 | pennylane | 2.78e-17 | 6.94e-18 | 4.44e-16 | pass |
| diagonal_2q_seed419 | nexqus | 1.14e-16 | 5.55e-17 | 6.66e-16 | pass |
| diagonal_2q_seed419 | qiskit | 3.86e-16 | 2.91e-16 | 0.00e+00 | pass |
| diagonal_2q_seed419 | pennylane | 1.57e-16 | 8.33e-17 | 7.77e-16 | pass |
| amplitude_damping_2q_seed419 | nexqus | 1.25e-16 | 1.39e-16 | 6.53e-18 | pass |
| amplitude_damping_2q_seed419 | qiskit | 1.68e-16 | 1.53e-16 | 8.91e-19 | pass |
| amplitude_damping_2q_seed419 | pennylane | 1.76e-14 | 3.35e-14 | 6.71e-14 | pass |
| depolarizing_2q_seed419 | nexqus | 2.28e-16 | 1.94e-16 | 6.66e-16 | pass |
| depolarizing_2q_seed419 | qiskit | 2.25e-16 | 1.67e-16 | 6.66e-16 | pass |
| depolarizing_2q_seed419 | pennylane | 6.48e-14 | 1.19e-13 | 2.40e-13 | pass |
| rotations_3q_seed7 | nexqus | 2.22e-16 | 2.60e-16 | 4.44e-16 | pass |
| rotations_3q_seed7 | qiskit | 3.33e-16 | 2.92e-16 | 4.44e-16 | pass |
| rotations_3q_seed7 | pennylane | 2.22e-16 | 2.60e-16 | 4.44e-16 | pass |
| qft_3q_seed7 | nexqus | 1.11e-16 | 1.56e-16 | 6.66e-16 | pass |
| qft_3q_seed7 | qiskit | 1.23e-16 | 1.09e-16 | 3.33e-16 | pass |
| qft_3q_seed7 | pennylane | 6.94e-18 | 0.00e+00 | 4.44e-16 | pass |
| diagonal_3q_seed7 | nexqus | 2.44e-16 | 1.94e-16 | 1.78e-15 | pass |
| diagonal_3q_seed7 | qiskit | 4.52e-16 | 7.67e-16 | 2.22e-16 | pass |
| diagonal_3q_seed7 | pennylane | 2.06e-16 | 1.39e-16 | 1.55e-15 | pass |
| amplitude_damping_3q_seed7 | nexqus | 1.97e-16 | 2.62e-16 | 4.45e-16 | pass |
| amplitude_damping_3q_seed7 | qiskit | 4.17e-16 | 4.70e-16 | 9.99e-16 | pass |
| amplitude_damping_3q_seed7 | pennylane | 1.49e-14 | 3.89e-14 | 7.77e-14 | pass |
| depolarizing_3q_seed7 | nexqus | 2.27e-16 | 2.60e-16 | 4.45e-16 | pass |
| depolarizing_3q_seed7 | qiskit | 2.03e-16 | 2.79e-16 | 4.44e-16 | pass |
| depolarizing_3q_seed7 | pennylane | 6.33e-14 | 1.79e-13 | 3.60e-13 | pass |
| rotations_3q_seed73 | nexqus | 2.10e-16 | 1.10e-16 | 4.44e-16 | pass |
| rotations_3q_seed73 | qiskit | 2.90e-16 | 2.89e-16 | 2.22e-16 | pass |
| rotations_3q_seed73 | pennylane | 2.10e-16 | 1.10e-16 | 4.44e-16 | pass |
| qft_3q_seed73 | nexqus | 1.39e-17 | 0.00e+00 | 5.55e-16 | pass |
| qft_3q_seed73 | qiskit | 1.67e-16 | 2.08e-16 | 3.33e-16 | pass |
| qft_3q_seed73 | pennylane | 2.78e-17 | 6.94e-18 | 5.55e-16 | pass |
| diagonal_3q_seed73 | nexqus | 3.38e-16 | 5.43e-16 | 2.22e-15 | pass |
| diagonal_3q_seed73 | qiskit | 3.86e-16 | 6.73e-16 | 2.22e-16 | pass |
| diagonal_3q_seed73 | pennylane | 2.99e-16 | 3.58e-16 | 1.78e-15 | pass |
| amplitude_damping_3q_seed73 | nexqus | 1.57e-16 | 9.02e-17 | 2.63e-16 | pass |
| amplitude_damping_3q_seed73 | qiskit | 1.37e-16 | 2.22e-16 | 9.75e-18 | pass |
| amplitude_damping_3q_seed73 | pennylane | 2.44e-14 | 4.34e-14 | 8.70e-14 | pass |
| depolarizing_3q_seed73 | nexqus | 1.78e-16 | 3.21e-16 | 6.67e-16 | pass |
| depolarizing_3q_seed73 | qiskit | 1.67e-16 | 2.10e-16 | 1.11e-15 | pass |
| depolarizing_3q_seed73 | pennylane | 7.20e-14 | 1.79e-13 | 3.60e-13 | pass |
| rotations_3q_seed419 | nexqus | 1.11e-16 | 1.46e-16 | 2.22e-16 | pass |
| rotations_3q_seed419 | qiskit | 2.24e-16 | 2.01e-16 | 0.00e+00 | pass |
| rotations_3q_seed419 | pennylane | 1.11e-16 | 1.46e-16 | 2.22e-16 | pass |
| qft_3q_seed419 | nexqus | 2.78e-17 | 9.11e-18 | 6.66e-16 | pass |
| qft_3q_seed419 | qiskit | 1.11e-16 | 6.98e-17 | 4.44e-16 | pass |
| qft_3q_seed419 | pennylane | 0.00e+00 | 0.00e+00 | 6.66e-16 | pass |
| diagonal_3q_seed419 | nexqus | 2.48e-16 | 2.70e-16 | 8.88e-16 | pass |
| diagonal_3q_seed419 | qiskit | 7.11e-16 | 8.46e-16 | 4.44e-16 | pass |
| diagonal_3q_seed419 | pennylane | 1.24e-16 | 1.63e-16 | 1.11e-15 | pass |
| amplitude_damping_3q_seed419 | nexqus | 1.58e-16 | 1.56e-16 | 9.76e-18 | pass |
| amplitude_damping_3q_seed419 | qiskit | 2.78e-16 | 4.23e-16 | 6.66e-16 | pass |
| amplitude_damping_3q_seed419 | pennylane | 2.47e-14 | 4.98e-14 | 9.99e-14 | pass |
| depolarizing_3q_seed419 | nexqus | 1.60e-16 | 2.43e-16 | 8.93e-16 | pass |
| depolarizing_3q_seed419 | qiskit | 1.98e-16 | 3.05e-16 | 6.66e-16 | pass |
| depolarizing_3q_seed419 | pennylane | 6.54e-14 | 1.79e-13 | 3.60e-13 | pass |
| rotations_5q_seed7 | nexqus | 2.78e-16 | 4.48e-16 | 4.44e-16 | pass |
| rotations_5q_seed7 | qiskit | 1.96e-16 | 3.15e-16 | 2.22e-16 | pass |
| rotations_5q_seed7 | pennylane | 2.78e-16 | 4.48e-16 | 4.44e-16 | pass |
| qft_5q_seed7 | nexqus | 1.67e-16 | 6.80e-17 | 9.99e-16 | pass |
| qft_5q_seed7 | qiskit | 3.72e-16 | 3.42e-16 | 1.11e-16 | pass |
| qft_5q_seed7 | pennylane | 1.39e-17 | 5.42e-18 | 1.11e-15 | pass |
| diagonal_5q_seed7 | nexqus | 1.93e-16 | 2.68e-16 | 2.66e-15 | pass |
| diagonal_5q_seed7 | qiskit | 6.11e-16 | 1.17e-15 | 4.44e-16 | pass |
| diagonal_5q_seed7 | pennylane | 1.56e-16 | 2.62e-16 | 2.78e-15 | pass |
| amplitude_damping_5q_seed7 | nexqus | 5.13e-17 | 2.41e-16 | 3.36e-16 | pass |
| amplitude_damping_5q_seed7 | qiskit | 6.61e-17 | 3.54e-16 | 3.33e-16 | pass |
| amplitude_damping_5q_seed7 | pennylane | 7.36e-15 | 7.42e-14 | 1.48e-13 | pass |
| depolarizing_5q_seed7 | nexqus | 6.36e-17 | 2.97e-16 | 1.33e-15 | pass |
| depolarizing_5q_seed7 | qiskit | 6.25e-17 | 3.63e-16 | 1.33e-15 | pass |
| depolarizing_5q_seed7 | pennylane | 2.54e-14 | 2.99e-13 | 6.00e-13 | pass |
| rotations_5q_seed73 | nexqus | 2.28e-16 | 4.04e-16 | 0.00e+00 | pass |
| rotations_5q_seed73 | qiskit | 1.74e-16 | 3.64e-16 | 2.22e-16 | pass |
| rotations_5q_seed73 | pennylane | 2.28e-16 | 4.04e-16 | 0.00e+00 | pass |
| qft_5q_seed73 | nexqus | 7.85e-17 | 7.76e-17 | 1.22e-15 | pass |
| qft_5q_seed73 | qiskit | 2.22e-16 | 3.82e-16 | 3.33e-16 | pass |
| qft_5q_seed73 | pennylane | 5.55e-17 | 8.67e-19 | 1.33e-15 | pass |
| diagonal_5q_seed73 | nexqus | 2.31e-16 | 2.48e-16 | 2.66e-15 | pass |
| diagonal_5q_seed73 | qiskit | 4.44e-16 | 8.90e-16 | 6.66e-16 | pass |
| diagonal_5q_seed73 | pennylane | 1.67e-16 | 2.60e-16 | 2.44e-15 | pass |
| amplitude_damping_5q_seed73 | nexqus | 1.11e-16 | 2.60e-16 | 1.25e-16 | pass |
| amplitude_damping_5q_seed73 | qiskit | 5.93e-17 | 2.81e-16 | 4.44e-16 | pass |
| amplitude_damping_5q_seed73 | pennylane | 1.08e-14 | 7.26e-14 | 1.45e-13 | pass |
| depolarizing_5q_seed73 | nexqus | 8.47e-17 | 3.38e-16 | 1.33e-15 | pass |
| depolarizing_5q_seed73 | qiskit | 8.89e-17 | 2.43e-16 | 1.78e-15 | pass |
| depolarizing_5q_seed73 | pennylane | 3.42e-14 | 2.99e-13 | 6.00e-13 | pass |
| rotations_5q_seed419 | nexqus | 1.70e-16 | 1.53e-16 | 2.22e-16 | pass |
| rotations_5q_seed419 | qiskit | 2.78e-16 | 3.40e-16 | 2.22e-16 | pass |
| rotations_5q_seed419 | pennylane | 1.70e-16 | 1.53e-16 | 2.22e-16 | pass |
| qft_5q_seed419 | nexqus | 9.02e-17 | 7.90e-17 | 8.88e-16 | pass |
| qft_5q_seed419 | qiskit | 2.22e-16 | 2.65e-16 | 3.33e-16 | pass |
| qft_5q_seed419 | pennylane | 0.00e+00 | 0.00e+00 | 8.88e-16 | pass |
| diagonal_5q_seed419 | nexqus | 2.37e-16 | 3.40e-16 | 2.22e-15 | pass |
| diagonal_5q_seed419 | qiskit | 6.11e-16 | 1.40e-15 | 0.00e+00 | pass |
| diagonal_5q_seed419 | pennylane | 2.95e-16 | 4.52e-16 | 2.00e-15 | pass |
| amplitude_damping_5q_seed419 | nexqus | 8.45e-17 | 3.11e-16 | 1.61e-17 | pass |
| amplitude_damping_5q_seed419 | qiskit | 1.13e-16 | 6.77e-16 | 8.88e-16 | pass |
| amplitude_damping_5q_seed419 | pennylane | 1.01e-14 | 7.02e-14 | 1.41e-13 | pass |
| depolarizing_5q_seed419 | nexqus | 8.88e-17 | 3.10e-16 | 1.55e-15 | pass |
| depolarizing_5q_seed419 | qiskit | 7.92e-17 | 4.51e-16 | 8.88e-16 | pass |
| depolarizing_5q_seed419 | pennylane | 3.68e-14 | 2.99e-13 | 6.00e-13 | pass |

Small circuits (up to 5 qubits) use an independent full-matrix NumPy oracle. Larger circuits use Qiskit Aer as the cross-simulator reference when available; the JSON labels every reference. Statevector errors are aligned for global phase. Density errors compare full matrix entries. Probability TVD, normalization, Hermiticity, and small-density eigenvalues are also checked.

All engines use identical gate/noise specifications, precision, zero shots and CPU settings. Fusion and idle-wire truncation are disabled. PennyLane uses default.qubit/default.mixed, not Lightning. Keep slower NexQus results; performance describes this host and these workloads.
