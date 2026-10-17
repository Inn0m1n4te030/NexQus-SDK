# Independent simulator comparison

Run: 2026-10-17T08:44:23.547356+00:00. CPU: AMD EPYC 9V74 80-Core Processor. Python 3.12.14.

Suite `correctness`; precision `double`; 1 CPU thread(s); 0 warmups and 1 measured repetitions per engine/case.

Full specifications, raw timings, versions, settings, hardware and errors: [JSON](results.json). Tabular data: [CSV](results.csv).

Timing covers prepared execution and full state/probability output. Imports, circuit construction, and compilation are excluded and preparation is reported separately. Gate fusion and idle-wire truncation are disabled. Every engine uses the same gates, noise, precision, zero shots and CPU. PennyLane uses default.qubit/default.mixed; this is not a benchmark of lightning.qubit.

RSS is process resident memory, including native allocations. Execution peak is sampled every 1 ms and includes warmups plus the untimed validation run; short peaks may be missed. OS process peak also includes imports and preparation. Baselines differ between SDKs; incremental RSS can be zero when an allocator reuses memory. Sampling is active during timing, so its overhead is included. No GPU memory or hardware execution is measured.

| Circuit | Engine | Median ms | Execution peak MiB | Process peak MiB | Max state error | Accuracy |
|---|---|---:|---:|---:|---:|---|
| rotations_2q_seed7 | nexqus | 4.271 | 38.5 | 38.5 | 1.86e-16 | pass |
| rotations_2q_seed7 | qiskit | 1.600 | 104.0 | 104.0 | 2.62e-16 | pass |
| rotations_2q_seed7 | pennylane | 1.192 | 171.0 | 171.0 | 1.76e-16 | pass |
| qft_2q_seed7 | nexqus | 4.041 | 39.1 | 39.1 | 5.55e-17 | pass |
| qft_2q_seed7 | qiskit | 1.742 | 104.2 | 104.3 | 2.25e-16 | pass |
| qft_2q_seed7 | pennylane | 1.179 | 170.9 | 170.9 | 2.69e-17 | pass |
| diagonal_2q_seed7 | nexqus | 4.585 | 38.6 | 38.7 | 2.00e-16 | pass |
| diagonal_2q_seed7 | qiskit | 1.913 | 107.1 | 107.1 | 2.83e-16 | pass |
| diagonal_2q_seed7 | pennylane | 1.695 | 171.0 | 171.0 | 1.24e-16 | pass |
| amplitude_damping_2q_seed7 | nexqus | 7.325 | 38.5 | 38.5 | 3.38e-16 | pass |
| amplitude_damping_2q_seed7 | qiskit | 1.561 | 105.3 | 105.3 | 2.84e-16 | pass |
| amplitude_damping_2q_seed7 | pennylane | 4.257 | 171.7 | 171.7 | 2.45e-14 | pass |
| depolarizing_2q_seed7 | nexqus | 7.493 | 39.2 | 39.3 | 2.78e-16 | pass |
| depolarizing_2q_seed7 | qiskit | 2.300 | 108.8 | 108.8 | 3.34e-16 | pass |
| depolarizing_2q_seed7 | pennylane | 4.519 | 169.9 | 170.0 | 7.87e-14 | pass |
| rotations_2q_seed73 | nexqus | 4.558 | 37.2 | 37.3 | 1.24e-16 | pass |
| rotations_2q_seed73 | qiskit | 2.007 | 104.3 | 104.3 | 2.78e-16 | pass |
| rotations_2q_seed73 | pennylane | 1.280 | 169.9 | 169.9 | 1.25e-16 | pass |
| qft_2q_seed73 | nexqus | 4.411 | 39.1 | 39.1 | 0.00e+00 | pass |
| qft_2q_seed73 | qiskit | 1.594 | 107.3 | 107.3 | 1.12e-16 | pass |
| qft_2q_seed73 | pennylane | 1.132 | 171.1 | 171.2 | 0.00e+00 | pass |
| diagonal_2q_seed73 | nexqus | 4.896 | 37.7 | 37.7 | 1.11e-16 | pass |
| diagonal_2q_seed73 | qiskit | 4.287 | 103.7 | 103.8 | 2.62e-16 | pass |
| diagonal_2q_seed73 | pennylane | 1.939 | 171.7 | 171.7 | 1.97e-16 | pass |
| amplitude_damping_2q_seed73 | nexqus | 5.744 | 38.4 | 38.5 | 1.49e-16 | pass |
| amplitude_damping_2q_seed73 | qiskit | 1.858 | 104.8 | 104.9 | 1.69e-16 | pass |
| amplitude_damping_2q_seed73 | pennylane | 4.722 | 169.9 | 169.9 | 2.04e-14 | pass |
| depolarizing_2q_seed73 | nexqus | 6.577 | 39.1 | 39.2 | 2.24e-16 | pass |
| depolarizing_2q_seed73 | qiskit | 1.943 | 105.5 | 105.6 | 2.22e-16 | pass |
| depolarizing_2q_seed73 | pennylane | 4.046 | 169.4 | 169.4 | 7.82e-14 | pass |
| rotations_2q_seed419 | nexqus | 4.081 | 39.0 | 39.1 | 2.22e-16 | pass |
| rotations_2q_seed419 | qiskit | 1.935 | 103.8 | 103.8 | 2.24e-16 | pass |
| rotations_2q_seed419 | pennylane | 1.493 | 169.2 | 169.3 | 2.48e-16 | pass |
| qft_2q_seed419 | nexqus | 4.299 | 39.2 | 39.2 | 1.96e-17 | pass |
| qft_2q_seed419 | qiskit | 1.751 | 103.9 | 103.9 | 1.11e-16 | pass |
| qft_2q_seed419 | pennylane | 1.169 | 171.3 | 171.4 | 2.78e-17 | pass |
| diagonal_2q_seed419 | nexqus | 4.722 | 38.6 | 38.6 | 7.85e-17 | pass |
| diagonal_2q_seed419 | qiskit | 1.874 | 103.7 | 103.8 | 3.86e-16 | pass |
| diagonal_2q_seed419 | pennylane | 1.674 | 170.4 | 170.4 | 1.57e-16 | pass |
| amplitude_damping_2q_seed419 | nexqus | 5.896 | 37.9 | 38.0 | 2.30e-16 | pass |
| amplitude_damping_2q_seed419 | qiskit | 2.224 | 108.6 | 108.7 | 1.68e-16 | pass |
| amplitude_damping_2q_seed419 | pennylane | 4.310 | 171.1 | 171.1 | 1.76e-14 | pass |
| depolarizing_2q_seed419 | nexqus | 6.929 | 37.2 | 37.3 | 2.11e-16 | pass |
| depolarizing_2q_seed419 | qiskit | 2.050 | 108.6 | 108.7 | 2.25e-16 | pass |
| depolarizing_2q_seed419 | pennylane | 4.359 | 170.1 | 170.1 | 6.48e-14 | pass |
| rotations_3q_seed7 | nexqus | 4.679 | 37.7 | 37.8 | 1.72e-16 | pass |
| rotations_3q_seed7 | qiskit | 1.816 | 107.4 | 107.5 | 3.33e-16 | pass |
| rotations_3q_seed7 | pennylane | 2.441 | 170.9 | 170.9 | 2.22e-16 | pass |
| qft_3q_seed7 | nexqus | 4.400 | 38.7 | 38.7 | 1.11e-16 | pass |
| qft_3q_seed7 | qiskit | 2.755 | 103.8 | 103.9 | 1.23e-16 | pass |
| qft_3q_seed7 | pennylane | 1.709 | 170.8 | 170.8 | 6.94e-18 | pass |
| diagonal_3q_seed7 | nexqus | 7.474 | 39.0 | 39.1 | 1.14e-16 | pass |
| diagonal_3q_seed7 | qiskit | 1.645 | 103.9 | 104.0 | 4.52e-16 | pass |
| diagonal_3q_seed7 | pennylane | 2.589 | 171.1 | 171.1 | 2.06e-16 | pass |
| amplitude_damping_3q_seed7 | nexqus | 8.119 | 37.7 | 37.8 | 1.43e-16 | pass |
| amplitude_damping_3q_seed7 | qiskit | 1.790 | 108.8 | 108.9 | 4.17e-16 | pass |
| amplitude_damping_3q_seed7 | pennylane | 7.441 | 169.5 | 169.6 | 1.49e-14 | pass |
| depolarizing_3q_seed7 | nexqus | 8.411 | 38.1 | 38.1 | 1.46e-16 | pass |
| depolarizing_3q_seed7 | qiskit | 1.662 | 105.5 | 105.5 | 2.03e-16 | pass |
| depolarizing_3q_seed7 | pennylane | 6.685 | 170.1 | 170.1 | 6.33e-14 | pass |
| rotations_3q_seed73 | nexqus | 5.155 | 37.6 | 37.7 | 2.08e-16 | pass |
| rotations_3q_seed73 | qiskit | 1.845 | 107.2 | 107.3 | 2.90e-16 | pass |
| rotations_3q_seed73 | pennylane | 2.314 | 169.8 | 169.8 | 2.10e-16 | pass |
| qft_3q_seed73 | nexqus | 5.007 | 37.7 | 37.7 | 1.39e-17 | pass |
| qft_3q_seed73 | qiskit | 1.461 | 104.0 | 104.0 | 1.67e-16 | pass |
| qft_3q_seed73 | pennylane | 1.432 | 170.7 | 170.7 | 2.78e-17 | pass |
| diagonal_3q_seed73 | nexqus | 4.764 | 37.7 | 37.8 | 2.62e-16 | pass |
| diagonal_3q_seed73 | qiskit | 2.210 | 106.8 | 106.8 | 3.86e-16 | pass |
| diagonal_3q_seed73 | pennylane | 2.545 | 172.3 | 172.3 | 2.99e-16 | pass |
| amplitude_damping_3q_seed73 | nexqus | 6.996 | 36.7 | 37.8 | 1.14e-16 | pass |
| amplitude_damping_3q_seed73 | qiskit | 1.701 | 105.5 | 105.5 | 1.37e-16 | pass |
| amplitude_damping_3q_seed73 | pennylane | 9.292 | 170.1 | 170.1 | 2.44e-14 | pass |
| depolarizing_3q_seed73 | nexqus | 9.580 | 38.0 | 38.1 | 1.16e-16 | pass |
| depolarizing_3q_seed73 | qiskit | 2.248 | 105.5 | 105.6 | 1.67e-16 | pass |
| depolarizing_3q_seed73 | pennylane | 8.386 | 170.6 | 170.6 | 7.20e-14 | pass |
| rotations_3q_seed419 | nexqus | 5.804 | 37.7 | 37.8 | 1.67e-16 | pass |
| rotations_3q_seed419 | qiskit | 1.666 | 104.1 | 104.1 | 2.24e-16 | pass |
| rotations_3q_seed419 | pennylane | 2.268 | 169.7 | 169.7 | 1.11e-16 | pass |
| qft_3q_seed419 | nexqus | 4.738 | 37.6 | 37.7 | 2.78e-17 | pass |
| qft_3q_seed419 | qiskit | 2.471 | 107.3 | 107.3 | 1.11e-16 | pass |
| qft_3q_seed419 | pennylane | 1.695 | 172.4 | 172.4 | 0.00e+00 | pass |
| diagonal_3q_seed419 | nexqus | 5.038 | 38.7 | 38.8 | 2.78e-16 | pass |
| diagonal_3q_seed419 | qiskit | 1.995 | 107.1 | 107.2 | 7.11e-16 | pass |
| diagonal_3q_seed419 | pennylane | 2.494 | 171.1 | 171.1 | 1.24e-16 | pass |
| amplitude_damping_3q_seed419 | nexqus | 8.607 | 39.1 | 39.2 | 1.26e-16 | pass |
| amplitude_damping_3q_seed419 | qiskit | 2.654 | 105.4 | 105.5 | 2.78e-16 | pass |
| amplitude_damping_3q_seed419 | pennylane | 8.809 | 169.8 | 169.8 | 2.47e-14 | pass |
| depolarizing_3q_seed419 | nexqus | 9.612 | 38.8 | 38.8 | 1.17e-16 | pass |
| depolarizing_3q_seed419 | qiskit | 2.162 | 105.6 | 105.6 | 1.98e-16 | pass |
| depolarizing_3q_seed419 | pennylane | 7.430 | 170.1 | 170.1 | 6.54e-14 | pass |
| rotations_5q_seed7 | nexqus | 6.362 | 38.0 | 39.2 | 1.78e-16 | pass |
| rotations_5q_seed7 | qiskit | 2.666 | 107.3 | 107.3 | 1.96e-16 | pass |
| rotations_5q_seed7 | pennylane | 4.072 | 169.7 | 169.7 | 2.78e-16 | pass |
| qft_5q_seed7 | nexqus | 4.478 | 37.9 | 38.0 | 1.67e-16 | pass |
| qft_5q_seed7 | qiskit | 3.850 | 103.7 | 103.8 | 3.72e-16 | pass |
| qft_5q_seed7 | pennylane | 2.326 | 170.4 | 170.5 | 1.39e-17 | pass |
| diagonal_5q_seed7 | nexqus | 6.274 | 37.7 | 37.8 | 1.14e-16 | pass |
| diagonal_5q_seed7 | qiskit | 3.057 | 107.2 | 107.2 | 6.11e-16 | pass |
| diagonal_5q_seed7 | pennylane | 3.852 | 171.2 | 171.2 | 1.56e-16 | pass |
| amplitude_damping_5q_seed7 | nexqus | 9.207 | 39.1 | 39.2 | 4.67e-17 | pass |
| amplitude_damping_5q_seed7 | qiskit | 2.708 | 108.9 | 108.9 | 6.61e-17 | pass |
| amplitude_damping_5q_seed7 | pennylane | 16.157 | 170.1 | 170.1 | 7.36e-15 | pass |
| depolarizing_5q_seed7 | nexqus | 12.750 | 37.8 | 37.9 | 4.43e-17 | pass |
| depolarizing_5q_seed7 | qiskit | 2.834 | 105.5 | 105.5 | 6.25e-17 | pass |
| depolarizing_5q_seed7 | pennylane | 15.023 | 169.4 | 169.4 | 2.54e-14 | pass |
| rotations_5q_seed73 | nexqus | 5.053 | 38.1 | 38.1 | 1.55e-16 | pass |
| rotations_5q_seed73 | qiskit | 1.979 | 104.0 | 104.0 | 1.74e-16 | pass |
| rotations_5q_seed73 | pennylane | 2.930 | 170.8 | 170.8 | 2.28e-16 | pass |
| qft_5q_seed73 | nexqus | 5.657 | 39.2 | 39.2 | 7.85e-17 | pass |
| qft_5q_seed73 | qiskit | 2.016 | 103.8 | 103.9 | 2.22e-16 | pass |
| qft_5q_seed73 | pennylane | 2.311 | 172.8 | 172.9 | 5.55e-17 | pass |
| diagonal_5q_seed73 | nexqus | 5.664 | 36.8 | 38.1 | 2.68e-16 | pass |
| diagonal_5q_seed73 | qiskit | 2.486 | 104.0 | 104.0 | 4.44e-16 | pass |
| diagonal_5q_seed73 | pennylane | 4.013 | 170.6 | 170.7 | 1.67e-16 | pass |
| amplitude_damping_5q_seed73 | nexqus | 10.182 | 38.1 | 38.2 | 6.21e-17 | pass |
| amplitude_damping_5q_seed73 | qiskit | 2.879 | 108.8 | 108.9 | 5.93e-17 | pass |
| amplitude_damping_5q_seed73 | pennylane | 15.220 | 169.4 | 169.4 | 1.08e-14 | pass |
| depolarizing_5q_seed73 | nexqus | 11.464 | 39.1 | 39.2 | 5.03e-17 | pass |
| depolarizing_5q_seed73 | qiskit | 2.397 | 105.7 | 105.8 | 8.89e-17 | pass |
| depolarizing_5q_seed73 | pennylane | 17.347 | 170.7 | 170.7 | 3.42e-14 | pass |
| rotations_5q_seed419 | nexqus | 4.861 | 37.7 | 37.7 | 2.00e-16 | pass |
| rotations_5q_seed419 | qiskit | 2.852 | 103.8 | 103.8 | 2.78e-16 | pass |
| rotations_5q_seed419 | pennylane | 3.059 | 170.9 | 170.9 | 1.70e-16 | pass |
| qft_5q_seed419 | nexqus | 5.866 | 39.1 | 39.2 | 9.02e-17 | pass |
| qft_5q_seed419 | qiskit | 1.957 | 107.1 | 107.1 | 2.22e-16 | pass |
| qft_5q_seed419 | pennylane | 2.263 | 170.9 | 170.9 | 0.00e+00 | pass |
| diagonal_5q_seed419 | nexqus | 5.453 | 38.7 | 38.7 | 2.78e-16 | pass |
| diagonal_5q_seed419 | qiskit | 2.113 | 107.5 | 107.5 | 6.11e-16 | pass |
| diagonal_5q_seed419 | pennylane | 3.264 | 171.9 | 172.0 | 2.95e-16 | pass |
| amplitude_damping_5q_seed419 | nexqus | 13.641 | 39.3 | 39.3 | 7.35e-17 | pass |
| amplitude_damping_5q_seed419 | qiskit | 2.367 | 105.4 | 105.4 | 1.13e-16 | pass |
| amplitude_damping_5q_seed419 | pennylane | 15.870 | 171.1 | 171.1 | 1.01e-14 | pass |
| depolarizing_5q_seed419 | nexqus | 12.461 | 38.9 | 38.9 | 5.31e-17 | pass |
| depolarizing_5q_seed419 | qiskit | 2.500 | 109.0 | 109.0 | 7.92e-17 | pass |
| depolarizing_5q_seed419 | pennylane | 16.434 | 170.7 | 170.7 | 3.68e-14 | pass |

Small circuits (up to 5 qubits) use an independent full-matrix NumPy oracle. Larger circuits use Qiskit Aer as the cross-simulator reference when available; the JSON labels every reference. Statevector errors are aligned for global phase. Density errors compare full matrix entries. Probability TVD, normalization, Hermiticity, and small-density eigenvalues are also checked.

These measurements describe this host and these workloads. Keep slower NexQus results; rerun on your own hardware before making performance claims. Prepared API overhead is included, so tiny circuits are often dominated by framework overhead.
