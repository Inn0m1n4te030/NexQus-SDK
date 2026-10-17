# Independent simulator comparison

Run: 2026-10-17T08:43:02.955371+00:00. CPU: AMD EPYC 9V74 80-Core Processor. Python 3.12.14.

Suite `standard`; precision `double`; 1 CPU thread(s); 2 warmups and 7 measured repetitions per engine/case.

Full specifications, raw timings, versions, settings, hardware and errors: [JSON](results.json). Tabular data: [CSV](results.csv).

Timing covers prepared execution and full state/probability output. Imports, circuit construction, and compilation are excluded and preparation is reported separately. Gate fusion and idle-wire truncation are disabled. Every engine uses the same gates, noise, precision, zero shots and CPU. PennyLane uses default.qubit/default.mixed; this is not a benchmark of lightning.qubit.

RSS is process resident memory, including native allocations. Execution peak is sampled every 1 ms and includes warmups plus the untimed validation run; short peaks may be missed. OS process peak also includes imports and preparation. Baselines differ between SDKs; incremental RSS can be zero when an allocator reuses memory. Sampling is active during timing, so its overhead is included. No GPU memory or hardware execution is measured.

| Circuit | Engine | Median ms | Execution peak MiB | Process peak MiB | Max state error | Accuracy |
|---|---|---:|---:|---:|---:|---|
| ghz_4q | nexqus | 0.099 | 37.2 | 37.3 | 0.00e+00 | pass |
| ghz_4q | qiskit | 0.480 | 107.4 | 107.5 | 1.11e-16 | pass |
| ghz_4q | pennylane | 0.303 | 170.7 | 170.7 | 0.00e+00 | pass |
| rotations_4q | nexqus | 1.248 | 38.1 | 38.2 | 2.88e-16 | pass |
| rotations_4q | qiskit | 1.616 | 107.5 | 107.5 | 3.48e-16 | pass |
| rotations_4q | pennylane | 2.981 | 169.7 | 169.7 | 3.19e-16 | pass |
| qft_4q | nexqus | 0.333 | 38.7 | 38.8 | 5.72e-17 | pass |
| qft_4q | qiskit | 0.748 | 103.8 | 103.8 | 1.75e-16 | pass |
| qft_4q | pennylane | 0.761 | 170.7 | 170.7 | 5.58e-17 | pass |
| diagonal_4q | nexqus | 1.005 | 37.8 | 37.9 | 3.08e-16 | pass |
| diagonal_4q | qiskit | 1.603 | 104.0 | 104.0 | 1.22e-15 | pass |
| diagonal_4q | pennylane | 3.629 | 172.3 | 172.3 | 2.86e-16 | pass |
| ghz_12q | nexqus | 0.205 | 38.6 | 38.6 | 1.11e-16 | pass |
| ghz_12q | qiskit | 0.679 | 107.3 | 107.4 | 0.00e+00 | pass |
| ghz_12q | pennylane | 0.995 | 170.8 | 170.8 | 1.11e-16 | pass |
| rotations_12q | nexqus | 9.064 | 39.4 | 39.5 | 5.97e-17 | pass |
| rotations_12q | qiskit | 6.678 | 108.3 | 108.3 | 0.00e+00 | pass |
| rotations_12q | pennylane | 16.172 | 170.6 | 170.7 | 6.14e-17 | pass |
| qft_12q | nexqus | 2.121 | 39.6 | 39.6 | 1.80e-16 | pass |
| qft_12q | qiskit | 4.254 | 107.6 | 107.6 | 0.00e+00 | pass |
| qft_12q | pennylane | 16.230 | 172.4 | 172.4 | 1.76e-16 | pass |
| diagonal_12q | nexqus | 6.620 | 38.3 | 38.4 | 5.83e-16 | pass |
| diagonal_12q | qiskit | 6.735 | 104.7 | 104.7 | 0.00e+00 | pass |
| diagonal_12q | pennylane | 18.512 | 171.6 | 171.6 | 5.83e-16 | pass |
| ghz_18q | nexqus | 9.150 | 52.5 | 56.4 | 1.11e-16 | pass |
| ghz_18q | qiskit | 18.095 | 110.3 | 114.1 | 0.00e+00 | pass |
| ghz_18q | pennylane | 8.723 | 184.5 | 186.5 | 1.11e-16 | pass |
| rotations_18q | nexqus | 542.374 | 58.0 | 57.9 | 1.37e-17 | pass |
| rotations_18q | qiskit | 317.561 | 117.3 | 121.1 | 0.00e+00 | pass |
| rotations_18q | pennylane | 402.133 | 186.9 | 186.8 | 1.20e-17 | pass |
| qft_18q | nexqus | 142.836 | 56.0 | 57.9 | 9.44e-17 | pass |
| qft_18q | qiskit | 227.874 | 111.1 | 114.9 | 0.00e+00 | pass |
| qft_18q | pennylane | 206.767 | 184.6 | 184.5 | 9.90e-17 | pass |
| diagonal_18q | nexqus | 388.473 | 57.9 | 58.0 | 3.06e-16 | pass |
| diagonal_18q | qiskit | 256.601 | 114.4 | 118.2 | 0.00e+00 | pass |
| diagonal_18q | pennylane | 394.102 | 183.6 | 183.4 | 3.12e-16 | pass |
| amplitude_damping_4q | nexqus | 4.890 | 39.2 | 39.3 | 1.13e-16 | pass |
| amplitude_damping_4q | qiskit | 1.392 | 109.1 | 109.1 | 1.57e-16 | pass |
| amplitude_damping_4q | pennylane | 11.713 | 170.6 | 170.6 | 2.27e-14 | pass |
| depolarizing_4q | nexqus | 7.406 | 38.9 | 39.0 | 9.93e-17 | pass |
| depolarizing_4q | qiskit | 1.448 | 105.7 | 105.7 | 1.14e-16 | pass |
| depolarizing_4q | pennylane | 10.238 | 170.9 | 171.0 | 6.85e-14 | pass |
| amplitude_damping_8q | nexqus | 156.523 | 45.4 | 45.4 | 1.74e-17 | pass |
| amplitude_damping_8q | qiskit | 41.675 | 107.1 | 108.1 | 0.00e+00 | pass |
| amplitude_damping_8q | pennylane | 423.884 | 173.4 | 173.4 | 3.56e-15 | pass |
| depolarizing_8q | nexqus | 235.452 | 45.9 | 45.8 | 2.08e-17 | pass |
| depolarizing_8q | qiskit | 41.489 | 110.3 | 111.3 | 0.00e+00 | pass |
| depolarizing_8q | pennylane | 620.912 | 172.7 | 172.7 | 1.22e-14 | pass |

Small circuits (up to 5 qubits) use an independent full-matrix NumPy oracle. Larger circuits use Qiskit Aer as the cross-simulator reference when available; the JSON labels every reference. Statevector errors are aligned for global phase. Density errors compare full matrix entries. Probability TVD, normalization, Hermiticity, and small-density eigenvalues are also checked.

These measurements describe this host and these workloads. Keep slower NexQus results; rerun on your own hardware before making performance claims. Prepared API overhead is included, so tiny circuits are often dominated by framework overhead.
