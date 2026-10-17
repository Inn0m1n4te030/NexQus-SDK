# NumPy optimization: matched before/after measurements

Both versions were rerun with the corrected performance harness on the same host and Python environment: double precision, one CPU thread, three untimed warmups and nine measured executions per engine/case. Memory sampling runs in a separate fresh subprocess. All specifications, settings and benchmark source hashes match.

The baseline runtime is the unoptimized SDK from commit `89050b4e35194a2587ebceb4e64a6be80a0b89b2`. It was imported from an extracted source snapshot; the optimized source was then measured with the same harness. These speedups compare code changes under a common protocol, rather than comparing to the earlier mixed timing/memory reports.

See [unoptimized baseline](baseline_cpu_20261017/report.md), [optimized three-simulator comparison](cpu_20261017/report.md), and [correctness-only checks](correctness_20261017/report.md). Raw repetitions and memory results are linked from each report.

| Workload | NexQus before ms | NexQus after ms | Before/after ratio | Aer after ms | PennyLane after ms |
|---|---:|---:|---:|---:|---:|
| ghz_4q | 0.075 | 0.043 | 1.74× | 0.389 | 0.242 |
| rotations_4q | 1.122 | 1.147 | 0.98× | 1.429 | 2.885 |
| qft_4q | 0.283 | 0.243 | 1.17× | 0.884 | 0.721 |
| diagonal_4q | 0.887 | 0.822 | 1.08× | 1.534 | 3.392 |
| ghz_12q | 0.186 | 0.227 | 0.82× | 0.749 | 0.688 |
| rotations_12q | 7.710 | 6.500 | 1.19× | 6.592 | 12.855 |
| qft_12q | 2.064 | 1.941 | 1.06× | 4.366 | 16.761 |
| diagonal_12q | 5.414 | 4.544 | 1.19× | 6.591 | 16.011 |
| ghz_18q | 8.950 | 7.079 | 1.26× | 18.966 | 9.511 |
| rotations_18q | 555.611 | 278.928 | 1.99× | 295.237 | 358.512 |
| qft_18q | 142.409 | 119.342 | 1.19× | 210.719 | 203.260 |
| diagonal_18q | 373.121 | 181.188 | 2.06× | 253.671 | 389.943 |
| amplitude_damping_4q | 4.454 | 1.678 | 2.65× | 1.482 | 8.687 |
| depolarizing_4q | 7.143 | 1.890 | 3.78× | 1.325 | 10.127 |
| amplitude_damping_8q | 149.594 | 83.420 | 1.79× | 42.745 | 419.933 |
| depolarizing_8q | 258.643 | 65.672 | 3.94× | 46.040 | 586.024 |

A ratio above 1 means NexQus was faster after optimization; below 1 means slower. All workloads are retained, including regressions and cases still slower than another simulator. Medians describe this host/run; consult raw ranges and standard deviations before interpreting small differences.

Changes: value-detected diagonal single-wire operations; reusable NumPy multiply/add buffers; direct exact density updates for Pauli mixtures and amplitude/phase damping; cached compiled channel recognition; no unused RNG setup for exact unseeded zero-shot static execution. No gate fusion, reduced precision, noise changes, or omitted operations are used.

Correctness reports contain no latency measurements. Performance uses equal shared NumPy random initialization and equal SDK warmups. Cold-start/preparation costs are excluded from steady-state timing and preparation is reported separately in the JSON/CSV.
