# Performance and numerical comparison

This guide applies to NexQus SDK 1.1.0. For order-finding workloads, call
`algorithms.shor_resources(modulus)` before simulation. Compact permutation gates
avoid dense modular matrices; state storage still grows exponentially in the
total qubit count. The API reference describes counting-precision and memory
budget controls. Drawing a circuit does not allocate or evolve a quantum state.

## Install benchmark tools

From the NexQus 1.1.0 source checkout:

```bash
python -m pip install --upgrade -e '.[benchmark]'
python -m unittest discover -s tests -v
python examples/validate_gpu.py
```

The benchmark extra adds threadpoolctl for controlling BLAS threads. Keep your
existing CUDA-capable PyTorch installation. The strict GPU validator fails if
PyTorch/CUDA is missing or a test is skipped.

## Re-run your workloads

```bash
python examples/benchmark_v3.py --compare --qubits 20 --cpu-threads 1 --output benchmark_20q_t1.json
python examples/benchmark_v3.py --compare --qubits 20 --cpu-threads 4 --output benchmark_20q_t4.json
python examples/benchmark_v3.py --compare --method trajectory --qubits 10 --noise 0.01 --trajectories 512 --batch-size 32 --cpu-threads 1 --random-engine native --output benchmark_noise.json
python examples/validate_noise.py --backend torch --device cuda:0 --random-engine native
```

`benchmark_v2.py` remains an alias-compatible entry point to the updated runner.
CPU thread controls set runtime BLAS limits and Torch intra-op threads. The
effective settings are reported. They apply only within the benchmark process.
CUDA still uses a CPU host, so CPU-thread limits can affect GPU timings too.
The benchmark warms up every configuration and excludes separate validation
runs from timing. CUDA peak allocation is captured before retained-state
validation, and includes live cached allocations.

## Execution paths

- Noise probabilities, sampling and branch selection stay on the chosen device.
  The trajectory loop has no per-channel `.cpu()`, `.numpy()` or `.item()` reads.
- Recognized Pauli mixtures apply selected I/X/Y/Z transformations directly,
  without computing state-dependent Kraus norms. Detection uses actual Kraus
  values, not a user-provided channel name.
- Canonical amplitude/phase damping uses analytic branch populations and local
  updates. Other custom Kraus channels use device-local generic sampling.
- NumPy single-qubit gates use direct amplitude-pair updates, avoiding tiny BLAS
  matrix multiplications. Diagonal gates broadcast over the tensor without
  flattening target axes. General multi-qubit matrices retain the fallback.
- CX gates use amplitude-slice swaps on CPU and Torch, detected from the actual
  unitary matrix. This avoids dense multiplication for common entangling gates.
- The initial state is transferred once per run; trajectory batches expand that
  device state. Probabilities accumulate on device and transfer once at the end.
- Retained output states avoid an extra host copy. Outputs remain NumPy arrays.

These are tensor-level optimizations. There is no fused CUDA extension,
distributed engine, native compiler, or universal speedup guarantee. Torch
single-qubit unitaries still use Torch matrix operations. Benchmark on the
target GPU and circuit, including both CPU backends.

## Randomness and compatibility

`Simulator.run(..., random_engine='portable')` is the default. It uses a seeded
32-bit counter mixer implemented on NumPy/Torch/CUDA, with matching uniform
draws across backends. It supports fewer than 2**32 noise-branch draws per run.
This is a noncryptographic simulation PRNG with basic distribution sanity
tests; it has not undergone large-scale statistical certification.

`random_engine='native'` selects NumPy's generator or a seeded Torch generator
on the requested device. Native mode may have lower kernel-launch overhead;
it is intended for performance experiments. A native CUDA generator draws
directly on the GPU. Native streams are not paired across backends.

Reproducibility requires a fixed SDK version,
engine, seed, backend, precision and batch layout. Changing batch size changes
which draws belong to which trajectory. Final classical shot sampling remains
on CPU and has its own seeded stream.

## Two different validation questions

### Paired numerical parity

`nexqus.validation.compare_results()` checks max probability error and total
variation distance. Double precision limits are 1e-9 and 1e-7; single precision
limits are 2e-5 and 1e-4. Trajectory results must have matching portable engine, seed,
trajectory count, batch size and fast-path configuration.

The benchmark performs a separate untimed portable validation run even when it
times native RNG. Validation error fields describe that paired run, not native
timed samples. Ideal and density-matrix validation retains states, catching
phase errors invisible in probabilities. Statevectors allow global phase.

Floating-point differences near a sampling threshold can produce branch
divergence, particularly in single precision. A strict failure should be
investigated, including a double-precision rerun; it is not silently waived.

### Statistical agreement with the exact noise model

`validate_noise.py` uses a small three-qubit circuit and compares trajectories
with deterministic density-matrix evolution. Its Hoeffding + union bound is
labelled separately and presumes independent trajectories. This is a
conservative sanity check, not the CPU/GPU numerical threshold or proof of
PRNG quality. Increase trajectories or repeat seeds for further experiments.

## Profiling switches

- `Simulator(local_gates=False)` or benchmark `--no-local-gates`: generic
  matrix/permutation kernels instead of local/broadcast kernels.
- `run(noise_fastpath=False)` or benchmark `--no-fast-noise`: generic Kraus
  sampling on device instead of specialized channels.
- `--cpu-threads 1`, `4`, or another positive count: controlled CPU comparisons.
- `--uncompiled`: ordinary execution with the bounded gate cache.

Compare one change at a time, with the same precision, circuit, repetitions,
trajectory count and batch size. Higher batches require more memory. These
changes do not remove exponential statevector/density storage requirements.

## Burmese notes

GPU အသုံးပြုရာတွင် noise branch ရွေးချယ်မှုကို GPU ပေါ်တွင် ပြုလုပ်ပါသည်။
CPU သို့ gate တစ်ခုချင်းစီ၏ probability များ ပြန်ပို့ရန် မလိုအပ်တော့ပါ။

`--cpu-threads 1` နှင့် `--cpu-threads 4` ကို သီးခြားစမ်းသပ်ပြီး JSON ရလဒ်များကို
နှိုင်းယှဉ်ပါ။ GPU ပိုမြန်မည်ဟု ကြိုတင်မယူဆဘဲ မိမိစက်ပေါ်တွင် တိုင်းတာပါ။

Random ရလဒ်ကို ပြန်လည်စမ်းသပ်ရန် SDK version နှင့် setting များကို မှတ်သားပါ။ CPU/GPU numerical check အတွက်
`portable` engine ကို သုံးပြီး performance စမ်းသပ်ရာတွင် `native` ကို ရွေးနိုင်ပါသည်။
