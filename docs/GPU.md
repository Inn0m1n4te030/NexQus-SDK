# CPU and CUDA across NexQus

These changes require the current source checkout; the published 1.2.0 wheel
does not contain them. Core installation still needs only NumPy. CUDA requires
an NVIDIA GPU, working driver and a CUDA-enabled PyTorch build. This work does
not provision cloud GPUs or submit QPU jobs. Apple MPS is not supported.

## Windows / Miniconda setup

Activate the environment you use for NexQus. Check `nvidia-smi`, then use the
[official PyTorch installer selector](https://pytorch.org/get-started/locally/)
for Windows, Pip and a CUDA build compatible with your driver/GPU. The earlier
image lesson's `--index-url .../cpu` installs CPU-only Torch; that build cannot
execute CUDA even when the machine has a GPU. If it is installed, replace Torch
using the CUDA command from the selector, adding `--upgrade --force-reinstall`.
Install compatible torchvision from the same index if using CIFAR-10.

Then, from the current NexQus checkout:

```powershell
python -m pip install -e ".[gpu,image-gan,tutorial]"
python -m nexqus.devices
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

Do not continue with a CPU-only Torch build when you intend CUDA execution.
Restart your Jupyter kernel after replacing Torch. Native Windows supports the
NexQus simulator, image models and fixed-H2 VQE on CUDA without PySCF. Generating
new molecular integrals via PySCF/GPU4PySCF requires Linux/macOS/WSL as described
below. The `gpu` extra cannot select your CUDA wheel/driver for you.

## Select a device consistently

```python
from nexqus import Simulator, Circuit, Parameter, QuantumFunction, QuantumDistribution

sim = Simulator.for_device('cuda:0')
result = sim.run(Circuit(2).h(0).cx(0, 1), shots=0, native_result=True)
print(result.probabilities.device)       # cuda:0; no state export
print(result.expectation('ZZ'))         # reduction on CUDA; one Python scalar
host = result.to_numpy()                # explicit host export for plotting

theta = Parameter('theta')
f = QuantumFunction(Circuit(1).ry(theta, 0), 'Z', device='cuda:0')
print(f.value_and_grad({'theta': .5}))
p = QuantumDistribution(Circuit(1).ry(theta, 0), device='cuda:0', native=True)
print(p({'theta': .5}).device)
```

`cpu` selects NumPy by default; `backend='torch', device='cpu'` selects Torch
CPU for parity testing. `cuda`/`cuda:N` select Torch and fail if unavailable.
Only `auto` permits choosing CPU when CUDA is unavailable. Results record the
actual backend/device. A supplied `simulator=` cannot be combined with separate
backend/device/precision options.

For code/examples that construct a default `Simulator()` or default quantum
functions, set an opt-in default in PowerShell:

```powershell
$env:NEXQUS_DEVICE = "cuda:0"
python examples/quickstart.py
python examples/qft.py
python examples/parameter_sweep.py
python examples/gradients.py
Remove-Item Env:NEXQUS_DEVICE
```

Explicit device/backend choices override this default. Existing public results
remain NumPy arrays unless `native_result=True` is requested. Counts/classical
records remain host Python dictionaries. Dynamic branch bookkeeping and final
histogram exports can synchronize with the host; their quantum state/channel
evolution uses the selected device. Circuit construction, parameter dictionaries,
control flow, file/data loading, scalar optimizers and rendering remain host work.

## Coverage and boundaries

| Area | Device execution | Host work / boundary |
|---|---|---|
| Static statevectors, density matrices, noise and permutations | Existing Torch CPU/CUDA kernels | Gate construction and default NumPy result export |
| Trajectories, measurements/reset/feed-forward | Quantum evolution on CPU/CUDA | Classical branch bookkeeping/count dictionaries |
| Compiled circuits and sweeps | Reuse selected simulator/device | Binding/iteration and optional exported results |
| QFT, Grover and Shor | Execute their circuits on selected simulator | Algorithm construction, continued fractions and factor recovery |
| QuantumFunction gradients | State evolution and expectation reductions on device | Scalar gate shifts/chain rule and dictionary optimizer |
| Regression/classification | Noiseless Torch path batches samples and differentiates on device | Small parameter-dictionary update; public predictions export to NumPy |
| Discrete QGAN | Probabilities, Jacobians, adversarial vector losses and metrics on device | Scalar dictionary optimizers; public default probability export |
| Patch/hybrid/classical image GANs | Models, batched circuit states, autograd and Adam on device | Image decoding/loading, minibatch transfer, shuffle order, reports |
| Hamiltonian/VQE | One evolved state per value/shift; weighted Pauli reductions on device | Scalar optimization/control |
| Bounded dense Hamiltonian reference | Torch matrix construction and eigensolve on selected device | Electron-sector index construction |
| Molecular RHF | Optional GPU4PySCF `scf_device='cuda:N'` | AO/MO integral generation and Jordan-Wigner mapping remain CPU |
| Circuit explanations | Gate/state evolution on selected device | Snapshot diagnostics and HTML/JSON export on CPU |
| Circuit drawings / molecular geometry viewer | No simulation required | Browser/host rendering; these are not CUDA compute workloads |
| External Braket adapter | Provider's selected execution target | A local SDK device option does not move a remote job to your GPU |

Noisy QuantumFunction learning retains the existing parameter-shift path. The
batched differentiable `TorchCircuit` path supports static pure states, fixed
unitaries/diagonals/permutations and symbolic RX/RY/RZ/P/CP arithmetic. It rejects
noise/dynamic operations; use `Simulator` for those instead. Small circuits can
be slower on GPU due to launch overhead; memory still grows exponentially.

## Image GANs

Append `--device cuda:0` to your existing command:

```powershell
python examples/image_gan.py --mode hybrid --device cuda:0 --data data/pets --size 16 --channels 3 --samples 10000 --steps 5000 --batch-size 16 --hidden 256 --latent-dim 16 --qubits 6 --depth 2 --learning-rate 0.0002 --output-dir results/pets-hybrid-cuda
```

`--mode all`, `patch` and `classical` also use the selected device. This adds
acceleration, not a validated improvement in animal-image quality. The existing
small image architectures still need data/training/model-quality evaluation.

```python
from nexqus.image_gan import HybridImageGenerator, ImageGAN
trainer = ImageGAN(HybridImageGenerator(), device='cuda:0')
# Or move a standalone generator with .to('cuda:0') before constructing trainer.
trainer.save('model.pt')
same_device = ImageGAN.load('model.pt', device='cuda:0')
on_cpu = ImageGAN.load('model.pt', device='cpu', reset_rng=True)
```

Same-device checkpoints restore optimizers, RNG streams and partial-epoch order.
CPU and CUDA use different noise RNG algorithms; cross-device continuation
requires `reset_rng=True` / CLI `--reset-rng`, and does not claim bitwise identity.
Old schema-1 CPU checkpoints load with circuit buffers rebuilt from the stored
model configuration; the new tensor algorithm can introduce numerical changes.
Do not move models after constructing a trainer; load on the desired device.
Sampling and parameter interventions return tensors on the training device.
Image files/plots explicitly export those tensors to CPU.

## General differentiable circuit tensors

```python
import torch
from nexqus import Circuit, Parameter
from nexqus.torch_circuit import TorchCircuit

q = TorchCircuit(Circuit(2).ry(Parameter('x'), 0).cx(0, 1), device='cuda:0')
angles = torch.tensor([.2, .5, .8], dtype=torch.float64, device='cuda:0', requires_grad=True)
probabilities = q({'x': angles})   # (batch, 2**n); all states stay on CUDA
loss = probabilities[:, 0].sum()
loss.backward()
print(angles.grad)
```

Shared scalar weights and 1D per-sample angles can be mixed. Bindings must match
the module's device/float32 or float64 dtype; `.float()`, `.double()` and `.to()`
move its registered buffers. State allocation checks the configured
`max_qubits` and estimated `max_memory_mb`; autograd graph storage adds cost.

## Chemistry: separate the two stages

Frozen-H2 VQE and visualization work on native Windows:

```powershell
python examples/molecular_viewer.py --device cuda:0
```

```python
from nexqus import Parameter
from nexqus.chemistry import h2_reference_problem, h2_pair_ansatz
problem = h2_reference_problem()
energy = problem.hamiltonian.function(h2_pair_ansatz(Parameter('theta')), device='cuda:0')
print(energy.value_and_grad({'theta': .4}))
print(problem.exact_energy(device='cuda:0'))
```

On Linux/WSL, install `.[chemistry]` to generate new molecular Hamiltonians.
GPU4PySCF is a separate optional dependency with CUDA-specific packages; follow
its [official installation guide](https://pyscf.org/user/gpu.html) and compatible
CuPy/cuTENSOR requirements. It is not installed automatically into native Windows.
Select it explicitly:

```bash
python examples/h2_vqe.py --device cuda:0 --scf-device cuda:0
```

The API is `molecular_hamiltonian(..., scf_device='cuda:0')`. This dispatches
RHF to GPU4PySCF inside the requested CuPy device context, then converts the
mean-field object to CPU for integral transformation/mapping. Missing GPU4PySCF
or CUDA raises; it never silently substitutes CPU SCF. The H2 example's
independent PySCF FCI reference remains CPU. `--device cuda:0 --scf-device cpu`
accelerates VQE while retaining the standard CPU molecular driver.

## Validate actual hardware

CPU parity tests are not proof of CUDA execution. On your GPU machine run:

```powershell
python scripts/check_gpu_workflows.py --device cuda:0
```

This fails on CPU-only Torch or any skipped/failed test, and writes
`.release-artifacts/cuda_workflows.json` with GPU identity, Torch/CUDA versions,
tests, failures/skips and hardware-validation status. It includes the full core
suite and image forward/backward/training/checkpoint tests. Add `--include-scf`
on a Linux/WSL GPU4PySCF environment to require a live CPU/GPU RHF parity check.
GPU4PySCF is otherwise recorded as not hardware validated.

The manual `CUDA hardware validation` workflow requires an existing self-hosted
runner labelled `gpu`; it creates no cloud resources and never runs automatically
on pull requests. Ordinary CI tests NumPy and Torch CPU, including Windows, and
reports unavailable CUDA tests as skips. This development environment has no
CUDA device: live CUDA and GPU4PySCF execution remain unverified until that strict
hardware check passes.
