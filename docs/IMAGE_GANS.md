# Image QGANs: patches and quantum features

These **unreleased source-checkout APIs** add both image-generation approaches,
conditional class labels, a classical GAN comparison and pixel explanations.
They are not in the published PyPI 1.2.0 wheel. Import from `nexqus.image_gan`;
core `import nexqus` still requires only NumPy. Training supports local CPU/CUDA.
See [GPU.md](GPU.md) before installing Torch if you want an NVIDIA GPU: the CPU
index command below deliberately installs CPU-only Torch.

## Install on Windows, Linux or macOS

Activate your environment, open the current checkout, and run:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[tutorial,image-gan]"
python -c "import sys, nexqus; from nexqus.image_gan import ImageGAN; print(sys.executable, nexqus.__file__)"
```

These commands work in PowerShell. PySCF/the chemistry extra are not needed.
If using Jupyter, register the environment with `python -m ipykernel install
--user --name nexqus --display-name "Python (NexQus)"`, restart the kernel and
select **Python (NexQus)**. Use lesson [11](../notebooks/11_image_gans.ipynb).
If the import points into an older `site-packages/nexqus`, install with that
kernel's Python; in a notebook `%pip install -e ".[tutorial,image-gan]"`
from the checkout, then restart. Version 1.2.0 alone cannot identify source APIs.

`image-datasets` adds optional torchvision. Install torch and torchvision from
the **same index** to obtain a compatible pair when using CIFAR-10:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Choose a generator

| Generator | Quantum component | Classical component | Image formation |
|---|---|---|---|
| `QuantumPatchGenerator` | One circuit per channel/patch; trainable RY/RZ/RX layers and CNOT entanglers | Latent-noise encoder and label embedding | Ancilla=0 probabilities become pixels; patches tile into an image |
| `HybridImageGenerator` | A trainable circuit returns Z expectation features | Angle encoder, label embedding and an MLP decoder | Decoder maps quantum features to sigmoid pixels |
| `ClassicalImageGenerator` | None | Conditional MLP | Noise and one-hot labels map to sigmoid pixels |

All accept `(noise, labels)` with float tensors shaped `(batch, latent_dim)`
and int64 labels shaped `(batch,)`, matching model device/dtype. Output shape is `(batch, channels, size,
size)`, with pixels in [0,1]. The label input permits requesting either class;
training and evaluation determine how well the generated images follow it.
The hybrid decoder receives noise/class information **through the circuit's
features**. Its pixel decoding remains classical. Neither quantum architecture
is entirely quantum: both learn a classical angle encoder and label embedding.

Patch probabilities are conditional on the last qubit being zero and normalized
by their maximum. Every nonzero patch has a brightest pixel of one. This imposes
a brightness constraint and can create patch seams. No classical image decoder
is hidden in this path. A 16×16 RGB image with 4×4 patches uses 48 separate
five-qubit circuits. The default 8×8 grayscale example uses four.

## First run: both models plus a classical GAN

```bash
python examples/image_gan.py --mode all --steps 20
```

This offline test uses **procedural horizontal/vertical bars**. Open
`.release-artifacts/image-gans/report.html` to see actual real/before/after
images, adversarial losses, parameter effects, settings and pixel diagnostics.
Each class occupies a row. Before/after use identical noise and labels.
Click through to a gate-by-gate report of the first patch/feature circuit.
The default is a short training demonstration, not a pretrained animal model.
An actual 200-update bars comparison is bundled in
[`examples/image_previews/report.html`](../examples/image_previews/report.html)
with its two circuit reports. Download the folder and open the HTML locally.
That saved preview predates batched Torch execution; its embedded timing and
precision settings describe the earlier implementation, not the new CUDA path.
Regenerate the report to measure your installed version and actual device.

## Train cats and dogs

Create `data/pets/cats/` and `data/pets/dogs/` and add your image files, then run:

```bash
python examples/image_gan.py --data data/pets --mode all --size 16 --channels 3 --samples 256 --steps 100 --batch-size 4 --output-dir results/pets
```

The folder loader sorts classes/files, takes at most `samples // class_count`
per class, corrects EXIF orientation, center-crops and resizes with bilinear
interpolation, and scales pixels to [0,1]. There is no augmentation. A seeded
stratified 80/20 split keeps training and validation sample indices disjoint.
Remove duplicates in your source images to avoid content leakage across splits.
Every class needs at least two images. Labels follow sorted folder names.

Alternatively, explicitly download the public CIFAR-10 dataset (~170 MB):

```bash
python examples/image_gan.py --cifar10 data/cifar10 --download --mode all --size 16 --channels 3 --samples 256 --steps 100 --batch-size 4 --output-dir results/cifar-pets
```

This selects genuine CIFAR-10 cat/dog images (original class IDs 3/5 → labels
0/1), then makes the same local train/validation split from the official training
set. `load_cifar10_cats_dogs(..., train=False)` selects the official test set
for separate later evaluation. Downloads never happen without `download=True`
or `--download`. No trained weights are bundled. Low-resolution recognizable
cats/dogs require substantial training and model selection; a few updates verify
the pipeline. Full-resolution photorealistic pet generation is beyond this
small CPU implementation.

## Python API

```python
import torch
from nexqus.image_gan import (
    ImageDataset, QuantumPatchGenerator, HybridImageGenerator,
    ClassicalImageGenerator, ImageGAN, save_image_grid,
)

data = ImageDataset.from_folder('data/pets', image_size=16, channels=3, max_images=256)
train, validation = data.split(seed=7)
generator = HybridImageGenerator(image_size=16, channels=3, latent_dim=4,
                                 num_classes=len(data.class_names), n_qubits=4)
# Or QuantumPatchGenerator(..., patch_size=4), with no n_qubits argument.
trainer = ImageGAN(generator, learning_rate=.002, seed=7)
trainer.fit(train, epochs=20, batch_size=4, max_steps=100)
cats = trainer.sample(16, labels=torch.zeros(16, dtype=torch.long), seed=42)
save_image_grid(cats, 'cats.png')
trainer.save('hybrid.pt')
```

`fit` performs alternating discriminator and generator updates. Losses use
balanced binary cross entropy for D and a non-saturating objective for G.
The two networks use Torch Adam with betas (.5,.999). `history` records losses
at their respective updates; G's recorded generated mean precedes its update.
Losses need not decrease monotonically and do not measure semantic image quality.

`fit(epochs=...)` finishes a partially completed epoch as its first epoch.
`max_steps` caps updates **in this call**, rather than the lifetime step count.
The CLI sets enough epochs to perform exactly the requested session updates.

## Resume exactly

```python
continued = ImageGAN.load('hybrid.pt')
continued.fit(train, epochs=1, batch_size=4, max_steps=10)
```

Checkpoints include both models, both optimizer states, local noise/shuffle
RNGs, epoch, history, current permutation and cursor. A SHA256 fingerprint of
preprocessed images/labels/class names rejects silent continuation on changed
data. Keep batch size fixed within a partial epoch. Fixed-seed `sample` calls
do not alter the training RNG. Loading uses `torch.load(weights_only=True)`
and reconstructs supported model types from a primitive config; no custom
model object is pickled. The save replaces its destination atomically.
Exact continuation is tested in the same software environment; cross-version
bitwise reproducibility is not promised.
CUDA training uses a CUDA noise generator. `ImageGAN.load(..., device='cuda:0')`
restores that device's training state. Moving CPU↔CUDA requires `reset_rng=True`
(`--reset-rng` in the CLI) because those noise streams differ. Old schema-1 CPU
weights load, but numerical trajectories can change with the new Torch algorithm.

For CLI resume use one mode, the same data/split/model/training flags, and
`--resume results/pets/hybrid/checkpoint.pt`. For example append it to the
cat/dog command above and change `--mode all` to `--mode hybrid`.

## Explain what the circuit changes

```python
noise = torch.tensor([[.2, -.4, .1, .8]])
label = torch.tensor([0])
effect = generator.parameter_effect(noise, label, index=0, delta=.25)
# minus, center, plus: actual image tensors; sensitivity: (plus-minus)/(2*delta)
generator.circuit_report(noise, label).to_html(filename='image_circuit.html')
```

The experiment holds noise/class fixed, intervenes on one circuit weight, then
restores the original weights and training mode. Use it between training steps.
The signed sensitivity is a **local parameter intervention**, not a dataset-wide
attribution or evidence of cat/dog understanding. The circuit report shows exact
states at bound angles/weights; for patch models it traces only the first patch.
Classical decoding and patch assembly lie outside that state trace, and are
shown by the pixel intervention. For RGB plots the displayed sensitivity is the
mean of signed channel sensitivities, so opposite channel changes can cancel.

## Comparison and output interpretation

`--mode all` holds the data/split, shuffled batches, latent-noise draws, D
initialization/architecture, losses, learning rate and update budget fixed.
It prints and saves the generator/quantum parameter counts; these are **not**
matched between architectures. It is one seeded configuration, not exhaustive
model tuning. Train several seeds and judge sample diversity and class fidelity
before drawing quality conclusions. Both quantum and classical parameters are
optimized in the quantum models.

`metrics.json` includes model/trainer configs, versions, platform, threads,
data hashes, losses, parameter counts, and measured training-loop seconds.
Those seconds include batched circuit autograd and optimizers; they exclude setup,
sampling and export. This is a workload measurement, not an isolated simulator
benchmark. Pre-training preview calls warm each generator before training.
Per-class held-out mean-pixel MAE and generated per-pixel standard deviation are
simple diagnostics: a blurry mean image can score well while looking poor.
There is no FID, pretrained classifier accuracy or claim of quantum advantage.

## Supported limits and validation

- CPU/CUDA float32 training; `QuantumProbabilityLayer` also supports float64.
  Batched native Torch circuits evolve complex64/complex128 states and use
  Torch autograd, including gradients into the classical angle encoder. NumPy
  and parameter shifts remain independent numerical oracles in tests.
- This differentiable image path supports exact static pure states, not noisy,
  finite-shot, dynamic-circuit or QPU training. Other simulator paths support
  CPU/CUDA noise and dynamic circuits as documented in [GPU.md](GPU.md).
- Images 4..64 square, 1 or 3 channels, batch 1..128, latent dimension ≤64,
  ≤32 labels. Patch size 2/4/8, at most 64 circuits/image; quantum layers ≤8
  qubits and depth ≤6. Batched autograd replaces per-sample parameter shifts
  in image training. State and gradient graph storage still grow with qubits,
  batch size, gate count and number of patch circuits.
- Optional tests independently check finite differences against Torch
  `gradcheck`, probability values, patch placement, encoder/circuit/decoder
  gradients, training updates, exact mid-epoch reload, labels, data split and
  RNG isolation. CI runs the image tests, all three CLI modes and lesson 11
  through real Jupyter on Linux/Python 3.12 and Windows/Python 3.13.

```bash
python -m unittest discover -s tests/image_gan -v
python scripts/run_notebooks.py --engine jupyter --kernel nexqus --require-image-gan
```

Method references: [PennyLane's quantum patch GAN tutorial](https://pennylane.ai/demos/tutorial_quantum_gans),
[PyTorch custom autograd](https://docs.pytorch.org/docs/stable/notes/extending.html),
and [torchvision CIFAR-10](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html).
NexQus builds and executes its own circuits; PennyLane/Qiskit are not dependencies.
