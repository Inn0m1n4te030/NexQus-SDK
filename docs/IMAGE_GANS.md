# Image GANs: convolutional models and quantum features

These **1.3.0 APIs** add both image-generation approaches,
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
| `HybridImageGenerator` | A trainable circuit returns Z expectations or full probabilities | Angle encoder, label embedding and an MLP or convolutional decoder | Decoder maps quantum features (optionally plus classical noise) to pixels |
| `ClassicalImageGenerator` | None | Conditional MLP or convolutional generator | Noise and one-hot labels map to pixels |

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

The earlier small MLP example produced coloured textures on a user's 10,000-image
pet dataset after 5,000 updates. It did not validate recognizable cat/dog
synthesis. More steps alone are not an established fix. The CLI now defaults
to `--architecture conv`: a learned 4×4 feature grid followed by nearest-neighbor
upsampling, 3×3 convolutions, BatchNorm and ReLU. The discriminator learns
strided convolutional features conditioned on spatial label planes and a class-projection score. With legacy settings, the hybrid
uses the same decoder design but receives Z features through its quantum
circuit. Optional probability readout and a classical latent bypass are
described in the regularized-training section below. These are new, untrained architectures;
real-pet generation quality still needs a full dataset experiment.

Create `data/pets/cats/` and `data/pets/dogs/`. Start with the classical baseline:

```powershell
python examples/image_gan.py --mode classical --architecture conv --data data/pets --size 64 --channels 3 --samples 10000 --epochs 100 --batch-size 64 --hidden 256 --latent-dim 64 --learning-rate 0.0002 --data-seed 7 --horizontal-flip --preview-every 500 --checkpoint-every 1000 --output-dir results/pets-conv-classical
```

This is a starting experiment, not a validated recipe for recognizable pets.
With 8,000 training images and batch size 64 there are 125 updates per epoch,
so 100 epochs means 12,500 updates. `--steps 5000` means 5,000 updates, not
5,000 epochs. CPU can run this command; use `--size 32 --hidden 128` for a
smaller CPU experiment. A 16×16 thumbnail discards much of the face detail.
Face-centred source crops reduce background variation; the loader does not
detect or align animal faces for you.

For an NVIDIA GPU with CUDA-enabled Torch append `--device cuda:0`; see
[GPU.md](GPU.md). The GPU change does not by itself improve image quality.
10,000 RGB images at 64×64 occupy about 469 MiB as float32; preprocessing and
split copies need additional host memory. Data loading remains in memory.

For the hybrid comparison use the **same command**, replacing `--mode classical`
with `--mode hybrid --qubits 6 --depth 2` and changing the output directory.
This preserves the data, augmentation, discriminator and update budget.
Generator parameter counts differ and are printed. It does not establish a
quantum advantage. The quantum patch model remains a separate bounded pixel
experiment; `--mode all` at 64×64 RGB exceeds its 64-circuit limit. Use separate
classical/hybrid runs for this resolution.

During training, inspect `classical/progress/step_XXXXXXXX.png` (or `hybrid/`).
They use fixed noise and requested labels, so changes are comparable over time.
`checkpoint.pt` is refreshed at the requested interval; the final `report.html`
embeds real training/validation images, progress samples, final samples and
loss curves. No semantic quality score or automatic success verdict is inferred
from loss/pixel statistics. A larger model may still fail to learn useful images.

`--data-seed 7` randomly selects the capped file subset reproducibly within
each class; omit it for the older sorted-file selection. Files have EXIF
orientation corrected, are center-cropped and resized bilinearly, and scaled
to [0,1]. `--horizontal-flip` adds seeded training-only random flips; it does
not alter validation images or sampling. A seeded stratified 80/20 split
keeps sample indices disjoint. Remove duplicate source files before splitting
to avoid content leakage. Every class needs at least two images; labels follow
sorted folder names. Keep data selection flags unchanged when resuming.

Alternatively, explicitly download CIFAR-10 (~170 MB):

```bash
python examples/image_gan.py --cifar10 data/cifar10 --download --mode classical --architecture conv --size 32 --channels 3 --samples 10000 --epochs 100 --batch-size 64 --hidden 128 --latent-dim 64 --learning-rate 0.0002 --output-dir results/cifar-pets
```

CIFAR-10 cat/dog IDs 3/5 become labels 0/1, using a local train/validation
split of the official training set. `load_cifar10_cats_dogs(..., train=False)`
selects the official test set for a separate evaluation. Downloads require
explicit permission through `--download`; no trained weights are bundled.

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
                                 num_classes=len(data.class_names), n_qubits=4,
                                 architecture='conv', hidden=128)
# Or QuantumPatchGenerator(..., patch_size=4), with no n_qubits argument.
trainer = ImageGAN(generator, learning_rate=.0002, seed=7, horizontal_flip=True)
trainer.fit(train, epochs=20, batch_size=4, max_steps=100)
cats = trainer.sample(16, labels=torch.zeros(16, dtype=torch.long), seed=42)
save_image_grid(cats, 'cats.png')
trainer.save('hybrid.pt')
```

`fit` performs alternating discriminator and generator updates. Losses use
balanced binary cross entropy for D and a non-saturating objective for G.
The two networks use Torch Adam with betas (.5,.999). The API defaults to the
legacy MLP for compatibility; pass `architecture='conv'` explicitly. The CLI
defaults to convolutional models. A trainer infers its discriminator architecture
from the generator unless `discriminator_architecture=` is supplied (use this
explicitly for a convolutional discriminator with the patch generator). `history` records losses
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
RNGs (including training augmentation), epoch, history, current permutation and cursor. A SHA256 fingerprint of
preprocessed images/labels/class names rejects silent continuation on changed
data. Keep batch size fixed within a partial epoch. Fixed-seed `sample` calls
do not alter the training RNG. Loading uses `torch.load(weights_only=True)`
and reconstructs supported model types from a primitive config; no custom
model object is pickled. The save replaces its destination atomically.
Exact continuation is verified on CPU in the same software environment;
cross-version bitwise reproducibility is not promised. Convolutional CUDA
reproducibility also depends on deterministic cuDNN/kernel settings; the CUDA
tests request deterministic cuDNN. Live CUDA remains unverified here.
CUDA training uses a CUDA noise generator. `ImageGAN.load(..., device='cuda:0')`
restores that device's training state. Moving CPU↔CUDA requires `reset_rng=True`
(`--reset-rng` in the CLI) because those noise streams differ. Old schema-1 CPU
weights load, but numerical trajectories can change with the new Torch algorithm.
Schema-1/2 MLP checkpoints reconstruct their MLP generator/discriminator; new
checkpoints use schema 4 (including optional EMA). A convolutional run starts from fresh weights; an old
MLP checkpoint cannot be converted into a trained convolutional model.

For CLI resume use one mode, the same data/split/model/training flags, and
`--resume results/pets-conv-classical/classical/checkpoint.pt`. `--epochs`
finishes a partial saved epoch as its first epoch; `--steps` always adds the
requested number of updates. Old MLP runs require `--architecture mlp`, no
augmentation, and the original sorted file selection (omit `--data-seed`).

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
sampling and export, including periodic previews/checkpoints. Console logging
remains inside the measured interval. Periodic export time is reported separately.
This is a workload measurement, not an isolated simulator
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
- Convolutional images require power-of-two sizes 4/8/16/32/64. MLP images
  support any integer size 4..64. Images have 1 or 3 channels, batch 1..128, latent dimension ≤64,
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
[PyTorch DCGAN tutorial](https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html),
[PyTorch custom autograd](https://docs.pytorch.org/docs/stable/notes/extending.html),
and [torchvision CIFAR-10](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html).
NexQus builds and executes its own circuits; PennyLane/Qiskit are not dependencies.

## Recorded learning experiment

[Actual before/after previews](../examples/convolutional_previews/report.html)
and [recorded settings/results](../benchmarks/results/conv_bars_20261018.json)
show a 1,000-update CPU experiment on 128 procedural 16×16 bars. The simple
requested-orientation heuristic scored 64/64 classical and 61/64 hybrid outputs
on 64 fixed noise vectors after training. Previews still show limited location
diversity and imperfect shapes. This checks some learned conditional structure;
it does not establish full distribution learning, pet-image quality or quantum
advantage. It is one seed and one small configuration, not a quality benchmark.
Reproduce it with `python benchmarks/image_learning.py`. No download is needed.

## Regularized training and larger spatial models (source checkout)

The classical and hybrid conv results still need image-quality improvements.
More training alone is not a demonstrated fix. The controls below are opt-in,
experimental settings, not a validated recipe for recognizable pets or a claim
of quantum advantage. Short real-image development trials still produced poor
images. Software checks and pixel diversity do not establish semantic quality.

`--training regularized` supplies this CLI preset:

| Control | Preset | Purpose |
|---|---|---|
| Generator learning rate | `--learning-rate` | Generator update size |
| Discriminator rate | Half generator rate | Independently control discriminator pace |
| R1 | gamma 1, every 16 updates | Penalize squared real-image discriminator gradients |
| Differentiable translations | On real and fake | Same policy during both losses |
| EMA sampling | decay 0.995 | Average generator parameters and copy buffers after each update |
| Hybrid readout | Full probabilities | Retain more output features than one Z expectation per wire |
| Hybrid angles | Linear with small initial scale | Avoid saturating tanh encoding; rotations remain periodic |

Override with `--generator-lr`, `--discriminator-lr`, `--r1-gamma`,
`--r1-every`, `--diff-augment`/`--no-diff-augment` and `--ema-decay` (0 disables).
R1 is `gamma * interval / 2 * mean(||grad D(real)||²)` at the scheduled updates.
History records adversarial D loss, R1, total D loss and G loss separately.
Regularized training reuses one generator forward for both losses, updating
BatchNorm once per alternating update. The legacy path remains available.
Translation RNG is checkpointed and sampling does not consume training RNG.

`--architecture dcgan` uses fresh transposed-convolution models with tanh output
rescaled to [0,1], DCGAN-style convolution/BatchNorm initialization, and more
channels in high-resolution stages. `--hidden` is the base channel width for
this architecture, capped internally at 512 channels. The old conv model keeps
its previous width semantics and state layout. dcgan supports sizes 8..64 in
powers of two. `--discriminator-hidden` can set D's width independently.

A fresh classical pilot (inspect early previews before using the full budget):

```bash
python examples/image_gan.py --mode classical --architecture dcgan --training regularized --data data/pets --size 64 --channels 3 --samples 10000 --epochs 100 --batch-size 64 --hidden 64 --latent-dim 64 --learning-rate 0.0002 --data-seed 7 --horizontal-flip --preview-every 500 --checkpoint-every 1000 --output-dir results/pets-dcgan-classical
```

Append `--device cuda:0` with CUDA-enabled Torch. For a smaller CPU experiment,
use size 32, hidden 32, fewer samples and bounded `--steps` instead of `--epochs`.
Local validation used CPU; real CUDA is unverified for the new controls.
For hybrid replace the mode with `--mode hybrid --qubits 6 --depth 2 --latent-skip`
and use a separate output folder. The bypass concatenates classical noise and
quantum features; it may dominate generation and does not establish quantum
contribution. Compare with `--no-latent-skip`. `--readout z|probabilities` and
`--angle-map tanh|linear` are explicit architecture options, fixed on resume.
API defaults stay mlp/legacy/z/tanh/no-bypass for old code.

### Retune an existing checkpoint

Explicit `--retune` preserves G/D weights, Adam moments and data order while
changing training controls. This is a changed experiment, not exact continuation
of the old settings. Newly enabled EMA starts from current weights. Keep the
original data, split seed, batch size, shape and architecture. For the previous
64x64 conv classical configuration, try 1000 additional updates with previews
every 100:

```bash
python examples/image_gan.py --mode classical --architecture conv --training regularized --retune --resume results/pets-conv-classical/classical/checkpoint.pt --data data/pets --size 64 --channels 3 --samples 10000 --steps 1000 --batch-size 64 --hidden 256 --latent-dim 64 --learning-rate 0.0002 --data-seed 7 --horizontal-flip --device cuda:0 --preview-every 100 --checkpoint-every 100 --keep-checkpoints --output-dir results/pets-conv-retuned
```

Replace the checkpoint path and match its actual flags. A fresh dcgan model
cannot inherit trained conv weights. Without `--retune`, changed controls are
rejected. `--keep-checkpoints` saves numbered periodic copies alongside the latest
checkpoint. A fresh run refuses to overwrite an existing output checkpoint
unless `--overwrite` is explicit. Use separate output folders for comparisons.

### Inspect actual generated outputs

The report leads with generated `after.png`; real training and validation photos
are clearly marked at the bottom. EMA previews are labelled, and `after_raw.png`
shows current weights with the same noise. `independent.png` uses a separate
fixed seed and `--diagnostic-samples` per class (default 32; max 128 total).
Metrics include pairwise pixel MAE, closest pair and fraction of pairs below
MAE 0.01, plus pooled 8x8 diversity. Identical outputs yield duplicate fraction
1. Random noise can be diverse without meaningful images: these are not quality
scores. `nearest_training.png` aligns same-class pooled-pixel references with
the preview positions; this is not perceptual similarity or a memorization proof.
Hybrid diagnostics record angle/feature spread, tanh saturation where applicable,
readout dimension and bypass status. They identify possible information
bottlenecks without proving the cause of bad images. Parameter interventions
and circuit reports use the same current/EMA model as the preview.
Schema 4 includes EMA and regularization/augmentation settings; schemas 1–3
remain loadable. CPU exact continuation including augmentation and EMA is tested.

References: [zero-centered gradient penalties / R1](https://arxiv.org/abs/1801.04406),
[differentiable augmentation](https://arxiv.org/abs/2006.10738),
[PyTorch DCGAN tutorial](https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html).
The preset combines selected techniques, without reproducing all settings or
quality claims of those works.

[Actual generated previews](../examples/image_stability_trials/report.html) and
[recorded settings](../benchmarks/results/image_stability_20261019.json) show a
five-case, 1000-update CPU pilot on 1600 training / 400 validation CIFAR cat/dog
photos at 32x32, seed 7. The outputs remain blurry; no convincing pet-image
improvement is demonstrated. Zero duplicate pairs did not imply recognizable
images. This does not validate the 64x64 Kaggle workflow. Recorded times are
single-session logs, not runtime comparisons; no CUDA or quantum advantage is
claimed. Download the HTML and open it locally to view the generated grids.
