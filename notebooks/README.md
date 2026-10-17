# NexQus learning path

Work through these lessons in order. Each notebook starts from its own imports
and can run independently after installing the SDK. English explanations include
Burmese summaries/comments; figures use English labels for broad font support.

| Notebook | Learn | Approximate study time |
|---|---|---:|
| [00 — Start here](00_start_here.ipynb) | Environment, first circuit, counts versus probabilities, 8-qubit drawing | 15 min |
| [01 — Qubits and gates](01_qubits_and_gates.ipynb) | Amplitudes, rotations, entanglement, QFT/iQFT, Grover, Shor and resources | 20 min |
| [02 — Noise and methods](02_noise_and_methods.ipynb) | Exact channels, trajectories, memory and conventions | 15 min |
| [03 — Functions and devices](03_functions_sweeps_and_devices.ipynb) | Decorators, parameters, sweeps, compilation, optional CUDA | 15 min |
| [04 — Gradients and learning](04_gradients_and_learning.ipynb) | Gradients, quantum/classical sine predictions, loss/error plots, toy classification | 25 min |
| [05 — Dynamic protocols](05_measurement_reset_teleportation.ipynb) | Collapse, reset, feed-forward, teleportation and correlations | 20 min |
| [06 — Your first experiment](06_your_first_experiment.ipynb) | Compare methods with controlled variables and an experiment report | 15 min |
| [07 — Inspection and QML templates](07_inspection_and_qml_templates.ipynb) | Angles, depth/counts, optional PNG/PDF, encoding, layers and reuploading | 20 min |
| [08 — QGAN distribution](08_qgan_distribution.ipynb) | Adversarial learning, classical reference and local probability sensitivity (source checkout) | 20 min |
| [09 — H₂ energy explained](09_h2_energy_explained.ipynb) | VQE, classical HF/FCI references and energy contributions (source checkout) | 25 min |
| [10 — Molecular visualization](10_molecular_visualization.ipynb) | Rotate H₂/CH₄/organic structures, measure distances/angles and export custom geometry | 20 min |

## Run with Jupyter

Download the source checkout for the notebook files. From its root, activate
the environment where you want to learn and install the tutorial dependencies:

```bash
python -m pip install -e ".[tutorial]"
python -m ipykernel install --user --name nexqus --display-name "Python (NexQus)"
jupyter lab notebooks
```

The original eight lessons target SDK 1.2.0. Lessons 08/09 require the unreleased
source checkout; the published 1.2.0 wheel does not include their new APIs.
The command above installs the current checkout into the activated environment.
Add `.[visualization]` separately if you need optional PNG/PDF export. Lessons
08/09 check required APIs and show the imported package and Python paths because
the source additions and published wheel both currently report version 1.2.0.
QGAN and the fixed-H₂ lesson work on native Windows without PySCF. The optional
PySCF driver requires Linux, macOS or WSL on Windows; install `.[chemistry]`
only in one of those environments. See [import recovery](../docs/QGAN_CHEMISTRY.md#recover-from-a-missing-qgan-import).
Restart the kernel after an upgrade. Select **Python (NexQus)**.
Use **Restart Kernel and Run All** to verify a lesson
from a clean state. A lesson should not depend on variables from another file.
An import/version error usually means the selected kernel uses a different
environment; select the kernel registered by the commands above.

To execute the entire series through a registered Jupyter kernel:

```bash
python scripts/run_notebooks.py --engine jupyter --kernel nexqus
```

If you want the optional CUDA check to be required:

```bash
python scripts/run_notebooks.py --engine jupyter --kernel nexqus --require-gpu
```

This does not install or replace your CUDA-enabled PyTorch. Lesson 03 checks
availability and otherwise explicitly skips its optional GPU cell.

## Standalone execution without Jupyter

These notebooks deliberately contain plain Python: no shell magics, widgets,
asynchronous cells or hidden IPython state. With NumPy, Matplotlib and NexQus:

```bash
python scripts/run_notebooks.py --engine python
```

The standalone runner uses a fresh subprocess/namespace for each notebook,
executes cells in order, checks assertions and captures stdout/Matplotlib PNGs
back into the notebook. It is not a Jupyter/IPython kernel implementation and
does not validate Jupyter front-end behavior. It is also not a sandbox: execute
only notebook code you trust, just as you would run a Python script.

`--engine auto` uses Jupyter when the required libraries are available, otherwise
the standalone runner. The chosen engine, per-notebook status, cell count and
optional skips and the SDK version reported by each kernel are recorded in
`.release-artifacts/notebook_execution.json` and notebook
metadata. Explicit `--engine jupyter` never silently falls back.
The runner rejects an otherwise passing notebook if its kernel imports a
different SDK version from this source release. `--report PATH` preserves a
separately named report. On supported Unix hosts `--transport ipc` can use local
IPC instead of TCP; neither transport is available in every restricted container.

## Checked behavior

CI executes all eleven notebooks through real Jupyter kernels on Linux and Windows. Their assertions
cover state normalization, bit order, gates, entanglement, noise conventions,
gradients, learning, reset, teleportation, native circuit drawing, quantum-path
factoring, QML templates, inspection and agreement between simulation methods.
The independent-correctness CI job tests real PNG/PDF exports with visualization installed.
GPU execution is optional for learners; maintainers can require it with the
command above. Reports remain separate from the public teaching material.

Each lesson includes a **Try it** exercise. Exercises are prompts for changing
the tested examples; they do not contain intentionally broken code cells.

## Rebuild the teaching source

`scripts/build_notebooks.py` is the editable lesson source. Running it regenerates
all notebooks and clears saved outputs. Re-run the executor afterward. The
source notebooks start with clean outputs so you can run each lesson yourself.
The generator also keeps examples/compare_sine.py synchronized with lesson 04.
For a visible preview without executing a notebook, see the
[saved comparison figure](../docs/images/sine_comparison.svg).

## Burmese guidance

Notebook များကို 00 မှ 07 အထိ အစဉ်လိုက် လေ့လာပါ။ Lesson တစ်ခုစီသည် သီးခြား
run နိုင်ပါသည်။ Jupyter တွင် **Python (NexQus)** kernel ကို ရွေးပါ။

Code ပြောင်းပြီး စမ်းသပ်ရာတွင် kernel ကို restart လုပ်ကာ cell အားလုံးကို
အစမှ ပြန် run ပါ။ Assertion မအောင်မြင်လျှင် expected result နှင့် မိမိပြောင်းလဲထားသော
circuit/noise setting ကို နှိုင်းယှဉ်ပါ။ GPU မရှိလည်း CPU lesson များကို ပြီးမြောက်နိုင်ပါသည်။
