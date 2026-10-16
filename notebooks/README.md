# NexQus SDK 1.1.0 learning path

Work through these lessons in order. Each notebook starts from its own imports
and can run independently after installing the SDK. English explanations include
Burmese summaries/comments; figures use English labels for broad font support.

| Notebook | Learn | Approximate study time |
|---|---|---:|
| [00 — Start here](00_start_here.ipynb) | Environment, first circuit, counts versus probabilities, 8-qubit drawing | 15 min |
| [01 — Qubits and gates](01_qubits_and_gates.ipynb) | Amplitudes, rotations, entanglement, QFT/iQFT, Grover, Shor and resources | 20 min |
| [02 — Noise and methods](02_noise_and_methods.ipynb) | Exact channels, trajectories, memory and conventions | 15 min |
| [03 — Functions and devices](03_functions_sweeps_and_devices.ipynb) | Decorators, parameters, sweeps, compilation, optional CUDA | 15 min |
| [04 — Gradients and learning](04_gradients_and_learning.ipynb) | Gate-local derivatives, Adam, regression, toy classification | 20 min |
| [05 — Dynamic protocols](05_measurement_reset_teleportation.ipynb) | Collapse, reset, feed-forward, teleportation and correlations | 20 min |
| [06 — Your first experiment](06_your_first_experiment.ipynb) | Compare methods with controlled variables and an experiment report | 15 min |

## Run with Jupyter

Download the source checkout for the notebook files. From its root, activate
the environment where you want to learn and install the tutorial dependencies:

```bash
python -m pip install 'nexqus-sdk[tutorial]==1.1.0'
python -m ipykernel install --user --name nexqus --display-name "Python (NexQus)"
jupyter lab notebooks
```

These notebooks target SDK 1.1.0; each setup cell checks the imported version.
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

CI executes all seven notebooks through real Jupyter kernels. Their assertions
cover state normalization, bit order, gates, entanglement, noise conventions,
gradients, learning, reset, teleportation, native circuit drawing, quantum-path
factoring and agreement between simulation methods.
GPU execution is optional for learners; maintainers can require it with the
command above. Reports remain separate from the public teaching material.

Each lesson includes a **Try it** exercise. Exercises are prompts for changing
the tested examples; they do not contain intentionally broken code cells.

## Rebuild the teaching source

`scripts/build_notebooks.py` is the editable lesson source. Running it regenerates
all notebooks and clears saved outputs. Re-run the executor afterward. The
source notebooks start with clean outputs so you can run each lesson yourself.

## Burmese guidance

Notebook များကို 00 မှ 06 အထိ အစဉ်လိုက် လေ့လာပါ။ Lesson တစ်ခုစီသည် သီးခြား
run နိုင်ပါသည်။ Jupyter တွင် **Python (NexQus)** kernel ကို ရွေးပါ။

Code ပြောင်းပြီး စမ်းသပ်ရာတွင် kernel ကို restart လုပ်ကာ cell အားလုံးကို
အစမှ ပြန် run ပါ။ Assertion မအောင်မြင်လျှင် expected result နှင့် မိမိပြောင်းလဲထားသော
circuit/noise setting ကို နှိုင်းယှဉ်ပါ။ GPU မရှိလည်း CPU lesson များကို ပြီးမြောက်နိုင်ပါသည်။
