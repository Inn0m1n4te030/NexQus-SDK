# Circuit explainability explorer

This is an unreleased source-checkout addition after SDK 1.2.0. It is not in
the existing PyPI 1.2.0 wheel. The explorer traces local, exact, double-precision
pure states. It makes no cloud calls and submits no QPU jobs.
Use `q.explain(device='cuda:0')` or `examples/explain_circuits.py --device cuda:0`
to evolve states on CUDA. Per-frame diagnostics and HTML/JSON export remain
host work, labelled in `execution` metadata. See [GPU.md](GPU.md).

## First report

From this checkout in an activated environment:

```bash
python -m pip install -e .
python examples/explain_circuits.py
```

Open `.release-artifacts/explanations/bell.html`, `interference.html` or
`qnn.html` in a browser. Each HTML file is standalone, works offline and needs
no plotting library, external scripts, server, API key or account. The runtime
dependency remains NumPy. Reports can also display in Jupyter in an isolated
script-enabled iframe, subject to notebook trust and frontend restrictions.

```python
from nexqus import Circuit

q = Circuit(2).h(0).cx(0, 1)
explanation = q.explain(observable='ZZ', title='Understanding Bell entanglement')
explanation.to_html(filename='bell.html')
data = explanation.to_dict()  # independent copy of every checkpoint and metric
```

The report provides:

- Clickable circuit operations and keyboard-accessible gate selection.
- Initial state, previous/next, slider and playback controls.
- Author-defined block navigation, alongside individual gate navigation.
- Full basis probabilities in the trace, with up to 16 visible basis states.
- Complex amplitudes, a relative-phase plane and per-step phase references.
- Reduced-qubit Bloch vectors, purity and entropy.
- Z-basis covariance, labelled separately from entanglement.
- A chosen Pauli observable's exact expectation at every checkpoint.
- Parameter values and occurrence highlighting for QNN features/weights.
- Trace JSON download for auditing or reuse in other visualizations.

Playback follows recorded operation order; it is not a hardware timing model.
The circuit view pages through 12 operations at a time. Basis views can use
probability ranking or basis order and always disclose the displayed probability
mass. Unshown basis states remain in the downloadable data.

## QNN structure and parameter provenance

```python
from nexqus import Circuit, Parameter

x, weight = Parameter('x'), Parameter('weight')
model = Circuit(2).ry(x, 0).ry(.5*x, 1).rx(weight, 0).cx(0, 1)
report = model.explain(
    bindings={'x': .7, 'weight': -.2},
    observable='ZZ',
    parameter_roles={'x': 'feature', 'weight': 'weight'},
    blocks=[
        {'name': 'Angle encoding', 'start': 0, 'stop': 2, 'role': 'encoding'},
        {'name': 'Learned layer', 'start': 2, 'stop': 4, 'role': 'trainable'},
    ],
)
report.to_html(filename='qnn.html')
```

Blocks use nonoverlapping, half-open operation ranges `[start, stop)`. Roles are
`encoding`, `trainable`, `entangling`, `readout` or `block`; they are supplied
by the author, not inferred. Uncovered operations stay unannotated. A parameter
role is `feature`, `weight` or `parameter`. Highlighting follows actual symbolic
parameter occurrence, including shared parameters and expressions, rather than
guessing from gate names. Bindings must contain exactly the required names.

The example generator trains a small two-qubit variational model on synthetic
cosine values using 40 Adam updates. The QNN report shows one input and the
learned parameters. Its output is the ZZ expectation itself. No classifier
probability, attribution, feature importance or counterfactual is fabricated.
These prediction-explanation methods are future extensions, not part of this
first circuit-execution explorer. Templates work as ordinary circuits; supply
block ranges when you want layer annotations.

## Mathematical meanings

For pure state `psi`, basis probability is `abs(psi[b])**2`. Amplitudes retain
the simulator's complex convention. The phase plot removes the phase of the
first basis amplitude with probability above `1e-14`; this reference is named
at each step and can change. Zero-probability phases are undefined.

For a qubit's reduced density matrix `rho`, the Bloch vector is
`(2 Re rho01, -2 Im rho01, rho00-rho11)`. The sphere uses an orthographic
projection, so the vector's screen length is not its three-dimensional length;
the numerical coordinates are also shown. Purity is `Tr(rho**2)`, and entropy
is `-sum(lambda * log2(lambda))`, in bits. Here, because the global state is
pure, reduced entropy describes that qubit's entanglement with **all remaining
qubits**, not necessarily with an individual partner. A Bell state's reduced
Bloch vectors vanish although its joint state is pure and entangled.

The heatmap shows `⟨Zi Zj⟩ - ⟨Zi⟩⟨Zj⟩`, with diagonal variances omitted.
This is basis-dependent covariance, not an entanglement witness. Pairwise
correlation is insufficient to describe the full quantum state.

Probability change is total variation distance from the previous checkpoint:
`0.5 * sum(abs(p_after-p_before))`. Fidelity is
`abs(vdot(psi_before, psi_after))**2`, invariant to global phase. Phase change
is the largest wrapped relative-phase change among shared nonzero amplitudes,
anchored to the first shared basis amplitude. It is undefined when fewer than
two such amplitudes exist. Initial-state changes/fidelity have no previous
checkpoint and are `null`. H–Z–H demonstrates why phase changes can matter even
when a gate leaves current basis probabilities unchanged.

Gate descriptions recognize verified matrices; user-defined names alone cannot
establish semantics. Unrecognized unitaries receive a generic description plus
calculated evidence. This is deterministic explanation, not generated narration
from an LLM. The selected observable is a Pauli string in q0-first order; all
I/X/Y/Z combinations are supported. Postprocessing from an expectation to a
classification or other score must be documented separately by the model author.

## API and limits

`Circuit.explain(**options)` delegates to `nexqus.explain.explain(circuit, ...)`.
Options are `bindings`, `observable`, `blocks`, `parameter_roles`, `initial_state`,
`title`, `max_qubits=10`, `max_steps=256`, `max_report_mb=32`.
`Explanation.to_dict()` returns a deep copy. `to_html(filename=None)` returns a
notebook-displayable `HTMLReport` string and optionally writes UTF-8 HTML.

Trace size grows as checkpoints times 2**qubits, with additional reduced-state
and correlation metrics. Limits are checked before state allocation and after
trace serialization. The payload estimate is conservative for ordinary reports,
but `max_report_mb` is not a hard process-RAM cap and does not include all
Python, gate-storage or HTML-renderer overhead. Raise limits only deliberately.

This first version supports static unitary, diagonal and compact permutation
operations on local pure states. It rejects noise, measurement, reset and
classical conditions instead of presenting a mixed or branched state as pure.
It accepts normalized custom pure initial states. It has no QPU state inspection,
tomography, hardware-state reconstruction or sampled uncertainty display.

Exported HTML/JSON includes all retained amplitudes, supplied parameter values,
labels and annotations. Review these before sharing a report. Labels are escaped
as JSON data and inserted through text nodes, and reports load no remote assets.

## Verification

```bash
python -m unittest discover -s tests -p test_explain.py -v
python examples/explain_circuits.py
npm install --no-save --no-package-lock jsdom@26.1.0
node scripts/check_explain_ui.cjs
```

Analytic regressions cover Bell, GHZ, interference, global phase and Bloch Y
orientation. Checkpoints are compared with ordinary simulation for rotations,
QFT-related phases, reversed/nonadjacent wires, permutations and inverses.
DOM tests execute the real report script without loading external resources and
check navigation, playback, annotations, highlighting, readout, trace download,
empty circuits, SVG numerical bounds and malicious-label handling. They do not
replace browser-engine layout testing across all notebook/browser frontends.
