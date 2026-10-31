# Circuit explainability explorer

This explorer is included in the 1.3.0 release and is absent from
the existing PyPI 1.2.0 wheel. It traces local, exact, double-precision
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
learned parameters. Its output is the ZZ expectation itself. The trained example now opts into local prediction sensitivity and independent
block removals. The explained output remains the ZZ expectation; no classifier
probability mapping or global feature importance is inferred. Templates work as ordinary circuits; supply
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

## Before/after views and offline circuit interventions

The explorer now shows the input and output of the selected operation together:
complex amplitudes, probabilities and their differences, reduced Bloch vectors,
and qubit-versus-rest entanglement entropy. Basis tables show at most 16 states;
the JSON download includes the full state history.

Precompute independent alternatives when exporting a report:

```python
from nexqus import Circuit

circuit = Circuit(2).rx(0.5, 0).cx(0, 1)
circuit.explain(
    blocks=[dict(name='Encoding', start=0, stop=1, role='encoding'),
            dict(name='Entangling', start=1, stop=2, role='entangling')],
    interventions=[dict(operation=0, angle=0.0),
                   dict(operation=0, angle=1.5707963267948966),
                   dict(operation=1, skip=True)],
).to_html(filename='circuit_laboratory.html')
```

Open the HTML and choose a calculated alternative. All later states are
recomputed in Python before export. No server, QPU, or external asset is needed.
Each choice edits the original bound circuit independently; choices do not stack.
A removed gate is represented by identity to preserve stage and block indices.
Logical depth/counts in that alternative therefore include the identity stage.
Original parameter chips remain context, while the gate label shows the new
angle. The downloaded JSON contains the selected trace and intervention.

Angle edits support validated RX/RY/RZ matrices, not arbitrary gate names. At
zero angle, axes are indistinguishable from the matrix alone: specify, for
example, `dict(operation=0, angle=0.5, axis='ry')` to select the intended axis.
There are at most 16 alternatives, and their combined serialized size must fit
`max_report_mb`. Larger circuits or many alternatives may exceed that budget.
These are finite exported presets; to add an arbitrary angle, regenerate from
Python. Comparing a gate removal on one input is not a general importance or
feature-attribution score. Noise and dynamic circuits remain unsupported.

Run `python examples/explain_circuits.py` and open
`.release-artifacts/explanations/laboratory.html` for the complete example.


## Local prediction sensitivity (development checkout)

Use `sensitivities=True` to compute signed derivatives of the **final Pauli
expectation** at the supplied bindings. This is opt-in and is not in the
published 1.3.0 wheel. Install this checkout to use it.

```python
from nexqus import Circuit, Parameter
x, w = Parameter('x'), Parameter('w')
q = Circuit(2).ry(x, 0).ry(w, 1).cx(0, 1)
report = q.explain(
    bindings={'x': .7, 'w': .3}, observable='ZZ',
    parameter_roles={'x': 'feature', 'w': 'weight'},
    blocks=[{'name': 'Encoding', 'start': 0, 'stop': 1},
            {'name': 'Learned layer', 'start': 1, 'stop': 3}],
    sensitivities=True,
    interventions=[{'block': 1, 'skip': True}],
)
report.to_html(filename='prediction.html')
```

The offline explorer ranks derivative magnitudes and shows signed values and
roles. JSON stores them under `sensitivity.parameters`. Derivatives use
**gate-local parameter shift and the arithmetic chain rule**: shared parameters
sum contributions from every occurrence. The supplied initial state is retained.
For a rotation angle a, the gate derivative is
`(f(a + pi/2) - f(a - pi/2))/2`; chain factors map this to binding derivatives.
This costs two state evolutions per active symbolic gate occurrence; at most
32 symbolic parameters are accepted, alongside the existing trace limits.
CUDA execution uses the selected simulator; derivative scalars and UI data are
exported to the CPU. CUDA performance has not been measured for this feature.

Feature sensitivities concern **encoded circuit bindings**, not raw words or
features before preprocessing. Units and scaling affect magnitude comparisons.
For a small change h, `f(v+h) - f(v)` is approximately `h * df/dv`.
A zero derivative at one input can be a stationary point, not irrelevance.
These are not global importance, Shapley values, calibrated classifier scores,
or real-world causal effects. No output-transform derivative is implied.

A block intervention uses its zero-based index in the **start-sorted** block
list. It replaces every operation in the half-open range with identity,
preserving stage navigation. Each edit is independent of all other choices;
the existing limit of 16 alternatives and aggregate report size limit apply.
The selector shows the recomputed trace and final output difference. The
sensitivity panel always describes the original circuit, even while an
alternative is selected. Omitted layers measure this circuit's response on
one supplied input, not their global necessity.


## Compare predictions and replay training (development checkout)

Pass `comparisons=[{'name': 'Input B', 'bindings': complete_bindings}]` to
`explain()` for a synchronized second prediction. Up to eight comparisons are
accepted. The circuit topology, observable, initial state and backend stay the
same. Each binding dictionary must contain every parameter. The UI shows both
bound gate labels, step/final expectations, aligned basis probabilities, binding
values, and final-output derivatives when `sensitivities=True`. State navigation
advances both traces together. The full states are in the downloaded JSON.
Comparisons inherit the same gate/block interventions, allowing matched removal
effects on both inputs. Their bindings stay fixed; choosing a training checkpoint
changes the left trace only. Different weights are displayed explicitly, so a
readout difference is not silently attributed to input changes alone.

Pass `training_history=[{'epoch': 0, 'bindings': complete_bindings, 'loss': .2},
...]` to replay recorded training checkpoints. Epochs must be nonnegative,
strictly increasing integers no larger than `2**53-1`; loss is optional, finite, nonnegative caller
metadata. Supply up to 32 checkpoints. The epoch selector/play button updates
weights, the entire trace, prediction and sensitivities. It retains the current
gate step. Missing epochs are not interpolated and missing losses are omitted
from the loss scatterplot. Layer interventions are available only for the
original supplied final bindings. Comparison interventions add further traces. All alternate traces share an aggregate
`max_report_mb` budget, conservatively checked before evolution and after
serialization. Each checkpoint/comparison adds full trace and, optionally,
gradient computations; export is bounded, not a live training dashboard.

`examples/explain_circuits.py` now records five actual optimizer checkpoints
(epoch 0/10/20/30/40, losses measured before the next update), compares two
encoded inputs, and exports everything to `qnn.html`. Open it offline after:

```bash
python examples/explain_circuits.py
```

Series JSON downloads retain the complete comparison/checkpoint collection;
ordinary reports retain the existing selected-intervention download behavior. Reports
export the parameter values supplied by you; review them before sharing.


## All-zero probability readout

`observable='P0'` selects `P(00...0)` instead of a Pauli expectation. It is
the expectation of the all-zero projector, so gate-local parameter-shift rules
and expression chain rules apply unchanged. Frame `expectation` fields, local
sensitivities, comparisons and interventions all use this chosen probability.
It works with any supported initial pure state, and is not a sampled estimate.
The [NLP laboratory](NLP_SIMILARITY.md) uses it to explain squared state overlap.
