# NLP building blocks (development checkout)

`from nexqus import nlp` provides NumPy-only helpers for text projects. These
additions are not in the published 1.3.0 wheel. Install this checkout with
`python -m pip install -e .`. No text model, parser, dataset or network request
is loaded by the module. Text processing and task evaluation remain project code.

## Fixed-width feature encoding

Extract token embeddings or sentence features using your chosen classical
library. Reduce their dimension outside NexQus when necessary. Fit every learned
preprocessing step on training data only; never refit it on test sentences.

```python
from nexqus import nlp

# Example numeric training features, not a learned language representation.
scaler = nlp.FeatureAngleScaler().fit([[0., 0.], [1., 1.]])
angles = scaler.transform([[0.2, 0.7], [0.3, 0.6]])
left = nlp.sentence_encoder(angles[0])
right = nlp.sentence_encoder(angles[1])
print(nlp.pair_fidelity(left, right))
```

The scaler maps each training feature's minimum/maximum to 0/pi and clips new
values to this range. Constant training features always map to zero. Transform
requires a (samples, features) numeric matrix and never updates fitted ranges.
Scaling is lossy and not a semantic model. Angle periodicity also affects the
representation: different features can produce identical physical states.

`pool_token_features(matrix, mask=...)` mean-pools a (tokens, features) matrix.
The optional mask contains one boolean per token, including at least one True;
padding is excluded from the mean. Token order is lost by mean pooling. Empty,
complex, nonfinite and nonnumeric matrices are rejected.

`sentence_encoder(features, weights=None, rotation='Y', entanglement='linear',
max_qubits=10)` returns an ordinary Circuit. Each feature is one angle/qubit.
Without weights it applies encoding followed by entanglers. Optional weights
have shape (layers, qubits, 1); they use Y rotations and reupload the feature
angles in each layer. Parameters and Expressions are supported. Entanglement
uses the existing linear/ring/full or explicit-edge template contract.

## Sentence-pair circuits and trainable models

`sentence_pair_circuit(left, right)` builds **U_right† U_left**. Starting from
all zeros, its all-zero probability equals the squared overlap of the encoder
states. Both Circuits must have equal width and static unitary operations.
They are not modified. This is compute/uncompute overlap, not a SWAP test.

`pair_fidelity(left, right, bindings=None, device='cpu', backend=None)` computes
that probability exactly using the existing simulator. `cuda`/`cuda:N` requires
CUDA-enabled PyTorch and compatible hardware; no QPU is used. Classical parsing,
pooling and scaling remain on CPU. No new real-GPU validation is claimed.

Overlap measures these encoded states; it does not establish semantic
similarity, retrieval quality, classification accuracy or quantum advantage.
A shared trailing unitary cancels from overlap. Multiple feature-reuploading
layers can avoid this simple cancellation, but do not guarantee useful
gradients or performance. Evaluate trained encoders against classical baselines.

```python
from nexqus import Parameter, QuantumFunction, Simulator

left = nlp.sentence_encoder([Parameter('left_x')])
right = nlp.sentence_encoder([Parameter('right_x')])
pair = nlp.sentence_pair_circuit(left, right)
# For this one-qubit example, fidelity = (1 + <Z>)/2.
readout = QuantumFunction(pair, 'Z', Simulator())
print(readout({'left_x': 0.2, 'right_x': 0.8}))
```

For more than one qubit, all-zero probability is not `(1+<Z on q0>)/2`.
Use the full output distribution (for example through TorchCircuit) for a custom
trainable overlap loss. A project can instead build pair classifiers using
expectation readouts. This module does not implement a retrieval index or loss.
Symbolic names are preserved: different sentence inputs need different names
or separate bindings before composition. Identical weight names share weights.

## Syntax-guided adapter

```python
parse = nlp.DependencyParse(['Alice', 'likes', 'Bob'], [1, 1, 1])
circuit = parse.circuit([0.2, 0.6, 0.4])
print(parse.annotations())
circuit.explain(title='Alice likes Bob').to_html(filename='sentence.html')
```

Heads are zero-based token indices. Exactly one root heads itself. Cycles,
multiple roots and external heads are rejected. Each token becomes one qubit;
one supplied Y-angle per token is followed by CX(head, child), ordered by child
index. Optional reuploading layers use the same edges. Word order determines
wire order. The default ten-qubit cap is explicit; nothing is silently truncated.
This is an experimental syntax-guided ansatz, not a semantic interpretation of
individual gates, nor an implementation of DisCoCat/DisCoCirc or lambeq.

Optionally install `python -m pip install -e ".[nlp-parser]"` for spaCy.
Language model installation and parsing remain the project's responsibility.
For an existing, dependency-parsed spaCy Doc or sentence Span:

```python
# 'sentence' comes from a parser already installed and run by your project.
parse = nlp.DependencyParse.from_spacy(sentence)
circuit = parse.circuit(token_angles)
```

The adapter reads `Token.i`, `Token.text`, `Token.dep_` and `Token.head.i`, and
remaps document offsets to local wire indices. All heads must be inside the
supplied single sentence; retain punctuation or explicitly remap dependencies
if filtering tokens. Unparsed tokens are rejected. NexQus neither imports
spaCy nor downloads its language models. Protocol tests run without spaCy;
real spaCy integration can be checked with a constructed Doc, without a model.
See the [spaCy Token API](https://spacy.io/api/token).

## Run the offline lesson

```bash
python examples/nlp_building_blocks.py
```

Open `.release-artifacts/nlp/sentence_pair.html` and `dependency.html`. The
example uses supplied demonstration features, labels token encoding blocks, and
shows computed states. It does not report NLP accuracy on any dataset.

## Interactive sentence-pair project

See [Sentence-pair similarity laboratory](NLP_SIMILARITY.md) for an offline
TF-IDF/SVD example, classical cosine baselines, optional external vectors and
P0 circuit explanations. Run `python examples/nlp_similarity.py` from this checkout.


## Reproducible trainable sentence-pair experiment

```bash
python examples/train_sentence_pairs.py --epochs 30 --seeds 7 13 21
```

Open `.release-artifacts/qnlp-training/report.html`. NumPy CPU is sufficient;
no Torch, GPU, external text model or downloaded dataset is required. The
included hand-authored CSV has 18 train, 8 validation and 12 test pairs, with
paraphrase, identity, unrelated, negation and word-order cases. It is a software
and learning fixture, not a semantic benchmark. Exact/reversed duplicate pairs
are rejected, but individual sentences/vocabulary may occur across splits.

Vocabulary/IDF, two-component SVD and angle ranges are fitted on **training
sentences only**. Validation selects the minimum-MSE checkpoint; the fixed 0.5
threshold and declared seeds never use test performance for selection. Unknown
words are recorded. Every model gets the same two encoded components per
sentence, same full-batch MSE/Adam learning rate and update budget. The classical
sigmoid linear model uses absolute differences and elementwise products of
normalized angles (5 parameters); an eight-hidden-unit tanh MLP uses the same
pair features (49 parameters). The shared quantum encoder uses four weights:
RY feature upload -> trainable RZ -> CX -> feature upload -> trainable RX -> CX
-> feature upload. Trainable noncommuting rotations are between data uploads,
so this is not a shared terminal-unitary-only kernel. Gate-local probability
Jacobians handle shared weights in both encoders and the inverse chain rule.

The fixed first seed supplies the playback example; it is not chosen by its test
score. Checkpoints record actual pre-update parameters and training losses.
The displayed word-order failure is intentional evidence: bag-of-words features
cannot distinguish reversed text with identical token counts. No downstream
model can recover the lost order. All model/seed predictions and category-tagged
errors are exported, including cases where NexQus loses. Repeated seeds measure
initialization variability, not uncertainty across datasets. Fit seconds include
training and validation, exclude preprocessing/export and are **not** a warmed
simulator benchmark or a matched-parameter-budget comparison.

Use `--data PATH` for a larger CSV with columns
`split,left,right,label,category`; split is train/validation/test and label is
0 or 1. Every split needs both labels, and each sentence pair must be unique
including reversed order. Start with at least four distinct training sentences
with enough vocabulary for two SVD components. Examples with no known words
are rejected. Shared sentences across splits are allowed and should be reported
when interpreting generalization. Hyperparameters are fixed, not tuned to this
fixture; domain datasets should use separately justified splits and budgets.

The checked-in `benchmarks/qnlp_cpu_fixture.json` records an actual three-seed,
30-update CPU run, the dataset hash, losses, held-out predictions and environment.
Reproduce before making task-performance claims; no quantum advantage is claimed.
