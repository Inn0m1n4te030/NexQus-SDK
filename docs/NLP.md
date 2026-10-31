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
