# NLP building blocks (1.4.0)

`from nexqus import nlp` provides NumPy-only helpers for text projects. These
additions are included in 1.4.0 and absent from 1.3.0. Install this checkout with
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


## Larger controlled feature study

The unigram and unigram+bigram conditions preserve the same two-qubit model,
encoded-component count, pair features, optimizer, update count, seed list and
CSV splits. Only the feature pipeline changes. A training-label-prior baseline
is included to reveal majority-class performance. The word vocabulary is capped
at 512 terms, selected by training-corpus frequency with deterministic lexical
tie breaks. IDF, SVD and angle ranges remain training-fitted. Bigrams add adjacent
word pairs; they are order-sensitive features, not a guarantee that the final
**two-dimensional** representation preserves every distinction. The report
measures a training-sentence reversal before and after compression.

Prepare public data explicitly (requires optional `pyarrow` only at this step):

```bash
python -m pip install pyarrow
python examples/prepare_mrpc.py --download
python examples/qnlp_feature_study.py --data .release-artifacts/mrpc.csv
```

Preparation fetches MRPC from the NYU GLUE dataset mirror at pinned revision
`68d3d763fc36f89a8d047f4b8e51b98ea36b98d0` and verifies source SHA-256 checksums.
See the [original Microsoft dataset](https://www.microsoft.com/en-us/download/details.aspx?id=52398)
and [mirror](https://huggingface.co/datasets/nyu-mll/glue/tree/68d3d763fc36f89a8d047f4b8e51b98ea36b98d0/mrpc)
for provenance and terms. Third-party sentences are downloaded locally; they
are not bundled with the SDK.

Defaults are 256 train / 64 validation / 128 test pairs. Test is sampled from
**labelled GLUE validation**; our train and validation subsets come from GLUE
train. This is a custom subset study, not an official GLUE test submission.
Sampling is stratified with seed 17, fixed before fitting. Duplicate/reversed
pairs are removed; conflicting-label duplicate groups are excluded. Training
candidates sharing a case-folded, whitespace-normalized sentence with selected
test pairs are excluded. Validation is then sampled, and training candidates
sharing validation sentences are excluded. The three selected splits are
exact-sentence-disjoint. Near-duplicates, shared news topics and paraphrased
sentences can remain; no article-level grouping is claimed. Metadata records
source counts/checksums, filtering counts, class counts, selected CSV hash and
split protocol. Cache files are not silently re-downloaded or trusted after a
checksum mismatch.

`--sizes TRAIN VALIDATION TEST` on the preparer changes subset sizes;
`--epochs` / `--seeds` on the study changes the declared training budget.
The original tiny fixture remains available through
`python examples/train_sentence_pairs.py`. The study runs on NumPy CPU and has
no claimed CUDA validation. Larger classical workloads may need more memory;
quantum state size remains bounded to two qubits in this controlled comparison.

For optional context-aware representations, export local sentence vectors using
your chosen external model, then pass:

```bash
python examples/qnlp_feature_study.py --data .release-artifacts/mrpc.csv \
  --representations embeddings --embeddings vectors.json
```

Use the `fit_sentences` / `vectors` JSON format described in
[NLP_SIMILARITY.md](NLP_SIMILARITY.md). The runner overrides fit sentences with
**only the CSV training sentences**, regardless of an external file's fit list.
Every used sentence needs a vector. No model is downloaded/imported automatically;
model dependencies stay in the user's project. Embedding results must be reported
separately with model/version/provenance; the checked-in measured study covers
unigrams and bigrams only.

Open `.release-artifacts/qnlp-features/report.html` for all conditions. Each links
to a per-condition error report and training playback. `comparison.json` records
all aggregate scores and preparation provenance; condition `results.json` files
retain complete per-seed data. Error groups overlap: provided class/category,
`negation_marker_mismatch` (presence of no/not/never differs),
`same_tokens_different_order` (equal token multisets, different sequences), and
`unknown_words` (at least one term outside the fitted vocabulary). These are
transparent token heuristics, not gold negation/syntax annotations or reliable
semantic error diagnoses. Category counts include repeated seeds; they are not
independent extra test samples. Missing groups must not be interpreted as proof
that a linguistic behavior is handled correctly.

## Trainable sentence-pair classifier (development checkout)

`SentencePairClassifier` is a reusable NumPy-core model. It accepts externally
prepared angle pairs shaped `(samples, 2, n_qubits)`, with binary labels. Fit any
text vectorizer, dimensionality reduction and angle scaler on training text only.

```python
from nexqus import SentencePairClassifier
model = SentencePairClassifier(n_qubits=2, seed=7)
model.fit(train_pairs, labels, epochs=30,
          validation_data=(validation_pairs, validation_labels))
probabilities = model.predict_proba(test_pairs)  # columns: class 0, class 1
predictions = model.predict(test_pairs)  # threshold P0 >= 0.5
model.save('model.json')
restored = SentencePairClassifier.load('model.json')
model.explain(test_pairs[0], comparison=test_pairs[1]).to_html(filename='prediction.html')
```

A fit starts fresh from the configured seed and uses full-batch analytic
parameter-shift gradients with Adam/MSE. Validation selects the minimum-MSE
checkpoint; without validation the final epoch is retained. Up to 32 recorded
checkpoints support playback. Save/load includes model weights and history,
**not** the external text encoder. Keep that preprocessing with your application.
Use `device='cuda:0', backend='torch'` only with a compatible Torch installation;
CPU checks do not validate CUDA hardware.

Run the complete text-to-prediction lesson:

```bash
python examples/sentence_pair_classifier.py --epochs 20
```

Open `.release-artifacts/sentence-pair-classifier/report.html`. It shows the
sentences, TF-IDF/SVD features, angles, P0 score and thresholded class, with the
native circuit explorer embedded below. Step through states/gates, inspect local
angle derivatives, compare a second held-out pair and replay training. The paired
`circuit.html` must remain beside the report. A checkpoint and held-out metric
JSON are also saved. The example compares a linear classical model trained on
absolute differences/products of the same encoded inputs, with the same update
budget and validation selection; parameter budgets differ.

This symmetric fidelity classifier always scores identical encodings as one.
It does not model arbitrary directional relations or calibrated confidence.
Angle sensitivities are not word attribution or causal effects. The small
fixture is a functional demonstration; broader evaluation is still required.

CPU smoke evidence (seed 7, 20 epochs, bundled fixture, 12 held-out pairs):

| Model | Test accuracy | Test MSE | Test F1 |
|---|---:|---:|---:|
| Quantum fidelity classifier | 0.6667 | 0.2649 | 0.7143 |
| Classical linear baseline | 0.7500 | 0.1963 | 0.6667 |

Quantum training MSE decreased from 0.2744 to 0.2392; validation selected epoch
20. These fixture results show the workflow executes, not general superiority.
