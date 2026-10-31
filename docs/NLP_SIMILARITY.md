# Sentence-pair similarity laboratory (development checkout)

This example connects classical text features to the existing NexQus sentence
encoders and offline circuit explorer. It is not in the published 1.3.0 wheel.
There are no accounts, model downloads or QPU calls, and no new dependencies
beyond the SDK's NumPy requirement.

## Run it

From a checkout containing this example:

```bash
python -m pip install -e .
python examples/nlp_similarity.py
```

The browser opens a server bound to `127.0.0.1`, normally port 8767. Enter four
sentences (two pairs), then click **Compare sentences**. Stop with Ctrl+C.
Use `--no-browser` to open the printed address yourself; `--port 0` chooses a
free port. Each sentence is limited to 1,000 characters. No entered text is
sent to a remote service. The three built-in examples also work offline:

```bash
python examples/nlp_similarity.py --export .release-artifacts/nlp-similarity.html
```

Open that file directly and choose an example. Arbitrary text needs the local
Python server because scores are calculated by the actual SDK, not a JavaScript
approximation. Optional `--qubits 1` through `--qubits 6` changes the representation
size. `--device cuda:0` requests CUDA using optional PyTorch; missing CUDA fails
rather than silently reporting GPU execution. Text processing, SVD and HTML
export remain CPU work. GPU hardware was not validated locally for this example.

## What the screen explains

1. **Classical baseline:** cosine similarity of the full normalized TF-IDF
   vectors; also cosine after dimension reduction. Zero-vector cosine is shown
   as undefined, not manufactured as a semantic score.
2. **Preprocessing:** Unicode word tokenization and case folding; count-based
   term frequency times `log((1+N)/(1+document_frequency))+1`; L2 normalization;
   truncated SVD fitted on the displayed 12-sentence corpus; training-fitted
   per-component scaling to `[0, pi]`, with clipping. All fit data, vocabulary,
   projection coefficients, intermediate vectors and clipped components are
   visible. Entered sentences never update any fit state.
3. **Quantum score:** construct two `nlp.sentence_encoder()` circuits and execute
   `U_right† U_left`. `P(00...0)` is the squared pure-state overlap. Both pairs
   use the same fitted preprocessing, circuit and simulator.
4. **Explanation:** the SDK's `observable='P0'` readout traces that exact all-zero
   probability and computes analytic derivatives of its final value. The
   synchronized comparison shows both pairs' angles, step readouts, probabilities
   and final-output sensitivities. Derivatives refer to encoded angle bindings,
   not raw words or discrete token changes.

The encoder is fixed, so this demo performs no optimizer training and fabricates
no training history. To study trained models, use the SDK QNN example's checkpoint
playback and the NLP module's reuploading helpers. A shared trailing entangler
cancels in fidelity. For this specific RY encoding the score is also exactly
`product(cos((left_angle-right_angle)/2)**2)`, which the tests use as an independent
check; this example does not require quantum hardware or offer a speed advantage.

Word order is discarded. Unknown words are listed and ignored; sentences with
no known words are rejected. Negation, unseen synonyms and grammar can therefore
be mishandled. Dimensionality reduction, clipping and angle geometry can merge
different texts. The corpus is a small demonstration fixture, not a trained
semantic encoder or held-out evaluation benchmark. Cosine and fidelity have
different ranges/geometries; a larger number does not establish better accuracy.

## Optional external text representations

Keep your preferred text model and its dependencies in your project. Export
sentence embeddings to a local JSON file:

```json
{
  "fit_sentences": ["first", "second", "third", "fourth"],
  "vectors": {
    "first": [1, 0, 0],
    "second": [0, 1, 0],
    "third": [0, 0, 1],
    "fourth": [1, 1, 1],
    "new query": [0.2, 0.3, 0.8]
  }
}
```

```bash
python examples/nlp_similarity.py --embeddings my_vectors.json
```

This is a lookup adapter, not live text-model inference. Every entered sentence
must have a finite, real, equal-width vector. Fit sentences must be distinct,
number at least four, and have entries in `vectors`. SVD and range scaling use
**only** `fit_sentences`; query-only vectors never refit the preprocessing. Select
fit sentences without test data leakage. The toy vectors above illustrate file
format only and carry no linguistic meaning. No new core dependency is needed.

## Export and reproduce

The outer **Download analysis JSON** saves text, features, preprocessing and
scores. The embedded explorer's **Download trace JSON** saves both full circuit
traces and their sensitivities. HTML exports include the source sentences and
bindings; review them before sharing. Live requests are limited to JSON bodies
of 16 KiB and foreign browser origins are rejected.

Validation:

```bash
python -m unittest discover -s tests -p test_nlp_similarity_demo.py -v
python examples/nlp_similarity.py --export .release-artifacts/nlp-similarity.html
# For development UI verification only:
npm install --no-save --no-package-lock jsdom@26.1.0
node scripts/check_nlp_similarity_ui.cjs .release-artifacts/nlp-similarity.html
```
