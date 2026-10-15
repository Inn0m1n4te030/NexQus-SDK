# NexQus SDK 0.4.0 — 15 October 2026

Focus: trainable quantum functions, explicit gradients and learning examples.

## Added

- QuantumFunction with expectation, gradient, and value_and_grad APIs.
- Gate-local parameter-shift for RX/RY/RZ/P/CP with exact arithmetic chain
  rules, including shared parameters and inverse/composed circuits.
- Exact density-matrix noise differentiation for fixed angle-independent noise.
- Explicit Adam/SGD dictionary-based optimization.
- Full-batch MSE QuantumRegressor and 0/1 QuantumBinaryClassifier.
- Four new examples: gradients, sine learning, toy classification, two-qubit VQE.
- New tests and English/Burmese guidance.

## Local results

- 89 tests discovered: 81 passed, 8 optional Torch CPU/CUDA tests skipped.
- Sine example: training MSE 0.293166 -> 0.000002365 after 80 epochs;
  dense evaluation-grid MSE 0.000002099.
- Toy classifier correctly labelled its six demonstration inputs.
- VQE energy -1.414212765 versus exact -1.414213562.
- These are small functional demonstrations, not quantum-advantage claims.

The existing v0.3 simulation API remains. No Torch autograd, trajectory-gradient
estimation, mid-circuit measurement, cloud hardware, or automatic hybrid QCNN
integration is added. See V04_GUIDE.md for costs, limitations and validation.
Previous-version logs and benchmarks retain their original versioned names.
