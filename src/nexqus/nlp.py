"""Small NLP circuit building blocks; only NumPy is required.

Text models, parsing and feature extraction remain external. These helpers
encode supplied features and syntax, not linguistic meaning automatically.
"""
from dataclasses import dataclass
import numbers
import numpy as np

from .circuit import Circuit, positive_integer
from .simulator import Simulator
from . import templates

__all__ = ['pool_token_features', 'FeatureAngleScaler', 'sentence_encoder',
           'sentence_pair_circuit', 'pair_fidelity', 'DependencyParse']


def _matrix(values):
    raw = np.asarray(values)
    if raw.dtype.kind not in 'iuf' or raw.ndim != 2 or min(raw.shape) < 1:
        raise ValueError('Features must be a nonempty real numeric matrix')
    result = raw.astype(float)
    if not np.isfinite(result).all(): raise ValueError('Features must be finite')
    return result


def pool_token_features(features, *, mask=None):
    """Mean-pool (tokens, features); optional boolean mask excludes padding.

    No tokenization or embedding model is run. Masked rows must also be finite.
    """
    values = _matrix(features)
    if mask is not None:
        selected = np.asarray(mask)
        if selected.dtype.kind != 'b' or selected.shape != (len(values),) or not selected.any():
            raise ValueError('mask must select at least one token with one boolean per row')
        values = values[selected]
    return values.mean(axis=0)


class FeatureAngleScaler:
    """Fit per-feature training ranges and map to [0, pi], clipping new data.

    Fit on training data only, after external dimensionality reduction if used.
    Constant training columns always map to zero. No vocabulary or semantic
    representation is learned. transform never updates the fitted ranges.
    """
    def fit(self, features):
        values = _matrix(features)
        self._minimum = values.min(axis=0)
        self._span = values.max(axis=0) - self._minimum
        if not np.isfinite(self._span).all(): raise ValueError('Feature range overflow')
        return self

    def transform(self, features):
        if not hasattr(self, '_minimum'): raise ValueError('Fit the scaler on training data first')
        values = _matrix(features)
        if values.shape[1] != len(self._minimum): raise ValueError('Feature dimension differs from fit')
        with np.errstate(over='ignore'):
            scaled = (values-self._minimum)/np.where(self._span == 0, 1, self._span)
        scaled[:, self._span == 0] = 0
        return np.clip(scaled, 0, 1)*np.pi

    def fit_transform(self, features):
        return self.fit(features).transform(features)


def sentence_encoder(features, *, weights=None, rotation='Y',
                     entanglement='linear', max_qubits=10):
    """One angle feature per qubit, optionally reuploaded in variational layers.

    Features and weights may be Parameters/Expressions. weights uses the
    existing (layers, qubits, 1) Y-rotation template convention. Fixed-width
    text features must be prepared externally and angles are in radians.
    """
    values = np.asarray(features, dtype=object)
    if values.ndim != 1 or not 1 <= len(values) <= positive_integer(max_qubits, 'max_qubits'):
        raise ValueError('Provide a feature vector within max_qubits')
    circuit = Circuit(len(values))
    if weights is None:
        templates.angle_encoding(circuit, values, rotation=rotation)
        templates.entangling_layer(circuit, entanglement=entanglement)
    else:
        templates.data_reuploading(circuit, values, weights,
            encoding_rotation=rotation, entanglement=entanglement)
    return circuit


def sentence_pair_circuit(left, right):
    """Return U_right† U_left; P(all zeros) is pure-state fidelity.

    Both encoders start from all zeros, have equal width and must be static
    unitaries. Symbolic names remain unchanged: use distinct input names or bind
    encoders separately. Shared weight names intentionally share parameters.
    A shared trailing unitary cancels from fidelity; use feature reuploading
    between trainable layers for a feature-dependent trainable kernel.
    """
    if not isinstance(left, Circuit) or not isinstance(right, Circuit):
        raise TypeError('Both sentence encoders must be Circuits')
    if left.n_qubits != right.n_qubits: raise ValueError('Sentence encoders must have equal width')
    for circuit in (left, right):
        if circuit.dynamic or any(op.kind not in ('unitary', 'diagonal', 'permutation', 'parameterized')
                                  for op in circuit.operations):
            raise ValueError('Sentence-pair fidelity requires static unitary encoders')
    return Circuit(left.n_qubits).compose(left).compose(right.inverse())


def pair_fidelity(left, right, *, bindings=None, device='cpu', backend=None):
    """Exact local squared overlap in [0,1], not a semantic similarity guarantee.

    device/backend select existing NumPy or Torch CPU/CUDA execution. No QPU,
    sampling, noise or classical text model is used.
    """
    circuit = sentence_pair_circuit(left, right).bind({} if bindings is None else bindings)
    result = Simulator.for_device(device, backend=backend).run(circuit, shots=0)
    return float(np.clip(result.probabilities[0], 0, 1))


@dataclass(frozen=True)
class DependencyParse:
    """One rooted dependency tree: tokens and zero-based head indices.

    The root heads itself. Token position determines wire order. This is a
    syntax-guided ansatz, not a DisCoCat/lambeq compiler or semantic parser.
    """
    tokens: tuple
    heads: tuple

    def __post_init__(self):
        if isinstance(self.tokens, str) or isinstance(self.heads, str):
            raise ValueError('Provide token and head sequences')
        tokens, heads = tuple(self.tokens), tuple(self.heads)
        if not tokens or len(tokens) != len(heads) or any(not isinstance(t, str) or not t.strip() for t in tokens):
            raise ValueError('Provide nonempty token strings and one head per token')
        if any(isinstance(h, bool) or not isinstance(h, numbers.Integral) or not 0 <= h < len(tokens) for h in heads):
            raise ValueError('Head indices must be integers within the sentence')
        roots = [i for i, h in enumerate(heads) if i == h]
        if len(roots) != 1: raise ValueError('Exactly one self-headed root is required')
        for start in range(len(tokens)):
            seen = set(); node = start
            while node != roots[0]:
                if node in seen: raise ValueError('Dependency heads contain a cycle')
                seen.add(node); node = heads[node]
        object.__setattr__(self, 'tokens', tokens)
        object.__setattr__(self, 'heads', tuple(int(h) for h in heads))

    @classmethod
    def from_spacy(cls, sentence):
        """Adapt an already parsed spaCy sentence Span/Doc without importing spaCy.

        No model loading/download. All heads must be inside the supplied single
        sentence; original document offsets are remapped to local wire indices.
        """
        tokens = list(sentence)
        if not tokens: raise ValueError('Supply a parsed, nonempty sentence')
        try:
            indices = [token.i for token in tokens]
            if any(not token.dep_ for token in tokens): raise ValueError('Dependency annotations are missing')
            if any(isinstance(i, bool) or not isinstance(i, numbers.Integral) for i in indices):
                raise ValueError('Token indices must be integers')
            mapping = {index: i for i, index in enumerate(indices)}
            if len(mapping) != len(tokens): raise ValueError('Duplicate token indices')
            return cls(tuple(token.text for token in tokens), tuple(mapping[token.head.i] for token in tokens))
        except (AttributeError, KeyError) as exc:
            raise ValueError('Expected parsed tokens with i, text, dep_ and an internal head.i') from exc

    def circuit(self, features, *, weights=None, max_qubits=10):
        """Encode one angle/token, then ordered CX(head, child) dependency edges.

        Optional Y variational layers use the same directed edges. Features are
        supplied externally; grammatical labels are not learned gate meanings.
        """
        if len(self.tokens) > positive_integer(max_qubits, 'max_qubits'):
            raise ValueError('Sentence exceeds max_qubits; shorten the sentence explicitly')
        values = np.asarray(features, dtype=object)
        if values.shape != (len(self.tokens),): raise ValueError('Provide one angle per token')
        edges = [(head, child) for child, head in enumerate(self.heads) if child != head]
        return sentence_encoder(values, weights=weights, entanglement=edges, max_qubits=max_qubits)

    def annotations(self):
        """JSON-ready word/wire and head mapping for project-level visualizations."""
        return [dict(wire=i, token=t, head=self.heads[i]) for i, t in enumerate(self.tokens)]
