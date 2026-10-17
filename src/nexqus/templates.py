"""Reusable QML blocks that append ordinary gates to a Circuit.

All helpers return the supplied circuit. Wires are ordered, so the first
feature/weight belongs to wires[0]. Numeric angles and NexQus Expressions are
supported; no external autodifferentiation framework is required.
"""
import numbers
import numpy as np
from .circuit import Circuit, positive_integer
from .parameters import Expression, real_scalar

__all__ = ['basis_encoding', 'angle_encoding', 'entangling_layer',
           'layer_shape', 'variational_layers', 'data_reuploading']


def _target(circuit, wires):
    if not isinstance(circuit, Circuit): raise TypeError('Expected a Circuit')
    return circuit._wires(range(circuit.n_qubits) if wires is None else wires)


def _rotations(rotations):
    if not isinstance(rotations, str) or not rotations or set(rotations)-set('XYZ'):
        raise ValueError("rotations must be a nonempty string of X, Y, Z axes")
    return rotations


def _angles(values, name):
    array = np.asarray(values, dtype=object)
    checked = [v if isinstance(v, Expression) else real_scalar(v) for v in array.flat]
    return np.array(checked, dtype=object).reshape(array.shape)


def _edges(wires, entanglement):
    if isinstance(entanglement, str):
        if entanglement == 'linear': return list(zip(wires[:-1], wires[1:]))
        if entanglement == 'ring':
            pairs = list(zip(wires[:-1], wires[1:]))
            return pairs + ([(wires[-1], wires[0])] if len(wires) > 2 else [])
        if entanglement == 'full':
            return [(a, b) for i, a in enumerate(wires) for b in wires[i+1:]]
        raise ValueError("entanglement must be 'linear', 'ring', 'full' or an iterable of wire pairs")
    try:
        pairs = [tuple(pair) for pair in entanglement]
    except TypeError as exc:
        raise ValueError('Entanglement edges must be wire pairs') from exc
    if any(len(pair) != 2 or any(isinstance(q, bool) or not isinstance(q, numbers.Integral)
                               or q not in wires for q in pair) or pair[0] == pair[1] for pair in pairs):
        raise ValueError('Each edge must join two distinct selected wires')
    if len(set(pairs)) != len(pairs): raise ValueError('Duplicate entanglement edge')
    return pairs


def _entangler(entangler):
    if entangler not in ('CX', 'CZ'): raise ValueError("entangler must be 'CX' or 'CZ'")
    return entangler.lower()


def basis_encoding(circuit, bits, wires=None):
    """Apply X for each 1 bit; prepares the basis state from an all-zero input.

    Exactly one integer 0/1 is required per selected wire. On a nonzero input
    this is an XOR with the supplied bit string, not an unconditional reset.
    """
    wires = _target(circuit, wires)
    bits = np.asarray(bits, dtype=object)
    if bits.shape != (len(wires),) or any(not isinstance(b, numbers.Integral) or b not in (0, 1) for b in bits):
        raise ValueError('Provide one integer 0/1 bit per selected wire')
    for bit, wire in zip(bits, wires):
        if bit: circuit.x(wire)
    return circuit


def angle_encoding(circuit, features, wires=None, *, rotation='Y', scale=1.0):
    """Encode 1..len(wires) features as R_axis(scale * feature), in radians.

    Unused selected wires remain untouched. Scaling is explicit; this helper
    does not fit a scaler to data. Features may be symbolic Expressions.
    """
    wires = _target(circuit, wires)
    if _rotations(rotation) not in ('X', 'Y', 'Z'):
        raise ValueError('Encoding rotation must be a single X, Y or Z axis')
    scale = real_scalar(scale)
    features = _angles(features, 'features')
    if features.ndim != 1 or not 1 <= len(features) <= len(wires):
        raise ValueError('features must be a nonempty vector with at most one value per selected wire')
    # Build separately, so even expression/overflow errors cannot partly edit
    # the caller's circuit.
    block = Circuit(circuit.n_qubits)
    for feature, wire in zip(features, wires):
        angle = feature if scale == 1 else scale * feature
        getattr(block, 'r'+rotation.lower())(angle, wire)
    return circuit.compose(block)


def entangling_layer(circuit, wires=None, *, entanglement='linear', entangler='CX'):
    """Append CX/CZ edges in order: linear, ring, full, or explicit wire pairs.

    Ring uses a single edge for two wires and no edges for one wire. Explicit
    pairs use actual circuit wire indices. An empty list disables entangling.
    """
    wires = _target(circuit, wires)
    gate = _entangler(entangler)
    pairs = _edges(wires, entanglement)
    for a, b in pairs: getattr(circuit, gate)(a, b)
    return circuit


def layer_shape(n_layers, n_wires, *, rotations='Y'):
    """Weight shape (layers, selected wires, number of rotation axes)."""
    return (positive_integer(n_layers, 'n_layers'), positive_integer(n_wires, 'n_wires'),
            len(_rotations(rotations)))


def _layer_spec(circuit, weights, wires, rotations, entanglement, entangler):
    wires = _target(circuit, wires)
    rotations = _rotations(rotations)
    weights = _angles(weights, 'weights')
    if (weights.ndim != 3 or weights.shape[0] < 1 or
            weights.shape[1:] != (len(wires), len(rotations))):
        raise ValueError('weights must have shape (layers, len(wires), len(rotations)); use layer_shape()')
    return wires, weights, rotations, _edges(wires, entanglement), _entangler(entangler)


def _append_layer(block, weights, wires, rotations, pairs, gate):
    for wire, angles in zip(wires, weights):
        for axis, angle in zip(rotations, angles): getattr(block, 'r'+axis.lower())(angle, wire)
    for a, b in pairs: getattr(block, gate)(a, b)


def variational_layers(circuit, weights, wires=None, *, rotations='Y',
                       entanglement='linear', entangler='CX'):
    """Append trainable rotations followed by entangling edges in every layer.

    weights has layer_shape(layers, wires, rotations=...) and may contain shared
    Parameters or Expressions. Axes apply left to right on each selected wire.
    """
    wires, weights, rotations, pairs, gate = _layer_spec(circuit, weights, wires, rotations,
                                                       entanglement, entangler)
    block = Circuit(circuit.n_qubits)
    for layer in weights: _append_layer(block, layer, wires, rotations, pairs, gate)
    return circuit.compose(block)


def data_reuploading(circuit, features, weights, wires=None, *, encoding_rotation='Y',
                     scale=1.0, rotations='Y', entanglement='linear', entangler='CX'):
    """Repeat [encode the same features, trainable rotations, entangle] per layer.

    This constructs a circuit, not a training loop. Combine it with
    QuantumFunction for binding and parameter-shift gradients.
    """
    wires, weights, rotations, pairs, gate = _layer_spec(circuit, weights, wires, rotations,
                                                       entanglement, entangler)
    features = _angles(features, 'features')
    block = Circuit(circuit.n_qubits)
    for layer in weights:
        angle_encoding(block, features, wires, rotation=encoding_rotation, scale=scale)
        _append_layer(block, layer, wires, rotations, pairs, gate)
    return circuit.compose(block)
