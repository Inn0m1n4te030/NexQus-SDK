"""Validated single-qubit CPTP channels and post-gate noise rules."""
from dataclasses import dataclass
import numpy as np
from .circuit import I, X, Y, Z


def _probability(p):
    p = float(p)
    if not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("Noise probability must lie in [0, 1]")
    return p


@dataclass(frozen=True, eq=False)
class Noise:
    kraus: tuple
    name: str = "custom"

    def __post_init__(self):
        ks = tuple(np.array(k, dtype=complex, copy=True) for k in self.kraus)
        if not ks or any(k.shape != (2,2) or not np.isfinite(k).all() for k in ks):
            raise ValueError("Kraus operators must be finite 2x2 matrices")
        if not np.allclose(sum(k.conj().T @ k for k in ks), I, atol=1e-10, rtol=1e-10):
            raise ValueError("Kraus operators must satisfy sum(K†K) = I")
        for k in ks: k.flags.writeable = False
        object.__setattr__(self, "kraus", ks)

    @classmethod
    def bit_flip(cls, p):
        p = _probability(p)
        return cls((np.sqrt(1-p)*I, np.sqrt(p)*X), "bit_flip")

    @classmethod
    def phase_flip(cls, p):
        p = _probability(p)
        return cls((np.sqrt(1-p)*I, np.sqrt(p)*Z), "phase_flip")

    @classmethod
    def depolarizing(cls, p):
        """E(rho) = (1-p)rho + p I/2 on the affected qubit."""
        p = _probability(p)
        return cls((np.sqrt(1-3*p/4)*I, np.sqrt(p/4)*X,
                    np.sqrt(p/4)*Y, np.sqrt(p/4)*Z), "depolarizing")

    @classmethod
    def amplitude_damping(cls, gamma):
        g = _probability(gamma)
        return cls((np.diag([1, np.sqrt(1-g)]), np.array([[0,np.sqrt(g)],[0,0]])),
                   "amplitude_damping")

    @classmethod
    def phase_damping(cls, gamma):
        """Populations unchanged; off-diagonal terms scaled by sqrt(1-gamma)."""
        g = _probability(gamma)
        return cls((np.diag([1, np.sqrt(1-g)]), np.diag([0, np.sqrt(g)])), "phase_damping")


class NoiseModel:
    """Apply channels after each gate, independently to each gate-touched wire.

    NoiseModel(Noise.depolarizing(.01)) applies globally.
    NoiseModel().after('CX', Noise.amplitude_damping(.02)) selects gate names.
    Explicit Circuit.noise() operations do not trigger post-gate rules.
    """
    def __init__(self, *channels):
        self.rules = []
        for channel in channels: self.after(None, channel)

    def after(self, gates, channel):
        if not isinstance(channel, Noise):
            raise TypeError("channel must be a Noise")
        names = None if gates is None else frozenset([gates] if isinstance(gates, str) else gates)
        self.rules.append((names, channel))
        return self

    def channels_for(self, gate_name):
        return [channel for names, channel in self.rules if names is None or gate_name in names]
