"""Noise: quantum channels after gates, and readout errors on measurement.

Build a model yourself, or from a device calibration in Lucivant's twin format, so the same
noise a Lucivant twin uses is available offline::

    noise = NoiseModel.uniform(single=1e-3, two=1e-2, readout=2e-2)
    noise = NoiseModel.from_calibration("ibm_device.json")

A calibration gives per-qubit and per-coupler errors where the maker publishes them and the
medians elsewhere. Lucivant Orrery applies them to the gates your circuit contains; it doesn't compile
to the device's own gates and couplers first (a Lucivant twin in Studio does), so compile your
circuit for the device beforehand for a faithful emulation.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from lucivant_sim.gates import PAULIS, Matrix, canonical_name

#: Gates that are frame changes on real devices: no error is attached to them by default.
IDEAL_GATES = frozenset({"rz", "p", "phase", "u1", "id", "i", "u0"})


@dataclass(frozen=True, eq=False)
class Channel:
    """A completely positive, trace-preserving map given by Kraus operators.

    ``mixture`` is set when the channel is a probabilistic mix of unitaries (Pauli channels), so
    trajectories can pick a branch without computing norms.
    """

    name: str
    num_qubits: int
    kraus: tuple[Matrix, ...]
    params: tuple[float, ...] = ()
    mixture: tuple[tuple[float, Matrix], ...] | None = None
    #: Set only by the depolarizing factory: permits the equivalent partial-trace density update.
    analytic_depolarizing: bool = False

    def __post_init__(self) -> None:
        d = 2**self.num_qubits
        total = sum(k.conj().T @ k for k in self.kraus)
        if not np.allclose(total, np.eye(d), atol=1e-9):
            raise ValueError(f"{self.name}: the Kraus operators don't preserve probability")

    def describe(self) -> list[Any]:
        return [self.name, self.num_qubits, [round(p, 15) for p in self.params]]


def _probability(p: float, what: str, upper: float = 1.0) -> float:
    if not (0.0 <= p <= upper) or math.isnan(p):
        raise ValueError(f"{what} must be between 0 and {upper:g}, not {p}")
    return float(p)


def pauli_channel(probabilities: Mapping[str, float]) -> Channel:
    """Apply Pauli string P with probability probabilities[P] (identity otherwise)."""
    k = len(next(iter(probabilities)))
    total = sum(probabilities.values())
    _probability(total, "the total Pauli probability")
    branches: list[tuple[float, Matrix]] = []
    if total < 1:
        branches.append((1 - total, np.eye(2**k, dtype=np.complex128)))
    for label, p in sorted(probabilities.items()):
        if len(label) != k or set(label) - set("IXYZ"):
            raise ValueError(f"{label!r} isn't a {k}-qubit Pauli string")
        if p <= 0:
            continue
        m = np.array([[1]], dtype=np.complex128)
        for letter in label:
            m = np.kron(m, PAULIS[letter]).astype(np.complex128)
        branches.append((p, m))
    kraus = tuple(math.sqrt(p) * m for p, m in branches)
    params = tuple(probabilities[key] for key in sorted(probabilities))
    return Channel("pauli", k, kraus, params, tuple(branches))


def depolarizing(p: float, num_qubits: int = 1) -> Channel:
    """With probability p, replace the state by the maximally mixed state.

    Equivalent to each non-identity Pauli with probability p/4**k. A maker's average gate error
    r corresponds to p = r * d / (d - 1), d = 2**k (see :func:`from_gate_error`).
    """
    d2 = 4**num_qubits
    _probability(p, "a depolarising probability", d2 / (d2 - 1))
    probs = {"".join(label): p / d2 for label in itertools.product("IXYZ", repeat=num_qubits) if set(label) != {"I"}}
    ch = pauli_channel(probs)
    return Channel("depolarizing", num_qubits, ch.kraus, (p,), ch.mixture, analytic_depolarizing=True)


def from_gate_error(error: float, num_qubits: int) -> Channel:
    """The depolarising channel whose average gate infidelity equals ``error``."""
    d = 2**num_qubits
    p = min(error * d / (d - 1), 4**num_qubits / (4**num_qubits - 1))
    return depolarizing(p, num_qubits)


def bit_flip(p: float) -> Channel:
    ch = pauli_channel({"X": _probability(p, "a bit-flip probability")})
    return Channel("bit_flip", 1, ch.kraus, (p,), ch.mixture)


def phase_flip(p: float) -> Channel:
    ch = pauli_channel({"Z": _probability(p, "a phase-flip probability")})
    return Channel("phase_flip", 1, ch.kraus, (p,), ch.mixture)


def amplitude_damping(gamma: float) -> Channel:
    """Energy relaxation: |1> decays to |0> with probability gamma."""
    g = _probability(gamma, "gamma")
    k0 = np.array([[1, 0], [0, math.sqrt(1 - g)]], dtype=np.complex128)
    k1 = np.array([[0, math.sqrt(g)], [0, 0]], dtype=np.complex128)
    return Channel("amplitude_damping", 1, (k0, k1), (g,))


def phase_damping(lam: float) -> Channel:
    """Loss of phase coherence without energy loss."""
    lam = _probability(lam, "lambda")
    k0 = np.array([[1, 0], [0, math.sqrt(1 - lam)]], dtype=np.complex128)
    k1 = np.array([[0, 0], [0, math.sqrt(lam)]], dtype=np.complex128)
    return Channel("phase_damping", 1, (k0, k1), (lam,))


def thermal_relaxation(t1: float, t2: float, time: float) -> Channel:
    """Relaxation (T1) and dephasing (T2 <= 2*T1) over ``time``, all in the same unit."""
    if t1 <= 0 or t2 <= 0 or time < 0:
        raise ValueError("T1 and T2 must be positive and the time not negative")
    if t2 > 2 * t1 + 1e-12:
        raise ValueError("T2 can't exceed 2*T1")
    gamma = 1 - math.exp(-time / t1)
    lam = max(0.0, 1 - math.exp(-2 * time / t2 + time / t1))
    ad, pd = amplitude_damping(gamma), phase_damping(min(lam, 1.0))
    kraus = tuple(b @ a for a in ad.kraus for b in pd.kraus)
    return Channel("thermal_relaxation", 1, kraus, (t1, t2, time))


@dataclass(frozen=True)
class ReadoutError:
    """p01: chance of reading 1 when the qubit was 0; p10: reading 0 when it was 1."""

    p01: float
    p10: float

    def __post_init__(self) -> None:
        _probability(self.p01, "p01")
        _probability(self.p10, "p10")

    @classmethod
    def symmetric(cls, p: float) -> ReadoutError:
        return cls(p, p)


@dataclass
class _Rule:
    channel: Channel
    gates: frozenset[str] | None
    qubits: tuple[int, ...] | None

    def score(self, name: str, qubits: tuple[int, ...]) -> float:
        if self.channel.num_qubits != len(qubits):
            return -1
        if self.gates is not None and name not in self.gates:
            return -1
        if self.qubits is not None and tuple(sorted(self.qubits)) != tuple(sorted(qubits)):
            return -1
        if self.gates is None and name in IDEAL_GATES:
            return -1
        return (2 if self.qubits is not None else 0) + (1 if self.gates is not None else 0)


@dataclass
class NoiseModel:
    """Which channel follows which gate, and which readout error each qubit has.

    The most specific rule wins: a rule for a gate on given qubits, then for any gate on those
    qubits, then for a gate anywhere, then for any gate of that size. Among equally specific
    rules the last one added wins. Frame-change gates (rz, p, u1, id) are error-free unless a
    rule names them.
    """

    rules: list[_Rule] = field(default_factory=list)
    readout: dict[int | None, ReadoutError] = field(default_factory=dict)
    source: str | None = None

    def add_gate_error(
        self,
        channel: Channel,
        gates: str | Iterable[str] | None = None,
        qubits: Sequence[int] | None = None,
    ) -> NoiseModel:
        names = (
            frozenset(canonical_name(g) for g in ([gates] if isinstance(gates, str) else gates))
            if gates is not None
            else None
        )
        if qubits is not None and len(qubits) != channel.num_qubits:
            raise ValueError(f"a {channel.num_qubits}-qubit channel can't attach to {len(qubits)} qubit(s)")
        self.rules.append(_Rule(channel, names, tuple(qubits) if qubits is not None else None))
        return self

    def add_readout_error(self, error: float | ReadoutError, qubits: Iterable[int] | None = None) -> NoiseModel:
        err = error if isinstance(error, ReadoutError) else ReadoutError.symmetric(float(error))
        for q in [None] if qubits is None else qubits:
            self.readout[q] = err
        return self

    # -- lookups --------------------------------------------------------------------------

    def gate_error(self, name: str, qubits: tuple[int, ...]) -> Channel | None:
        best, best_score = None, -1.0
        for rule in self.rules:
            score = rule.score(name, qubits)
            if score >= best_score and score >= 0:
                best, best_score = rule.channel, score
        return best

    def readout_error(self, qubit: int) -> ReadoutError | None:
        return self.readout.get(qubit, self.readout.get(None))

    def has_gate_errors(self) -> bool:
        return bool(self.rules)

    def is_empty(self) -> bool:
        return not self.rules and not any(e.p01 or e.p10 for e in self.readout.values())

    def describe(self) -> dict[str, Any]:
        return {
            "rules": [
                [sorted(r.gates) if r.gates is not None else None, list(r.qubits or []) or None, r.channel.describe()]
                for r in self.rules
            ],
            "readout": sorted(
                [[-1 if q is None else q, round(e.p01, 15), round(e.p10, 15)] for q, e in self.readout.items()]
            ),
        }

    def digest(self) -> str:
        data = json.dumps(self.describe(), separators=(",", ":"), sort_keys=True).encode()
        return "sha256:" + hashlib.sha256(data).hexdigest()

    # -- builders -------------------------------------------------------------------------

    @classmethod
    def uniform(cls, single: float = 0.0, two: float = 0.0, readout: float = 0.0) -> NoiseModel:
        """The same average gate error for every one- and two-qubit gate, and one readout error."""
        model = cls(source="uniform")
        if single > 0:
            model.add_gate_error(from_gate_error(single, 1))
        if two > 0:
            model.add_gate_error(from_gate_error(two, 2))
        if readout > 0:
            model.add_readout_error(readout)
        return model

    @classmethod
    def from_calibration(cls, calibration: Mapping[str, Any] | str | Path) -> NoiseModel:
        """A model from a device calibration in Lucivant's twin format (``DeviceCalibration``).

        Uses ``single_qubit_error``, ``two_qubit_error`` and ``readout_error`` (medians) and,
        when present, ``qubits[].single_qubit_error``/``readout_error`` and
        ``couplers[].qubits``/``error``. Fields that are missing or null add no noise.
        """
        if isinstance(calibration, (str, Path)):
            cal: dict[str, Any] = json.loads(Path(calibration).read_text(encoding="utf-8"))
        else:
            cal = dict(calibration)
        if "calibration" in cal and isinstance(cal["calibration"], Mapping):
            cal = dict(cal["calibration"])  # a twin export wraps it
        model = cls(source=str(cal.get("device") or cal.get("profile_key") or "calibration"))
        single, two, readout = cal.get("single_qubit_error"), cal.get("two_qubit_error"), cal.get("readout_error")
        if single:
            model.add_gate_error(from_gate_error(float(single), 1))
        if two:
            model.add_gate_error(from_gate_error(float(two), 2))
        if readout:
            model.add_readout_error(float(readout))
        for row in cal.get("qubits") or []:
            q = int(row["qubit"])
            if row.get("single_qubit_error") is not None:
                model.add_gate_error(from_gate_error(float(row["single_qubit_error"]), 1), qubits=[q])
            if row.get("readout_error") is not None:
                model.add_readout_error(float(row["readout_error"]), qubits=[q])
        for row in cal.get("couplers") or []:
            if row.get("error") is not None:
                a, b = (int(x) for x in row["qubits"])
                model.add_gate_error(from_gate_error(float(row["error"]), 2), qubits=[a, b])
        return model
