"""Running circuits: :func:`run` for counts, and :func:`statevector`, :func:`probabilities` and
:func:`expectation` for exact answers.

Every run is seeded. When no seed is given one is drawn and recorded, so any result can be
repeated exactly: the same program, noise, shots, seed, method and Lucivant Orrery version give the
same counts. :attr:`Result.fingerprint` identifies those inputs and :attr:`Result.counts_digest`
the outcome, and :func:`verify` reruns a saved result to check it.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from lucivant_sim import __version__
from lucivant_sim.circuit import Barrier, Circuit, Gate, GlobalPhase, IfElse, Measure, Operation, Reset
from lucivant_sim.engines import (
    MAX_DENSITY_QUBITS,
    DensityMatrix,
    SimulationError,
    Statevector,
    marginal,
)
from lucivant_sim.gates import PAULIS
from lucivant_sim.noise import NoiseModel
from lucivant_sim.qasm import load, parse
from lucivant_sim.stabilizer import clifford_problem
from lucivant_sim.stabilizer import run_counts as stabilizer_counts

Method = Literal[
    "automatic", "statevector", "density_matrix", "trajectories", "stabilizer", "statevector_aer", "stabilizer_stim"
]
METHODS: tuple[str, ...] = (
    "automatic",
    "statevector",
    "density_matrix",
    "trajectories",
    "stabilizer",
    # opt-in accelerators (lucivant_sim.accel): never chosen automatically
    "statevector_aer",
    "stabilizer_stim",
)
#: Noisy circuits up to this many qubits use the exact density matrix when the method is automatic.
AUTO_DENSITY_QUBITS = 10
#: Clifford circuits (Pauli noise at most) from this many qubits use the stabilizer engine when the
#: method is automatic: measured, the statevector is faster at 8 qubits and the stabilizer about 10x
#: faster at 16 (QF-002 Phase 3 Stage C, benchmarks/qf002_phase3/results/competitive).
AUTO_STABILIZER_QUBITS = 12
MAX_SHOTS = 10_000_000

Program = str | os.PathLike[str] | Circuit | Any


# ------------------------------------------------------------------------------- programs


def to_circuit(program: Program, params: Mapping[str, float] | None = None) -> Circuit:
    """A Circuit from OpenQASM text, a .qasm file, a Qiskit QuantumCircuit or a Cirq Circuit."""
    if isinstance(program, Circuit):
        if params:
            raise ValueError("params apply to OpenQASM source, not to an already-built Circuit")
        return program
    if isinstance(program, os.PathLike):
        return load(Path(program), params)
    if isinstance(program, str):
        return parse(program, params)
    module = type(program).__module__
    if module.startswith("qiskit"):
        from qiskit import qasm3

        return parse(qasm3.dumps(program), params)
    if module.startswith("cirq"):
        import cirq

        return parse(cirq.qasm(program), params)
    raise TypeError(f"can't simulate a {type(program).__name__}; give OpenQASM, a path, or a Qiskit or Cirq circuit")


# -------------------------------------------------------------------------------- results


def _digest(data: Any) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(data, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class Result:
    counts: dict[str, int]
    shots: int
    seed: int
    method: str
    num_qubits: int
    num_clbits: int
    circuit_digest: str
    noise_digest: str | None
    version: str
    elapsed_seconds: float
    #: The final state of an ideal, static circuit before its measurements (statevector method).
    statevector: NDArray[np.complex128] | None = field(default=None, repr=False, compare=False)

    @property
    def fingerprint(self) -> str:
        """Identifies what was run: same fingerprint, same counts."""
        return _digest(
            {
                "engine": "lucivant-sim",
                "version": self.version,
                "circuit": self.circuit_digest,
                "noise": self.noise_digest,
                "shots": self.shots,
                "seed": self.seed,
                "method": self.method,
            }
        )

    @property
    def counts_digest(self) -> str:
        return _digest(self.counts)

    def frequencies(self) -> dict[str, float]:
        return {k: v / self.shots for k, v in self.counts.items()} if self.shots else {}

    def most_frequent(self) -> str | None:
        return max(self.counts, key=lambda k: (self.counts[k], k)) if self.counts else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": "lucivant-sim",
            "version": self.version,
            "method": self.method,
            "shots": self.shots,
            "seed": self.seed,
            "num_qubits": self.num_qubits,
            "num_clbits": self.num_clbits,
            "counts": self.counts,
            "circuit_digest": self.circuit_digest,
            "noise_digest": self.noise_digest,
            "fingerprint": self.fingerprint,
            "counts_digest": self.counts_digest,
            "elapsed_seconds": self.elapsed_seconds,
        }


# ---------------------------------------------------------------------------- execution


def choose_method(circuit: Circuit, noise: NoiseModel | None, method: str = "automatic") -> str:
    if method not in METHODS:
        raise ValueError(f"method must be one of {', '.join(METHODS)}")
    gate_noise = noise is not None and noise.has_gate_errors()
    dynamic = circuit.is_dynamic()
    if method == "automatic":
        if circuit.num_qubits >= AUTO_STABILIZER_QUBITS and clifford_problem(circuit, noise) is None:
            return "stabilizer"
        if dynamic:
            return "trajectories"
        if gate_noise:
            return "density_matrix" if circuit.num_qubits <= AUTO_DENSITY_QUBITS else "trajectories"
        return "statevector"
    if method == "stabilizer":
        problem = clifford_problem(circuit, noise)
        if problem:
            raise SimulationError(f"the stabilizer method runs Clifford circuits only: {problem}")
        return method
    if method == "stabilizer_stim":
        if noise is not None and not noise.is_empty():
            raise SimulationError("stabilizer_stim runs ideal circuits; use 'stabilizer' for Pauli noise")
        if any(isinstance(op, IfElse) for op in circuit.operations):
            raise SimulationError("stabilizer_stim runs circuits without classical conditions; use 'stabilizer'")
        problem = clifford_problem(circuit, None)
        if problem:
            raise SimulationError(f"the stabilizer methods run Clifford circuits only: {problem}")
        return method
    if method in ("statevector", "statevector_aer") and (gate_noise or dynamic):
        raise SimulationError(
            "the statevector method runs ideal circuits without mid-circuit measurement; "
            "use 'density_matrix' or 'trajectories'"
        )
    if method == "density_matrix" and dynamic:
        raise SimulationError("this circuit uses measurement results mid-circuit; use 'trajectories'")
    if method == "density_matrix" and circuit.num_qubits > MAX_DENSITY_QUBITS:
        raise SimulationError(f"the density-matrix method handles up to {MAX_DENSITY_QUBITS} qubits")
    return method


def fuse_single_qubit_runs(ops: tuple[Operation, ...]) -> tuple[Operation, ...]:
    """Merge consecutive one-qubit gates on the same qubit into one gate (their matrix product), so
    the state is swept once per run instead of once per gate. Exact up to floating-point rounding;
    used only for ideal circuits (noise is attached to named gates). Anything else on a qubit (a
    multi-qubit gate, a measurement, a reset, a branch) ends that qubit's run."""
    out: list[Operation] = []
    pending: dict[int, int] = {}  # qubit -> index in out of its open one-qubit gate
    for op in ops:
        if isinstance(op, Gate) and len(op.qubits) == 1:
            (q,) = op.qubits
            at = pending.get(q)
            if at is not None:
                prev = out[at]
                assert isinstance(prev, Gate)
                matrix = np.asarray(op.matrix, dtype=np.complex128) @ np.asarray(prev.matrix, dtype=np.complex128)
                out[at] = Gate(name="fused", qubits=(q,), matrix=matrix)
                continue
            pending[q] = len(out)
            out.append(op)
            continue
        touched: tuple[int, ...] = ()
        if isinstance(op, Gate):
            touched = op.qubits
        elif isinstance(op, (Measure, Reset)):
            touched = (op.qubit,)
        elif isinstance(op, (IfElse, Barrier)):
            pending.clear()  # conservative: a branch or barrier ends every run
        for q in touched:
            pending.pop(q, None)
        out.append(op)
    return tuple(out)


def _evolve(
    state: Statevector | DensityMatrix,
    ops: tuple[Operation, ...],
    noise: NoiseModel | None,
) -> None:
    """Static evolution: gates, noise and resets of untouched qubits; measurements deferred."""
    if noise is None and isinstance(state, Statevector):
        ops = fuse_single_qubit_runs(ops)
    for op in ops:
        if isinstance(op, Gate):
            state.apply(op.matrix, op.qubits)
            channel = noise.gate_error(op.name, op.qubits) if noise else None
            if channel is not None:
                if isinstance(state, DensityMatrix):
                    state.apply_channel(channel, op.qubits)
                else:  # pragma: no cover - choose_method prevents it
                    raise SimulationError("gate noise needs the density matrix or trajectories")
        elif isinstance(op, GlobalPhase):
            state.scale(np.exp(1j * op.angle))
        elif isinstance(op, Reset) and isinstance(state, DensityMatrix):
            state.reset(op.qubit)
        # Measure: deferred to the end (nothing touches the qubit afterwards). Reset of an
        # untouched qubit: no effect on |0>. Barrier: nothing.


def _sample(
    circuit: Circuit,
    probs: NDArray[np.float64],
    shots: int,
    noise: NoiseModel | None,
    rng: np.random.Generator,
) -> dict[str, int]:
    pairs = circuit.final_measurements()
    if not pairs or shots == 0:
        return {}
    measured = sorted({q for q, _ in pairs})
    position = {q: i for i, q in enumerate(measured)}
    dist = marginal(probs, circuit.num_qubits, measured)
    cumulative = np.cumsum(dist)
    cumulative /= cumulative[-1]
    outcomes = np.searchsorted(cumulative, rng.random(shots), side="right").astype(np.int64)
    outcomes = np.minimum(outcomes, len(dist) - 1)
    # more than 62 classical bits don't fit an int64: use Python integers
    bits = np.zeros(shots, dtype=object if circuit.num_clbits > 62 else np.int64)
    for q, c in pairs:
        value = (outcomes >> position[q]) & 1
        err = noise.readout_error(q) if noise else None
        if err is not None and (err.p01 or err.p10):
            flip = rng.random(shots) < np.where(value == 1, err.p10, err.p01)
            value = value ^ flip.astype(np.int64)
        if bits.dtype == object:
            bits = bits + np.array([int(v) << c for v in value], dtype=object)
        else:
            bits |= value << c
    values, counts = np.unique(bits, return_counts=True)
    return {circuit.format_bits(int(v)): int(n) for v, n in zip(values, counts, strict=True)}


def _trajectory(
    circuit: Circuit, noise: NoiseModel | None, rng: np.random.Generator
) -> int:  # one shot; returns the classical bits
    state = Statevector(circuit.num_qubits)
    clbits = [0] * circuit.num_clbits

    def run_ops(ops: tuple[Operation, ...]) -> None:
        for op in ops:
            if isinstance(op, Gate):
                state.apply(op.matrix, op.qubits)
                channel = noise.gate_error(op.name, op.qubits) if noise else None
                if channel is not None:
                    state.apply_channel(channel, op.qubits, rng)
            elif isinstance(op, Measure):
                outcome = state.measure(op.qubit, rng)
                err = noise.readout_error(op.qubit) if noise else None
                if err is not None and rng.random() < (err.p10 if outcome else err.p01):
                    outcome ^= 1
                if op.clbit is not None:
                    clbits[op.clbit] = outcome
            elif isinstance(op, Reset):
                state.reset(op.qubit, rng)
            elif isinstance(op, GlobalPhase):
                state.scale(np.exp(1j * op.angle))
            elif isinstance(op, IfElse):
                run_ops(op.then if op.condition.holds(clbits) else op.otherwise)
            elif isinstance(op, Barrier):
                continue

    run_ops(circuit.operations)
    return sum(b << i for i, b in enumerate(clbits))


def run(
    program: Program,
    shots: int = 1024,
    *,
    seed: int | None = None,
    noise: NoiseModel | None = None,
    method: Method | str = "automatic",
    params: Mapping[str, float] | None = None,
) -> Result:
    """Simulate ``program`` ``shots`` times and return the counts.

    ``method`` is chosen for you by default: the exact statevector for ideal circuits, the exact
    density matrix for noisy ones (up to 10 qubits), and per-shot trajectories for circuits that
    use measurement results mid-circuit or are too wide for a density matrix.
    Clifford circuits (with at most Pauli noise) of 12 qubits or more use the stabilizer engine
    (up to 2,048 qubits; :mod:`lucivant_sim.stabilizer`); ``method="stabilizer"`` asks for it at any
    size and refuses any non-Clifford gate by name.
    """
    if not 0 <= shots <= MAX_SHOTS:
        raise ValueError(f"shots must be between 0 and {MAX_SHOTS:,}")
    circuit = to_circuit(program, params)
    chosen = choose_method(circuit, noise, method)
    seed = secrets.randbelow(2**63) if seed is None else int(seed)
    if seed < 0:
        raise ValueError("seed must not be negative")
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    vector = None
    version = __version__
    if chosen == "statevector_aer":
        from lucivant_sim.accel import aer_statevector, aer_version

        vector = aer_statevector(circuit)
        version = f"{__version__}+qiskit-aer-{aer_version()}"
        counts = _sample(circuit, np.abs(vector) ** 2, shots, noise, rng)
    elif chosen == "stabilizer_stim":
        from lucivant_sim.accel import stim_counts, stim_version

        tally_stim = stim_counts(circuit, shots, seed)
        version = f"{__version__}+stim-{stim_version()}"
        counts = {circuit.format_bits(b): n for b, n in sorted(tally_stim.items())} if circuit.num_clbits else {}
    elif chosen == "stabilizer":
        tally_stab = stabilizer_counts(circuit, shots, noise, rng)
        counts = {circuit.format_bits(b): n for b, n in sorted(tally_stab.items())} if circuit.num_clbits else {}
    elif chosen == "trajectories":
        if circuit.num_qubits > 0:
            Statevector(circuit.num_qubits)  # size check before the loop
        tally: dict[int, int] = {}
        for _ in range(shots):
            bits = _trajectory(circuit, noise, rng)
            tally[bits] = tally.get(bits, 0) + 1
        counts = {circuit.format_bits(b): n for b, n in sorted(tally.items())} if circuit.num_clbits else {}
    else:
        state: Statevector | DensityMatrix = (
            Statevector(circuit.num_qubits) if chosen == "statevector" else DensityMatrix(circuit.num_qubits)
        )
        _evolve(state, circuit.operations, noise)
        if isinstance(state, Statevector):
            vector = state.vector()
        counts = _sample(circuit, state.probabilities(), shots, noise, rng)
    elapsed = round(time.perf_counter() - started, 6)
    return Result(
        counts=dict(sorted(counts.items())),
        shots=shots,
        seed=seed,
        method=chosen,
        num_qubits=circuit.num_qubits,
        num_clbits=circuit.num_clbits,
        circuit_digest=circuit.digest(),
        noise_digest=noise.digest() if noise is not None and not noise.is_empty() else None,
        version=version,
        elapsed_seconds=elapsed,
        statevector=vector,
    )


# ------------------------------------------------------------------------------ exact answers


def _static(circuit: Circuit, what: str) -> None:
    if circuit.is_dynamic():
        raise SimulationError(f"{what} needs a circuit without mid-circuit measurement or conditions")


def statevector(program: Program, params: Mapping[str, float] | None = None) -> NDArray[np.complex128]:
    """The ideal final state (measurements ignored), little-endian: index bit q is qubit q."""
    circuit = to_circuit(program, params)
    _static(circuit, "a statevector")
    state = Statevector(circuit.num_qubits)
    _evolve(state, circuit.operations, None)
    return state.vector()


def density_matrix(
    program: Program, noise: NoiseModel | None = None, params: Mapping[str, float] | None = None
) -> NDArray[np.complex128]:
    """The final mixed state under ``noise`` (measurements ignored)."""
    circuit = to_circuit(program, params)
    _static(circuit, "a density matrix")
    state = DensityMatrix(circuit.num_qubits)
    _evolve(state, circuit.operations, noise)
    return state.matrix()


def probabilities(
    program: Program,
    noise: NoiseModel | None = None,
    params: Mapping[str, float] | None = None,
    *,
    threshold: float = 1e-12,
) -> dict[str, float]:
    """Exact probabilities of every basis state of all qubits (qubit 0 rightmost), before any
    readout error. Ideal circuits use the statevector, noisy ones the density matrix."""
    circuit = to_circuit(program, params)
    _static(circuit, "exact probabilities")
    gate_noise = noise is not None and noise.has_gate_errors()
    state: Statevector | DensityMatrix = (
        DensityMatrix(circuit.num_qubits) if gate_noise else Statevector(circuit.num_qubits)
    )
    _evolve(state, circuit.operations, noise)
    probs = state.probabilities()
    n = circuit.num_qubits
    return {format(i, f"0{n}b"): float(p) for i, p in enumerate(probs) if p > threshold}


def outcome_probabilities(
    program: Program,
    noise: NoiseModel | None = None,
    params: Mapping[str, float] | None = None,
    *,
    threshold: float = 1e-12,
) -> dict[str, float]:
    """Exact probabilities of the classical outcomes, keyed as :attr:`Result.counts` is, including
    readout error. For static circuits (no measurement results used mid-circuit)."""
    circuit = to_circuit(program, params)
    _static(circuit, "exact outcome probabilities")
    pairs = circuit.final_measurements()
    if not pairs:
        return {}
    gate_noise = noise is not None and noise.has_gate_errors()
    state: Statevector | DensityMatrix = (
        DensityMatrix(circuit.num_qubits) if gate_noise else Statevector(circuit.num_qubits)
    )
    _evolve(state, circuit.operations, noise)
    measured = sorted({q for q, _ in pairs})
    position = {q: i for i, q in enumerate(measured)}
    dist = marginal(state.probabilities(), circuit.num_qubits, measured)
    out: dict[int, float] = {}
    for index, p in enumerate(dist):
        if p <= threshold:
            continue
        # every combination of readout flips, weighted by its probability
        branches = [(0, float(p))]
        for q, c in pairs:
            bit = (index >> position[q]) & 1
            err = noise.readout_error(q) if noise else None
            flip = (err.p10 if bit else err.p01) if err is not None else 0.0
            nxt = []
            for bits, w in branches:
                if flip < 1:
                    nxt.append((bits | (bit << c), w * (1 - flip)))
                if flip > 0:
                    nxt.append((bits | ((bit ^ 1) << c), w * flip))
            branches = nxt
        for bits, w in branches:
            out[bits] = out.get(bits, 0.0) + w
    return {circuit.format_bits(b): w for b, w in sorted(out.items()) if w > threshold}


def expectation(
    program: Program,
    observable: str | Mapping[str, float],
    noise: NoiseModel | None = None,
    params: Mapping[str, float] | None = None,
) -> float:
    """<O> for a Pauli string ("ZZ", qubit 0 rightmost) or a weighted sum {"ZI": 0.5, "XX": -1}."""
    circuit = to_circuit(program, params)
    _static(circuit, "an expectation value")
    terms = {observable: 1.0} if isinstance(observable, str) else dict(observable)
    n = circuit.num_qubits
    for label in terms:
        if len(label) != n or set(label.upper()) - set("IXYZ"):
            raise ValueError(f"{label!r} must be a Pauli string of length {n} (qubit 0 rightmost)")
    gate_noise = noise is not None and noise.has_gate_errors()
    total = 0.0
    if gate_noise:
        dm = DensityMatrix(n)
        _evolve(dm, circuit.operations, noise)
        for label, weight in terms.items():
            probe = DensityMatrix.__new__(DensityMatrix)
            probe.n, probe.tensor = n, dm.tensor
            for q, letter in enumerate(reversed(label.upper())):
                if letter != "I":
                    probe.tensor = _left(probe, PAULIS[letter], q)
            d = 2**n
            total += weight * float(np.trace(probe.tensor.reshape(d, d)).real)
        return total
    sv = Statevector(n)
    _evolve(sv, circuit.operations, None)
    ket = sv.vector()
    for label, weight in terms.items():
        ket_probe = Statevector(n)
        ket_probe.tensor = ket.reshape([2] * n).copy() if n else ket.copy()  # gates apply in place
        for q, letter in enumerate(reversed(label.upper())):
            if letter != "I":
                ket_probe.apply(PAULIS[letter], (q,))
        total += weight * float(np.vdot(ket, ket_probe.vector()).real)
    return total


def _left(dm: DensityMatrix, matrix: NDArray[np.complex128], qubit: int) -> NDArray[np.complex128]:
    from lucivant_sim.engines import _apply

    return _apply(dm.tensor, matrix, [dm.n - 1 - qubit])


# ------------------------------------------------------------------------------ verification


def verify(saved: Mapping[str, Any] | str | Path, program: Program, noise: NoiseModel | None = None) -> bool:
    """Rerun a saved result (``Result.to_dict()``) and check the counts are identical."""
    data = json.loads(Path(saved).read_text(encoding="utf-8")) if isinstance(saved, (str, Path)) else dict(saved)
    again = run(program, int(data["shots"]), seed=int(data["seed"]), noise=noise, method=str(data["method"]))
    if again.circuit_digest != data.get("circuit_digest"):
        return False
    if (again.noise_digest or None) != (data.get("noise_digest") or None):
        return False
    return again.counts_digest == data.get("counts_digest") and again.counts == data.get("counts")
