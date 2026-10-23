"""The stabilizer (Clifford) engine: a tableau, not a state vector.

Clifford circuits (H, S, S†, the Paulis, CX, CZ, SWAP, and any other gate on up to three qubits
whose matrix maps every Pauli to a Pauli) are simulated exactly with the Aaronson-Gottesman tableau
(Phys. Rev. A 70, 052328, 2004): 2n Pauli rows of n qubits, so memory grows as n² and each gate
costs O(n), instead of the 2**n amplitudes a statevector needs. Measurement in the computational
basis and reset are supported, with outcomes drawn from the run's seeded generator.

What it is not: a general simulator. A non-Clifford gate (T, a rotation by an angle that isn't a
multiple of π/2, Toffoli) is refused by name, never approximated, and the engine doesn't give
amplitudes (global phase isn't tracked). Gate noise is accepted only when every branch is a Pauli
(depolarizing, bit and phase flips, Pauli channels); readout errors are applied as elsewhere.

Sampling a static circuit (measurements at the end) needs one simulation: the outcomes of a
stabilizer state are uniform over an affine subspace x0 + V of bit strings, where V is spanned by
the X parts of the stabilizers. x0 comes from one measurement pass and every shot is x0 plus a
random combination of V's basis. Dynamic circuits and gate noise are run one shot at a time.

Qubit q is bit q of an outcome (little-endian), as in the other engines.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from lucivant_sim.circuit import Barrier, Circuit, Gate, GlobalPhase, IfElse, Measure, Operation, Reset
from lucivant_sim.engines import SimulationError
from lucivant_sim.noise import Channel, NoiseModel

#: Memory and time guard: the tableau is 4n² bytes, and a measurement pass costs O(n³).
MAX_STABILIZER_QUBITS = 2048
#: Gates wider than this are never Clifford-checked (4**k Pauli images of 2**k matrices).
MAX_CLIFFORD_GATE_QUBITS = 3

_PAULI_1Q = {
    (0, 0): np.eye(2, dtype=np.complex128),
    (1, 0): np.array([[0, 1], [1, 0]], dtype=np.complex128),
    (1, 1): np.array([[0, -1j], [1j, 0]], dtype=np.complex128),  # x=z=1 is Y, as in the tableau
    (0, 1): np.array([[1, 0], [0, -1]], dtype=np.complex128),
}
_LETTER = {"I": (0, 0), "X": (1, 0), "Y": (1, 1), "Z": (0, 1)}

U8 = NDArray[np.uint8]


# ------------------------------------------------------------------------------- Clifford maps


@dataclass(frozen=True)
class CliffordMap:
    """How a k-qubit Clifford conjugates each local Pauli.

    Index i = the Pauli's bits (x0, z0, x1, z1, ...) read as a binary number, first qubit most
    significant. ``bits[i]`` are the image's bits in the same order and ``sign[i]`` is 1 when the
    image carries a minus sign.
    """

    k: int
    bits: NDArray[np.uint8]  # (4**k, 2k)
    sign: NDArray[np.uint8]  # (4**k,)


def _pauli_string(bits: Sequence[int]) -> NDArray[np.complex128]:
    m = np.array([[1]], dtype=np.complex128)
    for q in range(len(bits) // 2):
        m = np.kron(m, _PAULI_1Q[(bits[2 * q], bits[2 * q + 1])]).astype(np.complex128)
    return m


@lru_cache(maxsize=4)
def _pauli_basis(k: int) -> tuple[NDArray[np.uint8], NDArray[np.complex128]]:
    labels = np.array(list(itertools.product((0, 1), repeat=2 * k)), dtype=np.uint8)
    mats = np.stack([_pauli_string(row) for row in labels])
    return labels, mats


@lru_cache(maxsize=1024)
def _clifford_map_cached(key: bytes, k: int) -> CliffordMap | None:
    u = np.frombuffer(key, dtype=np.complex128).reshape(2**k, 2**k)
    labels, mats = _pauli_basis(k)
    d = 2**k
    # coefficients of U P U† in the Pauli basis: c[p, q] = tr(Q† U P U†) / d
    images = np.einsum("ij,pjk,lk->pil", u, mats, u.conj())
    coeffs = np.einsum("qji,pjk->pqik", mats.conj(), images).trace(axis1=2, axis2=3) / d
    bits = np.zeros((len(labels), 2 * k), dtype=np.uint8)
    sign = np.zeros(len(labels), dtype=np.uint8)
    for p in range(len(labels)):
        q = int(np.argmax(np.abs(coeffs[p])))
        c = coeffs[p, q]
        if abs(abs(c) - 1) > 1e-8 or abs(c.imag) > 1e-8:
            return None
        bits[p] = labels[q]
        sign[p] = 1 if c.real < 0 else 0
    return CliffordMap(k, bits, sign)


_RAW: dict[bytes, CliffordMap | None] = {}


def clifford_map(matrix: NDArray[np.complex128]) -> CliffordMap | None:
    """The Pauli map of a gate matrix, or None when the gate isn't Clifford."""
    raw = matrix.tobytes()
    if raw in _RAW:
        return _RAW[raw]
    d = matrix.shape[0]
    k = d.bit_length() - 1
    found = None
    if 0 < k <= MAX_CLIFFORD_GATE_QUBITS:
        rounded = np.round(np.ascontiguousarray(matrix, dtype=np.complex128), 12) + 0.0
        found = _clifford_map_cached(rounded.tobytes(), k)
    if len(_RAW) < 4096:
        _RAW[raw] = found
    return found


def _kind(cmap: CliffordMap) -> str | None:
    """The standard gate a map is, for the fast paths (None: use the generic table)."""
    return _KINDS.get((cmap.k, cmap.bits.tobytes(), cmap.sign.tobytes()))


def _standard_kinds() -> dict[tuple[int, bytes, bytes], str]:
    s2 = 1 / np.sqrt(2)
    one = {
        "h": np.array([[s2, s2], [s2, -s2]]),
        "s": np.diag([1, 1j]),
        "sdg": np.diag([1, -1j]),
        "x": np.array([[0, 1], [1, 0]]),
        "y": np.array([[0, -1j], [1j, 0]]),
        "z": np.diag([1, -1]),
        "id": np.eye(2),
    }
    cx = np.eye(4)[[0, 1, 3, 2]]  # first qubit (most significant) controls
    two = {"cx": cx, "cz": np.diag([1, 1, 1, -1])}
    kinds = {}
    for name, m in {**one, **two}.items():
        k = m.shape[0].bit_length() - 1
        rounded = np.round(np.asarray(m, dtype=np.complex128), 12) + 0.0
        cmap = _clifford_map_cached(rounded.tobytes(), k)
        assert cmap is not None
        kinds[(k, cmap.bits.tobytes(), cmap.sign.tobytes())] = name
    return kinds


def _pauli_label(matrix: NDArray[np.complex128]) -> list[tuple[int, int]] | None:
    """A Pauli string's (x, z) bits per qubit (first qubit first), or None if it isn't one."""
    k = matrix.shape[0].bit_length() - 1
    labels, mats = _pauli_basis(k)
    coeffs = np.einsum("qji,ij->q", mats.conj(), matrix) / 2**k
    q = int(np.argmax(np.abs(coeffs)))
    if abs(abs(coeffs[q]) - 1) > 1e-8:
        return None
    return [(int(labels[q][2 * i]), int(labels[q][2 * i + 1])) for i in range(k)]


@dataclass(frozen=True)
class PauliNoise:
    """A gate-noise channel as Pauli branches: cumulative probabilities and per-branch bits."""

    cumulative: NDArray[np.float64]
    branches: tuple[tuple[tuple[int, int], ...], ...]


@lru_cache(maxsize=256)
def _pauli_noise_cached(channel: Channel) -> PauliNoise | None:
    if channel.mixture is None:
        return None
    probs, branches = [], []
    for p, m in channel.mixture:
        label = _pauli_label(m)
        if label is None:
            return None
        probs.append(p)
        branches.append(tuple(label))
    return PauliNoise(np.cumsum(probs) / sum(probs), tuple(branches))


def pauli_noise(channel: Channel) -> PauliNoise | None:
    return _pauli_noise_cached(channel)


_KINDS = _standard_kinds()


# ------------------------------------------------------------------------------------ tableau


class Tableau:
    """Rows 0..n-1 are destabilizers, n..2n-1 stabilizers; ``x[q, row]`` and ``z[q, row]`` hold
    each row's Pauli on qubit q (x=z=1 is Y) and ``r[row]`` its sign. Starts in |0...0>."""

    def __init__(self, num_qubits: int) -> None:
        if num_qubits > MAX_STABILIZER_QUBITS:
            raise SimulationError(
                f"{num_qubits} qubits are more than the stabilizer engine allows (up to {MAX_STABILIZER_QUBITS})"
            )
        n = self.n = num_qubits
        self.x: U8 = np.zeros((n, 2 * n), dtype=np.uint8)
        self.z: U8 = np.zeros((n, 2 * n), dtype=np.uint8)
        self.r: U8 = np.zeros(2 * n, dtype=np.uint8)
        idx = np.arange(n)
        self.x[idx, idx] = 1  # destabilizer i = X_i
        self.z[idx, idx + n] = 1  # stabilizer i = Z_i

    def copy(self) -> Tableau:
        t = Tableau.__new__(Tableau)
        t.n, t.x, t.z, t.r = self.n, self.x.copy(), self.z.copy(), self.r.copy()
        return t

    # -- gates ------------------------------------------------------------------------------

    def apply(self, cmap: CliffordMap, qubits: Sequence[int]) -> None:
        kind = _kind(cmap)
        if kind is not None:
            self._fast(kind, qubits)
            return
        self.apply_generic(cmap, qubits)

    def _fast(self, kind: str, qubits: Sequence[int]) -> None:
        """Aaronson-Gottesman's update rules for the common gates (checked against the generic
        table in the tests)."""
        x, z, r = self.x, self.z, self.r
        a = qubits[0]
        if kind == "h":
            r ^= x[a] & z[a]
            x[a], z[a] = z[a].copy(), x[a].copy()
        elif kind == "s":
            r ^= x[a] & z[a]
            z[a] ^= x[a]
        elif kind == "sdg":
            r ^= x[a] & (z[a] ^ 1)
            z[a] ^= x[a]
        elif kind == "x":
            r ^= z[a]
        elif kind == "z":
            r ^= x[a]
        elif kind == "y":
            r ^= x[a] ^ z[a]
        elif kind == "cx":
            b = qubits[1]
            r ^= x[a] & z[b] & (x[b] ^ z[a] ^ 1)
            x[b] ^= x[a]
            z[a] ^= z[b]
        elif kind == "cz":
            b = qubits[1]
            r ^= x[a] & x[b] & (z[a] ^ z[b])
            z[a] ^= x[b]
            z[b] ^= x[a]
        # "id": nothing

    def apply_generic(self, cmap: CliffordMap, qubits: Sequence[int]) -> None:
        index = np.zeros(2 * self.n, dtype=np.int64)
        for q in qubits:  # first qubit most significant, (x, z) per qubit
            index = (index << 2) | (self.x[q].astype(np.int64) << 1) | self.z[q]
        new = cmap.bits[index]  # (2n, 2k)
        self.r ^= cmap.sign[index]
        for i, q in enumerate(qubits):
            self.x[q] = new[:, 2 * i]
            self.z[q] = new[:, 2 * i + 1]

    def pauli(self, label: Sequence[tuple[int, int]], qubits: Sequence[int]) -> None:
        """Multiply the state by a Pauli: flips the sign of every row it anticommutes with."""
        for (px, pz), q in zip(label, qubits, strict=True):
            if px:
                self.r ^= self.z[q]
            if pz:
                self.r ^= self.x[q]

    # -- measurement --------------------------------------------------------------------------

    def _rowsum_into(self, targets: NDArray[np.int64], source: int) -> None:
        """Row[t] := Row[source] · Row[t] for every t in targets, with the sign tracked."""
        if len(targets) == 0:
            return
        xs, zs = self.x[:, source].astype(np.int8)[:, None], self.z[:, source].astype(np.int8)[:, None]
        xt, zt = self.x[:, targets].astype(np.int8), self.z[:, targets].astype(np.int8)
        g = (xs & zs) * (zt - xt) + (xs & (1 - zs)) * (zt * (2 * xt - 1)) + ((1 - xs) & zs) * (xt * (1 - 2 * zt))
        total = 2 * self.r[targets].astype(np.int64) + 2 * int(self.r[source]) + g.sum(axis=0, dtype=np.int64)
        self.r[targets] = (np.mod(total, 4) == 2).astype(np.uint8)
        self.x[:, targets] ^= self.x[:, source][:, None]
        self.z[:, targets] ^= self.z[:, source][:, None]

    def _deterministic(self, a: int) -> int:
        n = self.n
        sx = np.zeros(n, dtype=np.int8)
        sz = np.zeros(n, dtype=np.int8)
        sr = 0
        for i in np.flatnonzero(self.x[a, :n]):
            row = int(i) + n
            xs, zs = self.x[:, row].astype(np.int8), self.z[:, row].astype(np.int8)
            g = (xs & zs) * (sz - sx) + (xs & (1 - zs)) * (sz * (2 * sx - 1)) + ((1 - xs) & zs) * (sx * (1 - 2 * sz))
            total = 2 * sr + 2 * int(self.r[row]) + int(g.sum())
            sr = 1 if total % 4 == 2 else 0
            sx ^= xs
            sz ^= zs
        return sr

    def measure(self, a: int, rng: np.random.Generator | None = None, forced: int | None = None) -> int:
        """Measure qubit a in the computational basis. A random outcome comes from ``rng``, or is
        ``forced`` (used to find one possible outcome when sampling)."""
        n = self.n
        candidates = np.flatnonzero(self.x[a, n:])
        if len(candidates) == 0:
            return self._deterministic(a)
        p = int(candidates[0]) + n
        others = np.flatnonzero(self.x[a])
        others = others[others != p]
        self._rowsum_into(others, p)
        self.x[:, p - n] = self.x[:, p]
        self.z[:, p - n] = self.z[:, p]
        self.r[p - n] = self.r[p]
        self.x[:, p] = 0
        self.z[:, p] = 0
        self.z[a, p] = 1
        if forced is not None:
            outcome = forced
        else:
            if rng is None:  # pragma: no cover - callers always pass one
                raise ValueError("a random measurement needs a generator")
            outcome = int(rng.integers(2))
        self.r[p] = outcome
        return outcome

    def reset(self, a: int, rng: np.random.Generator) -> None:
        if self.measure(a, rng):
            self.pauli([(1, 0)], [a])

    def support(self, qubits: Sequence[int]) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        """(x0, basis): outcomes on ``qubits`` are uniform over x0 + span(basis rows).

        Gaussian elimination on the stabilizers, bit-packed: rows with an X part span V; the rows
        left with only Z parts are ±Z constraints (z·x = r), and x0 solves them.
        """
        n = self.n
        sx = _pack_rows(self.x[:, n:].T)
        sz = _pack_rows(self.z[:, n:].T)
        sr = self.r[n:].astype(np.uint8).copy()
        pivots: list[int] = []
        used = np.zeros(n, dtype=bool)
        for q in range(n):
            word, bit = q // 64, np.uint64(1) << np.uint64(q % 64)
            has = (sx[:, word] & bit) != 0
            candidates = np.flatnonzero(has & ~used)
            if len(candidates) == 0:
                continue
            p = int(candidates[0])
            used[p] = True
            pivots.append(p)
            targets = np.flatnonzero(has)
            targets = targets[targets != p]
            if len(targets):
                _rowsum_packed(sx, sz, sr, targets, p)
        z_rows = np.flatnonzero(~used)
        x_full = _solve_z_constraints(sz[z_rows], sr[z_rows], n)
        index = np.asarray(qubits, dtype=np.int64)
        x0 = x_full[index]
        vectors = _unpack_rows(sx[pivots], n)[:, index] if pivots else np.zeros((0, len(index)), dtype=np.uint8)
        if len(index) == n and np.array_equal(index, np.arange(n)):
            return x0, vectors  # pivot rows are already independent
        return x0, _row_basis(vectors)


def _pack_rows(bits: NDArray[np.uint8]) -> NDArray[np.uint64]:
    """Rows of bits (little-endian within each row) as uint64 words."""
    packed = np.packbits(np.ascontiguousarray(bits, dtype=np.uint8), axis=1, bitorder="little")
    pad = (-packed.shape[1]) % 8
    if pad:
        packed = np.pad(packed, ((0, 0), (0, pad)))
    return np.ascontiguousarray(packed).view(np.uint64)


def _unpack_rows(words: NDArray[np.uint64], n: int) -> NDArray[np.uint8]:
    return np.unpackbits(np.ascontiguousarray(words).view(np.uint8), axis=1, bitorder="little")[:, :n]


def _rowsum_packed(
    sx: NDArray[np.uint64], sz: NDArray[np.uint64], sr: NDArray[np.uint8], targets: NDArray[np.int64], p: int
) -> None:
    """Row[t] := Row[p] · Row[t] for packed rows, with the sign from popcounts (Aaronson-Gottesman's
    g summed over qubits: +1 and -1 cases as bit masks)."""
    x1, z1 = sx[p], sz[p]
    x2, z2 = sx[targets], sz[targets]
    n1x, n1z, n2x, n2z = ~x1, ~z1, ~x2, ~z2
    plus = (x1 & z1 & n2x & z2) | (x1 & n1z & x2 & z2) | (n1x & z1 & x2 & n2z)
    minus = (x1 & z1 & x2 & n2z) | (x1 & n1z & n2x & z2) | (n1x & z1 & x2 & z2)
    g = np.bitwise_count(plus).sum(axis=1, dtype=np.int64) - np.bitwise_count(minus).sum(axis=1, dtype=np.int64)
    total = 2 * sr[targets].astype(np.int64) + 2 * int(sr[p]) + g
    sr[targets] = (np.mod(total, 4) == 2).astype(np.uint8)
    sx[targets] ^= x1
    sz[targets] ^= z1


def _solve_z_constraints(rows: NDArray[np.uint64], rhs: NDArray[np.uint8], n: int) -> NDArray[np.uint8]:
    """One x with rows·x = rhs (mod 2); the stabilizer state guarantees a solution."""
    m = rows.copy()
    b = rhs.astype(np.uint8).copy()
    x = np.zeros(n, dtype=np.uint8)
    pivot_of: list[tuple[int, int]] = []
    used = np.zeros(len(m), dtype=bool)
    for q in range(n):
        word, bit = q // 64, np.uint64(1) << np.uint64(q % 64)
        has = (m[:, word] & bit) != 0
        candidates = np.flatnonzero(has & ~used)
        if len(candidates) == 0:
            continue
        p = int(candidates[0])
        used[p] = True
        pivot_of.append((q, p))
        others = np.flatnonzero(has)
        others = others[others != p]
        m[others] ^= m[p]
        b[others] ^= b[p]
    for q, p in pivot_of:  # fully reduced: with free variables 0, each pivot variable is its row's rhs
        x[q] = b[p]
    if np.any(b[~used]):  # pragma: no cover - a valid tableau is always consistent
        raise SimulationError("inconsistent stabilizer constraints")
    return x


def _row_basis(rows: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """A basis of the GF(2) row space (Gaussian elimination)."""
    m = rows.copy()
    basis = []
    for col in range(m.shape[1]):
        pivots = np.flatnonzero(m[:, col])
        if len(pivots) == 0:
            continue
        pivot = int(pivots[0])
        row = m[pivot].copy()
        basis.append(row)
        mask = m[:, col].astype(bool)
        m[mask] ^= row
        if len(basis) == m.shape[1]:
            break
    return np.array(basis, dtype=np.uint8).reshape(len(basis), rows.shape[1])


# ------------------------------------------------------------------------------- capability


def _gates(ops: Sequence[Operation]) -> list[Gate]:
    found: list[Gate] = []
    for op in ops:
        if isinstance(op, Gate):
            found.append(op)
        elif isinstance(op, IfElse):
            found.extend(_gates(op.then))
            found.extend(_gates(op.otherwise))
    return found


def clifford_problem(circuit: Circuit, noise: NoiseModel | None = None) -> str | None:
    """Why the stabilizer engine can't run this circuit exactly, or None if it can."""
    if circuit.num_qubits > MAX_STABILIZER_QUBITS:
        return f"it has {circuit.num_qubits} qubits; the stabilizer engine allows up to {MAX_STABILIZER_QUBITS}"
    for gate in _gates(circuit.operations):
        if clifford_map(gate.matrix) is None:
            return f"{gate.name} on qubits {list(gate.qubits)} isn't a Clifford gate"
        channel = noise.gate_error(gate.name, gate.qubits) if noise else None
        if channel is not None and pauli_noise(channel) is None:
            return f"the {channel.name} noise after {gate.name} isn't a Pauli channel"
    return None


def is_clifford(circuit: Circuit) -> bool:
    return clifford_problem(circuit) is None


# ------------------------------------------------------------------------------------ running


def _apply_gate(state: Tableau, gate: Gate, noise: NoiseModel | None, rng: np.random.Generator | None) -> None:
    cmap = clifford_map(gate.matrix)
    if cmap is None:  # pragma: no cover - clifford_problem refuses it first
        raise SimulationError(f"{gate.name} isn't a Clifford gate")
    state.apply(cmap, gate.qubits)
    channel = noise.gate_error(gate.name, gate.qubits) if noise else None
    if channel is not None:
        branches = pauli_noise(channel)
        if branches is None or rng is None:  # pragma: no cover - refused earlier
            raise SimulationError(f"the {channel.name} noise isn't a Pauli channel")
        pick = int(np.searchsorted(branches.cumulative, rng.random(), side="right"))
        state.pauli(branches.branches[min(pick, len(branches.branches) - 1)], gate.qubits)


def _has_gate_noise(circuit: Circuit, noise: NoiseModel | None) -> bool:
    return noise is not None and any(noise.gate_error(g.name, g.qubits) is not None for g in _gates(circuit.operations))


def run_counts(circuit: Circuit, shots: int, noise: NoiseModel | None, rng: np.random.Generator) -> dict[int, int]:
    """Counts keyed by the classical bits as an integer (bit c = classical bit c)."""
    problem = clifford_problem(circuit, noise)
    if problem:
        raise SimulationError(f"the stabilizer method runs Clifford circuits only: {problem}")
    if shots == 0 or circuit.num_clbits == 0:
        # still simulate, so a too-large or non-Clifford circuit fails the same way with no shots
        state = Tableau(circuit.num_qubits)
        if not circuit.is_dynamic() and not _has_gate_noise(circuit, noise):
            for g in _gates(circuit.operations):
                state.apply(clifford_map(g.matrix), g.qubits)  # type: ignore[arg-type]
        return {}
    if circuit.is_dynamic():
        return _per_shot(circuit, shots, noise, rng)
    if _has_gate_noise(circuit, noise):
        return _frames(circuit, shots, noise, rng)
    return _sampled(circuit, shots, noise, rng)


def _frames(circuit: Circuit, shots: int, noise: NoiseModel | None, rng: np.random.Generator) -> dict[int, int]:
    """A static circuit with Pauli gate noise, all shots at once (Pauli frames; QF-002 Stage C).

    One noiseless simulation gives a reference outcome x0. Each shot then carries a Pauli frame
    through the circuit: it starts as a random Z string (harmless on |0...0>, and once conjugated
    through the circuit a uniformly random stabilizer, which spreads the shots uniformly over the
    ideal outcomes), noise Paulis are multiplied in where they occur, and a shot's outcome is x0
    with the bits flipped where its frame has an X part. Exact, like the per-shot tableau, and
    vectorised over shots.
    """
    n = circuit.num_qubits
    reference = Tableau(n)
    for op in circuit.operations:
        if isinstance(op, Gate):
            reference.apply(clifford_map(op.matrix), op.qubits)  # type: ignore[arg-type]
    pairs = circuit.final_measurements()
    if not pairs:
        return {}
    measured = sorted({q for q, _ in pairs})
    x0, _ = reference.support(measured)
    fx = np.zeros((n, shots), dtype=np.uint8)
    fz = rng.integers(0, 2, size=(n, shots), dtype=np.uint8)
    for op in circuit.operations:
        if not isinstance(op, Gate):
            continue  # measurements are final; resets touch untouched qubits only; phases don't matter
        cmap = clifford_map(op.matrix)
        assert cmap is not None  # clifford_problem refused anything else
        _frame_gate(fx, fz, cmap, op.qubits)
        channel = noise.gate_error(op.name, op.qubits) if noise else None
        if channel is None:
            continue
        branches = pauli_noise(channel)
        assert branches is not None
        picks = np.minimum(
            np.searchsorted(branches.cumulative, rng.random(shots), side="right"), len(branches.branches) - 1
        )
        for b, label in enumerate(branches.branches):
            if not any(px or pz for px, pz in label):
                continue
            mask = picks == b
            if not mask.any():
                continue
            for (px, pz), q in zip(label, op.qubits, strict=True):
                if px:
                    fx[q, mask] ^= 1
                if pz:
                    fz[q, mask] ^= 1
    position = {q: i for i, q in enumerate(measured)}
    measured_bits = np.stack([fx[q] ^ x0[position[q]] for q in measured], axis=1)
    return _count(circuit, pairs, position, measured_bits, noise, rng, shots)


def _frame_gate(fx: U8, fz: U8, cmap: CliffordMap, qubits: Sequence[int]) -> None:
    """Conjugate every shot's frame by a Clifford (signs don't matter for a frame)."""
    kind = _kind(cmap)
    a = qubits[0]
    if kind in ("x", "y", "z", "id"):
        return
    if kind == "h":
        fx[a], fz[a] = fz[a].copy(), fx[a].copy()
        return
    if kind in ("s", "sdg"):
        fz[a] ^= fx[a]
        return
    if kind == "cx":
        b = qubits[1]
        fx[b] ^= fx[a]
        fz[a] ^= fz[b]
        return
    if kind == "cz":
        b = qubits[1]
        fz[a] ^= fx[b]
        fz[b] ^= fx[a]
        return
    index = np.zeros(fx.shape[1], dtype=np.int64)
    for q in qubits:
        index = (index << 2) | (fx[q].astype(np.int64) << 1) | fz[q]
    new = cmap.bits[index]
    for i, q in enumerate(qubits):
        fx[q] = new[:, 2 * i]
        fz[q] = new[:, 2 * i + 1]


def _sampled(circuit: Circuit, shots: int, noise: NoiseModel | None, rng: np.random.Generator) -> dict[int, int]:
    state = Tableau(circuit.num_qubits)
    for op in circuit.operations:
        if isinstance(op, Gate):
            state.apply(clifford_map(op.matrix), op.qubits)  # type: ignore[arg-type]
        # GlobalPhase: not tracked; Reset of an untouched qubit: nothing; Measure: below
    pairs = circuit.final_measurements()
    if not pairs:
        return {}
    measured = sorted({q for q, _ in pairs})
    position = {q: i for i, q in enumerate(measured)}
    x0, basis = state.support(measured)
    m = len(measured)
    words = (m + 63) // 64
    packed_x0 = _pack(x0, words)
    outcomes = np.repeat(packed_x0[None, :], shots, axis=0)
    if len(basis):
        coefficients = rng.integers(0, 2, size=(shots, len(basis)), dtype=np.uint8).astype(bool)
        packed_basis = np.stack([_pack(row, words) for row in basis])
        for i in range(len(basis)):
            outcomes[coefficients[:, i]] ^= packed_basis[i]
    measured_bits = np.unpackbits(outcomes.view(np.uint8), axis=1, bitorder="little")[:, :m]
    return _count(circuit, pairs, position, measured_bits, noise, rng, shots)


def _count(
    circuit: Circuit,
    pairs: list[tuple[int, int]],
    position: dict[int, int],
    measured_bits: U8,
    noise: NoiseModel | None,
    rng: np.random.Generator,
    shots: int,
) -> dict[int, int]:
    """Outcome bits per shot (shots by measured qubits) placed on their classical bits, with
    readout errors, counted."""
    clbits = np.zeros((shots, circuit.num_clbits), dtype=np.uint8)
    for q, c in pairs:
        value = measured_bits[:, position[q]]
        err = noise.readout_error(q) if noise else None
        if err is not None and (err.p01 or err.p10):
            flip = rng.random(shots) < np.where(value == 1, err.p10, err.p01)
            value = value ^ flip.astype(np.uint8)
        clbits[:, c] = value
    packed = np.packbits(clbits, axis=1, bitorder="little")
    rows, counts = np.unique(packed, axis=0, return_counts=True)
    return {int.from_bytes(row.tobytes(), "little"): int(k) for row, k in zip(rows, counts, strict=True)}


def _pack(bits: NDArray[np.uint8], words: int) -> NDArray[np.uint64]:
    out = np.zeros(words, dtype=np.uint64)
    for i in np.flatnonzero(bits):
        out[i // 64] |= np.uint64(1) << np.uint64(i % 64)
    return out


def _per_shot(circuit: Circuit, shots: int, noise: NoiseModel | None, rng: np.random.Generator) -> dict[int, int]:
    """One tableau per shot. The noiseless gate prefix is simulated once and copied."""
    ops = circuit.operations
    prefix = Tableau(circuit.num_qubits)
    start = 0
    for op in ops:
        if isinstance(op, Gate) and (noise is None or noise.gate_error(op.name, op.qubits) is None):
            prefix.apply(clifford_map(op.matrix), op.qubits)  # type: ignore[arg-type]
            start += 1
        elif isinstance(op, (Barrier, GlobalPhase)):
            start += 1
        else:
            break
    rest = ops[start:]
    tally: dict[int, int] = {}
    for _ in range(shots):
        state = prefix.copy()
        clbits = [0] * circuit.num_clbits
        _run_ops(state, rest, clbits, noise, rng)
        key = sum(b << i for i, b in enumerate(clbits))
        tally[key] = tally.get(key, 0) + 1
    return tally


def _run_ops(
    state: Tableau, ops: Sequence[Operation], clbits: list[int], noise: NoiseModel | None, rng: np.random.Generator
) -> None:
    for op in ops:
        if isinstance(op, Gate):
            _apply_gate(state, op, noise, rng)
        elif isinstance(op, Measure):
            outcome = state.measure(op.qubit, rng)
            err = noise.readout_error(op.qubit) if noise else None
            if err is not None and rng.random() < (err.p10 if outcome else err.p01):
                outcome ^= 1
            if op.clbit is not None:
                clbits[op.clbit] = outcome
        elif isinstance(op, Reset):
            state.reset(op.qubit, rng)
        elif isinstance(op, IfElse):
            _run_ops(state, op.then if op.condition.holds(clbits) else op.otherwise, clbits, noise, rng)
        # Barrier, GlobalPhase: nothing to do


__all__ = [
    "MAX_STABILIZER_QUBITS",
    "CliffordMap",
    "Tableau",
    "clifford_map",
    "clifford_problem",
    "is_clifford",
    "run_counts",
]
