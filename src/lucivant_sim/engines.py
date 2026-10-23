"""The three engines.

* :class:`Statevector`: the exact state, 2**n amplitudes. Ideal circuits, sampled once per run.
* :class:`DensityMatrix`: the exact mixed state, 4**n entries. Noisy circuits, sampled once.
* Trajectories (in :mod:`lucivant_sim.simulator`): a statevector per shot, with noise branches,
  mid-circuit measurements and conditions chosen at random. Anything at all, one shot at a time.

Qubit q sits on tensor axis n-1-q, so a flattened state is indexed little-endian (qubit 0 is the
least significant bit), the convention OpenQASM tools print results in.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from lucivant_sim.gates import Matrix
from lucivant_sim.noise import Channel

#: The slice kernel is used from these state sizes (entries): diagonal and permutation gates
#: (rz, s, t, cz, x, cx, swap) from 2**14, general matrices from 2**18; smaller states use the
#: reference contraction (QF-002 Phase 3 Stage C, benchmarks/qf002_phase3/kernel_ab.py).
SLICE_MIN_MOVES = 2**14
SLICE_MIN_GENERAL = 2**18
#: General one-qubit gates use one batched matrix product when the block below the qubit's axis has
#: at least this many entries (measured: faster from 8 up; slower for the last three qubit axes).
MATMUL_MIN_INNER = 8

#: Memory guards: 2**26 amplitudes and 4**13 matrix entries are 1 GiB each.
MAX_STATEVECTOR_QUBITS = 26
MAX_DENSITY_QUBITS = 13


class SimulationError(RuntimeError):
    """The circuit can't be simulated as asked (too large, or the method doesn't fit it)."""


def _apply(tensor: NDArray[np.complex128], matrix: Matrix, axes: list[int]) -> NDArray[np.complex128]:
    """Reference contraction, retained for differential verification and wider gates."""
    k = len(axes)
    out = np.tensordot(matrix.reshape([2] * (2 * k)), tensor, axes=(list(range(k, 2 * k)), axes))
    return np.moveaxis(out, list(range(k)), axes)


def _apply_slices(
    tensor: NDArray[np.complex128], matrix: Matrix, axes: list[int], inplace: bool = True
) -> NDArray[np.complex128]:
    """One- and two-qubit gates by slices of the state, without transposing it (QF-002 Phase 3,
    Stage C). The tensor is updated in place when ``inplace`` (else a copy is), and returned.

    For a gate on axes (a1, a2), the state splits into 2**k slices by those axes' values (the first
    axis is the matrix's most significant bit); each new slice is a combination of the old ones.
    Diagonal gates (rz, s, t, cz, cp) only scale slices, and permutations with phases (x, y, cx,
    swap) only move them. Wider gates use the reference contraction :func:`_apply`.
    """
    k = len(axes)
    # cheapest checks first: small states stay exactly as they were
    if k == 0 or k > 2 or tensor.size < SLICE_MIN_MOVES:
        return _apply(tensor, matrix, axes)
    m = np.asarray(matrix, dtype=np.complex128)
    nonzero = m != 0
    moves_only = bool((nonzero.sum(axis=1) == 1).all() and (nonzero.sum(axis=0) == 1).all())
    if k == 1 and not moves_only:
        # A general one-qubit gate as one batched 2x2 product over (outer, 2, inner) blocks: one pass
        # over the state (Orrery 0.3, benchmarks/orrery_kernels). It loses to slices when the
        # inner block is tiny (the qubits nearest the fast axis), so those keep the slice path.
        inner = 2 ** (tensor.ndim - 1 - axes[0])
        if inner >= MATMUL_MIN_INNER:
            outer = tensor.size // (2 * inner)
            product: NDArray[np.complex128] = np.matmul(m, tensor.reshape(outer, 2, inner))
            return product.reshape(tensor.shape)
    # measured (benchmarks/qf002_phase3/kernel_ab.py): slices win on large states and lose a little
    # to Python overhead on small ones, more so for general matrices
    if tensor.size < (SLICE_MIN_MOVES if moves_only else SLICE_MIN_GENERAL):
        return _apply(tensor, matrix, axes)
    if inplace:
        t = tensor if tensor.flags.c_contiguous else np.ascontiguousarray(tensor)
    else:
        t = np.array(tensor, dtype=np.complex128, order="C", copy=True)
    nd = t.ndim
    d = 2**k

    def where(r: int) -> tuple[slice, ...]:
        # length-1 slices, not integers, so the result is always a writable view (even for one qubit)
        index: list[slice] = [slice(None)] * nd
        for i, axis in enumerate(axes):
            b = (r >> (k - 1 - i)) & 1
            index[axis] = slice(b, b + 1)
        return tuple(index)

    if moves_only:
        src = nonzero.argmax(axis=1)
        saved = {int(c): t[where(int(c))].copy() for r, c in enumerate(src) if c != r}
        for r in range(d):
            c = int(src[r])
            phase = m[r, c]
            if c == r:
                if phase != 1:
                    t[where(r)] *= phase
            elif phase == 1:
                t[where(r)] = saved[c]
            else:
                np.multiply(saved[c], phase, out=t[where(r)])
        return t
    old = [t[where(c)].copy() for c in range(d)]
    for r in range(d):
        out = t[where(r)]
        first = True
        for c in range(d):
            if m[r, c] == 0:
                continue
            if first:
                np.multiply(old[c], m[r, c], out=out)
                first = False
            else:
                out += m[r, c] * old[c]
        if first:  # an all-zero row (a projector): the slice is zero
            out[...] = 0
    return t


def _depolarizing_trace(
    tensor: NDArray[np.complex128], p: float, rows: list[int], cols: list[int]
) -> NDArray[np.complex128]:
    """E(rho)=(1-p)rho + p I/d tensor Tr_targets(rho), without enumerating Pauli Kraus terms."""
    k = len(rows)
    d = 2**k
    axes = rows + cols
    moved = np.moveaxis(tensor, axes, range(2 * k))
    matrix = moved.reshape((d, d, *moved.shape[2 * k :]))
    partial = np.trace(matrix, axis1=0, axis2=1)
    result: NDArray[np.complex128] = np.multiply(matrix, 1 - p, dtype=np.complex128)
    for i in range(d):
        result[i, i] += (p / d) * partial
    return np.moveaxis(result.reshape(moved.shape), range(2 * k), axes)


class Statevector:
    def __init__(self, num_qubits: int) -> None:
        if num_qubits > MAX_STATEVECTOR_QUBITS:
            raise SimulationError(
                f"{num_qubits} qubits need more memory than the statevector engine allows "
                f"(up to {MAX_STATEVECTOR_QUBITS})"
            )
        self.n = num_qubits
        state = np.zeros(2**num_qubits, dtype=np.complex128)
        state[0] = 1
        self.tensor = state.reshape([2] * num_qubits) if num_qubits else state

    def axes(self, qubits: tuple[int, ...]) -> list[int]:
        return [self.n - 1 - q for q in qubits]

    def apply(self, matrix: Matrix, qubits: tuple[int, ...]) -> None:
        self.tensor = _apply_slices(self.tensor, matrix, self.axes(qubits))

    def scale(self, factor: complex) -> None:
        self.tensor = self.tensor * factor

    def apply_channel(self, channel: Channel, qubits: tuple[int, ...], rng: np.random.Generator) -> None:
        """Pick one Kraus branch with its Born probability (a quantum trajectory)."""
        r = rng.random()
        if channel.mixture is not None:
            acc = 0.0
            for p, unitary in channel.mixture:
                acc += p
                if r < acc:
                    if not np.allclose(unitary, np.eye(unitary.shape[0])):
                        self.apply(unitary, qubits)
                    return
            return
        axes = self.axes(qubits)
        acc = 0.0
        candidate = self.tensor
        for kraus in channel.kraus:
            candidate = _apply_slices(self.tensor, kraus, axes, inplace=False)
            weight = float(np.vdot(candidate, candidate).real)
            acc += weight
            if r < acc and weight > 0:
                self.tensor = candidate / np.sqrt(weight)
                return
        norm = float(np.vdot(candidate, candidate).real)
        if norm > 0:
            self.tensor = candidate / np.sqrt(norm)

    def probability_of_one(self, qubit: int) -> float:
        one = np.take(self.tensor, 1, axis=self.n - 1 - qubit)
        return float(np.vdot(one, one).real)

    def measure(self, qubit: int, rng: np.random.Generator) -> int:
        p1 = min(max(self.probability_of_one(qubit), 0.0), 1.0)
        outcome = int(rng.random() < p1)
        self.collapse(qubit, outcome, p1 if outcome else 1 - p1)
        return outcome

    def collapse(self, qubit: int, outcome: int, probability: float) -> None:
        axis = self.n - 1 - qubit
        index: list[slice | int] = [slice(None)] * self.n
        index[axis] = 1 - outcome
        self.tensor[tuple(index)] = 0
        if probability > 0:
            self.tensor = self.tensor / np.sqrt(probability)

    def reset(self, qubit: int, rng: np.random.Generator) -> None:
        if self.measure(qubit, rng):
            self.apply(np.array([[0, 1], [1, 0]], dtype=np.complex128), (qubit,))

    def probabilities(self) -> NDArray[np.float64]:
        flat = self.tensor.reshape(-1)
        return np.abs(flat) ** 2

    def vector(self) -> NDArray[np.complex128]:
        return self.tensor.reshape(-1).copy()


class DensityMatrix:
    def __init__(self, num_qubits: int) -> None:
        if num_qubits > MAX_DENSITY_QUBITS:
            raise SimulationError(
                f"{num_qubits} qubits need more memory than the density-matrix engine allows "
                f"(up to {MAX_DENSITY_QUBITS}); use method='trajectories'"
            )
        self.n = num_qubits
        rho = np.zeros((2**num_qubits, 2**num_qubits), dtype=np.complex128)
        rho[0, 0] = 1
        self.tensor = rho.reshape([2] * (2 * num_qubits)) if num_qubits else rho

    def _rows(self, qubits: tuple[int, ...]) -> list[int]:
        return [self.n - 1 - q for q in qubits]

    def _cols(self, qubits: tuple[int, ...]) -> list[int]:
        return [2 * self.n - 1 - q for q in qubits]

    def apply(self, matrix: Matrix, qubits: tuple[int, ...]) -> None:
        t = _apply_slices(self.tensor, matrix, self._rows(qubits))
        self.tensor = _apply_slices(t, matrix.conj(), self._cols(qubits))

    def scale(self, factor: complex) -> None:
        """A global phase leaves a density matrix unchanged."""

    def apply_channel(self, channel: Channel, qubits: tuple[int, ...]) -> None:
        rows, cols = self._rows(qubits), self._cols(qubits)
        if channel.analytic_depolarizing and len(qubits) in (1, 2):
            self.tensor = _depolarizing_trace(self.tensor, channel.params[0], rows, cols)
            return
        total = None
        for kraus in channel.kraus:
            term = _apply_slices(_apply_slices(self.tensor, kraus, rows, inplace=False), kraus.conj(), cols)
            total = term if total is None else total + term
        if total is not None:
            self.tensor = total

    def reset(self, qubit: int) -> None:
        k0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
        k1 = np.array([[0, 1], [0, 0]], dtype=np.complex128)
        self.apply_channel(Channel("reset", 1, (k0, k1)), (qubit,))

    def probabilities(self) -> NDArray[np.float64]:
        d = 2**self.n
        diag = np.diagonal(self.tensor.reshape(d, d)).real
        return np.asarray(np.clip(diag, 0.0, None), dtype=np.float64)

    def matrix(self) -> NDArray[np.complex128]:
        d = 2**self.n
        return self.tensor.reshape(d, d).copy()


def marginal(probabilities: NDArray[np.float64], num_qubits: int, qubits: list[int]) -> NDArray[np.float64]:
    """Probabilities of ``qubits`` alone (sorted ascending); index bit i is qubits[i]."""
    keep = sorted(qubits)
    if num_qubits == 0:
        return probabilities
    tensor = probabilities.reshape([2] * num_qubits)
    drop = tuple(num_qubits - 1 - q for q in range(num_qubits) if q not in keep)
    reduced = tensor.sum(axis=drop) if drop else tensor
    return np.asarray(reduced).reshape(-1)
