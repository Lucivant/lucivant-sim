"""The standard gate library as unitary matrices.

Convention: a k-qubit matrix is written with its *first argument as the most significant bit*,
the textbook order, so ``cx a, b`` is ``[[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]`` with ``a`` the
control. The engines take care of placing arguments on the right axes; results still follow the
usual little-endian convention (qubit 0 is the rightmost bit of a bitstring).
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Matrix = NDArray[np.complex128]

I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2)
PAULIS: dict[str, Matrix] = {"I": I2, "X": X, "Y": Y, "Z": Z}


def _m(rows: list[list[complex]]) -> Matrix:
    return np.array(rows, dtype=np.complex128)


def rx(t: float) -> Matrix:
    c, s = math.cos(t / 2), math.sin(t / 2)
    return _m([[c, -1j * s], [-1j * s, c]])


def ry(t: float) -> Matrix:
    c, s = math.cos(t / 2), math.sin(t / 2)
    return _m([[c, -s], [s, c]])


def rz(t: float) -> Matrix:
    return _m([[cmath.exp(-0.5j * t), 0], [0, cmath.exp(0.5j * t)]])


def phase(t: float) -> Matrix:
    return _m([[1, 0], [0, cmath.exp(1j * t)]])


def u3(theta: float, phi: float, lam: float) -> Matrix:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return _m(
        [
            [c, -cmath.exp(1j * lam) * s],
            [cmath.exp(1j * phi) * s, cmath.exp(1j * (phi + lam)) * c],
        ]
    )


def controlled(u: Matrix, controls: int = 1) -> Matrix:
    """``u`` controlled on ``controls`` leading qubits (all must be 1)."""
    d = u.shape[0]
    total = d * 2**controls
    out = np.eye(total, dtype=np.complex128)
    out[total - d :, total - d :] = u
    return out


def negcontrolled(u: Matrix) -> Matrix:
    """``u`` applied when the leading qubit is 0."""
    d = u.shape[0]
    out = np.eye(2 * d, dtype=np.complex128)
    out[:d, :d] = u
    return out


def pauli_rotation(paulis: str, t: float) -> Matrix:
    """exp(-i t/2 P) for a Pauli string P, first letter on the first argument."""
    p = np.array([[1]], dtype=np.complex128)
    for letter in paulis:
        p = np.kron(p, PAULIS[letter]).astype(np.complex128)
    return math.cos(t / 2) * np.eye(p.shape[0], dtype=np.complex128) - 1j * math.sin(t / 2) * p


SX = _m([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2
SWAP: Matrix = _m([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
ISWAP: Matrix = _m([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]])
# Echoed cross-resonance, (XI - YX)/sqrt(2) with the first argument (the control) on the left.
ECR = ((np.kron(X, I2) - np.kron(Y, X)) / math.sqrt(2)).astype(np.complex128)
# Double CNOT: cx a,b then cx b,a.
DCX = _m([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0]])


@dataclass(frozen=True)
class GateDef:
    """A built-in gate: how many parameters and qubits, and its matrix."""

    params: int
    qubits: int
    matrix: Callable[..., Matrix]


def _fixed(m: Matrix) -> Callable[..., Matrix]:
    return lambda: m


GATES: dict[str, GateDef] = {
    # one qubit
    "id": GateDef(0, 1, _fixed(I2)),
    "i": GateDef(0, 1, _fixed(I2)),
    "x": GateDef(0, 1, _fixed(X)),
    "y": GateDef(0, 1, _fixed(Y)),
    "z": GateDef(0, 1, _fixed(Z)),
    "h": GateDef(0, 1, _fixed(H)),
    "s": GateDef(0, 1, _fixed(phase(math.pi / 2))),
    "sdg": GateDef(0, 1, _fixed(phase(-math.pi / 2))),
    "t": GateDef(0, 1, _fixed(phase(math.pi / 4))),
    "tdg": GateDef(0, 1, _fixed(phase(-math.pi / 4))),
    "sx": GateDef(0, 1, _fixed(SX)),
    "sxdg": GateDef(0, 1, _fixed(SX.conj().T)),
    "rx": GateDef(1, 1, rx),
    "ry": GateDef(1, 1, ry),
    "rz": GateDef(1, 1, rz),
    "p": GateDef(1, 1, phase),
    "phase": GateDef(1, 1, phase),
    "u1": GateDef(1, 1, phase),
    "u2": GateDef(2, 1, lambda phi, lam: u3(math.pi / 2, phi, lam)),
    "u3": GateDef(3, 1, u3),
    "u": GateDef(3, 1, u3),
    "U": GateDef(3, 1, u3),
    # two qubits
    "cx": GateDef(0, 2, _fixed(controlled(X))),
    "CX": GateDef(0, 2, _fixed(controlled(X))),
    "cnot": GateDef(0, 2, _fixed(controlled(X))),
    "cy": GateDef(0, 2, _fixed(controlled(Y))),
    "cz": GateDef(0, 2, _fixed(controlled(Z))),
    "ch": GateDef(0, 2, _fixed(controlled(H))),
    "csx": GateDef(0, 2, _fixed(controlled(SX))),
    "cp": GateDef(1, 2, lambda t: controlled(phase(t))),
    "cphase": GateDef(1, 2, lambda t: controlled(phase(t))),
    "cu1": GateDef(1, 2, lambda t: controlled(phase(t))),
    "crx": GateDef(1, 2, lambda t: controlled(rx(t))),
    "cry": GateDef(1, 2, lambda t: controlled(ry(t))),
    "crz": GateDef(1, 2, lambda t: controlled(rz(t))),
    "cu3": GateDef(3, 2, lambda a, b, c: controlled(u3(a, b, c))),
    "cu": GateDef(4, 2, lambda a, b, c, g: controlled(cmath.exp(1j * g) * u3(a, b, c))),
    "swap": GateDef(0, 2, _fixed(SWAP)),
    "iswap": GateDef(0, 2, _fixed(ISWAP)),
    "ecr": GateDef(0, 2, _fixed(ECR)),
    "dcx": GateDef(0, 2, _fixed(DCX)),
    "rxx": GateDef(1, 2, lambda t: pauli_rotation("XX", t)),
    "ryy": GateDef(1, 2, lambda t: pauli_rotation("YY", t)),
    "rzz": GateDef(1, 2, lambda t: pauli_rotation("ZZ", t)),
    "rzx": GateDef(1, 2, lambda t: pauli_rotation("ZX", t)),
    # three qubits
    "ccx": GateDef(0, 3, _fixed(controlled(X, 2))),
    "toffoli": GateDef(0, 3, _fixed(controlled(X, 2))),
    "ccz": GateDef(0, 3, _fixed(controlled(Z, 2))),
    "cswap": GateDef(0, 3, _fixed(controlled(SWAP))),
    "fredkin": GateDef(0, 3, _fixed(controlled(SWAP))),
    # four and five qubits (qelib1.inc)
    "c3x": GateDef(0, 4, _fixed(controlled(X, 3))),
    "c3sqrtx": GateDef(0, 4, _fixed(controlled(SX, 3))),
    "c4x": GateDef(0, 5, _fixed(controlled(X, 4))),
    "u0": GateDef(1, 1, lambda _gamma: I2),
}


#: Spellings of the same gate; noise rules and reports use the canonical name on the right.
ALIASES: dict[str, str] = {
    "U": "u",
    "u3": "u",
    "CX": "cx",
    "cnot": "cx",
    "phase": "p",
    "u1": "p",
    "cphase": "cp",
    "cu1": "cp",
    "toffoli": "ccx",
    "fredkin": "cswap",
    "i": "id",
}


def canonical_name(name: str) -> str:
    return ALIASES.get(name, name)


def gate_matrix(name: str, params: tuple[float, ...]) -> Matrix:
    spec = GATES[name]
    return spec.matrix(*params)


def matrix_power(u: Matrix, k: float) -> Matrix:
    """u**k for a unitary: integer powers exactly, fractional ones through its eigenvalues."""
    if float(k).is_integer():
        n = int(k)
        return np.linalg.matrix_power(u if n >= 0 else u.conj().T, abs(n))
    vals, vecs = np.linalg.eig(u)
    return (vecs @ np.diag(vals**k) @ np.linalg.inv(vecs)).astype(np.complex128)
