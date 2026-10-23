"""The circuit the engines run: flat operations on numbered qubits and classical bits.

Every gate is already resolved to a matrix (user-defined gates, ``ctrl @``, ``inv @`` and
``pow(k) @`` included), so engines never need to know OpenQASM. Circuits are immutable, and
:meth:`Circuit.digest` identifies one exactly, independent of whitespace, comments or how the
source spelled it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from lucivant_sim.gates import Matrix


@dataclass(frozen=True, eq=False)
class Gate:
    """A unitary on ``qubits`` (first qubit is the matrix's most significant bit)."""

    name: str
    qubits: tuple[int, ...]
    matrix: Matrix
    params: tuple[float, ...] = ()

    kind: Literal["gate"] = "gate"


@dataclass(frozen=True)
class Measure:
    qubit: int
    clbit: int | None  # None: measured and the outcome discarded (OpenQASM 3 ``measure q;``)

    kind: Literal["measure"] = "measure"


@dataclass(frozen=True)
class Reset:
    qubit: int

    kind: Literal["reset"] = "reset"


@dataclass(frozen=True)
class Barrier:
    qubits: tuple[int, ...]

    kind: Literal["barrier"] = "barrier"


@dataclass(frozen=True)
class GlobalPhase:
    angle: float

    kind: Literal["gphase"] = "gphase"


@dataclass(frozen=True)
class Condition:
    """True when the classical bits ``clbits`` (least significant first) read ``value``."""

    clbits: tuple[int, ...]
    value: int
    negate: bool = False

    def holds(self, bits: list[int]) -> bool:
        actual = sum(bits[c] << i for i, c in enumerate(self.clbits))
        return (actual == self.value) != self.negate


@dataclass(frozen=True)
class IfElse:
    condition: Condition
    then: tuple[Operation, ...]
    otherwise: tuple[Operation, ...] = ()

    kind: Literal["if"] = "if"


Operation = Gate | Measure | Reset | Barrier | GlobalPhase | IfElse


@dataclass(frozen=True)
class Register:
    name: str
    offset: int
    size: int


_MATRIX_JSON: dict[bytes, list[list[list[float]]]] = {}


def _matrix_json(matrix: np.ndarray) -> list[list[list[float]]]:
    """A matrix rounded to 12 decimals as JSON-ready lists (cached: circuits repeat their gates)."""
    raw = matrix.tobytes()
    found = _MATRIX_JSON.get(raw)
    if found is None:
        m = np.round(matrix, 12) + 0.0  # normalise -0.0
        found = [[[float(z.real), float(z.imag)] for z in row] for row in m]
        if len(_MATRIX_JSON) < 8192:
            _MATRIX_JSON[raw] = found
    return found


@dataclass(frozen=True)
class Circuit:
    num_qubits: int
    num_clbits: int
    operations: tuple[Operation, ...]
    qregs: tuple[Register, ...] = ()
    cregs: tuple[Register, ...] = ()
    #: OpenQASM inputs the circuit was built with (``input float theta``), for the record.
    inputs: dict[str, float] = field(default_factory=dict)

    # -- shape --------------------------------------------------------------------------------

    def is_dynamic(self) -> bool:
        """True when a run can't be one simulation sampled many times: an outcome feeds a
        condition, or a qubit is used again after it is measured or reset mid-circuit."""
        touched: set[int] = set()
        finished: set[int] = set()
        for op in self.operations:
            if isinstance(op, IfElse):
                return True
            if isinstance(op, Gate):
                if finished.intersection(op.qubits):
                    return True
                touched.update(op.qubits)
            elif isinstance(op, Reset):
                if op.qubit in touched or op.qubit in finished:
                    return True
            elif isinstance(op, Measure):
                finished.add(op.qubit)
        return False

    def final_measurements(self) -> list[tuple[int, int]]:
        """(qubit, clbit) pairs of a static circuit, in order; a later write to a clbit wins."""
        pairs: dict[int, int] = {}
        for op in self.operations:
            if isinstance(op, Measure) and op.clbit is not None:
                pairs[op.clbit] = op.qubit
        return [(q, c) for c, q in pairs.items()]

    def count_ops(self) -> dict[str, int]:
        counts: dict[str, int] = {}

        def visit(ops: tuple[Operation, ...]) -> None:
            for op in ops:
                if isinstance(op, IfElse):
                    visit(op.then)
                    visit(op.otherwise)
                    continue
                name = op.name if isinstance(op, Gate) else op.kind
                counts[name] = counts.get(name, 0) + 1

        visit(self.operations)
        return dict(sorted(counts.items()))

    # -- identity -----------------------------------------------------------------------------

    def canonical(self) -> dict[str, object]:
        """A JSON-ready description: matrices rounded to 12 decimals so equal circuits from
        different spellings compare equal."""

        def op_json(op: Operation) -> object:
            if isinstance(op, Gate):
                return ["g", list(op.qubits), _matrix_json(op.matrix)]
            if isinstance(op, Measure):
                return ["m", op.qubit, op.clbit]
            if isinstance(op, Reset):
                return ["r", op.qubit]
            if isinstance(op, Barrier):
                return ["b"]
            if isinstance(op, GlobalPhase):
                return ["p", round(op.angle, 12)]
            c = op.condition
            return [
                "if",
                [list(c.clbits), c.value, c.negate],
                [op_json(o) for o in op.then],
                [op_json(o) for o in op.otherwise],
            ]

        return {
            "qubits": self.num_qubits,
            "clbits": self.num_clbits,
            "cregs": [[r.name, r.offset, r.size] for r in self.cregs],
            "ops": [op_json(o) for o in self.operations],
        }

    def digest(self) -> str:
        """sha256 of the canonical description."""
        data = json.dumps(self.canonical(), separators=(",", ":"), sort_keys=True).encode()
        return "sha256:" + hashlib.sha256(data).hexdigest()

    # -- output -------------------------------------------------------------------------------

    def format_bits(self, bits: int) -> str:
        """A classical bit pattern as OpenQASM tools print it: registers last-declared first,
        separated by spaces, each register's bit 0 on the right."""
        if not self.cregs:
            return format(bits, f"0{self.num_clbits}b") if self.num_clbits else ""
        parts = []
        for reg in reversed(self.cregs):
            value = (bits >> reg.offset) & ((1 << reg.size) - 1)
            parts.append(format(value, f"0{reg.size}b"))
        return " ".join(parts)
