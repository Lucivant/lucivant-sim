"""An OpenQASM 2.0 and 3 reader, written for Lucivant Orrery (no Qiskit or other SDK needed).

It covers what circuits are written in practice: ``qreg``/``creg`` and ``qubit``/``bit``
declarations, the qelib1.inc and stdgates.inc gates, user-defined ``gate`` blocks, the ``ctrl @``,
``negctrl @``, ``inv @`` and ``pow(k) @`` modifiers, ``input`` parameters and constants,
broadcasting over registers, slices, ``measure`` in both spellings, ``reset``, ``barrier``,
``gphase``, ``box`` and ``if``/``else`` on classical bits. Loops, subroutines, pulse-level
calibration and classical arithmetic are refused with a clear message rather than guessed at.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from lucivant_sim.circuit import (
    Barrier,
    Circuit,
    Condition,
    Gate,
    GlobalPhase,
    IfElse,
    Measure,
    Operation,
    Register,
    Reset,
)
from lucivant_sim.gates import (
    GATES,
    Matrix,
    canonical_name,
    controlled,
    gate_matrix,
    matrix_power,
    negcontrolled,
)

#: Declarations beyond this are refused before any memory is used. The stabilizer engine's limit;
#: each engine refuses what it can't hold (statevector 26, density matrix 13).
MAX_QUBITS = 2048


class QasmError(ValueError):
    """The program can't be read. ``line`` and ``column`` point at the problem."""

    def __init__(self, message: str, line: int = 0, column: int = 0) -> None:
        self.line = line
        self.column = column
        super().__init__(f"line {line}, column {column}: {message}" if line else message)


# ---------------------------------------------------------------------------------- tokens


class Token(NamedTuple):
    """A lexical token. A named tuple: tokens are made by the tens of thousands for large
    programs, and a frozen dataclass costs several times more to build (Orrery 0.3)."""

    kind: str  # "id", "num", "str", "op", "phys", "eof"
    value: str
    line: int
    col: int


_TOKEN = re.compile(
    r"""
    (?P<ws>[ \t\r\f\v]+|\n)
  | (?P<lc>//[^\n]*)
  | (?P<bc>/\*.*?\*/)
  | (?P<num>0[bB][01_]+|0[xX][0-9a-fA-F_]+|0[oO][0-7_]+|(?:\d[\d_]*\.?[\d_]*|\.\d[\d_]*)(?:[eE][+-]?\d+)?)
  | (?P<str>"[^"\n]*"|'[^'\n]*')
  | (?P<phys>\$\d+)
  | (?P<id>[^\W\d]\w*)
  | (?P<op>->|==|!=|<=|>=|&&|\|\||\*\*|<<|>>|[\[\](){};,:=+\-*/^%<>!~@&|.])
    """,
    re.VERBOSE | re.DOTALL,
)


#: End-of-input tokens after the last real one: more than the parser ever looks ahead (two).
_EOF_PAD = 8


def tokenize(source: str) -> list[Token]:
    tokens: list[Token] = []
    append = tokens.append
    line, line_start, pos = 1, 0, 0
    # Pragmas and annotations are hints for other tools; they don't change the circuit.
    source = re.sub(r"(?m)^[ \t]*(#?pragma|@)[^\n]*", "", source)
    end = len(source)
    match = _TOKEN.match
    while pos < end:
        m = match(source, pos)
        if not m:
            raise QasmError(f"unexpected character {source[pos]!r}", line, pos - line_start + 1)
        kind = m.lastgroup
        next_pos = m.end()
        if kind == "ws":
            if source[pos] == "\n":
                line += 1
                line_start = next_pos
        elif kind == "lc":
            pass
        elif kind == "bc":
            text = m.group()
            newlines = text.count("\n")
            if newlines:
                line += newlines
                line_start = pos + text.rfind("\n") + 1
        else:
            append(Token(kind or "", m.group(), line, pos - line_start + 1))
        pos = next_pos
    # padded with end tokens, so the parser's lookahead never needs a bounds check
    tokens.extend([Token("eof", "", line, pos - line_start + 1)] * _EOF_PAD)
    return tokens


# ----------------------------------------------------------------------------- expressions

Expr = tuple[Any, ...]

_CONSTANTS = {"pi": math.pi, "π": math.pi, "tau": math.tau, "τ": math.tau, "euler": math.e, "ℇ": math.e}
_FUNCTIONS: dict[str, Callable[[float], float]] = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "arcsin": math.asin,
    "asin": math.asin,
    "arccos": math.acos,
    "acos": math.acos,
    "arctan": math.atan,
    "atan": math.atan,
    "exp": math.exp,
    "ln": math.log,
    "log": math.log,
    "sqrt": math.sqrt,
}


def _number(text: str) -> float:
    t = text.replace("_", "").lower()
    if t.startswith("0b"):
        return float(int(t[2:], 2))
    if t.startswith("0x"):
        return float(int(t[2:], 16))
    if t.startswith("0o"):
        return float(int(t[2:], 8))
    return float(t)


# ------------------------------------------------------------------------------ gate defs


@dataclass(frozen=True)
class _Call:
    modifiers: tuple[tuple[str, Expr | None], ...]
    name: str
    params: tuple[Expr, ...]
    args: tuple[str, ...]
    token: Token


@dataclass
class _GateDef:
    name: str
    params: tuple[str, ...]
    qubits: tuple[str, ...]
    body: list[_Call | tuple[str, Expr]]  # ("gphase", angle) entries too


# Gates qelib1.inc defines by decomposition rather than as a matrix (relative-phase Toffolis).
_PRELUDE = """
gate rccx a,b,c { u2(0,pi) c; u1(pi/4) c; cx b,c; u1(-pi/4) c; cx a,c; u1(pi/4) c; cx b,c;
  u1(-pi/4) c; u2(0,pi) c; }
gate rc3x a,b,c,d { u2(0,pi) d; u1(pi/4) d; cx c,d; u1(-pi/4) d; u2(0,pi) d; cx a,d; u1(pi/4) d;
  cx b,d; u1(-pi/4) d; cx a,d; u1(pi/4) d; cx b,d; u1(-pi/4) d; u2(0,pi) d; u1(pi/4) d; cx c,d;
  u1(-pi/4) d; u2(0,pi) d; }
"""

_STANDARD_INCLUDES = {"qelib1.inc", "stdgates.inc"}


def apply_local(op: Matrix, sub: Matrix, positions: tuple[int, ...], k: int) -> Matrix:
    """``sub`` (on local qubits ``positions``, first = most significant) composed after ``op``,
    a k-qubit operator whose local qubit j is its j-th most significant bit."""
    m = len(positions)
    tensor = op.reshape([2] * k + [2**k])
    moved = np.tensordot(sub.reshape([2] * (2 * m)), tensor, axes=(list(range(m, 2 * m)), list(positions)))
    moved = np.moveaxis(moved, list(range(m)), list(positions))
    return moved.reshape(2**k, 2**k)


# ---------------------------------------------------------------------------------- parser


@dataclass
class _Scope:
    qregs: dict[str, Register] = field(default_factory=dict)
    cregs: dict[str, Register] = field(default_factory=dict)
    values: dict[str, float] = field(default_factory=dict)
    num_qubits: int = 0
    num_clbits: int = 0
    physical: int = -1  # highest $n used


class _Parser:
    def __init__(
        self,
        source: str,
        inputs: Mapping[str, float],
        include: Callable[[str], str] | None,
    ) -> None:
        self.tokens = tokenize(source)
        self.pos = 0
        self.inputs = dict(inputs)
        self.used_inputs: dict[str, float] = {}
        self.include = include
        self.scope = _Scope()
        self.gates: dict[str, _GateDef] = {}
        self.user_defined: set[str] = set()
        self.cache: dict[tuple[Any, ...], Matrix] = {}
        self.version = 2
        prelude = _Parser.__new__(_Parser)
        prelude.tokens, prelude.pos, prelude.gates = tokenize(_PRELUDE), 0, self.gates
        prelude.user_defined = set()
        while prelude.peek().kind != "eof":
            prelude.gate_definition()

    # -- token helpers --------------------------------------------------------------------

    def peek(self, ahead: int = 0) -> Token:
        # the token list ends in _EOF_PAD end tokens; past them, stay on the last
        index = self.pos + ahead
        tokens = self.tokens
        return tokens[index] if index < len(tokens) else tokens[-1]

    def next(self) -> Token:
        tok = self.peek()
        self.pos += 1
        return tok

    def error(self, message: str, tok: Token | None = None) -> QasmError:
        tok = tok or self.peek()
        return QasmError(message, tok.line, tok.col)

    def accept(self, value: str) -> bool:
        tok = self.peek()
        if tok.value == value and tok.kind in ("op", "id"):
            self.pos += 1
            return True
        return False

    def expect(self, value: str) -> Token:
        tok = self.peek()
        if tok.value != value or tok.kind not in ("op", "id"):
            found = "the end of the program" if tok.kind == "eof" else repr(tok.value)
            raise self.error(f"expected {value!r} but found {found}", tok)
        self.pos += 1
        return tok

    def ident(self) -> Token:
        tok = self.next()
        if tok.kind != "id":
            raise self.error(f"expected a name but found {tok.value!r}", tok)
        return tok

    # -- expressions ----------------------------------------------------------------------

    def expr(self) -> Expr:
        left = self.term()
        while self.peek().value in ("+", "-") and self.peek().kind == "op":
            op = self.next().value
            left = ("bin", op, left, self.term())
        return left

    def term(self) -> Expr:
        left = self.unary()
        while self.peek().value in ("*", "/", "%") and self.peek().kind == "op":
            op = self.next().value
            left = ("bin", op, left, self.unary())
        return left

    def unary(self) -> Expr:
        if self.peek().kind == "op" and self.peek().value in ("-", "+"):
            op = self.next().value
            inner = self.unary()
            return ("neg", inner) if op == "-" else inner
        return self.power()

    def power(self) -> Expr:
        base = self.atom()
        if self.peek().kind == "op" and self.peek().value in ("^", "**"):
            self.next()
            return ("bin", "**", base, self.unary())
        return base

    def atom(self) -> Expr:
        tok = self.next()
        if tok.kind == "num":
            return ("num", _number(tok.value), tok)
        if tok.kind == "op" and tok.value == "(":
            inner = self.expr()
            self.expect(")")
            return inner
        if tok.kind == "id":
            if tok.value in _FUNCTIONS and self.peek().value == "(":
                self.expect("(")
                arg = self.expr()
                self.expect(")")
                return ("call", tok.value, arg, tok)
            return ("var", tok.value, tok)
        raise self.error(f"expected a number or an expression but found {tok.value!r}", tok)

    def evaluate(self, e: Expr, env: Mapping[str, float] | None = None) -> float:
        kind = e[0]
        if kind == "num":
            return float(e[1])
        if kind == "var":
            name, tok = e[1], e[2]
            if env and name in env:
                return env[name]
            if name in self.scope.values:
                return self.scope.values[name]
            if name in _CONSTANTS:
                return _CONSTANTS[name]
            raise self.error(f"unknown name {name!r}", tok)
        if kind == "neg":
            return -self.evaluate(e[1], env)
        if kind == "call":
            try:
                return _FUNCTIONS[e[1]](self.evaluate(e[2], env))
            except ValueError as exc:
                raise self.error(f"{e[1]}: {exc}", e[3]) from exc
        op, a, b = e[1], self.evaluate(e[2], env), self.evaluate(e[3], env)
        try:
            if op == "+":
                return a + b
            if op == "-":
                return a - b
            if op == "*":
                return a * b
            if op == "/":
                return a / b
            if op == "%":
                return math.fmod(a, b)
            return float(a**b)
        except (ZeroDivisionError, OverflowError, TypeError) as exc:
            raise self.error(f"can't evaluate the expression: {exc}") from exc

    def integer(self, e: Expr, what: str) -> int:
        value = self.evaluate(e)
        if not math.isfinite(value) or abs(value - round(value)) > 1e-9:
            raise self.error(f"{what} must be a whole number")
        return round(value)

    # -- program --------------------------------------------------------------------------

    def program(self) -> list[Operation]:
        ops: list[Operation] = []
        if self.peek().value == "OPENQASM":
            self.next()
            tok = self.next()
            if tok.kind != "num" or tok.value.split(".")[0] not in ("2", "3"):
                raise self.error("only OpenQASM 2 and 3 are supported", tok)
            self.version = int(tok.value.split(".")[0])
            self.expect(";")
        while self.peek().kind != "eof":
            ops.extend(self.statement())
        return ops

    def block_or_statement(self) -> list[Operation]:
        if self.accept("{"):
            ops: list[Operation] = []
            while not self.accept("}"):
                if self.peek().kind == "eof":
                    raise self.error("this block is never closed with '}'")
                ops.extend(self.statement())
            return ops
        return self.statement()

    def statement(self) -> list[Operation]:
        tok = self.peek()
        word = tok.value if tok.kind == "id" else None
        if tok.kind == "op" and tok.value == ";":
            self.next()
            return []
        if word == "include":
            self.next()
            name = self.next()
            if name.kind != "str":
                raise self.error("include needs a file name in quotes", name)
            self.expect(";")
            file = name.value[1:-1]
            if file in _STANDARD_INCLUDES:
                return []
            if self.include is None:
                raise self.error(f"can't include {file!r}: only qelib1.inc and stdgates.inc are built in", name)
            nested = _Parser(self.include(file), self.inputs, self.include)
            nested.scope, nested.gates, nested.user_defined = self.scope, self.gates, self.user_defined
            ops = nested.program()
            self.used_inputs |= nested.used_inputs
            return ops
        if word in ("qreg", "creg"):
            self.next()
            name = self.ident()
            self.expect("[")
            size = self.integer(self.expr(), "a register size")
            self.expect("]")
            self.expect(";")
            self.declare(name, size, quantum=word == "qreg")
            return []
        if word in ("qubit", "bit"):
            self.next()
            size = 1
            if self.accept("["):
                size = self.integer(self.expr(), "a register size")
                self.expect("]")
            name = self.ident()
            self.declare(name, size, quantum=word == "qubit")
            if word == "bit" and self.accept("="):
                self.expect("measure")
                targets = self.qubit_operand()
                self.expect(";")
                return self.measurements(targets, self.clbit_operand_of(name.value))
            self.expect(";")
            return []
        if word == "gate":
            self.gate_definition()
            return []
        if word in ("input", "const", "float", "angle", "int", "uint", "bool"):
            self.classical_declaration()
            return []
        if word == "measure":
            self.next()
            targets = self.qubit_operand()
            if self.accept("->"):
                bits = self.clbit_operand()
                self.expect(";")
                return self.measurements(targets, bits)
            self.expect(";")
            return [Measure(q, None) for q in targets[0]]
        if word == "reset":
            self.next()
            targets = self.qubit_operand()
            self.expect(";")
            return [Reset(q) for q in targets[0]]
        if word == "barrier":
            self.next()
            qubits: list[int] = []
            while self.peek().value != ";":
                qubits.extend(self.qubit_operand()[0])
                if not self.accept(","):
                    break
            self.expect(";")
            return [Barrier(tuple(qubits) or tuple(range(self.scope.num_qubits)))]
        if word == "if":
            return [self.if_statement()]
        if word == "box":
            self.next()
            if self.accept("["):
                self.skip_until("]")
            return self.block_or_statement()
        if word == "delay":
            self.next()
            self.expect("[")
            self.skip_until("]")
            while self.peek().value != ";":
                self.next()
            self.expect(";")
            return []
        if word in ("for", "while", "def", "defcal", "cal", "extern", "opaque", "return", "break", "continue"):
            raise self.error(f"'{word}' isn't supported by Lucivant Orrery yet; unroll it or define a gate instead")
        if word == "gphase":
            self.next()
            self.expect("(")
            angle = self.evaluate(self.expr())
            self.expect(")")
            self.expect(";")
            return [GlobalPhase(angle)]
        if tok.kind == "id" and self.peek(1).value in ("=", "[") and self.is_classical_target(tok.value):
            # c = measure q;  or  c[0] = measure q[0];
            bits = self.clbit_operand()
            self.expect("=")
            if not self.accept("measure"):
                raise self.error("only measurement results can be assigned to bits")
            targets = self.qubit_operand()
            self.expect(";")
            return self.measurements(targets, bits)
        if tok.kind == "id":
            return self.gate_statement()
        raise self.error(f"unexpected {tok.value!r}" if tok.kind != "eof" else "unexpected end of program")

    def skip_until(self, closing: str) -> None:
        depth = 0
        while True:
            tok = self.next()
            if tok.kind == "eof":
                raise self.error(f"expected {closing!r}")
            if tok.value == "[":
                depth += 1
            elif tok.value == closing:
                if depth == 0:
                    return
                depth -= 1

    def is_classical_target(self, name: str) -> bool:
        return name in self.scope.cregs

    # -- declarations ---------------------------------------------------------------------

    def declare(self, name: Token, size: int, *, quantum: bool) -> None:
        s = self.scope
        if name.value in s.qregs or name.value in s.cregs or name.value in s.values:
            raise self.error(f"{name.value!r} is already declared", name)
        if size < 1:
            raise self.error("a register needs at least one bit", name)
        if quantum:
            if s.physical >= 0:
                raise self.error("a program can't mix declared qubits with physical qubits ($n)", name)
            if s.num_qubits + size > MAX_QUBITS:
                raise self.error(f"more than {MAX_QUBITS} qubits", name)
            s.qregs[name.value] = Register(name.value, s.num_qubits, size)
            s.num_qubits += size
        else:
            s.cregs[name.value] = Register(name.value, s.num_clbits, size)
            s.num_clbits += size

    def classical_declaration(self) -> None:
        is_input = self.accept("input")
        self.accept("const")
        type_tok = self.ident()
        if type_tok.value not in ("float", "angle", "int", "uint", "bool"):
            raise self.error(f"'{type_tok.value}' parameters aren't supported; use float or angle", type_tok)
        if self.accept("["):
            self.skip_until("]")
        name = self.ident()
        if name.value in self.scope.values:
            raise self.error(f"{name.value!r} is already declared", name)
        if is_input:
            if name.value not in self.inputs:
                raise self.error(f"the program needs a value for its input {name.value!r}", name)
            value = float(self.inputs[name.value])
            self.used_inputs[name.value] = value
        else:
            self.expect("=")
            value = self.evaluate(self.expr())
        self.expect(";")
        self.scope.values[name.value] = value

    # -- operands -------------------------------------------------------------------------

    def indices(self, size: int, tok: Token) -> list[int]:
        """The index part of ``name[...]``; the opening bracket has been read."""
        if self.accept("{"):
            out = [self.integer(self.expr(), "an index")]
            while self.accept(","):
                out.append(self.integer(self.expr(), "an index"))
            self.expect("}")
        else:
            first = self.integer(self.expr(), "an index")
            if self.accept(":"):
                second = self.integer(self.expr(), "an index")
                if self.accept(":"):
                    step, stop = second, self.integer(self.expr(), "an index")
                else:
                    step, stop = 1, second
                if step == 0:
                    raise self.error("a range's step can't be 0", tok)
                out = list(range(first, stop + (1 if step > 0 else -1), step))
            else:
                out = [first]
        self.expect("]")
        resolved = []
        for i in out:
            j = i + size if i < 0 else i
            if not 0 <= j < size:
                raise self.error(f"index {i} is outside {tok.value!r}, which has {size}", tok)
            resolved.append(j)
        return resolved

    def qubit_operand(self, local: Mapping[str, int] | None = None) -> tuple[list[int], bool]:
        """Qubits an operand names, and whether it named a whole register (for broadcasting)."""
        tok = self.next()
        if tok.kind == "phys":
            if self.scope.qregs:
                raise self.error("a program can't mix declared qubits with physical qubits ($n)", tok)
            n = int(tok.value[1:])
            if n >= MAX_QUBITS:
                raise self.error(f"more than {MAX_QUBITS} qubits", tok)
            self.scope.physical = max(self.scope.physical, n)
            self.scope.num_qubits = self.scope.physical + 1
            return [n], False
        if tok.kind != "id":
            raise self.error(f"expected a qubit but found {tok.value!r}", tok)
        if local is not None:
            if tok.value not in local:
                raise self.error(f"{tok.value!r} isn't an argument of this gate", tok)
            return [local[tok.value]], False
        reg = self.scope.qregs.get(tok.value)
        if reg is None:
            raise self.error(f"unknown qubit register {tok.value!r}", tok)
        if self.accept("["):
            return [reg.offset + i for i in self.indices(reg.size, tok)], False
        return list(range(reg.offset, reg.offset + reg.size)), True

    def clbit_operand(self) -> tuple[list[int], bool]:
        tok = self.ident()
        return self.clbit_operand_of(tok.value, tok)

    def clbit_operand_of(self, name: str, tok: Token | None = None) -> tuple[list[int], bool]:
        reg = self.scope.cregs.get(name)
        if reg is None:
            raise self.error(f"unknown classical register {name!r}", tok)
        if tok is not None and self.accept("["):
            return [reg.offset + i for i in self.indices(reg.size, tok)], False
        return list(range(reg.offset, reg.offset + reg.size)), True

    def measurements(self, qubits: tuple[list[int], bool], bits: tuple[list[int], bool]) -> list[Operation]:
        if len(qubits[0]) != len(bits[0]):
            raise self.error(f"can't measure {len(qubits[0])} qubits into {len(bits[0])} bits")
        return [Measure(q, c) for q, c in zip(qubits[0], bits[0], strict=True)]

    # -- conditions -----------------------------------------------------------------------

    def if_statement(self) -> IfElse:
        self.expect("if")
        self.expect("(")
        negate = self.accept("!")
        tok = self.ident()
        bits, _ = self.clbit_operand_of(tok.value, tok)
        value = 1
        if self.peek().value in ("==", "!="):
            if self.next().value == "!=":
                negate = not negate
            num = self.next()
            if num.kind == "num":
                value = int(_number(num.value))
            elif num.value in ("true", "false"):
                value = int(num.value == "true")
            else:
                raise self.error("a condition compares with a whole number", num)
        elif len(bits) != 1:
            raise self.error("compare a register with a value, e.g. if (c == 1)", tok)
        self.expect(")")
        then = self.block_or_statement()
        otherwise: list[Operation] = []
        if self.accept("else"):
            otherwise = self.block_or_statement()
        return IfElse(Condition(tuple(bits), value, negate), tuple(then), tuple(otherwise))

    # -- gates ----------------------------------------------------------------------------

    def modifiers(self) -> list[tuple[str, Expr | None]]:
        mods: list[tuple[str, Expr | None]] = []
        while self.peek().value in ("ctrl", "negctrl", "inv", "pow") and self.peek().kind == "id":
            # a modifier is always followed (eventually) by '@'
            look = 1
            if self.peek(1).value == "(":
                depth = 0
                while True:
                    v = self.peek(look).value
                    if v == "(":
                        depth += 1
                    elif v == ")":
                        depth -= 1
                        if depth == 0:
                            break
                    elif self.peek(look).kind == "eof":
                        break
                    look += 1
                look += 1
            if self.peek(look).value != "@":
                break
            word = self.next().value
            arg: Expr | None = None
            if self.accept("("):
                arg = self.expr()
                self.expect(")")
            self.expect("@")
            mods.append((word, arg))
        return mods

    def call(self) -> tuple[list[tuple[str, Expr | None]], Token, list[Expr]]:
        mods = self.modifiers()
        name = self.ident()
        params: list[Expr] = []
        if self.accept("(") and not self.accept(")"):
            params.append(self.expr())
            while self.accept(","):
                params.append(self.expr())
            self.expect(")")
        return mods, name, params

    def gate_statement(self) -> list[Operation]:
        mods, name, param_exprs = self.call()
        params = tuple(self.evaluate(p) for p in param_exprs)
        operands: list[tuple[list[int], bool]] = []
        while self.peek().value != ";":
            operands.append(self.qubit_operand())
            if not self.accept(","):
                break
        self.expect(";")
        if name.value == "gphase" and not operands:
            if len(params) != 1:
                raise self.error("gphase takes one angle", name)
            return [GlobalPhase(params[0])]
        matrix, arity = self.resolve(mods, name, params)
        if len(operands) != arity:
            raise self.error(f"{name.value} acts on {arity} qubit(s) but was given {len(operands)}", name)
        widths = {len(qs) for qs, is_reg in operands if is_reg or len(qs) > 1}
        if len(widths) > 1:
            raise self.error("registers of different sizes can't be combined in one gate", name)
        times = widths.pop() if widths else 1
        ops: list[Operation] = []
        label = "".join(f"{m}@" for m, _ in mods) + canonical_name(name.value)
        for i in range(times):
            qubits = tuple(qs[i] if len(qs) > 1 else qs[0] for qs, _ in operands)
            if len(set(qubits)) != len(qubits):
                raise self.error(f"{name.value} is given the same qubit twice", name)
            ops.append(Gate(label, qubits, matrix, params))
        return ops

    def resolve(
        self,
        mods: list[tuple[str, Expr | None]],
        name: Token,
        params: tuple[float, ...],
        env: Mapping[str, float] | None = None,
    ) -> tuple[Matrix, int]:
        matrix, arity = self.base_matrix(name, params)
        for word, arg in reversed(mods):
            if word == "inv":
                matrix = matrix.conj().T
            elif word == "pow":
                if arg is None:
                    raise self.error("pow needs an exponent, e.g. pow(2) @", name)
                matrix = matrix_power(matrix, self.evaluate(arg, env))
            else:
                n = 1 if arg is None else self.integer(arg, "the number of controls")
                for _ in range(n):
                    matrix = controlled(matrix) if word == "ctrl" else negcontrolled(matrix)
                arity += n
        return matrix, arity

    def base_matrix(self, name: Token, params: tuple[float, ...]) -> tuple[Matrix, int]:
        key = (name.value, tuple(round(p, 14) for p in params))
        definition = self.gates.get(name.value)
        if definition is not None and (name.value in self.user_defined or name.value not in GATES):
            if len(params) != len(definition.params):
                raise self.error(
                    f"{name.value} takes {len(definition.params)} parameter(s) but was given {len(params)}", name
                )
            if key not in self.cache:
                self.cache[key] = self.unitary(definition, params)
            return self.cache[key], len(definition.qubits)
        spec = GATES.get(name.value)
        if spec is None:
            raise self.error(f"unknown gate {name.value!r}", name)
        if len(params) != spec.params:
            raise self.error(f"{name.value} takes {spec.params} parameter(s) but was given {len(params)}", name)
        if key not in self.cache:
            self.cache[key] = gate_matrix(name.value, params)
        return self.cache[key], spec.qubits

    def unitary(self, definition: _GateDef, params: tuple[float, ...]) -> Matrix:
        k = len(definition.qubits)
        env = dict(zip(definition.params, params, strict=True))
        local = {q: i for i, q in enumerate(definition.qubits)}
        op = np.eye(2**k, dtype=np.complex128)
        for item in definition.body:
            if isinstance(item, tuple):
                op = op * np.exp(1j * self.evaluate(item[1], env))
                continue
            values = tuple(self.evaluate(p, env) for p in item.params)
            sub, arity = self.resolve(list(item.modifiers), item.token, values, env)
            if len(item.args) != arity:
                raise self.error(f"{item.name} acts on {arity} qubit(s) but was given {len(item.args)}", item.token)
            positions = tuple(local[a] for a in item.args)
            if len(set(positions)) != len(positions):
                raise self.error(f"{item.name} is given the same qubit twice", item.token)
            op = apply_local(op, sub, positions, k)
        return op

    def gate_definition(self) -> None:
        self.expect("gate")
        name = self.ident()
        if name.value in self.user_defined:
            raise self.error(f"gate {name.value!r} is already defined", name)
        params: list[str] = []
        if self.accept("(") and not self.accept(")"):
            params.append(self.ident().value)
            while self.accept(","):
                params.append(self.ident().value)
            self.expect(")")
        qubits = [self.ident().value]
        while self.accept(","):
            qubits.append(self.ident().value)
        if len(set(qubits)) != len(qubits):
            raise self.error("a gate's qubit arguments must have different names", name)
        self.expect("{")
        body: list[_Call | tuple[str, Expr]] = []
        while not self.accept("}"):
            tok = self.peek()
            if tok.kind == "eof":
                raise self.error(f"gate {name.value!r} is never closed with '}}'", name)
            if tok.value == "barrier":
                while self.next().value != ";":
                    pass
                continue
            mods, call_name, call_params = self.call()
            if call_name.value == "gphase":
                self.expect(";")
                if len(call_params) != 1:
                    raise self.error("gphase takes one angle", call_name)
                body.append(("gphase", call_params[0]))
                continue
            args: list[str] = []
            while self.peek().value != ";":
                arg = self.ident()
                if arg.value not in qubits:
                    raise self.error(f"{arg.value!r} isn't an argument of gate {name.value!r}", arg)
                args.append(arg.value)
                if not self.accept(","):
                    break
            self.expect(";")
            if call_name.value == name.value:
                raise self.error("a gate can't call itself", call_name)
            body.append(_Call(tuple(mods), call_name.value, tuple(call_params), tuple(args), call_name))
        self.gates[name.value] = _GateDef(name.value, tuple(params), tuple(qubits), body)
        self.user_defined.add(name.value)


# ------------------------------------------------------------------------------------- API


def parse(
    source: str,
    params: Mapping[str, float] | None = None,
    *,
    include_dir: str | Path | None = None,
) -> Circuit:
    """Read an OpenQASM 2 or 3 program into a :class:`Circuit`.

    ``params`` gives values for the program's ``input`` parameters. ``include_dir`` allows
    ``include "file.inc";`` for files inside that folder (never outside it).
    """
    include: Callable[[str], str] | None = None
    if include_dir is not None:
        root = Path(include_dir).resolve()

        def include(name: str) -> str:
            path = (root / name).resolve()
            if root not in path.parents:
                raise QasmError(f"include {name!r} is outside {root}")
            return path.read_text(encoding="utf-8")

    parser = _Parser(source, params or {}, include)
    ops = parser.program()
    s = parser.scope
    qregs = tuple(s.qregs.values())
    if s.physical >= 0:
        qregs = (Register("$", 0, s.num_qubits),)
    unused = set(params or {}) - set(parser.used_inputs)
    if unused:
        raise QasmError(f"the program has no input named {', '.join(sorted(unused))}")
    return Circuit(
        num_qubits=s.num_qubits,
        num_clbits=s.num_clbits,
        operations=tuple(ops),
        qregs=qregs,
        cregs=tuple(s.cregs.values()),
        inputs=dict(parser.used_inputs),
    )


def load(path: str | Path, params: Mapping[str, float] | None = None) -> Circuit:
    """Read an OpenQASM file; includes resolve beside it."""
    p = Path(path)
    return parse(p.read_text(encoding="utf-8"), params, include_dir=p.parent)
