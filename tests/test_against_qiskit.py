"""Lucivant Orrery's states and probabilities against Qiskit's, gate by gate and on random circuits."""

from __future__ import annotations

import math

import numpy as np
import pytest
from qiskit import QuantumCircuit, qasm2, qasm3
from qiskit.circuit.library import C3SXGate, RC3XGate, RCCXGate
from qiskit.quantum_info import DensityMatrix, SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel as AerNoise
from qiskit_aer.noise import depolarizing_error

import lucivant_sim as ls
from lucivant_sim.gates import GATES

pytestmark = pytest.mark.unit

rng = np.random.default_rng(2026)


def _prep(qc: QuantumCircuit, n: int, angles: np.ndarray) -> str:
    """A generic entangled starting state, so phases and ordering errors can't hide."""
    lines = []
    for q in range(n):
        a, b, c = (float(x) for x in angles[q])
        qc.u(a, b, c, q)
        lines.append(f"u({a!r},{b!r},{c!r}) q[{q}];")
    for q in range(n - 1):
        qc.cx(q, q + 1)
        lines.append(f"cx q[{q}],q[{q + 1}];")
    return "\n".join(lines)


def _qiskit_apply(qc: QuantumCircuit, name: str, params: list[float], qubits: list[int]) -> None:
    alias = {
        "i": "id",
        "CX": "cx",
        "cnot": "cx",
        "phase": "p",
        "u1": "p",
        "U": "u",
        "u3": "u",
        "cphase": "cp",
        "cu1": "cp",
        "toffoli": "ccx",
        "fredkin": "cswap",
    }
    name = alias.get(name, name)
    if name == "u2":
        qc.u(math.pi / 2, params[0], params[1], qubits[0])
    elif name == "u0":
        qc.id(qubits[0])
    elif name == "cu3":
        qc.cu(*params, 0.0, *qubits)
    elif name == "c3x":
        qc.mcx(qubits[:3], qubits[3])
    elif name == "c4x":
        qc.mcx(qubits[:4], qubits[4])
    elif name == "c3sqrtx":
        qc.append(C3SXGate(), qubits)
    else:
        getattr(qc, name)(*params, *qubits)


@pytest.mark.parametrize("name", sorted(GATES))
def test_every_gate_matches_qiskit(name: str) -> None:
    spec = GATES[name]
    n = max(spec.qubits + 1, 3)
    qubits = [int(q) for q in rng.permutation(n)[: spec.qubits]]
    params = [float(x) for x in rng.uniform(-math.pi, math.pi, spec.params)]
    qc = QuantumCircuit(n)
    prep = _prep(qc, n, rng.uniform(0, 2 * math.pi, (n, 3)))
    _qiskit_apply(qc, name, params, qubits)
    args = ",".join(f"q[{q}]" for q in qubits)
    call = f"{name}({','.join(repr(p) for p in params)})" if params else name
    source = f'OPENQASM 3;\ninclude "stdgates.inc";\nqubit[{n}] q;\n{prep}\n{call} {args};'
    ours = ls.statevector(source)
    theirs = Statevector(qc).data
    assert np.allclose(ours, theirs, atol=1e-10), name


@pytest.mark.parametrize(("gate", "cls"), [("rccx", RCCXGate), ("rc3x", RC3XGate)])
def test_relative_phase_toffolis_match_qelib1(gate: str, cls: type) -> None:
    n = 4 if gate == "rc3x" else 3
    qc = QuantumCircuit(n)
    prep = _prep(qc, n, rng.uniform(0, 2 * math.pi, (n, 3)))
    qc.append(cls(), list(range(n)))
    args = ",".join(f"q[{i}]" for i in range(n))
    source = f'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[{n}];\n{prep}\n{gate} {args};'
    assert np.allclose(ls.statevector(source), Statevector(qc).data, atol=1e-10)


def _same_up_to_phase(a: np.ndarray, b: np.ndarray) -> bool:
    """Qiskit's OpenQASM export drops the global phase of the gates it defines (sxdg becomes
    s; h; s), so exported circuits equal the original only up to a global phase."""
    return bool(abs(abs(np.vdot(a, b)) - 1) < 1e-9 and np.isclose(np.linalg.norm(a), 1))


def _random_circuit(n: int, depth: int, seed: int) -> QuantumCircuit:
    r = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    one = ["h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "sxdg"]
    rot = ["rx", "ry", "rz", "p"]
    two = ["cx", "cz", "cy", "ch", "swap", "iswap", "ecr", "rzz", "rxx", "ryy", "cp", "crx", "rzx"]
    for _ in range(depth):
        kind = r.integers(3)
        if kind == 0:
            getattr(qc, str(r.choice(one)))(int(r.integers(n)))
        elif kind == 1:
            getattr(qc, str(r.choice(rot)))(float(r.uniform(-4, 4)), int(r.integers(n)))
        else:
            a, b = (int(x) for x in r.choice(n, 2, replace=False))
            g = str(r.choice(two))
            if g in ("rzz", "rxx", "ryy", "cp", "crx", "rzx"):
                getattr(qc, g)(float(r.uniform(-4, 4)), a, b)
            else:
                getattr(qc, g)(a, b)
    return qc


@pytest.mark.parametrize("seed", range(8))
def test_random_circuits_through_qasm3_export(seed: int) -> None:
    qc = _random_circuit(n=5, depth=60, seed=seed)
    ours = ls.statevector(qasm3.dumps(qc))
    assert _same_up_to_phase(ours, Statevector(qc).data)


def test_qiskit_and_cirq_objects_are_accepted() -> None:
    qc = _random_circuit(4, 30, 99)
    assert _same_up_to_phase(ls.statevector(qc), Statevector(qc).data)
    cirq = pytest.importorskip("cirq")
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit([cirq.H(a), cirq.CNOT(a, b), cirq.measure(a, b, key="m")])
    result = ls.run(circuit, shots=400, seed=3)
    assert set(result.counts) <= {"00", "11"} and sum(result.counts.values()) == 400


def test_qasm2_export_with_custom_gate_definitions() -> None:
    qc = QuantumCircuit(3)
    sub = QuantumCircuit(2, name="mygate")
    sub.h(0)
    sub.rzz(0.7, 0, 1)
    sub.sx(1)
    qc.append(sub.to_gate(), [2, 0])
    qc.append(sub.to_gate().control(1), [1, 0, 2])
    qc.rzx(0.3, 0, 1)
    text = qasm2.dumps(qc.decompose(reps=0))
    assert _same_up_to_phase(ls.statevector(text), Statevector(qc).data)


def test_expectation_matches_qiskit() -> None:
    qc = _random_circuit(4, 40, 5)
    obs = {"ZZII": 0.5, "XIXY": -1.2, "IIIZ": 2.0}
    expected = Statevector(qc).expectation_value(SparsePauliOp.from_list(list(obs.items()))).real
    assert ls.expectation(qasm3.dumps(qc), obs) == pytest.approx(expected, abs=1e-9)


def test_density_matrix_matches_aer_depolarizing() -> None:
    qc = _random_circuit(3, 25, 11)
    p1, p2 = 0.02, 0.08
    aer_noise = AerNoise(basis_gates=["u", "cx"])
    aer_noise.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["u"])
    aer_noise.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    from qiskit import transpile

    native = transpile(qc, basis_gates=["u", "cx"], optimization_level=0)
    native.save_density_matrix()
    aer = AerSimulator(method="density_matrix", noise_model=aer_noise)
    theirs = DensityMatrix(aer.run(native).result().data()["density_matrix"]).data
    noise = (
        ls.NoiseModel()
        .add_gate_error(ls.depolarizing(p1), gates="u")
        .add_gate_error(ls.depolarizing(p2, 2), gates="cx")
    )
    native.remove_final_measurements()
    native.data = [i for i in native.data if i.operation.name != "save_density_matrix"]
    ours = ls.density_matrix(qasm3.dumps(native), noise)
    assert np.allclose(ours, theirs, atol=1e-9)


def test_counts_format_matches_aer_for_several_registers() -> None:
    source = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    creg a[1];
    creg b[2];
    x q[0];
    x q[2];
    measure q[0] -> a[0];
    measure q[1] -> b[0];
    measure q[2] -> b[1];
    """
    ours = ls.run(source, shots=10, seed=1).counts
    qc = qasm2.loads(source)
    theirs = AerSimulator().run(qc, shots=10, seed_simulator=1).result().get_counts()
    assert ours == dict(theirs)
