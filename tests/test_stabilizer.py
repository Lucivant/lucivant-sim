"""The stabilizer engine (QF-002 Phase 3, Stage B), checked against the exact statevector engine.

The checks are exact where they can be: every stabilizer row of the final tableau, sign included,
must leave the statevector engine's state unchanged, and the outcomes the tableau allows must be
exactly the outcomes with non-zero probability, each equally likely. Sampled paths (mid-circuit
measurement, reset, conditions, Pauli noise) are compared with the trajectories and density-matrix
engines statistically, with fixed seeds.
"""

from __future__ import annotations

import math
import re
from itertools import product

import numpy as np
import pytest

import lucivant_sim as ls
from lucivant_sim.circuit import Gate
from lucivant_sim.engines import SimulationError
from lucivant_sim.simulator import to_circuit
from lucivant_sim.stabilizer import MAX_STABILIZER_QUBITS, Tableau, _pauli_string, clifford_map

ONE = ["h", "s", "sdg", "x", "y", "z", "sx", "sxdg", "id"]
ONE_PARAM = ["rz(pi/2)", "rz(-pi/2)", "rx(pi)", "ry(pi/2)", "p(pi/2)", "u(pi/2, 0, pi)", "rz(pi)"]
TWO = ["cx", "cz", "cy", "swap", "ecr", "iswap"]


def random_clifford(n: int, depth: int, seed: int, measure: bool = True) -> str:
    rng = np.random.default_rng(seed)
    lines = ["OPENQASM 3.0;", 'include "stdgates.inc";', f"qubit[{n}] q;", f"bit[{n}] c;"]
    for _ in range(depth):
        if n > 1 and rng.random() < 0.45:
            a, b = rng.choice(n, size=2, replace=False)
            lines.append(f"{rng.choice(TWO)} q[{a}], q[{b}];")
        else:
            gate = rng.choice(ONE + ONE_PARAM)
            lines.append(f"{gate} q[{rng.integers(n)}];")
    if measure:
        lines.append("c = measure q;")
    return "\n".join(lines) + "\n"


def tableau_of(source: str) -> Tableau:
    circuit = to_circuit(source)
    state = Tableau(circuit.num_qubits)
    for op in circuit.operations:
        if isinstance(op, Gate):
            cmap = clifford_map(op.matrix)
            assert cmap is not None, op.name
            state.apply(cmap, op.qubits)
    return state


def row_operator(t: Tableau, row: int) -> np.ndarray:
    """The full Pauli matrix of a tableau row in the statevector's little-endian order."""
    bits: list[int] = []
    for q in reversed(range(t.n)):  # most significant tensor factor is the highest qubit
        bits += [int(t.x[q, row]), int(t.z[q, row])]
    return (-1) ** int(t.r[row]) * _pauli_string(bits)


@pytest.mark.parametrize("seed", range(40))
def test_every_stabilizer_row_stabilizes_the_exact_state(seed: int) -> None:
    n = 1 + seed % 5
    source = random_clifford(n, depth=6 + 3 * n, seed=seed, measure=False)
    psi = ls.statevector(source)
    t = tableau_of(source)
    for row in range(t.n, 2 * t.n):
        assert np.allclose(row_operator(t, row) @ psi, psi, atol=1e-9), f"row {row} of seed {seed}"
    # destabilizers anticommute with their own stabilizer and commute with the rest
    for i in range(t.n):
        d, s = row_operator(t, i), row_operator(t, i + t.n)
        assert np.allclose(d @ s, -s @ d)


@pytest.mark.parametrize("seed", range(40))
def test_the_outcomes_allowed_are_exactly_the_possible_ones_and_equally_likely(seed: int) -> None:
    n = 1 + seed % 6
    source = random_clifford(n, depth=8 + 3 * n, seed=1000 + seed)
    exact = ls.probabilities(source)  # {bitstring: p}, little-endian strings
    possible = {k for k, p in exact.items() if p > 1e-12}
    t = tableau_of(source)
    x0, basis = t.support(list(range(n)))
    allowed = set()
    for coeffs in product((0, 1), repeat=len(basis)):
        v = x0.copy()
        for c, row in zip(coeffs, basis, strict=True):
            if c:
                v ^= row
        allowed.add("".join(str(int(v[q])) for q in reversed(range(n))))
    assert allowed == possible
    assert all(math.isclose(exact[k], 1 / len(allowed), abs_tol=1e-9) for k in possible)


def test_endianness_and_registers_match_the_other_engines() -> None:
    source = (
        'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[3] q;\nbit[2] a;\nbit[1] b;\n'
        "x q[1];\na[0] = measure q[0];\na[1] = measure q[1];\nb[0] = measure q[2];\n"
    )
    for method in ("stabilizer", "statevector"):
        assert ls.run(source, shots=50, seed=1, method=method).counts == {"0 10": 50}


def test_seeded_runs_repeat_exactly_and_differ_with_the_seed() -> None:
    source = random_clifford(8, 40, seed=5)
    a = ls.run(source, shots=2000, seed=11, method="stabilizer")
    b = ls.run(source, shots=2000, seed=11, method="stabilizer")
    c = ls.run(source, shots=2000, seed=12, method="stabilizer")
    assert a.counts == b.counts and a.counts_digest == b.counts_digest
    assert a.method == "stabilizer" and a.statevector is None
    assert c.counts != a.counts or len(a.counts) == 1


@pytest.mark.parametrize(
    ("snippet", "why"),
    [
        ("t q[0];", "t on qubits [0] isn't a Clifford gate"),
        ("rx(0.3) q[0];", "isn't a Clifford gate"),
        ("ccx q[0], q[1], q[2];", "ccx on qubits [0, 1, 2] isn't a Clifford gate"),
        ("crz(pi/2) q[0], q[1];", "isn't a Clifford gate"),
    ],
)
def test_non_clifford_gates_are_refused_by_name(snippet: str, why: str) -> None:
    source = f'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[3] q;\nbit[3] c;\nh q[0];\n{snippet}\nc = measure q;\n'
    assert not ls.is_clifford(to_circuit(source))
    with pytest.raises(SimulationError, match=re.escape(why)):
        ls.run(source, shots=10, seed=1, method="stabilizer")


def test_a_thousand_qubit_ghz_state_runs_and_only_all_zeros_or_ones_appear() -> None:
    n = 1000
    lines = ["OPENQASM 3.0;", 'include "stdgates.inc";', f"qubit[{n}] q;", f"bit[{n}] c;", "h q[0];"]
    lines += [f"cx q[{i}], q[{i + 1}];" for i in range(n - 1)]
    lines.append("c = measure q;")
    result = ls.run("\n".join(lines) + "\n", shots=400, seed=3, method="stabilizer")
    assert set(result.counts) == {"0" * n, "1" * n}
    assert 150 < result.counts["0" * n] < 250


def test_the_size_limit_is_explicit() -> None:
    with pytest.raises(SimulationError, match=f"up to {MAX_STABILIZER_QUBITS}"):
        Tableau(MAX_STABILIZER_QUBITS + 1)


def _tvd(a: dict[str, int], b: dict[str, float]) -> float:
    shots = sum(a.values())
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0) / shots - b.get(k, 0.0)) for k in keys)


def test_mid_circuit_measurement_reset_and_conditions_match_trajectories() -> None:
    source = (
        'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[3] q;\nbit[3] c;\n'
        "h q[0];\ncx q[0], q[1];\nc[0] = measure q[0];\nreset q[0];\nh q[0];\n"
        "if (c[0] == 1) { x q[2]; }\ncx q[1], q[2];\nc[1] = measure q[1];\nc[2] = measure q[2];\n"
    )
    shots = 20000
    stab = ls.run(source, shots=shots, seed=21, method="stabilizer").counts
    traj = ls.run(source, shots=shots, seed=22, method="trajectories").counts
    reference = {k: v / shots for k, v in traj.items()}
    assert set(stab) == set(traj)
    assert _tvd(stab, reference) < 0.03


def test_pauli_noise_and_readout_errors_match_the_density_matrix() -> None:
    source = random_clifford(4, 24, seed=77)
    noise = ls.NoiseModel.uniform(single=0.02, two=0.05, readout=0.03)
    shots = 40000
    stab = ls.run(source, shots=shots, seed=5, method="stabilizer", noise=noise).counts
    exact = ls.run(source, shots=shots, seed=6, method="density_matrix", noise=noise).counts
    assert _tvd(stab, {k: v / shots for k, v in exact.items()}) < 0.03


def test_noise_that_is_not_a_pauli_channel_is_refused() -> None:
    source = 'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[1] q;\nbit[1] c;\nh q[0];\nc = measure q;\n'
    noise = ls.NoiseModel().add_gate_error(ls.amplitude_damping(0.1), ["h"])
    with pytest.raises(SimulationError, match="isn't a Pauli channel"):
        ls.run(source, shots=10, seed=1, method="stabilizer", noise=noise)


def test_automatic_selection_uses_the_measured_crossover() -> None:
    """Stage C: Clifford circuits of 12 qubits or more go to the stabilizer engine automatically."""
    assert ls.run(random_clifford(8, 10, seed=3), shots=10, seed=1).method == "statevector"
    assert ls.run(random_clifford(16, 10, seed=3), shots=10, seed=1).method == "stabilizer"
    with_t = random_clifford(16, 10, seed=3, measure=False) + "t q[0];\nc = measure q;\n"
    assert ls.run(with_t, shots=10, seed=1).method == "statevector"
    # wider than any statevector: a 40-qubit GHZ state now runs without asking
    lines = ["OPENQASM 3.0;", 'include "stdgates.inc";', "qubit[40] q;", "bit[40] c;", "h q[0];"]
    lines += [f"cx q[{i}], q[{i + 1}];" for i in range(39)] + ["c = measure q;"]
    wide = ls.run("\n".join(lines) + "\n", shots=100, seed=2)
    assert wide.method == "stabilizer" and set(wide.counts) <= {"0" * 40, "1" * 40}
    # noise that isn't a Pauli channel keeps the general engines
    noise = ls.NoiseModel().add_gate_error(ls.amplitude_damping(0.01))  # every one-qubit gate
    assert ls.run(random_clifford(16, 4, seed=4), shots=10, seed=1, noise=noise).method != "stabilizer"


@pytest.mark.parametrize("seed", range(10))
def test_fast_paths_equal_the_generic_pauli_table(seed: int) -> None:
    """H, S, S†, the Paulis, CX and CZ use Aaronson-Gottesman's rules (Stage C); each must give
    exactly the generic table's tableau, signs included, on a scrambled state."""
    from lucivant_sim import gates
    from lucivant_sim.stabilizer import _kind

    t = tableau_of(random_clifford(5, 30, seed=500 + seed, measure=False))
    matrices = {
        "h": gates.H,
        "s": np.diag([1, 1j]),
        "sdg": np.diag([1, -1j]),
        "x": gates.X,
        "y": gates.Y,
        "z": gates.Z,
        "cx": np.eye(4, dtype=np.complex128)[[0, 1, 3, 2]],
        "cz": np.diag([1, 1, 1, -1]).astype(np.complex128),
    }
    rng = np.random.default_rng(seed)
    for name, m in matrices.items():
        cmap = clifford_map(np.asarray(m, dtype=np.complex128))
        assert cmap is not None and _kind(cmap) == name
        qubits = tuple(int(q) for q in rng.choice(5, size=cmap.k, replace=False))
        fast, generic = t.copy(), t.copy()
        fast.apply(cmap, qubits)
        generic.apply_generic(cmap, qubits)
        assert np.array_equal(fast.x, generic.x) and np.array_equal(fast.z, generic.z), name
        assert np.array_equal(fast.r, generic.r), name


@pytest.mark.parametrize("seed", range(6))
def test_pauli_frames_match_the_per_shot_tableau(seed: int) -> None:
    """Noisy static circuits use Pauli frames (Stage C); the per-shot tableau is the reference."""
    from lucivant_sim.stabilizer import _frames, _per_shot

    circuit = to_circuit(random_clifford(5, 30, seed=900 + seed))
    noise = ls.NoiseModel.uniform(single=0.03, two=0.06, readout=0.02)
    shots = 20000
    a = _frames(circuit, shots, noise, np.random.default_rng(seed))
    b = _per_shot(circuit, shots, noise, np.random.default_rng(seed + 100))
    keys = set(a) | set(b)
    tvd = 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys) / shots
    assert tvd < 0.03, tvd
