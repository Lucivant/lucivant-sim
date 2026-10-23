"""Behaviour: the OpenQASM reader, noise, engines, sampling statistics, reproducibility and the CLI."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

import lucivant_sim as ls
from lucivant_sim.cli import main

pytestmark = pytest.mark.unit

BELL = """
OPENQASM 3;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c = measure q;
"""


def tvd(counts: dict[str, int], probs: dict[str, float]) -> float:
    total = sum(counts.values())
    keys = set(counts) | set(probs)
    return 0.5 * sum(abs(counts.get(k, 0) / total - probs.get(k, 0.0)) for k in keys)


# ------------------------------------------------------------------------------- reader


def test_qasm2_and_qasm3_spellings_give_the_same_circuit() -> None:
    q2 = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\nh q[0];\nCX q[0],q[1];\nmeasure q -> c;'
    assert ls.parse(q2).digest() == ls.parse(BELL).digest()


def test_comments_and_whitespace_dont_change_the_digest() -> None:
    noisy = BELL.replace("h q[0];", "/* prepare */ h   q[0]; // superpose")
    assert ls.parse(noisy).digest() == ls.parse(BELL).digest()


def test_broadcast_slices_and_sets() -> None:
    c = ls.parse("OPENQASM 3; qubit[4] q; qubit[4] r; h q; cx q, r; x q[1:2]; z q[{0, 3}]; y q[-1];")
    names = [(op.name, op.qubits) for op in c.operations]
    assert names[:4] == [("h", (0,)), ("h", (1,)), ("h", (2,)), ("h", (3,))]
    assert names[4:8] == [("cx", (0, 4)), ("cx", (1, 5)), ("cx", (2, 6)), ("cx", (3, 7))]
    assert names[8:10] == [("x", (1,)), ("x", (2,))]
    assert names[10:] == [("z", (0,)), ("z", (3,)), ("y", (3,))]


def test_user_gates_modifiers_and_inputs() -> None:
    source = """
    OPENQASM 3;
    include "stdgates.inc";
    input float theta;
    const float half = theta / 2;
    gate rot(a) t { ry(a) t; }
    gate pair(a) x, y { rot(a) x; ctrl @ rot(2 * a) x, y; }
    qubit[3] q;
    pair(half) q[0], q[1];
    inv @ s q[2];
    pow(2) @ t q[2];
    negctrl @ x q[0], q[2];
    """
    sv = ls.statevector(source, params={"theta": 1.1})
    # Build the same state by hand.
    reference = """
    OPENQASM 3;
    include "stdgates.inc";
    qubit[3] q;
    ry(0.55) q[0];
    cry(1.1) q[0], q[1];
    sdg q[2];
    s q[2];
    x q[0]; cx q[0], q[2]; x q[0];
    """
    assert np.allclose(sv, ls.statevector(reference), atol=1e-12)


def test_missing_and_unknown_inputs_are_reported() -> None:
    with pytest.raises(ls.QasmError, match="needs a value for its input 'theta'"):
        ls.parse("OPENQASM 3; input float theta; qubit q; rx(theta) q;")
    with pytest.raises(ls.QasmError, match="no input named phi"):
        ls.parse("OPENQASM 3; qubit q; h q;", params={"phi": 1.0})


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("OPENQASM 3; qubit q; foo q;", "line 1, column 22: unknown gate 'foo'"),
        ("OPENQASM 3;\nqubit[2] q;\ncx q[0], q[0];", "line 3, column 1: cx is given the same qubit twice"),
        ("OPENQASM 3; qubit[2] q; h q[2];", "index 2 is outside 'q'"),
        ("OPENQASM 3; qubit q; for uint i in [0:3] { h q; }", "'for' isn't supported"),
        ("OPENQASM 3; qubit q; rx q;", "rx takes 1 parameter"),
        ("OPENQASM 3; qubit[2] q; qubit[3] r; cx q, r;", "different sizes"),
        ('OPENQASM 2.0; include "mine.inc"; qreg q[1];', "only qelib1.inc and stdgates.inc"),
        ("OPENQASM 4; qubit q;", "only OpenQASM 2 and 3"),
        ("OPENQASM 3; qubit q; h q", "expected ';'"),
    ],
)
def test_errors_point_at_the_problem(source: str, message: str) -> None:
    with pytest.raises(ls.QasmError, match=message.replace("(", r"\(").replace("[", r"\[")):
        ls.parse(source)


def test_includes_resolve_only_inside_the_folder(tmp_path: Path) -> None:
    (tmp_path / "mine.inc").write_text("gate flip a { x a; }")
    (tmp_path / "main.qasm").write_text('OPENQASM 3; include "mine.inc"; qubit q; flip q;')
    assert np.allclose(ls.statevector(ls.load(tmp_path / "main.qasm")), [0, 1])
    (tmp_path / "bad.qasm").write_text('OPENQASM 3; include "../../etc/passwd"; qubit q;')
    with pytest.raises(ls.QasmError, match="outside"):
        ls.load(tmp_path / "bad.qasm")


def test_physical_qubits() -> None:
    sv = ls.statevector("OPENQASM 3; x $2;")
    assert sv.shape == (8,) and abs(sv[4]) == pytest.approx(1)


def test_gphase_and_pragmas() -> None:
    sv = ls.statevector("OPENQASM 3;\npragma something\nqubit q;\ngphase(pi);\nx q;")
    assert np.allclose(sv, [0, -1])


# ------------------------------------------------------------------------------- engines


def test_bell_counts_are_balanced_and_correlated() -> None:
    result = ls.run(BELL, shots=20_000, seed=11)
    assert set(result.counts) == {"00", "11"}
    assert tvd(result.counts, {"00": 0.5, "11": 0.5}) < 0.01
    assert result.method == "statevector" and result.statevector is not None


def test_noisy_counts_match_exact_noisy_probabilities() -> None:
    noise = ls.NoiseModel.uniform(single=0.02, two=0.06)
    exact = ls.probabilities(BELL, noise)
    sampled = ls.run(BELL, shots=40_000, seed=5, noise=noise)
    assert sampled.method == "density_matrix"
    assert tvd(sampled.counts, exact) < 0.01


def test_trajectories_agree_with_the_density_matrix() -> None:
    noise = (
        ls.NoiseModel()
        .add_gate_error(ls.amplitude_damping(0.15), gates="h")
        .add_gate_error(ls.depolarizing(0.1, 2), gates="cx")
    )
    exact = ls.probabilities(BELL, noise)
    traj = ls.run(BELL, shots=20_000, seed=9, noise=noise, method="trajectories")
    assert tvd(traj.counts, exact) < 0.015


def test_readout_error_flips_the_record() -> None:
    noise = ls.NoiseModel().add_readout_error(ls.ReadoutError(p01=0.1, p10=0.3))
    source = "OPENQASM 3; qubit[2] q; bit[2] c; x q[1]; c = measure q;"
    result = ls.run(source, shots=40_000, seed=2, noise=noise)
    expected = {"10": 0.9 * 0.7, "00": 0.9 * 0.3, "11": 0.1 * 0.7, "01": 0.1 * 0.3}
    assert result.method == "statevector"
    assert tvd(result.counts, expected) < 0.01


def test_teleportation_with_mid_circuit_measurement_and_conditions() -> None:
    theta = 1.234
    source = f"""
    OPENQASM 3;
    include "stdgates.inc";
    qubit[3] q;
    bit[2] m;
    bit[1] out;
    ry({theta}) q[0];
    h q[1]; cx q[1], q[2];
    cx q[0], q[1]; h q[0];
    m[0] = measure q[0];
    m[1] = measure q[1];
    if (m[1]) x q[2];
    if (m[0] == 1) {{ z q[2]; }}
    out[0] = measure q[2];
    """
    result = ls.run(source, shots=20_000, seed=4)
    assert result.method == "trajectories"
    ones = sum(n for k, n in result.counts.items() if k.split(" ")[0] == "1")
    assert ones / 20_000 == pytest.approx(math.sin(theta / 2) ** 2, abs=0.012)


def test_reset_mid_circuit() -> None:
    source = "OPENQASM 3; qubit q; bit c; x q; reset q; c = measure q;"
    assert ls.run(source, shots=100, seed=1).counts == {"0": 100}


def test_if_else_uses_the_value_before_the_branch() -> None:
    source = """
    OPENQASM 3; include "stdgates.inc";
    qubit[2] q; bit[2] c;
    x q[0];
    c[0] = measure q[0];
    if (c[0] == 1) { c[0] = measure q[1]; } else { x q[1]; }
    c[1] = measure q[1];
    """
    assert ls.run(source, shots=50, seed=1).counts == {"00": 50}


def test_expectation_with_noise_is_exact() -> None:
    source = 'OPENQASM 3; include "stdgates.inc"; qubit[2] q; h q[0]; cx q[0], q[1];'
    noise = ls.NoiseModel().add_gate_error(ls.depolarizing(0.2, 2), gates="cx")
    assert ls.expectation(source, "XX", noise) == pytest.approx(0.8)
    assert ls.expectation(source, {"ZZ": 1.0, "ZI": 3.0}) == pytest.approx(1.0)


def test_thermal_relaxation_decays_one_to_zero() -> None:
    noise = ls.NoiseModel().add_gate_error(ls.thermal_relaxation(t1=10.0, t2=8.0, time=3.0), gates="x")
    probs = ls.probabilities("OPENQASM 3; qubit q; x q;", noise)
    assert probs["1"] == pytest.approx(math.exp(-0.3))


def test_invalid_channels_and_limits() -> None:
    with pytest.raises(ValueError, match="T2 can't exceed"):
        ls.thermal_relaxation(1.0, 3.0, 0.1)
    with pytest.raises(ValueError, match="between 0 and"):
        ls.bit_flip(1.5)
    with pytest.raises(ls.SimulationError, match="statevector method"):
        ls.run(BELL, noise=ls.NoiseModel.uniform(single=0.01), method="statevector")
    with pytest.raises(ls.SimulationError, match="up to 26"):
        ls.run("OPENQASM 3; qubit[30] q; h q; t q;", shots=1)  # not Clifford: the statevector's limit
    # a Clifford circuit that wide runs on the stabilizer engine (Stage C)
    assert ls.run("OPENQASM 3; qubit[30] q; h q;", shots=1).method == "stabilizer"


def test_calibration_noise_prefers_per_qubit_values() -> None:
    cal = {
        "device": "example_device",
        "num_qubits": 2,
        "single_qubit_error": 0.001,
        "two_qubit_error": 0.01,
        "readout_error": 0.02,
        "qubits": [{"qubit": 1, "single_qubit_error": 0.3, "readout_error": None}],
        "couplers": [{"qubits": [0, 1], "error": 0.05}],
    }
    model = ls.NoiseModel.from_calibration(cal)
    assert model.gate_error("sx", (1,)).params[0] == pytest.approx(0.3 * 2)  # type: ignore[union-attr]
    assert model.gate_error("sx", (0,)).params[0] == pytest.approx(0.001 * 2)  # type: ignore[union-attr]
    assert model.gate_error("cx", (1, 0)).params[0] == pytest.approx(0.05 * 4 / 3)  # type: ignore[union-attr]
    assert model.gate_error("rz", (0,)) is None
    assert model.readout_error(1) == ls.ReadoutError(0.02, 0.02)


# ------------------------------------------------------------------------- reproducibility


def test_same_inputs_same_counts_and_verification(tmp_path: Path) -> None:
    noise = ls.NoiseModel.uniform(single=0.01, two=0.03, readout=0.01)
    a = ls.run(BELL, shots=3000, seed=42, noise=noise)
    b = ls.run(BELL, shots=3000, seed=42, noise=noise)
    assert a.counts == b.counts and a.fingerprint == b.fingerprint
    assert ls.run(BELL, shots=3000, seed=43, noise=noise).counts != a.counts
    saved = tmp_path / "result.json"
    saved.write_text(json.dumps(a.to_dict()))
    assert ls.verify(saved, BELL, noise)
    assert not ls.verify(saved, BELL, None)  # different noise
    tampered = a.to_dict() | {"counts": {"00": 1}}
    assert not ls.verify(tampered, BELL, noise)


def test_unseeded_runs_record_their_seed() -> None:
    a = ls.run(BELL, shots=500)
    assert ls.run(BELL, shots=500, seed=a.seed).counts == a.counts


# ----------------------------------------------------------------------------------- CLI


def test_cli_run_state_expect_verify(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    program = tmp_path / "bell.qasm"
    program.write_text(BELL)
    assert main(["run", str(program), "--shots", "200", "--seed", "3"]) == 0
    assert "fingerprint sha256:" in capsys.readouterr().out
    assert main(["run", str(program), "--shots", "200", "--seed", "3", "--two", "0.02", "--json"]) == 0
    saved = tmp_path / "r.json"
    saved.write_text(capsys.readouterr().out)
    assert json.loads(saved.read_text())["method"] == "density_matrix"
    assert main(["verify", str(saved), str(program), "--two", "0.02"]) == 0
    assert "verified" in capsys.readouterr().out
    assert main(["verify", str(saved), str(program)]) == 1
    capsys.readouterr()
    assert main(["state", str(program), "--amplitudes"]) == 0
    assert "+0.707107" in capsys.readouterr().out
    assert main(["expect", str(program), "ZZ"]) == 0
    assert capsys.readouterr().out.strip() == "1"
    bad = tmp_path / "bad.qasm"
    bad.write_text("OPENQASM 3; qubit q; nope q;")
    assert main(["run", str(bad)]) == 2
    assert "unknown gate 'nope'" in capsys.readouterr().err


def test_outcome_probabilities_are_exact_and_include_readout_error() -> None:
    noise = ls.NoiseModel.uniform(two=0.05, readout=0.1)
    exact = ls.outcome_probabilities(BELL, noise)
    assert sum(exact.values()) == pytest.approx(1)
    assert tvd(ls.run(BELL, shots=40_000, seed=8, noise=noise).counts, exact) < 0.01
    several = "OPENQASM 2.0; qreg q[2]; creg a[1]; creg b[1]; x q[1]; measure q[0] -> a[0]; measure q[1] -> b[0];"
    assert ls.outcome_probabilities(several) == {"1 0": 1.0}
