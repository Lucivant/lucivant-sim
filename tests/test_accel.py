"""Opt-in accelerators (lucivant_sim.accel): other engines doing the heavy step, Orrery's record kept."""

from __future__ import annotations

import pytest

import lucivant_sim as ls
from lucivant_sim.engines import SimulationError

ROTATIONS = """OPENQASM 3.0;
include "stdgates.inc";
qubit[5] q;
bit[5] c;
rx(0.3) q[0]; ry(1.1) q[1]; h q[2]; t q[2]; cx q[0], q[1]; cp(0.7) q[1], q[3];
ry(0.9) q[4]; cz q[2], q[4]; swap q[0], q[4]; rz(0.4) q[3]; ccx q[0], q[1], q[2];
c = measure q;
"""

CLIFFORD = """OPENQASM 3.0;
include "stdgates.inc";
qubit[6] q;
bit[6] c;
h q[0]; s q[1]; h q[1]; cx q[0], q[2]; sdg q[2]; cz q[1], q[3]; sx q[4]; swap q[4], q[5]; cx q[3], q[5];
c = measure q;
"""


def test_automatic_never_picks_an_accelerator() -> None:
    assert ls.run(ROTATIONS, shots=10, seed=1).method == "statevector"


def test_statevector_aer_samples_like_orrery() -> None:
    pytest.importorskip("qiskit_aer")
    native = ls.run(ROTATIONS, shots=5000, seed=42)
    fast = ls.run(ROTATIONS, shots=5000, seed=42, method="statevector_aer")
    # Aer computes the state; Orrery samples it with the same seeded generator
    assert fast.counts == native.counts
    assert fast.method == "statevector_aer" and fast.version.startswith(f"{ls.__version__}+qiskit-aer-")
    assert fast.fingerprint != native.fingerprint  # the engine is part of what was run
    assert ls.verify(fast.to_dict(), ROTATIONS)


def test_statevector_aer_refuses_what_the_statevector_refuses() -> None:
    pytest.importorskip("qiskit_aer")
    dynamic = (
        "OPENQASM 3.0;\nqubit[2] q;\nbit[2] c;\nh q[0];\nc[0] = measure q[0];\nif (c[0]) x q[1];\n"
        "c[1] = measure q[1];\n"
    )
    with pytest.raises(SimulationError):
        ls.run(dynamic, shots=10, method="statevector_aer")


def test_stabilizer_stim_matches_the_exact_distribution() -> None:
    pytest.importorskip("stim")
    result = ls.run(CLIFFORD, shots=20000, seed=7, method="stabilizer_stim")
    exact = ls.probabilities(ls.parse(CLIFFORD))
    tvd = 0.5 * sum(abs(result.counts.get(k, 0) / 20000 - exact.get(k, 0.0)) for k in set(exact) | set(result.counts))
    assert tvd < 0.02
    assert result.version.startswith(f"{ls.__version__}+stim-")
    assert ls.verify(result.to_dict(), CLIFFORD)


def test_stabilizer_stim_refuses_non_clifford_gates() -> None:
    with pytest.raises(SimulationError, match="Clifford"):
        ls.run(ROTATIONS, shots=10, method="stabilizer_stim")
