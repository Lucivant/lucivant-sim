"""Differential numerical checks for the optimized Orrery kernel paths."""

from __future__ import annotations

import numpy as np
import pytest

from lucivant_sim.engines import DensityMatrix, _apply
from lucivant_sim.noise import Channel, depolarizing


@pytest.mark.parametrize(("n", "qubits", "p"), [(2, (0,), 0.0), (4, (3,), 0.002), (5, (0, 4), 0.01), (6, (4, 1), 0.3)])
def test_density_depolarizing_matches_kraus_reference(n: int, qubits: tuple[int, ...], p: float) -> None:
    rng = np.random.default_rng(20261021 + n)
    vector = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    vector /= np.linalg.norm(vector)
    rho = np.outer(vector, vector.conj()).reshape((2,) * (2 * n))
    channel = depolarizing(p, len(qubits))
    baseline = rho
    rows = [n - 1 - q for q in qubits]
    cols = [2 * n - 1 - q for q in qubits]
    expected = sum(_apply(_apply(baseline, k, rows), k.conj(), cols) for k in channel.kraus)
    actual = DensityMatrix(n)
    actual.tensor = rho.copy()
    actual.apply_channel(channel, qubits)
    assert np.allclose(actual.tensor, expected, atol=1e-12)
    assert np.trace(actual.matrix()) == pytest.approx(1)


def test_custom_channel_named_depolarizing_uses_its_actual_kraus() -> None:
    x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    channel = Channel("depolarizing", 1, (x,))
    state = DensityMatrix(1)
    state.apply_channel(channel, (0,))
    assert np.allclose(state.matrix(), np.diag([0, 1]))


# QF-002 Phase 3, Stage A: the analytic depolarizing path against an independent reference (Qiskit's
# DensityMatrix and Kraus evolution, little-endian like Orrery), every target placement and order,
# mixed states, and the channel's invariants.
def _mixed_state(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(2**n, 2**n)) + 1j * rng.normal(size=(2**n, 2**n))
    rho = a @ a.conj().T  # full rank, with coherences
    return rho / np.trace(rho)


def _orrery(rho: np.ndarray, channel: Channel, qubits: tuple[int, ...]) -> np.ndarray:
    n = int(np.log2(rho.shape[0]))
    state = DensityMatrix(n)
    state.tensor = rho.reshape((2,) * (2 * n)).copy()
    state.apply_channel(channel, qubits)
    return state.matrix()


def _qiskit(rho: np.ndarray, channel: Channel, qubits: tuple[int, ...]) -> np.ndarray:
    from qiskit.quantum_info import DensityMatrix as QiskitDensity
    from qiskit.quantum_info import Kraus

    return np.asarray(QiskitDensity(rho).evolve(Kraus(list(channel.kraus)), qargs=list(qubits)).data)


_PLACEMENTS = [(q,) for q in range(3)] + [(a, b) for a in range(3) for b in range(3) if a != b]


@pytest.mark.parametrize("qubits", _PLACEMENTS, ids=lambda q: "q" + "-".join(map(str, q)))
@pytest.mark.parametrize("p", [0.0, 1e-4, 0.05, 0.5, 1.0])
def test_analytic_depolarizing_matches_an_independent_reference(qubits: tuple[int, ...], p: float) -> None:
    rho = _mixed_state(3, 7 + len(qubits))
    channel = depolarizing(p, len(qubits))
    assert channel.analytic_depolarizing
    assert np.allclose(_orrery(rho, channel, qubits), _qiskit(rho, channel, qubits), atol=1e-12)


@pytest.mark.parametrize("k", [1, 2])
def test_depolarizing_invariants(k: int) -> None:
    n = 3
    rho = _mixed_state(n, 11)
    qubits = tuple(range(k))
    # identity at p = 0, exactly
    assert np.array_equal(_orrery(rho, depolarizing(0.0, k), qubits), rho)
    for p in (0.01, 0.3, 1.0, 4**k / (4**k - 1)):  # up to the largest physical value
        out = _orrery(rho, depolarizing(p, k), qubits)
        assert np.trace(out) == pytest.approx(1, abs=1e-12)
        assert np.allclose(out, out.conj().T, atol=1e-13)
        assert np.linalg.eigvalsh(out).min() > -1e-12
        # the untouched qubits are untouched: their reduced state is unchanged
        from qiskit.quantum_info import DensityMatrix as QiskitDensity
        from qiskit.quantum_info import partial_trace

        assert np.allclose(
            partial_trace(QiskitDensity(out), list(qubits)).data,
            partial_trace(QiskitDensity(rho), list(qubits)).data,
            atol=1e-12,
        )
    # p = 1 on every qubit is the maximally mixed state
    small = _mixed_state(k, 3)
    assert np.allclose(_orrery(small, depolarizing(1.0, k), tuple(range(k))), np.eye(2**k) / 2**k, atol=1e-12)
    # two applications compose: 1 - (1 - p1)(1 - p2)
    twice = _orrery(_orrery(rho, depolarizing(0.1, k), qubits), depolarizing(0.2, k), qubits)
    assert np.allclose(twice, _orrery(rho, depolarizing(1 - 0.9 * 0.8, k), qubits), atol=1e-12)


def test_fast_path_equals_the_generic_kraus_path_on_the_same_operators() -> None:
    rho = _mixed_state(4, 5)
    for k, qubits in ((1, (2,)), (2, (3, 0))):
        fast = depolarizing(0.37, k)
        generic = Channel("custom", k, fast.kraus)  # same operators, not marked by the factory
        assert not generic.analytic_depolarizing
        assert np.allclose(_orrery(rho, fast, qubits), _orrery(rho, generic, qubits), atol=1e-12)


def test_three_qubit_depolarizing_keeps_the_generic_path() -> None:
    rho = _mixed_state(4, 9)
    channel = depolarizing(0.2, 3)
    assert np.allclose(_orrery(rho, channel, (3, 0, 2)), _qiskit(rho, channel, (3, 0, 2)), atol=1e-12)
