"""The Stage C slice kernel against the reference contraction, forced on for every size."""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

from lucivant_sim import engines, gates


@pytest.fixture(autouse=True)
def _always_slices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engines, "SLICE_MIN_MOVES", 0)
    monkeypatch.setattr(engines, "SLICE_MIN_GENERAL", 0)


def _random_state(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    return (v / np.linalg.norm(v)).reshape((2,) * n) if n else v


def _unitary(k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(2**k, 2**k)) + 1j * rng.normal(size=(2**k, 2**k))
    q, _ = np.linalg.qr(a)
    return q


MATRICES_1Q = {
    "general": _unitary(1, 1),
    "h": gates.H,
    "x": gates.X,
    "y": gates.Y,
    "rz": gates.rz(0.3),
    "phase": gates.phase(1.1),
    "projector": np.array([[1, 0], [0, 0]], dtype=np.complex128),
}
MATRICES_2Q = {
    "general": _unitary(2, 2),
    "cx": np.eye(4, dtype=np.complex128)[[0, 1, 3, 2]],
    "cz": np.diag([1, 1, 1, -1]).astype(np.complex128),
    "swap": np.eye(4, dtype=np.complex128)[[0, 2, 1, 3]],
    "iswap": np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=np.complex128),
    "lower": np.array([[1, 0, 0, 0], [0.5, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.complex128),
}


@pytest.mark.parametrize("n", [1, 2, 3, 4])
@pytest.mark.parametrize("name", list(MATRICES_1Q))
def test_one_qubit_gates_match_the_reference_on_every_axis(n: int, name: str) -> None:
    m = MATRICES_1Q[name]
    for axis in range(n):
        state = _random_state(n, 10 * n + axis)
        expected = engines._apply(state.copy(), m, [axis])
        got = engines._apply_slices(state.copy(), m, [axis])
        assert np.allclose(got, expected, atol=1e-13), (name, n, axis)


@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("name", list(MATRICES_2Q))
def test_two_qubit_gates_match_the_reference_for_every_ordered_pair(n: int, name: str) -> None:
    m = MATRICES_2Q[name]
    for a, b in permutations(range(n), 2):
        state = _random_state(n, 100 * n + 10 * a + b)
        expected = engines._apply(state.copy(), m, [a, b])
        got = engines._apply_slices(state.copy(), m, [a, b])
        assert np.allclose(got, expected, atol=1e-13), (name, n, a, b)


def test_out_of_place_leaves_the_input_untouched() -> None:
    state = _random_state(3, 7)
    before = state.copy()
    engines._apply_slices(state, MATRICES_2Q["general"], [0, 2], inplace=False)
    engines._apply_slices(state, MATRICES_1Q["x"], [1], inplace=False)
    assert np.array_equal(state, before)


def test_density_matrices_and_channels_use_the_kernel_correctly() -> None:
    import lucivant_sim as ls

    program = (
        'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\nh q[0];\ncx q[0],q[2];\nrx(0.4) q[1];\n'
        "cz q[1],q[2];\nu3(0.3,0.2,0.1) q[0];\n"
    )
    noise = ls.NoiseModel().add_gate_error(ls.amplitude_damping(0.1), ["h", "rx"])
    fast = ls.density_matrix(program, noise)
    engines.SLICE_MIN_MOVES = engines.SLICE_MIN_GENERAL = 2**60  # the reference everywhere
    slow = ls.density_matrix(program, noise)
    assert np.allclose(fast, slow, atol=1e-13)
