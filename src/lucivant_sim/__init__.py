"""Lucivant Orrery: an open-source quantum circuit simulator from Lucivant.

    import lucivant_sim as ls

    result = ls.run('''
        OPENQASM 3;
        include "stdgates.inc";
        qubit[2] q; bit[2] c;
        h q[0]; cx q[0], q[1];
        c = measure q;
    ''', shots=1000, seed=7)
    print(result.counts)        # {'00': ..., '11': ...}

Exact statevector and density-matrix engines, per-shot trajectories for mid-circuit
measurement, a stabilizer (Clifford) engine for wide Clifford circuits, noise channels and
device-calibration noise, and seeded results anyone can verify.
"""

__version__ = "0.3.0"

from lucivant_sim.circuit import Circuit
from lucivant_sim.engines import SimulationError
from lucivant_sim.noise import (
    Channel,
    NoiseModel,
    ReadoutError,
    amplitude_damping,
    bit_flip,
    depolarizing,
    from_gate_error,
    pauli_channel,
    phase_damping,
    phase_flip,
    thermal_relaxation,
)
from lucivant_sim.qasm import QasmError, load, parse
from lucivant_sim.simulator import (
    Result,
    density_matrix,
    expectation,
    outcome_probabilities,
    probabilities,
    run,
    statevector,
    verify,
)
from lucivant_sim.stabilizer import is_clifford

__all__ = [
    "Channel",
    "Circuit",
    "NoiseModel",
    "QasmError",
    "ReadoutError",
    "Result",
    "SimulationError",
    "__version__",
    "amplitude_damping",
    "bit_flip",
    "density_matrix",
    "depolarizing",
    "expectation",
    "from_gate_error",
    "is_clifford",
    "load",
    "outcome_probabilities",
    "parse",
    "pauli_channel",
    "phase_damping",
    "phase_flip",
    "probabilities",
    "run",
    "statevector",
    "thermal_relaxation",
    "verify",
]
