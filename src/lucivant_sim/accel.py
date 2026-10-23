"""Optional accelerators: other engines doing Orrery's heavy step, with Orrery's record kept.

Orrery is pure NumPy on one CPU thread, so wide statevector circuits and large Clifford circuits
run faster on specialised engines (docs/SIMULATOR.md, competitive results). These methods are
**opt-in** (``method="statevector_aer"`` or ``method="stabilizer_stim"``) and need the engine
installed (``pip install lucivant-sim[accel]``); ``automatic`` never picks them, so a result never
depends silently on another package.

- ``statevector_aer``: Qiskit Aer computes the final statevector of an ideal, static circuit (no
  randomness involved). Orrery then samples it with its own seeded generator, exactly as the native
  ``statevector`` method does, so the sampling path and seed handling are Orrery's.
- ``stabilizer_stim``: Stim samples an ideal Clifford circuit, seeded from the run's seed. Stim's
  own sampler draws the outcomes, so counts differ from Orrery's stabilizer engine for the same seed.

Either way the result records the accelerator and its version in ``Result.version`` (and so in the
fingerprint), and :func:`lucivant_sim.verify` reruns with the same method: the same inputs and the
same versions give the same counts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from lucivant_sim.circuit import Barrier, Circuit, Gate, GlobalPhase, Measure, Reset
from lucivant_sim.engines import SimulationError

#: Gate names Stim knows, from Orrery's gate names.
STIM_GATES: dict[str, str] = {
    "id": "I",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "h": "H",
    "s": "S",
    "sdg": "S_DAG",
    "sx": "SQRT_X",
    "sxdg": "SQRT_X_DAG",
    "cx": "CX",
    "cnot": "CX",
    "cy": "CY",
    "cz": "CZ",
    "swap": "SWAP",
}


def _require(module: str, extra: str) -> Any:
    try:
        return __import__(module)
    except ImportError as exc:
        raise SimulationError(f"this method needs {extra}: pip install lucivant-sim[accel]") from exc


def aer_version() -> str:
    import qiskit_aer

    return str(qiskit_aer.__version__)


def stim_version() -> str:
    import stim

    return str(stim.__version__)


def aer_statevector(circuit: Circuit) -> NDArray[np.complex128]:
    """The final state of an ideal static circuit, computed by Qiskit Aer, little-endian like
    :meth:`lucivant_sim.engines.Statevector.vector`."""
    _require("qiskit_aer", "Qiskit Aer")
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    qc = QuantumCircuit(max(circuit.num_qubits, 1))
    phase = 0.0
    for op in circuit.operations:
        if isinstance(op, Gate):
            # Orrery's first qubit is the matrix's most significant bit; Qiskit's is the least.
            qc.unitary(np.asarray(op.matrix, dtype=np.complex128), list(reversed(op.qubits)))
        elif isinstance(op, GlobalPhase):
            phase += op.angle
        elif isinstance(op, (Measure, Barrier, Reset)):
            continue  # static circuit: measurements are final, resets touch untouched qubits
        else:  # pragma: no cover - the caller checks the circuit is static
            raise SimulationError("statevector_aer runs static circuits only")
    qc.global_phase = phase
    qc.save_statevector()
    simulator = AerSimulator(method="statevector", max_parallel_threads=0)
    result = simulator.run(qc, shots=1).result()
    state = np.asarray(result.get_statevector(qc), dtype=np.complex128)
    return state if circuit.num_qubits else state[:1]


def stim_counts(circuit: Circuit, shots: int, seed: int) -> dict[int, int]:
    """Counts (classical bits as integers) of an ideal Clifford circuit, sampled by Stim."""
    stim = _require("stim", "Stim")
    program = stim.Circuit()
    clbit_of_measurement: list[int | None] = []
    for op in circuit.operations:
        if isinstance(op, Gate):
            name = STIM_GATES.get(op.name)
            if name is None:
                raise SimulationError(f"stabilizer_stim runs Clifford gates only; {op.name!r} isn't one")
            program.append(name, list(op.qubits))
        elif isinstance(op, Measure):
            program.append("M", [op.qubit])
            clbit_of_measurement.append(op.clbit)
        elif isinstance(op, Reset):
            program.append("R", [op.qubit])
        elif isinstance(op, (Barrier, GlobalPhase)):
            continue
        else:
            raise SimulationError("stabilizer_stim runs circuits without classical conditions")
    if not clbit_of_measurement or shots == 0:
        return {}
    samples = program.compile_sampler(seed=seed).sample(shots)  # shots x measurements, bool
    # each classical bit holds its last measurement (a later write wins)
    last: dict[int, int] = {}
    for index, clbit in enumerate(clbit_of_measurement):
        if clbit is not None:
            last[clbit] = index
    if not last:
        return {}
    width = max(last) + 1
    columns = np.zeros((shots, width), dtype=np.uint8)  # classical bit i in column i
    sampled = np.asarray(samples)
    for clbit, index in last.items():
        columns[:, clbit] = sampled[:, index]
    rows, counts = np.unique(np.packbits(columns, axis=1, bitorder="little"), axis=0, return_counts=True)
    return {int.from_bytes(row.tobytes(), "little"): int(n) for row, n in zip(rows, counts, strict=True)}


__all__ = ["STIM_GATES", "aer_statevector", "aer_version", "stim_counts", "stim_version"]
