# Lucivant Orrery

Orrery is published as the `lucivant-sim` package (`import lucivant_sim`). Its command is `lucivant-sim`, also installed as `orrery`.

An open-source quantum circuit simulator from [Lucivant](https://lucivant.com). It reads
OpenQASM 2 and 3 directly, needs nothing but NumPy, and every result can be repeated exactly
and checked by anyone.

```bash
pip install lucivant-sim
```

```python
import lucivant_sim as ls

bell = """
OPENQASM 3;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c = measure q;
"""

result = ls.run(bell, shots=1000, seed=7)
print(result.counts)  # {'00': 502, '11': 498}
print(result.fingerprint)  # identifies program, noise, shots, seed, method and version
```

## What it does

| | |
|---|---|
| **Reads OpenQASM directly** | Its own OpenQASM 2.0 and 3 reader: `qreg`/`qubit`, qelib1.inc and stdgates.inc, user-defined `gate`s, `ctrl @`, `negctrl @`, `inv @`, `pow(k) @`, `input` parameters, register broadcasting and slices, `measure`, `reset`, `barrier`, `gphase`, `box`, and `if`/`else` on measurement results. Unsupported constructs (loops, subroutines, pulse calibration) are refused with the line and column, never guessed. |
| **Three engines, chosen for you** | Exact **statevector** for ideal circuits; exact **density matrix** for noisy ones; per-shot **trajectories** for circuits that use measurement results mid-circuit, or are too wide for a density matrix. |
| **A stabilizer engine for wide Clifford circuits** | Clifford circuits of 12 qubits or more use it automatically, and `method="stabilizer"` simulates Clifford circuits (H, S, the Paulis, CX, CZ, SWAP, and any gate that maps Paulis to Paulis) exactly with a tableau, up to 2,048 qubits, with measurement, reset, conditions, Pauli noise and readout errors. Any other gate is refused by name; it never approximates. `ls.is_clifford(circuit)` tells you beforehand. |
| **Noise** | Depolarising, Pauli, bit- and phase-flip, amplitude and phase damping, thermal relaxation (T1/T2) and readout errors, attached per gate, per qubit or per coupler. |
| **Device noise from calibrations** | `NoiseModel.from_calibration()` reads a device calibration in Lucivant's twin format, with per-qubit and per-coupler errors where the maker publishes them. |
| **Exact answers** | `statevector()`, `density_matrix()`, `probabilities()` and `expectation()` of Pauli observables, with or without noise. |
| **Reproducible and verifiable** | Every run is seeded (a seed is drawn and recorded if you don't give one). The same program, noise, shots, seed, method and version give the same counts. `verify()` and `lucivant-sim verify` rerun a saved result and check it. |
| **Works with what you use** | Pass OpenQASM text, a `.qasm` file, a Qiskit `QuantumCircuit` or a Cirq `Circuit`. |

## Command line

```bash
lucivant-sim run bell.qasm --shots 2000 --seed 7
lucivant-sim run bell.qasm --single 1e-3 --two 1e-2 --readout 2e-2
lucivant-sim run bell.qasm --noise device.json --json > result.json
lucivant-sim verify result.json bell.qasm --noise device.json
lucivant-sim state bell.qasm --amplitudes
lucivant-sim expect bell.qasm ZZ
lucivant-sim expect ansatz.qasm '{"ZI": 0.5, "XX": -1}' --param theta=0.3
```

## Noise

```python
noise = ls.NoiseModel.uniform(single=1e-3, two=1e-2, readout=2e-2)

noise = (
    ls.NoiseModel()
    .add_gate_error(ls.depolarizing(0.002), gates=["sx", "x"])
    .add_gate_error(ls.from_gate_error(0.012, 2), gates="cx", qubits=[0, 1])
    .add_gate_error(ls.thermal_relaxation(t1=120.0, t2=90.0, time=0.05), gates="h")
    .add_readout_error(ls.ReadoutError(p01=0.01, p10=0.03), qubits=[0])
)

noise = ls.NoiseModel.from_calibration("device.json")

result = ls.run(bell, shots=4000, seed=1, noise=noise)
```

The most specific rule wins: a gate on given qubits, then any gate on those qubits, then a gate
anywhere, then any gate of that size. Frame-change gates (`rz`, `p`, `u1`, `id`) are error-free
unless a rule names them, as on real devices.

**What it doesn't model:** Lucivant Orrery applies noise to the gates your circuit contains. It
doesn't compile to a device's native gates and couplers, and it doesn't model idle decoherence,
crosstalk, leakage or drift. For a faithful emulation of a specific device, compile your circuit
for it first, or use a device twin in [Lucivant Quantum Studio](https://lucivant.com/studio),
which compiles for the device and compares its predictions with the device's real results.

## Conventions

Bitstrings and statevector indices are little-endian: qubit 0 is the rightmost bit, and several
classical registers print last-declared first, separated by spaces, as OpenQASM tools do. A
k-qubit gate matrix in `lucivant_sim.gates` is written with its first argument as the most
significant bit.

## Limits

The statevector engine handles up to 26 qubits and the density matrix up to 13 (about 1 GiB
each). Trajectories cost one statevector simulation per shot.

## Checked against Qiskit

The test suite compares Lucivant Orrery's statevectors with Qiskit's on every supported gate and on
random circuits, its noisy probabilities with exact channel algebra, and its sampled counts with
the exact distributions. Run it with `pytest`.

## Licence

Apache-2.0. Copyright 2026 Lucivant Technologies Pty Ltd (ACN 703 268 571).
