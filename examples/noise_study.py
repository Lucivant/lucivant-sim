"""How a GHZ state's fidelity falls as two-qubit error grows, computed exactly (density matrix)."""

import lucivant_sim as ls

GHZ = """
OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q;
h q[0];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
"""

for error in (0.0, 0.005, 0.01, 0.02, 0.05):
    probs = ls.probabilities(GHZ, ls.NoiseModel.uniform(two=error))
    kept = probs.get("0000", 0.0) + probs.get("1111", 0.0)
    print(f"two-qubit error {error:<6} P(0000) + P(1111) = {kept:.4f}")
