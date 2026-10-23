OPENQASM 3.0;
include "stdgates.inc";
// Teleport ry(1.234)|0> from q[0] to q[2] with mid-circuit measurement and corrections.
qubit[3] q;
bit[2] m;
bit[1] out;
ry(1.234) q[0];
h q[1];
cx q[1], q[2];
cx q[0], q[1];
h q[0];
m[0] = measure q[0];
m[1] = measure q[1];
if (m[1]) x q[2];
if (m[0]) z q[2];
out[0] = measure q[2];
