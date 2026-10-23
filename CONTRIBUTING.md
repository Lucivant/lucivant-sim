# Contributing to Lucivant Orrery

Issues and pull requests are welcome. Before you open a pull request:

1. `pip install -e ".[qiskit,cirq]" qiskit-aer pytest ruff`
2. `ruff check . && ruff format --check . && pytest`
3. New gates or features come with a test against an independent reference (Qiskit, exact
   algebra or a known analytic result), as the existing tests do.

By contributing you agree that your contribution is licensed under Apache-2.0, and you sign off
each commit (`git commit -s`) to certify the Developer Certificate of Origin
(https://developercertificate.org). Security issues: see SECURITY.md in github.com/lucivant/.github,
never a public issue.
