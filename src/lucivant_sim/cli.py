"""``lucivant-sim``: run OpenQASM files from the terminal.

lucivant-sim run bell.qasm --shots 2000 --seed 7
lucivant-sim run bell.qasm --noise device.json --json > result.json
lucivant-sim verify result.json bell.qasm --noise device.json
lucivant-sim state bell.qasm
lucivant-sim expect bell.qasm ZZ
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lucivant_sim import __version__
from lucivant_sim.engines import SimulationError
from lucivant_sim.noise import NoiseModel
from lucivant_sim.qasm import QasmError, load, parse
from lucivant_sim.simulator import METHODS, expectation, probabilities, run, statevector, verify


def _program(path: str, params: dict[str, float]):  # type: ignore[no-untyped-def]
    if path == "-":
        return parse(sys.stdin.read(), params)
    return load(Path(path), params)


def _params(items: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--param needs name=value, not {item!r}")
        out[name.strip()] = float(value)
    return out


def _noise(args: argparse.Namespace) -> NoiseModel | None:
    if args.noise:
        return NoiseModel.from_calibration(args.noise)
    if args.single or args.two or args.readout:
        return NoiseModel.uniform(args.single, args.two, args.readout)
    return None


def _bar(p: float, width: int = 32) -> str:
    return "█" * round(p * width)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lucivant-sim", description="Lucivant Orrery: a quantum circuit simulator.")
    parser.add_argument("--version", action="version", version=f"lucivant-sim {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser, noise: bool = True) -> None:
        p.add_argument("program", help="an OpenQASM 2 or 3 file, or - for standard input")
        p.add_argument("--param", action="append", default=[], metavar="NAME=VALUE", help="an input parameter")
        if noise:
            p.add_argument("--noise", metavar="FILE", help="a device calibration (Lucivant twin JSON)")
            p.add_argument("--single", type=float, default=0.0, help="uniform one-qubit gate error")
            p.add_argument("--two", type=float, default=0.0, help="uniform two-qubit gate error")
            p.add_argument("--readout", type=float, default=0.0, help="uniform readout error")

    p_run = sub.add_parser("run", help="sample measurement counts")
    common(p_run)
    p_run.add_argument("--shots", type=int, default=1024)
    p_run.add_argument("--seed", type=int)
    p_run.add_argument("--method", choices=METHODS, default="automatic")
    p_run.add_argument("--json", action="store_true", help="print the full result as JSON")

    p_state = sub.add_parser("state", help="exact probabilities (and amplitudes for ideal circuits)")
    common(p_state)
    p_state.add_argument("--amplitudes", action="store_true")

    p_exp = sub.add_parser("expect", help="exact expectation of a Pauli observable")
    common(p_exp)
    p_exp.add_argument("observable", help='a Pauli string such as ZZ, or JSON like {"ZI": 0.5, "XX": 1}')

    p_ver = sub.add_parser("verify", help="rerun a saved --json result and check it matches")
    p_ver.add_argument("result", help="the saved result JSON")
    common(p_ver)

    args = parser.parse_args(argv)
    try:
        params = _params(args.param)
        program = _program(args.program, params)
        noise = _noise(args)
        if args.command == "run":
            result = run(program, args.shots, seed=args.seed, noise=noise, method=args.method)
            if args.json:
                print(json.dumps(result.to_dict(), indent=2))
                return 0
            print(f"{result.num_qubits} qubits · {result.shots} shots · {result.method} · seed {result.seed}")
            for key, n in sorted(result.counts.items(), key=lambda kv: (-kv[1], kv[0])):
                print(f"  {key or '(no bits)'}  {n:>8}  {_bar(n / max(result.shots, 1))}")
            if not result.counts:
                print("  The program measures nothing; try `lucivant-sim state`.")
            print(f"fingerprint {result.fingerprint}")
        elif args.command == "state":
            probs = probabilities(program, noise)
            for key, p in sorted(probs.items(), key=lambda kv: -kv[1])[:64]:
                print(f"  {key}  {p:.6f}  {_bar(p)}")
            if args.amplitudes:
                if noise is not None and noise.has_gate_errors():
                    raise SimulationError("amplitudes exist only for ideal circuits")
                vec = statevector(program)
                n = max(1, (len(vec) - 1).bit_length())
                for i, a in enumerate(vec):
                    if abs(a) > 1e-12:
                        print(f"  |{format(i, f'0{n}b')}>  {a.real:+.6f}{a.imag:+.6f}i")
        elif args.command == "expect":
            text = args.observable.strip()
            observable = json.loads(text) if text.startswith("{") else text
            print(f"{expectation(program, observable, noise):.10g}")
        else:
            ok = verify(args.result, program, noise)
            print("verified: the rerun matches the saved counts" if ok else "MISMATCH: the rerun differs")
            return 0 if ok else 1
    except (QasmError, SimulationError, ValueError, OSError, TypeError) as exc:
        print(f"lucivant-sim: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
