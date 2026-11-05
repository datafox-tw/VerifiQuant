"""Child process: run one untrusted candidate program and the card's invariants.

Reads a JSON job on stdin, writes a JSON report on stdout. Started with
`python -I` by sandbox.py, under a CPU limit and a wall-clock timeout.
"""

import json
import math
import resource
import sys

resource.setrlimit(resource.RLIMIT_CPU, (2, 2))

ALLOWED_MODULES = {"math"}


def _import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".")[0] not in ALLOWED_MODULES:
        raise ImportError(f"import of '{name}' is not allowed in the sandbox")
    return __import__(name, globals, locals, fromlist, level)


SAFE_BUILTINS = {
    n: __builtins__[n] if isinstance(__builtins__, dict) else getattr(__builtins__, n)
    for n in ["abs", "min", "max", "sum", "range", "len", "float", "int", "round", "pow", "enumerate",
              "zip", "list", "tuple", "dict", "isinstance", "ValueError", "ZeroDivisionError",
              "Exception", "True", "False", "None", "bool", "str", "any", "all", "reversed", "sorted"]
}
SAFE_BUILTINS["__import__"] = _import


def close(a, b, rel=1e-6, abs_tol=1e-9):
    # structured outputs (schedules, bands) compare element by element
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(close(x, y, rel, abs_tol) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k], rel, abs_tol) for k in a)
    return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=rel, abs_tol=abs_tol)


def pick(value, path):
    for k in path:
        value = value[k]
    return value


def bump(value, path):
    """Copy of value with the number at path moved by 0.1% (by 1e-3 when it is zero)."""
    if not path:
        return value + (abs(value) * 1e-3 if value else 1e-3)
    k, rest = path[0], path[1:]
    if isinstance(value, dict):
        return {**value, k: bump(value[k], rest)}
    out = list(value)
    out[k] = bump(value[k], rest)
    return tuple(out) if isinstance(value, tuple) else out


def amortize(principal, monthly_rate, months, payment):
    balance = principal
    for _ in range(int(months)):
        balance = balance * (1 + monthly_rate) - payment
    return balance


def main():
    job = json.loads(sys.stdin.read())
    ALLOWED_MODULES.update(job.get("allow", []))  # only trusted reference code is given extra modules
    # path: the program returns a structure and the question asks for the number at this path
    path = job.get("path")
    out = {"ok": True, "result": None, "error": None, "checks": {}}
    ns = {"__builtins__": SAFE_BUILTINS, "__name__": "candidate"}
    try:
        exec(compile(job["code"], "<candidate>", "exec"), ns)
        f = ns["solve"]
        if job.get("mutate"):  # element probe: every call of f has the asked number moved by 0.1%
            g = f
            f = lambda *a, **k: bump(g(*a, **k), path or [])
        full = f(**job["inputs"])
        if job.get("raw"):  # card self-check: the invariants see whatever the reference returns, nothing is projected
            result = full
        else:
            value = float(pick(full, path) if path is not None else full)
            if not math.isfinite(value):
                raise ValueError("result is not finite")
            out["result"] = value
            result = full if path is not None else value
        if path is not None:
            out["preview"] = json.dumps(full, default=str)[:300]
    except Exception as exc:  # the candidate failed to even run
        out.update(ok=False, error=f"{type(exc).__name__}: {exc}")
        print(json.dumps(out))
        return

    env = {"f": f, "close": close, "math": math, "amortize": amortize, "result": result, **job["inputs"]}
    env.update(job.get("extra", {}))
    for inv in job["invariants"]:
        try:
            # one globals dict, so generator expressions inside a check can see the inputs
            ok = bool(eval(inv["check"], {"__builtins__": SAFE_BUILTINS, **env}))
            out["checks"][inv["id"]] = {"ok": ok, "error": None}
        except Exception as exc:
            out["checks"][inv["id"]] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(out))


main()
