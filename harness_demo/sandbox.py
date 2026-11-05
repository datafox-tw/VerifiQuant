"""Run candidate code out of process. The harness never execs LLM code in its own interpreter."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

RUNNER = Path(__file__).with_name("_runner.py")


def run(code: str, inputs: dict, invariants: list[dict] | None = None, extra: dict | None = None,
        timeout: float = 5.0, allow: list[str] | None = None, path: list | None = None,
        mutate: bool = False) -> dict:
    """path: pick the result out of a structured return value; mutate: run the element probe."""
    job = {"code": code, "inputs": inputs, "invariants": invariants or [], "extra": extra or {},
           "allow": allow or [], "path": path, "mutate": mutate}
    try:
        proc = subprocess.run([sys.executable, "-I", str(RUNNER)], input=json.dumps(job),
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "result": None, "error": "Timeout: 程式執行超過時間上限", "checks": {}}
    if proc.returncode != 0 or not proc.stdout.strip():
        err = (proc.stderr.strip().splitlines() or ["程式異常終止"])[-1]
        return {"ok": False, "result": None, "error": err, "checks": {}}
    return json.loads(proc.stdout)


def eval_pre(invariants: list[dict], inputs: dict) -> dict:
    """Input-contract checks need no candidate; run them through the same sandbox anyway."""
    code = "def solve(**kw):\n    return 0.0\n"
    return run(code, inputs, invariants)["checks"]
