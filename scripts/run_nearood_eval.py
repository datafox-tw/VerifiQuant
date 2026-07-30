"""
run_nearood_eval.py
-------------------
P2 retrieval near-OOD evaluation. Runs VerifiQuant (K=1) and/or the plain
Retrieval-PAL baseline over the frozen near-OOD probe set under two conditions:

  bare      target family + nearest neighbors removed  -> cleanest abstain test
  nearmiss  only the target family removed             -> neighbors bait force-fit

Correct behavior is a scope refusal (VQ: N / abstain; R-PAL: "CANNOT SOLVE").
The headline metric is the *unsafe force-fit rate*: a confident numeric answer
computed against a wrong, adjacent function. Hypothesis: VQ's scope gate keeps
force-fit below R-PAL's, and both conditions are harder than in-library, with
`nearmiss` harder than `bare`.

The same ExclusionStore ablates the library identically for both systems.

`--dry-run` validates the exclusion/retrieval path with ZERO model calls.

Usage (dry, offline):
    python3 scripts/run_nearood_eval.py --system rpal --condition both \
        --probes verifiquant/data/runs/nearood/probes.jsonl \
        --db verifiquant/data/runs/paper_v2_250/fic/cards.db --dry-run

Usage (real; needs GEMINI_API_KEY, run after P1):
    python3 scripts/run_nearood_eval.py --system vq --condition both \
        --probes verifiquant/data/runs/nearood/probes.jsonl \
        --db verifiquant/data/runs/paper_v2_250/fic/cards.db \
        --out-dir verifiquant/data/runs/nearood --run-name vq_run1
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_pal_baseline import deny_reason, extract_code, run_code  # noqa: E402
from run_retrieval_pal_baseline import (  # noqa: E402
    DEFAULT_FUNCTIONS,
    _candidate_payloads,
    _generate_with_retries,
    _load_function_library,
    _prompt_for,
)
from verifiquant.pipeline.nearood_common import ExclusionStore, exclusion_for  # noqa: E402

CONDITIONS = ("bare", "nearmiss")


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


# --------------------------------------------------------------------------- #
# Outcome classifiers
# --------------------------------------------------------------------------- #
def classify_rpal(text: str, code: str | None) -> Dict[str, Any]:
    if not code:
        return {"outcome": "refused", "reason": "no_code", "value": None}
    deny = deny_reason(code)
    if deny:
        return {"outcome": "refused", "reason": deny, "value": None}
    result = run_code(code)
    if result is None or "CANNOT SOLVE" in str(result).upper():
        return {"outcome": "refused", "reason": "cannot_solve_or_error", "value": None}
    return {"outcome": "forcefit", "reason": "numeric", "value": str(result)}


COT_PROMPT = (
    "You are a financial analyst. Solve the question using only the given context.\n"
    "If the question requires a formula, method, or instrument you cannot reliably "
    "apply from the given information, do not guess.\n"
    "Think briefly, then end with exactly one line:\n"
    "  ANSWER: <number>        (your final numeric answer), or\n"
    "  ANSWER: CANNOT SOLVE    (if you cannot reliably answer).\n\n"
    "Question:\n{question}\n\nContext:\n{context}\n"
)

_ANS_RE = __import__("re").compile(r"ANSWER:\s*(.+)\s*$", __import__("re").MULTILINE)
_NUM_RE = __import__("re").compile(r"[-+]?\d[\d,]*\.?\d*")


def classify_cot(text: str) -> Dict[str, Any]:
    m = _ANS_RE.findall(text or "")
    tail = (m[-1] if m else (text or "")).strip()
    if "CANNOT SOLVE" in tail.upper():
        return {"outcome": "refused", "reason": "cannot_solve", "value": None}
    num = _NUM_RE.search(tail) or _NUM_RE.search(text or "")
    if num:
        return {"outcome": "forcefit", "reason": "numeric", "value": num.group(0)}
    return {"outcome": "refused", "reason": "no_number", "value": None}


def classify_vq(report: Dict[str, Any]) -> Dict[str, Any]:
    status = report.get("status")
    dt = report.get("diagnostic_type")
    if status == "success":
        return {"outcome": "forcefit", "reason": "numeric",
                "diagnostic_type": dt, "selected_fic": report.get("fic_id")}
    return {"outcome": "safe", "reason": status,
            "diagnostic_type": dt, "ideal_scope_refusal": dt == "N",
            "selected_fic": report.get("fic_id")}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(system: str, conditions: List[str], probes_path: str, db_path: str,
        top_k: int, model: str, out_dir: str, run_name: str,
        dry_run: bool, limit: int) -> Dict[str, Any]:
    probes = _load_jsonl(probes_path)
    if limit:
        probes = probes[:limit]
    store = ExclusionStore(f"sqlite:///{db_path}")

    client = None
    api = None
    function_by_id: Dict[str, Any] = {}
    core_by_id: Dict[str, Any] = {}
    if not dry_run:
        from verifiquant.pipeline.run_error_classification_pipeline import (
            create_genai_client_from_env, ErrorClassificationAPI,
        )
        client = create_genai_client_from_env()
        if system == "vq":
            api = ErrorClassificationAPI(
                core_by_id=store.load_core_by_id(),
                retrieval_cards=[],
                store=store,
                repair_index=store.build_repair_index(),
                client=client,
                top_k=top_k,
            )
        elif system == "rpal":
            function_by_id = _load_function_library(DEFAULT_FUNCTIONS)
            core_by_id = store.load_core_by_id()

    # CoT has no library, so the bare/near-miss ablation does not apply: run once.
    if system == "cot":
        conditions = ["ood"]

    records: List[Dict[str, Any]] = []
    for cond in conditions:
        for p in probes:
            if system != "cot":
                store.current_exclude = exclusion_for(cond, p["target_fic"], p["neighbors"])
            q, c = p["question"], p["context"]
            rec: Dict[str, Any] = {"probe_id": p["probe_id"], "condition": cond,
                                   "system": system, "target_fic": p["target_fic"]}
            if dry_run:
                rows = store.retrieve_candidates(query=f"{q}\n{c}", top_k=top_k)
                cand = [r["fic_id"] for r in rows]
                rec.update({
                    "dry_run": True,
                    "candidates_after_exclusion": cand,
                    "target_excluded": p["target_fic"] not in cand,
                    "neighbors_excluded": not any(n in cand for n in p["neighbors"]),
                })
            elif system == "cot":
                prompt = COT_PROMPT.format(question=q, context=c)
                text, _status = _generate_with_retries(client, model=model, prompt=prompt, max_retries=3)
                rec.update(classify_cot(text))
            elif system == "vq":
                report = api.diagnose(question=q, context=c, case_id=p["probe_id"], top_k=top_k)
                rec.update(classify_vq(report))
            else:
                candidates = _candidate_payloads(
                    store=store, core_by_id=core_by_id, function_by_id=function_by_id,
                    question=q, context=c, top_k=top_k)
                prompt = _prompt_for({"question": q, "context": c}, candidates)
                text, _status = _generate_with_retries(client, model=model, prompt=prompt, max_retries=3)
                code = extract_code(text)
                res = classify_rpal(text, code)
                res["candidate_used"] = candidates[0]["fic_id"] if candidates else None
                rec.update(res)
            records.append(rec)

    # summary: force-fit rate per condition
    summary: Dict[str, Any] = {"system": system, "n_probes": len(probes),
                               "dry_run": dry_run, "by_condition": {}}
    for cond in conditions:
        sub = [r for r in records if r["condition"] == cond]
        if dry_run:
            ok = sum(1 for r in sub if r["target_excluded"]
                     and (cond != "bare" or r["neighbors_excluded"]))
            summary["by_condition"][cond] = {"n": len(sub), "exclusion_valid": ok}
        else:
            oc = Counter(r["outcome"] for r in sub)
            ff = oc.get("forcefit", 0)
            summary["by_condition"][cond] = {
                "n": len(sub), "forcefit": ff,
                "forcefit_rate": round(ff / len(sub), 4) if sub else None,
                "outcomes": dict(oc),
            }

    if not dry_run:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        tag = run_name or f"{system}_{datetime.now():%H%M%S}"
        with open(out / f"nearood_{tag}.jsonl", "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(out / f"nearood_{tag}_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Retrieval near-OOD eval (P2).")
    ap.add_argument("--system", choices=("vq", "rpal", "cot"), required=True)
    ap.add_argument("--condition", choices=("bare", "nearmiss", "both"), default="both")
    ap.add_argument("--probes", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--out-dir", default="verifiquant/data/runs/nearood")
    ap.add_argument("--run-name", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    conds = list(CONDITIONS) if args.condition == "both" else [args.condition]
    run(args.system, conds, args.probes, args.db, args.top_k, args.model,
        args.out_dir, args.run_name, args.dry_run, args.limit)


if __name__ == "__main__":
    main()
