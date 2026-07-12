"""
compare_trigger_runs.py — canonical (triggers OFF) vs VQ_HINT_TRIGGERS=1 對照。

輸出:
1. 分層三元對照表(medium/hard/all)
2. 逐案 flip 清單:canonical correct → triggers 非 correct(regression),
   canonical SW/abstain → triggers correct(improvement),SW→flagged(轉性)
3. 觸發統計:fires / recall misses / hard vs soft / 誤觸(觸發且 regression)

用法:
    python3 scripts/compare_trigger_runs.py \
        --questions verifiquant/data/runs/paper_v2_250/questions_250.jsonl \
        --base verifiquant/data/runs/paper_v2_250/results/vq_flash/output.jsonl \
        --trig verifiquant/data/runs/paper_v2_250/results/vq_flash_250q_triggers/output.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

HARD_MIN = 4.15


def load(path: str) -> dict[str, dict]:
    out = {}
    for l in Path(path).read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        out[r["case_id"]] = r["framework_guided"]
    return out


def cls(fg: dict) -> str:
    if fg.get("final_is_correct") is True:
        return "correct"
    return "sw" if fg.get("final_status") == "success" else "abstain"


def has_soft(fg: dict) -> bool:
    return any(t.get("diagnostic", {}).get("has_i_soft") for t in fg.get("history", []))


def tern(run: dict[str, dict], qd: dict[str, float], tier: str | None) -> Counter:
    c: Counter = Counter()
    for cid, fg in run.items():
        t = "hard" if qd[cid] >= HARD_MIN else "medium"
        if tier and t != tier:
            continue
        c[cls(fg)] += 1
    return c


def row(name: str, c: Counter) -> str:
    n = sum(c.values())
    sw, cor, ab = c["sw"], c["correct"], c["abstain"]
    swr = sw / n * 100 if n else 0
    return f"  {name:<18} N={n:<4} correct={cor:<4} SW={sw:<3} abstain={ab:<3} SWR={swr:.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--trig", required=True)
    args = ap.parse_args()

    qd = {json.loads(l)["question_id"]: float(json.loads(l)["difficulty"])
          for l in Path(args.questions).read_text().splitlines() if l.strip()}
    base, trig = load(args.base), load(args.trig)

    print("=== 分層三元對照(base=canonical / trig=hint-triggers ON)===")
    for tier in ("medium", "hard", None):
        label = tier or "all"
        print(f"[{label}]")
        print(row("canonical", tern(base, qd, tier)))
        print(row("triggers", tern(trig, qd, tier)))

    print("\n=== 逐案 flips ===")
    regress, improve, sw_to_flagged, other = [], [], [], []
    for cid in base:
        b, t = cls(base[cid]), cls(trig.get(cid, {}))
        if b == t:
            if b == "sw" and not has_soft(base[cid]) and has_soft(trig[cid]):
                sw_to_flagged.append(cid)
            continue
        tier = "hard" if qd[cid] >= HARD_MIN else "med"
        if b == "correct":
            regress.append((cid, tier, f"correct→{t}"))
        elif t == "correct":
            improve.append((cid, tier, f"{b}→correct"))
        else:
            other.append((cid, tier, f"{b}→{t}"))
    print(f"REGRESSIONS (correct→non-correct): {len(regress)}")
    for x in regress: print("   ", x)
    print(f"IMPROVEMENTS (→correct): {len(improve)}")
    for x in improve: print("   ", x)
    print(f"SW→SW 但轉 flagged(silent→disclosed): {len(sw_to_flagged)}  {sw_to_flagged}")
    print(f"其他變動: {other}")

    print("\n=== 觸發統計(triggers run)===")
    fires = misses = hard_fires = 0
    fired_cases = set()
    for cid, fg in trig.items():
        for tu in fg.get("history", []):
            ct = (tu.get("diagnostic") or {}).get("critic_trace") or {}
            det = ct.get("deterministic_hint_triggers") or []
            if det:
                fires += len(det)
                misses += len(ct.get("critic_recall_misses") or [])
                hard_fires += sum(1 for d in det if d.get("i_level") == "hard")
                fired_cases.add(cid)
    print(f"fires={fires} on {len(fired_cases)} cases | recall_misses={misses} | hard fires={hard_fires}")
    reg_fired = [c for c, _, _ in regress if c in fired_cases]
    print(f"regression 中曾觸發者(疑似誤觸傷害): {reg_fired}")

    print("\n=== 舊 D 類三題(canonical SW)在 triggers run 的下場 ===")
    for cid in ("test-1004", "test-1063", "test-1771"):
        fg = trig.get(cid)
        if fg:
            print(f"  {cid}: {cls(fg)} (soft_flag={has_soft(fg)}, fired={cid in fired_cases})")


if __name__ == "__main__":
    main()
