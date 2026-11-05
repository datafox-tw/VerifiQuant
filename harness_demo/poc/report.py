"""G1-G5 numbers for one PoC run, next to the paper's VQ and CoT runs on the same questions.

    python3 -m harness_demo.poc.report --run full      -> runs/full/REPORT.md
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics

from verifiquant.pipeline.run_error_classification_pipeline import _answer_match

from .data import RUN, RUNS, questions, sample_ids

STOP_LABEL = {
    "scope": "1 範圍閘（選卡）", "intent": "2 意圖規格（出處／卡片不適用）", "contract": "3 卡片無不變量",
    "precheck": "4 輸入檢查", "verify": "6 驗證（沒有程式通過）", "ambiguity": "7 歧義（追問或無法區分）",
    "inspect": "8 檢查員不一致", None: "走完全程（輸出）",
}


def _load(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def paper_baselines(ids):
    vq, cot = {}, {}
    for r in _load(RUN / "results/vq_flash/output.jsonl"):
        fg = r["framework_guided"]
        vq[r["case_id"]] = "correct" if fg["final_status"] == "success" and fg["final_is_correct"] else \
            "silent_wrong" if fg["final_status"] == "success" else "abstain"
    for r in _load(RUN / "results/cot_single_shot_flash/output.jsonl"):
        c = r["cot_self_improve"]
        cot[r["case_id"]] = "abstain" if c["final_answer"] is None else \
            "correct" if c["final_is_correct"] else "silent_wrong"
    return {i: vq.get(i) for i in ids}, {i: cot.get(i) for i in ids}


def outcome(row):
    if row["action"] == "EMIT":
        return "correct" if row.get("correct") else "silent_wrong"
    return {"ASK": "ask", "ABSTAIN": "abstain"}.get(row["action"], "error")


def majority_vote(row):
    """CodeT-style answer: the largest cluster among every candidate that ran, ignoring invariants."""
    clusters = row.get("clusters") or []
    if not clusters:
        return None
    best = max(clusters, key=lambda c: len(c["members"]))
    q = questions()[row["question_id"]]
    gold = q["ground_truth"]
    ok = _answer_match(q["question"], float(best["value"]), float(gold))[1] if isinstance(gold, (int, float)) else None
    return {"value": best["value"], "size": len(best["members"]), "correct": ok}


def pct(a, b):
    return f"{a}/{b} ({a / b:.0%})" if b else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="full")
    args = ap.parse_args()
    d = RUNS / args.run
    rows = {r["question_id"]: r for r in _load(d / "results.jsonl")}
    sample = sample_ids()
    ids = [i for i in sample if i in rows] + [i for i in rows if i not in set(sample)]  # a 250 run reports all
    rows = [rows[i] for i in ids]
    grades = {g["question_id"]: g for g in _load(d / "card_grades.jsonl")} if (d / "card_grades.jsonl").exists() else {}
    vq, cot = paper_baselines(ids)
    n = len(rows)
    L = [f"# PoC 結果：harness 在 {n} 題上（run `{args.run}`）", ""]

    # G1 + G2 ---------------------------------------------------------------
    oc = collections.Counter(outcome(r) for r in rows)
    L += ["## G1 泛用性、G2 安全性：三個系統在同一批題目上", "",
          "| 系統 | 正確輸出 | 輸出但錯（silent wrong） | 追問 | 拒答 | 錯誤 |", "|---|---|---|---|---|---|"]
    L.append(f"| harness（自動卡片，單次，不回答追問） | {pct(oc['correct'], n)} | {pct(oc['silent_wrong'], n)} | "
             f"{pct(oc['ask'], n)} | {pct(oc['abstain'], n)} | {oc['error']} |")
    for name, base in (("VQ（論文，公式卡，含模擬使用者多輪）", vq), ("CoT 單次（論文）", cot)):
        c = collections.Counter(base.values())
        L.append(f"| {name} | {pct(c['correct'], n)} | {pct(c['silent_wrong'], n)} | — | {pct(c['abstain'], n)} | |")
    answered = oc["correct"] + oc["silent_wrong"]
    L += ["", f"harness 有輸出時的正確率：{pct(oc['correct'], answered)}。", ""]
    by_level = collections.defaultdict(collections.Counter)
    for r in rows:
        by_level[r["level"]][outcome(r)] += 1
    L += ["| 難度 | 題數 | 正確 | silent wrong | 追問 | 拒答 |", "|---|---|---|---|---|---|"]
    for lv, c in sorted(by_level.items()):
        t = sum(c.values())
        L.append(f"| {lv} | {t} | {c['correct']} | {c['silent_wrong']} | {c['ask']} | {c['abstain']} |")

    # funnel ----------------------------------------------------------------
    stop = collections.Counter(r.get("stopped_at") if r["action"] != "ERROR" else "error" for r in rows)
    L += ["", "### 漏斗：每題停在哪一步", "", "| 停在 | 題數 |", "|---|---|"]
    for k, v in stop.most_common():
        L.append(f"| {STOP_LABEL.get(k, k)} | {v} |")
    sel = [r for r in rows if r.get("selected_card")]
    L += ["", f"選卡正確率（選到出題用的那張卡）：{pct(sum(bool(r.get('retrieval_ok')) for r in sel), n)}；"
          f"選了別張卡：{sum(r.get('retrieval_ok') is False for r in sel)} 題；選 none：{n - len(sel)} 題。"]
    sw = [r for r in rows if outcome(r) == "silent_wrong"]
    issues = json.loads((RUNS.parent / "gold_issues.json").read_text()) if (RUNS.parent / "gold_issues.json").exists() else {}
    if sw:
        L += ["", "### silent wrong 逐題", "", "| 題號 | 輸出 | 標準答案 | 選卡正確 | 卡片完整 | 卡片殺死率 | 人工檢查 |",
              "|---|---|---|---|---|---|---|"]
        for r in sw:
            g = grades.get(r["question_id"], {})
            L.append(f"| {r['question_id']} | {r['value']:.6g} | {r['gold']} | {r.get('retrieval_ok')} | "
                     f"{g.get('complete')} | {g.get('kill_rate')} | {issues.get(r['question_id'], '—')} |")
        n_gold = sum(r["question_id"] in issues for r in sw)
        if n_gold:
            L += ["", f"其中 {n_gold} 題經人工檢查判定為標準答案有誤（上表「人工檢查」欄），表中仍照原始評分計入 silent wrong。"]

    # G3 --------------------------------------------------------------------
    if grades:
        gs = [grades[i] for i in ids if i in grades]
        ok = [g for g in gs if g.get("gradeable")]
        kills = [g["kill_rate"] for g in ok if g.get("kill_rate") is not None]
        family = lambda m: next((p for p in ("off_by_one", "percent_input", "drop_last") if m.startswith(p)), m)
        surv = collections.Counter(family(m) for g in ok for m in g.get("survived", []))
        L += ["", "## G3 約束強度：自動卡片的 VERINA 式評分", "",
              f"可評分 {len(ok)}/{len(gs)} 張（其餘：參考函式回傳多個值，或抽不出輸入）。", "",
              "| 指標 | 自動卡片 | 手寫卡片（展示網站） | 及格線 |", "|---|---|---|---|",
              f"| Complete（正確程式通過全部不變量） | {pct(sum(g['complete'] for g in ok), len(ok))} | 4/4 | ≥ 95% |",
              f"| 不變量對正確程式成立的比例（平均） | {statistics.mean(g['invariant_validity'] for g in ok):.1%} | 100% | |",
              f"| Sound：錯誤變異被抓到的比例（中位數） | {statistics.median(kills):.0%}（平均 {statistics.mean(kills):.0%}） | 約 97% | ≥ 80% |" if kills else "| Sound | — | | |",
              f"| 釘死答案（有 base case 且有關係式） | {pct(sum(g['shape']['pins'] for g in gs), len(gs))} | 3/4 | 報告 |",
              f"| 弱約束（只剩範圍／不等式） | {pct(sum(g['shape']['weak'] for g in gs), len(gs))} | 0/4 | ≤ 30% |", ""]
        if surv:
            L += ["沒被抓到的變異類型：" + "、".join(f"{k} ×{v}" for k, v in surv.most_common()), ""]
        fa = [g for g in ok if not g["complete"]]
        if fa:
            L += ["不完整的卡片（會誤殺正確程式）：", ""]
            for g in fa:
                L.append(f"- {g['question_id']} {g['title']}：誤報 {', '.join(g['false_alarms']) or '—'}"
                         f"{'；約定不合 ' + ', '.join(g['unfit_conventions']) if g['unfit_conventions'] else ''}")
            L.append("")

    # G4 --------------------------------------------------------------------
    asks = [r for r in rows if r["action"] == "ASK"]
    L += ["## G4 追問校準", "", f"追問 {pct(len(asks), n)}。逐題（需人工判斷是否為真歧義）：", ""]
    for r in asks:
        p = r.get("pending") or {}
        what = ", ".join(f["name"] for f in p.get("fields", [])) if p.get("type") == "slots" else \
            ", ".join(c["name"] for c in p.get("conventions", []))
        L.append(f"- {r['question_id']}：{p.get('type')} → {what}；選項中有正確答案：{r.get('ask_has_correct_option')}")
    ung = collections.Counter(s["status"] for r in rows for s in r.get("slots") or [])
    L += ["", "欄位狀態統計：" + "、".join(f"{k} {v}" for k, v in ung.most_common()), ""]

    # G5 --------------------------------------------------------------------
    saves, harmed = [], []
    for r in rows:
        mv = majority_vote(r)
        if not mv:
            continue
        if mv["correct"] is False and outcome(r) in ("correct", "abstain", "ask"):
            saves.append((r, mv))
        if mv["correct"] and outcome(r) in ("abstain", "ask"):
            harmed.append((r, mv))
    mvc = collections.Counter(str(majority_vote(r)["correct"]) for r in rows if majority_vote(r))
    L += ["## G5 不變量的貢獻：跟「多數決」比", "",
          f"只看候選程式多數決（CodeT 式，不管不變量）：對 {mvc['True']}、錯 {mvc['False']}（有候選可投票的 {sum(mvc.values())} 題）。", "",
          f"- 多數決會錯、harness 沒有輸出錯答案：{len(saves)} 題 "
          + ", ".join(f"{r['question_id']}（harness {outcome(r)}）" for r, _ in saves),
          f"- 多數決會對、harness 卻沒輸出（驗證的代價）：{len(harmed)} 題 "
          + ", ".join(f"{r['question_id']}（停在 {r.get('stopped_at')}）" for r, _ in harmed), ""]
    (d / "REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
