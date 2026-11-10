#!/usr/bin/env python3
"""
build_paper_demo.py
-------------------
Build static/data/paper_replay.json, the data behind the website's "Paper replay" demo.
Nothing here calls a model: every row is a recorded output from the runs the ICAIF '26
camera-ready reports, scored with the same normalizers and scorer as the paper's tables
(scripts/analyze_governance_eval.py).

    traps   the 80 F/E/I/N traps (datasets/03_traps_80/trap_set.jsonl)
              VerifiQuant K=1   governance_confirmatory_1014/vq_k1_top4_run{1,2,3}.jsonl
              R-PAL+            governance_confirmatory/conf_prompted_run{1,2,3}.jsonl
              R-PAL             governance_confirmatory/conf_plain_run1.jsonl
    clean   three of the 250 clean questions, used for the M, C and coverage-cost stories
              VerifiQuant K=1   paper_v2_250/results/vq_flash/output.jsonl (history[0])
              R-PAL             retrieval_pal_baseline/rpal_top4_gemini_25_flash_audited.jsonl

Before writing, it asserts that the aggregates equal the paper's printed numbers (trap FNAR
10.4 / 27.5 / 48.8, clean K=1 81.2 / 2.8 / 84.0, R-PAL 96.8 / 2.4 / 99.2), so the replay
cannot drift from the tables.

    PYTHONPATH=. .venv/bin/python scripts/build_paper_demo.py
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from scripts.analyze_governance_eval import (  # noqa: E402
    clean_labels_from_rpal, clean_labels_from_vq, clean_summary,
    load_jsonl, normalize_rpal, normalize_vq, score_traps,
)

RUNS = ROOT / "verifiquant/data/runs"
G = RUNS / "governance_confirmatory"
R = RUNS / "governance_confirmatory_1014"
TRAPS = ROOT / "datasets/03_traps_80/trap_set.jsonl"
CARDS_DB = RUNS / "paper_v2_250/fic/cards.db"
CLEAN_QUESTIONS = RUNS / "paper_v2_250/questions_250.jsonl"
CLEAN_VQ = RUNS / "paper_v2_250/results/vq_flash/output.jsonl"
CLEAN_RPAL = RUNS / "retrieval_pal_baseline/rpal_top4_gemini_25_flash_audited.jsonl"
OUT = ROOT / "static/data/paper_replay.json"

SYSTEMS = {
    "vq": [R / f"vq_k1_top4_run{i}.jsonl" for i in (1, 2, 3)],
    "rpal_plus": [G / f"conf_prompted_run{i}.jsonl" for i in (1, 2, 3)],
    "rpal": [G / "conf_plain_run1.jsonl"],
}
PAPER_FNAR = {"vq": 0.104, "rpal_plus": 0.275, "rpal": 0.488}

CODE_LIMIT = 2400
TEXT_LIMIT = 600

# The story the demo walks through, in order. Captions are written to stay inside what the
# paper shows (see the calibration notes in docs and the paper's section 5). The *_zh fields
# are the Traditional Chinese interface's versions of the same captions.
FEATURED: List[Dict[str, str]] = [
    {
        "id": "E-test-1080", "set": "trap", "chapter": "The problem",
        "title": "A program that runs is not an answer",
        "caption": "One input was changed so that no valid number exists: the previous price is $2.5, not $150. "
                   "Both baselines see the same four contracts, write code that runs, and report a 6,500% weekly gain "
                   "in every run. VerifiQuant binds the two prices to the contract's inputs, and the contract's declared "
                   "boundary check fires before anything is computed.",
        "chapter_zh": "問題所在", "title_zh": "程式跑得起來，不代表答案成立",
        "caption_zh": "我們只改了一個輸入，讓這題沒有合理答案：週初股價從 $150 改成 $2.5。兩個基線看到同樣的四張合約，寫出能執行的程式，每一次都回報這週漲了 6,500%。VerifiQuant 把兩個價格填進合約的輸入欄位，在任何計算之前，合約宣告的邊界檢查就先擋下來了。",
    },
    {
        "id": "F-test-1023", "set": "trap", "chapter": "Exit F",
        "title": "Missing input: ask for it by name",
        "caption": "The tax liability was redacted. VerifiQuant's input gate finds the contract's required field empty "
                   "and asks for exactly that field. R-PAL+ asks for it too: a missing field is conspicuous enough for "
                   "a prompt to catch, which is why F and N show little gap in the paper.",
        "chapter_zh": "出口 F", "title_zh": "少了輸入：直接講出缺的欄位",
        "caption_zh": "稅額被遮掉了。VerifiQuant 的輸入關卡發現合約的必要欄位是空的，就只問那一個欄位。R-PAL+ 這題也會問。缺欄位這種問題夠顯眼，寫在提示裡就擋得下來，這也是論文裡 F 和 N 差距很小的原因。",
    },
    {
        "id": "I-test-2163", "set": "trap", "chapter": "Exit I (hard)",
        "title": "An open convention that changes the number: stop and ask",
        "caption": "The word \"call\" was removed, so the option type is no longer stated. The contract declares this "
                   "ambiguity as HARD, so VerifiQuant shows the contract's options instead of a number. R-PAL+ assumes "
                   "and answers in all three runs.",
        "chapter_zh": "出口 I（hard）", "title_zh": "沒講清楚的慣例會改變數字：先停下來問",
        "caption_zh": "我們刪掉 call 這個字，題目就沒說是哪一種選擇權。合約把這個歧義宣告成 HARD，所以 VerifiQuant 改成列出合約的選項，不給數字。R-PAL+ 三次都自己假設了一種，直接作答。",
    },
    {
        "id": "I-test-1026", "set": "trap", "chapter": "Exit I (soft)",
        "title": "An open convention that changes the reading: answer and warn",
        "caption": "\"As a percentage\" was removed. The contract declares output scale as a SOFT ambiguity, so the "
                   "number goes out with a warning that names the assumption (decimal). R-PAL+ answers in all three "
                   "runs with no warning (0.06 twice, 6.0 once), which the paper counts as a false acceptance.",
        "chapter_zh": "出口 I（soft）", "title_zh": "只影響解讀的慣例：照樣作答，附上警告",
        "caption_zh": "題目裡的 as a percentage 被刪掉了。合約把輸出的單位宣告成 SOFT 歧義，所以數字照樣給出，同時附上警告，寫明它假設的是小數。R-PAL+ 三次都直接作答（兩次 0.06、一次 6.0），沒有任何警告，論文把這算成誤收。",
    },
    {
        "id": "N-test-1006", "set": "trap", "chapter": "Exit N",
        "title": "Out of scope: say so instead of guessing",
        "caption": "The question was replaced with a Value-at-Risk ask that no contract covers. VerifiQuant's selector "
                   "finds no applicable contract among the four retrieved and refuses. R-PAL+ also refuses: a blatant "
                   "out-of-scope ask is easy to spot.",
        "chapter_zh": "出口 N", "title_zh": "超出範圍：說找不到，不要用猜的",
        "caption_zh": "問題被換成問風險值（VaR），沒有任何合約涵蓋。VerifiQuant 的選擇器在檢索到的四張合約裡找不到適用的，就拒答。R-PAL+ 這題也拒答，這麼明顯的超範圍問題本來就容易看出來。",
    },
    {
        "id": "test-2231", "set": "clean", "chapter": "Exit M",
        "title": "Unclear intent: which model did you mean?",
        "caption": "A well-posed question from the clean set. Two retrieved contracts price a European call (Black-Scholes "
                   "and Heston), and the question does not say which, so VerifiQuant asks. R-PAL picks Black-Scholes and "
                   "matches the gold answer. On clean questions an M exit is a cost, not a catch.",
        "chapter_zh": "出口 M", "title_zh": "意圖不明：你要的是哪一個模型？",
        "caption_zh": "這是乾淨題裡一題定義清楚的問題。檢索到的合約裡有兩張都能算歐式買權的價格（Black-Scholes 和 Heston），題目沒說用哪一個，所以 VerifiQuant 開口問。R-PAL 選了 Black-Scholes，答案和標準答案一致。在乾淨題上停在 M 出口是一種成本，並沒有攔下任何錯誤。",
    },
    {
        "id": "test-1969", "set": "clean", "chapter": "Exit C",
        "title": "Every gate passes: the contract's own code computes",
        "caption": "Inputs complete and within bounds. The contract's Python returns 579.98, which matches the gold "
                   "answer, and the declared SOFT convention (payment timing) still travels with the number as a warning.",
        "chapter_zh": "出口 C", "title_zh": "每一關都通過：由合約自己的程式碼計算",
        "caption_zh": "輸入齊全，也都在範圍內。合約的 Python 算出 579.98，和標準答案一致。合約宣告的 SOFT 慣例（付款時點）仍然以警告的形式，跟著數字一起給出。",
    },
    {
        "id": "F-test-1263", "set": "trap", "chapter": "Where it fails",
        "title": "A trap that gets through",
        "caption": "The share count was redacted, but the contract marks that input optional with a default of 100, so "
                   "the input gate has nothing to flag and VerifiQuant computes with the default. R-PAL+ asks. The paper "
                   "reports this: the two F acceptances come from traps that break the F definition.",
        "chapter_zh": "失效的地方", "title_zh": "一題被放過的陷阱題",
        "caption_zh": "股數被遮掉了，但合約把這個輸入標成選填、預設 100 股，所以輸入關卡沒有東西可以攔，VerifiQuant 就用預設值算了。R-PAL+ 則開口問。論文有寫到這一點：VerifiQuant 的兩題 F 誤收，都來自這種不符合 F 定義的陷阱題。",
    },
    {
        "id": "test-1993", "set": "clean", "chapter": "What it costs",
        "title": "Declining a question it could have answered",
        "caption": "A plain mortgage payment. Retrieval ranked a different amortization card first and the loan contract "
                   "outside the top four, and the selector judged that none of the four fit. VerifiQuant refuses at N; "
                   "R-PAL writes its own program and is right. This is the coverage the paper says safety costs "
                   "(84.0% vs 99.2% on the clean set).",
        "chapter_zh": "代價", "title_zh": "拒答了一題其實答得出來的問題",
        "caption_zh": "一般的房貸月付額。檢索把另一張攤還相關的合約排在第一，真正的貸款合約沒有進前四名，選擇器判斷這四張都不適用。VerifiQuant 停在 N 拒答，R-PAL 自己寫程式，答對了。論文說安全要付出覆蓋率，指的就是這種情況（乾淨題 84.0% 對 99.2%）。",
    },
]


def _clip(text: Any, limit: int = TEXT_LIMIT) -> str:
    s = str(text or "").strip()
    return s if len(s) <= limit else s[: limit - 1].rstrip() + "…"


def _num(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    return value


def load_card_titles() -> Dict[str, str]:
    con = sqlite3.connect(str(CARDS_DB))
    try:
        return {fid: title for fid, title in con.execute("SELECT fic_id, title FROM retrieval_cards")}
    finally:
        con.close()


def _selector_reason(diag: Mapping[str, Any]) -> str:
    sel = (diag.get("selection_trace") or {}).get("selector") or {}
    if sel.get("reason"):
        return str(sel["reason"])
    for line in diag.get("pipeline_logs") or []:
        if str(line).startswith("selector:"):
            return str(line).split("reason=", 1)[-1]
    return ""


def _candidates(diag: Mapping[str, Any], titles: Mapping[str, str]) -> List[Dict[str, Any]]:
    trace = (diag.get("selection_trace") or {}).get("retrieval_candidates") or []
    if trace:
        return [
            {"fic_id": c.get("fic_id"), "title": c.get("title") or titles.get(c.get("fic_id"), ""),
             "score": round(float(c.get("score") or 0.0), 3)}
            for c in trace
        ]
    return [{"fic_id": fid, "title": titles.get(fid, ""), "score": None} for fid in diag.get("candidate_ids") or []]


def _vq_message(diag: Mapping[str, Any]) -> str:
    """The text a user would read: repair hint, clarification question, warning, or reason."""
    clar = diag.get("clarification_request") or {}
    if clar.get("questions"):
        return " ".join(str(q) for q in clar["questions"])
    hints = diag.get("repair_hints") or []
    if hints and hints[0].get("user_message"):
        # Drop the trailing routing tag some hints carry, e.g. "[i_level=hard; impact_scope=...]".
        return re.sub(r"\s*\[i_level=[^\]]*\]\s*$", "", str(hints[0]["user_message"]))
    soft = diag.get("soft_warnings") or []
    if soft:
        first = soft[0]
        return f"{first.get('clarification_question') or ''} Assumed: {first.get('assumption_if_not_clarified') or ''}".strip()
    if diag.get("status") in ("refusal",):
        return _selector_reason(diag) or str(diag.get("reason") or "")
    return str(diag.get("reason") or "")


def vq_record(row: Mapping[str, Any], titles: Mapping[str, str], flags: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    diag = row["framework_guided"]["history"][0]["diagnostic"]
    norm = normalize_vq(row)
    clar = diag.get("clarification_request") or {}
    options = []
    for opt in clar.get("options") or []:
        options.append({"label": opt.get("label"), "default": bool(opt.get("is_default")),
                        "has_transform": bool(opt.get("has_transform"))})
    soft = [
        {"hint_id": s.get("hint_id"), "question": s.get("clarification_question"),
         "assumed": s.get("assumption_if_not_clarified")}
        for s in diag.get("soft_warnings") or [] if isinstance(s, dict)
    ]
    rec = {
        "exit": norm["diagnostic"],
        "status": diag.get("status"),
        "numeric": norm["numeric"],
        "output": _num(norm["output"]),
        "fic_id": diag.get("fic_id"),
        "fic_title": titles.get(diag.get("fic_id") or "", ""),
        "rule_ids": list(diag.get("triggered_rule_ids") or (diag.get("echeck_trace") or {}).get("auto_triggered_rule_ids") or []),
        "requested_fields": list(diag.get("requested_fields") or []),
        "inputs": {k: _num(v) for k, v in (diag.get("provided_inputs") or {}).items()},
        "message": _clip(_vq_message(diag)),
        "selector_reason": _clip(_selector_reason(diag)),
        "candidates": _candidates(diag, titles),
        "options": options,
        "soft_warnings": soft,
    }
    if flags is not None:
        rec["false_accept"] = bool(flags["unsafe_numeric"])
        rec["exit_correct"] = bool(flags["diagnostic_correct"])
        rec["action_correct"] = bool(flags["action_correct"])
    return rec


def rpal_record(row: Mapping[str, Any], flags: Optional[Mapping[str, Any]], with_code: bool) -> Dict[str, Any]:
    norm = normalize_rpal(row)
    rec = {
        "exit": norm["diagnostic"],
        "decision": norm["decision"],
        "numeric": norm["numeric"],
        "output": _num(norm["output"]),
        "message": _clip(norm["clarification_question"] or norm["warning"] or norm["reason"]),
    }
    if with_code:
        rec["code"] = _clip(row.get("generated_code") or "", CODE_LIMIT)
    if flags is not None:
        rec["false_accept"] = bool(flags["unsafe_numeric"])
        rec["exit_correct"] = bool(flags["diagnostic_correct"])
        rec["action_correct"] = bool(flags["action_correct"])
    return rec


def trap_edit(trap: Mapping[str, Any]) -> Dict[str, Any]:
    """Describe what the trap operator changed, in display-ready form."""
    g = trap.get("grounding") or {}
    op = trap["operator"]
    if op == "E":
        return {"kind": "value", "field": g.get("field"), "from": g.get("clean_value"),
                "to": g.get("violating_value"), "rule_id": g.get("targeted_rule_id"),
                "expression": g.get("expression")}
    if op == "F":
        return {"kind": "removed_field", "field": g.get("removed_field"), "value": g.get("removed_value")}
    if op == "I":
        return {"kind": "removed_phrase", "phrase": g.get("removed_phrase"), "rule_id": g.get("rule_id"),
                "family": g.get("ambiguity_family")}
    return {"kind": "replaced_question", "template": g.get("template")}


def build_traps(titles: Mapping[str, str]) -> List[Dict[str, Any]]:
    traps = load_jsonl(TRAPS)
    runs: Dict[str, List[Dict[str, Dict[str, Any]]]] = {}
    for name, paths in SYSTEMS.items():
        normalizer = normalize_vq if name == "vq" else normalize_rpal
        per_run = []
        fnar_runs = []
        for path in paths:
            rows = load_jsonl(path)
            scored = score_traps(traps, rows, normalizer)
            fnar_runs.append(scored["overall"]["fnar"])
            flags = {d["case_id"]: d for d in scored["details"]}
            by_id = {str(r["case_id"]): r for r in rows}
            out = {}
            for i, trap in enumerate(traps):
                cid = str(trap["trap_id"])
                row = by_id[cid]
                if name == "vq":
                    out[cid] = vq_record(row, titles, flags[cid])
                else:
                    out[cid] = rpal_record(row, flags[cid], with_code=(len(per_run) == 0))
            per_run.append(out)
        mean = sum(fnar_runs) / len(fnar_runs)
        # The paper rounds half-up (39/80 = 0.4875 prints as 48.8%), so compare within half a unit.
        assert abs(mean - PAPER_FNAR[name]) <= 0.0005 + 1e-9, f"{name}: FNAR {mean:.4f} != paper {PAPER_FNAR[name]}"
        runs[name] = per_run

    # Later VQ runs mostly repeat run 1's retrieval and selector trace; keep a field only
    # where it differs, which roughly halves the file the page downloads.
    for later in runs["vq"][1:]:
        for cid, rec in later.items():
            first = runs["vq"][0][cid]
            for key in ("candidates", "selector_reason", "inputs"):
                if rec.get(key) == first.get(key):
                    rec.pop(key, None)

    items = []
    for trap in traps:
        cid = str(trap["trap_id"])
        items.append({
            "id": cid,
            "set": "trap",
            "operator": trap["operator"],
            "family": trap.get("family"),
            "expected": trap["expected_diagnostic"],
            "question": trap["question"],
            "context": trap["context"],
            "edit": trap_edit(trap),
            "edit_note": trap.get("reason"),
            "target_fic": (trap.get("grounding") or {}).get("fic_id") or (trap.get("grounding") or {}).get("base_fic_id"),
            "systems": {name: [run[cid] for run in per_run] for name, per_run in runs.items()},
        })
    return items


def build_clean(titles: Mapping[str, str], ids: List[str]) -> List[Dict[str, Any]]:
    questions = {str(q["question_id"]): q for q in load_jsonl(CLEAN_QUESTIONS)}
    vq_rows = load_jsonl(CLEAN_VQ)
    rp_rows = load_jsonl(CLEAN_RPAL)
    vq_labels = clean_labels_from_vq(vq_rows, set(questions))
    rp_labels = clean_labels_from_rpal(rp_rows, questions, use_stored_verdict=True)
    vq_sum, rp_sum = clean_summary(vq_labels), clean_summary(rp_labels)
    assert (round(vq_sum["accuracy"], 3), round(vq_sum["swr"], 3), round(vq_sum["coverage"], 3)) == (0.812, 0.028, 0.84), vq_sum
    assert (round(rp_sum["accuracy"], 3), round(rp_sum["swr"], 3), round(rp_sum["coverage"], 3)) == (0.968, 0.024, 0.992), rp_sum

    vq_by = {str(r["case_id"]): r for r in vq_rows}
    rp_by = {str(r["case_id"]): r for r in rp_rows}
    items = []
    for cid in ids:
        q = questions[cid]
        vq = vq_record(vq_by[cid], titles, None)
        vq["verdict"] = vq_labels[cid]
        rp = rpal_record(rp_by[cid], None, with_code=True)
        rp["verdict"] = rp_labels[cid]
        items.append({
            "id": cid,
            "set": "clean",
            "operator": None,
            "level": q.get("level"),
            "expected": None,
            "question": q["question"],
            "context": q["context"],
            "gold": _num(q.get("ground_truth")),
            "systems": {"vq": [vq], "rpal": [rp]},
        })
    return items


def main() -> None:
    titles = load_card_titles()
    traps = build_traps(titles)
    clean_ids = [f["id"] for f in FEATURED if f["set"] == "clean"]
    clean = build_clean(titles, clean_ids)
    trap_ids = {t["id"] for t in traps}
    for f in FEATURED:
        assert f["id"] in trap_ids or f["id"] in clean_ids, f["id"]

    payload = {
        "meta": {
            "paper": "VerifiQuant: Governance-by-Execution for Financial LLM Workflows (ACM ICAIF '26)",
            "backbone": "gemini-2.5-flash",
            "top_k": 4,
            "library": "250 contracts (datasets/02_contract_library)",
            "scorer": "scripts/analyze_governance_eval.py (score_traps, normalize_vq, normalize_rpal)",
            "sources": {
                "traps": str(TRAPS.relative_to(ROOT)),
                "vq_traps": [str(p.relative_to(ROOT)) for p in SYSTEMS["vq"]],
                "rpal_plus_traps": [str(p.relative_to(ROOT)) for p in SYSTEMS["rpal_plus"]],
                "rpal_traps": [str(p.relative_to(ROOT)) for p in SYSTEMS["rpal"]],
                "clean_vq": str(CLEAN_VQ.relative_to(ROOT)),
                "clean_rpal": str(CLEAN_RPAL.relative_to(ROOT)),
            },
            "fnar": PAPER_FNAR,
            # Short form for the page footer (paths relative to verifiquant/data/runs/).
            "sources_display": {
                "trap": "governance_confirmatory_1014/vq_k1_top4_run{1,2,3}.jsonl · "
                        "governance_confirmatory/conf_prompted_run{1,2,3}.jsonl · governance_confirmatory/conf_plain_run1.jsonl",
                "clean": "paper_v2_250/results/vq_flash/output.jsonl (turn 1, K = 1) · "
                         "retrieval_pal_baseline/rpal_top4_gemini_25_flash_audited.jsonl",
            },
        },
        "featured": FEATURED,
        "cases": traps + clean,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(traps)} traps + {len(clean)} clean, {OUT.stat().st_size / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
