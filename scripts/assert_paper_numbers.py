#!/usr/bin/env python3
"""
assert_paper_numbers.py
-----------------------
Pre-submission guard: every headline number in paper/main_final.tex is re-derived
from the raw run artifacts and compared against what the paper prints.

Two kinds of check:
  [SRC]  the value is recomputed from a run artifact (jsonl / summary.json)
  [INT]  no single artifact holds the value, so we assert internal consistency
         instead (e.g. acc + SWR == coverage, acc / coverage == selective acc)

Run from the repo root:  python3 scripts/assert_paper_numbers.py
Exit code 0 = all pass, 1 = at least one mismatch.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# main_final.tex is the submitted version; main_v2.tex is its immediate predecessor and
# is kept only for diffing. Override with PAPER_TEX to check a draft before it lands.
TEX = ROOT / os.environ.get("PAPER_TEX", "paper/main_final.tex")
RUNS = ROOT / "verifiquant/data/runs"

FAILURES: list[str] = []
CHECKS = 0


def load(p: Path):
    return json.load(open(p))


def rows(p: Path):
    return [json.loads(l) for l in open(p) if l.strip()]


def round_half_up(x: float, nd: int) -> float:
    """Round half away from zero, which is what the paper's numbers were typeset with.

    Python's built-in round() is banker's rounding, so round(48.75, 1) == 48.8 but
    round(11.25, 1) == 11.2. Comparing against that would flag correctly-typeset
    values as errors.
    """
    factor = 10 ** nd
    return float(int(abs(x) * factor + 0.5)) / factor * (1 if x >= 0 else -1)


def check(label: str, got: float, want: float, kind: str = "SRC", nd: int = 1) -> None:
    """Assert that the artifact value, rounded as the paper rounds, equals what is printed."""
    global CHECKS
    CHECKS += 1
    got_r = round_half_up(got, nd)
    ok = abs(got_r - want) < 1e-9
    mark = "ok  " if ok else "FAIL"
    print(f"  [{kind}] {mark} {label:<52} paper={want:<7} artifact={got:.4g} -> {got_r}")
    if not ok:
        FAILURES.append(f"{label}: paper={want} artifact={got:.4g} (rounds to {got_r})")


# ── macros the paper defines once and reuses in the tables ────────────────────
tex = TEX.read_text(encoding="utf8")
MACRO = {
    m.group(1): m.group(2)
    for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", tex)
}


# Which VQ trap run the paper cites. The submitted paper used one July run made at the
# CLI default top-3; the camera-ready drafts cite the mean of three top-4 reruns
# (2026-10-14) and name those files in a comment next to \govVQfnar.
VQ_TOP4 = "vq_k1_top4_run" in tex
GOV_SUMMARY = (RUNS / "governance_confirmatory_1014/confirmatory_summary_top4.json" if VQ_TOP4
               else RUNS / "governance_confirmatory/confirmatory_summary.json")
VQ_TRAP_FILES = ([RUNS / f"governance_confirmatory_1014/vq_k1_top4_run{i}.jsonl" for i in (1, 2, 3)]
                 if VQ_TOP4 else [RUNS / "governance_confirmatory/vq_k1_traps.jsonl"])
# The gate ladder, library sweep and out-of-library split also ran at top-3 in July and
# were rerun at top-4 on 2026-10-15 (E13-E15). A draft citing a rerun names its directory
# in a comment or a figure path, and the audit follows it.
LADDER_DIR = RUNS / ("gate_ladder_top4" if "gate_ladder_top4" in tex else "gate_ladder")
SWEEP_TOP4 = "libsweep_top4" in tex
SWEEP_AGG = RUNS / ("libsweep_top4" if SWEEP_TOP4 else "libsweep") / "aggregate.json"
OOD_RESULTS = (RUNS / "ood_eval_top4/ood_results_e13_top4.jsonl" if "ood_eval_top4" in tex
               else RUNS / "ood_eval/ood_results_20260713_005129.jsonl")


def text_check(label: str, text: str) -> None:
    """Assert that a sentence carrying a re-derived number is present verbatim."""
    global CHECKS
    CHECKS += 1
    ok = text in tex
    print(f"  [TXT] {'ok  ' if ok else 'FAIL'} {label:<52} \"{text[:60]}\"")
    if not ok:
        FAILURES.append(f"{label}: phrase not found: {text}")


def macro_pct(name: str) -> float:
    """Read a \\newcommand-defined percentage macro as a float."""
    raw = MACRO[name].replace(r"\%", "").replace("%", "").strip()
    return float(raw)


def table_body(label: str) -> str:
    """Return the source of the table environment carrying \\label{label}.

    Row anchors like "CoT (single-shot)" appear in more than one table, so every
    lookup must be scoped to the table it belongs to.
    """
    for m in re.finditer(r"\\begin\{table\}(.*?)\\end\{table\}", tex, re.S):
        if f"\\label{{{label}}}" in m.group(1):
            return m.group(1)
    raise KeyError(f"no table with label {label}")


def tex_table_cell(label: str, anchor: str, col: int) -> float:
    """Pull column `col` (1-indexed after the row label) from row `anchor` of table `label`.

    Cells may be literals (``96.8\\%``) or macro references (``\\rpalAccAll{}``);
    macro references are resolved through the \\newcommand table so a table row and
    its macro definition can never silently drift apart.
    """
    for line in table_body(label).splitlines():
        if anchor in line and "&" in line:
            cells = [c.strip() for c in line.split("&")]
            raw = cells[col]
            m = re.search(r"\\([A-Za-z]+)\{\}", raw)
            if m and m.group(1) in MACRO:
                raw = MACRO[m.group(1)]
            raw = re.sub(r"\\textbf\{|\{|\}|\\%|%|\\\\", "", raw).strip()
            return float(raw)
    raise KeyError(f"{label}:{anchor}")


print("=" * 78)
print("Table 2 Panel A — clean 250Q")
print("=" * 78)

# CoT single-shot: recount correct / silent-wrong / abstain from the raw run
cot = rows(RUNS / "paper_v2_250/results/cot_single_shot_flash/output.jsonl")
c = sw = ab = 0
for r in cot:
    rec = r["cot_self_improve"]
    a = rec.get("final_answer")
    empty = a is None or (isinstance(a, str) and not a.strip())
    if rec.get("final_is_correct"):
        c += 1
    elif empty:
        ab += 1
    else:
        sw += 1
n = len(cot)
check("CoT accuracy", 100 * c / n, tex_table_cell("tab:v2main", "CoT (single-shot)", 1))
check("CoT SWR", 100 * sw / n, tex_table_cell("tab:v2main", "CoT (single-shot)", 2))
check("CoT selective accuracy", 100 * c / (c + sw), tex_table_cell("tab:v2main", "CoT (single-shot)", 3))
check("CoT abstain (gate-ladder row)", ab, tex_table_cell("tab:v2gateladder", "CoT single-shot (LLM computes)", 3), nd=0)

# PAL / R-PAL / VQ all live in the R-PAL comparison file under the unified scorer
cmp_ = load(RUNS / "retrieval_pal_baseline/rpal_top4_gemini_25_flash_comparison.json")
M = cmp_["system_metrics"]
for key, anchor in [
    ("plain_pal_unified_vq_scorer", "PAL (no retrieval)"),
    ("retrieval_pal_top4_audited_unified_vq_scorer", "R-PAL (top-4 retrieval)"),
    ("vq_k1", r"\textbf{VerifiQuant} ($K{=}1$, autonomous)"),
    ("vq_k3", r"\quad $K{=}3$ (clarification upper bound)"),
]:
    if key not in M:
        print(f"  [SRC] SKIP {anchor} (key {key} absent)")
        continue
    m = M[key]
    name = anchor.split("(")[0].strip()
    try:
        check(f"{name} accuracy", 100 * m["accuracy"], tex_table_cell("tab:v2main", anchor, 1))
        check(f"{name} SWR", 100 * m["swr"], tex_table_cell("tab:v2main", anchor, 2))
        check(f"{name} selective accuracy", 100 * m["selective_accuracy"], tex_table_cell("tab:v2main", anchor, 3))
    except KeyError:
        # macro-driven row: compare against the macro instead of a literal cell
        pass

# VQ K=3 row is macro-driven
check("VQ K=3 accuracy (macro)", 100 * M["vq_k3"]["accuracy"], macro_pct("vqAccAll"))
check("VQ K=3 SWR (macro)", 100 * M["vq_k3"]["swr"], macro_pct("vqSWRAll"))
check("VQ K=3 sel.acc (macro)", 100 * M["vq_k3"]["selective_accuracy"], macro_pct("vqSelAccAll"))
check("VQ K=3 coverage (macro)", 100 * M["vq_k3"]["coverage"], macro_pct("vqCovAll"))
check("R-PAL accuracy (macro)", 100 * M["retrieval_pal_top4_audited_unified_vq_scorer"]["accuracy"],
      macro_pct("rpalAccAll"))
check("R-PAL SWR (macro)", 100 * M["retrieval_pal_top4_audited_unified_vq_scorer"]["swr"],
      macro_pct("rpalSWRAll"))
check("R-PAL coverage (macro)", 100 * M["retrieval_pal_top4_audited_unified_vq_scorer"]["coverage"],
      macro_pct("rpalCovAll"))

print()
print("=" * 78)
print("Table 2 Panel B + Table 3 — governance traps (80Q)")
print("=" * 78)
g = load(GOV_SUMMARY)
print(f"  VQ trap source: {', '.join(p.name for p in VQ_TRAP_FILES)}")
o = g["overall"]
check("VQ FNAR", 100 * o["vq_k1"]["fnar"], macro_pct("govVQfnar"))
check("VQ exact diagnostic", 100 * o["vq_k1"]["diagnostic_accuracy"], macro_pct("govVQdiag"))
check("VQ action accuracy", 100 * o["vq_k1"]["action_accuracy"], macro_pct("govVQaction"))
check("R-PAL plain FNAR", 100 * o["rpal_plain"]["fnar"], macro_pct("govPlainFnar"))
check("R-PAL+ FNAR", 100 * o["rpal_prompted_mean"]["fnar"], macro_pct("govPromptFnar"))
check("R-PAL+ diagnostic", 100 * o["rpal_prompted_mean"]["diagnostic_accuracy"], macro_pct("govPromptDiag"))
check("R-PAL+ action", 100 * o["rpal_prompted_mean"]["action_accuracy"], macro_pct("govPromptAction"))
check("prompted-minus-VQ gap", 100 * g["prompted_minus_vq_fnar_gap"]["point"], float(MACRO["govGap"]))
lo, hi = g["prompted_minus_vq_fnar_gap"]["ci95_base_clustered"]
check("gap CI low", 100 * lo, float(MACRO["govCIlo"]))
check("gap CI high", 100 * hi, float(MACRO["govCIhi"]))

# per-operator rows of Table 3
OP_ANCHOR = {"E": "E (boundary)", "I": "I (disambiguator)",
             "F": "F (removed input)", "N": "N (out-of-scope)"}
for op, anchor in OP_ANCHOR.items():
    check(f"Table3 {op} FNAR R-PAL", 100 * g["per_operator_fnar"]["rpal_plain"][op],
          tex_table_cell("tab:v2trapop", anchor, 1))
    check(f"Table3 {op} FNAR R-PAL+", 100 * g["per_operator_fnar"]["rpal_prompted_mean"][op],
          tex_table_cell("tab:v2trapop", anchor, 2))
    check(f"Table3 {op} FNAR VQ", 100 * g["per_operator_fnar"]["vq_k1"][op],
          tex_table_cell("tab:v2trapop", anchor, 3))

# VQ Diag / Act and R-PAL+ Diag columns of Table 3 (whole percent)
pda = g["per_operator_diag_action"]
for op, anchor in OP_ANCHOR.items():
    check(f"Table3 {op} VQ Diag", 100 * pda["vq_k1"][op]["diagnostic_accuracy"],
          tex_table_cell("tab:v2trapop", anchor, 4), nd=0)
    check(f"Table3 {op} VQ Act", 100 * pda["vq_k1"][op]["action_accuracy"],
          tex_table_cell("tab:v2trapop", anchor, 5), nd=0)
    check(f"Table3 {op} R-PAL+ Diag", 100 * pda["rpal_prompted_mean"][op]["diagnostic_accuracy"],
          tex_table_cell("tab:v2trapop", anchor, 6), nd=0)

# the Table 3 per-operator FNAR columns must average to the Overall row
for col, key in [(1, "rpal_plain"), (2, "rpal_prompted_mean"), (3, "vq_k1")]:
    mean = sum(tex_table_cell("tab:v2trapop", a, col) for a in OP_ANCHOR.values()) / 4
    printed = tex_table_cell("tab:v2trapop", "Overall", col)
    check(f"Table3 col{col} per-op mean == Overall", mean, printed, kind="INT")

print()
print("=" * 78)
print("Table 4 — cumulative gate ablation")
print("=" * 78)
LADDER = [
    ("r1_compute_only", "FIC select + bind + execute (no gates)"),
    ("r2_fe", r"\quad + F/E gates"),
    ("r3_fei", r"\quad + I gate"),
    ("r4_full", r"\quad + M/N gates (full funnel)"),
    # r5_full_hint is deliberately absent: the hint-trigger rung was an in-sample,
    # non-held-out result and was cut from Table 4. runs/gate_ladder/r5_full_hint_*
    # is retained for the record but is no longer a paper number.
]
print(f"  ladder source: {LADDER_DIR.name}")
for tag, anchor in LADDER:
    s = load(LADDER_DIR / f"{tag}_summary.json")
    # the paper reports unit-normalized figures: near-misses count as correct
    corr = s["correct"] + s["near_miss"]
    swr = s["silent_wrong"] - s["near_miss"]
    check(f"ladder {tag} acc", 100 * corr / s["n"], tex_table_cell("tab:v2gateladder", anchor, 1))
    check(f"ladder {tag} SWR", 100 * swr / s["n"], tex_table_cell("tab:v2gateladder", anchor, 2))
    check(f"ladder {tag} abstain", s["abstain"], tex_table_cell("tab:v2gateladder", anchor, 3), nd=0)
    check(f"ladder {tag} rows sum to {s['n']}", corr + swr + s["abstain"], s["n"], kind="INT", nd=0)

print()
print("=" * 78)
print("Table 5 — human-labeled external validity (80Q)")
print("=" * 78)
sr = load(RUNS / "messy_query_pilot/agreement/score_report.json")
RQ4 = {
    "CoT (single-shot)": "CoT (single-shot)",
    "R-PAL (top-4 retrieval)": "R-PAL (top-4 retrieval)",
    "R-PAL+ (taxonomy prompt)": r"R-PAL$^{+}$ (taxonomy prompt)",
    "VerifiQuant (K=1, M->O)": r"\textbf{VerifiQuant} ($K{=}1$)",
}
# Answer coverage: share of human-ANSWER items each system actually answered. This
# needs no gold value, unlike the retired Sel.Acc column: the messy queries were
# rewritten with fresh numbers, so the source question's gold does not transfer.
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("_sme", ROOT / "scripts/score_messy_external.py")
_sme = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_sme)
_gt = _sme.load_gt()
_sysrecs = {
    "CoT (single-shot)": _sme.adapt_cot(),
    "R-PAL (top-4 retrieval)": _sme.adapt_rpal_plain(),
    "R-PAL+ (taxonomy prompt)": _sme.adapt_rpal_plus(),
    "VerifiQuant (K=1, M->O)": _sme.adapt_vq(m_as_clarify=False),
}
_ans_items = [i for i, l in _gt.items() if l == "A"]
cov = {
    name: sum(1 for i in _ans_items if _sysrecs[name].get(i) and _sysrecs[name][i]["emit"])
    / len(_ans_items)
    for name in _sysrecs
}
check("RQ4 |ANSWER| items", len(_ans_items), 34, nd=0)

for src_name, anchor in RQ4.items():
    e = next(x for x in sr if x["system"] == src_name)
    check(f"RQ4 {src_name} Human-FNAR", 100 * e["human_fnar"], tex_table_cell("tab:v2ext", anchor, 1))
    check(f"RQ4 {src_name} answer coverage", 100 * cov[src_name], tex_table_cell("tab:v2ext", anchor, 2))
    check(f"RQ4 {src_name} Match", 100 * e["match"], tex_table_cell("tab:v2ext", anchor, 3))
# the M->clarify sensitivity figure quoted in the scoring-protocol paragraph
mc = next(x for x in sr if x["system"] == "VerifiQuant (K=1, M->C)")
check("RQ4 M->clarify Match (quoted 35.0%)", 100 * mc["match"], 35.0)

print()
print("=" * 78)
print("RQ4 prose — every count stated in §3.5 / §5.4 body text")
print("=" * 78)
# Table cells have a guard above; the surrounding prose did not, which is how a stale
# 31.7%/34.1% survived a set-size change. These checks re-derive each number the RQ4
# and construction paragraphs assert, then confirm the paper actually prints it.
MQ = RUNS / "messy_query_pilot"
gt = {r["id"]: r["label"] for r in rows(MQ / "ground_truth.jsonl")}
prov = load(MQ / "messy_pilot_provenance.json")
agree = load(MQ / "agreement_report.json")
vq_rows = {r["case_id"]: r for r in rows(MQ / "agreement/runs/vq_k1.jsonl")}
dist = agree["majority_label_distribution"]
n_items = len(gt)
rco = [i for i in gt if gt[i] != "A"]
answerable = [i for i in gt if gt[i] == "A"]
vq_ans = [i for i in answerable if vq_rows[i]["status"] == "success"]
shortfall = [i for i in answerable if vq_rows[i]["status"] != "success"]
n_refusals = [i for i in shortfall if vq_rows[i]["diagnostic_type"] == "N"]
cond = {}
for r in prov:
    cond[r["intended"]] = cond.get(r["intended"], 0) + 1


def phrase_check(label: str, ok: bool, detail_ok: str, detail_bad: str) -> None:
    """String-level guard: the recomputed value must actually appear in the paper."""
    global CHECKS
    CHECKS += 1
    print(f"  [TXT] {'ok  ' if ok else 'FAIL'} {label:<58} {detail_ok if ok else detail_bad}")
    if not ok:
        FAILURES.append(f"{label}: {detail_bad}")


check("RQ4 set size n", n_items, 80, nd=0)
check("RQ4 conditions sum to n", sum(cond.values()), n_items, kind="INT", nd=0)
for k, want in (("clean", 27), ("missing", 18), ("ambiguous", 18), ("judgment", 17)):
    check(f"RQ4 condition {k}", cond[k], want, nd=0)
check("RQ4 majority ANSWER", dist["A"], 34, nd=0)
check("RQ4 majority REQUEST-INFO", dist["R"], 20, nd=0)
check("RQ4 majority CLARIFY", dist["C"], 10, nd=0)
check("RQ4 majority OUT-OF-SCOPE", dist["O"], 16, nd=0)
check("RQ4 majority splits sum to n", sum(dist.values()), n_items, kind="INT", nd=0)
check("RQ4 |RCO| (non-ANSWER items)", len(rco), 46, nd=0)
check("RQ4 |RCO| + |A| == n", len(rco) + len(answerable), n_items, kind="INT", nd=0)
check("RQ4 Fleiss kappa (2dp)", agree["fleiss_kappa"], 0.84, nd=2)
check("RQ4 majority reached on all items", agree["majority_resolved"], n_items, kind="INT", nd=0)
# per-system emitted counts over RCO, quoted in the body as "4, 6, 10, and 26"
for src_name, want in (("VerifiQuant (K=1, M->O)", 4), ("R-PAL+ (taxonomy prompt)", 6),
                       ("CoT (single-shot)", 10), ("R-PAL (top-4 retrieval)", 26)):
    e = next(x for x in sr if x["system"] == src_name)
    check(f"RQ4 RCO emits {src_name}", e["fnar_hits"], want, nd=0)
vq = next(x for x in sr if x["system"] == "VerifiQuant (K=1, M->O)")
check("RQ4 VQ answered count (quoted 11/80)", vq["n_answered"], 11, nd=0)
check("RQ4 VQ answers on human-ANSWER items", len(vq_ans), 7, nd=0)
check("RQ4 human-answerable item count", len(answerable), 34, nd=0)
check("RQ4 VQ hit rate on answerable (quoted 20.6%)", 100 * len(vq_ans) / len(answerable), 20.6)
check("RQ4 shortfall count", len(shortfall), 27, nd=0)
check("RQ4 shortfalls that are N-refusals", len(n_refusals), 24, nd=0)
check("RQ4 answered + shortfall == answerable",
      len(vq_ans) + len(shortfall), len(answerable), kind="INT", nd=0)
for phrase in ("80 naturalistic queries", "all 80 items", "46 of the 80 queries",
               "$11/80$", "7 of the 34 items", "In 24 of the 27 shortfalls",
               "$n{=}80$ items"):
    phrase_check(f"prose states {phrase!r}", phrase in tex, "present", "MISSING from main_v2.tex")
# 82-era values that must not survive anywhere in the paper. This is a migration
# guard from the 2026-08-10 mq-81/mq-82 removal; retire it once the draft is final,
# since these are bare decimals that a future unrelated number could legitimately use.
for stale in ("87.8", "68.3", "65.9", "31.7", "34.1", "11/82", "7 of the 36",
              "19.4", "26 of the 29", "plus two further"):
    phrase_check(f"no stale 82-era {stale!r}", stale not in tex,
                 "absent", "STILL PRESENT in main_v2.tex")

print()
print("=" * 78)
print("Table 1 — RepairBench")
print("=" * 78)
rb = load(RUNS / "repairbench/repairbench_summary_20260719_184634.json")
tot = rb.get("mutants_total")
eq = rb.get("equivalent_mutants_total")
fa = rb.get("true_false_accept_k5_total")
if tot and eq is not None and fa is not None:
    noneq = tot - eq
    check("RepairBench mutants total", tot, 906, nd=0)
    check("RepairBench equivalent excluded", eq, 27, nd=0)
    check("RepairBench non-equivalent", noneq, 879, nd=0)
    check("RepairBench true false accepts", fa, 20, nd=0)
    check("RepairBench rejection rate", 100 * (noneq - fa) / noneq, 97.7)
    check("RepairBench FA rate", 100 * fa / noneq, 2.3)
else:
    print(f"  [SRC] SKIP RepairBench (keys present: {sorted(rb)[:8]})")

print()
print("=" * 78)
print("Figure 3 — library-coverage sweep")
print("=" * 78)
print(f"  sweep source: {SWEEP_AGG.parent.name}")
sweep = load(SWEEP_AGG)["points"]
ood = [p for p in sweep if p["ood_abstain"] is not None]
if SWEEP_TOP4:
    # The sentence carries the endpoints; check each printed value against the rerun.
    def _p(x, nd=1):
        return f"{round_half_up(100 * x, nd):.{nd}f}"
    gaps = [100 * p["selectivity_gap"] for p in ood]
    accs = [100 * p["id_acc"] for p in sweep]
    text_check("sweep OOD abstain endpoints",
               f"${_p(ood[0]['ood_abstain'])}\\%{{\\to}}{_p(ood[-1]['ood_abstain'])}\\%$")
    text_check("sweep ID abstain endpoints",
               f"${_p(sweep[0]['id_abstain'])}\\%{{\\to}}{_p(sweep[-1]['id_abstain'])}\\%$")
    text_check("sweep gap range",
               f"between ${round_half_up(min(gaps), 0):.0f}$ and ${round_half_up(max(gaps), 0):.0f}$ points")
    text_check("sweep ID accuracy range",
               f"${round_half_up(min(accs), 1):.1f}$--${round_half_up(max(accs), 1):.1f}\\%$")
    text_check("sweep OOD silent-wrong endpoints",
               f"${_p(ood[0]['ood_sw'])}\\%{{\\to}}{_p(ood[-1]['ood_sw'])}\\%$")
if not SWEEP_TOP4:
  check("OOD abstain at 10%", 100 * ood[0]["ood_abstain"], 91.6)
  check("OOD abstain at 75%", 100 * ood[-1]["ood_abstain"], 80.6)
  check("ID abstain at 10%", 100 * sweep[0]["id_abstain"], 8.0)
  check("ID abstain at 100%", 100 * sweep[-1]["id_abstain"], 14.4)
  check("OOD silent-wrong at 10%", 100 * ood[0]["ood_sw"], 1.3)
  check("OOD silent-wrong at 75%", 100 * ood[-1]["ood_sw"], 4.8)
  gaps = [100 * p["selectivity_gap"] for p in ood]
  check("selectivity gap min (paper says 67)", min(gaps), 67, nd=0)
  check("selectivity gap max (paper says 84)", max(gaps), 84, nd=0)
  accs = [100 * p["id_acc"] for p in sweep]
  check("ID accuracy min (paper says 81.6)", min(accs), 81.6)
  check("ID accuracy max (paper says 84.0)", max(accs), 84.0)

print()
print("=" * 78)
print("Out-of-library split (199 OOD + 51 ID), VerifiQuant macros")
print("=" * 78)
print(f"  OOD-split source: {OOD_RESULTS.name}")
_ood_rows = rows(OOD_RESULTS)
def _split(name):
    sub = [r for r in _ood_rows if r["split"] == name]
    return sub, {v: sum(r["verdict"] == v for r in sub) for v in ("abstain", "correct", "silent_wrong")}
_o, _oc = _split("OOD")
_i, _ic = _split("ID")
check("OOD split: out-of-library questions", len(_o), 199, kind="INT", nd=0)
check("OOD split: in-library questions", len(_i), 51, kind="INT", nd=0)
check("VQ OOD abstain", 100 * _oc["abstain"] / len(_o), macro_pct("oodVQabsOOD"))
check("VQ ID abstain", 100 * _ic["abstain"] / len(_i), macro_pct("oodVQabsID"))
check("VQ OOD-ID abstain gap", 100 * (_oc["abstain"] / len(_o) - _ic["abstain"] / len(_i)), float(MACRO["oodVQgap"]))
check("VQ OOD correct", 100 * _oc["correct"] / len(_o), macro_pct("oodVQcorrOOD"))
check("VQ OOD silent-wrong", 100 * _oc["silent_wrong"] / len(_o), macro_pct("oodVQswOOD"))
check("VQ OOD selective accuracy", 100 * _oc["correct"] / (_oc["correct"] + _oc["silent_wrong"]),
      macro_pct("oodVQselOOD"))

print()
print("=" * 78)
print("Selective-prediction reference point — self-consistency vote-share sweep")
print("=" * 78)
# The related-work paragraph quotes a threshold sweep over the stored SC@5 samples.
# Recompute it here with the run's own convention: an item counts as answered only
# when a winner exists (28 questions yield no parseable answer at all), which is what
# keeps the no-threshold coverage at 88.8% rather than 100%.
sc = rows(RUNS / "sc_baseline/sc5_gemini_25_flash.jsonl")
check("SC set size", len(sc), 250, nd=0)
check("SC unanimous questions", sum(1 for r in sc if r["share"] >= 1.0), macro_pct("scUnanimous"), nd=0)
for tau, cov_macro, swr_macro in ((0.0, "scCovOpen", "scSWRopen"), (1.0, "scCovTight", "scSWRtight")):
    ans = [r for r in sc if r["winner"] is not None and r["share"] >= tau]
    corr = sum(1 for r in ans if r["correct"])
    check(f"SC tau={tau} coverage", 100 * len(ans) / len(sc), macro_pct(cov_macro))
    check(f"SC tau={tau} SWR", 100 * (len(ans) - corr) / len(sc), macro_pct(swr_macro))
    if tau == 1.0:
        check("SC tau=1 selective accuracy", 100 * corr / len(ans), macro_pct("scSelAccTight"))
        # the paragraph's whole point is that this threshold lands on VQ's coverage exactly
        check("SC tau=1 coverage == VQ K=1 coverage", 100 * len(ans) / len(sc),
              tex_table_cell("tab:v2kabl", r"$K{=}1$ (autonomous)", 3), kind="INT")

print()
print("=" * 78)
print("Table 6 — clarification budgets (internal consistency)")
print("=" * 78)
for budget, anchor in [("K=1", r"$K{=}1$ (autonomous)"), ("K=2", r"$K{=}2$ &")]:
    acc = tex_table_cell("tab:v2kabl", anchor, 1)
    swr = tex_table_cell("tab:v2kabl", anchor, 2)
    cov = tex_table_cell("tab:v2kabl", anchor, 3)
    sel = tex_table_cell("tab:v2kabl", anchor, 4)
    check(f"Table6 {budget}: acc + SWR == coverage", acc + swr, cov, kind="INT", nd=1)
    check(f"Table6 {budget}: acc / coverage == sel.acc", 100 * acc / cov, sel, kind="INT", nd=1)
k3 = M["vq_k3"]
check("Table6 K=3: acc + SWR == coverage",
      100 * (k3["accuracy"] + k3["swr"]), 100 * k3["coverage"], kind="INT")

# CoT + oracle reference row. The submitted version has no such row, so the block runs
# only on a draft that carries it. Counts use the CoT single-shot convention above, and
# the prose counts beside the table are rebuilt from the two runs' own turn histories.
if "CoT + oracle" in table_body("tab:v2kabl"):
    coto = [r["cot_self_improve"] for r in rows(RUNS / "paper_v2_250/results/cot_basic_oracle_flash/output.jsonl")]
    c = sum(1 for r in coto if r.get("final_is_correct"))
    ab = sum(1 for r in coto if not r.get("final_is_correct")
             and (r.get("final_answer") is None or not str(r.get("final_answer")).strip()))
    sw = len(coto) - c - ab
    check("Table6 CoT+oracle accuracy", 100 * c / len(coto), tex_table_cell("tab:v2kabl", "CoT + oracle", 1))
    check("Table6 CoT+oracle SWR", 100 * sw / len(coto), tex_table_cell("tab:v2kabl", "CoT + oracle", 2))
    check("Table6 CoT+oracle coverage", 100 * (c + sw) / len(coto), tex_table_cell("tab:v2kabl", "CoT + oracle", 3))
    check("Table6 CoT+oracle selective accuracy", 100 * c / (c + sw), tex_table_cell("tab:v2kabl", "CoT + oracle", 4))
    check("Table6 CoT+oracle coverage == VQ K=3 coverage", 100 * (c + sw) / len(coto), 100 * k3["coverage"], kind="INT")

    first_ok = [bool(r["history"][0].get("is_correct")) for r in coto]
    rescued = sum(1 for r, ok in zip(coto, first_ok) if not ok and r.get("final_is_correct"))
    broken = sum(1 for r, ok in zip(coto, first_ok) if ok and not r.get("final_is_correct"))
    phrase = f"rescues {rescued} answers but breaks {broken} of the {sum(first_ok)}"
    phrase_check(f"prose states {phrase!r}", phrase in tex, "present", "MISSING or stale")

    # VerifiQuant side: a first-turn abstention is a structured non-answer. The three
    # I_SOFT_MISMATCH re-entries are answered-but-wrong turns and are not abstentions.
    vq_fg = [r["framework_guided"] for r in rows(RUNS / "paper_v2_250/results/vq_flash/output.jsonl")]
    done = lambda f: f["final_status"] == "success" and bool(f["final_is_correct"])
    abst = [f for f in vq_fg if f["history"][0].get("recorded_error") not in ("NONE", "I_SOFT_MISMATCH")]
    right = [f for f in vq_fg if f["history"][0].get("recorded_error") == "NONE"
             and f["history"][0]["diagnostic"].get("is_correct")]
    phrase = f"converts {sum(done(f) for f in abst)} of the {len(abst)} first-turn abstentions"
    phrase_check(f"prose states {phrase!r}", phrase in tex, "present", "MISSING or stale")
    kept = sum(done(f) for f in right)
    phrase = f"none of the {len(right)} answers" if kept == len(right) else f"{len(right) - kept} of the {len(right)} answers"
    phrase_check(f"prose states {phrase!r}", phrase in tex, "present", "MISSING or stale")

# ── camera-ready additions (skeleton v3, 2026-10-14) ─────────────────────────────
# Numbers the nine-page version will print that the submitted paper does not. They are
# recomputed from the artifacts and compared with the values frozen in
# paper/skeleton_9p_zh.md, so this block runs before the prose exists and does not read
# the .tex. Once a number lands in the prose, add a phrase_check for it as well.
print()
print("Camera-ready additions (data side, independent of the .tex)")
import csv

MQ = RUNS / "messy_query_pilot"
mq_gt = {r["id"]: r["label"] for r in rows(MQ / "ground_truth.jsonl")}
mq_vq = {r["case_id"]: r for r in rows(MQ / "agreement/runs/vq_k1.jsonl")}
q250 = {r["question_id"] for r in rows(RUNS / "paper_v2_250/questions_250.jsonl")}
prov = load(MQ / "messy_pilot_provenance.json")
check("CR human set: sources inside the 250", sum(p["source_id"] in q250 for p in prov), 0, nd=0)

# Answerable items the system declined, and where they exited.
answerable = [i for i, lab in mq_gt.items() if lab == "A"]
declined = [i for i in answerable if mq_vq[i]["status"] != "success"]
check("CR human set: answerable items", len(answerable), 34, nd=0)
check("CR human set: answerable items declined", len(declined), 27, nd=0)
check("CR human set: declined at N", sum(mq_vq[i]["diagnostic_type"] == "N" for i in declined), 24, nd=0)

# Item-level attribution of those N refusals (scripts/attribute_messy_shortfalls.py).
attr = [a for a in csv.DictReader(open(MQ / "shortfall_attribution.csv")) if a["vq_exit"] == "N"]
present = [a for a in attr if a["verdict"] == "present"]
absent = [a for a in attr if a["verdict"] == "absent"]
ARITH = {"part / whole share", "ratio of two values", "absolute difference"}
check("CR attribution: applicable card present", len(present), 7, nd=0)
check("CR attribution: no applicable card", len(absent), 17, nd=0)
check("CR attribution: absent, elementary arithmetic", sum(a["asked"] in ARITH for a in absent), 7, nd=0)
check("CR attribution: absent, named formula", sum(a["asked"] not in ARITH for a in absent), 10, nd=0)
check("CR attribution: present cards recovered at k=16",
      sum(1 for a in present if a["depth_needed"] and int(a["depth_needed"]) <= 16), 1, nd=0)

# Non-answer items: how many got no number, and how many of those exited at N.
rco = [i for i, lab in mq_gt.items() if lab != "A"]
no_number = [i for i in rco if mq_vq[i]["status"] != "success"]
check("CR human set: non-answer items", len(rco), 46, nd=0)
check("CR human set: non-answer items without a number", len(no_number), 42, nd=0)
check("CR human set: ...of which exited at N", sum(mq_vq[i]["diagnostic_type"] == "N" for i in no_number), 28, nd=0)


def from_selector(diag) -> bool:
    return "abstain_n" in " ".join(diag.get("pipeline_logs") or [])


# Every observed N refusal came from the LLM selector (answer to R2 on Figure 1).
human_n = [r for r in mq_vq.values() if r["status"] != "success" and r["diagnostic_type"] == "N"]
check("CR human set: N refusals", len(human_n), 52, nd=0)
check("CR human set: N refusals from the LLM selector", sum(from_selector(r) for r in human_n), 52, nd=0)
trap_n = [r["framework_guided"]["history"][0]["diagnostic"] for p in VQ_TRAP_FILES for r in rows(p)
          if r["framework_guided"].get("final_diagnostic_type") == "N"]
n_want = 47 if VQ_TOP4 else 15
check("CR traps: N refusals (all runs)", len(trap_n), n_want, nd=0)
check("CR traps: N refusals from the LLM selector", sum(from_selector(d) for d in trap_n), n_want, nd=0)

# The FNAR gap split by operator (four classes, equal weight).
po = g["per_operator_fnar"]
if VQ_TOP4:
    shares, total, nd_share = (("I", 14.6), ("E", 5.4), ("F", -2.5), ("N", -0.4)), 17.1, 1
else:
    shares, total, nd_share = (("I", 15.0), ("E", 5.0), ("F", -2.5), ("N", -1.25)), 16.25, 2
for op, want in shares:
    check(f"CR FNAR gap share, operator {op}", 100 * (po["rpal_prompted_mean"][op] - po["vq_k1"][op]) / 4,
          want, nd=nd_share)
check("CR FNAR gap, sum of the four shares",
      sum(100 * (po["rpal_prompted_mean"][o] - po["vq_k1"][o]) / 4 for o in "FEIN"), total, nd=nd_share)

if VQ_TOP4:
    # Prose that carries the three-run numbers (E10, 2026-10-14).
    sys.path.insert(0, str(ROOT / "scripts"))
    from analyze_governance_eval import normalize_vq, score_traps  # noqa: E402
    traps = rows(RUNS / "governance_confirmatory/trap_set.jsonl")
    i_wrong = [d for p in VQ_TRAP_FILES for d in score_traps(traps, rows(p), normalize_vq)["details"]
               if d["operator"] == "I" and not d["diagnostic_correct"]]
    n_ready = sum(d["got_diagnostic"] == "READY" for d in i_wrong)
    n_neighbor = sum(d["got_diagnostic"] in {"M", "E", "F"} for d in i_wrong)
    check("CR traps: wrong I exits over three runs", len(i_wrong), 23, nd=0)
    check("CR traps: ...passing as ready", n_ready, 11, nd=0)
    check("CR traps: ...stopping at M, E, or F", n_neighbor, 9, nd=0)
    text_check("CR text: I-error profile",
           "of its 23 wrong I exits over the three runs, 11 pass as ready and 9 stop at a neighboring gate")
    text_check("CR text: N refusals on the traps", "all 47 on the traps over three runs")
    text_check("CR text: integer gap", f"{round_half_up(100 * g['prompted_minus_vq_fnar_gap']['point'], 0):.0f}-point")
    text_check("CR text: I-trap diagnostic", f"(I, ${round_half_up(100 * pda['vq_k1']['I']['diagnostic_accuracy'], 0):.0f}\\%$)")
    text_check("CR text: N FNAR", f"N: ${round_half_up(100 * po['vq_k1']['N'], 1)}\\%$ vs.")
    m = re.search(r"VerifiQuant loses ([0-9.]+) \(\\govVQdiag", tex)
    if m:
        check("CR text: VQ diagnosis-to-action drop", 100 * (o["vq_k1"]["diagnostic_accuracy"]
              - o["vq_k1"]["action_accuracy"]), float(m.group(1)))

# E1: the three answer-conditioned re-entries in the canonical K=3 run.
vq3 = [r["framework_guided"] for r in rows(RUNS / "paper_v2_250/results/vq_flash/output.jsonl")]
reentry = [f for f in vq3 if any(h.get("recorded_error") == "I_SOFT_MISMATCH" for h in f["history"])]
rescued = sum(1 for f in reentry if f["final_status"] == "success" and f["final_is_correct"])
check("CR K=3: answer-conditioned re-entries", len(reentry), 3, nd=0)
check("CR K=3: ...that ended correct", rescued, 1, nd=0)
k3_answered = sum(f["final_status"] == "success" for f in vq3)
k3_correct = sum(1 for f in vq3 if f["final_status"] == "success" and f["final_is_correct"])
check("CR K=3 accuracy without them", 100 * (k3_correct - rescued) / len(vq3), 93.2)
check("CR K=3 SWR without them", 100 * (k3_answered - k3_correct + rescued) / len(vq3), 3.6)
check("CR K=3 selective accuracy without them", 100 * (k3_correct - rescued) / k3_answered, 96.3)

# E4: K=1 silent wrongs that carried an I_SOFT warning (the scorer counts them).
first = [f["history"][0] for f in vq3]
k1_wrong = [h["diagnostic"] for h in first
            if h["diagnostic"].get("output_value") is not None
            and (h["diagnostic"].get("status") == "success" or h.get("recorded_error") == "I_SOFT_MISMATCH")
            and not h["diagnostic"].get("is_correct")]
warned = [d for d in k1_wrong if d.get("has_i_soft") or d.get("soft_warnings")]
check("CR K=1 silent wrongs", len(k1_wrong), 7, nd=0)
check("CR K=1 silent wrongs carrying an I_SOFT warning", len(warned), 3, nd=0)
check("CR K=1 SWR if warned numbers counted as flagged", 100 * (len(k1_wrong) - len(warned)) / len(vq3), 1.6)

print()
print("=" * 78)
if FAILURES:
    print(f"RESULT: {len(FAILURES)} MISMATCH of {CHECKS} checks")
    for f in FAILURES:
        print("  - " + f)
    sys.exit(1)
print(f"RESULT: all {CHECKS} checks pass")
