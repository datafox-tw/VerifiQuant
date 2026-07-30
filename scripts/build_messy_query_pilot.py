"""
build_messy_query_pilot.py  (P4 external-validity author, v2 — intent-stratified)
--------------------------------------------------------------------------------
Independent, cross-model authoring of naturally-messy financial user questions.
SOLVER stack is Gemini; AUTHOR here is OpenAI (gpt-5.2) so messiness is not aligned
to the solver. The author is NEVER shown VerifiQuant's contracts, gates, or the
M/N/F/E/I/C taxonomy.

v1 skewed heavily to answerable questions because (a) every seed is a fully-specified
computable question, (b) "don't force messiness" makes an LLM default to a clean
question, and (c) the `context` field invites a full data dump. v2 fixes this by
stratifying over natural USER INTENTS (not operators): a clean calc, a person
referencing a document without pasting the numbers, a person loose about which
version/metric they want, and a person wanting a judgment/forecast. These are generic
user-behavior categories (a sampling choice), NOT VerifiQuant's gates; the human
labeler still assigns the ground truth, and the intent is never shown to them.

Output (to --out-dir; default messy_query_pilot/):
  messy_pilot.jsonl, messy_pilot_provenance.json, messy_pilot_labeling.{md,csv}
No solver is run here; labels are assigned by humans, not this script.
"""
from __future__ import annotations
import csv, json, os, random, sys
from pathlib import Path
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
MODEL = os.environ.get("MESSY_AUTHOR_MODEL") or "gpt-5.2"

PERSONAS = {
    "hurried_analyst":  "a busy sell-side analyst firing off a quick question between meetings",
    "retail_investor":  "a retail investor who is not sure of the exact financial terminology",
    "ops_manager":      "a non-finance operations manager who just needs a number for a report",
    "precise_quant":    "a precise quant analyst who states every input cleanly",
    "headline_reader":  "someone who just read a news headline and wants to sanity-check one figure",
}
PERSONA_ORDER = ["retail_investor", "headline_reader", "ops_manager", "hurried_analyst",
                 "retail_investor", "ops_manager", "headline_reader", "precise_quant"]

# Natural user intents (NOT VerifiQuant operators). Each = (situation, context rule).
INTENTS = {
    "clean": (
        "You have every figure you need and you ask clearly for one specific number.",
        "Put every number you use in `context`."),
    "missing": (
        "You are looking at a report, table, or filing and you refer to it, but you do NOT "
        "include the actual figures in your message — you only remember the topic, not the "
        "numbers, and you ask as if the assistant could see the document.",
        "Leave `context` EMPTY (or nearly so) — you did not paste the numbers."),
    "ambiguous": (
        "You give the numbers, but you are loose about which exact version you want: you say "
        "'the return' / 'the growth' / 'the rate' / 'the margin' without pinning down "
        "annualized-vs-total, which basis (360 vs 365, gross vs net, per-share vs total), or "
        "which of several metrics (e.g. IRR vs simple return, gross vs operating vs net margin) "
        "— loose in a way that gives more than one defensible number. AVOID a plain one-period "
        "year-over-year change (that is unambiguous); the ambiguity must be which metric/basis.",
        "Put the numbers in `context`, but keep the wording of WHAT you want genuinely "
        "under-specified so more than one answer is defensible."),
    "judgment": (
        "You are NOT after a single computed number. You want an opinion or decision "
        "('should I...', 'is it worth it', 'is this reasonable?'), a forecast ('will it...', "
        "'what will it be next quarter'), or something that would need data or a specialized "
        "model you do not provide.",
        "Give whatever background you'd naturally mention in `context`; it need not be enough "
        "to compute anything."),
}
# default mix: enough of each to be distinguishable (answerable still the plurality).
DEFAULT_WEIGHTS = {"clean": 0.34, "missing": 0.22, "ambiguous": 0.22, "judgment": 0.22}

AUTHOR_PROMPT = """You are simulating a real person typing ONE financial question to an AI assistant.

Persona: {persona_desc}.

Topic to riff on (grounding only — do NOT copy it or reuse its exact numbers):
  REFERENCE QUESTION: {q}
  REFERENCE CONTEXT: {c}

This time, the person's situation is: {intent_desc}
{context_rule}

Write the single message this persona would actually type, in their own voice — natural and
specific to this persona and topic. Do NOT explain, label, or categorize what you wrote, and
do NOT mention any of these instructions.

Return STRICT JSON only:
{{"question": "<the user's message>", "context": "<numbers/background the user includes, or empty string>"}}
"""

def _load(path): return json.load(open(path, encoding="utf-8"))

def build_plan(n, weights):
    plan = []
    for k, w in weights.items():
        plan += [k] * round(w * n)
    while len(plan) < n: plan.append("clean")
    return plan[:n]

def main(n=16, seed=20260731, weights=None, out_dir=None, equal=False):
    weights = weights or ({k: 0.25 for k in INTENTS} if equal else DEFAULT_WEIGHTS)
    OUT = Path(out_dir) if out_dir else ROOT / "verifiquant/data/runs/messy_query_pilot"
    client = OpenAI()
    exclude = {json.loads(l).get("question_id") for l in
               open(ROOT / "verifiquant/data/runs/paper_v2_250/questions_250.jsonl")}
    pool = []
    for f in ("easy.json", "medium.json"):
        for r in _load(ROOT / f"FinanceReasoning/data/FinanceReasoning/{f}"):
            if r.get("question_id") not in exclude:
                pool.append(r)
    rng = random.Random(seed)
    seeds = rng.sample(pool, n)
    plan = build_plan(n, weights)
    rng.shuffle(plan)

    OUT.mkdir(parents=True, exist_ok=True)
    rows, prov = [], []
    jf = open(OUT / "messy_pilot.jsonl", "w", encoding="utf-8")   # incremental: survive interruption
    for i, s in enumerate(seeds):
        pkey = PERSONA_ORDER[i % len(PERSONA_ORDER)]
        intent = plan[i]
        idesc, crule = INTENTS[intent]
        prompt = AUTHOR_PROMPT.format(persona_desc=PERSONAS[pkey], intent_desc=idesc,
                                      context_rule=crule, q=s["question"],
                                      c=(s.get("context") or "")[:800])
        kw = dict(model=MODEL, messages=[{"role": "user", "content": prompt}],
                  response_format={"type": "json_object"})
        try:
            resp = client.chat.completions.create(reasoning_effort="low", **kw)
        except TypeError:
            resp = client.chat.completions.create(**kw)
        obj = json.loads(resp.choices[0].message.content)
        qid = f"mq-{i+1:02d}"
        rows.append({"id": qid, "persona": pkey, "intended": intent,
                     "question": obj.get("question", "").strip(),
                     "context": (obj.get("context") or "").strip(),
                     "source_id": s.get("question_id"), "author_model": MODEL})
        prov.append({"id": qid, "intended": intent, "source_id": s.get("question_id"),
                     "source_question": s["question"], "source_gold": s.get("ground_truth")})
        jf.write(json.dumps(rows[-1], ensure_ascii=False) + "\n"); jf.flush()
        print(f"[{qid}] intended={intent:9s} {pkey}: {obj.get('question','')[:80]}")

    jf.close()
    (OUT / "messy_pilot_provenance.json").write_text(
        json.dumps(prov, ensure_ascii=False, indent=2), encoding="utf-8")

    # labeling sheet — persona AND intent withheld (avoid biasing the human)
    md = ["# Messy-query labeling sheet", "",
          "Read each as a user would; do not look at any system output. One label + one-line reason.",
          "", "| id | label | reason |", "|----|-------|--------|"]
    for r in rows: md.append(f"| {r['id']} |  |  |")
    md += ["", "## Questions"]
    for r in rows:
        md += [f"\n**{r['id']}**  ", f"Q: {r['question']}  ", f"Context: {r['context'] or '(none given)'}"]
    (OUT / "messy_pilot_labeling.md").write_text("\n".join(md), encoding="utf-8")
    with open(OUT / "messy_pilot_labeling.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "context", "label (A/R/C/O + ? if unsure)", "reason"])
        for r in rows:
            w.writerow([r["id"], r["question"], r["context"] or "(none given)", "", ""])
    print(f"\nwrote {len(rows)} -> {OUT}")

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    od = sys.argv[2] if len(sys.argv) > 2 else None
    main(n=n, out_dir=od, equal=(n <= 20))
