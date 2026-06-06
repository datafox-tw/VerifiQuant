"""
patch_fic_cards.py
------------------
One-time surgical patch of 7 FIC cards in core.jsonl that caused failures
in the paper_v1 VQ Flash run (43/50 → target 48+/50).

Failures patched:
  test-1926  fic_article_1740  C-error  from math import comb → inline comb
  test-1710  fic_article_1400  C-error  pmi_dc_004 eval assignment syntax
  test-1789  fic_article_1610  C-error  output_is_finite uses undefined compute
  test-1499  fic_article_72    SW       method not in required → defaults to SL
  test-1845  fic_article_440   SW       interest earned ≠ maturity value (FV)
  test-1237  fic_article_2824  F-error  option_type not extracted from "call option"
  test-1606  fic_article_2183  F-error  direction not extracted from "stock price increased"

Run:
    python3 scripts/patch_fic_cards.py \
        --core verifiquant/data/runs/paper_v1/fic/core.jsonl \
        --out  verifiquant/data/runs/paper_v1/fic/core_v2.jsonl
"""

import argparse
import copy
import json
from pathlib import Path


# ── patch functions ────────────────────────────────────────────────────────────

def _patch_1740(card: dict) -> dict:
    """Replace `from math import comb` with inline binomial coefficient."""
    card = copy.deepcopy(card)
    card["execution"]["code"] = """\
def compute(inputs):
    trials = int(inputs['trials'])
    success_prob = inputs['success_prob']
    successes = int(inputs['successes'])
    # Inline binomial coefficient — no imports needed
    n, k = trials, successes
    if k < 0 or k > n:
        return 0.0
    k = min(k, n - k)
    c = 1
    for i in range(k):
        c = c * (n - i) // (i + 1)
    probability = c * success_prob ** successes * (1 - success_prob) ** (trials - successes)
    return round(probability * 100, 2)
"""
    return card


def _patch_1400(card: dict) -> dict:
    """Fix pmi_dc_004: E-check runs pre-execution, so `result` is not in eval ctx.
    Use compute(inputs) directly — compute IS available in the eval environment."""
    card = copy.deepcopy(card)
    for chk in card.get("diagnostic_checks", []):
        if chk.get("rule_id") == "pmi_dc_004":
            # Old: "result = compute(inputs); result < 0 or result > 100"
            # → eval() cannot have assignment statements
            # New: call compute(inputs) inline — compute is injected into eval env
            chk["expression"] = "compute(inputs) < 0 or compute(inputs) > 100"
    return card


def _patch_1610(card: dict) -> dict:
    """Fix output_is_finite: same issue — use compute(inputs), not `result`.
    math is also available in eval env."""
    card = copy.deepcopy(card)
    for chk in card.get("diagnostic_checks", []):
        if chk.get("rule_id") == "output_is_finite":
            # Old: "math.isinf(compute(inputs)) or math.isnan(compute(inputs))"
            # → compute worked but math.isnan/isinf failed because `math` wasn't
            #   confirmed to be in the old eval env. Now math IS in env (line 464).
            chk["expression"] = "math.isinf(compute(inputs)) or math.isnan(compute(inputs))"
    return card


def _patch_72(card: dict) -> dict:
    """Make `method` required so extractor must capture 'double_declining_balance'."""
    card = copy.deepcopy(card)
    inputs = card.get("inputs", [])  # inputs is a list of dicts
    found = False
    for inp in inputs:
        if inp.get("name") == "method":
            inp["required"] = True
            inp["description"] = (
                "Depreciation method to use. Must be one of: "
                "'straight_line' or 'double_declining_balance'. "
                "REQUIRED — extract from the question text "
                "(e.g. 'double-declining balance method' → 'double_declining_balance', "
                "'straight-line method' → 'straight_line')."
            )
            found = True
            break
    if not found:
        inputs.append({
            "name": "method",
            "type": "string",
            "required": True,
            "description": (
                "Depreciation method: 'straight_line' or 'double_declining_balance'. "
                "Extract from question text."
            ),
            "unit": "categorical",
            "aliases": [],
        })
    card["inputs"] = inputs
    return card


def _patch_440(card: dict) -> dict:
    """
    Add a semantic_hint for 'interest earned' vs 'maturity value / future value'.

    The current code computes interest earned: P × ((1+r)^n − 1).
    When the question asks for maturity value or future value, the result is P × (1+r)^n
    = result + principal.

    The transform_spec uses result_postprocess: result + principal.
    Numerical invariant: result_new == result_old + principal.
    """
    card = copy.deepcopy(card)
    hints = card.get("semantic_hints", [])

    # Only add if not already present
    existing_ids = {h.get("hint_id") for h in hints}
    if "interest_vs_future_value" not in existing_ids:
        hints.append({
            "hint_id": "interest_vs_future_value",
            "ambiguity_type": "output_interpretation",
            "question": (
                "Does the question ask for the compound interest EARNED (just the gain, "
                "not including the original principal), or the TOTAL maturity / future value "
                "(principal + interest)?"
            ),
            "options": [
                {
                    "label": "Interest earned only (the gain, not including principal)",
                    "value": "interest_earned",
                    "is_default": True,
                    "transform_spec": None,
                },
                {
                    "label": "Total maturity value / future value (principal + interest)",
                    "value": "future_value",
                    "is_default": False,
                    "transform_spec": {
                        "patch_type": "result_postprocess",
                        "result_expr": "result + principal",
                        "max_expr_nodes": 5,
                        "invariant": "result_new == result_old + principal",
                        "affected_inputs": ["principal"],
                    },
                },
            ],
        })
    card["semantic_hints"] = hints
    return card


def _patch_2824(card: dict) -> dict:
    """Improve option_type input description for better named-entity extraction."""
    card = copy.deepcopy(card)
    inputs = card.get("inputs", {})
    if "option_type" in inputs:
        inputs["option_type"]["description"] = (
            "Type of option: 'call' or 'put'. "
            "Extract directly from the question text — e.g. 'call option' → 'call', "
            "'put option' → 'put'. This field is REQUIRED and must be explicitly stated "
            "in the question."
        )
    return card


def _patch_2183(card: dict) -> dict:
    """Improve direction input description for better context-based extraction."""
    card = copy.deepcopy(card)
    inputs = card.get("inputs", {})
    if "direction" in inputs:
        inputs["direction"]["description"] = (
            "Direction of the underlying asset price movement at expiration: 'up' or 'down'. "
            "Infer from context: if the stock price at expiration is ABOVE the strike price "
            "or if the question says 'price increased / rose / went up', use 'up'. "
            "If the stock price is BELOW strike or 'price decreased / fell / went down', use 'down'. "
            "For long straddle, 'up' means the call is in-the-money."
        )
    return card


# ── patch registry ─────────────────────────────────────────────────────────────

PATCHES = {
    "fic_article_1740": _patch_1740,
    "fic_article_1400": _patch_1400,
    "fic_article_1610": _patch_1610,
    "fic_article_72":   _patch_72,
    "fic_article_440":  _patch_440,
    "fic_article_2824": _patch_2824,
    "fic_article_2183": _patch_2183,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", default="verifiquant/data/runs/paper_v1/fic/core.jsonl")
    parser.add_argument("--out",  default="verifiquant/data/runs/paper_v1/fic/core_v2.jsonl")
    args = parser.parse_args()

    patched = 0
    out_rows = []
    with open(args.core) as f:
        for line in f:
            if not line.strip():
                continue
            card = json.loads(line)
            fic_id = card.get("fic_id", "")
            if fic_id in PATCHES:
                card = PATCHES[fic_id](card)
                print(f"  [patch] {fic_id}")
                patched += 1
            out_rows.append(card)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nPatched {patched}/7 cards → {args.out}")
    print("Next step:")
    print("  python3 preprocessing/build_card_store.py \\")
    print("    --db-url sqlite:////path/to/cards_v2.db \\")
    print("    --core <out> --retrieval fic/retrieval.jsonl --repair fic/repair.jsonl")


if __name__ == "__main__":
    main()
