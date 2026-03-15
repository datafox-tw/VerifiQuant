from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover - runtime dependency guard
    genai = None
    genai_types = None

from verifiquant_v2.contracts import REFUSAL_CATEGORY_DESCRIPTIONS, USER_REPAIR_OPTIONS
from verifiquant_v2.taxonomy import is_valid_domain_topic, taxonomy_json


if genai_types is not None:
    STAGE1_CORE_SCHEMA = genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "id": genai_types.Schema(type=genai_types.Type.STRING),
            "name": genai_types.Schema(type=genai_types.Type.STRING),
            "short_description": genai_types.Schema(type=genai_types.Type.STRING),
            "domain": genai_types.Schema(type=genai_types.Type.STRING),
            "topic": genai_types.Schema(type=genai_types.Type.STRING),
            "inputs": genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=genai_types.Schema(
                    type=genai_types.Type.OBJECT,
                    properties={
                        "name": genai_types.Schema(type=genai_types.Type.STRING),
                        "type": genai_types.Schema(type=genai_types.Type.STRING),
                        "description": genai_types.Schema(type=genai_types.Type.STRING),
                        "required": genai_types.Schema(type=genai_types.Type.BOOLEAN),
                        "unit": genai_types.Schema(type=genai_types.Type.STRING),
                    },
                    required=["name", "type", "description", "required"],
                ),
            ),
            "output_var": genai_types.Schema(type=genai_types.Type.STRING),
            "execution": genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "language": genai_types.Schema(type=genai_types.Type.STRING),
                    "entrypoint": genai_types.Schema(type=genai_types.Type.STRING),
                    "code": genai_types.Schema(type=genai_types.Type.STRING),
                    "deterministic": genai_types.Schema(type=genai_types.Type.BOOLEAN),
                },
                required=["language", "entrypoint", "code", "deterministic"],
            ),
        },
        required=[
            "id",
            "name",
            "short_description",
            "domain",
            "topic",
            "inputs",
            "output_var",
            "execution",
        ],
    )

    STAGE2_ENRICH_SCHEMA = genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "diagnostics": genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "invariants": genai_types.Schema(
                        type=genai_types.Type.ARRAY,
                        items=genai_types.Schema(
                            type=genai_types.Type.OBJECT,
                            properties={
                                "id": genai_types.Schema(type=genai_types.Type.STRING),
                                "rule": genai_types.Schema(type=genai_types.Type.STRING),
                                "category": genai_types.Schema(
                                    type=genai_types.Type.STRING,
                                    enum=["F", "E"],
                                ),
                                "severity": genai_types.Schema(
                                    type=genai_types.Type.STRING,
                                    enum=["error", "alert"],
                                ),
                                "message": genai_types.Schema(type=genai_types.Type.STRING),
                            },
                            required=["id", "rule", "category", "severity", "message"],
                        ),
                    ),
                    "scale_checks": genai_types.Schema(
                        type=genai_types.Type.ARRAY,
                        items=genai_types.Schema(
                            type=genai_types.Type.OBJECT,
                            properties={
                                "id": genai_types.Schema(type=genai_types.Type.STRING),
                                "params": genai_types.Schema(
                                    type=genai_types.Type.ARRAY,
                                    items=genai_types.Schema(type=genai_types.Type.STRING),
                                ),
                                "expected": genai_types.Schema(type=genai_types.Type.STRING),
                                "category": genai_types.Schema(
                                    type=genai_types.Type.STRING,
                                    enum=["F", "E"],
                                ),
                                "severity": genai_types.Schema(
                                    type=genai_types.Type.STRING,
                                    enum=["error", "alert"],
                                ),
                                "message": genai_types.Schema(type=genai_types.Type.STRING),
                            },
                            required=[
                                "id",
                                "params",
                                "expected",
                                "category",
                                "severity",
                                "message",
                            ],
                        ),
                    ),
                },
                required=["invariants", "scale_checks"],
            ),
            "selection_hints": genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "self_description": genai_types.Schema(type=genai_types.Type.STRING),
                    "applicable_when": genai_types.Schema(type=genai_types.Type.ARRAY),
                    "not_applicable_when": genai_types.Schema(type=genai_types.Type.ARRAY),
                    "common_confusions": genai_types.Schema(type=genai_types.Type.ARRAY),
                    "required_input_summary": genai_types.Schema(type=genai_types.Type.ARRAY),
                    "disambiguation_prompts": genai_types.Schema(type=genai_types.Type.ARRAY),
                },
                required=[
                    "self_description",
                    "applicable_when",
                    "not_applicable_when",
                    "common_confusions",
                    "required_input_summary",
                    "disambiguation_prompts",
                ],
            ),
            "refusal_hints": genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "m_refusal_message": genai_types.Schema(type=genai_types.Type.STRING),
                    "f_error_message": genai_types.Schema(type=genai_types.Type.STRING),
                    "e_alert_message": genai_types.Schema(type=genai_types.Type.STRING),
                    "clarification_prompts": genai_types.Schema(type=genai_types.Type.ARRAY),
                    "user_repair_options": genai_types.Schema(type=genai_types.Type.ARRAY),
                },
                required=[
                    "m_refusal_message",
                    "f_error_message",
                    "e_alert_message",
                    "clarification_prompts",
                    "user_repair_options",
                ],
            ),
        },
        required=["diagnostics", "selection_hints", "refusal_hints"],
    )
else:
    STAGE1_CORE_SCHEMA = None
    STAGE2_ENRICH_SCHEMA = None


STAGE1_PROMPT = """Role:
You are a financial software engineer generating a Core FIC (stage 1).

Task:
Convert the input definition into a minimal, stable core JSON with only:
- id, name, short_description
- domain, topic
- inputs
- output_var
- execution (python deterministic code)

CRITICAL PRIORITY:
1) `python_solution` is the PRIMARY source for executable logic.
2) `function` is SECONDARY for concept naming and generalization.
3) Core FIC must solve the given question/context, but be generalized beyond this single sample.

Taxonomy constraint:
- Choose domain/topic ONLY from provided taxonomy.
- Never invent new domain/topic values.

Generalization rules:
1) Replace sample constants with inputs where formula-relevant.
2) Remove story details; keep financial meaning.
3) Keep units/time basis in input descriptions when important.
4) Do not hardcode sample numbers from context/question.

Execution rules:
1) Must define: `def compute(inputs): ...`
2) Return dict that includes output_var key.
3) deterministic=true; no network/file IO/randomness.

Input Definition:
<DEFINITION_JSON>
{definition_json}
</DEFINITION_JSON>

Allowed Taxonomy:
<TAXONOMY_JSON>
{taxonomy_json}
</TAXONOMY_JSON>

Output:
Return JSON only. No markdown.
"""


STAGE2_PROMPT = """Role:
You are a financial verification engineer generating Stage-2 diagnostics metadata.

Task:
Given a formalized Core FIC (stage 1), produce only:
- diagnostics (invariants + scale_checks)
- selection_hints
- refusal_hints

Do NOT modify core fields (id, domain, topic, inputs, execution, output_var).

Rules:
1) Create 2-5 invariants (category in F/E).
2) Create 1-4 scale checks for unit/scale/time consistency.
3) refusal_hints must support M/F/E interactions.
4) user_repair_options must use only allowed enum values.

Refusal taxonomy:
{refusal_taxonomy_json}

Allowed repair options:
{user_repair_options_json}

Formalized Core FIC:
<CORE_FIC_JSON>
{core_fic_json}
</CORE_FIC_JSON>

Output:
Return JSON only. No markdown.
"""


@dataclass
class ConversionInput:
    source_meta: Dict[str, Any]
    function: str
    python_solution: str
    context: str
    question: str


def _load_records(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"Unsupported JSON payload type in {path}")


def _to_definition(record: Dict[str, Any]) -> ConversionInput:
    return ConversionInput(
        source_meta={
            "function_id": record.get("function_id"),
            "article_title": record.get("article_title"),
            "source": record.get("source"),
            "question_id": record.get("question_id"),
            "difficulty": record.get("difficulty"),
            "level": record.get("level"),
        },
        function=str(record.get("function", "") or ""),
        python_solution=str(record.get("python_solution", "") or ""),
        context=str(record.get("context", "") or ""),
        question=str(record.get("question", "") or ""),
    )


def _safe_id(raw: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", raw.strip()).strip("_")
    if not cleaned:
        return "fic_v2_generated"
    if not cleaned.startswith("fic_"):
        return f"fic_{cleaned.lower()}"
    return cleaned.lower()


def _normalize_input_item(item: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(item)
    out["name"] = str(out.get("name", "")).strip()
    out["type"] = str(out.get("type", "float")).strip() or "float"
    out["description"] = str(out.get("description", "")).strip()
    out["required"] = bool(out.get("required", True))
    return out


def build_stage1_prompt(defn: ConversionInput) -> str:
    definition_json = json.dumps(
        {
            "source_meta": defn.source_meta,
            "function": defn.function,
            "python_solution": defn.python_solution,
            "context": defn.context,
            "question": defn.question,
        },
        ensure_ascii=False,
        indent=2,
    )
    return STAGE1_PROMPT.format(
        definition_json=definition_json,
        taxonomy_json=taxonomy_json(indent=2),
    )


def build_stage2_prompt(core_fic: Dict[str, Any]) -> str:
    return STAGE2_PROMPT.format(
        refusal_taxonomy_json=json.dumps(
            REFUSAL_CATEGORY_DESCRIPTIONS, ensure_ascii=False, indent=2, sort_keys=True
        ),
        user_repair_options_json=json.dumps(USER_REPAIR_OPTIONS, ensure_ascii=False),
        core_fic_json=json.dumps(core_fic, ensure_ascii=False, indent=2),
    )


def _call_gemini_json(
    client: Any,
    *,
    model: str,
    prompt: str,
    schema: Any,
) -> Dict[str, Any]:
    if genai is None or genai_types is None:
        raise RuntimeError(
            "google.genai is not available in this environment. Install/upgrade the dependency before running conversion."
        )
    config = genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    try:
        return json.loads(response.text)
    except json.JSONDecodeError as err:
        raise ValueError(f"Gemini returned invalid JSON: {response.text}") from err


def formalize_stage1_core(core: Dict[str, Any], fallback_id: str) -> Dict[str, Any]:
    out = dict(core)
    out["id"] = _safe_id(str(out.get("id", fallback_id)))
    out["name"] = str(out.get("name", "")).strip()
    out["short_description"] = str(out.get("short_description", "")).strip()
    out["domain"] = str(out.get("domain", "")).strip().lower()
    out["topic"] = str(out.get("topic", "")).strip().lower()
    out["output_var"] = str(out.get("output_var", "")).strip()

    raw_inputs = out.get("inputs", [])
    if not isinstance(raw_inputs, list) or not raw_inputs:
        raise ValueError("Stage1 core must include non-empty inputs list")
    out["inputs"] = [_normalize_input_item(i) for i in raw_inputs]

    execution = dict(out.get("execution") or {})
    execution["language"] = "python"
    execution["entrypoint"] = "compute"
    execution["deterministic"] = True
    code = str(execution.get("code", "")).strip()
    if "def compute(inputs)" not in code:
        raise ValueError("Stage1 execution.code must define compute(inputs)")
    execution["code"] = code
    out["execution"] = execution

    if not out["name"]:
        raise ValueError("Stage1 core missing name")
    if not out["short_description"]:
        raise ValueError("Stage1 core missing short_description")
    if not out["output_var"]:
        raise ValueError("Stage1 core missing output_var")
    if not is_valid_domain_topic(out["domain"], out["topic"]):
        raise ValueError(
            f"Invalid taxonomy selection from stage1: domain={out['domain']}, topic={out['topic']}"
        )
    return out


def validate_stage2_enrichment(payload: Dict[str, Any]) -> None:
    diagnostics = payload.get("diagnostics")
    selection_hints = payload.get("selection_hints")
    refusal_hints = payload.get("refusal_hints")
    if not isinstance(diagnostics, dict):
        raise ValueError("Stage2 payload missing diagnostics object")
    if "invariants" not in diagnostics or "scale_checks" not in diagnostics:
        raise ValueError("diagnostics must include invariants and scale_checks")
    if not isinstance(selection_hints, dict):
        raise ValueError("Stage2 payload missing selection_hints object")
    if not isinstance(refusal_hints, dict):
        raise ValueError("Stage2 payload missing refusal_hints object")

    repair_opts = refusal_hints.get("user_repair_options", [])
    if not isinstance(repair_opts, list):
        raise ValueError("refusal_hints.user_repair_options must be a list")
    illegal_opts = [opt for opt in repair_opts if opt not in USER_REPAIR_OPTIONS]
    if illegal_opts:
        raise ValueError(f"Unknown user_repair_options: {illegal_opts}")


def validate_final_fic(fic: Dict[str, Any]) -> None:
    domain = str(fic.get("domain", ""))
    topic = str(fic.get("topic", ""))
    if not is_valid_domain_topic(domain, topic):
        raise ValueError(f"Invalid taxonomy selection: domain={domain}, topic={topic}")

    execution = fic.get("execution") or {}
    if execution.get("language") != "python":
        raise ValueError("execution.language must be 'python'")
    if execution.get("entrypoint") != "compute":
        raise ValueError("execution.entrypoint must be 'compute'")
    if not execution.get("deterministic", False):
        raise ValueError("execution.deterministic must be true")
    code = str(execution.get("code", ""))
    if "def compute(inputs)" not in code:
        raise ValueError("execution.code must define compute(inputs)")

    diagnostics = fic.get("diagnostics") or {}
    if "invariants" not in diagnostics or "scale_checks" not in diagnostics:
        raise ValueError("diagnostics must include invariants and scale_checks")
    if not isinstance(fic.get("selection_hints"), dict):
        raise ValueError("selection_hints must be an object")
    if not isinstance(fic.get("refusal_hints"), dict):
        raise ValueError("refusal_hints must be an object")


def convert_record_two_stage(
    record: Dict[str, Any],
    *,
    client: genai.Client,
    stage1_model: str,
    stage2_model: str,
) -> Dict[str, Any]:
    defn = _to_definition(record)
    if not defn.python_solution.strip():
        raise ValueError("Record missing python_solution")
    fallback_id = (
        f"{defn.source_meta.get('function_id') or 'unknown'}_{defn.source_meta.get('question_id') or 'unknown'}"
    )

    stage1_prompt = build_stage1_prompt(defn)
    if STAGE1_CORE_SCHEMA is None or STAGE2_ENRICH_SCHEMA is None:
        raise RuntimeError("Stage schemas are unavailable because google.genai failed to import.")

    raw_core = _call_gemini_json(
        client,
        model=stage1_model,
        prompt=stage1_prompt,
        schema=STAGE1_CORE_SCHEMA,
    )
    core_fic = formalize_stage1_core(raw_core, fallback_id=fallback_id)

    stage2_prompt = build_stage2_prompt(core_fic)
    enrichment = _call_gemini_json(
        client,
        model=stage2_model,
        prompt=stage2_prompt,
        schema=STAGE2_ENRICH_SCHEMA,
    )
    validate_stage2_enrichment(enrichment)

    final_fic = {
        **core_fic,
        "diagnostics": enrichment["diagnostics"],
        "selection_hints": enrichment["selection_hints"],
        "refusal_hints": enrichment["refusal_hints"],
    }
    validate_final_fic(final_fic)
    return {
        "core_fic": core_fic,
        "final_fic": final_fic,
    }


def convert_records(
    records: Iterable[Dict[str, Any]],
    *,
    client: genai.Client,
    stage1_model: str,
    stage2_model: str,
) -> Dict[str, List[Dict[str, Any]]]:
    final_cards: List[Dict[str, Any]] = []
    core_cards: List[Dict[str, Any]] = []
    for idx, record in enumerate(records, start=1):
        print(f"[v2-two-stage] processing record {idx} ...")
        payload = convert_record_two_stage(
            record,
            client=client,
            stage1_model=stage1_model,
            stage2_model=stage2_model,
        )
        core_cards.append(payload["core_fic"])
        final_cards.append(payload["final_fic"])
    return {
        "core_cards": core_cards,
        "final_cards": final_cards,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-stage conversion: dataset problem records -> formalized FIC v2 (stage1 core + stage2 diagnostics)."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input JSON or JSONL file.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path. Defaults to <input>.fic_v2.json",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Default Gemini model for both stages unless stage-specific args are given.",
    )
    parser.add_argument(
        "--stage1-model",
        help="Optional model override for stage 1 core generation.",
    )
    parser.add_argument(
        "--stage2-model",
        help="Optional model override for stage 2 diagnostics enrichment.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Optional cap for number of records to process (0 = all).",
    )
    parser.add_argument(
        "--dump-stage1-core",
        type=Path,
        help=(
            "Optional path to write Stage 1 formalized core FIC JSON. "
            "If omitted, core output is not persisted."
        ),
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Missing GEMINI_API_KEY in environment.", file=sys.stderr)
        sys.exit(1)
    if genai is None:
        print(
            "Missing dependency: google.genai import failed. Please install a compatible google-genai package.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    records = _load_records(args.input)
    if args.max_records > 0:
        records = records[: args.max_records]

    stage1_model = args.stage1_model or args.model
    stage2_model = args.stage2_model or args.model

    client = genai.Client(api_key=api_key)
    converted = convert_records(
        records,
        client=client,
        stage1_model=stage1_model,
        stage2_model=stage2_model,
    )
    core_fics = converted["core_cards"]
    final_fics = converted["final_cards"]

    output_path = args.output or args.input.with_suffix(".fic_v2.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(final_fics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(final_fics)} final FIC v2 cards to {output_path}")

    if args.dump_stage1_core:
        args.dump_stage1_core.parent.mkdir(parents=True, exist_ok=True)
        args.dump_stage1_core.write_text(
            json.dumps(core_fics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote {len(core_fics)} stage1 core FIC cards to {args.dump_stage1_core}")


if __name__ == "__main__":
    main()
