from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from verifiquant_v3.preprocessing.common import (
    dump_records,
    load_records,
    require_gemini_client,
    to_conversion_input,
)
from verifiquant_v3.preprocessing.stage_core import generate_core
from verifiquant_v3.preprocessing.stage_repair import generate_repair_rules
from verifiquant_v3.preprocessing.stage_retrieval import generate_retrieval


def _default_output_paths(input_path: Path) -> Dict[str, Path]:
    stem = input_path.with_suffix("")
    return {
        "core": Path(f"{stem}.fic_core.json"),
        "retrieval": Path(f"{stem}.fic_retrieval.json"),
        "repair": Path(f"{stem}.repair_rules.json"),
    }


def run_pipeline(
    *,
    rows: List[Dict[str, Any]],
    client: Any,
    stage1_model: str,
    stage2_model: str,
    stage3_model: str,
    allow_new_topic: bool,
) -> Dict[str, List[Dict[str, Any]]]:
    core_cards: List[Dict[str, Any]] = []
    retrieval_cards: List[Dict[str, Any]] = []
    repair_rules: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        conversion_input = to_conversion_input(row)
        print(f"[v3-fic-pipeline] processing record {idx} ...")

        core = generate_core(
            client=client,
            model=stage1_model,
            defn=conversion_input,
            allow_new_topic=allow_new_topic,
        )
        retrieval = generate_retrieval(
            client=client,
            model=stage2_model,
            core=core,
        )
        repairs = generate_repair_rules(
            client=client,
            model=stage3_model,
            core=core,
        )

        core_cards.append(core)
        retrieval_cards.append(retrieval)
        repair_rules.extend(repairs)

    return {
        "core": core_cards,
        "retrieval": retrieval_cards,
        "repair": repair_rules,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "v3 multi-stage generator: dataset case -> fic_core -> fic_retrieval + repair_rule."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Input JSON or JSONL file")
    parser.add_argument("--core-output", type=Path, help="Output path for fic_core JSON/JSONL")
    parser.add_argument("--retrieval-output", type=Path, help="Output path for fic_retrieval JSON/JSONL")
    parser.add_argument("--repair-output", type=Path, help="Output path for repair_rule JSON/JSONL")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Default model for all stages")
    parser.add_argument("--stage1-model", help="Stage 1 override model")
    parser.add_argument("--stage2-model", help="Stage 2 override model")
    parser.add_argument("--stage3-model", help="Stage 3 override model")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument(
        "--disallow-new-topic",
        action="store_true",
        help="If set, topic must already exist under chosen taxonomy domain.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    rows = load_records(args.input)
    if args.max_records > 0:
        rows = rows[: args.max_records]

    client = require_gemini_client()
    stage1_model = args.stage1_model or args.model
    stage2_model = args.stage2_model or args.model
    stage3_model = args.stage3_model or args.model

    result = run_pipeline(
        rows=rows,
        client=client,
        stage1_model=stage1_model,
        stage2_model=stage2_model,
        stage3_model=stage3_model,
        allow_new_topic=not args.disallow_new_topic,
    )

    default_paths = _default_output_paths(args.input)
    core_output = args.core_output or default_paths["core"]
    retrieval_output = args.retrieval_output or default_paths["retrieval"]
    repair_output = args.repair_output or default_paths["repair"]

    dump_records(core_output, result["core"])
    dump_records(retrieval_output, result["retrieval"])
    dump_records(repair_output, result["repair"])

    print(f"Wrote {len(result['core'])} core cards to {core_output}")
    print(f"Wrote {len(result['retrieval'])} retrieval cards to {retrieval_output}")
    print(f"Wrote {len(result['repair'])} repair rules to {repair_output}")


if __name__ == "__main__":
    main()
