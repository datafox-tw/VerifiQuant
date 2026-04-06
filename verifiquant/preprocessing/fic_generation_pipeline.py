from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from verifiquant.preprocessing.common import (
    dump_records,
    load_records,
    require_gemini_client,
    to_conversion_input,
)
from verifiquant.preprocessing.stage_core import generate_core
from verifiquant.preprocessing.stage_repair import generate_repair_rules
from verifiquant.preprocessing.stage_retrieval import generate_retrieval
from verifiquant.preprocessing.seed_builder import build_seed_rows_from_config
from verifiquant.preprocessing.validate_relations import validate_artifact_relations
from verifiquant.card_store import SQLAlchemyArtifactStore


def _default_output_paths(input_path: Path) -> Dict[str, Path]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("verifiquant/data/runs") / f"fic_generation_{ts}"
    stem = run_dir / input_path.stem
    return {
        "core": Path(f"{stem}.fic_core.json"),
        "retrieval": Path(f"{stem}.fic_retrieval.json"),
        "repair": Path(f"{stem}.repair_rules.json"),
    }


def _load_rows_if_exists(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return load_records(path)


def _merge_core_rows(existing: List[Dict[str, Any]], new_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for row in existing + new_rows:
        fic_id = str(row.get("fic_id", "")).strip()
        if not fic_id:
            continue
        by_id[fic_id] = row
    return list(by_id.values())


def _merge_retrieval_rows(existing: List[Dict[str, Any]], new_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for row in existing + new_rows:
        fic_id = str(row.get("fic_id", "")).strip()
        if not fic_id:
            continue
        by_id[fic_id] = row
    return list(by_id.values())


def _merge_repair_rows(existing: List[Dict[str, Any]], new_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key: Dict[tuple[str, str], Dict[str, Any]] = {}
    for row in existing + new_rows:
        fic_id = str(row.get("fic_id", "")).strip()
        rule_id = str(row.get("rule_id", "")).strip()
        if not fic_id or not rule_id:
            continue
        by_key[(fic_id, rule_id)] = row
    return list(by_key.values())


def _collect_processed_source_ids(existing_core_rows: List[Dict[str, Any]]) -> tuple[set[str], set[str]]:
    function_ids: set[str] = set()
    question_ids: set[str] = set()
    for row in existing_core_rows:
        sm = row.get("source_meta", {}) if isinstance(row, dict) else {}
        fid = str(sm.get("function_id", "")).strip()
        qid = str(sm.get("question_id", "")).strip()
        if fid:
            function_ids.add(fid)
        if qid:
            question_ids.add(qid)
    return function_ids, question_ids


def _normalize_token(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    out = []
    for ch in text:
        if ch.isalnum() or ch == "_":
            out.append(ch)
        elif ch in {"-", " "}:
            out.append("_")
    token = "".join(out).strip("_")
    return token or "case"


def _candidate_suffix(row: Dict[str, Any], idx: int) -> str:
    return _normalize_token(
        row.get("question_id")
        or row.get("source")
        or row.get("function_id")
        or f"row_{idx}"
    )


def _resolve_duplicate_fic_id(
    *,
    core: Dict[str, Any],
    seen_fic_ids: set[str],
    row: Dict[str, Any],
    idx: int,
    policy: str,
) -> Dict[str, Any] | None:
    fic_id = str(core.get("fic_id", "")).strip()
    if not fic_id:
        raise ValueError(f"record {idx}: generated empty fic_id")
    if fic_id not in seen_fic_ids:
        return core

    if policy == "skip":
        return None
    if policy == "error":
        raise ValueError(f"record {idx}: duplicate fic_id generated: {fic_id}")

    # policy == "suffix": keep card but make ID unique by source-derived suffix.
    suffix = _candidate_suffix(row, idx)
    candidate = f"{fic_id}__{suffix}"
    serial = 2
    while candidate in seen_fic_ids:
        candidate = f"{fic_id}__{suffix}_{serial}"
        serial += 1
    core["fic_id"] = candidate
    return core


def run_pipeline(
    *,
    rows: List[Dict[str, Any]],
    client: Any,
    stage1_model: str,
    stage2_model: str,
    stage3_model: str,
    allow_new_topic: bool,
    duplicate_fic_policy: str,
    initial_seen_fic_ids: Optional[set[str]] = None,
    on_record_complete: Optional[Callable[[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], int], None]] = None,
) -> Dict[str, Any]:
    core_cards: List[Dict[str, Any]] = []
    retrieval_cards: List[Dict[str, Any]] = []
    repair_rules: List[Dict[str, Any]] = []
    skipped_duplicates = 0
    seen_fic_ids: set[str] = set(initial_seen_fic_ids or set())
    processed_records = 0

    for idx, row in enumerate(rows, start=1):
        conversion_input = to_conversion_input(row)
        print(f"[fic-pipeline] processing record {idx} ...")

        core = generate_core(
            client=client,
            model=stage1_model,
            defn=conversion_input,
            allow_new_topic=allow_new_topic,
        )
        resolved = _resolve_duplicate_fic_id(
            core=core,
            seen_fic_ids=seen_fic_ids,
            row=row,
            idx=idx,
            policy=duplicate_fic_policy,
        )
        if resolved is None:
            skipped_duplicates += 1
            print(f"[fic-pipeline] skipped duplicate fic_id at record {idx}")
            continue
        core = resolved
        seen_fic_ids.add(str(core["fic_id"]))

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
        processed_records += 1
        if on_record_complete is not None:
            on_record_complete(core, retrieval, repairs, idx)

    return {
        "core": core_cards,
        "retrieval": retrieval_cards,
        "repair": repair_rules,
        "skipped_duplicates": skipped_duplicates,
        "processed_records": processed_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "multi-stage FIC v2 generator: dataset case -> fic_core_v2 -> fic_retrieval_v2 + repair_rule_v2."
        )
    )
    parser.add_argument("--input", type=Path, help="Input JSON or JSONL file")
    parser.add_argument(
        "--seed-from-config",
        action="store_true",
        help=(
            "Build FIC seed rows from config target_ids + function/article/docs sources "
            "instead of reading --input directly."
        ),
    )
    parser.add_argument("--config-path", type=Path, default=Path("verifiquant/data/config.yaml"))
    parser.add_argument("--qa-dataset-path", type=Path, default=Path("verifiquant/data/medium.json"))
    parser.add_argument("--functions-catalog-path", type=Path, default=Path("verifiquant/data/functions-article-all.json"))
    parser.add_argument("--financial-docs-path", type=Path, default=Path("verifiquant/data/financial_documents.json"))
    parser.add_argument(
        "--max-doc-chars",
        type=int,
        default=4000,
        help="When --seed-from-config is used, max document characters kept in generated seed context.",
    )
    parser.add_argument(
        "--no-dedupe-function-seeds",
        action="store_true",
        help="When --seed-from-config is used, keep multiple seed rows per same function_id.",
    )
    parser.add_argument(
        "--seed-report-output",
        type=Path,
        help="Optional JSON report path for --seed-from-config assembly diagnostics.",
    )
    parser.add_argument("--core-output", type=Path, help="Output path for fic_core JSON/JSONL")
    parser.add_argument("--retrieval-output", type=Path, help="Output path for fic_retrieval JSON/JSONL")
    parser.add_argument("--repair-output", type=Path, help="Output path for repair_rule JSON/JSONL")
    parser.add_argument(
        "--checkpoint-every-record",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When enabled (default), persist core/retrieval/repair outputs after each processed record.",
    )
    parser.add_argument(
        "--skip-existing-core",
        action="store_true",
        help=(
            "If set, read existing core output file and skip rows whose source ids were already processed "
            "(source_meta.function_id or source_meta.question_id)."
        ),
    )
    parser.add_argument("--model", default="gemini-2.5-flash", help="Default model for all stages")
    parser.add_argument("--stage1-model", help="Stage 1 override model")
    parser.add_argument("--stage2-model", help="Stage 2 override model")
    parser.add_argument("--stage3-model", help="Stage 3 override model")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--db-url", help="Optional SQLAlchemy DB URL for direct ingestion after generation")
    parser.add_argument(
        "--duplicate-fic-policy",
        choices=["error", "suffix", "skip"],
        default="suffix",
        help=(
            "How to handle duplicate fic_id generated across records. "
            "suffix=append stable source suffix (default), skip=drop duplicate card, error=raise."
        ),
    )
    parser.add_argument(
        "--on-validation-error",
        choices=["raise", "save"],
        default="raise",
        help="raise=fail immediately on artifact relation validation error; save=write generated files first and continue.",
    )
    parser.add_argument(
        "--validation-report",
        type=Path,
        help="Optional path to persist validation result/error as JSON.",
    )
    parser.add_argument(
        "--disallow-new-topic",
        action="store_true",
        help="If set, topic must already exist under chosen taxonomy domain.",
    )
    args = parser.parse_args()

    if args.seed_from_config:
        rows, seed_report = build_seed_rows_from_config(
            config_path=args.config_path,
            qa_dataset_path=args.qa_dataset_path,
            functions_catalog_path=args.functions_catalog_path,
            financial_docs_path=args.financial_docs_path,
            max_doc_chars=max(0, args.max_doc_chars),
            dedupe_by_function_id=not args.no_dedupe_function_seeds,
        )
        if args.max_records > 0:
            rows = rows[: args.max_records]
        print(
            "[fic-pipeline] seed-from-config: "
            f"requested={seed_report['requested_question_ids']}, built={seed_report['built_seed_rows']}, "
            f"dedupe_by_function_id={seed_report['dedupe_by_function_id']}"
        )
        if args.seed_report_output:
            args.seed_report_output.parent.mkdir(parents=True, exist_ok=True)
            args.seed_report_output.write_text(json.dumps(seed_report, ensure_ascii=False, indent=2), encoding="utf-8")
        elif seed_report.get("missing_question_ids") or seed_report.get("missing_function_entries") or seed_report.get("missing_doc_titles"):
            # concise visibility without dumping huge lists
            print(
                "[fic-pipeline] seed warnings: "
                f"missing_question_ids={len(seed_report.get('missing_question_ids', []))}, "
                f"missing_function_entries={len(seed_report.get('missing_function_entries', []))}, "
                f"missing_doc_titles={len(seed_report.get('missing_doc_titles', []))}"
            )
    else:
        if args.input is None:
            raise ValueError("--input is required unless --seed-from-config is set.")
        if not args.input.exists():
            raise FileNotFoundError(args.input)
        rows = load_records(args.input)
        if args.max_records > 0:
            rows = rows[: args.max_records]

    client = require_gemini_client()
    stage1_model = args.stage1_model or args.model
    stage2_model = args.stage2_model or args.model
    stage3_model = args.stage3_model or args.model

    default_paths = _default_output_paths(args.input or Path("seed_from_config.jsonl"))
    core_output = args.core_output or default_paths["core"]
    retrieval_output = args.retrieval_output or default_paths["retrieval"]
    repair_output = args.repair_output or default_paths["repair"]

    existing_core_rows: List[Dict[str, Any]] = []
    existing_retrieval_rows: List[Dict[str, Any]] = []
    existing_repair_rows: List[Dict[str, Any]] = []
    if args.skip_existing_core:
        existing_core_rows = _load_rows_if_exists(core_output)
        existing_retrieval_rows = _load_rows_if_exists(retrieval_output)
        existing_repair_rows = _load_rows_if_exists(repair_output)
        done_function_ids, done_question_ids = _collect_processed_source_ids(existing_core_rows)
        before = len(rows)
        filtered: List[Dict[str, Any]] = []
        for row in rows:
            row_fid = str(row.get("function_id", "")).strip()
            row_qid = str(row.get("question_id", "")).strip()
            if row_fid and row_fid in done_function_ids:
                continue
            if row_qid and row_qid in done_question_ids:
                continue
            filtered.append(row)
        rows = filtered
        print(
            "[fic-pipeline] skip-existing-core: "
            f"existing_core={len(existing_core_rows)}, requested={before}, remaining={len(rows)}"
        )

    checkpoint_core_rows: List[Dict[str, Any]] = list(existing_core_rows if args.skip_existing_core else [])
    checkpoint_retrieval_rows: List[Dict[str, Any]] = list(existing_retrieval_rows if args.skip_existing_core else [])
    checkpoint_repair_rows: List[Dict[str, Any]] = list(existing_repair_rows if args.skip_existing_core else [])

    def _checkpoint(core: Dict[str, Any], retrieval: Dict[str, Any], repairs: List[Dict[str, Any]], idx: int) -> None:
        nonlocal checkpoint_core_rows, checkpoint_retrieval_rows, checkpoint_repair_rows
        checkpoint_core_rows = _merge_core_rows(checkpoint_core_rows, [core])
        checkpoint_retrieval_rows = _merge_retrieval_rows(checkpoint_retrieval_rows, [retrieval])
        checkpoint_repair_rows = _merge_repair_rows(checkpoint_repair_rows, repairs)
        dump_records(core_output, checkpoint_core_rows)
        dump_records(retrieval_output, checkpoint_retrieval_rows)
        dump_records(repair_output, checkpoint_repair_rows)
        print(
            "[fic-pipeline] checkpoint saved "
            f"(record={idx}, core={len(checkpoint_core_rows)}, retrieval={len(checkpoint_retrieval_rows)}, repair={len(checkpoint_repair_rows)})"
        )

    seen_ids_seed = {str(x.get("fic_id", "")).strip() for x in existing_core_rows if str(x.get("fic_id", "")).strip()}

    result = run_pipeline(
        rows=rows,
        client=client,
        stage1_model=stage1_model,
        stage2_model=stage2_model,
        stage3_model=stage3_model,
        allow_new_topic=not args.disallow_new_topic,
        duplicate_fic_policy=args.duplicate_fic_policy,
        initial_seen_fic_ids=seen_ids_seed,
        on_record_complete=_checkpoint if args.checkpoint_every_record else None,
    )

    if args.skip_existing_core:
        result["core"] = _merge_core_rows(existing_core_rows, result["core"])
        result["retrieval"] = _merge_retrieval_rows(existing_retrieval_rows, result["retrieval"])
        result["repair"] = _merge_repair_rows(existing_repair_rows, result["repair"])

    dump_records(core_output, result["core"])
    dump_records(retrieval_output, result["retrieval"])
    dump_records(repair_output, result["repair"])

    print(f"Wrote {len(result['core'])} core cards to {core_output}")
    print(f"Wrote {len(result['retrieval'])} retrieval cards to {retrieval_output}")
    print(f"Wrote {len(result['repair'])} repair rules to {repair_output}")
    if result.get("skipped_duplicates", 0):
        print(f"[fic-pipeline] skipped duplicate cards: {result['skipped_duplicates']}")

    validation_stats: Dict[str, Any] | None = None
    validation_error: str | None = None
    try:
        validation_stats = validate_artifact_relations(
            core_cards=result["core"],
            retrieval_cards=result["retrieval"],
            repair_rules=result["repair"],
        )
        print(
            "[fic-pipeline] relation check passed: "
            f"core={validation_stats['core_count']}, retrieval={validation_stats['retrieval_count']}, "
            f"repair={validation_stats['repair_count']}, diagnostic_rules={validation_stats['diagnostic_rule_count']}"
        )
    except ValueError as err:
        validation_error = str(err)
        if args.on_validation_error == "raise":
            if args.validation_report:
                args.validation_report.parent.mkdir(parents=True, exist_ok=True)
                args.validation_report.write_text(
                    json.dumps({"ok": False, "error": validation_error}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            raise
        print("[fic-pipeline] validation failed but files were saved due to --on-validation-error save")
        print(validation_error)

    if args.validation_report:
        args.validation_report.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "ok": validation_error is None,
            "error": validation_error,
            "validation_stats": validation_stats,
            "output": {
                "core": str(core_output),
                "retrieval": str(retrieval_output),
                "repair": str(repair_output),
            },
            "counts": {
                "core": len(result["core"]),
                "retrieval": len(result["retrieval"]),
                "repair": len(result["repair"]),
                "skipped_duplicates": result.get("skipped_duplicates", 0),
            },
        }
        args.validation_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.db_url and validation_error is None:
        store = SQLAlchemyArtifactStore(args.db_url)
        db_stats = store.ingest_artifacts(
            core_cards=result["core"],
            retrieval_cards=result["retrieval"],
            repair_rules=result["repair"],
            validate_relations=True,
        )
        print(
            "[fic-pipeline] db ingest complete: "
            f"core={db_stats['core_count']}, retrieval={db_stats['retrieval_count']}, "
            f"repair={db_stats['repair_count']}, diagnostic_rules={db_stats['diagnostic_rule_count']}"
        )
    elif args.db_url and validation_error is not None:
        print("[fic-pipeline] skipped DB ingest because validation failed.")


if __name__ == "__main__":
    main()
