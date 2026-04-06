from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "PyYAML is required for this script. Install with: pip install pyyaml"
    ) from exc


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML object in {path}, got {type(data).__name__}")
    return data


def load_json_array(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}, got {type(data).__name__}")
    rows: List[dict] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item #{idx} in {path} is not an object")
        rows.append(item)
    return rows


def collect_target_ids(config: dict) -> List[str]:
    target_ids = config.get("target_ids")
    if not isinstance(target_ids, dict):
        raise ValueError("config.yaml missing object field: target_ids")

    ordered: List[str] = []
    for key in ("correct_samples", "error_samples"):
        values = target_ids.get(key, [])
        if not isinstance(values, list):
            raise ValueError(f"target_ids.{key} must be a list")
        for raw in values:
            qid = str(raw).strip()
            if qid:
                ordered.append(qid)

    # Deduplicate while preserving config order
    seen = set()
    deduped: List[str] = []
    for qid in ordered:
        if qid in seen:
            continue
        seen.add(qid)
        deduped.append(qid)
    return deduped


def index_by_question_id(rows: Iterable[dict]) -> Dict[str, dict]:
    index: Dict[str, dict] = {}
    for row in rows:
        qid = str(row.get("question_id", "")).strip()
        if qid and qid not in index:
            index[qid] = row
    return index


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract question_id targets from config.yaml and write matched medium.json rows to JSONL."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("verifiquant/data/config.yaml"),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--medium",
        type=Path,
        default=Path("verifiquant/data/medium.json"),
        help="Path to medium.json (array of objects with question_id)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("verifiquant/data/medium_config_50.jsonl"),
        help="Output JSONL path",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    target_ids = collect_target_ids(config)
    medium_rows = load_json_array(args.medium)
    medium_index = index_by_question_id(medium_rows)

    selected_rows: List[dict] = []
    missing_ids: List[str] = []
    for qid in target_ids:
        row = medium_index.get(qid)
        if row is None:
            missing_ids.append(qid)
        else:
            selected_rows.append(row)

    write_jsonl(args.out, selected_rows)

    print(f"targets={len(target_ids)} selected={len(selected_rows)} missing={len(missing_ids)}")
    print(f"output={args.out}")
    if missing_ids:
        print("missing_ids:")
        for qid in missing_ids:
            print(f"- {qid}")


if __name__ == "__main__":
    main()
