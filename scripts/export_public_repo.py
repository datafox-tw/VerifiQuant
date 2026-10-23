#!/usr/bin/env python3
"""
export_public_repo.py
---------------------
Build the public release repository (https://github.com/datafox-tw/icaif-verifiquant) from
this working repository, by allowlist. Nothing outside the lists below is copied, so paper
drafts, meeting notes, sibling repos and old experiments never reach the public tree, and
the release is regenerated (not hand-edited) whenever a result changes.

    .venv/bin/python scripts/export_public_repo.py --target ../icaif-verifiquant

Layout of the public tree:

    verifiquant/   the package: card store, contracts, card build (preprocessing/), funnel (pipeline/)
    datasets/      inputs and labels (copied from datasets/, see its README)
    results/       the recorded runs behind every number (verifiquant/data/runs/<run>/ here)
    scripts/       dataset builders, experiment runners, analysis, assert_paper_numbers.py
    paper/         camera-ready source + extended version (Appendices A-K), PDFs, figures
    tests/         offline checks
    README.md ...  from release/

Text files are rewritten on the way out: `verifiquant/data/runs` becomes `results`, and the
absolute path of this checkout is stripped. datasets/MANIFEST.json is then re-hashed for any
file whose bytes changed. Finally the tree is scanned for absolute home paths, API keys and
private e-mail addresses, and the export fails if any is found.

The target's .git directory is never touched. Files in the target that this export did not
write are listed (not deleted), so a stale file is visible before it is committed.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "verifiquant/data/runs"

PACKAGE = [
    "verifiquant/__init__.py",
    "verifiquant/card_store.py",
    "verifiquant/contracts.py",
    "verifiquant/taxonomy.py",
    "verifiquant/preprocessing/__init__.py",
    "verifiquant/preprocessing/common.py",
    "verifiquant/preprocessing/dataset_case_to_fic.py",
    "verifiquant/preprocessing/fic_generation_pipeline.py",
    "verifiquant/preprocessing/lint_echecks.py",
    "verifiquant/preprocessing/seed_builder.py",
    "verifiquant/preprocessing/stage_core.py",
    "verifiquant/preprocessing/stage_repair.py",
    "verifiquant/preprocessing/stage_retrieval.py",
    "verifiquant/preprocessing/stage_transform.py",
    "verifiquant/preprocessing/validate_relations.py",
    "verifiquant/pipeline/__init__.py",
    "verifiquant/pipeline/run_error_classification_pipeline.py",
    "verifiquant/pipeline/run_framework_guided_self_improve_pipeline.py",
    "verifiquant/pipeline/run_cot_self_improve_pipeline.py",
    "verifiquant/pipeline/build_trap_set.py",
    "verifiquant/pipeline/build_trap_set_v2.py",
    "verifiquant/pipeline/score_trap_set.py",
]

SCRIPTS = [
    # datasets and contract libraries
    "sample_dataset.py", "sample_dataset_250.py", "supervise_card_build.sh", "build_subset_libraries.py",
    "package_datasets.py", "build_messy_query_pilot.py", "curate_messy_missing_items.py",
    "trim_messy_set_to_80.py", "aggregate_messy_labels.py", "build_messy_answer_gold.py",
    # experiment runners (these call a model)
    "api_guard.py", "run_paper_experiments.py", "run_pal_baseline.py", "run_retrieval_pal_baseline.py",
    "run_retrieval_pal_governance_eval.py", "run_staged_rpal_governance_eval.py", "run_vq_trap_rerun.py",
    "run_scripted_user_eval.py", "run_sc_baseline.py", "run_ood_family_eval.py", "run_vq50_pro.py",
    "run_gate_ladder.sh", "run_libsweep.sh", "run_repairbench.py", "score_rpal_plus_clean250.py",
    "score_messy_external.py", "score_messy_external_SPEC.md",
    # analysis, figures, audit (offline)
    "analyze_governance_eval.py", "analyze_governance_confirmatory.py", "analyze_governance_multirun.py",
    "analyze_cameraready_e10_e7.py", "analyze_cameraready_e6_e12.py", "analyze_cameraready_topk_reruns.py",
    "analyze_trap_sensitivity.py", "analyze_retrieval_pal_baseline.py", "analyze_v2_by_tier.py",
    "run_stats_250q.py", "attribute_messy_shortfalls.py", "attribute_messy_nonanswer.py",
    "measure_retrieval_recall.py", "build_mixed_risk_coverage.py", "plot_safety_utility.py",
    "plot_libsweep.py", "plot_risk_coverage.py", "export_prompts_appendix.py", "assert_paper_numbers.py",
]

# Run directories under verifiquant/data/runs/, each with an optional allowlist of globs
# (None = the whole directory). Logs and build backups are dropped everywhere (RUN_EXCLUDE).
RUN_DIRS: Dict[str, List[str] | None] = {
    "paper_v2_250": [
        "questions_250.jsonl", "experiment_config.yaml", "paper_results_summary.json", "SCORING_NOTE.md",
        "fic/cards.db", "fic/core.jsonl", "fic/retrieval.jsonl", "fic/repair.jsonl", "fic/card_repairs.json",
        "fic/validation_report.json", "fic/echeck_lint_report.json", "results/*",
    ],
    # The pilot: its inputs (the 50-family library is the out-of-library split's library)
    # and the trap results behind the appendix's pilot case.
    "paper_v1": [
        "questions_50.jsonl", "experiment_config.yaml", "fic/cards.db", "fic/core.jsonl",
        "fic/retrieval.jsonl", "fic/repair.jsonl", "trap/trap_set.jsonl", "trap/trap_manifest.json",
        "trap/results/*",
    ],
    "retrieval_pal_baseline": ["rpal_top4_gemini_25_flash*"],
    "pal_baseline": None,
    "governance_eval": None,
    "governance_confirmatory": None,
    "governance_confirmatory_1014": None,
    "governance_confirmatory_1015_pro": None,
    "scripted_user_1015": None,
    "ood_eval": None,
    "ood_eval_top4": None,
    "libsweep": None,
    "libsweep_top4": None,
    "gate_ladder": None,
    "gate_ladder_top4": None,
    "messy_query_pilot": None,
    "repairbench": None,
    "sc_baseline": None,
    "mixed_risk_coverage": None,
}
RUN_FILES = ["cameraready_e6_e12_summary.json", "cameraready_topk_reruns_summary.json"]
RUN_EXCLUDE = ["*/logs/*", "*.log", "*.bak*", "*.DS_Store", "*__pycache__*", "*prelint*", "*prescalar*"]

PAPER = {
    "paper/main_camera_ready_9p.tex": "paper/main_camera_ready_9p.tex",
    "paper/appendix_camera_ready.tex": "paper/appendix_camera_ready.tex",
    "paper/appendix_prompts.tex": "paper/appendix_prompts.tex",
    "paper/references.bib": "paper/references.bib",
    "paper/figures/pipeline_v6.pdf": "paper/figures/pipeline_v6.pdf",
    "paper/figures/safety_utility_250q_top4_staged_compact.pdf": "paper/figures/safety_utility_250q_top4_staged_compact.pdf",
    "paper/figures/libsweep_250q_top4_compact.png": "paper/figures/libsweep_250q_top4_compact.png",
    "paper/main_camera_ready_9p.pdf": "paper/verifiquant_icaif26.pdf",
    "paper/main_camera_ready_9p_arxiv.pdf": "paper/verifiquant_icaif26_extended.pdf",
}

OTHER = {
    "test_transform_poc.py": "tests/test_transform_poc.py",
}

TEXT_SUFFIXES = {".py", ".sh", ".md", ".json", ".jsonl", ".yaml", ".yml", ".csv", ".tex", ".bib", ".txt", ".cff", ""}
REWRITES: List[Tuple[str, str]] = [
    (str(ROOT) + "/", ""),
    (str(ROOT), "."),
    ("verifiquant/data/runs", "results"),
]
# The working repo's runners call root-level preprocessing/*.py CLI shims (12-line re-exports of
# the package). The public tree has no shims, so those calls become `python -m verifiquant...`.
# Each pair must match at least once, so a runner that changes upstream fails the export loudly.
FILE_REWRITES: Dict[str, List[Tuple[str, str]]] = {
    "scripts/run_paper_experiments.py": [
        ('"python3", "preprocessing/dataset_case_to_fic.py",', '"python3", "-m", "verifiquant.preprocessing.dataset_case_to_fic",'),
        ('"python3", "preprocessing/build_card_store.py",', '"python3", "-m", "verifiquant.card_store",'),
        ('"python3", "preprocessing/run_framework_guided_self_improve_pipeline.py",',
         '"python3", "-m", "verifiquant.pipeline.run_framework_guided_self_improve_pipeline",'),
        ('script = "preprocessing/run_cot_self_improve_pipeline.py"', 'script = "verifiquant.pipeline.run_cot_self_improve_pipeline"'),
        ('"python3", script,', '"python3", "-m", script,'),
    ],
    "scripts/supervise_card_build.sh": [
        (".venv/bin/python preprocessing/dataset_case_to_fic.py", ".venv/bin/python -m verifiquant.preprocessing.dataset_case_to_fic"),
    ],
    # The public tree ships the camera-ready source, not the submitted main_final.tex.
    "scripts/assert_paper_numbers.py": [
        ('os.environ.get("PAPER_TEX", "paper/main_final.tex")', 'os.environ.get("PAPER_TEX", "paper/main_camera_ready_9p.tex")'),
    ],
    # Moved from the repo root into tests/.
    "tests/test_transform_poc.py": [
        ("sys.path.insert(0, str(Path(__file__).parent))", "sys.path.insert(0, str(Path(__file__).resolve().parent.parent))"),
        ("Run: python3 test_transform_poc.py", "Run: python3 tests/test_transform_poc.py"),
    ],
}
# Patterns that must not appear anywhere in the public tree.
FORBIDDEN = {
    "absolute home path": re.compile(r"/Users/[A-Za-z0-9_.-]+"),
    "Google API key": re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),
    "OpenAI API key": re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
    "private e-mail": re.compile(r"vincent990622@"),
}


def excluded(rel: str) -> bool:
    return any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch("/" + rel, pat) for pat in RUN_EXCLUDE)


def run_files() -> Iterable[Tuple[Path, str]]:
    for run, globs in RUN_DIRS.items():
        base = RUNS / run
        if not base.is_dir():
            raise SystemExit(f"missing run directory: {base}")
        for path in sorted(p for p in base.rglob("*") if p.is_file()):
            rel = path.relative_to(base).as_posix()
            if excluded(f"{run}/{rel}"):
                continue
            # fnmatch's "*" also matches "/", so "results/*" covers the whole results/ subtree.
            if globs is not None and not any(fnmatch.fnmatch(rel, g) for g in globs):
                continue
            yield path, f"results/{run}/{rel}"
    for name in RUN_FILES:
        yield RUNS / name, f"results/{name}"


def all_sources() -> List[Tuple[Path, str]]:
    pairs: List[Tuple[Path, str]] = [(ROOT / p, p) for p in PACKAGE]
    pairs += [(ROOT / "scripts" / s, f"scripts/{s}") for s in SCRIPTS]
    pairs += [(ROOT / src, dst) for src, dst in OTHER.items()]
    pairs += [(ROOT / src, dst) for src, dst in PAPER.items()]
    for path in sorted(p for p in (ROOT / "datasets").rglob("*") if p.is_file()):
        rel = path.relative_to(ROOT).as_posix()
        if not excluded(rel):
            pairs.append((path, rel))
    pairs += list(run_files())
    for path in sorted(p for p in (ROOT / "release").rglob("*") if p.is_file()):
        pairs.append((path, path.relative_to(ROOT / "release").as_posix()))
    missing = [str(src) for src, _ in pairs if not src.exists()]
    if missing:
        raise SystemExit("missing sources:\n  " + "\n  ".join(missing))
    return pairs


def rewrite(data: bytes, dst: str) -> bytes:
    text = data.decode("utf-8")
    for old, new in FILE_REWRITES.get(dst, []):
        if old not in text:
            raise SystemExit(f"{dst}: expected to find {old!r}; update FILE_REWRITES")
        text = text.replace(old, new)
    for old, new in REWRITES:
        text = text.replace(old, new)
    return text.encode("utf-8")


def is_text(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES


def rehash_manifest(target: Path) -> int:
    """Refresh sha256/size in datasets/MANIFEST.json for files the rewrite changed."""
    manifest_path = target / "datasets/MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("files", manifest)
    changed = 0
    items = entries.items() if isinstance(entries, dict) else ((e.get("path"), e) for e in entries)
    for rel, entry in items:
        if not isinstance(entry, dict) or "sha256" not in entry:
            continue
        path = target / "datasets" / rel
        if not path.exists():
            continue
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != entry["sha256"]:
            entry["sha256"] = digest
            if "size" in entry:
                entry["size"] = len(data)
            if "bytes" in entry:
                entry["bytes"] = len(data)
            changed += 1
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return changed


def scan(target: Path) -> List[str]:
    hits = []
    for path in sorted(p for p in target.rglob("*") if p.is_file() and ".git" not in p.parts):
        if not is_text(path):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for name, pattern in FORBIDDEN.items():
            m = pattern.search(text)
            if m:
                hits.append(f"{path.relative_to(target)}: {name}: {m.group(0)[:40]}")
    return hits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=Path, default=ROOT.parent / "icaif-verifiquant")
    ap.add_argument("--dry-run", action="store_true", help="list what would be written and exit")
    args = ap.parse_args()
    target = args.target.resolve()
    if not (target / ".git").exists():
        raise SystemExit(f"{target} is not a git checkout; clone the public repo there first")

    pairs = all_sources()
    dests = [d for _, d in pairs]
    dupes = {d for d in dests if dests.count(d) > 1}
    if dupes:
        raise SystemExit(f"two sources map to the same destination: {sorted(dupes)}")
    total = sum(src.stat().st_size for src, _ in pairs)
    if args.dry_run:
        for src, dst in pairs:
            print(f"{src.stat().st_size:>10}  {dst}")
        print(f"{len(pairs)} files, {total / 2**20:.1f} MiB")
        return

    written = set()
    for src, dst in pairs:
        out = target / dst
        out.parent.mkdir(parents=True, exist_ok=True)
        data = src.read_bytes()
        out.write_bytes(rewrite(data, dst) if is_text(src) else data)
        shutil.copymode(src, out)
        written.add(dst)
    changed = rehash_manifest(target)

    stale = sorted(
        p.relative_to(target).as_posix() for p in target.rglob("*")
        if p.is_file() and ".git" not in p.parts and p.relative_to(target).as_posix() not in written
    )
    hits = scan(target)
    print(f"wrote {len(written)} files ({total / 2**20:.1f} MiB) to {target}; re-hashed {changed} manifest entries")
    if stale:
        print("files in the target that this export did not write (review before committing):")
        for s in stale:
            print("  " + s)
    if hits:
        print("FORBIDDEN CONTENT FOUND:")
        for h in hits:
            print("  " + h)
        sys.exit(1)
    print("scan: no absolute home paths, API keys or private e-mail addresses")


if __name__ == "__main__":
    main()
