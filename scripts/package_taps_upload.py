#!/usr/bin/env python3
"""
package_taps_upload.py
----------------------
Build the ACM TAPS upload for ICAIF '26 (proceedings acronym icaif26, TAPS paper ID 13) from
paper/main_camera_ready_9p.tex, in the layout TAPS asks for:

    icaif26-13.zip   (pdf/ and Source/ sit at the top of the zip, with no wrapper folder)
      pdf/icaif26-13.pdf
      Source/icaif26-13.tex  icaif26-13.bbl  references.bib  <the figures used, as PNG, flat, no subfolder>

TAPS recompiles the source with its own acmart, so the source must stand alone. The upload copy
(the working file is never modified) therefore

  1. resolves the arXiv switch to the ACM branch: \\extref{app:x} becomes the literal "App.~X",
     and the \\ifarxiv machinery, the listings package, and the appendix \\input are removed;
  2. drops every comment-only line and empties inline comments, keeping the "%" itself so that
     line ends behave exactly as before.

Two variants exist because TAPS prints the affiliations entered in its own system, not the ones in
the source (10/18). In TAPS, Mi-Yen Yeh has two affiliations, which adds two lines to the author
block; the long-affiliation variant reproduces that block and rewords three paragraphs (no claim,
number, or reference removed) so the paper still fits in 9 pages. The short-affiliation variant is
the working file as it stands, for use if TAPS agrees to the single affiliation.

For each variant the script compiles the cleaned source twice in empty directories: once with the
original vector figures (the reference) and once from the package alone, with PNG figures. It checks
9 pages, no undefined reference, no LaTeX error, and that every page of the package PDF has the same
text as the reference apart from the words inside the figures. For the short variant the reference
must also match paper/main_camera_ready_9p.pdf word for word, so run paper/build_camera_ready.sh
first. The arXiv PDF is copied next to the zip as an optional supplement; it is not in the zip.

Usage: .venv/bin/python scripts/package_taps_upload.py --variant long-affiliation [--tag ...]
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
MAIN = PAPER / "main_camera_ready_9p.tex"
NAME = "icaif26-13"
PAGE_LIMIT = 9
ZIP_LIMIT = 10 * 1024 * 1024

YEH_SHORT = r"""\author{Mi-Yen Yeh}
\orcid{0000-0001-7707-4487}
\email{miyen@iis.sinica.edu.tw}
\affiliation{
  \department{Data Science Degree Program}
  \institution{National Taiwan University and Academia Sinica}
  \city{Taipei}
  \country{Taiwan}
}
"""
YEH_LONG = r"""\author{Mi-Yen Yeh}
\orcid{0000-0001-7707-4487}
\affiliation{
  \department{Institute of Information Science}
  \institution{Academia Sinica}
  \city{Taipei}
  \country{Taiwan}
}
\affiliation{
  \department{Data Science Degree Program}
  \institution{National Taiwan University}
  \city{Taipei}
  \country{Taiwan}
}
\email{miyen@iis.sinica.edu.tw}
"""
# Each pair must match the working file exactly once.
VARIANTS = {
    "short-affiliation": [],
    # The three rewordings that made this variant fit are in the working file since 10/20, because the
    # long-affiliation upload is the published text (TAPS froze the body after the 10/18 18:42 proof).
    "long-affiliation": [(YEH_SHORT, YEH_LONG)],
}


def acm_source(tex: str) -> str:
    letters = dict(re.findall(r"\\expandafter\\def\\csname extletter-(app:[a-z]+)\\endcsname\{([A-Z])\}", tex))
    assert len(letters) == 11, f"expected the 11 fixed appendix letters, found {len(letters)}"
    tex = re.sub(r"\\extref\{(app:[a-z]+)\}", lambda m: "App.~" + letters[m.group(1)], tex)
    for line in (r"\newif\ifarxiv", r"\arxivfalse", r"\ifarxiv\usepackage{listings}\fi",
                 r"\newcommand{\extref}[1]{App.~\ifarxiv\ref{#1}\else\csname extletter-#1\endcsname\fi}"):
        assert tex.count(line + "\n") == 1, f"not found exactly once: {line}"
        tex = tex.replace(line + "\n", "")
    tex = re.sub(r"\\expandafter\\def\\csname extletter-app:[a-z]+\\endcsname\{[A-Z]\}\n", "", tex)
    assert "\\TODO{" not in tex.replace("\\newcommand{\\TODO}", ""), "a \\TODO{...} is still used in the paper"
    tex = tex.replace("\\newcommand{\\TODO}[1]{\\textbf{[TODO:~#1]}}\n", "")
    i = tex.index("\\ifarxiv\n"); j = tex.index("\\fi\n", i) + len("\\fi\n")
    assert "\\input{appendix_camera_ready}" in tex[i:j], "the arXiv-only tail block moved"
    tex = tex[:i] + tex[j:]
    out = []
    for line in tex.splitlines():
        if re.match(r"\s*%", line):
            continue
        out.append(re.sub(r"(?<!\\)%.*$", "%", line))
    tex = re.sub(r"\n{3,}", "\n\n", "\n".join(out) + "\n")
    for gone in ("ifarxiv", "extref", "extletter", "\\input", "listings", "TODO"):
        assert gone not in tex, f"left in the upload source: {gone}"
    return tex


def run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def page_words(pdf: Path, n: int):
    norm = lambda t: re.sub(r"USA, \d+ pages\.", "", t)  # noqa: E731  (the page count in the ACM reference line)
    return [Counter(norm(run(["pdftotext", "-f", str(p), "-l", str(p), str(pdf), "-"], ROOT).stdout).split())
            for p in range(1, n + 1)]


def compile_alone(files: Path, out_pdf: Path):
    """Compile NAME.tex from a copy of `files` in an empty directory; return (log, bbl text)."""
    with tempfile.TemporaryDirectory() as w:
        w = Path(w)
        shutil.copytree(files, w, dirs_exist_ok=True, symlinks=True)
        for f in ("hyperxmp.sty", "binhex.tex"):  # only fill gaps in the local TeX tree
            hit = next(iter(Path.home().glob(f"Library/Caches/Tectonic/**/{f}")), None)
            if hit:
                shutil.copy(hit, w / f)
        for cmd in (["pdflatex", "-interaction=nonstopmode", NAME], ["bibtex", NAME],
                    ["pdflatex", "-interaction=nonstopmode", NAME], ["pdflatex", "-interaction=nonstopmode", NAME]):
            run(cmd, w)
        shutil.copy(w / f"{NAME}.pdf", out_pdf)
        return (w / f"{NAME}.log").read_text(encoding="utf-8", errors="ignore"), (w / f"{NAME}.bbl").read_text(encoding="utf-8")


def pages(pdf: Path) -> int:
    return int(re.search(r"Pages:\s+(\d+)", run(["pdfinfo", str(pdf)], ROOT).stdout).group(1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    ap.add_argument("--tag", default=None, help="output folder under taps_upload/ (default: 2026-10-18_<variant>)")
    args = ap.parse_args()
    out_dir = ROOT / "taps_upload" / (args.tag or f"2026-10-18_{args.variant}")
    pkg = out_dir / NAME
    if pkg.exists():
        shutil.rmtree(pkg)
    src_dir, pdf_dir = pkg / "Source", pkg / "pdf"
    src_dir.mkdir(parents=True)
    pdf_dir.mkdir(parents=True)

    raw = MAIN.read_text(encoding="utf-8")
    for old, new in VARIANTS[args.variant]:
        assert raw.count(old) == 1, f"variant edit does not match exactly once: {old[:70]}"
        raw = raw.replace(old, new)
    tex = acm_source(raw)

    # reference: the same cleaned source with the original vector figures
    with tempfile.TemporaryDirectory() as r:
        r = Path(r)
        (r / f"{NAME}.tex").write_text(tex, encoding="utf-8")
        shutil.copy(PAPER / "references.bib", r)
        (r / "figures").symlink_to(PAPER / "figures")
        ref = out_dir / "reference_vector_figures.pdf"
        out_dir.mkdir(parents=True, exist_ok=True)
        compile_alone(r, ref)
    # TAPS does not see subfolders of Source/ (10/18: "File `figures/pipeline_v6.pdf' not found"),
    # so every figure sits next to the .tex and is referenced by its bare file name.
    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    names = [Path(f).name for f in figs]
    assert len(set(names)) == len(names), f"two figures share a file name: {names}"
    # TAPS also failed to find the PDF figures once they sat flat in Source/ (it rewrites the paths into
    # its own figures/ folder and evidently does not carry PDF files there), so each figure ships as a
    # high-resolution PNG, a format TAPS accepts everywhere. Widths in the .tex are unchanged.
    for i, (f, n) in enumerate(zip(figs, names)):
        stem = Path(n).stem
        dpi = 300 if (PAPER / f).stat().st_size > 50_000 else 600  # the large diagram renders at ~800 dpi on paper anyway
        run(["pdftoppm", "-png", "-r", str(dpi), "-singlefile", str(PAPER / f), str(src_dir / stem)], ROOT)
        assert (src_dir / f"{stem}.png").exists(), f"PNG conversion failed for {f}"
        names[i] = f"{stem}.png"
        tex = tex.replace("{" + f + "}", "{" + names[i] + "}")
    assert "figures/" not in tex, "a path into figures/ is left in the upload source"
    (src_dir / f"{NAME}.tex").write_text(tex, encoding="utf-8")
    shutil.copy(PAPER / "references.bib", src_dir)

    # compile from the package alone
    log, bbl = compile_alone(src_dir, pdf_dir / f"{NAME}.pdf")
    undefined = re.findall(r"(?:Reference|Citation) [^\n]* undefined", log)
    errors = [l for l in log.splitlines() if l.startswith("! ") and "Loading a class or package in a group" not in l]
    (src_dir / f"{NAME}.bbl").write_text(bbl, encoding="utf-8")

    pdf = pdf_dir / f"{NAME}.pdf"
    n = pages(pdf)
    # The PNG figures carry no extractable text, so a page may lack exactly the words printed inside the
    # vector figures and nothing else; any other difference means the layout moved.
    fig_words = Counter()
    for f in figs:
        fig_words += Counter(run(["pdftotext", str(PAPER / f), "-"], ROOT).stdout.split())
    same = pages(ref) == n and all(not (a - b) and not ((b - a) - fig_words)
                                   for a, b in zip(page_words(pdf, n), page_words(ref, n)))
    if args.variant == "short-affiliation":  # the cleaning must not change a single word of the ACM build
        acm = PAPER / "main_camera_ready_9p.pdf"
        same = same and pages(acm) == n and page_words(ref, n) == page_words(acm, n)

    zpath = out_dir / f"{NAME}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(pkg.rglob("*")):
            if f.is_file() and f.name != ".DS_Store":
                z.write(f, f.relative_to(pkg))  # TAPS wants pdf/ and Source/ at the top of the zip, no wrapper folder
    supp = out_dir / "supplement_optional"
    supp.mkdir(exist_ok=True)
    shutil.copy(PAPER / "main_camera_ready_9p_arxiv.pdf", supp / f"{NAME}-extended-with-appendix.pdf")

    print(f"source: {len(tex.splitlines())} lines, figures (flat in Source/): {', '.join(names)}")
    print(f"pages: {n} (limit {PAGE_LIMIT}) | undefined refs/cites: {len(undefined)} | LaTeX errors: {len(errors)}")
    print(f"variant: {args.variant} ({len(VARIANTS[args.variant])} edits to the working file)")
    print(f"every page matches the vector-figure reference apart from the words inside the figures"
          f"{' (and the reference matches paper/main_camera_ready_9p.pdf)' if args.variant == 'short-affiliation' else ''}: {same}")
    print(f"zip: {zpath.relative_to(ROOT)} ({zpath.stat().st_size / 1024:.0f} KB, limit 10 MB)")
    with zipfile.ZipFile(zpath) as z:
        for i in z.infolist():
            print(f"  {i.filename}  ({i.file_size / 1024:.0f} KB)")
    ok = n <= PAGE_LIMIT and not undefined and not errors and same and zpath.stat().st_size < ZIP_LIMIT
    print("READY" if ok else "NOT READY: fix the items above")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
