#!/usr/bin/env python3
"""
build_cut_proposals.py
----------------------
Write docs/2026-10-17_cut_proposals.md: every proposed cut for the 9-page ACM build, with the full
paragraph before and after (no ellipses), a word-level diff, where removed information goes, and
the number of lines it actually saves.

Lines saved are measured, not estimated: each proposal is applied alone to a copy of the paper,
the ACM build is compiled with pdfLaTeX, and the script counts the text lines of the whole PDF
(running heads excluded). A paragraph's line count depends only on the column width, so this
count is additive and does not jump when a float moves to another page, unlike the position of the
References heading, which acmart's flush-bottom columns and float placement make step-like.
Whether the ACM build fits in nine pages is then checked by compiling cumulative combinations.
Nothing in paper/ is modified.

Usage: .venv/bin/python scripts/build_cut_proposals.py
"""
from __future__ import annotations

import difflib
import html
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
MAIN = PAPER / "main_camera_ready_9p.tex"
APP = PAPER / "appendix_camera_ready.tex"
OUT = ROOT / "docs/2026-10-17_cut_proposals.md"

BASE = MAIN.read_text(encoding="utf-8")
BASE_APP = APP.read_text(encoding="utf-8")


def span(start: str, end: str, src: str = BASE) -> str:
    i = src.index(start)
    j = src.index(end, i) + len(end)
    return src[i:j]


# ── the proposals ──────────────────────────────────────────────────────────────
# Each: id, title, kind, edits on the body [(old, new)], edits on the appendix, and where any
# removed information goes. Old strings must occur exactly once.
P: List[Dict] = []

P.append(dict(id="A1", title="§3.3 修復段（尾行 wild.）", body=[
    ("Repairs hence use AST-bounded transforms.", "Repairs use AST-bounded transforms."),
    ("This controlled distribution probes verifier mechanisms.", "This controlled set probes verifier mechanisms.")],
    dest="只刪連接詞與同義詞，兩句界線說法與 879 分母都保留。"))
P.append(dict(id="A2", title="§2 範圍訊號定義（尾行 to CoT.）", body=[
    ("It is defined only for systems that carry a library, so it does not apply to CoT.",
     "It is defined only for systems that carry a library, so not for CoT.")],
    dest="意思不變。"))
P.append(dict(id="A3", title="Table 6 圖說（尾行 result.）", body=[
    (r"Every $K{>}1$ row is an intent-oracle upper bound, not an autonomous result.}" "\n" r"\label{tab:v2kabl}",
     r"Every $K{>}1$ row is an intent-oracle upper bound.}" "\n" r"\label{tab:v2kabl}")],
    dest="oracle 上限的標註保留（校準規則）；「not an autonomous result」在 Table 2 圖說與 §4.4 都有。"))
P.append(dict(id="A4", title="§5.2 Pro 那句（尾行 version).）", body=[
    (r"Repeating the trap comparison on Gemini 2.5 Pro narrows the gap to \proGap{} points (R-PAL$^{+}$ \proPromptFnar{}, VerifiQuant \proVQfnar{}, 95\% CI $[\proCIlo{},\proCIhi{}]$), and nearly all of what remains lies in I (\extref{app:pro}).",
     r"On Gemini 2.5 Pro the gap narrows to \proGap{} points (R-PAL$^{+}$ \proPromptFnar{}, VerifiQuant \proVQfnar{}, 95\% CI $[\proCIlo{},\proCIhi{}]$), nearly all of it in I (\extref{app:pro}).")],
    dest="數字全保留，意思不變。"))
P.append(dict(id="A5", title="§3.2 N/M 段（尾行 is resolved.）", body=[
    ("Nothing in the retriever is trained, tuned, or fitted to the evaluation sets",
     "Nothing in the retriever is fitted to the evaluation sets")],
    dest="「fitted」已涵蓋 trained 與 tuned。"))
P.append(dict(id="A6a", title="§5 開頭，修尾行（尾行 human set.）", body=[
    ("The results are read by library status. RQ1 and RQ2 stay in-library, RQ3 moves to the boundary, and RQ4 uses the library-agnostic human set.",
     "RQ1 and RQ2 stay in-library, RQ3 moves to the library boundary, and RQ4 uses the library-agnostic human set.")],
    dest="意思不變。"))
P.append(dict(id="A6b", title="§5 開頭，整段刪（A6a 的替代方案）", body=[
    ("The results are read by library status. RQ1 and RQ2 stay in-library, RQ3 moves to the boundary, and RQ4 uses the library-agnostic human set.\n", "")],
    dest="小節標題 RQ1–RQ4 已說明順序與庫內、邊界、人工集的區分。"))
P.append(dict(id="A7", title="貢獻三（尾行 vouches for.）", body=[
    ("the system's statement that the library it was given holds no applicable contract",
     "the system's statement that its library holds no applicable contract")],
    dest="意思不變。"))
P.append(dict(id="A8", title="§5.3 gate 消融（尾行 not accuracy.）", body=[
    (r"It is a dedicated autonomous $K{=}1$ run over all 250 clean questions with Gemini 2.5 Flash and the full contract library, using the same output-scale-normalized accuracy and SWR definitions as the main results.",
     r"It is a dedicated autonomous $K{=}1$ run over all 250 clean questions with the full contract library and the same scoring as the main results.")],
    dest="backbone 在 §5.1 已講（Every other system comparison holds the backbone fixed at Flash），計分定義在 §4.2。"))
P.append(dict(id="A9", title="§4.5 第二段（尾行 extended version.）", body=[
    (r"The full construction, label definitions, and scoring rules are in \extref{app:human}.",
     r"Full details are in \extref{app:human}.")],
    dest="意思不變。"))
P.append(dict(id="A10", title="結論（尾行 binding constraint.）", body=[
    ("VerifiQuant compiles scope, schema, boundary, and ambiguity checks into an executable funnel",
     "VerifiQuant compiles its checks into an executable funnel")],
    dest="四類檢查在摘要與貢獻一已列。"))

b1_old = span("The two systems therefore serve different engineering objectives.", "(\\S\\ref{sec:v2res}).\n\n")
P.append(dict(id="B1", title="導論「兩個系統服務不同目標」整段", body=[(b1_old, "")],
    dest="1.5 倍延遲來自 §5.3（12.2 秒對 8.2 秒），少 9 點覆蓋率在 Table 2 與 §5.3（92.9% 對 84.0%），"
         "「依部署目標取捨」在結論（Whether that exchange is worth making is a property of the deployment）。"
         "刪掉後也解決這段以 therefore 接在法規句後、語意接不上的問題。"))

b2_old = span("Sweeping a self-consistency threshold", "is not a conformal procedure.")
P.append(dict(id="B2", title="§2 自洽性細節", body=[
    (b2_old, r"On the same 250 clean questions, a five-sample self-consistency threshold set to VerifiQuant's $84.0\%$ coverage holds \scSelAccTight{} selective accuracy against $96.7\%$, because \scUnanimous{} of the 250 are already unanimous (\extref{app:evals}).")],
    app=[(r"\section{What each evaluation is for}\label{app:evals}" "\n",
          r"\section{What each evaluation is for}\label{app:evals}" "\n"
          r"\paragraph{Self-consistency reference point.} Sweeping a self-consistency threshold (five samples, vote share as the score) over the 250 clean questions moves coverage only from \scCovOpen{} to \scCovTight{} and SWR from \scSWRopen{} to \scSWRtight{}, because \scUnanimous{} of the 250 are already unanimous and the score has little resolution left to spend. That reference point carries no retrieval and is not a conformal procedure." "\n\n")],
    dest="掃描端點（覆蓋率、SWR 的起訖）與「不含檢索、不是 conformal」移到附錄 H。"))

b3a = span("Of the design targets in \\S1, a traceable record", "\\cite[Art.~12]{euaiact2024}. ")
b3b = span("When a number is emitted, the report records", "and what the user should supply.")
P.append(dict(id="B3", title="§3.2 Audit trail", body=[
    (b3a, ""),
    ("At the query level, every output carries a \\emph{DiagnosticReport}. " + b3b,
     "At the query level, every output carries a \\emph{DiagnosticReport} that records the contract and its version, the retrieved candidates and the selector's decision, the bound values, and each check passed or failed, with the evidence behind a stop and what the user should supply.")],
    app=[(r"This appendix expands the design consideration stated in the introduction." "\n",
          r"This appendix expands the design consideration stated in the introduction." "\n"
          r"Of the six targets, a traceable record (C5) is the one most directly written into regulation: the EU AI Act requires high-risk systems to log events automatically over their lifetime for traceability \cite[Art.~12]{euaiact2024}. " + b3b + "\n")],
    dest="AI Act 第 12 條那句與完整的報告欄位清單移到附錄 G（法規背景，對應 C5）。可重放的界線說法完整保留。提醒：這段對應設計目標 C5，審稿人可能在意。"))

b4_old = span("The comparison there is matched in the same way as the clean set.", "differ only in whether the funnel runs.")
P.append(dict(id="B4", title="§4.1 OOD split 的配對說明", body=[
    (b4_old, r"Both systems run the same backbone on the same 199 unchanged questions and retrieve top-4 from the \emph{same} 50-family library, so they differ only in whether the funnel runs.")],
    dest="同段三句講同一件事，合併成一句，無資訊損失。"))

b5_old = span(r"The $1\%$ tolerance is a general-purpose threshold", "independent of the tolerance entirely.")
P.append(dict(id="B5", title="§4.2 容忍度", body=[
    (b5_old, r"One scorer applies the $1\%$ tolerance to every system, and FNAR does not depend on it, because it counts only whether a number was emitted.")],
    app=[(r"\section{What each evaluation is for}\label{app:evals}" "\n",
          r"\section{What each evaluation is for}\label{app:evals}" "\n"
          r"The $1\%$ tolerance of \S\ref{sec:v2baselines} is a general-purpose threshold rather than the stricter convention the benchmark applies to its own leaderboard." "\n\n")],
    dest="「比 benchmark 排行榜慣例寬鬆」那句移到附錄 H。"))

b6a = span("These paired tests use per-question binary outcomes", "not averages over repeats.")
b6b = "\nClean scalar accuracy is the wrong axis on which to measure governance, and instead, the systems must be separated where an answer should not be emitted."
P.append(dict(id="B6", title="§5.1 RQ1 刪兩句", body=[(" " + b6a, ""), (b6b, "")],
    app=[(r"\section{Run-level results and retrieval depth}\label{app:reruns}" "\n",
          r"\section{Run-level results and retrieval depth}\label{app:reruns}" "\n"
          r"The paired tests of \S\ref{sec:v2res} " + b6a[len("These paired tests "):] + "\n")],
    dest="配對檢定的資料來源移到附錄 D；最後一句和前一句「Standing on the clean set therefore does not predict governed behavior」同義。"))

b7_old = span("These are wall-clock means that include provider-side variance", "not comparable (\\extref{app:reruns}).")
P.append(dict(id="B7", title="§5.3 延遲但書", body=[
    (b7_old, r"These wall-clock means come from an earlier sequential run of the ladder and bound the order of magnitude rather than the marginal cost of a deployment (\extref{app:reruns}).")],
    app=[("because the depth-4 rerun executed its rungs concurrently.",
          "because the depth-4 rerun executed its rungs concurrently. They are wall-clock means that include provider-side variance and retries, and we do not report token accounting.")],
    dest="「含供應商端變異與重試」「未計 token」併入附錄 D（附錄 D 原本已有 depth 3 循序、depth 4 並行的說明）。"))

b8_old = span("This is the same exchange as the out-of-library split", "rather than only its benefit.")
P.append(dict(id="B8", title="§5.4 最後一段", body=[
    (b8_old, "As on the out-of-library split, parametric knowledge buys coverage without a signal, and contract grounding buys a signal without coverage.")],
    dest="「量兩次才能講出邊界的價格」在貢獻三（we put a price on the boundary）與 §6 已講；保留金句。"))

P.append(dict(id="B9", title="§6 Where the claim holds 最後一句", body=[
    ("\nThe clean-set result is in-library by construction, and out-of-library behavior is exactly the trade-off examined in RQ3.", "")],
    dest="in-library by construction 在 §4.1，庫外取捨在 RQ3。"))

# C: how appendix pointers read in the ACM version
first_ext = r"marked C1--C6 in Figure~\ref{fig:v2pipeline} and detailed in \extref{app:regulatory}."
P.append(dict(id="C1", title="附錄引用縮短成「App. X」，第一次出現時加註腳說明延伸版在哪裡", body=[
    (r"\newcommand{\extref}[1]{Appendix~\ifarxiv\ref{#1}\else\csname extletter-#1\endcsname\fi{} of the extended version}",
     r"\newcommand{\extref}[1]{App.~\ifarxiv\ref{#1}\else\csname extletter-#1\endcsname\fi}"),
    (first_ext, first_ext[:-1] + r"\footnote{Appendices A--K are in the extended version, released with the code at \url{https://github.com/datafox-tw/icaif-verifiquant}.}.")],
    dest="14 處「Appendix X of the extended version」改成「App. X」，延伸版位置在第一次出現時的註腳說明一次。延伸版 PDF 需要放進 repo（或日後補上 arXiv 連結）。",
    macro=True))
P.append(dict(id="C2", title="正文完全不提附錄（C1 的替代方案）", body="REMOVE_ALL_EXTREF",
    dest="14 處附錄引用全部刪除。附錄內容仍在 arXiv 版，但 ACM 讀者不會知道。", macro=True))

RECOMMENDED = ["A1", "A2", "A3", "A4", "A5", "A6a", "A7", "A8", "A9", "A10", "B1", "B4", "B6"]


def remove_all_extref(src: str) -> str:
    body_start, body_end = src.index(r"\begin{document}"), src.index(r"\bibliographystyle")
    body = src[body_start:body_end]
    body = re.sub(r",? and detailed in \\extref\{[^}]*\}", "", body)
    body = re.sub(r"\s*\(\\extref\{[^}]*\}\)", "", body)
    body = re.sub(r",? summarized in \\extref\{[^}]*\}", "", body)
    body = re.sub(r"\s*Full details are in \\extref\{[^}]*\}\.", "", body)
    body = re.sub(r"\s*The full construction, label definitions, and scoring rules are in \\extref\{[^}]*\}\.", "", body)
    body = re.sub(r",? and print every prompt in \\extref\{[^}]*\}", "", body)
    body = re.sub(r"\(\\extref\{app:reruns\}\)", "", body)
    assert "\\extref{" not in body, re.findall(r".{60}\\extref\{[^}]*\}.{20}", body)
    return src[:body_start] + body + src[body_end:]


def extref_sentences(src: str) -> List[str]:
    """Every body sentence that contains an \\extref, split at sentence ends (not at decimals)."""
    body = src[src.index(r"\begin{document}"):src.index(r"\bibliographystyle")]
    body = re.sub(r"(?<!\\)%.*", "", body)
    out = []
    for m in re.finditer(r"\\extref\{[^}]*\}", body):
        a = max(x.end() for x in [re.match("", "")] + list(re.finditer(r"(?<=[a-z0-9\)\}\]]\.)\s+(?=[A-Z\\])|\n\n", body[:m.start()])))
        b = re.search(r"(?<=[a-z0-9\)\}\]]\.)(\s|$)", body[m.end() - 1:])
        out.append(body[a:m.end() - 1 + (b.start() if b else 1)].strip())
    return out


def apply(src: str, app: str, prop: Dict) -> Tuple[str, str]:
    if prop["body"] == "REMOVE_ALL_EXTREF":
        return remove_all_extref(src), app
    for old, new in prop["body"]:
        assert src.count(old) == 1, (prop["id"], src.count(old), old[:80])
        src = src.replace(old, new)
    for old, new in prop.get("app", []):
        assert app.count(old) == 1, (prop["id"], "appendix", app.count(old), old[:80])
        app = app.replace(old, new)
    return src, app


# ── compile and measure ────────────────────────────────────────────────────────
WORK = Path(tempfile.mkdtemp(prefix="vq_cuts_"))
for f in ("hyperxmp.sty", "binhex.tex"):
    hit = next(iter(Path.home().glob(f"Library/Caches/Tectonic/**/{f}")), None)
    if hit:
        shutil.copy(hit, WORK / f)
(WORK / "figures").symlink_to(PAPER / "figures")
shutil.copy(PAPER / "references.bib", WORK / "references.bib")


def build(tex: str, name: str, bib: bool = False) -> Path:
    (WORK / f"{name}.tex").write_text(tex, encoding="utf-8")
    run = lambda *a: subprocess.run(a, cwd=WORK, capture_output=True, text=True)  # noqa: E731
    run("pdflatex", "-interaction=nonstopmode", f"{name}.tex")
    if bib:
        run("bibtex", name)
    else:
        shutil.copy(WORK / "base.bbl", WORK / f"{name}.bbl")
    run("pdflatex", "-interaction=nonstopmode", f"{name}.tex")
    run("pdflatex", "-interaction=nonstopmode", f"{name}.tex")
    return WORK / f"{name}.pdf"


def references_position(pdf: Path) -> Tuple[int, float, float, float]:
    """(page, x, y) of the References heading and the body line pitch, from pdftotext -bbox."""
    xml = subprocess.run(["pdftotext", "-bbox", str(pdf), "-"], capture_output=True, text=True).stdout
    pages = xml.split("<page ")[1:]
    for pi, pg in enumerate(pages, 1):
        for m in re.finditer(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="[\d.]+" yMax="[\d.]+">References</word>', pg):
            return pi, float(m.group(1)), float(m.group(2)), 0.0
    raise RuntimeError("References heading not found")


def page_geometry(pdf: Path) -> Tuple[float, float, float, float]:
    """Text-block top, bottom, line pitch, and page width, measured on page 3 of the base build."""
    xml = subprocess.run(["pdftotext", "-bbox", "-f", "3", "-l", "3", str(pdf), "-"], capture_output=True, text=True).stdout
    width = float(re.search(r'<page width="([\d.]+)"', xml).group(1))
    ys = sorted({round(float(y), 1) for y in re.findall(r'yMin="([\d.]+)"', xml)})
    body = [y for y in ys if y > 70]
    gaps = [round(b - a, 1) for a, b in zip(body, body[1:]) if 9 < b - a < 13]
    pitch = max(set(gaps), key=gaps.count)
    return min(body), max(body), pitch, width


def linear(pdf: Path, geo) -> float:
    top, bottom, pitch, width = geo
    page, x, y, _ = references_position(pdf)
    col = 0 if x < width / 2 else 1
    per_col = (bottom - top) / pitch + 1
    return ((page - 1) * 2 + col) * per_col + (y - top) / pitch


def text_lines(pdf: Path) -> int:
    """All text lines of the PDF except running heads (top 60 pt of each page)."""
    xml = subprocess.run(["pdftotext", "-bbox-layout", str(pdf), "-"], capture_output=True, text=True).stdout
    return sum(1 for y in re.findall(r'<line xMin="[\d.]+" yMin="([\d.]+)"', xml) if float(y) > 60)


def pages(pdf: Path) -> int:
    info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    return int(re.search(r"Pages:\s+(\d+)", info).group(1))


base_pdf = build(BASE, "base", bib=True)
BASE_LINES = text_lines(base_pdf)
BASE_PAGES = pages(base_pdf)
saved: Dict[str, float] = {}
for prop in P:
    tex, _ = apply(BASE, BASE_APP, prop)
    saved[prop["id"]] = BASE_LINES - text_lines(build(tex, f"p_{prop['id']}"))

# cumulative path, least intrusive first; compile after each step and record the page count
PATH = RECOMMENDED + ["B9", "B8", "B7", "B5", "C1", "B2", "B3"]
path_rows = []
cum_tex, cum_app = BASE, BASE_APP
for k, pid in enumerate(PATH, 1):
    cum_tex, cum_app = apply(cum_tex, cum_app, next(p for p in P if p["id"] == pid))
    if pid in RECOMMENDED and pid != RECOMMENDED[-1]:
        continue
    pdf = build(cum_tex, f"cum_{k}")
    path_rows.append((pid, BASE_LINES - text_lines(pdf), pages(pdf)))
combo_saved, combo_pages = path_rows[0][1], path_rows[0][2]

# ── readable text ──────────────────────────────────────────────────────────────
MACRO = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", BASE))
aux = (WORK / "base.aux").read_text(encoding="utf-8", errors="ignore")
LABEL = {m.group(1): m.group(2) for m in re.finditer(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", aux)}
LETTER = dict(re.findall(r"extletter-(app:[a-z]+)\\endcsname\{([A-Z])\}", BASE))


def readable(t: str) -> str:
    t = re.sub(r"(?<!\\)%.*", "", t)
    t = re.sub(r"\\(sub)*section\*?\{[^}]*\}", "", t)
    t = re.sub(r"\\label\{[^}]*\}", "", t)
    t = re.sub(r"\\extref\{([^}]+)\}", lambda m: f"Appendix {LETTER.get(m.group(1), '?')} of the extended version", t)
    t = re.sub(r"\\S\\ref\{([^}]+)\}", lambda m: "§" + LABEL.get(m.group(1), "?"), t)
    t = re.sub(r"(Table|Figure)~\\ref\{([^}]+)\}", lambda m: f"{m.group(1)} {LABEL.get(m.group(2), '?')}", t)
    t = re.sub(r"\\ref\{([^}]+)\}", lambda m: LABEL.get(m.group(1), "?"), t)
    t = re.sub(r"\\cite(\[[^]]*\])?\{([^}]+)\}", lambda m: f"[{m.group(2)}]", t)
    t = re.sub(r"\\(\w+)\{\}", lambda m: MACRO.get(m.group(1), m.group(0)).replace("\\%", "%").replace("$\\times$", "×"), t)
    t = t.replace("\\_", "_").replace("\\$", "$")
    t = t.replace("\\%", "%").replace("R-PAL$^{+}$", "R-PAL+").replace("~", " ").replace("--", "–")
    t = re.sub(r"\\textsubscript\{([^}]*)\}", r"_\1", t)
    t = re.sub(r"\\textsc\{([^}]*)\}", lambda m: m.group(1).upper(), t)
    t = re.sub(r"\\emph\{([^}]*)\}", r"*\1*", t)
    t = re.sub(r"\\textbf\{([^}]*)\}", r"\1", t)
    t = re.sub(r"\\texttt\{([^}]*)\}", r"`\1`", t)
    t = re.sub(r"\\paragraph\{([^}]*)\}", r"**\1** ", t)
    t = re.sub(r"\\(caption|footnote|url)\{", "", t)
    t = t.replace("{=}", "=").replace("{>}", ">").replace("\\times", "×").replace("\\,", " ").replace("\\ ", " ")
    t = re.sub(r"\$([^$]*)\$", lambda m: m.group(1).replace("{", "").replace("}", "").replace("\\", ""), t)
    t = re.sub(r"\\[a-zA-Z]+\*?", "", t).replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", t).strip()


def paragraph_of(src: str, needle: str) -> str:
    """The source paragraph around needle: bounded by blank lines, headings, or float environments."""
    i = src.index(needle)
    starts = [src.rfind(m, 0, i) for m in ("\n\n", "\\paragraph{", "\\section", "\\subsection", "\\end{table}", "\\end{figure}", "\\caption{")]
    start = max(starts)
    ends = [x for x in (src.find(m, i + len(needle)) for m in ("\n\n", "\\paragraph{", "\\section", "\\subsection", "\\begin{table", "\\begin{figure", "\\label{tab")) if x > 0]
    return src[start:min(ends)].lstrip("\n")


def word_diff(a: str, b: str) -> str:
    aw, bw = a.split(" "), b.split(" ")
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=aw, b=bw, autojunk=False).get_opcodes():
        if op == "equal":
            out.append(" ".join(aw[i1:i2]))
        if op in ("delete", "replace") and " ".join(aw[i1:i2]).strip():
            out.append("~~" + " ".join(aw[i1:i2]) + "~~")
        if op in ("insert", "replace") and " ".join(bw[j1:j2]).strip():
            out.append("**" + " ".join(bw[j1:j2]) + "**")
    return " ".join(out)


# ── section 0: the provenance corrections already in the paper (before = archived pre-correction copy)
PRE = (PAPER / "archive/2026-10-17_main_9p_before_integrity.tex").read_text(encoding="utf-8")
SEC0 = [
    ("§3.1 建庫（三段都寫出來）", "The 250-card library used here is generated fully automatically", "The 250-card library used here is generated fully automatically"),
    ("§4.1 建庫與卡片重複題目數值（簡短版）", "The contract library is built from those reference functions", "The contract library is built from those reference functions"),
    ("§4.5 人工集的 18 題", "The F/E/I/N traps are contract-derived, so a skeptic", "The F/E/I/N traps are contract-derived, so a skeptic"),
    ("§7 限制", "This work is a first step.", "This work is a first step."),
]


def section0() -> List[str]:
    out = ["## 〇、已經改進論文的出處更正（改前＝更正前的存檔，改後＝目前版本）", ""]
    for title, a_pre, a_now in SEC0:
        rb, ra = readable(paragraph_of(PRE, a_pre)), readable(paragraph_of(BASE, a_now))
        out += [f"### {title}", "", f"**改前（整段）**\n\n> {rb}\n", f"**改後（整段）**\n\n> {ra}\n", f"**差異**\n\n> {word_diff(rb, ra)}\n", ""]
    out += ["---", ""]
    return out


def section_for(prop: Dict) -> str:
    lines = [f"### {prop['id']}　{prop['title']}　（實測省 {saved[prop['id']]:.0f} 行）\n"]
    if prop["body"] == "REMOVE_ALL_EXTREF":
        after_src = remove_all_extref(BASE)
        sents_b = extref_sentences(BASE)
        lines.append("每一處附錄引用所在的句子：\n")
        for k, s_ in enumerate(sents_b, 1):
            s_ = s_.strip()
            after = remove_all_extref(BASE[:BASE.index(r"\begin{document}")] + r"\begin{document}" + s_ + r"\bibliographystyle")
            after = after[after.index(r"\begin{document}") + len(r"\begin{document}"):after.index(r"\bibliographystyle")]
            lines.append(f"{k}. 改前：{readable(s_)}\n   改後：{readable(after)}\n")
    else:
        groups: Dict[str, List[Tuple[str, str]]] = {}
        for old, new in prop["body"]:
            if prop.get("macro") and old.startswith("\\newcommand"):
                lines.append("**巨集定義**：正文全部附錄引用由「Appendix X of the extended version」改印成「App. X」。逐句改前改後：\n")
                for k, s_ in enumerate(extref_sentences(BASE), 1):
                    rb = readable(s_)
                    lines.append(f"{k}. 改前：{rb}\n   改後：{re.sub(r'Appendix ([A-Z]) of the extended version', lambda m_: 'App. ' + m_.group(1), rb)}\n")
                lines.append("")
                continue
            anchor = old.strip("\n").split("\n")[0][:120]
            groups.setdefault(paragraph_of(BASE, anchor), []).append((old, new))
        for para_b, edits in groups.items():
            para_a = para_b
            for old, new in edits:
                o, n = old.strip("\n"), new.strip("\n")
                if o not in para_a and "\n\\label" in o:  # caption edits carry their label for uniqueness
                    o, n = o.split("\n\\label")[0], n.split("\n\\label")[0]
                assert o in para_a, (prop["id"], o[:60])
                para_a = para_a.replace(o, n)
            rb, ra = readable(para_b), readable(para_a)
            lines.append(f"**改前（整段）**\n\n> {rb}\n\n**改後（整段）**\n\n> {ra if ra else '（整段刪除）'}\n\n**差異**\n\n> {word_diff(rb, ra)}\n")
        for old, new in prop.get("app", []):
            added = new.replace(old, "", 1) if new.startswith(old) else new
            lines.append(f"**附錄新增**\n\n> {readable(added)}\n")
    lines.append(f"**資訊去向**：{prop['dest']}\n")
    return "\n".join(lines)


md = [
    "# ACM 版壓到九面：刪減提案（完整前後文）",
    "",
    f"由 `scripts/build_cut_proposals.py` 產生。目前 ACM 版 {BASE_PAGES} 頁。",
    "每一項都實際套用到一份副本、用 pdfLaTeX 編譯，數全文少了幾行文字，所以「實測省」是編譯出來的結果，不是估計。",
    "行數可以相加；能不能進九頁要看分頁，所以下面另外逐步累加、每一步都實際編譯看頁數。",
    "",
    "**改前、改後都是整段**，數字巨集換成實際數值、引用換成實際編號；差異欄用 ~~刪除線~~ 表示刪、**粗體** 表示新增。",
    "",
    *section0(),
    "## 總覽",
    "",
    "| 編號 | 位置 | 實測省（行） | 資訊去向 |",
    "|---|---|---|---|",
]
for prop in P:
    md.append(f"| {prop['id']} | {prop['title']} | {saved[prop['id']]:.0f} | {prop['dest'][:60]}{'…' if len(prop['dest']) > 60 else ''} |")
md += [
    "",
    f"**逐步累加（每一步都實際編譯）**：先套建議組合 {', '.join(RECOMMENDED)}，再依「對內容影響由小到大」加入其他項目。",
    "",
    "| 累加到 | 累計省（行） | ACM 頁數 |",
    "|---|---|---|",
]
for pid, n, pg in path_rows:
    md.append(f"| {'建議組合' if pid == RECOMMENDED[-1] else '+ ' + pid} | {n} | {pg} |")
md += ["", "---", ""]
for prop in P:
    md.append(section_for(prop))
    md.append("\n---\n")
OUT.write_text("\n".join(md), encoding="utf-8")
print(f"base {BASE_PAGES} pages, {BASE_LINES} text lines")
for pid, n, pg in path_rows:
    print(f"  cumulative through {pid}: {n} lines, {pg} pages")
for k, v in saved.items():
    print(f"  {k:4s} {v:5.0f}")
print("wrote", OUT.relative_to(ROOT))
shutil.rmtree(WORK, ignore_errors=True)
