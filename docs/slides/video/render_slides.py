"""Render the ICAIF deck slides to 1920x1080 PNGs with headless Chrome.

The deck source (docs/slides/icaif_deck/) uses the Slides artifact format:
one <section> per file, inline styles, plus <x-icon>, <x-shape> and
<x-connector> tags that only the artifact runtime draws. This script swaps
those tags for equivalent inline SVG so a plain browser renders the same look.

    python3 docs/slides/video/render_slides.py            # all slides
    python3 docs/slides/video/render_slides.py trap edits # some slides
"""
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
DECK = ROOT / "docs/slides/icaif_deck"
OUT = ROOT / "docs/slides/video/build/slides"
TMP = ROOT / "docs/slides/video/build/html"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

BLOBS = {
    "/_blob/903cb7977632e7b4967e2159c913e593": "poster/assets/qr-demo.svg",
    "/_blob/d8c82b2f627d801abaaab6516ea7fe17": "poster/assets/qr-github.svg",
    "/_blob/abf810f375de808dfd373d201dbdf954": "poster/assets/qr-doi.svg",
    "/_blob/adc12fd362c39ff61d21918a81ff1edc": "poster/assets/icaif-2026-mark-white.svg",
}

ICONS = {  # 24x24 stroke icons
    "Code": '<polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/>',
    "Warning": '<path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>'
               '<line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "CheckCircle": '<circle cx="12" cy="12" r="10"/><polyline points="8 12 11 15 16 9"/>',
    "Chat": '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    "Database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>'
                '<path d="M3 12c0 1.66 4 3 9 3s9-1.34 9-3"/>',
    "Search": '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>',
    "Trust": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><polyline points="9 12 11 14 15 10"/>',
}

CSS = """
*{margin:0;padding:0;box-sizing:border-box}
html,body{width:1920px;height:1080px;overflow:hidden;background:#F5F3EE}
section{width:1920px;height:1080px;position:relative;overflow:hidden}
table{border-collapse:collapse;width:100%}
td,th{padding:0.35em 0.6em;border:1px solid #C9D1D6;text-align:left}
th{font-weight:600}
"""


def style_get(style, prop, default=None):
    m = re.search(rf"(?:^|;)\s*{prop}\s*:\s*([^;]+)", style)
    return m.group(1).strip() if m else default


def px(v):
    return int(float(v.replace("px", "")))


def icon(m):
    name, style = m.group(1), m.group(2)
    c = style_get(style, "color", "#16324A")
    w, h = px(style_get(style, "width", "48px")), px(style_get(style, "height", "48px"))
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round" style="flex:none">{ICONS[name]}</svg>')


def shape(m):
    kind, style = m.group(1), m.group(2)
    assert kind == "arrow-right", kind
    c = style_get(style, "background", "#9FC3D3")
    w, h = px(style_get(style, "width")), px(style_get(style, "height"))
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 100 100" preserveAspectRatio="none" style="flex:none">'
            f'<polygon points="0,30 60,30 60,0 100,50 60,100 60,70 0,70" fill="{c}"/></svg>')


def connector(m):
    style = m.group(1)
    c = style_get(style, "color", "#4F6A7D")
    w = px(style_get(style, "width", "32px"))
    return (f'<svg width="{w}" height="20" viewBox="0 0 {w} 20" style="flex:none;align-self:center">'
            f'<line x1="2" y1="10" x2="{w - 8}" y2="10" stroke="{c}" stroke-width="2.5"/>'
            f'<polygon points="{w - 10},4 {w},10 {w - 10},16" fill="{c}"/></svg>')


def standalone(slide_html, faces):
    s = re.sub(r"<aside>.*?</aside>", "", slide_html, flags=re.S)
    # the footer row (title + page number) sits where the video subtitles go
    s = re.sub(r'<p style="position:absolute;left:\d+px;bottom:64px;[^"]*">[^<]*</p>\n?', "", s)
    for k, v in BLOBS.items():
        s = s.replace(k, (ROOT / v).as_uri())
    s = re.sub(r'<x-icon name="(\w+)" style="([^"]*)"></x-icon>', icon, s)
    s = re.sub(r'<x-shape kind="([\w-]+)" style="([^"]*)"></x-shape>', shape, s)
    s = re.sub(r'<x-connector style="([^"]*)"></x-connector>', connector, s)
    assert "<x-" not in s, "unhandled custom tag"
    links = "".join(f'<link rel="stylesheet" href="{f["href"]}">' for f in faces.values())
    return f'<!doctype html><html><head><meta charset="utf-8">{links}<style>{CSS}</style></head><body>{s}</body></html>'


def main(only):
    deck = json.loads((DECK / "deck.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    for i, sid in enumerate(deck["order"], 1):
        if only and sid not in only:
            continue
        page = TMP / f"{i:02d}_{sid}.html"
        page.write_text(standalone((DECK / f"slides/{sid}.html").read_text(), deck["faces"]))
        png = OUT / f"{i:02d}_{sid}.png"
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        "--force-device-scale-factor=1", "--window-size=1920,1080",
                        "--virtual-time-budget=10000", f"--screenshot={png}", page.as_uri()],
                       check=True, capture_output=True)
        print("rendered", png.name)


if __name__ == "__main__":
    main(set(sys.argv[1:]))
