"""Assemble slides + narration into an MP4 with burned-in subtitles and a sidecar .srt.

Inputs (from render_slides.py and build_audio.py):
  build/slides/NN_<id>.png   1920x1080
  build/audio/NN_<id>.wav    24 kHz mono, plus NN_<id>.json line timings
Outputs:
  verifiquant_icaif26_talk.mp4   subtitles burned in
  verifiquant_icaif26_talk.srt   same cues, for YouTube or a player

Subtitles are drawn with Pillow onto still frames, so any ffmpeg build works
(the imageio-ffmpeg binary has no libass). Needs: pip install imageio-ffmpeg pillow numpy

    python3 docs/slides/video/build_video.py
"""
import json
import pathlib
import re
import subprocess
import wave

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = pathlib.Path(__file__).resolve().parent
B = HERE / "build"
RATE = 24000
LEAD, TAIL, LAST_TAIL = 0.6, 0.9, 3.0  # seconds of silence around each slide's narration
MAX_CUE = 96  # characters; longer lines are split into two cues
HOLD = 0.35  # keep a cue up this long after its line ends, unless the next cue starts
NAME = "verifiquant_icaif26_talk"
SPEED = 1.1  # narration tempo (pitch kept by ffmpeg atempo); all timings are divided by this
FONT = ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", 44, index=0)
BOX, INK = (22, 50, 74, 225), (248, 246, 243, 255)


def split_cue(text, start, end):
    if len(text) <= MAX_CUE:
        return [(text, start, end)]
    mid = len(text) / 2
    cands = [m.end() for m in re.finditer(r"[,:;] ", text)] or [m.end() for m in re.finditer(r" ", text)]
    cut = min(cands, key=lambda c: abs(c - mid))
    a, b = text[:cut].strip(), text[cut:].strip()
    t = start + (end - start) * len(a) / (len(a) + len(b))
    return split_cue(a, start, t) + split_cue(b, t, end)


def srt_ts(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def wrap(text, width=1560):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if FONT.getlength(trial) <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    return lines + [cur]


def subtitled(png, text, out):
    img = Image.open(png).convert("RGBA")
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    lines = wrap(text)
    lh, pad_x, pad_y = 58, 32, 16
    w = max(FONT.getlength(l) for l in lines) + 2 * pad_x
    h = lh * len(lines) + 2 * pad_y
    x0, y0 = (1920 - w) / 2, 1080 - 36 - h
    d.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=14, fill=BOX)
    for i, l in enumerate(lines):
        d.text((960, y0 + pad_y + lh * i + lh / 2), l, font=FONT, fill=INK, anchor="mm")
    Image.alpha_composite(img, layer).convert("RGB").save(out)


def main():
    slides = sorted((B / "slides").glob("*.png"))
    audio_parts, spans, cues, t = [], [], [], 0.0
    for k, png in enumerate(slides):
        meta = json.loads((B / f"audio/{png.stem}.json").read_text())
        with wave.open(str(B / f"audio/{png.stem}.wav")) as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        tail = LAST_TAIL if k == len(slides) - 1 else TAIL
        n_lead, n_tail = int(LEAD * RATE), int(tail * RATE)
        audio_parts += [np.zeros(n_lead, np.int16), pcm, np.zeros(n_tail, np.int16)]
        dur = (n_lead + len(pcm) + n_tail) / RATE
        spans.append((png, t, t + dur))
        for line in meta["lines"]:
            cues += [(txt, s, e, k) for txt, s, e in
                     split_cue(line["text"], t + LEAD + line["start"], t + LEAD + line["end"])]
        t += dur

    # hold each cue briefly, never past the next cue or its own slide
    held = []
    for i, (txt, s, e, k) in enumerate(cues):
        nxt = cues[i + 1][1] if i + 1 < len(cues) else float("inf")
        held.append((txt, s, min(e + HOLD, nxt - 0.04, spans[k][2]), k))
    cues = [(txt, s / SPEED, e / SPEED, k) for txt, s, e, k in held]
    spans = [(png, a / SPEED, b / SPEED) for png, a, b in spans]

    track = np.concatenate(audio_parts)
    with wave.open(str(B / "narration.wav"), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(RATE), w.writeframes(track.tobytes())

    srt = [f"{i}\n{srt_ts(s)} --> {srt_ts(e)}\n{txt}\n" for i, (txt, s, e, _) in enumerate(cues, 1)]
    (HERE / f"{NAME}.srt").write_text("\n".join(srt))

    # piecewise-constant frames: plain slide between cues, slide + subtitle during a cue
    frames = B / "frames"
    frames.mkdir(exist_ok=True)
    concat, n = [], 0
    for k, (png, a, b) in enumerate(spans):
        cur = a
        for txt, s, e, ck in cues:
            if ck != k:
                continue
            if s > cur + 0.01:
                concat += [f"file '{png.as_posix()}'", f"duration {s - cur:.3f}"]
            n += 1
            f = frames / f"{n:04d}.png"
            subtitled(png, txt, f)
            concat += [f"file '{f.as_posix()}'", f"duration {e - s:.3f}"]
            cur = e
        if b > cur + 0.01:
            concat += [f"file '{png.as_posix()}'", f"duration {b - cur:.3f}"]
    concat.append(f"file '{slides[-1].as_posix()}'")
    (B / "concat.txt").write_text("\n".join(concat) + "\n")

    out = HERE / f"{NAME}.mp4"
    tmp = B / f"{NAME}.partial.mp4"  # renamed only after ffmpeg finishes, so an interrupt never leaves a broken file
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(B / "concat.txt"),
        "-i", str(B / "narration.wav"),
        "-vf", "fps=30,format=yuv420p", "-af", f"atempo={SPEED}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-tune", "stillimage",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-shortest", "-movflags", "+faststart", str(tmp),
    ], check=True)
    tmp.replace(out)
    print(f"{out.name}: {len(track) / RATE / SPEED / 60:.2f} min, {len(cues)} cues, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
