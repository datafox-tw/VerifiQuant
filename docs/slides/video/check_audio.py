"""Transcribe narration WAVs with Gemini to check nothing extra was read aloud.

    python3 docs/slides/video/check_audio.py cover trap
"""
import json
import pathlib
import sys

from google import genai
from google.genai import types

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from build_audio import OUT, load_env  # noqa: E402

load_env()
client = genai.Client()
for wav in sorted(OUT.glob("*.wav")):
    sid = wav.stem.split("_", 1)[1]
    if len(sys.argv) > 1 and sid not in sys.argv[1:]:
        continue
    meta = json.loads(wav.with_suffix(".json").read_text())
    r = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[types.Part.from_bytes(data=wav.read_bytes(), mime_type="audio/wav"),
                  "Transcribe this audio verbatim. Output only the transcript."])
    print(f"== {wav.name} ({meta['duration']}s)\n{r.text.strip()}\n")
