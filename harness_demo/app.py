"""Flask front for the verifiable-code harness demo.

Run from the repo root:  python3 -m harness_demo.app   (http://localhost:6230)
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque

from flask import Flask, jsonify, render_template, request

from . import harness
from .cards import CARDS
from .cases import CASES, CASES_BY_ID
from .papers import CONCEPTS, PAPERS
from .lenses import compute as compute_lenses
from .providers import LiveProvider, ScriptedProvider, live_available

app = Flask(__name__)
app.json.sort_keys = False  # step and paper order is meaningful

# live mode spends the owner's Gemini quota, so a public deploy caps it (in memory, per instance)
MAX_QUESTION_CHARS = 300
LIVE_PER_IP_HOUR = int(os.environ.get("HARNESS_LIVE_PER_IP_HOUR", 20))
LIVE_PER_DAY = int(os.environ.get("HARNESS_LIVE_PER_DAY", 200))
_hits: dict[str, deque] = {}
_day = {"date": "", "count": 0}
_hits_lock = threading.Lock()


def _live_quota_error() -> str | None:
    ip = (request.headers.get("X-Forwarded-For") or request.remote_addr or "").split(",")[0].strip()
    now = time.time()
    today = time.strftime("%Y-%m-%d")
    with _hits_lock:
        if _day["date"] != today:
            _day.update(date=today, count=0)
        if _day["count"] >= LIVE_PER_DAY:
            return "今天的即時模式額度已用完，請改用示範模式，或明天再試。"
        q = _hits.setdefault(ip, deque())
        while q and now - q[0] > 3600:
            q.popleft()
        if len(q) >= LIVE_PER_IP_HOUR:
            return f"即時模式每小時最多 {LIVE_PER_IP_HOUR} 次，請稍後再試，或先用示範模式。"
        q.append(now)
        _day["count"] += 1
    return None


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/meta")
def meta():
    return jsonify({
        "cases": [{k: c[k] for k in ("id", "title", "tag", "question", "story")} for c in CASES],
        "papers": PAPERS,
        "concepts": CONCEPTS,
        "steps": harness.STEP_TITLES,
        "cards": {cid: {"title": c["title"], "topic": c["topic"]} for cid, c in CARDS.items()},
        "live": live_available(),
    })


@app.get("/api/lenses")
def lenses():
    return jsonify(compute_lenses())


@app.post("/api/run")
def run():
    body = request.get_json(force=True)
    answers = body.get("answers") or {}
    if body.get("mode") == "live":
        if not live_available():
            return jsonify({"error": "即時模式需要 GEMINI_API_KEY 與 google-genai"}), 400
        question = (body.get("question") or "").strip()
        if not question:
            return jsonify({"error": "請輸入問題"}), 400
        if len(question) > MAX_QUESTION_CHARS:
            return jsonify({"error": f"問題請在 {MAX_QUESTION_CHARS} 字以內"}), 400
        if err := _live_quota_error():
            return jsonify({"error": err}), 429
        try:
            return jsonify(harness.run(LiveProvider(), question, answers))
        except Exception as exc:
            return jsonify({"error": f"Gemini 呼叫失敗：{type(exc).__name__}: {exc}"}), 502
    case = CASES_BY_ID.get(body.get("case_id"))
    if not case:
        return jsonify({"error": "unknown case"}), 404
    return jsonify(harness.run(ScriptedProvider(case), case["question"], answers))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 6230)), debug=False)
