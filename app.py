import os
from collections import defaultdict
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from google import genai

from card_store import DefinitionStore
from verifiquant import solve_question


def build_domain_topic_options(store: DefinitionStore):
    mapping = defaultdict(set)
    for card in store.cards:
        if card.domain:
            mapping[card.domain].add(card.topic)
    domain_options = sorted(mapping.keys())
    topic_options = {domain: sorted(list(topics)) for domain, topics in mapping.items()}
    return domain_options, topic_options


def create_app():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set.")

    store_path = Path(os.environ.get("DEFINITION_STORE_PATH", "artifacts/definition_store.pkl"))
    if not store_path.exists():
        raise FileNotFoundError(f"Definition store not found at {store_path}")

    store = DefinitionStore.load(store_path)
    domain_options, topic_options = build_domain_topic_options(store)
    client = genai.Client(api_key=api_key)

    app = Flask(__name__, static_folder="static", template_folder="templates")

    app.config["STORE"] = store
    app.config["GENAI_CLIENT"] = client
    app.config["DOMAIN_OPTIONS"] = domain_options
    app.config["TOPIC_OPTIONS"] = topic_options

    @app.route("/")
    def index():
        return render_template(
            "index.html",
            domain_options=app.config["DOMAIN_OPTIONS"],
            topic_options=app.config["TOPIC_OPTIONS"],
        )

    @app.route("/api/solve", methods=["POST"])
    def api_solve():
        data = request.get_json() or {}
        question = (data.get("question") or "").strip()
        domain = data.get("domain") or None
        topic = data.get("topic") or None
        if not question:
            return jsonify({"status": "error", "message": "問題內容不可為空"}), 400

        try:
            result = solve_question(
                question=question,
                store=app.config["STORE"],
                client=app.config["GENAI_CLIENT"],
                selector_model=os.environ.get("SELECTOR_MODEL", "gemini-2.5-flash"),
                extractor_model=os.environ.get("EXTRACTOR_MODEL", "gemini-2.5-flash"),
                domain=domain,
                topic=topic,
            )
            return jsonify(result)
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500

    return app


if __name__ == "__main__":
    application = create_app()
    application.run(host="0.0.0.0", port=int(os.environ.get("PORT", 6222)), debug=True)

