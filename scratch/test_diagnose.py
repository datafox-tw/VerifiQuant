import json
import os

from verifiquant.pipeline.run_error_classification_pipeline import ErrorClassificationAPI, create_genai_client_from_env

def _load_env_file(path=".env"):
    if not os.path.exists(path):
        return
    for line in open(path, "r", encoding="utf-8").read().splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")

_load_env_file()
client = create_genai_client_from_env()

db_url = "sqlite:///" + os.path.abspath(os.path.join("verifiquant", "data", "runs", "demo_50q_0415", "cards.db"))
api = ErrorClassificationAPI.from_db(
    db_url=db_url,
    client=client,
    selector_model="gemini-2.5-flash",
    extractor_model="gemini-2.5-flash",
    judge_model="gemini-2.5-flash",
    top_k=3,
)

payload = {
    "question": "這個project有辦法成功獲利嗎 這是一個capital budgeting題目",
    "context": "項目所需的初始投資為10萬美元（第一年年初）。預計該計畫在未來五年內將產生以下現金流：第一年2萬美元，第二年3萬美元，第三年3.5萬美元，第四年4萬美元，第五年4.5萬美元。該公司採用8%的折現率來評估其投資，是年底收到款項。",
}

result = api.diagnose_row(payload, top_k=3, m_min_top_score=0.05, debug_sanity=True)
print(json.dumps(result, indent=2, ensure_ascii=False))
