import urllib.request
import json
import socket

payload = {
    "question": "這個project有辦法成功獲利嗎 這是一個capital budgeting題目",
    "context": "項目所需的初始投資為10萬美元（第一年年初）。預計該計畫在未來五年內將產生以下現金流：第一年2萬美元，第二年3萬美元，第三年3.5萬美元，第四年4萬美元，第五年4.5萬美元。該公司採用8%的折現率來評估其投資，是年底收到款項。",
    "top_k": 3,
    "m_min_top_score": 0.05
}

req = urllib.request.Request("http://127.0.0.1:6222/api/diagnose", 
                            data=json.dumps(payload).encode('utf-8'),
                            headers={'Content-Type': 'application/json'})

try:
    with urllib.request.urlopen(req, timeout=30) as response:
        result = json.loads(response.read().decode('utf-8'))
        print(json.dumps(result, indent=2, ensure_ascii=False))
except Exception as e:
    print(f"Error: {e}")
