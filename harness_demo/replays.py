"""PoC replay cases: complete harness runs on the paper's questions, recorded by poc/record_replays.py.

The site cannot run the PoC (its 250 auto cards and card library are not in the image), so these cases
return a recorded run instead of executing one. Each story says how often the shown outcome occurred across
the PoC runs, and the failures are listed as failures.
"""

from __future__ import annotations

import json
from pathlib import Path

DIR = Path(__file__).with_name("replays")

REPLAYS = [
    {"id": "poc_amortization", "title": "房貸第一個月的利息", "tag": "PoC 實測 · 衍生量 → 輸出",
     "story": "論文 250 題之一（test-2218），卡片由 LLM 讀函式庫自動產生。函式回傳整張攤還表，題目只要第一個月的利息："
              "harness 用有原文出處的路徑 result[0]['interest_paid'] 取值，再用元素探針確認卡片管得到這一格。"
              "這張卡原本有兩條寫錯的不變量，卡片自我檢查刪掉之後才答得出來。v6c 三次執行都答對 625.00。",
     "baseline": "CoT 單次：625.00（正確）。論文 VQ：正確。"},
    {"id": "poc_keltner", "title": "肯特納通道最後的上軌", "tag": "PoC 實測 · 元素探針 → 拒答",
     "story": "test-2236。卡片只有「上軌 ≥ 中軌」這類不等式。把題目要的那一格改動 0.1%，沒有任何不變量發現，所以拒答："
              "候選程式的答案從 143.53 到 143.69 不等，輸出就等於給一個沒被驗證過的數字。每次執行都拒答。",
     "baseline": "CoT 單次：沒有給出答案。論文 VQ：正確。"},
    {"id": "poc_inspector", "title": "獨立檢查員擋下錯的答案", "tag": "PoC 實測 · 檢查員 → 拒答",
     "story": "test-1588。通過不變量的候選算出 70,750；只看規格的檢查員用不同方法算出 145,000，也就是標準答案。"
              "兩條路徑不一致，harness 拒答，而不是輸出錯的數字。我們量過「讓檢查員也先通過卡片」，"
              "它會讓檢查員連卡片的錯一起學走，所以保留檢查員的獨立性。錄製兩次都是這個結果。",
     "baseline": "CoT 單次：145,000（正確）。論文 VQ：正確。"},
    {"id": "poc_lwma", "title": "加權移動平均（失敗案例）", "tag": "PoC 實測 · 輸出但錯",
     "story": "test-1801，誠實列出的失敗。函式要價格「由新到舊」，題目列的是第 1 天到第 5 天。"
              "每個數值都有原文出處、程式也通過不變量，但順序的意義沒有被驗證，於是輸出 103.83（正確是 108.50）。"
              "這是規格層的錯，和陷阱題的結論相同。",
     "baseline": "CoT 單次：108.50（正確）。論文 VQ：正確。這一題 harness 比直接問模型差。"},
    {"id": "poc_trap_missing", "title": "陷阱題：拿掉一個必要數值", "tag": "PoC 實測 · 陷阱 F → 追問",
     "story": "論文的陷阱題 F-test-1023，題目刪掉了原本的稅負金額。每個欄位都要有原文出處，找不到就停下來問。"
              "缺欄位這一類陷阱，harness 在開發集與測試集的 FNAR 都是 0%。五次執行都追問。",
     "baseline": "R-PAL+：追問。論文 VQ：攔下並追問。"},
    {"id": "poc_trap_scope", "title": "陷阱題：換成範圍外的問題（失敗案例）", "tag": "PoC 實測 · 陷阱 N → 仍輸出",
     "story": "N-test-1086。問題被換成「用有限差分 PDE 解美式選擇權」，但情境段落沒有換。選卡與意圖步驟讀到情境段落，"
              "以為還是原題，照原題算出 15%。這是分析中的原因三：管線繞過了模型原本會有的拒答判斷。五次執行都輸出。",
     "baseline": "R-PAL+：拒答。論文 VQ：拒答。這一類 harness 比兩者都差。"},
    {"id": "poc_trap_absurd", "title": "陷阱題：荒謬的輸入（失敗案例）", "tag": "PoC 實測 · 陷阱 E → 仍輸出",
     "story": "E-test-1006。加工時間被改成 1,500 萬小時。自動卡片的前置條件只檢查正負，沒有「合理範圍」，於是照算。"
              "VQ 的手寫卡片有這條規則而攔下。這是需要卡片裡的領域知識才能補的一類。五次執行都輸出。",
     "baseline": "R-PAL+：照算 15,000,006.5。論文 VQ：攔下（輸入不合理）。"},
]
REPLAYS_BY_ID = {c["id"]: c for c in REPLAYS}


def load(case_id: str) -> dict:
    case = REPLAYS_BY_ID[case_id]
    res = json.loads((DIR / f"{case_id}.json").read_text())
    res["baseline"] = {"kind": "measured", "text": case["baseline"]}
    res["mode"] = "replay"
    return res


def meta_entries() -> list[dict]:
    """Same fields as the scripted cases; the question is what the harness saw (context + question)."""
    out = []
    for c in REPLAYS:
        res = json.loads((DIR / f"{c['id']}.json").read_text())
        out.append({"id": c["id"], "title": c["title"], "tag": c["tag"], "question": res["question"], "story": c["story"]})
    return out
