# 可驗證財金程式 Harness（給老師的示範）

LLM 寫財金程式，harness 決定這個程式能不能信。每一步用到的概念都連到真正討論它的論文（頁面底部的「概念 × 論文對照」），前端把每一步的證據攤開。

```bash
python3 harness_demo/serve.py      # http://localhost:6230
```

- **示範模式**：6 個簡單例子，不需要網路。候選程式是人工撰寫、重現常見 LLM 錯誤的替身（頁面上有標示）；沙箱執行、不變量檢查、檢查員重算全部是真的。
- **即時 Gemini**：讀 `.env` 的 `GEMINI_API_KEY`，規格抽取、4 份候選程式、檢查員、無 harness 對照組都由 Gemini 2.5 Flash 現場產生。問題類型限於四張卡。

## 單篇論文套用（第三個模式）

拿掉 harness 的其他步驟，只用一篇論文的機制處理同一批示範例子，程式真的執行，再和完整 harness 比較（`lenses.py`）。

- **VERINA**：不產生答案，而是用它的 soundness / completeness 定義替卡片打分。正例是標註正確的程式，反例是標註的 bug 加上變異程式（答案乘 1.001、加 1 元、四捨五入），在三組輸入上執行。目前結果：NPV 卡放行了「四捨五入到整數元」，其餘反例都被抓到；有些反例在題目那一組輸入上會被放行，換輸入才被拒絕。
- **PAL / PoT、CodeT / AlphaCode、ClarifyGPT、AIRepr**：各自只用自己的機制產生答案或提問，結果分成正確、自行假設、安靜地錯、追問、追問含雜訊、拒答、照樣作答。
- 評分用 `cases.py` 裡人工標註的 bug；harness 本身從不讀這些標註。

## 十個步驟

| # | 步驟 | 確定性？ | 主要文獻 |
|---|---|---|---|
| 1 | 判斷問題類型，沒有卡就拒答 | 是（即時模式關鍵字沒中才問 LLM） | Selective prediction |
| 2 | 意圖規格：每個欄位要有原文引用，數值由程式重新解析 | LLM 指出引用，程式驗證 | Astrogator, FinanceReasoning |
| 3 | 取出卡片上的不變量（前置、後置、變形關係、約定釘子） | 是 | Design by Contract, metamorphic testing |
| 4 | 輸入合約檢查 | 是 | Design by Contract |
| 5 | 生成多份候選程式，在題目輸入上執行並分群 | LLM 生成，沙箱執行 | PAL/PoT, CodeT |
| 6 | 每份程式跑完所有不變量；宣稱的約定要和行為一致 | 是 | metamorphic testing, VERAFI（對比） |
| 7 | 分辨 bug 與歧義；卡片列出的約定原文沒決定就問 | 是 | ClarifyGPT, Astrogator |
| 8 | 檢查員只看凍結規格，用不同方法重算 | LLM 生成，沙箱執行 | AIRepr, Selection-Inference |
| 9 | 規格項目 ↔ 程式行追溯 | 是 | Verifiable Literate Programming |
| 10 | 輸出 / 追問 / 拒答，附稽核紀錄 | 是 | Learning to Seek Evidence, Turpin et al. |

## 檔案

- `cards.py`：四張「不變量卡」（年金終值、房貸月付、CAGR、NPV）。卡片不存公式，只存欄位、約定與不變量。
- `cases.py`：示範例子與替身候選程式（每份錯誤程式有人工標註的成因）。
- `harness.py`：十個步驟。
- `sandbox.py`、`_runner.py`：候選程式在子行程中執行（只允許 `import math`，CPU 與時間上限）。
- `providers.py`：示範替身與 Gemini 兩種來源，介面相同。
- `papers.py`：概念 × 論文對照。`CONCEPTS` 列出 harness 用到的 18 個概念，每個概念記錄程式位置、做法、以及真實討論它的論文「怎麼說」與「差在哪」；`PAPERS` 記錄 29 筆出處與連結。`checked` 標出出處資訊怎麼確認（10/29 網路查證、ICAIF bib、先前讀過原文、未查證）；`read=False` 是建議補讀。論文主張摘自摘要或週會筆記，引用數字前請回原文核對。標為「本系統的設計」的兩個概念（base case + 遞迴、先淘汰 bug 再判歧義）沒有直接出處。

## 已知限制

- 卡片庫只有四種問題類型；問題類型判斷是關鍵字比對，換個說法可能沒命中而拒答。
- 不變量由人撰寫，寫得不夠強就可能放過錯誤程式；步驟 7 在通過驗證的程式仍有分歧時會拒答，但抓不到「全部程式一起錯、且不變量也沒覆蓋」的情況。
- 沙箱是子行程加受限 builtins，適合示範，不是正式的安全隔離。
