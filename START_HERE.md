# 專案導覽（依驗證方式分區）

最後更新：2026-09-27

**Agenthon 這條線 9/26 暫停、同日恢復。** 暫停時的判斷在
`docs/2026-09-23_verification_layers_reference.md` 第五部分（§23–26）；恢復之後的進度
（突變測試、L5 決策宣告 v0、官方逐題結果）在 `agenthon/docs/2026-10-02_meeting.md`。
每一次上傳的內容與結果在 `agenthon/submission/zips/README.md`。

這份文件是整個工作目錄的入口。它跟 `README.md` 的差別在切法：`README.md` 講
VerifiQuant 這一個系統怎麼運作，寫於 2026-06-14，不包含後來的論文實驗與 Agenthon。
這份文件橫跨兩個 repo，用**驗證方式**當座標軸重新排列所有檔案。

要看單一系統的技術細節，讀 `README.md` 與 `.claude/CLAUDE.md`。
要知道某個檔案為什麼存在、屬於哪一種檢查、憑什麼證據講話，讀這份。

---

## 0. 一句話

整個專案在處理同一件事：**語言模型在證據不足時會給出高自信的答案**，
而我們要讓這種失敗變成可以攔截、可以標記、可以交代的行為。

不是讓模型更準。是讓它錯得可預測。

所有程式碼都是這個主張的實作或量測。分區的邏輯因此不是「放在哪個資料夾」，
而是「**這個檢查憑什麼證據說話**」。

---

## 1. 為什麼用驗證方式分區

同一個資料夾裡的檔案常常做完全不同的事。`agenthon/agent/` 底下，`contract.py`
只讀題目文字，`cross_derive.py` 只讀產出的數字，`explore.py` 讀的是真實輸入檔。
三者的失效模式、誤告成本、能覆蓋的題目完全不同，但目錄把它們混在一起。

反過來，`verifiquant/pipeline/run_error_classification_pipeline.py` 的 E 層
和 `agenthon/agent/dimensions.py` 在兩個 repo、兩個語境，做的卻是同一種檢查：
**不呼叫模型，用數值範圍的硬規則判斷一個值合不合理**。

所以底下用證據來源分成 A 到 H 區。每一區列出：它問什麼問題、證據哪裡來、
什麼時候觸發、目前做到哪裡。

---

## 2. 分區總表

| 區 | 名稱 | 證據來源 | 有無模型呼叫 | 狀態 |
|---|---|---|---|---|
| **A** | 宣告式驗證（FIC 卡片） | 事先建好的卡片庫 | 建卡時有，用卡時無 | 上線 |
| **B** | 自證式驗證（產出物自洽） | 答案本身 | L3 有，其餘無 | L1–L4a 上線，L4b 已整合但預設關閉 |
| **C** | 指令遵循驗證 | 題目文字 | 有 | Critic 上線；L5 決策宣告 v0 已建，預設關閉 |
| **D** | 前置與環境驗證 | 真實輸入檔與執行環境 | 部分有 | 上線 |
| **E** | 驗證器的驗證（負控制） | 已知正確 / 已知錯誤的案例 | 無 | 上線，是本專案的方法論核心 |
| **F** | 實驗與證據 | 跑出來的數字 | 有 | 論文數字的唯一來源 |
| **G** | 論文與對齊文件 | 人寫的 | 無 | 進行中 |
| **H** | 外部基準與歷史 | 別人的 repo / 舊版本 | 無 | 唯讀 |

---

## A 區 — 宣告式驗證：FIC 卡片

**問的問題**：這道題對應到哪一條金融公式，那條公式要求什麼？

**證據來源**：一個**事先建好的卡片庫**（SQLite `cards.db`）。每張 FIC 卡片記錄
一條公式的輸入欄位、可執行的 Python、不變量、數值範圍檢查、語意歧義提示。
關鍵性質是：檢查所需的知識在題目進來**之前**就存在，所以檢查本身不需要模型判斷。

這是 VerifiQuant 的本體，也是六層 M/N/F/E/I/C 漏斗。

### 六層漏斗

| 層 | 代碼 | 攔什麼 | `status` | `gate_action` | 憑什麼 |
|---|---|---|---|---|---|
| 1 Intent | **M** | 語意模糊，對不到任何公式家族 | `refusal` | `refusal` | 選擇器模型判斷 |
| 2 Scope | **N** | 意圖清楚但卡片庫沒有這題 | `refusal` | `graceful_exit` | 檢索結果為空，**確定性** |
| 3 Schema | **F** | 選到卡片了，但必填輸入缺席 | `error` | `slot_filling` | 卡片的 `inputs` 宣告，**確定性** |
| 4 Boundary | **E** | 輸入有了但數值不合理 | `alert` | `deterministic_alert` | 卡片的 `scale_checks`，**確定性** |
| 5 Critic | **I** | 隱性語意歧義 | `needs_clarification` | `critic_intervention` | 卡片的 `semantic_hints` |
| 6 Logic | **C** | 確定性 Python 的執行錯誤 | `error` | 稽核日誌 | 執行結果 |

**注意每一層的 `status` 都不一樣。** 這不是命名潔癖。
N 是 `refusal`（我知道我不知道），F 是 `error`（你少給了東西），
E 是 `alert`（你給的東西有問題，但我沒死），I 是 `needs_clarification`（我需要你決定）。
呼叫端可以只看 `status` 就決定要顯示什麼 UI，不必理解漏斗。

### 實際執行順序

主引擎的步驟名稱就是漏斗（`_set_step` 的參數）：

```
rag_cards  →  mn_analysis  →  extract_values  →  fe_checks  →  ihard_check  →  python_execution  →  final_response
   ↓              ↓                                  ↓             ↓                ↓
   N 出口        M 出口                            F / E 出口     I 出口          C 出口
```

每一個提前出口都會把後面所有步驟標成 `skipped`，並寫明理由
（`"Blocked by I_hard ambiguity."`、`"Execution blocked by deterministic E alert."`）。
所以一份 `DiagnosticReport` 讀起來是一條可稽核的軌跡，不是一個分類標籤。

### 逐層的具體內容

**N 層（Scope）是最先跑的，而且它是確定性的。**
條件很簡單：檢索回來的候選卡片是空的，`support_gap_reason` 記成 `no_candidate_cards`。
把它放在 M 層之前有個好處，模型根本沒有機會對一個庫裡沒有的題目編答案。
這是整個系統裡唯一一層「承認知識邊界」的設計。

**M 層（Intent）交給選擇器模型，它同時決定 M / N / 選哪張卡。**
拒答的時候會附上 `clarification_request`，預設問題是
「請指定目標指標，例如 NPV、IRR 或回收期」。這一層依賴模型判斷，所以它是六層裡最不穩的。

**F 層（Schema）是純粹的樹走訪。** 卡片的 `inputs` 宣告了必填欄位，
抽取結果裡缺的就進 `requested_fields`，並產生 `repair_action: request_missing_fields`
以及一組 `ask_user_for` 的槽位定義，前端可以直接渲染成表單。沒有任何模型判斷的餘地。

**E 層（Boundary）用卡片的 `scale_checks`，輸出 `triggered_rule_ids`。**
抓的是數值上不合理但語法上合法的輸入：無風險利率高於市場報酬、負的標準差、
百分比與小數混用。注意它回的是 `alert` 而不是 `error`，因為輸入不合理不代表無法計算，
只代表計算出來的東西不該被信任。
`lint_echecks.py` 是這一層的建置期閘門，把模型寫成賦值形式的檢查自動改回表達式，
而 `ECHECK_SAFE_BUILTINS` 是執行期求值環境的唯一真實來源。

**I 層（Critic）分 hard 與 soft。**
`I_HARD` 會擋下執行並回 `needs_clarification`，附上 `clarification_request.groups`，
每一組對應一個 `semantic_hints` 的 hint id，帶著問題與選項。
`mode` 是 `hint_oriented`（卡片有提示）或 `open`（沒有）。
這一層抓的是匯率方向、時間基準、年化與否這類**語法完全正確但語意有兩種讀法**的東西。

### 這一區最重要的機制：可驗證的原子轉換

I 層不只是問問題。使用者選了一個替代讀法之後，系統會套用一個**數學上被驗證過**的轉換，
而不是叫模型重算一次。這是 VerifiQuant 跟一般「讓 LLM 自我修正」最根本的差別。

```
stage_core    : semantic_hints[].options = [{label, value, is_default, transform_spec?}]
   ↓ _normalize_semantic_hint()   保留 dict 選項（必須用 _normalize_hint_options）
   ↓ _normalize_hint_option()     安全網，過濾模型的錯誤
stage_repair  : _global_i_rules() → _parse_hint_options() → repair_action.transform_map
   ↓ get_transform_spec_for_choice(repair_rule, chosen_value)
stage_transform: verify_transform() | verify_code_patch()
```

兩種轉換，各有各的驗證層：

| | `result_postprocess` | `code_patch` |
|---|---|---|
| 改什麼 | 輸出值 | 程式碼本身 |
| 驗證 1 | AST 節點數 ≤ 宣告上限 | 樣式出現次數必須 **== 1** |
| 驗證 2 | 名稱白名單 | 淺層 AST 差異 ≤ `max_changed_nodes` |
| 驗證 3 | 數值的確定性與有限性 | 不得引入危險呼叫或 import |
| 驗證 4 | 數值不變量檢查 | **數值交叉驗證**再加不變量檢查 |

**`code_patch` 為什麼安全**：接受條件是改完的程式碼**跑出來的數值**，
必須等於 `eval(cross_verify_result_expr, {result: 原始輸出, **inputs})`。
這把一個自由形式的程式碼編輯，綁死在一條宣告出來的代數恆等式上。
數學錯的、AST 差異過大的、偷夾 import 或 dunder 的、樣式出現多次的，全部被拒。

`_shallow_node_key()` 只雜湊節點自己的基本欄位，不含子樹。
所以 `start=1` 改成 `start=0` 算 **1** 個節點變動，而不是整條父節點鏈。

**`_normalize_hint_option()` 是硬閘門**，因為 Gemini 不會完美遵守結構化輸出的提示。
它強制執行七條規則：

1. 預設選項的 `transform_spec` 一律強制設為 null（FIC 的程式碼本身就是預設）
2. 恆等變換（`result_expr` 是 `""` 或 `"result"`）→ 丟棄
3. 下標語法（`inputs['name']`）→ 丟棄。必須用裸的輸入名
4. 函式呼叫（`round()`、`exp()`、`compute()`）→ 丟棄。`result_expr` 只能是純算術
5. `result_expr` 裡沒有引用到 `result` → 丟棄。它必須是 `result` 的變換，不是重算
6. 無法解析 → 丟棄
7. `max_expr_nodes` 由實際表達式重算（Gemini 數不準）

**提示是理想性的，正規化才是硬閘門。** 改提示的時候一定要回來檢查這個過濾器還蓋不蓋得住。
這條教訓在 Agenthon 的 L3 又重複了一次，見 C 區末尾的對照表。

### 檔案

**核心型別與儲存**
- `verifiquant/contracts.py` — `DiagnosticReport` / `DiagnosticFinding`，所有層的共同輸出格式
- `verifiquant/card_store.py` — SQLAlchemy ORM，三張表 `fic_core` / `fic_retrieval` / `fic_repair_rules` 加全文檢索
- `verifiquant/taxonomy.py` — 金融領域分類，卡片打標籤用

**建卡（build-time，會呼叫模型）** — `verifiquant/preprocessing/`
- `stage_core.py` — 模型產生公式、不變量、範圍檢查、語意提示。`_normalize_hint_option()` 在這裡，它是硬閘門
- `stage_retrieval.py` — 模型產生檢索用的關鍵字與負向關鍵字
- `stage_repair.py` — 把診斷檢查與語意提示轉成修復規則
- `stage_transform.py` — 可驗證的原子轉換（`result_postprocess` / `code_patch`）
- `fic_generation_pipeline.py` — 串起上面三個階段，處理去重與合併
- `seed_builder.py` — 從 YAML / JSONL 設定產生種子列
- `lint_echecks.py` — E 層檢查的自動修正閘門
- `validate_relations.py` — 跨產出物的完整性檢查（core ↔ retrieval ↔ repair）
- `common.py` — 讀寫、模型 client、正規化工具
- `dataset_case_to_fic.py` — 單題薄包裝

**用卡（run-time）** — `verifiquant/pipeline/`
- `run_error_classification_pipeline.py` — **主引擎**，約 2100 行，`ErrorClassificationAPI` 在這裡
- `run_error_classification_pipeline_lg.py` — LangGraph 重構版，`ErrorClassificationLG` 介面對齊上面那支
- `run_framework_guided_self_improve_pipeline.py` — 多輪 VerifiQuant 加復原
- `run_cot_self_improve_pipeline.py` — CoT 基線
- `run_verifiquant_lite_cot_self_improve_pipeline.py` — 減配版，用來拆解「是哪個元件帶來效果」
- `run_iterative_agents_pipeline.py` — 探索性質，非主線
- `strict_binding.py` — 字面接地稽核。動機是強模型（Pro）會在綁定前偷偷重算一個題目字面給的輸入
- `expand_cases.py` / `build_trap_set.py` / `build_trap_set_v2.py` — 陷阱題變體產生器
- `score_trap_set.py` — 對陷阱題集評分
- `run_refusal_ablation.py` — 拒答消融
- `nearood_common.py` — 近 OOD 檢索實驗的共用機制

**資料**
- `verifiquant/data/` — 題庫（`medium_config_50_*.jsonl`、`testing_*Q.jsonl`）、卡片庫、設定 YAML
- `verifiquant/data/runs/<run>/` — 每次實驗的 `cards.db` 與輸出。命名即實驗名，對照 F 區

### 這一區的限制

卡片庫要先建，所以**只能處理卡片涵蓋得到的題目**。N 層就是承認這件事的層。
這是 A 區與 B 區的根本差別：A 區有先驗知識但有邊界，B 區沒有邊界但也沒有先驗。

---

## B 區 — 自證式驗證：產出物自洽

**問的問題**：這份答案自己合不合理？

**證據來源**：**答案本身**。沒有卡片庫，沒有標準答案，沒有另一個模型當裁判。
只看模型寫出來的東西內部有沒有矛盾。

這是 Agenthon 那一側，也是 VerifiQuant 精神在「沒有卡片可用」情況下的延伸。
Track 1 有 87 個題目單元，題目彼此不重複，所以建卡片庫這條路不成立。

### 五層一覽

| 層 | 檔案 | 讀什麼 | 什麼時候閉嘴 | 呼叫模型 |
|---|---|---|---|---|
| **L1 結構** | `agent/contract.py` | 題目文字 + 產出檔案 | 題目沒有可辨識的輸出段落 | 否 |
| **L2 量綱** | `agent/dimensions.py` | 產出的每個數值 + 欄位名 | 欄位名沒有公認的量綱，題目也沒宣告單位 | 否 |
| **L3 不變量** | `agent/invariants.py` | 產出 + 題目宣告的恆等式 | 模型寫不出通過閘門的不變量 | **是** |
| L3 輔助 | `agent/invariant_retrieval.py` | 練習題語料的**正規化**不變量模式 | 沒有同型模式 | 否 |
| **L4a 交叉推導** | `agent/cross_derive.py` | 產出檔案內部欄位彼此 | 找不到任何一組欄位能解釋衍生欄 | 否 |
| **L4b 蛻變測試** | `agent/metamorphic.py` | **重跑腳本**的兩次結果 | 四道閘門任何一道不過 | 否 |

「什麼時候閉嘴」那一欄比「檢查什麼」更重要。每一層的設計裡，**沉默是預設值**，
開口需要先把關係建立起來。這是精確率當硬約束的直接後果。

---

### L1 結構契約 — `agent/contract.py`

確定性，不呼叫模型，不連網。在 `solve` 裡跑**兩次**：寫程式之前跑一次當規格，
求解器寫完產出之後再跑一次驗收。這樣格式壞掉的交付物是被我們抓到，不是被官方閘門 g1/g3 抓到。

它能還原的東西，照可靠度由高到低：

1. 哪些檔案必須被寫出來，以及格式
2. JSON 的鍵名，包含草圖有畫出來的巢狀路徑
3. CSV / Parquet 的欄位名
4. 散文裡寫的列數關係（`one row per input option`、`exactly 24 rows`）

**它是啟發式的，而且這件事無法迴避。** `instruction.md` 是散文不是 schema。
所以它不宣稱自己正確，而是拿官方評分器實際要求的東西去對（`eval/measure.py`）。

具體的語料事實決定了實作長相：

- 輸出掛載點有兩種拼法，64 個單元寫 `/app/output`，23 個寫 `/output`。兩種都要接受
- `reward.json`、`reward.txt`、`pytest_report.json` 永遠是評分器的 `test.sh` 寫的，不是我們寫的。
  硬編在 `_NEVER_OURS` 排除
- **在輸入根目錄底下被提到的檔名是在被描述，不是在被要求。**
  「讀 `portfolio.csv`，把加工過的 `portfolio.csv` 寫到 `/app/output/`」是真實的題型，
  所以只從那些**沒有帶 `/output/` 路徑**的名字裡扣掉。這條規則量出來的效果是：
  精確率 0.899 → 0.914，召回率不變，修好三個單元（`params.json`、`order.json`、
  `forward_returns.csv` 各自是一個輸入檔被區段掃描誤升成交付物）

最後那一條是有代價才學到的。**L1 的幽靈交付物曾經害掉一個單元**：
agent 花回合去生成一個題目根本沒要求的檔案。

---

### L2 量綱契約 — `agent/dimensions.py`

這一層存在的理由是量出來的，不是假設的。對一份**正確解**做三個單行突變，
拿官方評分器自己的檢查去打分：

| 突變 | 得分 | 結果 |
|---|---|---|
| gamma 正負號翻轉 | 0.0 | 被不變量抓到 |
| vega 除以 100 | **1.0** | **沒抓到** |
| theta 正負號翻轉再乘 365 | **1.0** | **沒抓到** |

單位與正負號的錯誤，會保留所有便宜檢查在測的性質：數字仍然有限、仍然為正、仍然單調。
而且這正是**模型最無法懷疑自己的錯誤**，因為它自己的產出裡沒有任何地方看起來不對。
這句話就是整個碩論主張在單一層次上的縮影。

兩種規則來源，有優先序：

1. **宣告型** — 題目自己講了（"report as a positive magnitude"、"raw floats, not percentages"、
   "in basis points"）。信心最高，因為是題目在說話
2. **語意型** — 欄位名帶著每個 quant 都知道的量綱

語意辭典（`_SEMANTIC`）的實際內容，比對的是小寫化的葉節點名稱，先匹配先贏：

| 欄位名樣式 | 界限 | 理由 |
|---|---|---|
| `r_squared` / `r2` / `adj_r_squared` | [0, 1] | 判定係數 |
| `win_rate` / `hit_rate` / `accuracy` / `coverage` | [0, 1] | 成功率 |
| `prob` / `probability` / `pd` / `default_prob*` | [0, 1] | 機率 |
| `corr` / `correlation` / `rho` / `kendall_tau` / `spearman` / `ic` | [-1, 1] | 相關係數 |
| `discount_factor` | (0, 1] | 折現因子 |
| `delta` | [-1, 1] | 選擇權 delta |
| `gamma` / `vega` | ≥ 0 | 香草選擇權的 gamma 與 vega 非負 |
| `vol` / `volatility` / `sigma` / `std` / `stdev` | ≥ 0 | 波動度與標準差非負 |
| `variance` / `var_` | ≥ 0 | 變異數非負 |

**這一層刻意不宣稱抓得到的東西**：一個沒有宣告單位、也沒有自然界限的量被純粹縮放。
上面那個 vega 除以 100 就正是這種。它需要的是第二次推導，不是範圍檢查，那是 L4 的工作。
模組 docstring 原話是：在這裡假裝抓得到，只會讓這一層看起來比實際好。

---

### L3 不變量契約 — `agent/invariants.py`

這一層不可能是固定辭典。買賣權平價、無套利界限、二元樹與解析解的一致性，
每一條都跟題目綁死。所以不變量必須**逐題從指令產生**，也就是**由模型寫**。
而模型會寫錯。

**一個錯的不變量比沒有不變量更糟。** 它會對正確答案發出違反警報，
然後修復迴圈把剩下的預算全部花在破壞一個本來是對的東西上。

所以這裡的分工跟 A 區一模一樣：**產生用的提示是理想性的，這支模組才是硬閘門。**
VerifiQuant 的 `_normalize_hint_option()` 對 `semantic_hints` 做的是同一件事。
兩個 repo 用同一個模式，這不是巧合，是同一條教訓。

一條不變量要活著通過閘門，必須：

- 能解析，而且**它用到的每個名字都真的存在**於產出或題目輸入裡。
  幻覺出來的欄位名是最常見的失敗
- 只呼叫白名單內的東西（`_SAFE_CALLS`：`abs` `min` `max` `sum` `len` `all` `any` `round`
  `sorted` `exp` `log` `log10` `sqrt` `floor` `ceil` `isfinite` `mean` `std` `diff`
  `maximum` `minimum` `where` `sign` `cumsum`）。不得 import，不得碰 dunder
- **引用一句真的出現在指令裡的句子**。這樣每條不變量都可以追溯到題目，
  而不是追溯到模型的先驗
- 講的是**我們的答案**，不能只講輸入資料
- 浮點數比較必須帶容差。裸的 `==` 會被改寫而不是被信任

AST 白名單刻意放進了 numpy 的遮罩慣用法（`~` `&` `|` `^`）。沒有它們，
這個語料上最有價值的不變量根本寫不出來：買賣權平價必須先選出 call 的列和 put 的列，
才有辦法把兩者關聯起來。

**降級機制**：當結構與量綱都乾淨、只有某條不變量一直失敗，那條會被降成 advisory，
不再消耗回合。替代方案是讓一行生成出來的 Python 決定整個單元的命運。

---

### L4a 交叉推導 — `agent/cross_derive.py`

前面四層沒有一層**把兩個報出來的數字互相對照**。但交付物的 schema 經常把同一個量報兩次，
一次是組成分量，一次是由它們算出來的東西：

```
filing_resolution.csv   value_total_in_filing, summarypage_value_total, summarypage_value_diff
swaption_prices.json    analytical_bps, tree_bps, tree_analytical_diff_bps
parity.csv              parity_lhs, parity_rhs, parity_error
```

第三欄跟前兩欄對不上的時候，**產出在自我矛盾**。這是本專案在不多花任何一次模型請求的前提下
能拿到的最強證據，成本是零請求。

**為什麼關係要用「擬合」而不是從名字解析。** 直覺的做法是查表：`*_diff` 就是把名字裡提到的
兩個欄位相減。這個做法撞到語料就碎了。`summarypage_value_diff` 是
`value_total_in_filing` 減 `summarypage_value_total`，而**它的名字兩個都沒提到**。

所以實際做法是：把每一組兄弟欄位拿去試一小組封閉形式，只有在某個關係能解釋除了少數幾列
以外的全部資料、而且沒有別的欄位組合解釋得一樣多的時候，才接受它。
巧合不可能擬合四十列。**擬合本身就是「這個關係是題目原意」的證據。**

這也決定了**沉默的意思**。如果沒有任何一組欄位能解釋這個衍生欄，最可能的情況是
組成分量根本不在產出裡（跟輸入比、或跟一個這支模組看不到的常數比）。
那**不是違反**，所以什麼都不主張。

**量出來的結果**：在 19 個附有 `checks/reference_data`（評分器自己的正確產出）的單元上，
建立了 9 條關係，發出 **0 個指控**，而 9 次單格突變 **9 次全部抓到**。
19 個裡只有 3 個能建立起任何關係，所以這一層在多數單元上就是沉默的，
而那個沉默是設計而不是缺口。

**`mean` 和 `avg` 被刻意排除在聚合關係之外。** 金融裡的「平均」通常是數量加權的，
而名字從來不會講。第一次真實 sweep 就抓到了：`avg_fill_price` 被指控跟 `fill_price` 欄的
簡單平均對不上，但評分器算的是 `sum(fill_qty * fill_price) / sum(fill_qty)`，
而 agent 的值跟評分器的期望常數**完全相等**。這一層當時差一點就要花一個修復回合，
把一個正確的加權平均改成錯誤的簡單平均。

`_MIN_ROWS` 與 `_close` 裡的每一個界限，都是負控制抓到它指控正確答案之後才設的。
沒有一個是事先選的。

---

### L4b 蛻變測試 — `agent/metamorphic.py`

L4a 只能在產出把同一個量命名兩次的時候說話。**87 個單元裡有 23 個寫的是扁平 JSON，
同一個量只出現一次**，L4a 永遠碰不到它們。這一層靠**重跑答案**去搆到剩下那些。

**原設計被證據推翻了一半。** 原本是「同一個量用兩條獨立路徑各算一次再比對」。
Nogueira 等人（arXiv:2607.02808，224 題 × 12 模型 × 5 語言）量到 3 版與 5 版集成
只實現理論獨立性效益的 43% 與 44%，同模型集成更差，而且提示策略對失敗相關性的影響
「相對次要」。Knight & Leveson 1986 的根因在 LLM 複製實驗裡原封不動地活下來：
共享的失敗是**共享的規格誤讀**。而我們的 Silent Wrong 族群**正好就是**模型誤讀指令的那一群，
我們唯一負擔得起的多樣性又正好是「同一個模型換個提示」，也就是被量到沒有效果的那根槓桿。

**蛻變測試不需要兩條路徑。** 它斷言的是**同一條路徑的兩次執行之間**的關係。
這就是它在共模失效的結論下還能活下來的原因：對題目的共享誤讀，
仍然逃不過「把橫斷面檔案的列打亂，答案不可以動」。

**不要把兩件事混在一起。** 被推翻的是「叫同一個模型再寫一條路徑」，
**不是**「把同一支程式當函數，改輸入再跑一次」。後者正是這一層在做的事。
0911 設計過一條 vega 探針（用解答自己的定價器算價格差分，對照報出來的 vega），
0922 退役路徑目錄時被一起丟掉，但它其實是一條宣告型蛻變關係。
細節與機會量見 `docs/2026-09-23_verification_layers_reference.md` §15.6。

照 Li、Yang、Liu、Yan（arXiv:2606.17529）的方法實作。一個候選關係要變成一個測試，
必須過四道閘門，而且結果是**有型別的**而不是布林：

| 閘門 | 問什麼 |
|---|---|
| 1 basis | 有沒有一個明講出來的物理或軟體理由，說這個關係成立 |
| 2 preconditions | 這一次執行拿到的輸入，滿不滿足那些前提 |
| 3 validity | 變換後的輸入還是不是合法輸入，輸出的對應在變換後還說不說得通 |
| 4 decidability | 判定容差有沒有**支配**量測算子本身的誤差地板 |

判決是 `admit`、`admit_ood`、`reject`、`defer` 四選一。
每一條活下來的關係是一張 **MR card**，帶著自己的前提、容差的來源、以及排除規則。

**第四道閘門的數字不採信任何假設。** 這一層的量測算子就是「再跑一次腳本」，
所以它的誤差地板就是同一份腳本在兩次相同執行之間的實際差異。
那是**跑兩次量出來的**，在任何關係被允許下判決之前。一個沒有設種子的蒙地卡羅
會以自己的標準誤在動，容差落在那個動幅底下的關係在這個單元上不可判定，
所以是 `defer` 而不是 fail。`_DECIDABILITY_MARGIN = 100.0`，而且第四道閘門
**逐欄位**套用而不是逐卡片。

目前實作的關係家族有三個：`determinism`（同輸入重跑）、`permutation`（列打亂，
需要至少 8 列）、`scale`（齊次縮放）。

**單調性被刻意排除。** Perignon et al. 2006 在真實選擇權資料上量到，
光是微結構雜訊就造成 6% 到 35% 的違反率。把它放進來就是在製造誤告。

---

### 目前狀態

- L1、L2、L3、L4a 已接進 `agent/agent.py` 的回合迴圈
- **L4b 已接進迴圈，但預設關閉**（`T1_METAMORPHIC`）。離線控制 12/12 通過
  （7 負控制、3 正控制、2 結構）
- 語料負控制已跑（`eval/metamorphic_sweep.py`，2026-09-26）：reward=1.0 的 23 題上
  **0 次誤告**，三法則的 95% 上界約 13%。唯一一條 hard finding 是真陽性
- 預設關閉的理由：advisory 模式下它改變不了任何答案，卻要花重跑的牆鐘，
  而且它的報告在官方評分環境裡拿不回來。完整論證在參考文件 §15.5

---

## C 區 — 指令遵循驗證

**問的問題**：題目叫你做的事，你做了沒？

**證據來源**：**題目文字本身**。

A 區與 B 區都在問「答案內部合不合理」。這一區問的是完全不同的問題。
用 V&V 的術語，L1 到 L4 是 verification（方程式有沒有解對），這一區是 validation
（宣告的規則有沒有被遵守）。

### 義務地圖 — `agent/obligations.py`

在任何程式碼存在之前，先把題目**明文**要求的事情編號列出來。
「明文」是嚴格的：一條義務只在指令把事情講死的時候才成立。指令容許多種讀法的部分，
刻意留給 L5，不在這裡處理。

兩個性質決定了它在迴圈裡的位置。

**一、只從指令抽，絕不從產生出來的程式碼抽。** 理由是量出來的：
一個 LLM 讀完實作再去寫檢查，會跟著**實作**走而不是跟著**題目**走，
正確率低於 50%，而且程式碼有 bug 的時候更差。從程式碼推出來的義務地圖，
會同意程式碼做的任何事。這跟 L4 的設計背後是同一條理由。

**二、在寫程式的模型跑起來之前就抽完。** 這樣它可以當成**檢查清單放在模型面前**，
而不只是事後拿來稽核。這個 agent 上每一次把東西往前搬（L1 契約、Phase 0 的實測 schema、
絕對輸入路徑）都是當下投報率最高的改動。至於清單本身就有效、
還是事後稽核另外有加成，那正是 `T1_OBLIGATIONS` 與 `T1_CRITIC` 兩個實驗臂在分離的東西。

**兩種義務，交給不同的裁判：**

| 種類 | 裁判 | 為什麼 |
|---|---|---|
| **宣告參數**（有字面值） | `check_literals`，**不呼叫模型** | 樹走訪就能定案，花一次請求只會買到更差的判決 |
| **方法 / 時序 / 資料處理**（沒有字面值） | `critic.py`，必須引證據且過確定性閘門 | 「取對數報酬」「不得用當期之後的資料」「先前向填補再丟棄」沒有字面可比對 |

第一種的實例值得記住。`t1-evt-pot-var` 在 `params.json` 裡宣告
`backtest_window: 500`，答案用了習慣的 250，於是算出 `1506-250=1256`，
而參考答案是 `1506-500=1006`。**這件事完全不需要任何意見。**

兩個細節顯示這一層在防的是什麼：

- `MODEL_KINDS` 是 `method` / `temporal` / `data_handling` / `numeric` / `convention`。
  **`parameter` 刻意不在裡面**。參數來自宣告它的檔案，是確定性的，
  再叫模型抽一次只是重複 `from_params` 已經找到的東西
- `_UNINFORMATIVE = {0, 1, 2, -1, 100}`。字面值 1 或 0 幾乎出現在任何腳本裡，
  找到一個不構成「宣告參數被遵守」的證據。硬要檢查的話，
  只會在程式碼剛好不含 1 的時候報違反，那是兩個方向的雜訊

還有一個折衷值得學：`token` 欄位。當指令自己就把要求寫成程式碼的樣子
（`ddof=1`、`method='lower'`、`floc=0`），**模型負責抽出這條要求，但檢查變回確定性的**。
這比一個意見更銳利，而且少一件事要分散 critic 的注意力。

---

### 批判者與證據閘 — `agent/critic.py`

它在所有確定性層都乾淨之後才跑。那正是**最需要它、也最危險**的時刻。

**最需要**，因為一個回合同時滿足了 L1 結構、L2 量綱、L3 不變量卻還是錯，
那就是 Silent Wrong 本身（實測 87 個單元裡有 36 到 39 個）。

**最危險**，因為在那些**已經正確**的單元上，它是唯一還能把事情弄壞的元件。

這個危險不是假想的。一支競爭隊伍在同一個 benchmark 上公開了四輪實測結果：
他們的對抗式 reviewer 的**判決**是可靠的（它核可的時候，5/5 被評為通過），
但它的**改寫**不可靠（8 次只成功 3 次，而三次失敗是把本來正確的答案弄壞）。
這個結論可以直接遷移，所以這裡的設計是：

- **批判者永遠不寫程式碼，只產生 findings**
- 一個 finding 要能傳到寫程式的模型面前，必須先有一個**確定性閘門**確認它引的證據真的存在
- `contradicted` + `hard` + 證據通過確認，是**唯一**會觸發修復的組合，而且**每個單元最多一次**

**閘門是整個設計的承重牆，而且它是普通程式碼，不是第二個意見。**
叫第二個模型去評判第一個模型，會有跟第一個一模一樣的問題。
閘門做的事又便宜又徹底：不在義務地圖裡的 id、不在指令裡的引文、
不在腳本裡的表達式、不在產出裡的欄位，**一律直接駁回，連讀都不讀**。

出來三個桶子，而且**三個都要報告**：

| 桶子 | 內容 | 後果 |
|---|---|---|
| `repairable` | 有接地、hard、contradicted | 可以觸發一次修復 |
| `marked` | 有接地但 soft，或 contradicted 但證據只有部分 | 記錄，永不修復 |
| `gap` | 證據不足、指令歧義、被閘門駁回的、以及**批判者根本沒提到的義務** | 只記錄 |

**`gap` 桶不是剩菜，它是整個設計的重點。**
它存在的理由是：**一條沒有人檢查得了的義務，絕對不可以看起來像一條通過了的義務。**

這個失效模式在本專案已經從外部被抓到兩次。第一次，主辦方的評分器在**參考資料遺失的時候
仍然給出自信的判決**。第二次，一份**全部是 NaN** 的產出通過了 L1 到 L3，
因為一條**無法被求值**的不變量被算成「沒有違反」。
`critic.py` 是這件事可能發生的第三個地方，所以 `gap` 桶被顯式做出來。

這一點值得停下來看。「無法評估」被當成「通過」，
就是**在證據不足的時候給出高自信的答案**，只是這次犯錯的是我們自己的檢查器，
而不是被檢查的模型。碩論要證的那件事，在工具鏈的每一層都會重演。

---

### L5 決策宣告（v0 已建，預設關閉）— `agent/decisions.py`

validation 的另一半：指令容許**多種合理讀法**的時候。
義務地圖刻意不碰這塊，因為一條義務只在指令講死的時候才成立。

**9/27 的做法**：Writer 在腳本開頭寫一個 `DECISIONS = [...]` 常值，逐條交代慣例選擇
（慣例軸、選項、依據是題目/參數/自己假設、實作它的程式碼錨點）。每一條用**翻轉錨點再重跑**來驗證，
沒有模型參與。慣例軸詞彙表就是突變測試的運算子（`agent/conventions.py`，兩邊共用）。
開關是 `T1_DECISIONS`，只寫紀錄 `_agent_decisions.json`，不改答案。
離線控制 `eval/test_decisions.py` 11/11；8 題實測見 `agenthon/docs/2026-10-02_meeting.md` 5.2a。

下面這張表是 9/18 做成因分析時的原始動機：

**為什麼它排在下一個。** 18 個被標記過成因的 silent-wrong 單元，分佈是：

| 成因 | 數量 | 該由哪一層處理 | L5 碰得到嗎 |
|---|---|---|---|
| A 實作或數值錯誤 | 8 | L4 | 碰不到。量的定義明確，程式完全合規，錯在數學 |
| A + B | 3 | L4 標記、L5 救 | 部分 |
| **B 慣例讀法** | **6** | **L5** | 「明確參數沒被使用」碰得到而且是確定性的；用詞撞名與題目沒寫清楚碰不到 |
| F 輸出格式 | 3 | L1 封閉 schema | 部分重疊 |
| G 退化值 | 3 | L2 / L3 | 碰不到 |

L5 上看約 8/28，對照 L4 的 11/28 量級相當，但**成本低一個數量級**：
不需要路徑目錄、不需要六種路徑家族、不需要蒙地卡羅假設檢定。

---

### 對照 VerifiQuant：同一件事，兩種做法

A 區的 I 層（`semantic_hints`）做的是跟 L5 一樣的事，但走卡片路線：
**歧義是事先由卡片宣告的**，執行時比對到就要求澄清，而且附一個數值驗證過的轉換。
L5 沒有卡片可用，所以必須從指令文字當場推。

更值得注意的是 L3 與 A 區的關係。兩邊都遇到同一個結構性問題：
**需要模型產生檢查，但模型會產生錯的檢查。** 兩邊的答案也一樣：

| | VerifiQuant | Agenthon |
|---|---|---|
| 模型產生什麼 | `semantic_hints[].options[].transform_spec` | 一條可執行的不變量 |
| 提示的地位 | 理想性的 | 理想性的 |
| 硬閘門 | `_normalize_hint_option()` | `invariants.py` 的白名單與引文檢查 |
| 閘門駁回的例子 | 下標語法、函式呼叫、沒引用到 `result`、對預設選項做恆等變換 | 幻覺欄位名、import、裸 `==`、引了指令裡不存在的句子 |

這不是巧合，是同一條教訓在兩個語境下各記一次：
**模型的結構化輸出提示永遠是理想性的，真正把關的是後處理的過濾器。**

---

## D 區 — 前置與環境驗證

**問的問題**：這題在環境上根本跑不跑得起來？

**證據來源**：真實的輸入檔、執行環境、評分器的規則。

這一區不判斷答案對不對。它處理的是「連答錯的機會都沒有」的那些失敗。

### 檔案

- `agent/explore.py` — **第零階段**。在寫任何程式之前，先讀真實輸入並說出裡面有什麼。
  理由是量出來的：2026-09-16 的 87 單元 sweep 裡，19 個單元從來沒產出過跑得起來的腳本，
  而那些單元的 109 個腳本錯誤裡有 82 個（75%）是名稱或 schema 錯誤
- `agent/inputs.py` — **路徑改寫**。題目宣告輸入在 `/app/data/returns.csv`，
  那是主辦方映像檔裡的路徑，我們的容器只透過唯讀的 `/input` 掛載拿到檔案。
  87 個單元裡 69 個宣告在 `/app/` 底下，只有 1 個在 `/input/`。
  照字面執行的 agent 幾乎每題都死在 FileNotFoundError。
  **注意：這支不做 symlink。** symlink 方案在 2026-09-17 因為評分環境根目錄唯讀而退役，
  現在改寫的是題目**文字**，重跑機制是在腳本原始碼裡做絕對路徑字串替換
- `agent/canary.py` — **金絲雀隔離**。閘門 g2 會讓任何含有單元 canary GUID 的提交失敗，
  而那個 GUID 就在 87 個單元中 66 個的 `instruction.md` 前兩行。我們會把指令送給模型，
  所以沒有這支的話，GUID 離被複製進交付物只差一個抄過去的檔頭註解
- `agent/checks_reader.py` — 在執行時讀單元自己的檢查器。兩份官方文件對這個檔案在不在容器裡
  講法互相矛盾，所以策略是「在的時候讀，不在的時候退回」
- `agent/probe_solve.py` — 探測用的最小求解流程

---

## E 區 — 驗證器的驗證（負控制）

**問的問題**：這個檢查器本身會不會亂講話？

**證據來源**：**已知正確**與**已知錯誤**的案例。

這一區是整個專案的方法論核心，也是最該先讀的一區。規則只有一條：

> **先懷疑量尺，再懷疑被量的東西。**

一個從來沒有指控過任何東西的檢查器，說它「誤告率低」是沒有意義的。
所以每一層在上線之前，都要先對一份**已知正確**的解跑過，確認它**閉嘴**；
再對一份**已知錯誤**的解跑過，確認它**開口**。兩邊都做完才算數。

這不是抽象原則。以下每一條都是這條紀律實際抓到的東西，
而且每一條都是在「差一點就要根據錯誤數字下結論」的前一刻抓到的。

### 已經抓到的儀器錯誤

**一、分母差兩倍。** 語料負控制工具把 87 個單元數成 174 個，
因為 `--save-outputs` 會在每個 `<unit>.output` 旁邊寫一個 `<unit>.verifier`，
兩者都被當成單元。誤告率的分母會**剛好差一倍**。
修法是只接受 `gs-<unit>` 與 `<unit>.output` 兩種命名，其餘的**計數並印出來**而不是丟掉。
一個被靜默忽略的目錄，正是一個語料控制只量到一半語料卻什麼都不說的方式。

**二、L4a 差一點把正確答案改壞。** `avg_fill_price` 被指控跟 `fill_price` 欄的
簡單平均對不上。實際上評分器算的是 `sum(fill_qty * fill_price) / sum(fill_qty)`，
agent 的值跟評分器的期望常數完全相等。結論是把 `mean` / `avg` 整個家族從聚合關係裡拿掉，
因為金融裡的「平均」通常是數量加權的，而名字從來不會講。

**三、L1 把輸入檔升級成交付物。** `params.json`、`order.json`、`forward_returns.csv`
各自是一個被讀取的輸入，卻被輸出區段掃描當成必須產生的檔案。
agent 於是花回合去生成題目根本沒要求的東西。修完精確率 0.899 → 0.914，召回率不變。

**四、無法評估被當成通過。** 一份**全部是 NaN** 的產出通過了 L1 到 L3，
因為一條**無法被求值**的不變量被算成「沒有違反」。
這直接導致了 `critic.py` 的 `gap` 桶設計。

**五、來自外部的同一個錯誤。** 主辦方自己的評分器在**參考資料遺失的時候**
仍然給出自信的判決。這不是我們的程式碼，但它是同一個失效模式，
而且它證明這個模式不是我們特別笨才會犯。

### 最大的一次：通過數本身就不是一個可用的量尺

三次全量 87 單元 sweep 的結果是 **18 / 21 / 15**。看起來像是「後面加的層讓成績變差了」。

實際量下去：只有 **8** 個單元三次都過，**28** 個至少過一次。
拿同樣的程式碼重跑 15 個已知通過的單元，翻掉 2 個（**13%**）。

所以**幾個單元的差距是不可解讀的**，21 → 15 這件事本身沒有訊息。
再把層的訊號跟運氣分開看：所有層都回報乾淨的單元從 56 降到 49，
其中被評為 1.0 的從 20 降到 11，而被評為 0.0 的（Silent Wrong）從 36 升到 38。

**穩定的量是 Silent Wrong（約 38 到 39 / 87），不是通過數。**
動的是「模型剛好在所有層都沉默的地方答對」的頻率。
這個發現改變了整個專案要報告什麼指標，而它完全來自於懷疑量尺。

### 在建之前先量機會

`eval/l4_opportunity.py` 是這個做法的樣板：**在寫 L4 之前，先量 L4 碰得到幾題。**
碰不到就不要建。同樣的前置量測現在是 L5 的擋門條件。

這跟一般的「先做再說」相反，而且它救回來的時間比它花掉的多。
L4b 的原設計（路徑比對）就是這樣在寫任何程式碼之前被文獻推翻的。
被推翻的是同模型重寫路徑，不是重跑程式，見參考文件 §15.6。

### 這一區跟碩論的關係

這一區是整份研究跟多數同類工作最不一樣的地方。

多數「LLM 自我驗證」的工作只報告：我加了一個檢查，分數上升了。
**它們不報告那個檢查誤告了幾次，也不報告它在正確答案上是不是閉嘴。**
而誤告的成本在這個專案裡是實際量過的：一次誤告會花掉一個修復回合，
把一個本來正確的答案改壞。

E 區的存在，就是讓「我們的檢查器不會在證據不足時給出高自信判斷」這句話
變成一個有數字支撐的主張，而不是一個自我宣稱。

### Agenthon 側

**離線控制（不呼叫模型、不連網）**
- `eval/test_metamorphic.py` — L4b，12/12（7 負控制、3 正控制、2 結構）
- `eval/test_decisions.py` — L5，11/11（4 負控制、5 正控制、2 結構）
- `eval/test_mutation_score.py` — 突變測試工具本身，7/7
- `eval/test_cross_derive.py` — L4a
- `eval/test_invariants.py` — L3
- `eval/test_invariant_retrieval.py` — L3 檢索
- `eval/test_obligations.py` — 義務地圖
- `eval/test_clock.py` — 時鐘
- `eval/test_house_route.py` / `test_house_route_live.py` — 內建模型路由

**語料層級的負控制**
- `eval/metamorphic_sweep.py` — 對整個 87 單元語料量 L4b 的誤告率。2026-09-26 已跑：0/23 誤告，上界約 13%
- `eval/cross_derive_sweep.py` — 同樣的事，對 L4a
- `eval/mutation_test.sh` — exemplar 上的三個手寫突變，看 L2 抓不抓得到
- `eval/mutation_score.py` — **逐題突變分數，不需要 grader**。對每份腳本套慣例突變（拿掉 /365、252↔365、
  ddof、對數↔算術報酬等），重跑，看哪一層開口。9/27 全語料：62 題 1,106 個突變，我們抓 21.9%
- `eval/mutation_grade.py` — 同一批突變給官方 grader 評分：grader 75%、我們 22%、只有我們 3.5%
- `eval/l4_opportunity.py` — **在建之前先量機會**。L4 能碰到幾題？不值得就不建
- `eval/measure.py` / `measure_input_rewrite.py` / `measure_unpresented_inputs.py` — 各項前置量測
- `eval/loop_detector_check.py` / `loop_probe.py` — 迴圈偵測
- `eval/generic_verifier_diagnostics.py` — 通用驗證器的診斷
- `eval/truncation_analysis.py` — 截斷分析
- `eval/agent_loop_controls.py` — agent 迴圈的控制組

### VerifiQuant 側

- `test_transform_poc.py` — 根目錄，離線，不需 API，端到端驗證 `stage_transform.py`
- `verifiquant/preprocessing/validate_relations.py` — 跨產出物完整性
- `verifiquant/preprocessing/lint_echecks.py` — E 層檢查的 lint 閘門
- `scripts/validate_strict_binding.py` — 字面接地稽核的驗證
- `scripts/assert_paper_numbers.py` — **論文數字的斷言守衛**。論文裡寫的每個數字，
  都要能從 runs 目錄重新算出來

---

## F 區 — 實驗與證據

**問的問題**：上面那些主張，數字在哪裡？

**證據來源**：實際跑出來的結果。

論文裡的每一個數字都應該追溯到這一區的某個檔案。

### 執行與分析腳本 — `scripts/`

**建資料集**
- `build_trap_set.py`（在 `verifiquant/pipeline/`）/ `build_trap_set_v2.py` — 陷阱題集
- `build_messy_query_pilot.py` / `build_messy_answer_gold.py` — 雜訊查詢集
- `trim_messy_set_to_80.py` — 雜訊集從 82 修到 80。刪除 mq-81、mq-82 的依據是結構性的、
  與標註無關的條件
- `curate_messy_missing_items.py` — 補缺項
- `build_nearood_probes.py` — 近 OOD 探針
- `build_subset_libraries.py` — 子集卡片庫，給檢索覆蓋率掃描用
- `build_mixed_risk_coverage.py` — 混合 risk-coverage 曲線
- `sample_dataset.py` / `sample_dataset_250.py` — 抽樣

**跑實驗**
- `run_paper_experiments.py` — 論文實驗總入口
- `run_pal_baseline.py` / `run_retrieval_pal_baseline.py` / `run_retrieval_pal_governance_eval.py` — PAL 基線與檢索增強版
- `run_sc_baseline.py` — self-consistency 基線
- `run_multik_refusal.py` — K 值消融（K=1 自主 81.2%，K=3 93.6%）
- `run_refusal_ablation_grid.py` — 拒答提示消融網格
- `run_ood_family_eval.py` / `run_nearood_eval.py` — OOD 家族評估
- `run_repairbench.py` — 修復基準
- `run_stats_250q.py` — 250 題統計
- `run_vq50_pro.py` — 50 題 Pro 模型
- `run_gate_ladder.sh` / `run_libsweep.sh` / `run_wave1_chain.sh` / `supervise_card_build.sh` — 批次執行

**分析與繪圖**
- `analyze_governance_eval.py` / `analyze_governance_confirmatory.py` / `analyze_governance_multirun.py`
- `analyze_retrieval_pal_baseline.py` / `analyze_v2_by_tier.py`
- `aggregate_messy_labels.py` — 雜訊集標註彙總
- `score_messy_external.py` + `score_messy_external_SPEC.md` — 外部效度評分與其規格
- `score_rpal_plus_clean250.py`
- `measure_retrieval_recall.py` — 檢索召回率
- `compare_trigger_runs.py` — 觸發比較
- `plot_risk_coverage.py` / `plot_safety_utility.py` / `plot_libsweep.py` — 論文圖

**卡片維運**
- `import_m_cards.py` / `patch_fic_cards.py` / `promote_npv_timing_transform.py` / `promote_loan_timing_hard.py`

### 結果 — 讀這些而不是重跑

- `verifiquant/data/runs/<實驗名>/` — 每次跑的原始輸出。目錄名就是實驗名
  （`gate_ladder`、`governance_eval`、`governance_confirmatory`、`libsweep`、
  `messy_query_pilot`、`mixed_risk_coverage`、`nearood`、`ood_eval`、`pal_baseline`、
  `paper_v1`、`paper_v2_250`）
- `docs/results/` — 日期前綴的實驗紀錄
- `docs/2026-*.md` — 每次實驗的敘述性紀錄，比 JSON 好讀。特別是
  `2026-07-12_experiment_record_master.md`（總表）與 `2026-07-21_reviewer_evidence_scorecard.md`
- `agenthon/eval/*.txt` / `*.json` / `*.csv` — Agenthon 側的結果快照
  （`key-results-v*.txt`、`gold-results.txt`、`probe-results*.txt`、`measure-results.json`、
  `silent-wrong-38-0918.md`、`silent-wrong-value-28.csv`、`taxonomy-0918.json`）

### 兩個數字上的警告

**Agenthon 的通過數不穩定。** 三次全量 87 單元 sweep 的結果是 18 / 21 / 15。
只有 8 個單元三次都過，28 個至少過一次。同樣的程式碼重跑 15 個已知通過的單元，
有 2 個翻掉（13%）。所以**幾個單元的差距是不可解讀的**。
穩定的量是 Silent Wrong（約 38 到 39 / 87），不是通過數。

**VerifiQuant 的 CoT 準確率有 ±5% 的執行間變異。** 確定性的 E 層與 F 層才是穩定的部分。

---

## G 區 — 論文與對齊文件

### 論文 — `paper/`

- `main_final.tex` — **已投稿版本**（ICAIF 八頁，最後一天 2026-08-10）
- `main_v2.tex` / `main_v3.tex` — 演進中的版本
- `main_v2_08041100_best.tex` 等日期後綴檔 — 定點快照，不要當成現行版
- `preamble.tex` / `numbers.tex` / `references.bib` / `Makefile`
- `appendix_rq4_protocol.tex` — RQ4 協定附錄
- `figures/` — 論文圖，由 `scripts/plot_*.py` 產生
- `main_*.pdf` — 編譯產物快照，檔名帶日期時間
- `archive/` / `main_v1/` — 歷史版本

**注意**：根目錄的 `paper/` 有五個 `.tex` 的刪除尚未 commit
（`main.tex`、`main_body.tex`、`main_body_v2.tex`、`main_with_appendix.tex`、`appendix.tex`），
看起來是整併進 `main_final.tex` 的收尾，但沒有進版控。

### 對齊文件 — `docs/`

按日期排，是整個專案的決策軌跡。重要的幾份：
- `docs/2026-07-12_experiment_record_master.md` — 實驗總表
- `docs/2026-07-21_reviewer_evidence_scorecard.md` — 每個主張對應到哪個證據
- `docs/2026-07-29_phase_split_icaif_acl.md` — ICAIF / ACL 的階段切分
- `docs/2026-07-29_governance_go_no_go_protocol.md` — 治理評估的 go/no-go 協定
- `docs/2026-08-03_rq4_messy_external_validity.md` — 雜訊查詢的外部效度
- `docs/2026-08-04_review_response_map.md` — 審稿回應對照
- `docs/ICAIF_2026_8page_outline.md` — 八頁版大綱
- `docs/archive/` — 設計源流，讀**意圖**用。**不要假設這裡描述的功能存在**。
  舊的 NSFCA 願景寫了 SymPy 數學引擎與 PDF/RAG，那兩個從來沒有被實作

### Agenthon 對齊文件 — `agenthon/docs/`

- `2026-10-02_meeting.md` — **最新的一份，轉向提案**。Part 2 是局部最優的量測（我們是官方 grader
  較弱的子集），Part 4 是突變測試與導數探針的機會量，Part 5 是 L5 決策宣告與第一次實測
- `2026-09-25_meeting.md` — 八個 Part，涵蓋 L1 到 L4b 的
  全部設計、量測、以及被推翻的部分。Part 6 是 L4b，Part 7 是全部 17 次 sweep 的回溯
- `../submission/zips/README.md` — 每次上傳的內容、官方逐題結果、同一個 zip 被評兩次的翻轉率
- `2026-09-06_from_verifiquant_to_agenthon.md` — 兩個專案的銜接論述
- `2026-09-17_verifiquant_vs_agenthon_direction.md` — 方向上的差異
- `2026-09-11_advisor_briefing.md` / `2026-09-18_advisor_briefing.md` — 指導會議
- `research-plan.md` / `lit-review/` — 研究計畫與文獻
- `2026-09-17_agenthon_track1_overview.md` — 競賽規則整理

### 會議與報告

- `MEETING_BRIEF_advisor.md` / `weekly meeting 碩一下.txt` — 歷史會議紀錄
- `CASE_STUDIES.md` — 個案研究
- `台大_r14946005_資科碩二_柯宥圻.md` — 個人進度

---

## H 區 — 外部基準、展示層、歷史

### 外部基準（git submodule，但沒有 `.gitmodules`）

這三個是 gitlink，`git status` 會顯示 `m` 表示裡面有變更。它們有自己的 origin。

- `FinanceReasoning/` — FinanceReasoning 基準
- `finchain/` — FinChain 基準
- `reimplement-jpmogan/` — JPMorgan 多代理人基線的重實作（論文報 0.46 Pass@1）
- `financereasoning.pdf` — 對應論文

### 展示層

- `app.py` — Flask UI，包住 `ErrorClassificationAPI`。約 36KB
- `templates/` / `static/` — 前端。`demo.html` 與 `static/demo.js` 是 Demo Console
- `deploy.sh` / `Dockerfile` / `requirements-web.txt` / `.gcloudignore` — GCP 部署。
  **注意 `.gcloudignore`（不是 `.dockerignore`）才是控制上傳大小的那個檔案**，
  而且金鑰來自 Secret Manager 而不是 `.env`

### 歷史與廢棄，不要當成現行

- `preprocessing/` — 多數是 **12 行的 CLI 薄包裝**，只是把 `verifiquant/*` 重新匯出。
  **改那些沒有任何效果**，要改 `verifiquant/` 底下的真檔案。
  例外是五支獨立工具，它們有實作內容：`extract_config_questions_to_jsonl.py`（128 行）、
  `analyze_cot_log.py`（81 行）、`visualize_config_eval.py`（263 行）、
  `visualize_expand_eval.py`（372 行）、`visualize_lg_pipeline.py`（442 行）。
  `.claude/CLAUDE.md` 說只有三支，那是舊的
- `scratch/` — 一次性的除錯程式，不屬於任何流程
- `verfifiquantfinal/` — 舊的簡報與 PDF
- `template/` — ACM 論文模板
- `rag_pipeline/` — 只剩 `__pycache__`，等於空的
- `data/` — 空目錄，實際資料在 `verifiquant/data/`
- `output/` / `tmp/` — 暫存產物
- `ORACLE_PROMPT_FINAL.py` / `SKILL.md` / `QUICK_START.md` — 早期產物
- `agenthon/track1-coding-public/` / `agenthon/Agenthon2026-public/` — 上游 clone，
  共 235MB，已被 `.gitignore` 排除，用 `git clone` 重建

---

## 3. 兩個系統的對照

同樣的檢查精神，在有卡片與沒卡片兩種情況下長成不同樣子。

| 檢查什麼 | VerifiQuant（有卡片） | Agenthon（沒卡片） | 確定性 |
|---|---|---|---|
| 問題在不在能力範圍內 | N 層，檢索為空 | 無對應（題目一定要答） | 是 |
| 必填欄位齊不齊 | F 層，卡片宣告 `inputs` | L1，從 `instruction.md` 抽結構契約 | 是 |
| 數值合不合理 | E 層，卡片的 `scale_checks` | L2，單位與正負號的固定辭典 | 是 |
| 滿不滿足恆等式 | 卡片的 `invariants` | L3，模型每題現場產生 | 否（L3） |
| 報出來的數字彼此對不對 | 無對應 | L4a 交叉推導 | 是 |
| 改題目答案該不該變 | 無對應 | L4b 蛻變測試 | 是 |
| 隱性歧義 | I 層，卡片的 `semantic_hints` | L5 決策宣告（v0，預設關閉） | 否 |
| 指令有沒有被遵守 | 卡片本身即規格 | 義務地圖 + Critic | 部分 |

值得注意的是**空格在哪裡**。VerifiQuant 缺 L4a / L4b 那一類「答案內部互相對照」的檢查，
因為它的答案通常是單一數值。Agenthon 缺 N 層，因為競賽不允許拒答。

---

## 4. 碩論的方向：外部約束與內部約束

目前這個 repo 裡的**所有東西都是外部約束**。

卡片庫、六層漏斗、L1 到 L5、義務地圖、Critic，全部都在模型**外面**：
先讓模型產生，再用確定性的規則去檢查、攔截、要求澄清、或套用經過數值驗證的修復。
模型的決策過程本身是黑箱，我們只管輸出。

這條路的好處是可稽核。每一次攔截都有證據、有規則編號、有可重現的判定。
壞處是它永遠在事後。模型已經把錯誤的推理走完了，我們只是不讓它出門。

### 下一步想探索的：steerability

<https://steerability.github.io/>

從模型**自己的決策流程**裡面做限制，而不是在外面攔。具體來說是用 schema 去控制
生成過程本身，讓「在證據不足時硬答」這個行為在產生的當下就受到約束，
而不是產生完再被攔下來。

### 這份 repo 裡哪些地方是接點

如果要把外部約束換成內部約束，最自然的接點依序是：

1. **I 層的 `semantic_hints`（A 區）**。現在的流程是卡片事先宣告歧義，
   執行時比對到就要求澄清。steering 版本會是：讓模型在面對這個歧義時**無法**直接選一邊，
   而是必須先輸出一個歧義標記。歧義的宣告從卡片移到生成約束
2. **L2 量綱（B 區）**。單位與正負號是最像 schema 的東西。
   現在是事後查表，內部約束版本可以在生成時就綁定型別
3. **義務地圖（C 區）**。現在是「先列義務，事後查有沒有做」。
   內部約束版本是讓義務成為生成的骨架，沒有覆蓋到的義務就產不出東西
4. **M 層的拒答（A 區）**。這是最難也最值得的一個。目前拒答是外部判斷之後加上去的，
   而消融實驗顯示 CoT 可以被提示成會拒答，但**不具選擇性**（過度拒答遠多於正確拒答），
   Gemini 則幾乎不拒答。這正是「高自信硬答」在數字上的樣子

### 論文要證的那件事，證據目前在哪

| 主張 | 證據位置 |
|---|---|
| 模型在證據不足時會高自信硬答 | `docs/results/2026-06-09_refusal_prompt_ablation.md`、`agenthon/eval/silent-wrong-38-0918.md` |
| 提示式的拒答不具選擇性 | 同上，拒答消融 |
| 外部確定性約束可以選擇性地攔截 | `docs/2026-07-29_governance_matched_results.md`、risk-coverage 曲線 |
| 攔截行為可以被稽核與重現 | `scripts/assert_paper_numbers.py`、`verifiquant/data/runs/` |
| 這套精神可以遷移到沒有卡片庫的場景 | `agenthon/`，L1 到 L4b |
| 檢查器本身的誤告率是被量過的 | E 區全部 |

最後一列是這份研究跟多數同類工作最不一樣的地方，也是最該在論文裡寫足的地方。

---

## 5. 版本控制與 repo 邊界

這個工作目錄裡有**兩個獨立的 git repo**。

**`verifiquant-update/`**（本目錄）
- remote：`github.com/datafox-tw/VerifiQuant.git`，private
- 現行分支：`main`
- `paper-icaif-8page-trim` 已合併進 `main` 並刪除本機分支
- `.env` 有被 gitignore，未被追蹤

**`agenthon/`**（巢狀，不是 submodule）
- remote：`github.com/datafox-tw/responsible-agent.git`，private
- 刻意不設成 submodule，因為競賽程式碼有自己的節奏
- 已被本 repo 的 `.gitignore` 排除，避免 `git add -A` 把它整包吞掉
- 標籤 `pre-sweep-l4b-0923` 記錄 L4b 建好但未接線的狀態

**憑證規則**
- 金鑰只住在 `verifiquant-update/.env`，執行時才讀
- 上傳到競賽的 zip 是**公開的**，只能帶用金鑰算出來的 proof，不能帶金鑰本身
- 部署用的金鑰來自 GCP Secret Manager，不是 `.env`
- 列出環境變數時只印**名稱**，不印值

---

## 6. 從哪裡開始讀

**如果你是第一次看這個專案**
1. 這份文件的第 0 節與第 2 節
2. `README.md` 的架構段（六層漏斗的完整表格）
3. `agenthon/docs/2026-10-02_meeting.md` 的 Part 1 與 Part 2（目前所有控制元件的總覽，以及為什麼要轉向）

**如果你要改 VerifiQuant 的程式**
1. `.claude/CLAUDE.md`，特別是 Common Pitfalls
2. `verifiquant/pipeline/run_error_classification_pipeline.py`
3. 記住：改 `preprocessing/` 的薄包裝沒有效果

**如果你要改 Agenthon 的程式**
1. `agenthon/docs/2026-10-02_meeting.md` 的 Part 6（時程、停損點、刻意不做的事）
2. `agenthon/submission/zips/README.md` 的「下一步」與「每次上傳前的固定檢查」
3. 對應層的模組 docstring，每支開頭都寫了「這層為什麼存在」而且附量測數字
4. 動任何一層之前，先跑該層的 `eval/test_*.py`

**如果你要寫論文**
1. `docs/2026-07-21_reviewer_evidence_scorecard.md`
2. `scripts/assert_paper_numbers.py`，確認要引的數字算得出來
3. `paper/main_final.tex`
