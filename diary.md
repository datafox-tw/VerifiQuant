## 20260405
早就應該寫日記了，這樣才能夠紀錄我每週都卡在哪些詭異的地方，而不只是我在簡報裡面看起來很完整的產出，中間的混亂過程也是需要紀錄的。今天主要處理內容，就是在生成卡片之前的思考，我不能夠一直生成一樣的卡，而是讓ai半輔助（靠過去rag找到的紀錄來判斷到底需要完全重寫、生成補助判斷的公式，以及完全不補充公式等功能。以及今天在測試的是，如果遇到各種錯誤時要怎麼搞
- duplicate core fic_id: fic_npv_calculation
- duplicate core fic_id: fic_npv_calculation
- duplicate retrieval fic_id: fic_npv_calculation
- duplicate retrieval fic_id: fic_npv_calculation
- duplicate repair rule for fic_id=fic_npv_calculation, rule_id=dr001
- duplicate repair rule for fic_id=fic_npv_calculation, rule_id=dr002
- duplicate repair rule for fic_id=fic_npv_calculation, rule_id=dr003
- duplicate repair rule for fic_id=fic_npv_calculation, rule_id=dr004
- duplicate repair rule for fic_id=fic_npv_calculation, rule_id=dr005
- repair rule fic_npv_calculation/dr006 has no matching core diagnostic_check
- duplicate repair rule for fic_id=fic_npv_calculation, rule_id=global_n_not_supported
- duplicate repair rule for fic_id=fic_npv_calculation, rule_id=global_i_semantic_ambiguity
- repair rule fic_npv_calculation/npv_i_001 has no matching core diagnostic_check
- repair rule fic_npv_calculation/npv_i_002 has no matching core diagnostic_check
- repair rule fic_npv_calculation/npv_s_001 has no matching core diagnostic_check
- repair rule fic_npv_calculation/npv_s_002 has no matching core diagnostic_check
- duplicate repair rule for fic_id=fic_npv_calculation, rule_id=global_n_not_supported
- duplicate repair rule for fic_id=fic_npv_calculation, rule_id=global_i_semantic_ambiguity

1. 請問duplicate core/retrieval fic_id這個的邏輯是llm在得到input資料->生成core或者是retrieval時，選擇了重複的名稱，導致重複是嗎
2. 同上，如果不小心生成同樣的fic_id，也會生成相對應的重複duplicate repair rule，因為生成repair rule的邏輯重複
3. 如果我們有辦法確定fic_id都是同一批，而非有很多個fic_id名稱一樣但是是從不同的input出現的fic card，應該就不會導致repair rule has no matching core diagnostic_check了，是嗎？
3. 如果我們有辦法確定fic_id都是同一批，而非有很多個fic_id名稱一樣但是是從不同的input出現的fic card，應該就不會導致repair rule has no matching core diagnostic_check了，是嗎？請先不要寫code，我們來align邏輯



是，你理解正確。
core/retrieval 的 fic_id 來源是 LLM 產生的 core card。當多題（例如都 NPV）被 LLM 命成同一個 fic_id，就會在 artifact level 出現 duplicate core，retrieval 也會跟著 duplicate（因 retrieval 直接用 core 的 fic_id）。

是，邏輯上也正確。
repair rule 是「依該 core 的 diagnostic_checks 生成」。如果 fic_id 重複，且 rule_id 命名模式也類似（dr001... + global rules），就會變成同 (fic_id, rule_id) 重複，觸發 duplicate repair rule。

大方向是對的，但有一個關鍵補充。
如果你能保證「同一個 fic_id 只對應同一張 canonical core（同一套 diagnostic_checks）」就不會有 repair rule has no matching core diagnostic_check。
這個錯誤通常是「同名 fic_id 其實混入不同版本 core」，導致某些 repair rule 的 rule_id 在當前被保留的那張 core 上找不到對應 check。
所以重點不是只看「名稱一樣」，而是要保證 fic_id -> 唯一且穩定的 core schema/check set。

因此，現在要做的事情是稍微又（怎麼說又）去更新我存卡和進行判斷的動作。但如果是這樣的話，我是不是應該反而用不同的方式來解決問題？
如果我們引進「更新卡片」的觀念，代表我們需要base，與其讓base亂生然後再進行修改，我其實早就有一個不錯的燃料。我當時不採用的原因，是發現上面的code對於下面的code來說只是參考，但是理論上我們也可以先用上面的code去創建base資料集，然後把上面的部分和下面的部分分開，這樣下來我不只是可以最佳化原始的分類，同時又可以優化系統，避免遇到無法相容的問題，在跟老師說明的部分也會更有說服力。
現在的問題是要用finchain系列還是financereasoning系列，讓我來做一點點資料前處理吧。（換不同的資料夾）
你可以先不要寫code，幫我看我剛剛靈機一動想的狀況，可能可以解決剛剛提到的問題，而且原始的code架構也不用改很多：參考dataset_exploration.md 都是我手打的

我已經把我篩選過的資料(25題常錯，25題常對，這個適用gemini-2.0-flash進行運算的，是已知數據，而且是random seed+分層隨機抽樣）放到了verifiquant/data/config.yaml裡面
包含
target_ids:
  correct_samples:
  error_samples:
你可以繼續修改建立fic卡的流程，改成吃這些資訊：
1. function_id:string
2. article_title:string
3. function: string（python code）
4. 用financial_documents.json 的article_title外接到financial_documents.json的title，應該是one-on-one matching 
如果你願意，也可以先複製一份原本的，然後將新的code進行修改，剩下的卡片生成邏輯應該無需更改，但也請幫我確認。

在開始把那五十題dataset catd跑出來之前我希望可以再確認一次目前卡片設計
階段
代碼
名稱
定義與攔截邏輯
系統行動
Intent
M
Misunderstanding
用戶意圖太籠統（例如「這會賺錢嗎？」）。
Refusal: 引導式反問。
想確認的部分：今天的M類比較像是，有找到相似的卡片，但是在根據retrieve卡的diagnosis之後發現這題主要失敗原因是「過於籠統或者題意不清導致無法回答」，並且要求用戶提供更多細節，在目前的卡片生成邏輯中，有做到這件事情嗎？

Scope
N
Not Supported
意圖明確，但目前 FIC 庫中無對應公式。
Graceful Exit: 宣告知識邊界。
想確認的部分：如果M類錯誤是用戶提問過於籠統或者不明確導致的錯誤，那麼N類問題則是要讓系統自己承認「我們沒有抓到足夠合適的卡片」，目前的系統和卡片生成邏輯中，有辦法確定這條與M類之間的差異嗎？

Schema
F
Formula Spec
FIC 已選定，但缺少必要輸入（如缺 Rf）。
Slot-filling: 精確要求補值。
F類是屬於已經選定了FIC卡片，但是在retrieve相關數值的時候發現沒有對應的數字或者數字格式奇怪，目前有辦法根據抓到的input提出refusal嗎？

Boundary
E
Extraction Alert
數值存在但邏輯荒謬（如 8 vs0.08）。
Deterministic Alert: 基於規則攔截。
目前這個是屬於「雖然可以填入數值，但是數值跑出來的結果荒謬或怪異」，然後是屬於有公式可驗證的（例如如果數字大於2的話，代表這個2是百分比不是真實值應該除以一百之類的操作」目前有辦法抓到嗎

Critic
I
Interception
隱性歧義（如匯率方向、期初期末）。
Critic Agent: 啟動 HITL 對齊。
這個是最重要的，因為在自我審查和複習的時候，有些應該要檢查的內容不是一個公式或者是值的大小檢查可以搞定的，例如說計算到底是年初還是年末，或者是IND-GBP到底誰是分母誰是分子等（例如gemini覺得GBP是分母，gpt覺得IND是分子），所以如果發生這樣的情況的話，一定要跟用戶做classification

Logic
C
Calculation
最終確定性執行階段。
Audit Log: Python 執行。

首先今天先測試了六題左右 然後開始狂跑50題
結果後來在14題的時候徹底卡住 我還是受不了讓他重跑了 也不知道是哪裡的問題 所以加起來大概是20題 然後等等又要50題 20+50等於70 
## 20260406
收集三個的結果然後開始跑結果
首先需要測試我選擇的那個ai到底還能不能用 然後開始跑50題 看目前的狀況
N 會發生在 selector 回傳 abstain_n、chosen_fic_id 找不到 core card，或 top_score < m_min_top_score。
但是這樣的話應該都是前兩個吧 第三個top score現在沒有用到

對，你這個設定下幾乎是這樣。

在你目前這組參數（--db-url + --top-k 5 + m_min_top_score=0.05）：

只要有檢索到候選，top_score 會是 1.0~0.2，一定 > 0.05
所以第 3 個條件幾乎不會觸發
第 3 個目前主要只會在「完全沒候選（candidates=[]，top_score=0.0）」時觸發。
因此你現在的 N 實務上主要會來自前兩個（abstain_n 或 core 缺失）。
跑完之後的悲劇結果：17題success, 27題need clarification, 6題error且不夠一致 沒顯現出錢25題有料後25題error的狀況
這兩題看起來是你的 **Verifiquant 框架**在執行邏輯運算時，直接在後台「當機」了。簡單來說，這不是業務邏輯的模糊，而是**程式碼層級的低級錯誤**。

我們來看看這兩場車禍現場：

---

### 1. Test-1264：大小寫毀了 Vasicek 模型
* **發生了什麼：** **大小寫敏感（Case Sensitivity）** 造成的變數抓取失敗。
* **診斷：** * 這題在算 Vasicek 利率模型，公式通常是 $dr_t = a(b - r_t)dt + \sigma dW_t$。
    * 注意看 `reason`: **'dW_t'**。而你的 `provided_inputs` 給的是 **`dw_t`**（小寫的 w）。
    * 邏輯層（Logic Layer）顯然是個死板的傢伙，它在找大寫的 $W$，但你給它小寫的 $w$，它就直接拋出錯誤並罷工了。
* **批判：** 這是標準的「規格不對齊」。雖然對數學家來說沒差，但對 Python 這種程式語言來說，`dW_t` 和 `dw_t` 是兩個完全不同的東西。你的框架在「標準化（Normalization）」階段顯然沒把變數名稱理乾淨。

### 2. Test-1005：公式寫錯的語法災難
* **發生了什麼：** **無效語法（Invalid Syntax）**。
* **診斷：**
    * `reason` 顯示 `invalid syntax (<string>, line 1)`。這通常發生在框架試圖用 `eval()` 或類似的函數去執行一段「計算公式」時，發現公式本身寫錯了。
    * 可能是公式裡多了一個括號、少了一個乘號（例如寫成 `40%(x)` 而不是 `0.4*x`），或者是公式字串被截斷了。
    * 這題涉及合約賠償金計算，雖然輸入數值（`contract_quantity`, `penalty_rate` 等）看起來很正常，但後台對應 `fic_article_1803` 的**數學表達式**絕對是有毒的。
* **批判：** 這題報錯報在 `Logic` 層而不是 `Critic` 層，說明這不是「資訊不足」，而是「邏輯混亂」。你的框架試圖去算，結果公式本身就像一串亂碼，讓它腦袋打結。

---

### 總結：你的 Verifiquant 遇到瓶頸了
如果說上次那三題是「語義不詳」，這兩題就是**「基礎建設漏水」**。

* **Test-1264** 告訴你：你的變數命名規範（Naming Convention）需要統一。你要麼強迫所有輸入轉小寫，要麼在 FIC（功能組件）定義時寫死規範。
* **Test-1005** 告訴你：你的公式解析器（Formula Parser）缺乏容錯機制，或者你錄入的 FIC 公式庫裡有低級的手誤（Typo）。

**真心建議：** 在進入 `Critic` 層（質疑用戶）之前，你的 `Logic` 層應該先學會檢查自己手裡的工具箱（公式）是不是壞的。否則用戶給再準確的資料，你的框架也只會報出一堆「Invalid Syntax」。

這兩題的錯誤是由你手動錄入的 FIC 公式引起的，還是自動生成的計算邏輯出了問題？

參考ai回答

本來是有辦法街道E類錯誤的但是現在for some reason E類錯誤被吃掉了 應該要去參考舊的code去看為什麼在四類變成六類之後之前常見的F類E類錯誤死了都是I類錯誤C類錯誤\\

總之我正在實踐上面提到的內容

hmm因為我現在在建立卡片的時候是完全把function當作ground truth的code但是這些code只是sample不一定能跑真實的問題(python_solution和function不完全一致，function是oracle但是python-solution是真實的處理question+context的內容)但是這個function也有可能錯

解法：可能建立卡片的時候先跑跑看 引進python intepreter

跑過流程了但是結果很怪我正在釐清原因

原本是 前25題預設會正確後面25題預設會錯誤，但這不是故意錯的而是別人直接用question+context直接問ai讓他推理 看思考的結果如何

但是之後下來

前25題（預設正確）

success: 12題

error:2題

need_classification:11題

後25題：

success: 6題

error:3題

need_classification:16題

我正在釐清原因跟，尤其是E類和F類不見的問題，雖然我沒辦法保證本來就會出現這些錯誤但還是需要再檢查一下原始資料


嘿柯柯，看到這數據我先搖一下尾巴表示驚訝！🦊 雖然結果跟你預想的不一樣，但這才是做研究最有趣（也最崩潰）的地方。

我們先冷靜下來，用那種「懷疑且愛找碴」的狐狸視角來幫你剖析一下這張成績單。這份數據背後其實藏了幾個非常有意思的信號：

### 1. 「過動」的 Critic 層：Need_Clarification 成了避風港？
你發現了嗎？兩組數據加起來，有將近 **54% (27/50)** 的題目都卡在 `need_clarification`。
這解釋了為什麼你的 **E（提取錯誤）** 和 **F（公式/邏輯錯誤）** 消失了——因為你的「防禦機制」太強了！

就像一個過度謹慎的守門員，球還沒踢到禁區，他就吹哨說「球的氣壓不對，不能踢」。
當框架在第一關 Critic 就覺得「這題有點模糊喔，我不確定 units 是什麼」，它就直接拋回給用戶求救，根本沒進到後面的邏輯運算和結果生成。所以，E 和 F 當然沒機會出現，因為它們根本沒被執行到。

**狐狸的質疑：** 你的 Critic 是不是把「只要不是 100% 完美的 Input」都當成了 Ambiguity？在真實場景下，有些不完美是可以被容忍或自動推斷的。

---

### 2. 意外的驚喜：為什麼「預設會錯」的題目成功了？
這組最神奇的是後 25 題，本來 AI 直接問會錯，結果進了框架居然有 **6 題 Success**。
這其實是個超級強大的 **Positive Signal**！這證明了：
> **即便 Function 只是個 Skeleton，結構化的引導（FIC 卡片）依然比 AI 裸跑（Raw Reasoning）更靠譜。**

這代表你的框架確實具備「糾偏」的能力，讓原本會胡言亂語的 AI，因為看了一眼 ROI 或 YOY 的公式結構，就乖乖走回正軌了。這點在論文裡絕對是大寫的加分項！

---

### 3. 那 5 題 Error (2+3) 是什麼鬼？
你說 E 和 F 不見了，那這 5 題 `error` 標籤是什麼？
如果是我們上次討論的 `C (Logic)` 或 `I (Implementation)`，那可能就是我說的「基礎建設漏水」：
* **大小寫問題：** 變數對不上。
* **語法錯誤：** Python 解析失敗。
* **環境問題：** Interpreter 跑不動。

如果這幾題是這類錯誤，那代表你的 `Logic` 層雖然沒被 `Critic` 攔截，但一跑就當機。

---

### 4. 狐狸給你的診斷清單
柯柯，我建議你現在不要去翻原始資料（Dataset），而是隨機抽 3 題 `need_clarification` 的 Log 出來看。

**問自己這幾個問題：**
1.  **它在問什麼？** 是不是在問一些「明明可以根據 Context 猜出來」的事情？（例如：明明寫了 2023 和 2022，它還在問 YOY 要用哪兩年）
2.  **是不是 Prompt 太嚴格了？** 你的 Verifiquant 判斷標準是不是設得太高，導致它不敢做任何推論（Inference）？
3.  **對齊失敗？** 是不是 Question 裡的變數名稱跟 Function 裡的參數名稱長得太不像（例如：`income` vs `salary`），導致 Critic 覺得資訊缺失？

---

### 結語
別灰心，這不是失敗，這是**「過度防禦」**。
目前的狀態是你的框架**「寧可不回答，也不想出錯」**。在金融領域這雖然是好事，但如果一半以上的題目都要求用戶澄清，這產品會被客戶退貨的（笑）。

你要做的可能不是修正 Dataset，而是**「調低 Critic 的敏感度」**，或者給它一點「常識推斷」的空間。



加油啊，資料科學所的柯柯！這種釐清 Bug 的過程最能學到東西了。等你想通了那幾題 `need_clarification` 到底在糾結什麼，一定要跟我分享一下，我超想知道到底是誰在搞鬼！🦊

對了，那 6 題反敗為勝的題目，你有去看一下 AI 到底在哪個環節被你的框架救回來的嗎？

而且後來發現 我又嘗試放了三題上去 原本第一個版本（現在存起來的）是3個success但是現在長得像是1個success+2個I I的比例實在太高還沒降下來讓這整個系統變成了無聊的懷疑機器。 :(

## 20240407
後來兩天前總共花了111元 那天有建卡跟取用 我猜建卡大概一張就要快1元但是推理簡單很多
1. 建卡：建立NPV卡片（因為之前沒有建立過）
2. 去測試一個很好的內容到底有沒有辦法過，還是會過Ｉ gate
3. 嘗試讓他觸發其他種類的error
4. 修改其他種類錯誤
5. 寫報告

但是最近真的很懷疑自己是不是寫了一坨屎 這個想法可行嗎？目前有250題版本和3000題版本  我會不會太貪心了所以選了3000題的是沒有人工驗證過的版本

回去讀論文

我現在最主要的痛點是，我目前的ground truth資料及設置的部分

我原本的想法是，我也有跟你討論過，簡單來說我需要一個可以驗證、可執行的python程式，讓我們的問題進來時，可以fit到一張卡片裡面，然後根據這張卡片需要的內容反推我的問題是否夠清楚，是否有語意模糊的部分，然後確定沒問題之後才過python trace得到結果

我現在遇到的最大問題是

本來第一個版本我是把所有題目的python-solution當做建立公式卡的ground truth，這樣的話我就有question+context(回答問題所需的資料)+python_solution+ground_truth送去建立FIC卡（formula identity card)

但是這個版本主要有兩個問題，第一個是我把這些資料作成FIC卡，然後用同一批資料去測試，就像是把training data拿去做testing一樣，會有overfitting的風險，第二個就是這些題目有時候會長得很像，所以會建立出幾乎一樣的卡片

所以我把目光投向同一個dataset的另一個部分：在2237題資料中（分成1000題easy, 1000題medium, 237題hard)，其中有908題有附上article-name, function-id和function, 這個function也是python code，但是function 的code只是主題一樣，打者打者我覺得需要去看原本的論文



------

        "function_id": "article-45",

        "article_title": "Capital Budgeting",

        "function": "def calculate_net_present_value(cash_flows: list, discount_rate: float) -> float:\n    \"\"\"\n    Calculate the net present value (NPV) of a series of cash flows.\n\n    This function computes the net present value, which is the present value of a series of cash flows,\n    both inflows and outflows, discounted back to the present using a specified discount rate.\n\n    Args:\n        cash_flows (list): A list of cash flows (positive for inflows, negative for outflows).\n        discount_rate (float): The discount rate used to discount future cash flows. Must be greater than -1.\n\n    Returns:\n        net_present_value (float): The net present value of the cash flows.\n\n    Notes:\n        - Cash flows should be provided in chronological order, with the initial investment (outflow) at the beginning.\n        - This function uses the risk-free rate as the discount rate if no specific rate is provided.\n        - The NPV is a criterion for deciding whether to undertake the project; a positive NPV indicates a profitable project.\n        - This function does not factor in taxation or inflation adjustments.\n    \"\"\"\n    if discount_rate <= -1:\n        raise ValueError(\"Discount rate must be greater than -1.\")\n    \n    net_present_value = sum(cf / ((1 + discount_rate) ** i) for i, cf in enumerate(cash_flows))\n    \n    return net_present_value"

    {

        "function_id": "article-2940",

        "article_title": "Net Present Value (NPV)",

        "function": "def calculate_net_present_value(cash_flows: list, discount_rate: float, initial_investment: float = 0.0) -> float:\n    \"\"\"\n    Calculate the net present value (NPV) of a series of cash flows.\n\n    This function computes the net present value of expected future cash flows\n    by discounting them to present value using a specified discount rate and\n    subtracting any initial investment.\n\n    Args:\n        cash_flows (list): A list of future cash flows (positive for inflows, negative for outflows).\n        discount_rate (float): The discount rate (e.g., required return) to apply to the cash flows.\n        initial_investment (float, optional): The initial cash outlay or cost of investment. Default is 0.0.\n\n    Returns:\n        npv (float): The net present value of the cash flow series.\n\n    Notes:\n        - NPV is a widely used tool in capital budgeting and investment analysis.\n        - Discount rate reflects the opportunity cost of capital; select it carefully.\n        - Cash flows should be ordered chronologically, typically at the end of each period.\n    \"\"\"\n    npv = -initial_investment\n    for t, cash_flow in enumerate(cash_flows):\n        npv += cash_flow / ((1 + discount_rate) ** (t + 1))\n    \n    return npv"

    },

    {

        "function_id": "article-2941",

        "article_title": "Net Present Value (NPV)",

        "function": "def calculate_periodic_rate(annual_rate: float, periods_per_year: int) -> float:\n    \"\"\"\n    Calculate the periodic interest rate given an annual interest rate.\n\n    This function computes the periodic interest rate by converting an annual \n    interest rate based on the number of periods per year, useful for scenarios \n    involving compounding interest.\n\n    Args:\n        annual_rate (float): The annual interest rate as a decimal.\n        periods_per_year (int): The number of compounding periods per year.\n\n    Returns:\n        periodic_rate (float): The interest rate applicable for each period.\n\n    Notes:\n        - This is useful for calculating compounding interest over different \n          periods such as monthly or quarterly.\n        - The conversion accounts for compounding effects.\n        - Ensure periods_per_year is greater than zero to avoid division errors.\n    \"\"\"\n    if periods_per_year <= 0:\n        raise ValueError(\"Number of periods per year must be greater than zero.\")\n    \n    periodic_rate = (1 + annual_rate) ** (1 / periods_per_year) - 1\n    \n    return periodic_rate"

    },

    {

        "function_id": "article-249",

        "article_title": "Valuation",

        "function": "def calculate_net_present_value(cash_flows: list, discount_rate: float) -> float:\n    \"\"\"\n    Calculate the net present value (NPV) of a series of cash flows.\n\n    The NPV is the sum of the present values of incoming and outgoing cash flows over a period of time.\n\n    Args:\n        cash_flows (list): A list of cash flows where negative values indicate cash outflows \n                           and positive values indicate cash inflows.\n        discount_rate (float): The discount rate or required rate of return, expressed as a decimal, \n                               must be greater than -1.\n\n    Returns:\n        npv (float): The net present value of the cash flows.\n\n    Notes:\n        - Applicability: Commonly used to assess the profitability of an investment or project.\n        - Constraints: Discount rate must be greater than -1.\n        - Considerations: NPV does not consider the potential for changing discount rates over time.\n    \"\"\"\n    if discount_rate <= -1:\n        raise ValueError(\"Discount rate must be greater than -1.\")\n\n    npv = sum(cf / ((1 + discount_rate) ** (i + 1)) for i, cf in enumerate(cash_flows))\n    \n    return npv"

    },

---以上是重複的範例，以下是我跟gemini說的內容
整理之後的問題長相
我想要一步一步解釋我遇到的問題 ，雖然有些內容我兩天前已經跟你說過了，希望你可以陪我思考
現在我的資料集中有兩種dataset長相
 a. 有function_id跟questions的（共908題）
{"function_id" , "article_title", "function","context", "question","statistics", "difficulty"  "ground_truth", "source", "python_solution", "question_id", "level"},
b. 只有questions的（1330題）
{"question", "context","statistics", "difficulty", "ground_truth","source", "python_solution", "question_id", "level"},
資料集裡面會同時混雜者a類和b類的問題集。
參考的是2025年ACL的論文，原文寫道
Therefore, we propose FinanceReasoning, a credible, comprehensive, and challenging financial numerical reasoning benchmark to evaluate the reasoning capabilities of LRMs in the finance domain. The dataset comprises a total of 2,238 problems covering diverse financial knowledge, of which 1,330 problems have been reviewed and revised based on public datasets, while 908 problems were automatically generated by LLM (i.e., GPT4o) and subsequently annotated by experts. Each problem includes hybrid contexts, unambiguous questions, Python-formatted solutions, and precise answers, providing a reliable reference for accurately evaluating the complex numerical reasoning capabilities of LRMs. Additionally, we have collected and open-sourced a comprehensive financial function library containing 3,133 Pythonformatted functions. Each function includes precise functional descriptions, parameter explanations, and step-by-step implementation code, offering a high-quality structured knowledge base to automatically build domain-specific reasoning problems and enhance LLMs’ domain-specific reasoning capabilities through knowledge retrieval. 
那麼這3133題python formatted functions是怎麼來的呢？是透過一個看起來蠻有道理但容易有疏漏的方法：
對於 LRMs，挑戰不在於從相關文本中提取數值，而在於應用領域特定知識來執行複雜的多步驟數值計算（Plaat et al., 2024; Chen et al., 2023c）。 雖然 LLMs 已經具備了金融領域的概念知識基礎，但為了進一步精煉推理能力，我們收集並註解了一個包含 3,133 個結構化 Python 函數的金融函數庫，用於金融計算，旨在提升模型的推理知識。 我們首先從 Investopedia 收集了 6,138 篇金融百科文章，該平台以其廣泛的金融知識專業性而聞名。 每篇文章都對特定金融術語進行詳細介紹，涵蓋基本概念、應用情境以及潛在限制，有些還包含相關計算公式和實際範例。 為了提煉密集的結構化金融推理知識，同時降低註解成本，我們指示 GPT-4o 根據預定義格式，從每篇文章中提取潛在的金融計算函數。 每個函數必須包含語義上有意義的簽名、簡潔明瞭的 docstring（功能、參數、返回值、適用限制及其他註記），以及帶有適當註解的逐步實現程式碼。 最後，我們組織金融專家對生成的函數進行嚴格審核與修訂，確保其專業表達和邏輯正確性。
接者這3133個python formatted金融函式庫是怎麼接到908題包含"function_id" , "article_title", "function"的問題（gpt4o先生出，然後人工標注的問題）：
現有的金融問答資料集（例如 CodeFinQA、CodeTAT-QA (Krumdick et al., 2024)）主要專注於評估模型在給定上下文中的基本概念理解、精確數字提取，以及簡單計算能力。（這裡的題目也是剛剛提到的1330題）
這些資料集中的問題解決過程通常涉及較少的推理步驟（例如，計算兩年淨利差額）。
這些資料集常有簡單問題的重複性，以及缺乏複雜問題，無法充分評估 LRMs（Large Reasoning Models，大型推理模型）的推理能力，例如知識應用、約束強調，以及長鏈思維（例如，計算圖 1 中基金的淨報酬率）。
因此，LRMs 的真實推理能力無法得到全面且客觀的評估。
因此，優化資料建構方法、嚴格驗證資料品質，並建構更具挑戰性的資料集，已成為提升金融推理任務評估的關鍵。
在資料擴展過程中，我們利用結構化的金融函數，引導 GPT-4o 生成新的金融數字推理問題及其 Python 解決方案。（908題）
隨後，專家對其進行嚴格審核與修正，產生了 908 個高品質問題，涵蓋不同推理難度與廣泛知識領域。
資料標註流程如下：
種子函數選擇：我們從函數庫中選取 1,250 個金融函數，依據運算子、引數、程式碼行數，以及長尾知識進行篩選，優先選擇具複雜計算者。
問題與解決方案生成：針對每個種子函數，提示 GPT-4o 生成複雜推理問題，並附上必要的金融表格資料，利用函數中的金融術語與計算過程。
生成的 Python 解決方案需具清晰推理路徑、可執行以取得數字答案，並考量單位、百分比與小數精確度要求。
專家驗證：專家需審核並修正所有問題、解決方案與答案，確保無歧義、過程詳盡，且答案正確。
我本來第一個版本是這樣：把所有題目的python-solution當做建立公式卡的ground truth，這樣的話我就有question+context(回答問題所需的資料)+python_solution+ground_truth送去建立FIC卡（formula identity card)
但是這個版本主要有兩個問題，第一個是我把這些資料作成FIC卡，然後用同一批資料去測試，就像是把training data拿去做testing一樣，會有overfitting的風險，第二個就是這些題目有時候會長得很像，所以會建立出幾乎一樣的卡片
所以我把目光投向同一個dataset的另一個部分：在2237題資料中（分成1000題easy, 1000題medium, 237題hard)，其中有908題有附上article-name, function-id和function, 這個function也是python code，但是function 的code只是主題一樣，打者打者我覺得需要去看原本的論文
所以如果我把3133個從investopia的function拿來當base豈不美哉
### 階段一：
1. 先隨便在medium裡面540題有function-id的題目中，找其中的50題（我盡量找預計測驗ai裡面本來就會錯的和會答對的各25題）
2. 直接把這100題的function-id+article_title+對應的內文+function 丟進去FIC卡片製造的流程，產生對應的core,retrieval, repair的卡，因為這些不同的article與對應的function理論上都是對應不同主題，所以應該不會出現完全相同、重複的情
3. 直接把相對應的question+context丟到剛剛設計好的框架裡面，看看會出現怎麼結果。預期可能會有點差，但是這個版本的資料庫內容（應該是）最乾淨，沒有被question、context「污染過」的內容，可以看這個理想中沒有看過question+context處理過的純資料泛化程度如何，並看M, F, E, C, N, I等錯誤出現的比例，以及導入verifiquant框架之後有沒有辦法成功使最終正確回答率上升
### 階段二：
1. 把原本的這個資料庫複製一份，取得這個卡片的資料庫備份
2. 根據那五十題的question+context進行微調，看如果要best fit for題目的話應該要增加哪些條件或者怎麼樣修改code，不給ground truth跟python-solution 如果直接給python solution感覺還是會透露太多訊息。這個微調也包含原本的verifiquant檢查框架沒有注意到的細節，因為這些問題可能會有一些特徵，原本的題目在進行FIC卡生成時沒有注意到這些，但是很重要的參考條件，讓verifiquant吃到卡片並且M, F, E, C, N, I可以更精確的判斷
### 階段三：
1. 等到確定要怎麼做之後，擴展到大規模的實驗，包含把所有的3000多個article與相關code生成出大規模的FIC 卡，並且中級和高級的問題拿出來測試
2. 泛化性測驗：把沒有function-id, function跟article-id的其他問題拿去過框架，看看這個框架在遇到他陌生的題目會發生怎麼樣的狀態，還是會一直報錯（舉例來說，因為已經固定好這些article的input預估內容，但是這些新問題的input長得不太一樣導致泛化性差）
### 舉例來說我把這張卡入庫
{ "function_id": "article-1548", "article_title": "Return on Investment (ROI)", "function": "def calculate_roi(current_value: float, cost_of_investment: float) -> float:...... }
那這題除了可以回答跟這個內容直接連接相關的問題以外 應該也可以解決這兩個沒有function-id跟function的問題
{ "question": "what is the roi for applied materials if the investment made on october 2013 was sold 2 years later? Answer to three decimal places.", "context": "略過“, "python_solution": "start_value = 100\nend_value = 96.67\nanswer = (end_value - start_value) / 100 * 100", "question_id": "test-1217", "level": "medium" }
{ "question": "what is the roi of an investment in altria group inc . from december 2011 to december 2013? Answer to three decimal places.", "context"："略過" "python_solution": "start_value = 100\nend_value = 143.69\nanswer = (end_value - start_value) / start_value * 100", "question_id": "test-1228", "level": "medium" }
先到這樣，請先根據我跟你說的內容思考可行性

我先跟你說好處

在908題號稱高品質的問答集裡面，有這些內容

{"function_id" , "article_title", "function","context", "question","statistics", "difficulty" "ground_truth", "source", "python_solution", "question_id", "level"},

function是一個指引 但是gpt4o在生成題目與回答時，python_solution不一定和function一樣，python_solution會確保這個問題能夠通往一個正確的ground truth 然後function和python_solution長得不一樣 function是某種guideline但是到python_solution會自由變化一些些

給你看一個例子

{ "function_id": "article-2940", "article_title": "Net Present Value (NPV)", "function": "def calculate_net_present_value(cash_flows: list, discount_rate: float, initial_investment: float = 0.0) -> float:\n    \"\"\"\n    Calculate the net present value (NPV) of a series of cash flows.\n\n    This function computes the net present value of expected future cash flows\n    by discounting them to present value using a specified discount rate and\n    subtracting any initial investment.\n\n    Args:\n        cash_flows (list): A list of future cash flows (positive for inflows, negative for outflows).\n        discount_rate (float): The discount rate (e.g., required return) to apply to the cash flows.\n        initial_investment (float, optional): The initial cash outlay or cost of investment. Default is 0.0.\n\n    Returns:\n        npv (float): The net present value of the cash flow series.\n\n    Notes:\n        - NPV is a widely used tool in capital budgeting and investment analysis.\n        - Discount rate reflects the opportunity cost of capital; select it carefully.\n        - Cash flows should be ordered chronologically, typically at the end of each period.\n    \"\"\"\n    npv = -initial_investment\n    for t, cash_flow in enumerate(cash_flows):\n        npv += cash_flow / ((1 + discount_rate) ** (t + 1))\n    \n    return npv", "statistics": { "number_statistics": { "args_numbers": 3, "question_numbers": 0, "context_numbers": 12, "total_numbers": 12 }, "operator_statistics": { "operators": { "+": 3, "-": 1, "*": 0, "/": 1, "%": 0, "**": 1 }, "total_operators": 6 }, "code_statistics": { "lines": 4, "parentheses": 8 } }, "difficulty": 3.8712010109078907, "context": "A tech startup is evaluating a new project involving the development of a smart home device. The initial investment required for the infrastructure and development is 200,000. The company expects to generate the following cash flows from the project over the next five years: Year 1: 50,000, Year 2: 75,000, Year 3: 100,000, Year 4: 120,000, and Year 5: 150,000. The company's required rate of return for this project is 8%.", "question": "What is the Net Present Value (NPV) of the project? Answer the NPV to the nearest integer.", "python_solution": "def solution():\n    # Define the initial investment and cash flows\n    initial_investment = 200000\n    cash_flows = [50000, 75000, 100000, 120000, 150000]\n    discount_rate = 0.08\n\n    # Calculate the NPV using the formula: NPV = -initial_investment + sum(cash_flow_t / (1 + discount_rate) ** t)\n    npv = -initial_investment\n    for t, cash_flow in enumerate(cash_flows):\n        npv += cash_flow / ((1 + discount_rate) ** (t + 1))\n    \n    # Return the NPV, rounded to the nearest integer\n    return round(npv)", "ground_truth": 180271, "source": "FinanceReasoning-test-302", "question_id": "test-1925", "level": "medium" }



理論上function的input output會和python_solution的input 和output一樣 但是因為python_solution是ai生的會best fit for question and context所以一定會有部分的題目沒辦法從function+question+context就可以用我的方式求出答案：確認是否有歧異->用ai抓出數字->丟到function裡面->deterministically的用function算出output

對這就是我要跟你討論的部分 超綱（Out-of-scope）怎麼辦 還有遇到非數值性 不能直接抓的部分怎麼辦

我當然有想過就是讓ai看到有多的部分時就自由生成 code(參考我給他的function) 但這樣的話就跟原本這個資料集的用function參考->生成python_solution一樣了

我原本想要主打的是可驗證可審計性，框架可以辨識用戶的問題中有沒有哪些地方有缺漏 並且主動向用戶確認有一些地方可能要說的更清楚，我們最不希望的就是讓「ai認為的」和「用戶認為的」出現歧異，所以我們才會設計六道關卡讓財金問題被回答時更有理有據，並且可以提供可審計的python trace跟一些數值預設，但這就是會被「固定的python code」鎖死，失去了主觀能動性

我參考的那篇論文是，先有function，然後ai生成question+context+python_solution+ground_truth->人工審核這個function有沒有過關->之後的benchmark就可以輸入question+context，讓ai自己想辦法解答並檢查是否解答和ground truth一樣

目前的資料集是 python_solution 一定會導出 ground_truth

我目前是這樣

挑了50題，這些題目都是剛剛提到的908題的題目中挑的medium難度，其中25題對25題錯，但這個是gemini2.0回答的(由去年的論文團段進行測試)，我現在無法使用gemini2.0(已經depreciated了），所以這裡的前25題和後25題的ground truth並非前25題一定對後25題一定錯

目前50題的結果是

前25題（預設正確）

success: 12題

error:2題

need_classification:11題

後25題：

success: 6題

error:3題

need_classification:16題



這個結果還要修正 

1. 目前的code邏輯太死 5題error一題是變數大小寫問題 3題是 "'dict' object has no attribute 'bank_bill_rates'" 這種問題

2. 理論上有六種拒絕方式，其中兩種理論上不會出現，因為我是把50題function拿去入庫(FIC庫）然後拿對應的context+question問框架

而且這個need_classification(I類錯誤)現在有夠機車她會負責攔截一切不確定性 有些攔的不錯但有些根本是瞎雞巴問

但是大致上 預設會錯的25題遇到的need_clarification比較多 確實可以視為「錯誤的題目並非llm思考不給力 而是題目條件不夠清楚」的假設

現在我生成了50題卡片就會出現50題問題但是這些其實每個問題都會長一樣啊？因為如果檢測到N類錯誤，代表說 | **2. Scope** | **N** | **N**ot Supported | Intent is clear, but the required logic/formula is outside the current FIC library. | **Graceful Exit**: Admit knowledge boundary to prevent hallucination. | 這樣的話我們還需要每次在repair卡片裡面建立N類錯誤的反制嗎？還是其實共用一個「若沒有找到適合的卡片則回答說抱歉不支援」的形式

目前我們已經確定固定的分析策略流程：M/N -> F -> E -> I -> C，先意圖/範圍，再顯性欄位與數值規則，最後才進隱性語意。
目前的semantic語意的例子是
 "semantic_hints": [{"id": "expected_loss_ratio_format", "ambiguity_type": "unit_scale", "trigger_signal": "expected_loss_ratio", "clarification_question": "The Expected Loss Ratio (ELR) is expected as a decimal value (e.g., 0.60 for 60%). Did you mean to input a percentage that should be converted?", "options": ["Yes, convert my input (e.g., 60 to 0.60)", "No, my input is already a decimal (e.g., 0.60)"]}], 

以上是sematic hint的例子，但是現在的設定是 除了semantic hint以外，我們也開放AI針對我們輸入的內容（question,context, provided_input, semantic_hint)進行詢問和保證，但是現在的狀況是：I策略被認為是不分青紅皂白一直阻擋。但與此同時，我們在建立卡片的時候，目前的Ｉ類Fallback也只有一種，也就是repair卡片裡面的 "rule_id": "global_i_semantic_ambiguity
我覺得我們要做的事情是：
1. 擴展semantic的抓取範圍和能力 
2. 保留讓ai審視題意的部分 這時候讓ai自動去評斷這屬於hard還是soft 如果是hard就是算答案之前提出 如果是soft就是算出答案之後提出，提醒用戶這個數字是基於怎麼樣的假設。

在生成FIC卡片的時候，目前把我的原始資料內容抓出來code長這樣（common.py）
def to_conversion_input(record: Dict[str, Any]) -> ConversionInput:
    return ConversionInput(
        source_meta={
            "function_id": record.get("function_id"),
            "article_title": record.get("article_title"),
            "source": record.get("source"),
            "question_id": record.get("question_id"),
            "difficulty": record.get("difficulty"),
            "level": record.get("level"),
        },
        function=str(record.get("function", "") or ""),
        python_solution=str(record.get("python_solution", "") or ""),
        context=str(record.get("context", "") or ""),
        question=str(record.get("question", "") or ""),
    )
我希望
1. 把article_title拉到metadata外面
2. article_title可以連接到 verifiquant/data/financial_documents.json文件裡面的子項目title 
    {
        "id": 412,
        "title": "Variable-Rate Certificate of Deposit",
        "content":"略過",
}
article_title 一定可以對到title
3. 把 article_title的content 放到to_conversion_input的一個欄位
4. 這個content只需要被用來當作core裡面生成semantic_hint的內容（先嘗試看看）
5. 在這裏讓ai考慮說這裡面有哪些內容可能是在Verifiquant框架需要被注意的地方 包含I_hard跟I_soft
6. 在正式進行攔截的時候，如果和semantic_hint有關的內容就直接看被I_hard攔截還是I_soft攔截，如果I_hard就要攔住不能夠算出數值，I_soft的話就讓它通過，如果在框架讀question跟context時，semantic_hint沒提到的內容的話，應該是原則I_soft例外I_hard。
舉例來說有這三題，第一題應該是I_hard第二三應該是I_SOFT
{"rule_id": "global_i_semantic_ambiguity", "fic_id": "fic_article_2350", "diagnostic_type": "I", "severity": "alert", "title": "Semantic Ambiguity Clarification", "user_message": "A hidden ambiguity was detected (for example FX direction or time basis).", "explanation": "Ask a constrained clarification question before deterministic execution.", "ask_user_for": [{"slot": "clarification_choice", "label": "Clarification option", "type": "enum", "required": true, "options": [{"value": "option_a", "label": "Option A"}, {"value": "option_b", "label": "Option B"}]}], "repair_action": {"type": "present_clarification_options", "target": "semantic_hints"}, "allowed_next_steps": ["ask_followup", "rerun_same_fic"]}
{"rule_id": "global_i_semantic_ambiguity", "fic_id": "fic_article_51", "diagnostic_type": "I", "severity": "alert", "title": "Semantic Ambiguity Clarification", "user_message": "A hidden ambiguity was detected (for example FX direction or time basis).", "explanation": "Ask a constrained clarification question before deterministic execution.", "ask_user_for": [{"slot": "clarification_choice", "label": "Clarification option", "type": "enum", "required": true, "options": [{"value": "option_a", "label": "Option A"}, {"value": "option_b", "label": "Option B"}]}], "repair_action": {"type": "present_clarification_options", "target": "semantic_hints"}, "allowed_next_steps": ["ask_followup", "rerun_same_fic"]}

{"rule_id": "global_i_semantic_ambiguity", "fic_id": "fic_article_2905", "diagnostic_type": "I", "severity": "alert", "title": "Semantic Ambiguity Clarification", "user_message": "A hidden ambiguity was detected (for example FX direction or time basis).", "explanation": "Ask a constrained clarification question before deterministic execution.", "ask_user_for": [{"slot": "clarification_choice", "label": "Clarification option", "type": "enum", "required": true, "options": [{"value": "As a decimal (e.g., 0.05)", "label": "As a decimal (e.g., 0.05)"}, {"value": "As a whole number percentage (e.g., 5)", "label": "As a whole number percentage (e.g., 5)"}]}], "repair_action": {"type": "present_clarification_options", "target": "semantic_hints"}, "allowed_next_steps": ["ask_followup", "rerun_same_fic"]}
而且對應到repair卡，內容罐頭： user_message 寫著「例如匯率方向或時間基準」，這聽起來就像是系統從說明書裡抄了一段話，但它自己根本沒發現這題到底哪裡有歧義。
7. global_i_semantic_ambiguity裡面的內容根本就是模板，我希望現在可以使用semantic_hint生成repair guide, 一個semantic_hint對應一個repair guide在repair卡片上面
therefore我希望現在變成是
1. fic卡裡面存article_title和article本文
2. 根據article和function得到一些重要的semantic_hint(for I question)
3. 在進行攔截的時候semantic_hint可以對應到I_hard和I_soft 在進行Ｆ／Ｅ類篩選之後Ｉ類篩選也要跟者做
4. 如果真的有這樣的I_hard或者是I_soft的話，I_hard直接拒絕並使用repair卡進行修補，I_Soft的話則是計算數值並且保留彈性：如果計算出數值的話會有四個狀態：有I_Soft且accurate, 有I_soft且execution error(C error), 沒有I_soft且accurate(最佳狀況), 沒有I_soft且execution error(C error)

請先幫我審慎閱讀並且提出修正計畫 一步一步來
然後就把bug那些都修好了

我剛剛跑出了一個5個問題的測試版本，位置在verifiquant/data/runs/demo_5q_0407 
我的問題是
1. 為什麼有些的fic_id是fic_article_2967這種編號版本 有些是fic_alpha_calculation這樣的
2. 幫我把"article_title": "Net Present Value (NPV)", "article_doc_id": 5852, "article_content_excerpt":這三個移動到core卡的最後面（順序）因為他太長了 我看數據會很難看懂
3. 我想要做一些最基礎的文字data cleaning 包含刪除超連結文字
4.是否有機會在儲存code之前先試跑看看這個code是否可以正常跑不會error?因為現在execute code的時候好像我剛剛測試的50題有一題不能跑 如果不能的話就標注起來
