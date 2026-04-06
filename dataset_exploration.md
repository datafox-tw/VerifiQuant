在解釋我想要幹嘛之前，我先說明一下我預想中是怎麼處理dataset的，跟我發現用這種方式處理dataset可能會遇到的問題，以及最後我認為可能更好的方法，這種方法可能不能保證我的論文數據結果漂亮，但更真實和合理。

## 1. 先秀出目前的資料集中兩種dataset長相
### a. 有function_id跟questions的
{"function_id" , "article_title", "function","context", "question","statistics", "difficulty"  "ground_truth", "source", "python_solution", "question_id", "level"},
        "result": {
            "acc": 1,
            "execution_rate": 1,
            "executed_result": "1583"
        }

### b. 只有questions的
{"question", "context","statistics", "difficulty", "ground_truth","source", "python_solution", "question_id", "level"},

資料集裡面會同時混雜者a類和b類的問題集。

function_id, article_title跟function會對應到question, context, python solution等內容，但不是100%對應，他的python_solution的邏輯是，根據question, context和我預先標注的function（已經有一個算法去找說跟這個問題最為對應的function_id），然後把這個function作為參考，生成python code，所以python_solution跟function雖然都是處理同一個問題，但是只有python_solution是best fit for question_context，而function雖然也可以跑，但是在語境下只是skeleton.
question是問題，context是如果需要回答問題的話需要的資料。
我也有一個檔案專門存這些資訊
    {
        "function_id": "article-0",
        "article_title": "Year-End Bonus",
        "function": "def calculate_year_end_bonus(salary: float, bonus_percentage: float) -> float （code）"
    },
    {
        "function_id": "article-1",
        "article_title": "Year-Over-Year (YOY)",
        "function": "def calculate_yoy_growth(current_value: float, previous_value: float) -> float （code）
    },
然後這些article就會對應到真正的article(用functions-article-all的article_title對上financial_documents的title)

## 2. 原本的作法跟可能會造成問題的地方
我原本擔心的點是，因為在easy, medium, hard的問答集裡面，首先不是每一題都有function, function-id, article-title，加上我打算讓這些資料集的每個問題都能夠有對應的card，所以如果裡面有些card沒有function只有python-solutions就會無法對應。
在easy dataset裡面
在medium dataset裡面
- 在easy dataset裡面，1000題資料集只有212題有對應function
- 在mdium中，1000題中有540題有對應的function
- 在hard中，238題中有126題有對應的function
但是，根據剛剛的實驗，有一些危險的地方：
1. 我原本是拿python-solutions+questions+context+ground_truth去建卡，但是我用同個資料集去建卡又選卡的話，感覺會太overfitting，就像是我的train和test資料是同一筆資料一樣
2. 太多重複一樣的卡會導致duplicate的問題 像是剛剛建立卡片的時候就遇到因為長得太像導致duplicated core跟retrieval卡，以及 repair rule has no matching core diagnostic_check等問題

## 3. 我想要改成的方式（至少三階段）
### 階段一：
1. 先隨便在medium裡面540題有function-id的題目中，找其中的50題（我盡量找預計測驗ai裡面本來就會錯的和會答對的各25題）
2. 直接把這100題的function-id+article_title+對應的內文+function 丟進去FIC卡片製造的流程，產生對應的core,retrieval, repair的卡，因為這些不同的article與對應的function理論上都是對應不同主題，所以應該不會出現完全相同、重複的情形
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



## 4. 這樣做的好處與風險
1. 如果是被問到說，我們這樣的code是不是太自導自演，用同樣的code template作為FIC card的輸入燃料，以及測試框架有效性的時候使用的測試資料集，會不會有自導自演的嫌疑？我用這個架構做了天然的ablation實驗
2. 我們可以用這個架構證明擴展性佳，應該可以handle住不同方面的疑問
3. 但是也有可能一開始的表現就很差...