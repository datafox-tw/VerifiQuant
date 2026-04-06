import json
import random

def extract_sampled_questions(file_path, n_zero=25, n_one=25, seed=42):
    """
    從 JSON 檔案中根據 acc 分類並隨機抽樣。
    
    :param file_path: JSON 檔案路徑
    :param n_zero: acc=0 要抽取的數量
    :param n_one: acc=1 要抽取的數量
    :param seed: 隨機種子碼
    """
    # 讀取資料
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        return "找不到檔案，請確認路徑是否正確喔！"
    except json.JSONDecodeError:
        return "JSON 格式似乎有誤，檢查一下是不是多了或少了逗號？"

    # 1. 分類與計算
    # 這裡假設 result 裡面一定有 acc，如果擔心缺失可以用 .get() 處理
# 增加 function_id 的檢查：確保該 key 存在，且內容不是 None 或空字串
    acc_0_list = [
        item for item in data 
        if item.get('result', {}).get('acc') == 0 
        and item.get('function_id') is not None  # 確保欄位存在且有值
    ]

    acc_1_list = [
        item for item in data 
        if item.get('result', {}).get('acc') == 1 
        and item.get('function_id') is not None
    ]
    print("--- 統計資訊 ---")
    print(f"Total acc=0 (錯誤): {len(acc_0_list)} 題")
    print(f"Total acc=1 (正確): {len(acc_1_list)} 題")
    print("----------------\n")

    # 2. 固定隨機碼
    random.seed(seed)

    # 3. 隨機抽樣 (加入保護機制，避免抽取數量大於總數)
    def safe_sample(population, k):
        if len(population) < k:
            print(f"警告：要求的數量 {k} 超過現有總數 {len(population)}，將回傳全部資料。")
            return population
        return random.sample(population, k)

    sampled_0 = safe_sample(acc_0_list, n_zero)
    sampled_1 = safe_sample(acc_1_list, n_one)

    # 4. 提取 question_id 列表
    q_ids_0 = [item['question_id'] for item in sampled_0]
    q_ids_1 = [item['question_id'] for item in sampled_1]

    return {
        "acc_0_ids": q_ids_0,
        "acc_1_ids": q_ids_1,
        "seed_used": seed
    }

# --- 使用範例 ---
# 假設你的檔案叫做 'data.json'
file_name = './verifiquant/data/gemini-2.0-evaluation-medium.json' 

# 你可以隨意調整這些參數
results = extract_sampled_questions(
    file_path=file_name, 
    n_zero=25, 
    n_one=25, 
    seed=12345
)

if isinstance(results, dict):
    print("【正確題目的 ID 列表 (25題)】:")
    print(results['acc_1_ids'])
    print("\n【錯誤題目的 ID 列表 (25題)】:")
    print(results['acc_0_ids'])
else:
    print(results)