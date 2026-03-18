python3 preprocessing/dataset_case_to_fic_v3.py \
  --input verifiquant_v3/data/original_first10Q.jsonl \
  --core-output verifiquant_v3/data/10Q_ver1_core.fic_v3.json \
  --retrieval-output verifiquant_v3/data/10Q_ver1_retrieval.fic_v3.json \
  --repair-output verifiquant_v3/data/10Q_ver1_repair.fic_v3.json 
  --max-records 1 \

生成三份檔案加上入庫
python3 preprocessing/dataset_case_to_fic_v3.py \
  --input verifiquant_v3/data/original_first10Q.jsonl \
  --core-output verifiquant_v3/data/10Q_ver1_core.fic_v3.json \
  --retrieval-output verifiquant_v3/data/10Q_ver1_retrieval.fic_v3.json \
  --repair-output verifiquant_v3/data/10Q_ver1_repair.fic_v3.json \
  --db-url sqlite:///verifiquant_v3/data/v3_cards.db

或者是只有入庫
python3 preprocessing/build_card_store_v3.py \
  --db-url sqlite:///verifiquant_v3/data/v3_cards.db \
  --core verifiquant_v3/data/10Q_ver1_core.fic_v3.json \
  --retrieval verifiquant_v3/data/10Q_ver1_retrieval.fic_v3.json \
  --repair verifiquant_v3/data/10Q_ver1_repair.fic_v3.json

進行測驗
python3 preprocessing/run_mfe_pipeline_v3.py \
  --input verifiquant_v3/data/original_first10Q.jsonl \
  --db-url sqlite:///verifiquant_v3/data/v3_cards.db \
  --output verifiquant_v3/data/10Q_ver1_results.fic_v3.jsonl
3/15 22元這次不知道多少

存semi版本的expand
python3 preprocessing/expand_cases_v3.py \
--input verifiquant_v3/data/original_first10Q.jsonl \
--output verifiquant_v3/data/original_first10Q.expanded_40.semi_llm.jsonl \
--mode semi-llm 

python3 preprocessing/run_mfe_pipeline_v3.py \
  --input verifiquant_v3/data/original_first10Q.expanded_40.semi_llm.jsonl \
  --db-url sqlite:///verifiquant_v3/data/v3_cards.db \
  --output verifiquant_v3/data/10Q_expand40_ver1_results.fic_v3.jsonl

  生成圖片