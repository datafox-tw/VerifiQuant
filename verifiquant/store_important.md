python3 preprocessing/dataset_case_to_fic.py \
  --input verifiquant/data/original_first10Q.jsonl \
  --core-output verifiquant/data/10Q_ver1_core.fic.json \
  --retrieval-output verifiquant/data/10Q_ver1_retrieval.fic.json \
  --repair-output verifiquant/data/10Q_ver1_repair.fic.json 
  --max-records 1 \

生成三份檔案加上入庫
python3 preprocessing/dataset_case_to_fic.py \
  --input verifiquant/data/original_first10Q.jsonl \
  --core-output verifiquant/data/10Q_ver1_core.fic.json \
  --retrieval-output verifiquant/data/10Q_ver1_retrieval.fic.json \
  --repair-output verifiquant/data/10Q_ver1_repair.fic.json \
  --db-url sqlite:///verifiquant/data/cards.db

或者是只有入庫
python3 preprocessing/build_card_store.py \
  --db-url sqlite:///verifiquant/data/cards.db \
  --core verifiquant/data/10Q_ver1_core.fic.json \
  --retrieval verifiquant/data/10Q_ver1_retrieval.fic.json \
  --repair verifiquant/data/10Q_ver1_repair.fic.json

進行測驗
python3 preprocessing/run_mfe_pipeline.py \
  --input verifiquant/data/original_first10Q.jsonl \
  --db-url sqlite:///verifiquant/data/cards.db \
  --output verifiquant/data/10Q_ver1_results.fic.jsonl
3/15 22元這次不知道多少

存semi版本的expand
python3 preprocessing/expand_cases.py \
--input verifiquant/data/original_first10Q.jsonl \
--output verifiquant/data/original_first10Q.expanded_40.semi_llm.jsonl \
--mode semi-llm 

python3 preprocessing/run_mfe_pipeline.py \
  --input verifiquant/data/original_first10Q.expanded_40.semi_llm.jsonl \
  --db-url sqlite:///verifiquant/data/cards.db \
  --output verifiquant/data/10Q_expand40_ver1_results.fic.jsonl

  生成圖片