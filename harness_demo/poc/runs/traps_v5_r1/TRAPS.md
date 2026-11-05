# harness 在 dev 陷阱題上（80 題，1 次執行）

FNAR = 在陷阱題上仍給出數字的比例（論文的主要指標，越低越好），評分直接呼叫論文的 `score_traps`。
兩欄都報：含 I_SOFT（論文規則）與不含 I_SOFT（66 題；harness 沒有「附警告照做」模式，I_SOFT 一輸出就算錯，拿掉後比較不受計分方式影響）。

| 系統 | FNAR 含 I_SOFT | FNAR 不含 I_SOFT | F | E | I | N | 動作正確率 |
|---|---|---|---|---|---|---|---|
| harness（1 次） | **27.5%** | 19.7% | 0% | 30% | 60% | 20% | 53.8% |
| VQ（論文設定 top-4、K=1，3 次） | **10.4%**（12.5%、10.0%、8.8%） | 8.1% | 10% | 8% | 22% | 2% | 75.4% |
| R-PAL prompted（論文，三次平均） | 27.5% | — | 0% | 30% | 80% | 0% | — |
| R-PAL plain（論文） | 48.8% | — | 10% | 90% | 95% | 0% | — |

## 逐類（traps_v5_r1）

| 運算子 | 家族 | 題數 | 輸出（對/錯） | 追問 | 拒答 | 診斷對 |
|---|---|---|---|---|---|---|
| E | E / magnitude | 4 | 1（0/1） | 2 | 1 | 1 |
| E | E / negativity | 8 | 2（2/0） | 5 | 1 | 5 |
| E | E / nonpositive_zero | 4 | 1（0/1） | 3 | 0 | 3 |
| E | E / rate_out_of_unit | 4 | 2（0/2） | 1 | 1 | 1 |
| F | F / balance | 5 | 0（0/0） | 5 | 0 | 5 |
| F | F / count | 5 | 0（0/0） | 5 | 0 | 3 |
| F | F / denominator | 2 | 0（0/0） | 2 | 0 | 2 |
| F | F / other | 5 | 0（0/0） | 4 | 1 | 4 |
| F | F / price | 3 | 0（0/0） | 3 | 0 | 3 |
| I | I_HARD / hard | 6 | 3（3/0） | 3 | 0 | 0 |
| I | I_SOFT / soft | 14 | 9（9/0） | 2 | 3 | 0 |
| N | N / exotic | 14 | 2（2/0） | 0 | 12 | 12 |
| N | N / near_miss | 6 | 2（2/0） | 0 | 4 | 4 |

## 給出數字的陷阱題（逐題）

| 題號 | 預期 | 停在 | 輸出 | 答案對照原題 | 說明 |
|---|---|---|---|---|---|
| E-test-1006 | E | — | 1.5e+07 | False | {'targeted_rule_id': 'excessive_single_component_time', 'expression': "inputs['processing_ |
| E-test-1022 | E | — | 504 | True | {'targeted_rule_id': 'input_itemized_deductions_non_negative', 'expression': "inputs['item |
| E-test-1026 | E | — | 0.06 | True | {'targeted_rule_id': 'interest_expense_non_negative', 'expression': "inputs['interest_expe |
| E-test-1034 | E | — | 0.25 | False | {'targeted_rule_id': 'roae_e_001', 'expression': "inputs['beginning_equity'] <= 0 or input |
| E-test-1080 | E | — | 6500 | False | {'targeted_rule_id': 'price_coherence_extreme_change', 'expression': "abs(inputs['price_cu |
| E-test-1648 | E | — | 17.0334 | False | {'targeted_rule_id': 'plausible_ap_change', 'expression': "inputs['beginning_ap'] > 0 and  |
| I-test-1668 | I_HARD | — | 300 | True | {'ambiguity_family': 'option_type', 'removed_phrase': 'put', 'rule_id': 'i_option_type_har |
| I-test-1952 | I_HARD | — | 300 | True | {'ambiguity_family': 'option_type', 'removed_phrase': 'call', 'rule_id': 'i_option_type_cl |
| I-test-2163 | I_HARD | — | 6250 | True | {'ambiguity_family': 'option_type', 'removed_phrase': 'call', 'rule_id': 'i_option_type_im |
| I-test-1007 | I_SOFT | — | 36.6 | True | {'ambiguity_family': 'time_unit', 'removed_phrase': 'in months', 'rule_id': 'i_output_unit |
| I-test-1008 | I_SOFT | — | 0.884615 | True | {'ambiguity_family': 'scale', 'removed_phrase': 'as a percentage', 'rule_id': 'i_output_sc |
| I-test-1026 | I_SOFT | — | 0.06 | True | {'ambiguity_family': 'scale', 'removed_phrase': 'as a percentage', 'rule_id': 'i_output_sc |
| I-test-1034 | I_SOFT | — | 0.142857 | True | {'ambiguity_family': 'scale', 'removed_phrase': 'as a percentage', 'rule_id': 'i_output_sc |
| I-test-1037 | I_SOFT | — | 100 | True | {'ambiguity_family': 'scale', 'removed_phrase': 'as a percentage', 'rule_id': 'i_output_sc |
| I-test-1057 | I_SOFT | — | 0.25 | True | {'ambiguity_family': 'scale', 'removed_phrase': 'as a percentage', 'rule_id': 'i_output_sc |
| I-test-1062 | I_SOFT | — | 25 | True | {'ambiguity_family': 'scale', 'removed_phrase': 'as a percentage', 'rule_id': 'i_output_sc |
| I-test-1064 | I_SOFT | — | 1.02041 | True | {'ambiguity_family': 'scale', 'removed_phrase': 'as a percentage', 'rule_id': 'i_output_sc |
| I-test-1067 | I_SOFT | — | 25 | True | {'ambiguity_family': 'scale', 'removed_phrase': 'as a percentage', 'rule_id': 'i_output_sc |
| N-test-1086 | N | — | 15 | True | {'base_fic_id': 'fic_article_1726', 'template': 'exotic_american_pde'} |
| N-test-1105 | N | — | 25 | True | {'base_fic_id': 'fic_article_2532', 'template': 'exotic_american_pde'} |
| N-test-1097 | N | — | 5.29141 | True | {'base_fic_id': 'fic_article_2093', 'template': 'near_miss_keyrate'} |
| N-test-1100 | N | — | 24 | True | {'base_fic_id': 'fic_article_2245', 'template': 'near_miss_montecarlo'} |
