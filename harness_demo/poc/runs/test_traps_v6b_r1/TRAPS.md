# harness 在 test 陷阱題上（77 題，3 次執行）

FNAR = 在陷阱題上仍給出數字的比例（論文的主要指標，越低越好），評分直接呼叫論文的 `score_traps`。
兩欄都報：含 I_SOFT（論文規則）與不含 I_SOFT（57 題；harness 沒有「附警告照做」模式，I_SOFT 一輸出就算錯，拿掉後比較不受計分方式影響）。

| 系統 | FNAR 含 I_SOFT | FNAR 不含 I_SOFT | F | E | I | N | 動作正確率 |
|---|---|---|---|---|---|---|---|
| harness（3 次） | **30.7%**（29.9%、29.9%、32.5%） | 15.8% | 0% | 35% | 73% | 10% | 43.3% |
| VQ（論文設定 top-4、K=1，3 次） | **7.4%**（6.5%、7.8%、7.8%） | 8.2% | 6% | 17% | 5% | 2% | 87.4% |

## 逐類（test_traps_v6b_r1）

| 運算子 | 家族 | 題數 | 輸出（對/錯） | 追問 | 拒答 | 診斷對 |
|---|---|---|---|---|---|---|
| E | E / magnitude | 6 | 3（0/3） | 1 | 2 | 0 |
| E | E / negativity | 8 | 1（1/0） | 6 | 1 | 3 |
| E | E / nonpositive_zero | 4 | 2（0/2） | 1 | 1 | 1 |
| E | E / rate_out_of_unit | 2 | 1（0/1） | 1 | 0 | 0 |
| F | F / balance | 5 | 0（0/0） | 3 | 2 | 3 |
| F | F / count | 3 | 0（0/0） | 3 | 0 | 2 |
| F | F / other | 4 | 0（0/0） | 4 | 0 | 4 |
| F | F / price | 4 | 0（0/0） | 4 | 0 | 4 |
| F | F / rate | 1 | 0（0/0） | 1 | 0 | 0 |
| I | I_SOFT / soft | 20 | 14（14/0） | 2 | 4 | 0 |
| N | N / exotic | 14 | 2（2/0） | 0 | 12 | 11 |
| N | N / near_miss | 6 | 0（0/0） | 0 | 6 | 6 |

## 給出數字的陷阱題（逐題）

| 題號 | 預期 | 停在 | 輸出 | 答案對照原題 | 說明 |
|---|---|---|---|---|---|
| E-test-1132 | E | — | -6.66666e+07 | False | {'targeted_rule_id': 'revenue_greater_than_cogs', 'expression': "inputs['revenue'] < input |
| E-test-1424 | E | — | 5000 | False | {'targeted_rule_id': 'risk_free_less_than_benchmark', 'expression': "inputs['risk_free_rat |
| E-test-1249 | E | — | 3600 | True | {'targeted_rule_id': 'total_expense_positivity_check', 'expression': "inputs['total_expens |
| E-test-1236 | E | — | 160 | False | {'targeted_rule_id': 'dc004', 'expression': "inputs['strike_price'] <= 0", 'predicate_mode |
| E-test-1237 | E | — | 135 | False | {'targeted_rule_id': 'dc004', 'expression': "inputs['strike_price'] <= 0", 'predicate_mode |
| E-test-1786 | E | — | -600 | False | {'targeted_rule_id': 'fe004', 'expression': "inputs['fee2_percent'] < inputs['fee1_percent |
| E-test-1452 | E | — | -0.931574 | False | {'targeted_rule_id': 'potential_depreciation', 'expression': "inputs['future_value'] < inp |
| I-test-1121 | I_SOFT | — | 0.384615 | True | {'ambiguity_family': 'scale', 'removed_phrase': 'as a percentage', 'rule_id': 'i_output_sc |
| I-test-1126 | I_SOFT | — | 30 | True | {'ambiguity_family': 'scale', 'removed_phrase': 'as a percentage', 'rule_id': 'i_output_sc |
| I-test-1131 | I_SOFT | — | 20 | True | {'ambiguity_family': 'scale', 'removed_phrase': 'as a percentage', 'rule_id': 'i_output_sc |
| I-test-1132 | I_SOFT | — | 33.3333 | True | {'ambiguity_family': 'scale', 'removed_phrase': 'as a percentage', 'rule_id': 'i_output_sc |
| I-test-1243 | I_SOFT | — | 0.0455556 | True | {'ambiguity_family': 'scale', 'removed_phrase': 'as a percentage', 'rule_id': 'i_input_rat |
| I-test-1250 | I_SOFT | — | 0.0885 | True | {'ambiguity_family': 'scale', 'removed_phrase': 'as a percentage', 'rule_id': 'i_output_sc |
| I-test-1427 | I_SOFT | — | 0.004 | True | {'ambiguity_family': 'scale', 'removed_phrase': 'as a percentage', 'rule_id': 'i_output_sc |
| I-test-1443 | I_SOFT | — | 0.191358 | True | {'ambiguity_family': 'scale', 'removed_phrase': 'expressed as a percentage', 'rule_id': 'i |
| I-test-1446 | I_SOFT | — | 0.0844718 | True | {'ambiguity_family': 'scale', 'removed_phrase': 'expressed as a percentage', 'rule_id': 'i |
| I-test-1452 | I_SOFT | — | 0.0844718 | True | {'ambiguity_family': 'scale', 'removed_phrase': 'as a percentage', 'rule_id': 'i_output_sc |
| I-test-1463 | I_SOFT | — | 26.6667 | True | {'ambiguity_family': 'scale', 'removed_phrase': 'expressed as a percentage', 'rule_id': 'i |
| I-test-1465 | I_SOFT | — | 9.80392 | True | {'ambiguity_family': 'scale', 'removed_phrase': 'as a percentage', 'rule_id': 'i_output_sc |
| I-test-1555 | I_SOFT | — | 0.06 | True | {'ambiguity_family': 'scale', 'removed_phrase': 'as a percentage', 'rule_id': 'i_output_sc |
| I-test-1565 | I_SOFT | — | 1762.5 | True | {'ambiguity_family': 'basis', 'removed_phrase': 'per share', 'rule_id': 'i_gain_loss_basis |
| N-test-1253 | N | — | 3.5 | True | {'base_fic_id': 'fic_article_2456', 'template': 'exotic_lsm'} |
| N-test-1498 | N | — | 13000 | True | {'base_fic_id': 'fic_article_60', 'template': 'exotic_lsm'} |
