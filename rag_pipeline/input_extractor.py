import json
import re
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from google import genai
from google.genai import types as genai_types
from card_store import CardRecord

# 1. Schema 設定
# 注意：我們不強制 value 必填，因為當 status="missing" 時，value 可能不存在
EXTRACTION_SCHEMA = genai_types.Schema(
    type=genai_types.Type.OBJECT,
    properties={
        "inputs": genai_types.Schema(
            type=genai_types.Type.ARRAY,
            items=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "name": genai_types.Schema(
                        type=genai_types.Type.STRING,
                        description="The variable name exactly as defined in the card."
                    ),
                    "status": genai_types.Schema(
                        type=genai_types.Type.STRING,
                        enum=["provided", "missing"],
                        description="Whether the value is explicitly found in the text."
                    ),
                    "value": genai_types.Schema(
                        type=genai_types.Type.STRING,
                        description="The numeric value found. Convert percentages to decimals. If status is missing, use empty string."
                    ),
                },
                required=["name", "status"], # value 設為選填，增加容錯
            ),
        ),
    },
    required=["inputs"],
)

@dataclass
class ExtractionResult:
    provided_inputs: Dict[str, float]
    missing_inputs: List[str]

def _parse_numeric(value: str) -> float:
    """
    將字串轉換為浮點數。
    增強了對 '7.58%' -> 0.0758 的處理，同時防止 LLM 已經轉過一次的情況。
    """
    if not value:
        raise ValueError("Empty value")
        
    text = str(value).strip().lower()
    
    # 移除常見貨幣符號與逗號
    text = text.replace(",", "").replace("$", "").replace(" ", "")
    
    # 檢查是否包含百分比符號
    is_percent = "%" in text
    text = text.replace("%", "")
    
    # 使用正則表達式提取數字部分 (支援負數與小數)
    match = re.search(r"-?\d+(\.\d+)?", text)
    if not match:
        raise ValueError(f"Unable to parse numeric value from '{value}'")
        
    number = float(match.group())
    
    # 邏輯判斷：如果原本有%，絕對要除以 100
    if is_percent:
        number /= 100
        
    return number

def extract_inputs_with_llm(
    client: genai.Client,
    model: str,
    user_question: str,
    card: CardRecord,
) -> ExtractionResult:
    # 準備 Prompt 的輸入描述
    required_inputs = card.data.get("inputs", [])
    formatted_inputs = "\n".join(
        f"- {inp['name']} ({inp['type']}): {inp.get('description','')}"
        for inp in required_inputs
    )
    
    # 收集所有需要的變數名稱，作為後續檢查的 Source of Truth
    required_var_names = [inp['name'] for inp in required_inputs]

    instructions = """
Task: Extract numeric values for the required inputs from the User Question.

Rules:
1. **Accuracy**: Only mark status="provided" if the specific number is explicitly stated in the text.
2. **Percentages**: If a value is a percentage (e.g., "7.58%"), convert it to a decimal string in the value field (e.g., "0.0758").
3. **Completeness**: You MUST return an object for EVERY variable listed in "Required inputs".
4. **Inference**: Match variables by meaning (context). 
   - "initial investment" matches 'x'
   - "discount rate" matches 'r'
   - "environmental cleanup" matches 'cleanup'
5. **Missing**: If a value is not found, set status="missing" and value="".

Example JSON Output:
{
  "inputs": [
    {"name": "x", "status": "provided", "value": "107641"},
    {"name": "r", "status": "provided", "value": "0.0758"},
    {"name": "salvage", "status": "missing", "value": ""}
  ]
}
"""
    prompt = f"""
You are a financial data extraction assistant.

Card ID: {card.id}
Required inputs:
{formatted_inputs}

User question:
{user_question}

{instructions}
"""
    print(prompt)
    print("--------------------------")
    config = genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=EXTRACTION_SCHEMA,
    )
    print(config)
    print("--------------------------")
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        
        data = json.loads(response.text)
        print("==========================")
        print("data: ", data)
        print("==========================")
    except Exception as e:
        # 如果 API 失敗或 JSON 解析失敗，將所有變數視為缺失
        print(f"Extraction Error: {e}")
        return ExtractionResult(provided_inputs={}, missing_inputs=required_var_names)

    # 建立 LLM 輸出的查找字典 (Name -> Entry)
    extracted_map = {
        item.get("name"): item 
        for item in data.get("inputs", []) 
        if item.get("name")
    }

    provided: Dict[str, float] = {}
    missing: List[str] = []
    print("==========================")
    print("data: ", extracted_map)
    print("==========================")
    # 【關鍵修正】：遍歷 Card 定義的變數，而不是 LLM 的輸出
    # 這確保了我們檢查了卡片所需的每一個變數
    for req_input in required_inputs:
        print("req_input: ", req_input)
        var_name = req_input["name"]
        
        # 檢查 LLM 是否有回傳這個變數
        if var_name not in extracted_map:
            print("var_name missing: ", var_name)
            missing.append(var_name)
            continue
        print("var_name found: ", var_name)
        entry = extracted_map[var_name]
        status = entry.get("status")
        raw_value = entry.get("value", "")

        if status == "provided" and raw_value:
            try:
                # 嘗試解析數值
                parsed_val = _parse_numeric(str(raw_value))
                provided[var_name] = parsed_val
                print("parsed_val: ", parsed_val)
            except ValueError:
                # 如果數值解析失敗，視為缺失 (雖然 LLM 說有，但格式爛掉了)
                print("parsing failed for var_name: ", var_name)
                missing.append(var_name)
        else:
            print("status missing for var_name: ", var_name)
            missing.append(var_name)

    return ExtractionResult(
        provided_inputs=provided,
        missing_inputs=missing,
    )
# better old method(gemini)
# import json
# import re
# from dataclasses import dataclass
# from typing import Dict, List, Any, Optional
# from google import genai
# from google.genai import types as genai_types
# from card_store import CardRecord

# # 1. Schema 設定
# # 注意：我們不強制 value 必填，因為當 status="missing" 時，value 可能不存在
# EXTRACTION_SCHEMA = genai_types.Schema(
#     type=genai_types.Type.OBJECT,
#     properties={
#         "inputs": genai_types.Schema(
#             type=genai_types.Type.ARRAY,
#             items=genai_types.Schema(
#                 type=genai_types.Type.OBJECT,
#                 properties={
#                     "name": genai_types.Schema(
#                         type=genai_types.Type.STRING,
#                         description="The variable name exactly as defined in the card."
#                     ),
#                     "status": genai_types.Schema(
#                         type=genai_types.Type.STRING,
#                         enum=["provided", "missing"],
#                         description="Whether the value is explicitly found in the text."
#                     ),
#                     "value": genai_types.Schema(
#                         type=genai_types.Type.STRING,
#                         description="The numeric value found. Convert percentages to decimals. If status is missing, use empty string."
#                     ),
#                 },
#                 required=["name", "status"], # value 設為選填，增加容錯
#             ),
#         ),
#     },
#     required=["inputs"],
# )

# @dataclass
# class ExtractionResult:
#     provided_inputs: Dict[str, float]
#     missing_inputs: List[str]

# def _parse_numeric(value: str) -> float:
#     """
#     將字串轉換為浮點數。
#     增強了對 '7.58%' -> 0.0758 的處理，同時防止 LLM 已經轉過一次的情況。
#     """
#     if not value:
#         raise ValueError("Empty value")
        
#     text = str(value).strip().lower()
    
#     # 移除常見貨幣符號與逗號
#     text = text.replace(",", "").replace("$", "").replace(" ", "")
    
#     # 檢查是否包含百分比符號
#     is_percent = "%" in text
#     text = text.replace("%", "")
    
#     # 使用正則表達式提取數字部分 (支援負數與小數)
#     match = re.search(r"-?\d+(\.\d+)?", text)
#     if not match:
#         raise ValueError(f"Unable to parse numeric value from '{value}'")
        
#     number = float(match.group())
    
#     # 邏輯判斷：如果原本有%，絕對要除以 100
#     if is_percent:
#         number /= 100
        
#     return number

# def extract_inputs_with_llm(
#     client: genai.Client,
#     model: str,
#     user_question: str,
#     card: CardRecord,
# ) -> ExtractionResult:
#     # 準備 Prompt 的輸入描述
#     required_inputs = card.data.get("inputs", [])
#     formatted_inputs = "\n".join(
#         f"- {inp['name']} ({inp['type']}): {inp.get('description','')}"
#         for inp in required_inputs
#     )
    
#     # 收集所有需要的變數名稱，作為後續檢查的 Source of Truth
#     required_var_names = [inp['name'] for inp in required_inputs]

#     instructions = """
# Task: Extract numeric values for the required inputs from the User Question.

# Rules:
# 1. **Accuracy**: Only mark status="provided" if the specific number is explicitly stated in the text.
# 2. **Percentages**: If a value is a percentage (e.g., "7.58%"), convert it to a decimal string in the value field (e.g., "0.0758").
# 3. **Completeness**: You MUST return an object for EVERY variable listed in "Required inputs".
# 4. **Inference**: Match variables by meaning (context). 
#    - "initial investment" matches 'x'
#    - "discount rate" matches 'r'
#    - "environmental cleanup" matches 'cleanup'
# 5. **Missing**: If a value is not found, set status="missing" and value="".

# Example JSON Output:
# {
#   "inputs": [
#     {"name": "x", "status": "provided", "value": "107641"},
#     {"name": "r", "status": "provided", "value": "0.0758"},
#     {"name": "salvage", "status": "missing", "value": ""}
#   ]
# }
# """
#     prompt = f"""
# You are a financial data extraction assistant.

# Card ID: {card.id}
# Required inputs:
# {formatted_inputs}

# User question:
# {user_question}

# {instructions}
# """
    
#     config = genai_types.GenerateContentConfig(
#         response_mime_type="application/json",
#         response_schema=EXTRACTION_SCHEMA,
#     )

#     try:
#         response = client.models.generate_content(
#             model=model,
#             contents=prompt,
#             config=config,
#         )
#         data = json.loads(response.text)
#     except Exception as e:
#         # 如果 API 失敗或 JSON 解析失敗，將所有變數視為缺失
#         print(f"Extraction Error: {e}")
#         return ExtractionResult(provided_inputs={}, missing_inputs=required_var_names)

#     # 建立 LLM 輸出的查找字典 (Name -> Entry)
#     extracted_map = {
#         item.get("name"): item 
#         for item in data.get("inputs", []) 
#         if item.get("name")
#     }

#     provided: Dict[str, float] = {}
#     missing: List[str] = []

#     # 【關鍵修正】：遍歷 Card 定義的變數，而不是 LLM 的輸出
#     # 這確保了我們檢查了卡片所需的每一個變數
#     for req_input in required_inputs:
#         var_name = req_input["name"]
        
#         # 檢查 LLM 是否有回傳這個變數
#         if var_name not in extracted_map:
#             missing.append(var_name)
#             continue
            
#         entry = extracted_map[var_name]
#         status = entry.get("status")
#         raw_value = entry.get("value", "")

#         if status == "provided" and raw_value:
#             try:
#                 # 嘗試解析數值
#                 parsed_val = _parse_numeric(str(raw_value))
#                 provided[var_name] = parsed_val
#             except ValueError:
#                 # 如果數值解析失敗，視為缺失 (雖然 LLM 說有，但格式爛掉了)
#                 missing.append(var_name)
#         else:
#             missing.append(var_name)

#     return ExtractionResult(
#         provided_inputs=provided,
#         missing_inputs=missing,
#     )
# worst old  version(cursor)
# import json
# import re
# from dataclasses import dataclass
# from typing import Dict, List

# from google import genai
# from google.genai import types as genai_types

# from card_store import CardRecord

# EXTRACTION_SCHEMA = genai_types.Schema(
#     type=genai_types.Type.OBJECT,
#     properties={
#         "inputs": genai_types.Schema(
#             type=genai_types.Type.ARRAY,
#             items=genai_types.Schema(
#                 type=genai_types.Type.OBJECT,
#                 properties={
#                     "name": genai_types.Schema(type=genai_types.Type.STRING),
#                     "status": genai_types.Schema(
#                         type=genai_types.Type.STRING,
#                         enum=["provided", "missing"],
#                     ),
#                     "value": genai_types.Schema(type=genai_types.Type.STRING),
#                 },
#                 required=["name", "status"],
#             ),
#         ),
#     },
#     required=["inputs"],
# )


# @dataclass
# class ExtractionResult:
#     provided_inputs: Dict[str, float]
#     missing_inputs: List[str]


# def _parse_numeric(value: str) -> float:
#     text = value.strip().lower()
#     is_percent = text.endswith("%")
#     text = text.replace("%", "")
#     text = text.replace(",", "")
#     text = text.replace("$", "")
#     match = re.search(r"-?\d+(\.\d+)?", text)
#     if not match:
#         raise ValueError(f"Unable to parse numeric value from '{value}'")
#     number = float(match.group())
#     if is_percent:
#         number /= 100
#     return number


# def extract_inputs_with_llm(
#     client: genai.Client,
#     model: str,
#     user_question: str,
#     card: CardRecord,
# ) -> ExtractionResult:
#     required_inputs = card.data.get("inputs", [])
#     formatted_inputs = "\n".join(
#         f"- {inp['name']} ({inp['type']}): {inp.get('description','')}"
#         for inp in required_inputs
#     )

#     instructions = """
# Rules:
# 1. Remove currency symbols and commas before returning numbers.
# 2. Convert percentages to decimals (e.g., 7.58% => 0.0758).
# 3. Provide an entry for every required input. Use status="provided" when the value is in the text, otherwise status="missing".
# 4. Only mark inputs as missing when the question truly lacks the numeric value. Consider synonyms like "invest", "initial cost", "save", "sold for", "cleanup cost".
# 5. When status="provided", include the numeric string in the "value" field.

# Example:
# {
#   "inputs": [
#     {"name": "x", "status": "provided", "value": "50000"},
#     {"name": "r", "status": "provided", "value": "0.08"},
#     {"name": "C2", "status": "missing"}
#   ]
# }
# """

#     prompt = f"""
# You are helping parse user questions for financial calculations.

# Card ID: {card.id}
# Required inputs:
# {formatted_inputs}

# User question:
# {user_question}

# {instructions}

# Return JSON only.
# """
#     config = genai_types.GenerateContentConfig(
#         response_mime_type="application/json",
#         response_schema=EXTRACTION_SCHEMA,
#     )

#     response = client.models.generate_content(
#         model=model,
#         contents=prompt,
#         config=config,
#     )

#     data = json.loads(response.text)
#     provided: Dict[str, float] = {}
#     missing: List[str] = []

#     for entry in data.get("inputs", []):
#         name = entry.get("name")
#         status = entry.get("status")
#         if not name or not status:
#             continue
#         if status == "provided":
#             value = entry.get("value")
#             if value is None:
#                 missing.append(name)
#                 continue
#             try:
#                 provided[name] = _parse_numeric(value)
#             except ValueError:
#                 missing.append(name)
#         else:
#             missing.append(name)

#     return ExtractionResult(
#         provided_inputs=provided,
#         missing_inputs=missing,
#     )

