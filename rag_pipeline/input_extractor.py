import json
from dataclasses import dataclass
from typing import Dict, List, Any

from google import genai
from google.genai import types as genai_types

from card_store import CardRecord

EXTRACTION_SCHEMA = genai_types.Schema(
    type=genai_types.Type.OBJECT,
    properties={
        # 修正：改用 ARRAY 來表示變數及其值，以符合 API 規範
        "provided_inputs": genai_types.Schema(
            type=genai_types.Type.ARRAY,
            description="A list of input variables extracted from the context and their corresponding values.",
            items=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "variable": genai_types.Schema(type=genai_types.Type.STRING, description="The name of the variable (e.g., 'x', 'r')."),
                    "value": genai_types.Schema(type=genai_types.Type.STRING, description="The extracted numeric value as a raw string (e.g., '1000', '0.05')."),
                },
                required=["variable", "value"],
            ),
        ),
        "missing_inputs": genai_types.Schema(
            type=genai_types.Type.ARRAY,
            description="A list of required input variable names that could not be found in the question.",
            items=genai_types.Schema(type=genai_types.Type.STRING),
        ),
    },
    required=["provided_inputs", "missing_inputs"],
)


@dataclass
class ExtractionResult:
    provided_inputs: Dict[str, float]
    missing_inputs: List[str]


def extract_inputs_with_llm(
    client: genai.Client,
    model: str,
    user_question: str,
    card: Any, # 假設 CardRecord，這裡使用 Any 避免未定義錯誤
) -> ExtractionResult:
    required_inputs = card.data.get("inputs", [])
    
    # 創建格式化輸入列表，用於 Prompt
    formatted_inputs = "\n".join(
        f"- {inp['name']} ({inp['type']}): {inp.get('description','')}"
        for inp in required_inputs
    )

    prompt = f"""
You are helping parse user questions for financial calculations.

Card ID: {card.id}
Required inputs:
{formatted_inputs}

User question:
{user_question}

Extract the numeric values provided in the question for each required input listed above. If any are missing, list them in 'missing_inputs'.
Values should be given as raw numbers without formatting (e.g., '1000.5', '0.05'). Return JSON only.
"""
    config = genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=EXTRACTION_SCHEMA,
    )
    
    # 呼叫 Gemini API
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    
    data = json.loads(response.text)
    
    # 關鍵修正：將陣列轉換回字典
    # data["provided_inputs"] 現在是 [{'variable': 'x', 'value': '1000'}, ...]
    provided = {
        item["variable"]: float(item["value"]) # 將字串值轉換為 float
        for item in data.get("provided_inputs", [])
    }
    
    return ExtractionResult(
        provided_inputs=provided,
        missing_inputs=data.get("missing_inputs", []),
    )