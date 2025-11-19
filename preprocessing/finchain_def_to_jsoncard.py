import argparse
import ast
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

from google import genai
from google.genai import types as genai_types


from google import genai
from google.genai import types as genai_types

CARD_SCHEMA = genai_types.Schema(
    type=genai_types.Type.OBJECT,
    properties={
        "id": genai_types.Schema(
            type=genai_types.Type.STRING,
            description="Unique identifier for the card/model.",
        ),
        "name": genai_types.Schema(
            type=genai_types.Type.STRING,
            description="The full name of the formula or concept.",
        ),
        "short_description": genai_types.Schema(
            type=genai_types.Type.STRING,
            description="A brief summary of the card's content.",
        ),
        "domain": genai_types.Schema(type=genai_types.Type.STRING),
        "topic": genai_types.Schema(type=genai_types.Type.STRING),
        "inputs": genai_types.Schema(
            type=genai_types.Type.ARRAY,
            description="A list of required input variables for the calculation.",
            items=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "name": genai_types.Schema(type=genai_types.Type.STRING),
                    "type": genai_types.Schema(
                        type=genai_types.Type.STRING,
                        enum=["float", "integer", "string"],
                    ),
                    "description": genai_types.Schema(type=genai_types.Type.STRING),
                },
                required=["name", "type", "description"],
            ),
        ),
        "output_var": genai_types.Schema(type=genai_types.Type.STRING),
        
        # 👇【已修正】新的 sympy_formulas 欄位，改為 ARRAY 結構
        "sympy_formulas": genai_types.Schema(
            type=genai_types.Type.ARRAY,
            description="A list of SymPy-compatible string expressions for intermediate/final calculations.",
            items=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "variable": genai_types.Schema(type=genai_types.Type.STRING, description="The resulting variable name (e.g., 'NPV')."),
                    "formula": genai_types.Schema(type=genai_types.Type.STRING, description="The SymPy formula string (e.g., 'C1/(1+r) + C2/(1+r)**2')."),
                },
                required=["variable", "formula"],
            ),
        ),
        
        "tags": genai_types.Schema(
            type=genai_types.Type.ARRAY,
            items=genai_types.Schema(type=genai_types.Type.STRING),
        ),
    },
    required=[
        "id",
        "name",
        "short_description",
        "domain",
        "topic",
        "inputs",
        "output_var",
        "sympy_formulas", # 👈 【已修正】要求這個新的欄位
        "tags",
    ],
)

ONE_SHOT_EXAMPLE = """
## Example:
input: 
```python
def template_three_year_with_salvage_npv():
    investor = random.choice(investor_names)
    project  = random.choice(project_names)

    x        = random.randint(30_000, 100_000)      # $ initial
    C1       = random.randint(12_000, 30_000)       # $ after yr 1
    C2       = random.randint(12_000, 30_000)       # $ after yr 2
    salvage  = random.randint(40_000, 90_000)       # $ at yr 3
    r        = round(random.uniform(6, 14), 2)      # %

    question = (
        f"investor invests $x in project. It is expected to generate "
        f"$C1 after one year, $C2 after two years, and a salvage value of "
        f"$salvage at the end of year three. Using a r:.2f% discount rate, "
        f"what is the NPV?"
    )

    PV_op      = round(C1/(1+r/100) + C2/((1+r/100)**2), 2)
    PV_salvage = round(salvage/((1+r/100)**3), 2)
    NPV        = round(PV_op + PV_salvage - x, 2)

    solution = (
        f"Step 1: Present value of operating inflows:\n"
        f"  PV₁ = $C1/(1+{r/100:.4f}),  PV₂ = ${C2}/(1+{r/100:.4f})²\n"
        f"  PV_op = PV₁ + PV₂ = ${PV_op:,.2f}\n\n"
        f"Step 2: Present value of the year‑3 salvage:\n"
        f"  PV_salvage = ${salvage}/(1+{r/100:.4f})³ = ${PV_salvage:,.2f}\n\n"
        f"Step 3: Net present value:\n"
        f"  NPV = PV_op + PV_salvage − initial investment\n"
        f"      = ${PV_op:,.2f} + ${PV_salvage:,.2f} − ${x} = ${NPV:,.2f}"
    )
    return question, solution
```
output:
```json
{
  "id": "template_three_year_with_salvage_npv",
  "name": "NPV: two cash flows + salvage in year 3",
  "short_description": "Initial investment x, inflows C1 and C2 in years 1 and 2, and salvage in year 3, discounted at rate r.",
  "domain": "Corporate Finance", 
  "topic": "Net Present Value",  
  "inputs": [
    {"name": "x", "type": "float", "description": "initial investment"},
    {"name": "C1", "type": "float", "description": "cash flow at year 1"},
    {"name": "C2", "type": "float", "description": "cash flow at year 2"},
    {"name": "salvage", "type": "float", "description": "salvage value at year 3"},
    {"name": "r", "type": "float", "description": "discount rate as decimal"}
  ],
  "output_var": "NPV",
  "sympy_expr": {
    "PV_op": "C1/(1+r) + C2/(1+r)**2",
    "PV_salvage": "salvage/(1+r)**3",
    "NPV": "PV_op + PV_salvage - x"
  },
  "tags": ["npv", "multi-period", "salvage"]
}
```

"""
PROMPT_TEMPLATE = """You are a financial modeling expert tasked with turning Python template functions into structured JSON cards.

Context:
- Domain folder: {domain_slug} (use human-readable form "{domain_title}" in the card)
- Topic file: {topic_title}
- Definition name: {definition_name}
- Source path: {source_path}

Use the docstring and code block below to infer:
1. A concise user-facing `name`.
2. A `short_description` explaining the scenario and goal.
3. Inputs (variable type, role, and timing if relevant). If type is unclear, choose float.
4. **VERY IMPORTANT**: all of the deterministic and random inputs should be included. for example, if the code uses `year=4`, include `year` as an input, and don't set the variable to be 4. Make it be the same as random variable do.
5. Output variable (usually the final value calculated).
6. Symbolic expressions in SymPy-compatible strings. Use lower-case `r` for decimal rates unless the code clearly uses percentages.
7. Tags: include at least `{topic_slug}`, `npv`, `financial-template`.

Return ONLY JSON that conforms to the provided schema.

Before you start, please make sure you understand the following:
1. The docstring is the description of the function.
2. The code is the implementation of the function.
3. The id is the name of the function.
4. The name is the name of the function.
5. The short_description is the short description of the function.
6. The domain is the domain of the function.
7. The topic is the topic of the function.
8. The inputs are the inputs of the function.
9. The output_var is the output variable of the function.
10. The sympy_expr is the symbolic expression of the function.
11. The tags are the tags of the function.

Example:
{one_shot_example}

Docstring:
\"\"\"{docstring}\"\"\"

Code:
```
{code_block}
```
"""


@dataclass
class DefinitionSnippet:
    name: str
    docstring: str
    code: str


def extract_definitions(py_path: Path) -> List[DefinitionSnippet]:
    source = py_path.read_text(encoding="utf-8")
    module = ast.parse(source)
    lines = source.splitlines()
    snippets: List[DefinitionSnippet] = []

    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name != "main":
            start = node.lineno - 1
            end = node.end_lineno
            code_block = "\n".join(lines[start:end])
            doc = ast.get_docstring(node) or ""
            snippets.append(DefinitionSnippet(node.name, doc, code_block))

    if not snippets:
        raise ValueError(f"No function definitions found in {py_path}")
    return snippets


def infer_domain_info(py_path: Path) -> Dict[str, str]:
    parts = py_path.parts
    if "templates" not in parts:
        raise ValueError("Input path must contain 'templates'")
    temp_idx = parts.index("templates")
    domain_slug = parts[temp_idx + 1]
    domain_title = domain_slug.replace("_", " ").title()
    topic_slug = py_path.stem
    topic_title = topic_slug.replace("_", " ").title()
    return {
        "domain_slug": domain_slug,
        "domain_title": domain_title,
        "topic_slug": topic_slug,
        "topic_title": topic_title,
    }


def build_prompt(snippet: DefinitionSnippet, context: Dict[str, str], source_path: Path) -> str:
    return PROMPT_TEMPLATE.format(
        one_shot_example=ONE_SHOT_EXAMPLE,
        domain_slug=context["domain_slug"],
        domain_title=context["domain_title"],
        topic_title=context["topic_title"],
        topic_slug=context["topic_slug"],
        definition_name=snippet.name,
        source_path=str(source_path),
        docstring=snippet.docstring.strip(),
        code_block=snippet.code.strip(),
    )


def call_gemini(client: genai.Client, prompt: str, model: str) -> Dict[str, Any]:
    config = genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=CARD_SCHEMA,
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    try:
        return json.loads(response.text)
    except json.JSONDecodeError as err:
        raise ValueError(f"Gemini returned invalid JSON: {response.text}") from err


def ensure_output_path(path: Path) -> None:
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert finchain template definitions into JSON cards via Gemini."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the template .py file (e.g., finchain/data/templates/investment_analysis/npv.py)",
    )

    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model name to use.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Missing GEMINI_API_KEY in environment.", file=sys.stderr)
        sys.exit(1)
    start_time = time.time()
    client = genai.Client(api_key=api_key)

    py_path = args.input.resolve()
    if not py_path.exists():
        raise FileNotFoundError(py_path)

    context = infer_domain_info(py_path)
    snippets = extract_definitions(py_path)

    cards = []
    count = 1
    for snippet in snippets:
        # 初始化重試參數
        max_retries = 5 # 增加最大重試次數
        wait_time = 5   # 初始等待時間 (秒)
        
        for attempt in range(max_retries):
            try:
                print(f"Processing {count} of {len(snippets)}: {snippet.name} (Attempt {attempt + 1})...")
                prompt = build_prompt(snippet, context, py_path)
                card = call_gemini(client, prompt, args.model)
                cards.append(card)
                count += 1
                # 成功後，休息一小段時間以保持禮貌，然後跳出重試循環
                time.sleep(5) 
                break 
                
            except genai.errors.ServerError as e:
                if e.code == 503:
                    # 這是 503/過載錯誤
                    if attempt < max_retries - 1:
                        print(f"Server overloaded (503). Retrying in {wait_time} seconds: {e}")
                        time.sleep(wait_time)
                        # 指數增加等待時間（例如：5, 10, 20, 40, ... 秒）
                        wait_time *= 2
                    else:
                        # 最後一次嘗試失敗，跳過
                        print(f"Failed to process {snippet.name} after {max_retries} attempts, skipping: {e}")
                        # 使用 continue 進入下一個 snippet 的處理
                        break 
                else:
                    # 處理其他 ServerError (例如 500)
                    print(f"Other ServerError: {e}, skipping.")
                    break 
            
            except Exception as e:
                # 處理其他非服務器錯誤 (如 JSONDecodeError)
                print(f"General Error: {e}, skipping.")
                break
    output_path = Path("data") / context["domain_slug"] / f"{context['topic_slug']}.json"
    ensure_output_path(output_path)
    output_path.write_text(json.dumps(cards, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(cards)} cards to {output_path}")
    print(f"Total time taken: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()

# python preprocessing/finchain_def_to_jsoncard.py --input finchain/data/templates/investment_analysis/npv.py --output data/investment_analysis/npv.json