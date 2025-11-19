import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import sympy as sp
from google import genai
from google.genai import types as genai_types

from card_store import CardRecord


@dataclass
class CalculationStep:
    variable: str
    formula: str
    value: float


@dataclass
class CalculationResult:
    steps: List[CalculationStep]
    output_var: str
    output_value: float
    fallback: bool


FALLBACK_SCHEMA = genai_types.Schema(
    type=genai_types.Type.OBJECT,
    properties={
        "steps": genai_types.Schema(
            type=genai_types.Type.ARRAY,
            items=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "variable": genai_types.Schema(type=genai_types.Type.STRING),
                    "formula": genai_types.Schema(type=genai_types.Type.STRING),
                    "value": genai_types.Schema(type=genai_types.Type.NUMBER),
                },
                required=["variable", "value"],
            ),
        ),
        "output_value": genai_types.Schema(type=genai_types.Type.NUMBER),
    },
    required=["output_value"],
)


def _evaluate_expression(expr_str: str, env: Dict[str, float]) -> float:
    try:
        expr = sp.sympify(expr_str, locals=env)
        value = float(expr.evalf()) if hasattr(expr, "evalf") else float(expr)
        if math.isnan(value) or math.isinf(value):
            raise ValueError("Expression evaluated to invalid number.")
        return value
    except Exception:
        local_env = {**env, "math": math, "pow": pow}
        try:
            value = float(eval(expr_str, {"__builtins__": {}}, local_env))
            if math.isnan(value) or math.isinf(value):
                raise ValueError("Eval returned invalid number.")
            return value
        except Exception as err:
            raise ValueError(f"Failed to evaluate expression '{expr_str}'") from err


def _llm_fallback_calculation(
    client: genai.Client,
    model: str,
    card: CardRecord,
    inputs: Dict[str, float],
    question: str,
) -> CalculationResult:
    formulas_text = "\n".join(
        f"{item['variable']} = {item['formula']}"
        for item in card.data.get("sympy_formulas", [])
    )
    prompt = f"""
You are a financial modeling assistant with Python execution abilities.
The user asked: {question}

Card definition:
- Name: {card.data.get("name", "")}
- Short description: {card.data.get("short_description", "")}
- Formulas:
{formulas_text}

Inputs (JSON):
{json.dumps(inputs)}

Compute the final output ({card.data.get("output_var","")}) by running precise calculations.
Return JSON with the numeric steps you executed (variable/formula/value) and the final output_value.
"""
    config = genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=FALLBACK_SCHEMA,
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    data = json.loads(response.text)
    steps_payload = data.get("steps", [])
    steps: List[CalculationStep] = []
    for idx, step in enumerate(steps_payload):
        steps.append(
            CalculationStep(
                variable=step.get("variable", f"step_{idx+1}"),
                formula=step.get("formula", ""),
                value=float(step.get("value")),
            )
        )
    return CalculationResult(
        steps=steps,
        output_var=card.data.get("output_var", ""),
        output_value=float(data["output_value"]),
        fallback=True,
    )


def evaluate_card(
    card: CardRecord,
    inputs: Dict[str, float],
    *,
    question: Optional[str] = None,
    fallback_client: Optional[genai.Client] = None,
    fallback_model: Optional[str] = None,
) -> CalculationResult:
    env: Dict[str, float] = {}
    env.update(inputs)

    steps: List[CalculationStep] = []
    formulas = card.data.get("sympy_formulas", [])
    if not formulas:
        raise ValueError(f"Card {card.id} has no sympy_formulas.")
    FALLBACK = False
    try:
        for formula in formulas:
            variable = formula["variable"]
            expr_str = formula["formula"]
            value = _evaluate_expression(expr_str, env)
            env[variable] = value
            steps.append(
                CalculationStep(variable=variable, formula=expr_str, value=value)
            )
    except Exception as exc:
        FALLBACK = True
        if fallback_client and fallback_model and question:
            return _llm_fallback_calculation(
                client=fallback_client,
                model=fallback_model,
                card=card,
                inputs=inputs,
                question=question,
            )
        raise exc

    output_var = card.data.get("output_var", "")
    output_value = env.get(output_var)
    if output_var == "" or output_value is None:
        raise ValueError(f"Output variable {output_var} missing for card {card.id}.")

    return CalculationResult(
        steps=steps,
        output_var=output_var,
        output_value=float(output_value),
        fallback = FALLBACK,
    )

