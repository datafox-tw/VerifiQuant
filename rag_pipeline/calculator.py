from dataclasses import dataclass
from typing import Dict, List, Tuple

import sympy as sp

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


def evaluate_card(card: CardRecord, inputs: Dict[str, float]) -> CalculationResult:
    env: Dict[str, float] = {}
    env.update(inputs)

    steps: List[CalculationStep] = []
    formulas = card.data.get("sympy_formulas", [])
    if not formulas:
        raise ValueError(f"Card {card.id} has no sympy_formulas.")

    for formula in formulas:
        variable = formula["variable"]
        expr_str = formula["formula"]
        expr = sp.sympify(expr_str, locals=env)
        value = float(expr.evalf())
        env[variable] = value
        steps.append(CalculationStep(variable=variable, formula=expr_str, value=value))

    output_var = card.data.get("output_var", "")
    output_value = env.get(output_var)
    if output_var == "" or output_value is None:
        raise ValueError(f"Output variable {output_var} missing for card {card.id}.")

    return CalculationResult(
        steps=steps,
        output_var=output_var,
        output_value=float(output_value),
    )

