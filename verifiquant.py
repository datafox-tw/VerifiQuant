import argparse
import os
from pathlib import Path
from typing import Dict, Optional

from google import genai

from card_store import DefinitionStore
from rag_pipeline.calculator import evaluate_card
from rag_pipeline.card_selector import select_card_with_llm
from rag_pipeline.input_extractor import extract_inputs_with_llm
from rag_pipeline.retrieval import fetch_candidates


def solve_question(
    question: str,
    store: DefinitionStore,
    client: genai.Client,
    *,
    selector_model: str,
    extractor_model: str,
    top_k: int = 3,
    alpha: float = 0.4,
    domain: Optional[str] = None,
    topic: Optional[str] = None,
) -> Dict[str, object]:
    candidates = fetch_candidates(
        store,
        query=question,
        top_k=top_k,
        alpha=alpha,
        domain=domain,
        topic=topic,
    )
    if not candidates:
        return {
            "status": "refused",
            "reason": "No relevant cards found for the given question.",
        }

    selection = select_card_with_llm(
        client=client,
        model=selector_model,
        user_question=question,
        candidates=candidates,
    )

    chosen_card = next(
        (candidate.card for candidate in candidates if candidate.card.id == selection.chosen_id),
        None,
    )
    if not chosen_card:
        return {
            "status": "refused",
            "reason": f"LLM selected card {selection.chosen_id} which was not retrieved.",
        }

    extraction = extract_inputs_with_llm(
        client=client,
        model=extractor_model,
        user_question=question,
        card=chosen_card,
    )
    if extraction.missing_inputs:
        return {
            "status": "refused",
            "reason": "Missing numeric values for required inputs.",
            "missing_inputs": extraction.missing_inputs,
        }
    # if selection.missing_inputs:
    #     return {
    #         "status": "refused",
    #         "reason": "Missing inputs from user question.",
    #         "missing_inputs": selection.missing_inputs,
    #     }
    result = evaluate_card(chosen_card, extraction.provided_inputs)
    return {
        "status": "success",
        "card_id": chosen_card.id,
        "selection_reason": selection.reason,
        "inputs": extraction.provided_inputs,
        "steps": [
            {
                "variable": step.variable,
                "formula": step.formula,
                "value": step.value,
            }
            for step in result.steps
        ],
        "output_var": result.output_var,
        "output_value": result.output_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the end-to-end RAG workflow for financial data cards."
    )
    parser.add_argument(
        "--store",
        type=Path,
        default=Path("artifacts/definition_store.pkl"),
        help="Path to the serialized DefinitionStore.",
    )
    parser.add_argument(
        "--selector-model",
        default="gemini-2.5-flash",
        help="Model for card selection.",
    )
    parser.add_argument(
        "--extractor-model",
        default="gemini-2.5-flash",
        help="Model for input extraction.",
    )
    parser.add_argument(
        "--question",
        required=True,
        help="User question to solve.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
        help="Hybrid weighting for BM25 vs embedding similarity.",
    )
    parser.add_argument(
        "--domain",
        help="Optional domain filter.",
    )
    parser.add_argument(
        "--topic",
        help="Optional topic filter.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set.")
    client = genai.Client(api_key=api_key)

    store = DefinitionStore.load(args.store)
    outcome = solve_question(
        args.question,
        store,
        client,
        selector_model=args.selector_model,
        extractor_model=args.extractor_model,
        alpha=args.alpha,
        domain=args.domain,
        topic=args.topic,
    )
    print(outcome)


if __name__ == "__main__":
    main()

