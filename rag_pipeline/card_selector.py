import json
from dataclasses import dataclass
from typing import List

from google import genai
from google.genai import types as genai_types

from card_store import CardRecord
from rag_pipeline.retrieval import RetrievalCandidate


SELECTION_SCHEMA = genai_types.Schema(
    type=genai_types.Type.OBJECT,
    properties={
        "chosen_id": genai_types.Schema(
            type=genai_types.Type.STRING,
            description="The ID of the chosen card, or empty string '' if no card matches the question."
        ),
        "reason": genai_types.Schema(
            type=genai_types.Type.STRING,
            description="Explanation for the choice, or why no card matches if chosen_id is empty."
        ),
    },
    required=["chosen_id", "reason"],
)


@dataclass
class SelectionResult:
    chosen_id: str
    reason: str


def select_card_with_llm(
    client: genai.Client,
    model: str,
    user_question: str,
    candidates: List[RetrievalCandidate],
) -> SelectionResult:
    if not candidates:
        raise ValueError("No candidates available for selection.")

    formatted_candidates = "\n\n".join(
        f"Candidate {idx+1}:\n{candidate.as_context()}"
        for idx, candidate in enumerate(candidates)
    )

    prompt = f"""
You are a financial modeling expert helping to pick the best calculation template.

User Question:
{user_question}

Candidate Cards:
{formatted_candidates}

Instructions:
- If one of the candidate cards matches the question well, choose it and set chosen_id to that card's ID.
- If NONE of the candidate cards are suitable for answering this question (e.g., the question is about a topic not covered by any card, or the required calculation type is different), set chosen_id to an empty string '' and explain why in the reason field.

Return JSON only.
"""
    config = genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=SELECTION_SCHEMA,
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    data = json.loads(response.text)
    return SelectionResult(
        chosen_id=data["chosen_id"],
        reason=data["reason"],
    )

