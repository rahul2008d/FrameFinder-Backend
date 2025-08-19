# services/planner_llm.py
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel
from loguru import logger
import json

from settings import settings

try:
    # OpenAI SDK v1
    from openai import OpenAI
except Exception:  # optional import guard
    OpenAI = None  # type: ignore


class QueryPlan(BaseModel):
    normalized: str
    synonyms: List[str] = []
    objects: List[str] = []
    actions: List[str] = []
    person_names: List[str] = []
    negatives: List[str] = []
    notes: Optional[str] = None


def _make_prompt(query: str) -> str:
    return f"""
You are helping a video moment retriever that uses CLIP (image/text embeddings).
Given the user's query, output a compact JSON object with:

- "normalized": short, literal reformulation
- "synonyms": up to 3 short alternative phrasings
- "objects": key visual objects/entities (e.g., "bus", "woman", "screen text")
- "actions": key verbs or visual actions (e.g., "getting off", "waves", "walks in")
- "person_names": any literal names mentioned (e.g., "rahul datta")
- "negatives": short negative constraints if implied (e.g., "not intro slide")

GUIDELINES:
- Keep phrases short and visual (no long sentences).
- Prefer concrete nouns/verbs over abstract concepts.
- If a field is not applicable, return an empty list.

USER_QUERY: {query}
Return ONLY JSON.
""".strip()


def plan_query(query: str) -> Optional[QueryPlan]:
    if not settings.planner_enabled:
        return None
    if not settings.openai_api_key or OpenAI is None:
        logger.info("Planner disabled (no API key or OpenAI SDK missing).")
        return None

    client = OpenAI(api_key=settings.openai_api_key)
    try:
        resp = client.chat.completions.create(
            model=settings.planner_model,
            messages=[
                {"role": "system", "content": "You produce compact JSON for visual search planning."},
                {"role": "user", "content": _make_prompt(query)},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
            timeout=10,
        )
        content = resp.choices[0].message.content or "{}"
        data = json.loads(content)
        plan = QueryPlan.model_validate(data)
        logger.info(f"PLANNER plan: normalized={plan.normalized!r} "
                    f"syn={len(plan.synonyms)} objs={len(plan.objects)} acts={len(plan.actions)}")
        return plan
    except Exception as e:
        logger.warning(f"planner_llm failed: {e}")
        return None
