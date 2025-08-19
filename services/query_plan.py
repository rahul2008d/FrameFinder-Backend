# services/query_plan.py
from pydantic import BaseModel, Field
from typing import List, Dict

class QueryPlan(BaseModel):
    expansions: List[str] = Field(default_factory=list)
    weights: Dict[int, float] = Field(default_factory=dict)  # optional, by expansion index
    recall_k: int = 120
    final_k: int = 2
