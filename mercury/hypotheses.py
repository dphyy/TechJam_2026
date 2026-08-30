"""Deterministic, bounded intent hypotheses for ambiguous retrieval plans."""

from __future__ import annotations

from mercury.types import RetrievalHypothesis, RetrievalPlan


_USE_CASE_OBJECTS = {
    "running": ("sneakers",),
    "walking": ("shoes",),
    "hiking": ("boots",),
    "swimming": ("swimwear",),
    "wedding": ("dresses", "jewelry"),
    "work": ("shirts", "bags"),
    "travel": ("bags", "shoes"),
    "yoga": ("leggings",),
    "cycling": ("shorts",),
    "gym": ("tops", "shoes"),
    "winter": ("coats", "boots"),
    "summer": ("shirts", "sandals"),
    "beach": ("swimwear", "sandals"),
    "party": ("dresses", "jewelry"),
}


def build_intent_hypotheses(plan: RetrievalPlan, maximum: int = 2) -> tuple[RetrievalHypothesis, ...]:
    if type(maximum) is not int or not 1 <= maximum <= 2:
        raise ValueError("Intent hypotheses must be bounded to one or two")
    hypotheses = []
    base_query = plan.lexical_query or " ".join((*plan.object_types, *plan.use_case, *plan.positive_terms))
    if plan.object_types:
        for object_type in plan.object_types[:maximum]:
            terms = (object_type, *[value for value in plan.positive_terms if value != object_type])
            hypotheses.append(RetrievalHypothesis(" ".join(dict.fromkeys(terms)), (object_type,),
                                                  "explicit_object"))
    elif base_query:
        hypotheses.append(RetrievalHypothesis(base_query, (), "open_request"))
    if len(hypotheses) < maximum:
        for use_case in plan.use_case:
            for object_type in _USE_CASE_OBJECTS.get(use_case, ()):
                query = " ".join(dict.fromkeys((object_type, use_case, *plan.positive_terms)))
                candidate = RetrievalHypothesis(query, (object_type,), f"use_case:{use_case}")
                if candidate.query not in {item.query for item in hypotheses}:
                    hypotheses.append(candidate)
                if len(hypotheses) == maximum:
                    break
            if len(hypotheses) == maximum:
                break
    return tuple(hypotheses[:maximum])
