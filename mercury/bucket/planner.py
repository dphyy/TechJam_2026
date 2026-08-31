from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from mercury.lexical.dialogue import SessionState

from .index import CatalogIndex, tokens
from .ranking import COLORS, MATERIALS, RankedProduct


@dataclass(frozen=True, slots=True)
class Question:
    attribute: str | None
    message: str
    uncertainty: float = 0.0


QUESTIONS = {
    "material": "What material or fabric would you prefer?",
    "color": "Which color would you prefer?",
    "size": "What size, width, or fit do you need?",
    "style": "What style would work best for you?",
    "use_case": "What activity or occasion will you use it for?",
    "budget": "What price range would you like to stay within?",
    "other": "What other feature or requirement matters most to you?",
}
FACET_WORDS = {
    "material": MATERIALS, "color": COLORS,
    "size": frozenset({"small", "medium", "large", "wide", "narrow", "petite", "plus"}),
    "style": frozenset({"casual", "formal", "slim", "relaxed", "fitted", "classic", "vintage"}),
    "use_case": frozenset({"hiking", "running", "gym", "winter", "outdoor", "work", "travel"}),
}


def choose_question(state: SessionState, index: CatalogIndex, ranked: list[RankedProduct],
                    turn: int, policy: str) -> Question:
    if turn >= 10 or policy == "none" or not ranked:
        return Question(None, "Here are the closest catalog matches for your requirements.")
    # A free-form question is valuable before any specific requirement is known.
    # Afterwards, prefer a real varying catalog facet over a repeated question.
    if policy == "other" or not state.asked_attributes:
        if "other" not in state.no_preference_attributes and state.asked_attributes.count("other") < 2:
            return Question("other", QUESTIONS["other"])
    products = [index.products[item.identifier] for item in ranked[:30]]
    mentioned = {word for item in state.evidence if item.source != "category" for word in tokens(item.text)}
    choices = []
    for attribute, vocabulary in FACET_WORDS.items():
        if attribute in state.no_preference_attributes or state.asked_attributes.count(attribute) >= 2:
            continue
        counts = Counter(tuple(sorted(product.positive & vocabulary)) for product in products)
        counts.pop((), None)
        known = sum(counts.values())
        entropy = -sum(count / known * math.log2(count / known) for count in counts.values()) if known else 0.0
        information = entropy * known / len(products)
        # Existing constraints reduce the need to ask the same facet again.
        if mentioned & vocabulary:
            information /= 2
        information /= 1 + state.asked_attributes.count(attribute)
        if information > 0:
            choices.append((information, attribute))
    if choices:
        uncertainty, attribute = min(choices, key=lambda item: (-item[0], item[1]))
        return Question(attribute, QUESTIONS[attribute], uncertainty)
    if "other" not in state.no_preference_attributes and state.asked_attributes.count("other") < 2:
        return Question("other", QUESTIONS["other"])
    remaining = [attribute for attribute in ("budget", "size", "use_case")
                 if attribute not in state.no_preference_attributes and attribute not in state.asked_attributes]
    if remaining:
        return Question(remaining[0], QUESTIONS[remaining[0]])
    return Question(None, "Here are the closest catalog matches for your requirements.")
