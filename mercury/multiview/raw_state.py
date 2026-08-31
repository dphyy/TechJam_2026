"""Adapt active phrase evidence without searching retired conversation history."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256

from mercury.lexical.dialogue import GENERIC_PREFIX_RE, SessionState as PhraseState
from mercury.lexical.product_features import alternative_values, component_scope, component_value
from mercury.types import Preference


class RawEvidenceState:
    """Project a phrase ledger into the common retrieval/evidence interface."""

    def __init__(self, profile: dict) -> None:
        self.profile = deepcopy(profile)
        self.raw = PhraseState(user_profile=deepcopy(profile))
        self.preferences: list[Preference] = []
        self.turn = 0

    @staticmethod
    def _key(preference: Preference) -> tuple:
        return (preference.attribute, preference.value, preference.polarity,
                preference.source_turn, preference.source_text, preference.scope,
                preference.alternative_group, preference.hard, preference.confidence)

    def update(self, user_message: str, turn: int) -> None:
        self.raw.observe(user_message, turn)
        self.turn = self.raw.last_turn
        previous = {self._key(preference): preference for preference in self.preferences}
        for preference in self.preferences:
            preference.active = False
        for item in self.raw.evidence:
            text = GENERIC_PREFIX_RE.sub("", item.text).strip()
            if not text:
                continue
            polarity = -1 if item.source == "exclusion" else 1
            branches = alternative_values(text) if polarity == 1 else (text,)
            identity = f"{item.turn}:{item.source}:{item.text.casefold()}"
            group = "raw:" + sha256(identity.encode()).hexdigest() if len(branches) > 1 else None
            for branch in branches:
                scope = component_scope(branch)
                value = component_value(branch) if scope else branch
                preference = Preference(
                    attribute="category" if item.source == "category" else item.attribute or "other",
                    value=value, source_turn=item.turn, source_text=item.text,
                    hard=item.source in {"hard_constraint", "override", "exclusion"},
                    polarity=polarity, confidence=min(1.0, item.weight / 2.5),
                    alternative_group=group, scope=scope, source_kind="raw:" + item.source,
                )
                prior = previous.get(self._key(preference))
                if prior is not None:
                    prior.active = True
                else:
                    self.preferences.append(preference)

    def active_preferences(self) -> list[Preference]:
        return [preference for preference in self.preferences if preference.active]

    def query(self) -> str:
        # The phrase ledger, not messages or retired preferences, owns this query.
        return " ".join(GENERIC_PREFIX_RE.sub("", item.text).strip()
                        for item in self.raw.evidence if item.source != "exclusion")

    def source_alias_query(self) -> str:
        return ""

    def record_question(self, attribute: str | None, goal: str | None = None) -> None:
        if attribute is not None:
            self.raw.record_question(attribute)

    def semantic_signature(self) -> tuple[tuple, ...]:
        return tuple(sorted((preference.attribute, preference.value, preference.polarity,
                             preference.scope or "", preference.hard)
                            for preference in self.active_preferences()))
