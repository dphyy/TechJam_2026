from __future__ import annotations

import hashlib
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from mercury.lexical.dialogue import SessionState

from .index import CatalogIndex
from .planner import choose_question
from .ranking import compile_requirements, rank_candidates


@dataclass(frozen=True, slots=True)
class AgentConfig:
    candidate_limit: int = 2000
    lexical_limit: int = 600
    slate_size: int = 10
    max_sessions: int = 256
    question_policy: str = "adaptive"

    def __post_init__(self) -> None:
        for name in ("candidate_limit", "lexical_limit", "max_sessions"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.slate_size) is not int or not 1 <= self.slate_size <= 10:
            raise ValueError("slate_size must be an integer in [1, 10]")
        if self.candidate_limit < self.slate_size:
            raise ValueError("candidate_limit must cover the requested slate size")
        if self.question_policy not in {"adaptive", "other", "none"}:
            raise ValueError("question_policy must be adaptive, other, or none")


FULL_WIDTH_CONFIG = AgentConfig()


class Agent:
    """Bounded conversational search with independently ordered catalog evidence."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl", *,
                 config: AgentConfig = FULL_WIDTH_CONFIG) -> None:
        self.config = config
        self.index = CatalogIndex(catalog_path)
        self._sessions: OrderedDict[str, SessionState] = OrderedDict()
        self._responses: dict[str, tuple[tuple[int, str, int], dict]] = {}
        self._diagnostics: dict[str, dict] = {}
        self._closed = False

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("agent is closed")

    def _drop_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._responses.pop(session_id, None)
        self._diagnostics.pop(session_id, None)

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._require_open()
        if not isinstance(session_id, str) or not session_id or not isinstance(user_profile, dict):
            raise ValueError("reset requires a nonempty session ID and profile object")
        state = SessionState(user_profile=deepcopy(user_profile))
        self._drop_session(session_id)
        self._sessions[session_id] = state
        while len(self._sessions) > self.config.max_sessions:
            self._drop_session(next(iter(self._sessions)))

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        self._require_open()
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("respond requires a nonempty session ID")
        if not isinstance(user_message, str) or type(turn) is not int or not 1 <= turn <= 10:
            raise ValueError("respond requires text and a turn in [1, 10]")
        if type(top_k) is not int or not 1 <= top_k <= 10:
            raise ValueError("top_k must be an integer in [1, 10]")
        state = self._sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before respond")
        # Hash the complete request before bounding parser input: two different
        # long messages must never be accepted as an identical retry.
        request = (turn, hashlib.sha256(user_message.encode("utf-8")).hexdigest(), top_k)
        previous = self._responses.get(session_id)
        if previous is not None and previous[0] == request:
            self._sessions.move_to_end(session_id)
            return deepcopy(previous[1])
        if turn <= state.last_turn:
            raise ValueError("turn must advance; conflicting or stale retries are not accepted")
        working = deepcopy(state)
        working.observe(user_message[:8000], turn)
        requirements = compile_requirements(working)
        positive = [branch.phrase for item in requirements if not item.exclude for branch in item.branches]
        query = " ".join((working.category_text, *positive))
        candidate_ids, category_ids, diagnostics = self.index.candidates(
            working.category_text, query, tuple(positive), self.config.candidate_limit,
            self.config.lexical_limit,
        )
        ranked = rank_candidates(self.index, candidate_ids, category_ids, requirements)
        question = choose_question(working, self.index, ranked, turn, self.config.question_policy)
        if question.attribute is not None:
            working.record_question(question.attribute)
        width = min(top_k, self.config.slate_size)
        response = {
            "message": question.message,
            "ask_attribute": question.attribute,
            "recommendations": [{"parent_asin": item.identifier, "score": round(1 / (rank + 1), 6)}
                                for rank, item in enumerate(ranked[:width])],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
        diagnostics.update({
            "catalog_sha256": self.index.sha256, "catalog_size": len(self.index.products),
            "candidate_ids": candidate_ids, "raw_ranked_ids": [item.identifier for item in ranked],
            "presentation_width": width, "question_uncertainty": question.uncertainty,
            "requirement_count": len(requirements),
            "top_evidence": [{"parent_asin": item.identifier, **item.evidence} for item in ranked[:10]],
        })
        self._sessions[session_id] = working
        self._sessions.move_to_end(session_id)
        self._responses[session_id] = (request, deepcopy(response))
        self._diagnostics[session_id] = diagnostics
        return response

    def diagnostics(self, session_id: str) -> dict:
        self._require_open()
        return deepcopy(self._diagnostics.get(session_id, {}))

    def product(self, identifier: str) -> dict | None:
        self._require_open()
        product = self.index.products.get(identifier)
        return product.raw() if product is not None else None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._sessions.clear()
        self._responses.clear()
        self._diagnostics.clear()
        self.index.close()
