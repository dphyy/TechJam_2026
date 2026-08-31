"""Catalog-only agent with independent admission routes and open questions."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING

from mercury.catalog import Catalog
from mercury.multiview.retrieval import Config, MultiViewIndex, explain
from mercury.state import SessionState

if TYPE_CHECKING:
    from mercury.multiview.raw_state import RawEvidenceState


class Agent:
    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl",
                 config: Config | None = None) -> None:
        self.config = config or Config()
        self.catalog = Catalog(catalog_path)
        self.index = MultiViewIndex(self.catalog, self.config)
        self.sessions: OrderedDict[str, SessionState | RawEvidenceState] = OrderedDict()
        self._responses: dict[str, tuple[tuple, dict, dict]] = {}
        self.diagnostics: dict = {}

    def close(self) -> None:
        self.index.close()
        self.sessions.clear()
        self._responses.clear()
        self.diagnostics = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        if not isinstance(session_id, str) or not session_id or not isinstance(user_profile, dict):
            raise ValueError("reset requires a nonempty session ID and profile object")
        # Profiles are retained as priors, never interpreted as live constraints.
        profile = deepcopy(user_profile)
        if self.config.state_mode == "raw":
            from mercury.multiview.raw_state import RawEvidenceState

            self.sessions[session_id] = RawEvidenceState(profile)
        else:
            self.sessions[session_id] = SessionState(
                profile, mode="ledger", alternatives_mode="grouped", scoped_preferences=True,
            )
        self.sessions.move_to_end(session_id)
        self._responses.pop(session_id, None)
        self.diagnostics = {}
        while len(self.sessions) > self.config.max_sessions:
            retired, _ = self.sessions.popitem(last=False)
            self._responses.pop(retired, None)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        if session_id not in self.sessions:
            raise RuntimeError("reset must be called before respond")
        if not isinstance(user_message, str) or type(turn) is not int or not 1 <= turn <= 10:
            raise ValueError("respond requires text and a turn in [1, 10]")
        top_k = max(0, min(10, top_k)) if type(top_k) is int else 0
        user_message = user_message[:8000]
        request = (turn, user_message, top_k)
        state = self.sessions[session_id]
        cached = self._responses.get(session_id)
        if cached and cached[0] == request:
            self.sessions.move_to_end(session_id)
            self.diagnostics = deepcopy(cached[2])
            return deepcopy(cached[1])
        if turn <= state.turn:
            raise ValueError("turns must advance; only an identical latest request may be retried")
        state.update(user_message, turn)
        self.sessions.move_to_end(session_id)
        result = self.index.search(state)
        selected = result.candidates[:top_k]
        ask_attribute = "other" if turn < 10 else None
        message = (
            "What other features or preferences would help narrow these choices?"
            if ask_attribute else "Here are the closest catalog matches to your preferences."
        )
        state.record_question(ask_attribute, "open_preferences" if ask_attribute else None)
        response = {
            "message": message, "ask_attribute": ask_attribute,
            "recommendations": [{"parent_asin": item.product.parent_asin, "score": item.score}
                                for item in selected],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
        self.diagnostics = {
            "backend": "multiview", "catalog_sha256": self.catalog.sha256,
            "config": asdict(self.config),
            "runtime": {"requested_neural": False, "loaded_neural": False,
                        "effective_output_cap": 10, "requested_fullwidth": self.config.fullwidth},
            "candidate_ids": [item.product.parent_asin for item in result.candidates],
            "routes": result.routes, "queries": result.queries,
            "constraints_omitted_from_admission": result.constraints_omitted,
            "active_preferences": [asdict(item) for item in state.active_preferences()],
            "retired_preferences": [asdict(item) for item in state.preferences if not item.active],
            "evidence": {item.product.parent_asin: explain(item.product, state.active_preferences())
                         for item in selected},
        }
        self._responses[session_id] = (request, deepcopy(response), deepcopy(self.diagnostics))
        return response


def make_agent(catalog_path: str | Path, *, fullwidth: bool = True,
               config: Config | None = None) -> Agent:
    """Both control settings expose the same ranked ten without policy mutation."""
    return Agent(catalog_path, replace(config or Config(), fullwidth=fullwidth))
