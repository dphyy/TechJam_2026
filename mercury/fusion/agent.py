"""A single pipeline with an explicit off/on candidate admission arm."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

from mercury.fusion.retrieval import FusionCatalogSearch
from mercury.lexical.agent import Agent as LexicalAgent
from mercury.lexical.config import DEFAULT_AGENT_CONFIG
from mercury.lexical.retrieval import FEATURE_CACHE_SIZE


@dataclass(frozen=True, slots=True)
class Config:
    additional_admission: bool = True
    fullwidth: bool = False

    def __post_init__(self) -> None:
        if type(self.additional_admission) is not bool or type(self.fullwidth) is not bool:
            raise ValueError("admission and fullwidth controls must be boolean")


class Agent(LexicalAgent):
    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl", config: Config | None = None,
                 *, feature_cache_size: int = FEATURE_CACHE_SIZE,
                 catalog_index_path: str | Path | None = None) -> None:
        self.fusion_config = config or Config()
        self.diagnostics: dict = {}
        self._traces: dict[str, dict] = {}
        super().__init__(
            catalog_path, feature_cache_size=feature_cache_size,
            config=replace(DEFAULT_AGENT_CONFIG, full_width=self.fusion_config.fullwidth),
            catalog_index_path=catalog_index_path,
            search_factory=partial(FusionCatalogSearch,
                                   additional_admission=self.fusion_config.additional_admission),
        )

    def reset(self, session_id: str, user_profile: dict) -> None:
        super().reset(session_id, user_profile)
        self._traces = {key: value for key, value in self._traces.items()
                        if key in self._sessions and key != session_id}
        self.diagnostics = {}

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        previous = self._responses.get(session_id)
        response = super().respond(session_id, user_message, turn, top_k)
        if previous is not None and previous is self._responses.get(session_id):
            self.diagnostics = deepcopy(self._traces[session_id])
        else:
            self.diagnostics = deepcopy(self.search.diagnostics)
            self.diagnostics["stage_ids"]["presented"] = [item["parent_asin"]
                                                         for item in response["recommendations"]]
            self.diagnostics["fullwidth"] = self.fusion_config.fullwidth
            self._traces[session_id] = deepcopy(self.diagnostics)
        return response

    def close(self) -> None:
        self._traces.clear()
        self.diagnostics.clear()
        super().close()


def make_agent(catalog_path: str | Path, *, additional_admission: bool = True,
               fullwidth: bool = False) -> Agent:
    return Agent(catalog_path, Config(additional_admission, fullwidth))
