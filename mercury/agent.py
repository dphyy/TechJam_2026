from __future__ import annotations

import logging
import math
import time
from collections import OrderedDict
from pathlib import Path

from mercury.catalog import Catalog
from mercury.config import Config
from mercury.policy import choose_policy
from mercury.ranking import rank_candidates, rank_constraints
from mercury.retrieval import SparseIndex, fuse_routes, terms
from mercury.state import SessionState
from mercury.types import Candidate, Preference


LOGGER = logging.getLogger(__name__)


def _valid_ranking(original: list[Candidate], ranked: object) -> list[Candidate]:
    if not isinstance(ranked, list) or len(ranked) != len(original):
        raise ValueError("Ranker returned invalid candidate count/type")
    if any(not isinstance(item, Candidate) or not math.isfinite(item.score) for item in ranked):
        raise ValueError("Ranker returned invalid candidate/score")
    expected = {item.product.parent_asin for item in original}
    actual = [item.product.parent_asin for item in ranked]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("Ranker changed candidate identities")
    return ranked


def _apply_constraints(candidates: list[Candidate], preferences: list[Preference],
                       fallbacks: list[str]) -> list[Candidate]:
    try:
        return _valid_ranking(candidates, rank_constraints(candidates, preferences))
    except (RuntimeError, ValueError, TypeError) as error:
        if "constraints" not in fallbacks:
            fallbacks.append("constraints")
        LOGGER.warning("Constraint ranking failed: %s", type(error).__name__)
        return candidates


class Agent:
    """Offline shopping search with bounded, reversible session-local evidence."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl", config: Config | None = None) -> None:
        self.config = config or Config()
        self.catalog = Catalog(catalog_path)
        self.sparse = SparseIndex(self.catalog)
        self.sessions: OrderedDict[str, SessionState] = OrderedDict()
        self._cache: dict[str, tuple] = {}
        self._abstentions: dict[str, int] = {}
        self.dense = None
        self.reranker = None
        self.contrast = None
        self.startup_fallbacks: dict[str, str] = {}
        self.last_diagnostics: dict = {}
        artifacts = Path(self.config.artifact_dir)
        if self.config.dense:
            try:
                from mercury.neural import DenseIndex
                self.dense = DenseIndex(self.catalog, artifacts, self.config.device, self.config.threads)
            except (OSError, ValueError, KeyError, ImportError, RuntimeError, TypeError, AttributeError) as error:
                self._startup_failure("dense", error)
        if self.config.neural_rerank:
            try:
                from mercury.neural import NeuralRanker
                self.reranker = NeuralRanker(artifacts, self.config.device, self.config.threads)
            except (OSError, ValueError, KeyError, ImportError, RuntimeError, TypeError, AttributeError) as error:
                self._startup_failure("neural_rerank", error)
        if self.config.contrast:
            try:
                from mercury.contrast import ContrastIndex
                self.contrast = ContrastIndex(self.catalog, artifacts / "contrast")
            except (OSError, ValueError, KeyError, ImportError, RuntimeError, TypeError, AttributeError) as error:
                self._startup_failure("contrast", error)

    def _startup_failure(self, component: str, error: Exception) -> None:
        self.startup_fallbacks[component] = f"{type(error).__name__}: {error}"
        LOGGER.warning("%s unavailable; using sparse fallback (%s)", component, type(error).__name__)

    def reset(self, session_id: str, user_profile: dict) -> None:
        if not isinstance(session_id, str) or not isinstance(user_profile, dict):
            raise ValueError("reset requires a string session ID and profile object")
        self.sessions[session_id] = SessionState(user_profile, self.config.state_mode, self.config.alternatives_mode)
        self.sessions.move_to_end(session_id)
        self._cache.pop(session_id, None)
        self._abstentions[session_id] = 0
        while len(self.sessions) > self.config.max_sessions:
            old, _ = self.sessions.popitem(last=False)
            self._cache.pop(old, None)
            self._abstentions.pop(old, None)

    def _retrieve(self, query: str, state: SessionState, fallbacks: list[str]) -> tuple[list[Candidate], dict[str, list[str]]]:
        config = self.config
        routes = {"sparse": self.sparse.search(query, config.sparse_limit)}
        weights = {"sparse": 1.0}
        categories = [p.value for p in state.active_preferences()
                      if p.attribute == "category" and p.polarity == 1]
        if categories:
            routes["scoped"] = self.sparse.search(query, config.sparse_limit, categories)
            weights = {"sparse": 0.7, "scoped": 0.3}
        if self.dense is not None:
            try:
                routes["dense"] = self.dense.search(query, config.dense_limit)
                weights = {name: value * (1.0 - config.dense_weight) for name, value in weights.items()}
                weights["dense"] = config.dense_weight
            except (RuntimeError, ValueError, OSError, TimeoutError) as error:
                fallbacks.append("dense")
                LOGGER.warning("Dense retrieval failed: %s", type(error).__name__)
        fused = fuse_routes(routes, weights)
        candidates = [Candidate(self.catalog.by_id[identifier], score, parts)
                      for identifier, score, parts in fused if identifier in self.catalog.by_id]
        if not candidates:
            # No metadata match is not evidence that the catalog is empty.
            candidates = [Candidate(product, 0.0) for product in self.catalog.products[:config.candidate_limit]]
            fallbacks.append("no_matches")
        return candidates, routes

    def _tokens(self) -> int:
        return sum(getattr(model, "prompt_tokens", 0) for model in (self.dense, self.reranker) if model is not None)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        if session_id not in self.sessions:
            raise RuntimeError("reset must be called before respond")
        if not isinstance(user_message, str) or type(turn) is not int or not 1 <= turn <= 10:
            raise ValueError("respond requires text and a turn in [1, 10]")
        user_message = user_message[:8000]
        started = time.perf_counter()
        tokens_before = self._tokens()
        top_k = max(0, min(10, top_k)) if type(top_k) is int else 0
        state = self.sessions[session_id]
        self.sessions.move_to_end(session_id)
        state.update(user_message, turn)
        query = state.query()
        if not query and turn == 1:
            query = " ".join(terms(user_message))
        cache_key = (query, state.revision)
        fallbacks = list(self.startup_fallbacks)
        cached = self._cache.get(session_id)
        cache_hit = cached is not None and cached[0] == cache_key
        if cache_hit:
            candidates, routes, retrieved_ids, cached_fallbacks = cached[1:]
            fallbacks = list(cached_fallbacks)
        else:
            candidates, routes = self._retrieve(query, state, fallbacks)
            retrieved_ids = [item.product.parent_asin for item in candidates]
            preferences = state.active_preferences()
            if self.config.evidence_ranking:
                try:
                    candidates = _valid_ranking(candidates, rank_candidates(candidates, preferences))
                except (RuntimeError, ValueError, TypeError) as error:
                    fallbacks.append("ranking")
                    LOGGER.warning("Evidence ranking failed: %s", type(error).__name__)
            candidates = _apply_constraints(candidates, preferences, fallbacks)
            candidates = candidates[:self.config.candidate_limit]
            if self.contrast is not None:
                try:
                    candidates = _valid_ranking(candidates, self.contrast.rank(candidates, preferences, self.config.contrast_weight))
                except (RuntimeError, ValueError, KeyError, TypeError) as error:
                    fallbacks.append("contrast")
                    LOGGER.warning("Contrast ranking failed: %s", type(error).__name__)
            if self.reranker is not None:
                try:
                    candidates = _valid_ranking(candidates, self.reranker.rank(query, candidates, self.config.rerank_limit, self.config.neural_weight))
                except (RuntimeError, ValueError, TypeError, TimeoutError) as error:
                    fallbacks.append("neural_rerank")
                    LOGGER.warning("Neural ranking failed: %s", type(error).__name__)
            candidates = _apply_constraints(candidates, preferences, fallbacks)
            self._cache[session_id] = (cache_key, candidates, routes, retrieved_ids, tuple(fallbacks))
        decision = choose_policy(state, candidates, self.config, turn, top_k, self._abstentions[session_id])
        state.record_question(decision.ask_attribute)
        limit = min(top_k, max(0, decision.slate_size))
        ranked = list(dict.fromkeys(item.product.parent_asin for item in candidates))[:limit]
        self._abstentions[session_id] = 0 if ranked else self._abstentions[session_id] + 1
        self.last_diagnostics = {
            "query": query, "revision": state.revision, "cache_hit": cache_hit,
            "preferences": [{"attribute": p.attribute, "value": p.value, "polarity": p.polarity,
                             "source_turn": p.source_turn, "hard": p.hard,
                             **({"alternative_group": p.alternative_group} if p.alternative_group is not None else {})}
                            for p in state.active_preferences()],
            "unsupported_alternatives": [dict(item) for item in state.unsupported_alternatives],
            "routes": routes, "retrieved_ids": retrieved_ids,
            "ranked_ids": [item.product.parent_asin for item in candidates],
            "constraint_penalties": {item.product.parent_asin: item.route_scores.get("constraint_penalty", 0.0)
                                     for item in candidates},
            "fallbacks": fallbacks, "policy": decision.diagnostics,
            "latency_seconds": time.perf_counter() - started,
        }
        return {"message": decision.message, "ask_attribute": decision.ask_attribute,
                "recommendations": [{"parent_asin": identifier} for identifier in ranked],
                "usage": {"prompt_tokens": self._tokens() - tokens_before, "completion_tokens": 0}}

    def close(self) -> None:
        self.sparse.close()
        self.sessions.clear()
        self._cache.clear()
        self._abstentions.clear()
