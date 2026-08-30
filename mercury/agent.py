from __future__ import annotations

import logging
import math
import time
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

from mercury.admission import AdmissionModel, score_all_candidates, select_rerank_prefix
from mercury.cascade import decide_compute_cascade
from mercury.catalog import Catalog
from mercury.config import Config
from mercury.intent import IntentWeights, decide_intent
from mercury.neural import fuse_neural_logits
from mercury.hypotheses import build_intent_hypotheses
from mercury.planning import build_retrieval_plan
from mercury.policy import choose_policy
from mercury.profile import distill_profile, rank_profile_prior
from mercury.ranking import (rank_candidates, rank_composition_evidence, rank_constraints,
                             rank_product_compatibility, rank_role_evidence, rank_soft_prices)
from mercury.retrieval import SparseIndex, fuse_routes, terms
from mercury.state import SessionState
from mercury.sufficiency import decide_retrieval_sufficiency
from mercury.types import Candidate, ComputeCascadeDecision, Preference, RetrievalPlan
from mercury.vocabulary import CatalogVocabulary


LOGGER = logging.getLogger(__name__)


def _route_overlap(routes: dict[str, list[str]]) -> dict[str, float]:
    """Pairwise Jaccard overlap is target-independent route agreement evidence."""
    names = sorted(routes)
    result = {}
    for index, first in enumerate(names):
        left = set(routes[first])
        for second in names[index + 1:]:
            right = set(routes[second])
            union = left | right
            result[f"{first}:{second}"] = len(left & right) / len(union) if union else 1.0
    return result


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


def _neural_score_summary(candidates: list[Candidate]) -> dict:
    logits = [item.route_scores.get("neural_logit") for item in candidates
              if type(item.route_scores.get("neural_logit")) in (int, float)
              and math.isfinite(item.route_scores["neural_logit"])]
    ordered = sorted(logits, reverse=True)
    weights = sorted({item.route_scores.get("neural_fusion_weight") for item in candidates
                      if type(item.route_scores.get("neural_fusion_weight")) in (int, float)})
    return {
        "scored_pairs": len(ordered),
        "top_logit": ordered[0] if ordered else None,
        "second_logit": ordered[1] if len(ordered) > 1 else None,
        "logit_margin": ordered[0] - ordered[1] if len(ordered) > 1 else None,
        "logit_range": ordered[0] - ordered[-1] if len(ordered) > 1 else None,
        "fusion_weight": weights[0] if len(weights) == 1 else None,
    }


def _apply_constraints(candidates: list[Candidate], preferences: list[Preference],
                       fallbacks: list[str]) -> list[Candidate]:
    try:
        return _valid_ranking(candidates, rank_constraints(candidates, preferences))
    except (RuntimeError, ValueError, TypeError) as error:
        if "constraints" not in fallbacks:
            fallbacks.append("constraints")
        LOGGER.warning("Constraint ranking failed: %s", type(error).__name__)
        return candidates


def _apply_product_guard(candidates: list[Candidate], preferences: list[Preference],
                         fallbacks: list[str]) -> list[Candidate]:
    try:
        return _valid_ranking(candidates, rank_product_compatibility(candidates, preferences))
    except (RuntimeError, ValueError, TypeError) as error:
        if "product_guard" not in fallbacks:
            fallbacks.append("product_guard")
        LOGGER.warning("Product guard failed: %s", type(error).__name__)
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
        self._deferred_turns: dict[str, int] = {}
        self._last_candidates: dict[str, list[Candidate]] = {}
        self._cascade_counts: dict[str, int] = {}
        self._last_neural_margin: dict[str, float | None] = {}
        self._pages: dict[str, int] = {}
        self._last_ranked: dict[str, list[str]] = {}
        self._shown_ids: dict[str, set[str]] = {}
        self._frontier_revision: dict[str, int] = {}
        self._frontier_base: dict[str, list[Candidate]] = {}
        self._frontier_logits: dict[str, dict[str, float]] = {}
        self._page_rerank_revision: dict[str, int] = {}
        self._page_rerank_logits: dict[str, dict[str, float]] = {}
        self._page_rerank_batches: dict[str, int] = {}
        self._page_rerank_pairs: dict[str, int] = {}
        self.dense = None
        self.reranker = None
        self.contrast = None
        self.admission_model = None
        self.catalog_vocabulary = None
        self.startup_fallbacks: dict[str, str] = {}
        self._rerank_cost: float | None = None
        self.last_diagnostics: dict = {}
        artifacts = Path(self.config.artifact_dir)
        if self.config.catalog_vocabulary:
            try:
                self.catalog_vocabulary = CatalogVocabulary(
                    self.config.catalog_vocabulary_path, self.catalog.sha256,
                )
            except (OSError, ValueError, KeyError, TypeError) as error:
                self._startup_failure("catalog_vocabulary", error)
        if self.config.rerank_admission == "linear":
            try:
                self.admission_model = AdmissionModel.load(
                    self.config.admission_model_path, self.catalog.sha256,
                )
            except (OSError, ValueError, KeyError, TypeError) as error:
                self._startup_failure("admission_model", error)
        if self.config.dense:
            try:
                from mercury.neural import DenseIndex
                self.dense = DenseIndex(self.catalog, artifacts, self.config.device, self.config.threads)
            except (OSError, ValueError, KeyError, ImportError, RuntimeError, TypeError, AttributeError) as error:
                self._startup_failure("dense", error)
        if self.config.neural_rerank:
            try:
                from mercury.neural import NeuralRanker
                cache_options = (
                    {"cache_capacity": self.config.neural_logit_cache_size}
                    if self.config.neural_logit_cache else {}
                )
                if self.config.neural_batch_size != 16:
                    cache_options["batch_size"] = self.config.neural_batch_size
                self.reranker = NeuralRanker(artifacts, self.config.device, self.config.threads,
                                             self.config.reranker_model, **cache_options)
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
        if not isinstance(session_id, str) or not session_id or not isinstance(user_profile, dict):
            raise ValueError("reset requires a nonempty string session ID and profile object")
        self.sessions[session_id] = SessionState(
            user_profile, self.config.state_mode, self.config.alternatives_mode,
            self.config.scoped_preferences, self.catalog_vocabulary,
        )
        self.sessions.move_to_end(session_id)
        self._cache.pop(session_id, None)
        self._abstentions[session_id] = 0
        self._deferred_turns[session_id] = 0
        self._last_candidates.pop(session_id, None)
        self._cascade_counts[session_id] = 0
        self._last_neural_margin[session_id] = None
        self._pages.pop(session_id, None)
        self._last_ranked.pop(session_id, None)
        self._shown_ids[session_id] = set()
        self._frontier_revision.pop(session_id, None)
        self._frontier_base.pop(session_id, None)
        self._frontier_logits.pop(session_id, None)
        self._page_rerank_revision.pop(session_id, None)
        self._page_rerank_logits.pop(session_id, None)
        self._page_rerank_batches[session_id] = 0
        self._page_rerank_pairs[session_id] = 0
        while len(self.sessions) > self.config.max_sessions:
            old, _ = self.sessions.popitem(last=False)
            self._cache.pop(old, None)
            self._abstentions.pop(old, None)
            self._deferred_turns.pop(old, None)
            self._last_candidates.pop(old, None)
            self._cascade_counts.pop(old, None)
            self._last_neural_margin.pop(old, None)
            self._pages.pop(old, None)
            self._last_ranked.pop(old, None)
            self._shown_ids.pop(old, None)
            self._frontier_revision.pop(old, None)
            self._frontier_base.pop(old, None)
            self._frontier_logits.pop(old, None)
            self._page_rerank_revision.pop(old, None)
            self._page_rerank_logits.pop(old, None)
            self._page_rerank_batches.pop(old, None)
            self._page_rerank_pairs.pop(old, None)

    def _retrieve(self, plan: RetrievalPlan, state: SessionState, fallbacks: list[str],
                  source_alias_query: str = ""
                  ) -> tuple[list[Candidate], dict[str, list[str]], dict[str, float]]:
        config = self.config
        if config.multi_hypothesis_retrieval:
            return self._retrieve_hypotheses(plan, fallbacks)
        query = plan.lexical_query
        preferences = state.active_preferences()
        categories = [p.value for p in preferences if p.attribute == "category" and p.polarity == 1]
        requirements = [p.value for p in preferences
                        if p.polarity == 1 and p.attribute not in {"category", "budget"} and p.depends_on is None]

        def fielded(limit: int) -> list[str]:
            if categories or requirements:
                return self.sparse.search_factored(categories, requirements, limit)
            return self.sparse.search(query, limit)

        if config.retrieval_mode == "broad":
            routes = {"sparse": self.sparse.search(query, config.sparse_limit)}
            weights = {"sparse": 1.0}
            if categories:
                routes["scoped"] = self.sparse.search(query, config.sparse_limit, categories)
                weights = {"sparse": 0.7, "scoped": 0.3}
        elif config.retrieval_mode == "field_union":
            route_limit = max(1, config.sparse_limit // 2)
            routes = {
                "sparse": self.sparse.search(query, route_limit),
                "fielded": fielded(route_limit),
            }
            weights = {"sparse": 0.5, "fielded": 0.5}
            if categories:
                routes["scoped"] = self.sparse.search(query, route_limit, categories)
                weights = {"sparse": 0.35, "scoped": 0.15, "fielded": 0.5}
        else:
            routes = {"fielded": fielded(config.sparse_limit)}
            weights = {"fielded": 1.0}
        if config.routed_retrieval and config.retrieval_mode == "broad":
            if plan.mode == "buying" and categories:
                weights = {"sparse": 1.0 - config.buying_scoped_weight,
                           "scoped": config.buying_scoped_weight}
            elif plan.mode == "browsing" and plan.use_case:
                routes["scenario_sparse"] = self.sparse.search(" ".join(plan.use_case), config.sparse_limit)
                weights = {"sparse": 1.0 - config.browsing_scenario_weight,
                           "scenario_sparse": config.browsing_scenario_weight}
            elif plan.mode == "mixed" and categories:
                weights = {"sparse": 1.0 - config.mixed_scoped_weight,
                           "scoped": config.mixed_scoped_weight}
        if source_alias_query:
            routes["source_alias"] = self.sparse.search(source_alias_query, config.sparse_limit)
            weights = {name: weight * 0.85 for name, weight in weights.items()}
            weights["source_alias"] = 0.15
        if self.dense is not None:
            try:
                semantic_query = plan.semantic_queries[0] if config.routed_retrieval and plan.semantic_queries else query
                routes["dense"] = self.dense.search(semantic_query, config.dense_limit)
                dense_weight = config.dense_weight
                if config.routed_retrieval:
                    dense_weight = {
                        "buying": config.buying_dense_weight,
                        "browsing": config.browsing_dense_weight,
                        "mixed": config.mixed_dense_weight,
                    }[plan.mode]
                weights = {name: value * (1.0 - dense_weight) for name, value in weights.items()}
                weights["dense"] = dense_weight
            except (RuntimeError, ValueError, OSError, TimeoutError) as error:
                fallbacks.append("dense")
                LOGGER.warning("Dense retrieval failed: %s", type(error).__name__)
        fused = fuse_routes(routes, weights)
        candidates = [Candidate(self.catalog.by_id[identifier], score, parts)
                      for identifier, score, parts in fused if identifier in self.catalog.by_id]
        if len(self.catalog.products) <= max(10, config.candidate_limit):
            seen = {candidate.product.parent_asin for candidate in candidates}
            candidates.extend(Candidate(product, 0.0, {"tiny_catalog_tail": 0.0})
                              for product in self.catalog.products if product.parent_asin not in seen)
        if not candidates:
            # No metadata match is not evidence that the catalog is empty.
            candidates = [Candidate(product, 0.0) for product in self.catalog.products[:config.candidate_limit]]
            fallbacks.append("no_matches")
        return candidates, routes, weights

    def _retrieve_hypotheses(self, plan: RetrievalPlan, fallbacks: list[str]
                             ) -> tuple[list[Candidate], dict[str, list[str]], dict[str, float]]:
        hypotheses = plan.hypotheses[:self.config.max_intent_hypotheses]
        if not hypotheses:
            fallbacks.append("no_intent_hypothesis")
            routes = {"hypothesis_0": self.sparse.search(
                plan.lexical_query, self.config.hypothesis_candidate_budget,
            )}
            weights = {"hypothesis_0": 1.0}
            fused = fuse_routes(routes, weights)
            candidates = [Candidate(self.catalog.by_id[identifier], score, parts)
                          for identifier, score, parts in fused if identifier in self.catalog.by_id]
            return self._hypothesis_tail(candidates, routes, weights, fallbacks)
        quotient, remainder = divmod(self.config.hypothesis_candidate_budget, len(hypotheses))
        routes = {}
        for index, hypothesis in enumerate(hypotheses):
            limit = quotient + (1 if index < remainder else 0)
            routes[f"hypothesis_{index}"] = self.sparse.search(
                hypothesis.query, limit, list(hypothesis.object_types) or None,
            )
        weights = {name: 1.0 / len(routes) for name in routes}
        fused = fuse_routes(routes, weights)
        candidates = [Candidate(self.catalog.by_id[identifier], score, parts)
                      for identifier, score, parts in fused if identifier in self.catalog.by_id]
        return self._hypothesis_tail(candidates, routes, weights, fallbacks)

    def _hypothesis_tail(self, candidates: list[Candidate], routes: dict[str, list[str]],
                         weights: dict[str, float], fallbacks: list[str]
                         ) -> tuple[list[Candidate], dict[str, list[str]], dict[str, float]]:
        if len(self.catalog.products) <= max(10, self.config.candidate_limit):
            seen = {candidate.product.parent_asin for candidate in candidates}
            candidates.extend(Candidate(product, 0.0, {"tiny_catalog_tail": 0.0})
                              for product in self.catalog.products if product.parent_asin not in seen)
        if not candidates:
            candidates = [Candidate(product, 0.0)
                          for product in self.catalog.products[:self.config.hypothesis_candidate_budget]]
            fallbacks.append("no_matches")
        return candidates, routes, weights

    def _slate_page(self, session_id: str, ordered: list[str], turn: int, limit: int,
                    reset_page: bool = False) -> int:
        """Advance past an unchanged slate; any ranking change returns to the top.

        Re-serving an identical slate cannot score, because the session would
        already have ended had the wanted product been in it. A detected intent
        override can explicitly reset paging because earlier slates may have
        contained a target that was not eligible to score yet.
        """
        first_turn = self.config.slate_paging_first_turn
        previous = self._last_ranked.get(session_id)
        page = self._pages.get(session_id, 0)
        if reset_page:
            page = 0
        elif previous is not None and ordered == previous:
            if first_turn and turn >= first_turn:
                page += 1
        else:
            page = 0
        page = min(page, (len(ordered) - 1) // limit) if limit and ordered else 0
        self._pages[session_id] = page
        self._last_ranked[session_id] = ordered
        return page

    def _seen_aware_selection(self, session_id: str, ordered: list[str], limit: int,
                              reset_seen: bool = False) -> tuple[int, list[str], int]:
        """Select the strongest unseen products, clearing history only on override."""
        seen = self._shown_ids.setdefault(session_id, set())
        if reset_seen:
            seen.clear()
            self._pages[session_id] = 0
        page = self._pages.get(session_id, 0)
        if seen and not reset_seen:
            page += 1
        unseen = [identifier for identifier in ordered if identifier not in seen]
        selected = unseen[:limit]
        if len(selected) < limit:
            selected_set = set(selected)
            selected.extend([
                identifier for identifier in ordered if identifier not in selected_set
            ][:limit - len(selected)])
        new_count = sum(identifier not in seen for identifier in selected)
        seen.update(selected)
        self._pages[session_id] = page
        self._last_ranked[session_id] = ordered
        return page, selected, new_count

    def _affordable_rerank_limit(self, elapsed: float) -> int:
        """Reranking prefix this turn's remaining budget can pay for.

        Returns the configured limit when unbudgeted or before any cost is
        known. A timeout is scored as a miss, so an exhausted budget degrades
        to the ranking already computed instead of risking the whole session.
        """
        limit = self.config.rerank_limit
        budget = self.config.turn_budget_seconds
        if budget <= 0 or self._rerank_cost is None or self._rerank_cost <= 0:
            return limit
        remaining = budget - elapsed
        if remaining <= 0:
            return 0
        return max(0, min(limit, int((remaining + 1e-9) / self._rerank_cost)))

    def _record_rerank_cost(self, seconds: float, candidates: int) -> None:
        """Track observed seconds per reranked candidate on this machine."""
        if candidates < 1 or not math.isfinite(seconds) or seconds < 0:
            return
        measured = seconds / candidates
        self._rerank_cost = measured if self._rerank_cost is None else (self._rerank_cost + measured) / 2.0

    def _tokens(self) -> int:
        return sum(getattr(model, "prompt_tokens", 0) for model in (self.dense, self.reranker) if model is not None)

    def _neural_cache_stats(self) -> dict[str, int | bool]:
        empty = {
            "enabled": False, "capacity": 0, "size": 0, "hits": 0,
            "misses": 0, "evictions": 0, "evaluated_pairs": 0,
        }
        stats = getattr(self.reranker, "cache_stats", None)
        if not callable(stats):
            return empty
        try:
            snapshot = stats()
        except (RuntimeError, ValueError, TypeError, AttributeError):
            return empty
        return snapshot if isinstance(snapshot, dict) else empty

    def _minimal_probe(self, query: str, fallbacks: list[str]) -> tuple[
            list[Candidate], dict[str, list[str]], dict[str, float]]:
        identifiers = self.sparse.search(query, self.config.minimal_probe_limit)
        if not identifiers:
            identifiers = [product.parent_asin for product in self.catalog.products[:self.config.minimal_probe_limit]]
            fallbacks.append("minimal_probe_no_matches")
        routes = {"minimal_sparse_probe": identifiers}
        fused = fuse_routes(routes, {"minimal_sparse_probe": 1.0})
        candidates = [Candidate(self.catalog.by_id[identifier], score, parts)
                      for identifier, score, parts in fused if identifier in self.catalog.by_id]
        return candidates, routes, {"minimal_sparse_probe": 1.0}

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        if session_id not in self.sessions:
            raise RuntimeError("reset must be called before respond")
        if not isinstance(user_message, str) or type(turn) is not int or not 1 <= turn <= 10:
            raise ValueError("respond requires text and a turn in [1, 10]")
        user_message = user_message[:8000]
        started = time.perf_counter()
        tokens_before = self._tokens()
        neural_cache_before = self._neural_cache_stats()
        top_k = max(0, min(10, top_k)) if type(top_k) is int else 0
        state = self.sessions[session_id]
        self.sessions.move_to_end(session_id)
        state_started = time.perf_counter()
        state.update(user_message, turn)
        intent = decide_intent(
            state, user_message, self.config.router_buying_threshold,
            self.config.router_browsing_threshold, self.config.router_over_general_threshold,
            IntentWeights(
                object=self.config.intent_object_weight,
                slots=self.config.intent_slot_weight,
                hard=self.config.intent_hard_weight,
                buying_language=self.config.intent_buying_language_weight,
                browsing_language=self.config.intent_browsing_language_weight,
                use_case_without_object=self.config.intent_use_case_weight,
                unresolved=self.config.intent_unresolved_weight,
                sparse_request=self.config.intent_sparse_request_weight,
            ),
        )
        plan = build_retrieval_plan(state, intent)
        if self.config.multi_hypothesis_retrieval:
            plan = replace(plan, hypotheses=build_intent_hypotheses(
                plan, self.config.max_intent_hypotheses,
            ))
        component_latency = {"state_and_intent": time.perf_counter() - state_started}
        sufficiency = decide_retrieval_sufficiency(
            state, intent, plan, self.config, turn, self._deferred_turns[session_id],
        )
        if sufficiency.action == "retrieve":
            self._deferred_turns[session_id] = 0
        else:
            self._deferred_turns[session_id] += 1
        query = plan.lexical_query
        if not query and turn == 1:
            query = " ".join(terms(user_message))
        source_alias_query = state.source_alias_query() if self.config.source_alias_retrieval else ""
        cache_key = (
            query, source_alias_query, state.revision,
            plan.mode if self.config.routed_retrieval else None,
            turn if self.config.soft_preference_decay else None,
            sufficiency.action,
        )
        fallbacks = list(self.startup_fallbacks)
        cached = self._cache.get(session_id)
        cache_hit = sufficiency.action != "clarify_first" and cached is not None and cached[0] == cache_key
        minimal_probe = sufficiency.action == "minimal_probe"
        preferences = state.effective_preferences(
            self.config.soft_decay_turns if self.config.soft_preference_decay else 0,
        )
        admission_diagnostics = {
            "mode": self.config.rerank_admission,
            "pool_size": 0,
            "model_sha256": getattr(self.admission_model, "model_sha256", None),
            "fallback": None,
        }
        frontier_scored_this_turn: list[str] = []
        frontier_triggered = False
        if sufficiency.action == "clarify_first":
            candidates = list(self._last_candidates.get(session_id, ()))
            routes, route_weights, retrieved_ids, comparison_tail_ids, rerank_prefix_ids = {}, {}, [], [], []
            role_witnesses, composition_witnesses = {}, {}
            stage_counts = {"retrieved": 0, "clarify_first": 1}
            stage_ids = {"retrieved": [], "guarded_before_truncation": [],
                         "candidate_limited": [], "neural_scored": [], "final_ranked": []}
            cascade = ComputeCascadeDecision(False, self.config.rerank_limit, 0.0, ("retrieval_deferred",))
            component_latency["retrieval_and_ranking"] = 0.0
        elif cache_hit:
            (candidates, routes, route_weights, retrieved_ids, comparison_tail_ids,
             rerank_prefix_ids, role_witnesses, composition_witnesses,
             stage_counts, stage_ids, cascade, cached_fallbacks) = cached[1:]
            stage_counts = dict(stage_counts)
            stage_ids = {name: list(values) for name, values in stage_ids.items()}
            fallbacks = list(cached_fallbacks)
            component_latency["retrieval_and_ranking"] = 0.0
        else:
            retrieval_started = time.perf_counter()
            if minimal_probe:
                candidates, routes, route_weights = self._minimal_probe(query, fallbacks)
            else:
                candidates, routes, route_weights = self._retrieve(
                    plan, state, fallbacks, source_alias_query,
                )
            retrieved_ids = [item.product.parent_asin for item in candidates
                             if "tiny_catalog_tail" not in item.route_scores]
            comparison_tail_ids = [item.product.parent_asin for item in candidates
                                   if "tiny_catalog_tail" in item.route_scores]
            stage_counts = {"retrieved": len(candidates)}
            stage_ids = {"retrieved": list(retrieved_ids)}
            if self.config.evidence_ranking:
                try:
                    candidates = _valid_ranking(candidates, rank_candidates(candidates, preferences))
                except (RuntimeError, ValueError, TypeError) as error:
                    fallbacks.append("ranking")
                    LOGGER.warning("Evidence ranking failed: %s", type(error).__name__)
            candidates = _apply_constraints(candidates, preferences, fallbacks)
            if self.config.product_guard:
                candidates = _apply_product_guard(candidates, preferences, fallbacks)
            stage_counts["guarded_before_truncation"] = len(candidates)
            stage_ids["guarded_before_truncation"] = [
                item.product.parent_asin for item in candidates
            ]
            candidate_limit = self.config.candidate_limit
            if minimal_probe:
                candidate_limit = min(candidate_limit, self.config.minimal_probe_limit)
            if self.config.routed_retrieval and plan.mode == "buying" and intent.specificity >= 0.7:
                candidate_limit = max(self.config.rerank_limit, min(candidate_limit, 80))
            candidates = candidates[:candidate_limit]
            stage_counts["candidate_limited"] = len(candidates)
            stage_ids["candidate_limited"] = [item.product.parent_asin for item in candidates]
            cascade = decide_compute_cascade(
                intent, plan, _route_overlap(routes), len(candidates), self.config,
                self._cascade_counts[session_id], self.reranker is not None and not minimal_probe,
                self._last_neural_margin[session_id],
            )
            if cascade.escalate:
                self._cascade_counts[session_id] += 1
            stage_counts["cascade_rerank_limit"] = cascade.rerank_limit
            role_witnesses: dict[str, list[dict]] = {}
            if self.config.role_evidence:
                candidates, role_witnesses = rank_role_evidence(candidates, preferences)
            if self.contrast is not None and not minimal_probe:
                try:
                    candidates = _valid_ranking(candidates, self.contrast.rank(candidates, preferences, self.config.contrast_weight))
                except (RuntimeError, ValueError, KeyError, TypeError) as error:
                    fallbacks.append("contrast")
                    LOGGER.warning("Contrast ranking failed: %s", type(error).__name__)
            rerank_prefix_ids = []
            admission_diagnostics = {
                "mode": self.config.rerank_admission,
                "pool_size": len(candidates),
                "model_sha256": getattr(self.admission_model, "model_sha256", None),
                "fallback": None,
            }
            neural_base: list[Candidate] | None = None
            affordable = (
                cascade.rerank_limit
                if self.config.turn_budget_seconds <= 0
                else self._affordable_rerank_limit(time.perf_counter() - started)
            )
            rerank_limit = min(cascade.rerank_limit, affordable)
            if self.reranker is not None and not minimal_probe and rerank_limit < 1:
                fallbacks.append("latency_budget")
                LOGGER.warning("Turn budget exhausted; serving the pre-reranking order")
            elif self.reranker is not None and not minimal_probe:
                try:
                    rerank_query = plan.rerank_context if self.config.structured_rerank else query
                    if (self.config.over_general_cutoff and intent.over_general
                            and len(candidates) > self.config.over_general_candidate_threshold):
                        rerank_limit = min(rerank_limit, self.config.over_general_rerank_limit)
                        stage_counts["over_general_cutoff"] = rerank_limit
                    admission_mode = self.config.rerank_admission
                    if admission_mode == "linear" and self.admission_model is None:
                        if "admission_model" not in fallbacks:
                            fallbacks.append("admission_model")
                        admission_diagnostics["fallback"] = "prefix"
                        admitted = select_rerank_prefix(candidates, preferences, rerank_limit, "prefix")
                    elif admission_mode in {"fusion", "linear"}:
                        scored_pool, admission_diagnostics = score_all_candidates(
                            candidates, preferences, plan, admission_mode, self.admission_model,
                        )
                        admitted = scored_pool[:rerank_limit]
                    else:
                        admitted = select_rerank_prefix(
                            candidates, preferences, rerank_limit, admission_mode, plan,
                        )
                    rerank_prefix_ids = [item.product.parent_asin for item in admitted]
                    admitted_ids = set(rerank_prefix_ids)
                    ordered = admitted + [
                        item for item in candidates if item.product.parent_asin not in admitted_ids
                    ]
                    neural_base = list(ordered)
                    reranked_at = time.perf_counter()
                    if self.config.structured_rerank:
                        ranked = self.reranker.rank(
                            rerank_query, ordered, rerank_limit, self.config.neural_weight,
                            structured=True,
                        )
                    elif self.config.rerank_document_mode != "head":
                        ranked = self.reranker.rank(
                            rerank_query, ordered, rerank_limit, self.config.neural_weight,
                            preferences, self.config.rerank_document_mode,
                        )
                    elif self.config.neural_margin_fusion:
                        ranked = self.reranker.rank(
                            rerank_query, ordered, rerank_limit, self.config.neural_weight,
                            low_margin_weight=self.config.neural_low_margin_weight,
                            margin_threshold=self.config.neural_margin_threshold,
                        )
                    else:
                        # Keep the selected 30-prefix release on the original
                        # four-argument ranker contract and document view.
                        ranked = self.reranker.rank(
                            rerank_query, ordered, rerank_limit, self.config.neural_weight,
                        )
                    candidates = _valid_ranking(candidates, ranked)
                    self._record_rerank_cost(time.perf_counter() - reranked_at, len(admitted))
                    if self.config.progressive_frontier_rerank:
                        self._frontier_revision[session_id] = state.revision
                        self._frontier_base[session_id] = list(neural_base)
                        self._frontier_logits[session_id] = {
                            item.product.parent_asin: float(item.route_scores["neural_logit"])
                            for item in candidates if "neural_logit" in item.route_scores
                        }
                except (RuntimeError, ValueError, TypeError, TimeoutError) as error:
                    fallbacks.append("neural_rerank")
                    LOGGER.warning("Neural ranking failed: %s", type(error).__name__)
                    rerank_prefix_ids = []
            else:
                rerank_prefix_ids = []
                admission_diagnostics = {
                    "mode": self.config.rerank_admission,
                    "pool_size": len(candidates),
                    "model_sha256": getattr(self.admission_model, "model_sha256", None),
                    "fallback": None,
                }
            if self.config.progressive_frontier_rerank and not rerank_prefix_ids:
                self._frontier_revision.pop(session_id, None)
                self._frontier_base.pop(session_id, None)
                self._frontier_logits.pop(session_id, None)
            stage_counts["reranked"] = len(candidates)
            stage_ids["neural_scored"] = list(rerank_prefix_ids)
            composition_witnesses: dict[str, list[dict]] = {}
            if self.config.composition_evidence:
                candidates, composition_witnesses = rank_composition_evidence(candidates, preferences)
            candidates = _apply_constraints(candidates, preferences, fallbacks)
            candidates = rank_soft_prices(candidates, preferences, self.config.soft_price_weight)
            if self.config.profile_prior:
                candidates = rank_profile_prior(
                    candidates, distill_profile(state.profile), self.config.profile_weight,
                )
                candidates = _apply_constraints(candidates, preferences, fallbacks)
            if self.config.product_guard:
                candidates = _apply_product_guard(candidates, preferences, fallbacks)
            stage_counts["guarded_after_rerank"] = len(candidates)
            if minimal_probe:
                stage_counts["minimal_probe"] = len(candidates)
            stage_ids["final_ranked"] = [item.product.parent_asin for item in candidates]
            if self.config.page_local_rerank:
                if rerank_prefix_ids:
                    self._page_rerank_revision[session_id] = state.revision
                    self._page_rerank_logits[session_id] = {
                        item.product.parent_asin: float(item.route_scores["neural_logit"])
                        for item in candidates if "neural_logit" in item.route_scores
                    }
                else:
                    self._page_rerank_revision.pop(session_id, None)
                    self._page_rerank_logits.pop(session_id, None)
            self._cache[session_id] = (
                cache_key, candidates, routes, route_weights, retrieved_ids, comparison_tail_ids,
                rerank_prefix_ids, role_witnesses, composition_witnesses,
                stage_counts, stage_ids, cascade, tuple(fallbacks),
            )
            component_latency["retrieval_and_ranking"] = time.perf_counter() - retrieval_started
            self._last_candidates[session_id] = list(candidates)

        frontier_eligible = (
            cache_hit
            and self.config.progressive_frontier_rerank
            and self.reranker is not None
            and not minimal_probe
            and not state.last_override.detected
            and (not state.last_update_informative
                 or state.last_feedback.scope in {"item", "product_type"})
            and self._frontier_revision.get(session_id) == state.revision
        )
        if frontier_eligible:
            base = self._frontier_base.get(session_id, [])
            logits = dict(self._frontier_logits.get(session_id, {}))
            shown = self._shown_ids.get(session_id, set())
            frontier = [
                item for item in base
                if item.product.parent_asin not in logits
                and item.product.parent_asin not in shown
            ][:self.config.rerank_limit]
            if frontier:
                frontier_started = time.perf_counter()
                try:
                    rerank_query = plan.rerank_context if self.config.structured_rerank else query
                    if self.config.structured_rerank:
                        new_logits = self.reranker.score(
                            rerank_query, frontier, structured=True,
                        )
                    elif self.config.rerank_document_mode != "head":
                        new_logits = self.reranker.score(
                            rerank_query, frontier, preferences, self.config.rerank_document_mode,
                        )
                    else:
                        new_logits = self.reranker.score(rerank_query, frontier)
                    logits.update(new_logits)
                    candidates = fuse_neural_logits(
                        base, logits, self.config.neural_weight,
                        low_margin_weight=(self.config.neural_low_margin_weight
                                           if self.config.neural_margin_fusion else None),
                        margin_threshold=self.config.neural_margin_threshold,
                    )
                    composition_witnesses = {}
                    if self.config.composition_evidence:
                        candidates, composition_witnesses = rank_composition_evidence(
                            candidates, preferences,
                        )
                    candidates = _apply_constraints(candidates, preferences, fallbacks)
                    candidates = rank_soft_prices(
                        candidates, preferences, self.config.soft_price_weight,
                    )
                    if self.config.profile_prior:
                        candidates = rank_profile_prior(
                            candidates, distill_profile(state.profile), self.config.profile_weight,
                        )
                        candidates = _apply_constraints(candidates, preferences, fallbacks)
                    if self.config.product_guard:
                        candidates = _apply_product_guard(candidates, preferences, fallbacks)
                    frontier_scored_this_turn = [item.product.parent_asin for item in frontier]
                    frontier_triggered = True
                    self._frontier_logits[session_id] = logits
                    rerank_prefix_ids = [
                        item.product.parent_asin for item in base
                        if item.product.parent_asin in logits
                    ]
                    stage_counts["frontier_scored_this_turn"] = len(frontier_scored_this_turn)
                    stage_counts["frontier_scored_total"] = len(logits)
                    stage_ids["neural_scored"] = list(rerank_prefix_ids)
                    stage_ids["final_ranked"] = [item.product.parent_asin for item in candidates]
                    self._record_rerank_cost(
                        time.perf_counter() - frontier_started, len(frontier),
                    )
                    self._cache[session_id] = (
                        cache_key, candidates, routes, route_weights, retrieved_ids,
                        comparison_tail_ids, rerank_prefix_ids, role_witnesses,
                        composition_witnesses, stage_counts, stage_ids, cascade,
                        tuple(fallbacks),
                    )
                    self._last_candidates[session_id] = list(candidates)
                except (RuntimeError, ValueError, TypeError, TimeoutError, AttributeError) as error:
                    fallbacks.append("frontier_rerank")
                    LOGGER.warning("Frontier neural ranking failed: %s", type(error).__name__)
                component_latency["retrieval_and_ranking"] = time.perf_counter() - frontier_started
        policy_started = time.perf_counter()
        decision = choose_policy(
            state, candidates, self.config, turn, top_k, self._abstentions[session_id], intent,
        )
        component_latency["policy"] = time.perf_counter() - policy_started
        state.record_question(decision.ask_attribute, decision.question_goal)
        limit = min(top_k, max(0, decision.slate_size))
        ordered = list(dict.fromkeys(item.product.parent_asin for item in candidates))
        override_page_reset = (
            self.config.slate_reset_on_override and "intent_override" in intent.reasons
        )
        shown_before = len(self._shown_ids.get(session_id, set()))
        if self.config.seen_aware_slate:
            page, ranked, newly_shown = self._seen_aware_selection(
                session_id, ordered, limit, override_page_reset,
            ) if limit else (0, [], 0)
        else:
            page = self._slate_page(session_id, ordered, turn, limit, override_page_reset)
            ranked = ordered[page * limit:page * limit + limit] if limit else []
            newly_shown = 0
        control_page = list(ranked)
        page_local_attempted = False
        page_local_triggered = False
        page_local_scored: list[str] = []
        page_local_reason = "disabled"
        page_local_revision_match = (
            self._page_rerank_revision.get(session_id) == state.revision
        )
        page_logits = (
            dict(self._page_rerank_logits.get(session_id, {}))
            if page_local_revision_match else {}
        )
        by_identifier = {item.product.parent_asin: item for item in candidates}
        page_candidates = [by_identifier[identifier] for identifier in control_page]
        missing_page_scores = [
            item for item in page_candidates
            if item.product.parent_asin not in page_logits
        ]
        page_batches = self._page_rerank_batches.get(session_id, 0)
        page_pairs = self._page_rerank_pairs.get(session_id, 0)
        remaining_pairs = self.config.page_local_max_pairs - page_pairs
        page_local_base_eligible = (
            self.config.page_local_rerank
            and self.reranker is not None
            and bool(control_page)
            and page > 0
            and not state.last_override.detected
            and not state.last_update_informative
            and (cache_hit or state.last_feedback.scope == "item")
            and page_local_revision_match
            and bool(missing_page_scores)
        )
        page_local_eligible = (
            page_local_base_eligible
            and len(missing_page_scores) <= self.config.page_local_rerank_limit
            and len(missing_page_scores) <= remaining_pairs
            and page_batches < self.config.page_local_max_batches
        )
        if page_local_eligible:
            page_local_attempted = True
            page_local_scored = [item.product.parent_asin for item in missing_page_scores]
            self._page_rerank_batches[session_id] = page_batches + 1
            self._page_rerank_pairs[session_id] = page_pairs + len(missing_page_scores)
            page_started = time.perf_counter()
            try:
                rerank_query = plan.rerank_context if self.config.structured_rerank else query
                if self.config.structured_rerank:
                    new_logits = self.reranker.score(
                        rerank_query, missing_page_scores, structured=True,
                    )
                elif self.config.rerank_document_mode != "head":
                    new_logits = self.reranker.score(
                        rerank_query, missing_page_scores, preferences,
                        self.config.rerank_document_mode,
                    )
                else:
                    new_logits = self.reranker.score(rerank_query, missing_page_scores)
                elapsed = time.perf_counter() - page_started
                if set(new_logits) != set(page_local_scored):
                    raise ValueError("Page-local reranker returned the wrong candidate set")
                page_logits.update(new_logits)
                local_logits = {
                    identifier: page_logits[identifier] for identifier in control_page
                }
                local_ranking = fuse_neural_logits(
                    page_candidates, local_logits, self.config.neural_weight,
                    low_margin_weight=(self.config.neural_low_margin_weight
                                       if self.config.neural_margin_fusion else None),
                    margin_threshold=self.config.neural_margin_threshold,
                )
                reordered = [item.product.parent_asin for item in local_ranking]
                if len(reordered) != len(control_page) or set(reordered) != set(control_page):
                    raise ValueError("Page-local reranking changed page membership")
                if elapsed > self.config.page_local_budget_seconds:
                    raise TimeoutError("Page-local reranking exceeded its latency budget")
                ranked = reordered
                page_local_triggered = True
                page_local_reason = "page_reordered"
                self._page_rerank_logits[session_id] = page_logits
                self._record_rerank_cost(elapsed, len(missing_page_scores))
            except (RuntimeError, ValueError, TypeError, TimeoutError, AttributeError) as error:
                ranked = control_page
                page_local_reason = type(error).__name__
                fallbacks.append("frontier_page_rerank")
                LOGGER.warning("Page-local neural ranking failed: %s", type(error).__name__)
            component_latency["page_local_rerank"] = time.perf_counter() - page_started
        elif self.config.page_local_rerank:
            if state.last_override.detected:
                page_local_reason = "override"
            elif state.last_update_informative:
                page_local_reason = "informative_update"
            elif page == 0:
                page_local_reason = "first_page"
            elif not missing_page_scores:
                page_local_reason = "already_scored"
            elif page_batches >= self.config.page_local_max_batches:
                page_local_reason = "batch_budget"
            elif len(missing_page_scores) > remaining_pairs:
                page_local_reason = "pair_budget"
            elif not cache_hit and state.last_feedback.scope != "item":
                page_local_reason = "not_continuation"
            elif not page_local_revision_match:
                page_local_reason = "revision_mismatch"
            else:
                page_local_reason = "ineligible"
        stage_ids["control_page"] = list(control_page)
        stage_ids["returned_page"] = list(ranked)
        neural_scores = _neural_score_summary(candidates)
        if neural_scores["logit_margin"] is not None:
            self._last_neural_margin[session_id] = neural_scores["logit_margin"]
        self._abstentions[session_id] = 0 if ranked else self._abstentions[session_id] + 1
        neural_cache_after = self._neural_cache_stats()
        neural_cache_turn = {
            key: int(neural_cache_after.get(key, 0)) - int(neural_cache_before.get(key, 0))
            for key in ("hits", "misses", "evictions", "evaluated_pairs")
        }
        self.last_diagnostics = {
            "query": query, "source_alias_query": source_alias_query,
            "revision": state.revision, "cache_hit": cache_hit, "slate_page": page,
            "slate_page_reset": "intent_override" if override_page_reset else None,
            "slate_exposure": {
                "seen_aware": self.config.seen_aware_slate,
                "shown_before": shown_before,
                "newly_shown": newly_shown,
                "shown_total": len(self._shown_ids.get(session_id, set())),
            },
            "frontier_rerank": {
                "enabled": self.config.progressive_frontier_rerank,
                "eligible": frontier_eligible,
                "triggered": frontier_triggered,
                "scored_this_turn": list(frontier_scored_this_turn),
                "scored_total": len(self._frontier_logits.get(session_id, {})),
                "state_revision": self._frontier_revision.get(session_id),
            },
            "page_local_rerank": {
                "enabled": self.config.page_local_rerank,
                "eligible": page_local_eligible,
                "attempted": page_local_attempted,
                "triggered": page_local_triggered,
                "reason": page_local_reason,
                "control_page": list(control_page),
                "returned_page": list(ranked),
                "membership_preserved": (
                    len(control_page) == len(ranked) and set(control_page) == set(ranked)
                ),
                "scored_this_turn": list(page_local_scored),
                "batches_used": self._page_rerank_batches.get(session_id, 0),
                "additional_pairs_used": self._page_rerank_pairs.get(session_id, 0),
                "state_revision": self._page_rerank_revision.get(session_id),
            },
            "neural_logit_cache": {
                **neural_cache_after,
                "turn": neural_cache_turn,
            },
            "override": {
                "detected": state.last_override.detected,
                "confidence": state.last_override.confidence,
                "changed_attributes": list(state.last_override.changed_attributes),
                "retired": [
                    {"attribute": fact.attribute, "value": fact.value, "polarity": fact.polarity,
                     "scope": fact.scope}
                    for fact in state.last_override.retired
                ],
                "added": [
                    {"attribute": fact.attribute, "value": fact.value, "polarity": fact.polarity,
                     "scope": fact.scope}
                    for fact in state.last_override.added
                ],
                "retained": [
                    {"attribute": fact.attribute, "value": fact.value, "polarity": fact.polarity,
                     "scope": fact.scope}
                    for fact in state.last_override.retained
                ],
                "reasons": list(state.last_override.reasons),
            },
            "retrieval_sufficiency": {
                "action": sufficiency.action, "sufficient": sufficiency.sufficient,
                "reasons": list(sufficiency.reasons),
                "deferred_turns": self._deferred_turns[session_id],
                "avoided_neural_pairs": self.config.rerank_limit if sufficiency.action != "retrieve" else 0,
                "used_previous_slate": sufficiency.action == "clarify_first" and bool(candidates),
            },
            "compute_cascade": {
                "escalated": cascade.escalate, "rerank_limit": cascade.rerank_limit,
                "uncertainty": cascade.uncertainty, "reasons": list(cascade.reasons),
                "session_escalations": self._cascade_counts[session_id],
                "max_session_escalations": self.config.cascade_max_turns,
            },
            "neural_scores": neural_scores,
            "intent": {"mode": intent.mode, "specificity": intent.specificity,
                       "confidence": intent.confidence,
                       "hard_constraint_count": intent.hard_constraint_count,
                       "over_general": intent.over_general, "reasons": list(intent.reasons)},
            "retrieval_plan": {
                "mode": plan.mode, "object_types": list(plan.object_types),
                "category_terms": list(plan.category_terms), "positive_terms": list(plan.positive_terms),
                "negative_terms": list(plan.negative_terms), "use_case": list(plan.use_case),
                "semantic_queries": list(plan.semantic_queries), "lexical_query": plan.lexical_query,
                "rerank_context": plan.rerank_context,
                "hypotheses": [{"query": item.query, "object_types": list(item.object_types),
                                "reason": item.reason} for item in plan.hypotheses],
                "scoped_features": [{"attribute": item.attribute, "value": item.value,
                                     "polarity": item.polarity, "scope": item.scope}
                                    for item in plan.scoped_features],
            },
            "preferences": [{"attribute": p.attribute, "value": p.value, "polarity": p.polarity,
                             "source_turn": p.source_turn, "hard": p.hard,
                             "source_kind": p.source_kind,
                             "provenance": "current_turn" if p.source_turn == turn else "earlier_session",
                             **({"scope": p.scope} if p.scope is not None else {}),
                             **({"alternative_group": p.alternative_group} if p.alternative_group is not None else {})}
                            for p in state.active_preferences()],
            "preference_groups": {
                "hard": [p.value for p in state.active_preferences() if p.hard and p.polarity != 0],
                "soft": [p.value for p in state.active_preferences() if not p.hard and p.polarity == 1],
                "negative": [p.value for p in state.active_preferences() if p.polarity == -1],
                "neutral": [p.attribute for p in state.active_preferences() if p.polarity == 0],
            },
            "profile_prior_terms": list(distill_profile(state.profile)) if self.config.profile_prior else [],
            "unsupported_alternatives": [dict(item) for item in state.unsupported_alternatives],
            "negative_feedback": {
                "scope": state.last_feedback.scope, "attribute": state.last_feedback.attribute,
                "reason": state.last_feedback.reason,
            },
            "routes": routes, "route_weights": route_weights, "route_overlap": _route_overlap(routes),
            "stage_counts": stage_counts,
            "stage_ids": stage_ids,
            "retrieved_ids": retrieved_ids, "comparison_tail_ids": comparison_tail_ids,
            "rerank_admission": self.config.rerank_admission,
            "admission": admission_diagnostics,
            "catalog_vocabulary": {
                "enabled": self.catalog_vocabulary is not None,
                "version": getattr(self.catalog_vocabulary, "version", None),
                "model_sha256": getattr(self.catalog_vocabulary, "model_sha256", None),
            },
            "rerank_document_mode": self.config.rerank_document_mode, "rerank_prefix_ids": rerank_prefix_ids,
            "role_evidence": role_witnesses,
            "composition_evidence": composition_witnesses,
            "ranked_ids": [item.product.parent_asin for item in candidates],
            "constraint_penalties": {item.product.parent_asin: item.route_scores.get("constraint_penalty", 0.0)
                                     for item in candidates},
            "object_penalties": {item.product.parent_asin: item.route_scores.get("object_penalty", 0.0)
                                 for item in candidates},
            "price_adjustments": {item.product.parent_asin: item.route_scores.get("price_preference", 0.0)
                                  for item in candidates},
            "fallbacks": fallbacks, "policy": decision.diagnostics,
            "question": {"attribute": decision.ask_attribute, "goal": decision.question_goal,
                         "reason": decision.diagnostics.get("decision", "policy")},
            "component_latency_seconds": component_latency,
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
        self._deferred_turns.clear()
        self._last_candidates.clear()
        self._cascade_counts.clear()
        self._last_neural_margin.clear()
        self._pages.clear()
        self._last_ranked.clear()
        self._shown_ids.clear()
        self._frontier_revision.clear()
        self._frontier_base.clear()
        self._frontier_logits.clear()
        self._page_rerank_revision.clear()
        self._page_rerank_logits.clear()
        self._page_rerank_batches.clear()
        self._page_rerank_pairs.clear()
