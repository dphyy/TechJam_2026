from collections import OrderedDict
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import time

from .config import AgentConfig, DEFAULT_AGENT_CONFIG
from .dialogue import SessionState
from .diagnostics import capability_receipt, runtime_identity, turn_receipt
from .memory import UserProfileStore
from .paging import ContextItem, PagingState, explicit_override, select_page, semantic_signature
from .question_planner import AdaptiveQuestionPlanner
from .ranking import DEFAULT_RANKING_POLICIES, RankingPolicies
from .retrieval import FEATURE_CACHE_SIZE, CatalogSearch
from .vector_index import VectorIndex


AMBIGUOUS_FIELD_SCORE_THRESHOLD = 2.0
ASK_ATTRIBUTES = {"category", "material", "color", "size", "style", "brand", "budget", "feature", "use_case", "other"}


class Agent:
    """Deterministic conversational product-search agent."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        feature_cache_size: int = FEATURE_CACHE_SIZE,
        *,
        config: AgentConfig = DEFAULT_AGENT_CONFIG,
        ranking_policies: RankingPolicies = DEFAULT_RANKING_POLICIES,
        vector_index: VectorIndex | None = None,
        catalog_index_path: str | Path | None = None,
        max_sessions: int = 256,
        share_profile_memory: bool = False,
        search_factory=CatalogSearch,
    ) -> None:
        if type(max_sessions) is not int or max_sessions < 1:
            raise ValueError("max_sessions must be a positive integer")
        self.catalog_path = Path(catalog_path)
        self.config = config
        self.max_sessions = max_sessions
        self.share_profile_memory = share_profile_memory
        self.search = search_factory(
            self.catalog_path,
            feature_cache_size=feature_cache_size,
            enable_vector_reranker=config.enable_vector_reranker,
            ranking_policies=ranking_policies,
            vector_index=vector_index,
            catalog_index_path=catalog_index_path,
        )
        self.question_planner = AdaptiveQuestionPlanner(self.search.feature_store)
        self._sessions: OrderedDict[str, SessionState] = OrderedDict()
        self._profile_ids: dict[str, str] = {}
        self._ambiguity_deferred: set[str] = set()
        self._responses: dict[str, tuple[tuple[int, str, int], dict]] = {}
        self._diagnostics: dict[str, dict] = {}
        self._sources: dict[str, dict[int, dict]] = {}
        self._pages: dict[str, PagingState] = {}
        self._closed = False
        self._last_diagnostics: dict = {}
        self.profile_store = UserProfileStore()
        try:
            self._identity = runtime_identity(
                self.catalog_path, config, ranking_policies, self.search,
                agent=self, planner=self.question_planner, search_factory=search_factory,
                feature_cache_size=feature_cache_size, max_sessions=max_sessions,
                share_profile_memory=share_profile_memory,
            )
        except Exception:
            self.search.close()
            raise
        startup_fallbacks = list(capability_receipt(self._identity, config, self.search, {})["fallbacks"])
        if self._identity["catalog_index"]["artifact_present"] and not self._identity["catalog_index"]["prebuilt_loaded"]:
            startup_fallbacks.append("catalog_index_rebuilt")
        self.startup_fallbacks = tuple(startup_fallbacks)

    @property
    def last_diagnostics(self) -> dict:
        return deepcopy(self._last_diagnostics)

    @last_diagnostics.setter
    def last_diagnostics(self, value: dict) -> None:
        """Allow specialized wrappers to publish a detached receipt."""
        if not isinstance(value, dict):
            raise ValueError("diagnostics must be an object")
        self._last_diagnostics = deepcopy(value)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._sessions.clear()
        self._profile_ids.clear()
        self._ambiguity_deferred.clear()
        self._responses.clear()
        self._diagnostics.clear()
        self._sources.clear()
        self._pages.clear()
        self._last_diagnostics.clear()
        self.profile_store.profiles.clear()
        self.search.close()

    def _drop_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._ambiguity_deferred.discard(session_id)
        self._responses.pop(session_id, None)
        self._diagnostics.pop(session_id, None)
        self._sources.pop(session_id, None)
        self._pages.pop(session_id, None)
        self._last_diagnostics = {}
        profile_id = self._profile_ids.pop(session_id, None)
        if profile_id is not None and profile_id not in self._profile_ids.values():
            self.profile_store.forget(profile_id)

    def reset(self, session_id: str, user_profile: dict) -> None:
        if self._closed:
            raise RuntimeError("Agent is closed")
        if not isinstance(session_id, str) or not session_id or not isinstance(user_profile, dict):
            raise ValueError("reset requires a nonempty session ID and profile object")
        user_profile = deepcopy(user_profile)
        self._drop_session(session_id)
        profile_id = str(user_profile.get("profile_id") or user_profile.get("user_id") or session_id)
        if not self.share_profile_memory:
            self.profile_store.forget(profile_id)
        profile = self.profile_store.get(profile_id, user_profile)
        self._sessions[session_id] = SessionState(user_profile=user_profile, long_term_profile=profile)
        self._profile_ids[session_id] = profile_id
        self._ambiguity_deferred.discard(session_id)
        while len(self._sessions) > self.max_sessions:
            self._drop_session(next(iter(self._sessions)))

    def export_profile(self, profile_id: str) -> dict | None:
        profile = self.profile_store.profiles.get(profile_id)
        return profile.snapshot() if profile else None

    def forget_profile(self, profile_id: str) -> None:
        profile = self.profile_store.profiles.get(profile_id)
        if profile is not None:
            profile.clear()
        for session_id, current_id in self._profile_ids.items():
            if current_id != profile_id:
                continue
            state = self._sessions[session_id]
            if state.long_term_profile is not None:
                state.long_term_profile.clear()
            state.long_term_profile = None
            state.user_profile = {}
            state.forget_provenance()
            self._responses.pop(session_id, None)
            self._diagnostics.pop(session_id, None)
            self._sources.pop(session_id, None)
            self._pages.pop(session_id, None)
            self._ambiguity_deferred.discard(session_id)
        self._last_diagnostics = {}
        self.profile_store.forget(profile_id)

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        try:
            return self._respond(session_id, user_message, turn, top_k)
        except Exception as exc:
            self._last_diagnostics = {
                "request_succeeded": False, "state_committed": False,
                "turn": turn if type(turn) is int else None,
                "error_type": type(exc).__name__, "fallbacks": [], "fallbacks_complete": False,
                "identity": deepcopy(self._identity),
            }
            raise

    def _respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        started = time.perf_counter()
        if self._closed:
            raise RuntimeError("Agent is closed")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("respond requires a nonempty session ID")
        if not isinstance(user_message, str) or type(turn) is not int or not 1 <= turn <= 10:
            raise ValueError("respond requires text and a turn in [1, 10]")
        if type(top_k) is not int or not 1 <= top_k <= 10:
            raise ValueError("top_k must be an integer in [1, 10]")
        state = self._sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before respond")
        received_characters = len(user_message)
        message_sha256 = hashlib.sha256(user_message.encode("utf-8")).hexdigest()
        request = (turn, message_sha256, top_k)
        user_message = user_message[:8000]
        previous = self._responses.get(session_id)
        if previous is not None and previous[0] == request:
            self._sessions.move_to_end(session_id)
            self._last_diagnostics = deepcopy(self._diagnostics.get(session_id, {}))
            self._last_diagnostics.update(cache_hit=True, latency_seconds=time.perf_counter() - started,
                                          current_call={"search_executed": False, "inference_executed": False})
            original_vector_receipt = self._last_diagnostics.get("vector_stage", {})
            self._last_diagnostics["vector_stage"] = {
                "attempted": False, "inference_attempted": False, "status": "cached_response",
                "returned_count": 0, "contribution_count": 0, "origin_receipt": original_vector_receipt,
            }
            vector_capability = self._last_diagnostics.get("effective_capabilities", {}).get("components", {}).get("vector_rerank", {})
            vector_capability.update(attempted=False, status="cached_response")
            return deepcopy(previous[1])
        if turn <= state.last_turn:
            raise ValueError("turn must advance; conflicting or stale retries are not accepted")
        original_state = state
        state = deepcopy(original_state)
        state.observe(user_message, turn, category_names=getattr(self.search, "category_names", frozenset()))
        result = self.search.search_with_context(
            state, limit=max(1, min(int(top_k), 10))
        )

        question_plan = self.question_planner.choose(
            state, result.candidates, turn
        )

        top_candidate = result.candidates[0] if result.candidates else {}
        hard_count = int(top_candidate.get("_hard_constraint_count") or 0)
        exact_count = int(top_candidate.get("_hard_constraint_exact_count") or 0)
        clarification_count = (
            state.asked_attributes.count(question_plan.attribute)
            if question_plan.attribute
            else 0
        )
        recommendation_limit = self.config.recommendation_policy.limit_for(
            turn,
            top_k,
            scores=tuple(score for _, score in result.recommendations),
            hard_constraint_coverage=(exact_count / hard_count if hard_count else 0.0),
            has_hard_constraints=hard_count > 0,
            has_answerable_clarification=(
                question_plan.attribute is not None
                and clarification_count <= 2
                and len(state.asked_attributes) <= 2
            ),
            clarification_expected_value=question_plan.expected_value,
            turns_remaining=max(0, 10 - turn),
            has_intent_override=any(
                item.source == "override" for item in state.evidence
            ),
            has_no_preference_reply=bool(state.no_preference_attributes),
        )
        if self.config.full_width:
            recommendation_limit = max(1, min(int(top_k), 10))
        # When the leading catalog records are observational siblings, their
        # popularity score is not evidence that one satisfies the request
        # better. Use the already-planned clarification before exposing an
        # arbitrary sibling order; the next answer commonly supplies the rare
        # feature phrase that disambiguates the records.
        unresolved_siblings = (
            not self.config.full_width
            and recommendation_limit > 1
            and session_id not in self._ambiguity_deferred
            and turn < 9
            and question_plan.attribute is not None
            and len(result.candidates) >= 2
            and float(
                (result.candidates[0].get("_catalog_tiebreak") or (0.0,))[0]
            ) < AMBIGUOUS_FIELD_SCORE_THRESHOLD
            and result.candidates[0].get("_catalog_tiebreak")
            == result.candidates[1].get("_catalog_tiebreak")
            and result.candidates[0].get("_hard_constraint_exact_count")
            == result.candidates[1].get("_hard_constraint_exact_count")
            and result.candidates[0].get("_category_leaf_match")
            == result.candidates[1].get("_category_leaf_match")
        )
        tentative = (unresolved_siblings and self.config.tentative_on_ambiguity
                     and result.candidates[0].get("_semantic_violation") is False
                     and bool(result.recommendations))
        ranked = (result.recommendations[:1] if tentative else [] if unresolved_siblings
                  else result.recommendations[:recommendation_limit])
        # Stage paging with the copied dialogue state. Failed turns and retries
        # must not consume exposure history or advance to a different page.
        base_ranked = list(ranked)
        page_state = None
        paging_receipt = {"enabled": False, "triggered": False, "reason": "disabled"}
        if self.config.guarded_paging:
            context = tuple(ContextItem(product["parent_asin"], product["_rank_score"],
                                        bool(product.get("_semantic_violation")))
                            for product in result.candidates)
            selected, page_state, paging_receipt = select_page(
                context, tuple(key for key, _ in ranked), semantic_signature(state),
                explicit_override(user_message, state, turn), self._pages.get(session_id), enabled=True,
            )
            scores = {item.identifier: item.score for item in context}
            ranked = [(key, scores[key]) for key in selected]
        response = {
            "message": question_plan.message,
            "ask_attribute": question_plan.attribute,
            "recommendations": [
                {"parent_asin": parent_asin, "score": round(score, 6)}
                for parent_asin, score in ranked
            ],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": 0,
            },
        }
        self._validate_response(response, top_k)
        cached_response = deepcopy(response)
        sources = dict(self._sources.get(session_id, {}))
        sources[turn] = {"text": user_message, "message_sha256": message_sha256,
                         "received_characters": received_characters, "accepted_characters": len(user_message),
                         "complete": received_characters <= 8000}
        diagnostics = turn_receipt(
            before=original_state.evidence, state=state, previous=self._diagnostics.get(session_id, {}),
            sources=sources, turn=turn, top_k=top_k, result=result, response=response,
            identity=self._identity, config=self.config, search=self.search,
            latency_seconds=time.perf_counter() - started, deferred=unresolved_siblings and not tentative,
        )
        diagnostics["state_committed"] = True
        diagnostics["paging"] = paging_receipt
        diagnostics["base_recommendations"] = [
            {"parent_asin": key, "score": round(score, 6)} for key, score in base_ranked
        ]
        diagnostics["output_width"].update(
            policy_limit=1 if tentative else recommendation_limit,
            reason="tentative_ambiguity" if tentative else "ambiguity_deferred" if unresolved_siblings else
            "candidate_shortfall" if len(response["recommendations"]) < recommendation_limit else
            "full_width" if self.config.full_width else
            "adaptive_policy" if self.config.recommendation_policy.adaptive else "fixed_width",
        )
        diagnostics["question"] = {"attribute": question_plan.attribute,
                                    "information_gain": question_plan.information_gain,
                                    "answerability": question_plan.answerability,
                                    "expected_value": question_plan.expected_value}
        diagnostics["current_call"] = {"search_executed": True,
                                       "inference_executed": result.vector_stage.get("inference_attempted", False)}
        cached_diagnostics = deepcopy(diagnostics)
        # Commit only after search, planning and response validation succeed.
        # Shared sessions keep the same profile object rather than a detached copy.
        original_profile = original_state.long_term_profile
        if original_profile is not None and state.long_term_profile is not None:
            original_profile.learned.clear()
            original_profile.learned.update(state.long_term_profile.learned)
            original_profile._observations.clear()
            original_profile._observations.update(state.long_term_profile._observations)
            state.long_term_profile = original_profile
        original_state.__dict__.update(state.__dict__)
        self._sessions.move_to_end(session_id)
        if unresolved_siblings:
            self._ambiguity_deferred.add(session_id)
        self._responses[session_id] = (request, cached_response)
        self._diagnostics[session_id] = cached_diagnostics
        self._sources[session_id] = sources
        if page_state is not None:
            self._pages[session_id] = page_state
        self._last_diagnostics = diagnostics
        return response

    def _validate_response(self, response: dict, top_k: int) -> None:
        if not isinstance(response["message"], str) or (
            response["ask_attribute"] is not None and (
                not isinstance(response["ask_attribute"], str) or response["ask_attribute"] not in ASK_ATTRIBUTES)
        ):
            raise ValueError("invalid clarification response")
        rows = response["recommendations"]
        identifiers = [row["parent_asin"] for row in rows]
        if (len(rows) > top_k or len(set(identifiers)) != len(rows)
                or any(not isinstance(value, str) or value not in self.search._row_id_by_asin for value in identifiers)
                or any(type(row["score"]) not in {int, float} or not math.isfinite(row["score"]) for row in rows)):
            raise ValueError("invalid recommendation response")
        if any(type(value) is not int or value < 0 for value in response["usage"].values()):
            raise ValueError("invalid response usage")
