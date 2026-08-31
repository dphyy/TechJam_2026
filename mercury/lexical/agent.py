from collections import OrderedDict
from pathlib import Path

from .config import AgentConfig, DEFAULT_AGENT_CONFIG
from .dialogue import SessionState
from .memory import UserProfileStore
from .question_planner import AdaptiveQuestionPlanner
from .ranking import DEFAULT_RANKING_POLICIES, RankingPolicies
from .retrieval import FEATURE_CACHE_SIZE, CatalogSearch
from .vector_index import VectorIndex


AMBIGUOUS_FIELD_SCORE_THRESHOLD = 2.0


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
    ) -> None:
        if type(max_sessions) is not int or max_sessions < 1:
            raise ValueError("max_sessions must be a positive integer")
        self.catalog_path = Path(catalog_path)
        self.config = config
        self.max_sessions = max_sessions
        self.share_profile_memory = share_profile_memory
        self.search = CatalogSearch(
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
        self.profile_store = UserProfileStore()

    def close(self) -> None:
        self._sessions.clear()
        self._profile_ids.clear()
        self._ambiguity_deferred.clear()
        self.profile_store.profiles.clear()
        self.search.close()

    def _drop_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._ambiguity_deferred.discard(session_id)
        profile_id = self._profile_ids.pop(session_id, None)
        if profile_id is not None and profile_id not in self._profile_ids.values():
            self.profile_store.forget(profile_id)

    def reset(self, session_id: str, user_profile: dict) -> None:
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
        self.profile_store.forget(profile_id)

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        state = self._sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before respond")
        self._sessions.move_to_end(session_id)

        state.observe(user_message, turn)
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
        if unresolved_siblings:
            self._ambiguity_deferred.add(session_id)
        ranked = (
            []
            if unresolved_siblings
            else result.recommendations[:recommendation_limit]
        )
        return {
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
