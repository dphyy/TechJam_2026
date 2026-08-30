from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from mercury.model_assets import RERANKERS


@dataclass(frozen=True, slots=True)
class Config:
    dense: bool = False
    neural_rerank: bool = False
    contrast: bool = False
    evidence_ranking: bool = True
    routed_retrieval: bool = False
    product_guard: bool = False
    structured_rerank: bool = False
    over_general_cutoff: bool = False
    profile_prior: bool = False
    soft_preference_decay: bool = False
    scoped_preferences: bool = False
    retrieval_sufficiency_gate: bool = False
    compute_cascade: bool = False
    multi_hypothesis_retrieval: bool = False
    semantic_question_goals: bool = False
    require_positive_question_value: bool = False
    role_evidence: bool = False
    composition_evidence: bool = False
    source_alias_retrieval: bool = False
    typed_plan_mode: str = "off"
    intent_conditioned_ranking: bool = False
    state_mode: str = "ledger"
    alternatives_mode: str = "off"
    question_policy: str = "other"
    slate_policy: str = "fixed"
    slate_size: int = 10
    other_question_limit: int = 2
    over_general_candidate_threshold: int = 80
    over_general_rerank_limit: int = 12
    soft_decay_turns: int = 3
    rerank_admission: str = "prefix"
    retrieval_mode: str = "broad"
    rerank_document_mode: str = "head"
    insufficient_action: str = "minimal_probe"
    reranker_model: str = "reranker"
    turn_budget_seconds: float = 0.0
    slate_paging_first_turn: int = 0
    candidate_limit: int = 120
    sparse_limit: int = 180
    dense_limit: int = 100
    rerank_limit: int = 30
    intent_browsing_pool_limit: int = 30
    max_deferred_turns: int = 1
    minimal_probe_limit: int = 30
    cascade_max_rerank_limit: int = 60
    cascade_max_turns: int = 2
    cascade_candidate_threshold: int = 100
    max_intent_hypotheses: int = 2
    hypothesis_candidate_budget: int = 120
    max_sessions: int = 256
    dense_weight: float = 0.35
    neural_weight: float = 0.25
    contrast_weight: float = 0.12
    router_buying_threshold: float = 0.50
    router_browsing_threshold: float = 0.50
    router_over_general_threshold: float = 0.35
    intent_object_weight: float = 0.20
    intent_slot_weight: float = 0.25
    intent_hard_weight: float = 0.00
    intent_buying_language_weight: float = 0.20
    intent_browsing_language_weight: float = 0.50
    intent_use_case_weight: float = 0.25
    intent_unresolved_weight: float = 0.25
    intent_sparse_request_weight: float = 0.50
    buying_scoped_weight: float = 0.55
    browsing_scenario_weight: float = 0.25
    mixed_scoped_weight: float = 0.45
    buying_dense_weight: float = 0.15
    browsing_dense_weight: float = 0.55
    mixed_dense_weight: float = 0.35
    question_turn_cost: float = 0.02
    profile_weight: float = 0.005
    soft_price_weight: float = 0.02
    soft_negative_weight: float = 0.02
    typed_plan_weight: float = 0.10
    intent_buying_hard_weight: float = 0.10
    intent_browsing_diversity_strength: float = 0.20
    minimum_retrieval_specificity: float = 0.35
    cascade_threshold: float = 0.65
    cascade_low_overlap: float = 0.15
    cascade_low_confidence: float = 0.65
    cascade_previous_margin_threshold: float = 0.0
    artifact_dir: str = "artifacts"
    device: str = "cpu"
    threads: int = 4

    def __post_init__(self) -> None:
        choices = {
            "state_mode": {"ledger", "latest", "history"},
            "alternatives_mode": {"off", "parse", "grouped"},
            "question_policy": {"other", "schedule", "entropy", "rank_value", "intent", "none"},
            "slate_policy": {"fixed", "gap", "lookahead"},
            "rerank_admission": {"prefix", "stratified", "cover"},
            "retrieval_mode": {"broad", "field_union", "factored"},
            "rerank_document_mode": {"head", "lexical", "protected"},
            "insufficient_action": {"minimal_probe", "clarify_first"},
            "reranker_model": set(RERANKERS),
            "typed_plan_mode": {"off", "shadow", "active"},
            "device": {"cpu", "mps", "cuda"},
        }
        for key, allowed in choices.items():
            if getattr(self, key) not in allowed:
                raise ValueError(f"Invalid {key}: {getattr(self, key)!r}")
        if self.alternatives_mode == "grouped" and self.state_mode != "ledger":
            raise ValueError("Grouped alternatives require ledger state")
        for key in ("dense", "neural_rerank", "contrast", "evidence_ranking",
                    "routed_retrieval", "product_guard", "structured_rerank", "over_general_cutoff",
                    "profile_prior", "soft_preference_decay", "scoped_preferences",
                    "retrieval_sufficiency_gate", "compute_cascade", "multi_hypothesis_retrieval",
                    "semantic_question_goals", "require_positive_question_value", "role_evidence",
                    "composition_evidence", "source_alias_retrieval", "intent_conditioned_ranking"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be a boolean")
        if self.role_evidence and self.composition_evidence:
            raise ValueError("Role and composition evidence require separate configurations")
        if self.source_alias_retrieval and self.retrieval_mode != "broad":
            raise ValueError("Source alias retrieval requires broad retrieval")
        for key in ("candidate_limit", "sparse_limit", "dense_limit", "rerank_limit",
                    "max_sessions", "threads", "over_general_candidate_threshold", "over_general_rerank_limit",
                    "soft_decay_turns", "max_deferred_turns", "minimal_probe_limit",
                    "cascade_max_rerank_limit", "cascade_max_turns", "cascade_candidate_threshold",
                    "max_intent_hypotheses", "hypothesis_candidate_budget",
                    "intent_browsing_pool_limit"):
            value = getattr(self, key)
            if type(value) is not int or not 1 <= value <= 10000:
                raise ValueError(f"{key} must be an integer in [1, 10000]")
        if type(self.slate_size) is not int or not 0 <= self.slate_size <= 10:
            raise ValueError("slate_size must be an integer in [0, 10]")
        if type(self.other_question_limit) is not int or not 0 <= self.other_question_limit <= 9:
            raise ValueError("other_question_limit must be an integer in [0, 9]")
        if type(self.slate_paging_first_turn) is not int or not 0 <= self.slate_paging_first_turn <= 10:
            raise ValueError("slate_paging_first_turn must be an integer in [0, 10]; 0 disables paging")
        for key in ("dense_weight", "neural_weight", "contrast_weight", "router_buying_threshold",
                    "router_browsing_threshold", "router_over_general_threshold", "intent_object_weight",
                    "intent_slot_weight", "intent_hard_weight", "intent_buying_language_weight",
                    "intent_browsing_language_weight", "intent_use_case_weight", "intent_unresolved_weight",
                    "intent_sparse_request_weight", "buying_scoped_weight", "browsing_scenario_weight",
                    "mixed_scoped_weight",
                    "buying_dense_weight", "browsing_dense_weight", "mixed_dense_weight", "question_turn_cost",
                    "profile_weight", "soft_price_weight", "soft_negative_weight", "typed_plan_weight",
                    "intent_buying_hard_weight", "intent_browsing_diversity_strength",
                    "minimum_retrieval_specificity",
                    "cascade_threshold", "cascade_low_overlap", "cascade_low_confidence"):
            value = getattr(self, key)
            if type(value) not in (int, float) or not 0 <= value <= 1:
                raise ValueError(f"{key} must be a finite number in [0, 1]")
        if self.compute_cascade and not (
                self.rerank_limit <= self.cascade_max_rerank_limit <= 60
                and self.cascade_max_rerank_limit <= self.candidate_limit
                and self.cascade_max_turns <= 10):
            raise ValueError("compute cascade requires base <= max <= min(60, candidates) and at most 10 turns")
        if self.multi_hypothesis_retrieval and not (
                self.max_intent_hypotheses <= 2
                and self.hypothesis_candidate_budget <= self.candidate_limit):
            raise ValueError("multi-hypothesis retrieval allows at most two hypotheses within candidate_limit")
        if self.intent_conditioned_ranking and self.intent_browsing_pool_limit > self.candidate_limit:
            raise ValueError("intent browsing pool must fit within candidate_limit")
        if type(self.cascade_previous_margin_threshold) not in (int, float) \
                or not 0 <= self.cascade_previous_margin_threshold <= 100:
            raise ValueError("cascade_previous_margin_threshold must be in [0, 100]")
        if type(self.turn_budget_seconds) not in (int, float) or \
                not math.isfinite(self.turn_budget_seconds) or self.turn_budget_seconds < 0:
            raise ValueError("turn_budget_seconds must be a finite number of seconds >= 0")
        if not isinstance(self.artifact_dir, str) or not self.artifact_dir:
            raise ValueError("artifact_dir must be a nonempty path")

    @classmethod
    def from_dict(cls, values: dict) -> Config:
        if not isinstance(values, dict):
            raise ValueError("Configuration must be an object")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        return cls(**values)

    @classmethod
    def load(cls, path: str | Path) -> Config:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict:
        return asdict(self)
