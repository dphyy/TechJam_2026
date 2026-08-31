from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RecommendationPolicy:
    """Choose recommendation breadth from live ranking confidence."""

    # Fixed defaults; compare policy changes with a separate full-width control.
    high_margin: float = 0.08
    low_margin: float = 0.01
    low_entropy: float = 0.80
    high_entropy: float = 0.98
    clarification_horizon: int = 1
    moderate_width: int = 2
    valuable_question_threshold: float = 0.20
    low_value_width: int = 5
    adaptive: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.low_margin <= self.high_margin:
            raise ValueError("margin thresholds must be ordered and non-negative")
        if not 0.0 <= self.low_entropy <= self.high_entropy <= 1.0:
            raise ValueError("entropy thresholds must be ordered within [0, 1]")
        if not 0.0 <= self.valuable_question_threshold <= 1.0:
            raise ValueError("question value threshold must be within [0, 1]")
        if (
            self.clarification_horizon < 0
            or self.moderate_width < 1
            or self.low_value_width < 1
        ):
            raise ValueError("policy widths and horizons must be non-negative")

    def limit_for(
        self,
        turn: int,
        requested: int,
        *,
        scores: tuple[float, ...] = (),
        hard_constraint_coverage: float = 0.0,
        has_hard_constraints: bool = False,
        has_answerable_clarification: bool = False,
        clarification_expected_value: float | None = None,
        turns_remaining: int | None = None,
        has_intent_override: bool = False,
        has_no_preference_reply: bool = False,
    ) -> int:
        requested = max(1, min(int(requested), 10))
        if not self.adaptive or requested == 1 or len(scores) < 2:
            return requested

        remaining = max(0, 10 - int(turn)) if turns_remaining is None else max(0, turns_remaining)
        margin = self._relative_margin(scores)
        entropy = self._normalized_entropy(scores)
        constraints_satisfied = not has_hard_constraints or hard_constraint_coverage >= 1.0
        ambiguous = margin <= self.low_margin or entropy >= self.high_entropy
        incomplete_constraints = has_hard_constraints and hard_constraint_coverage < 1.0
        question_is_valuable = has_answerable_clarification and (
            clarification_expected_value is None
            or clarification_expected_value >= self.valuable_question_threshold
        )

        # A question with little expected ranking value should not hide viable
        # alternatives. Ambiguous rankings get full recall; otherwise expose a
        # useful shortlist immediately.
        if has_answerable_clarification and not question_is_valuable:
            if ambiguous or incomplete_constraints:
                return requested
            return min(requested, self.low_value_width)

        # Once the useful clarification budget is exhausted, use the remaining
        # turns for recall. A numerical leader should not suppress alternatives
        # when there is no customer answer left that can validate that lead.
        if not has_answerable_clarification:
            if (
                turn == 3
                and not has_intent_override
                and not has_no_preference_reply
            ):
                return min(requested, self.low_value_width)
            if remaining <= 6:
                return requested

        # A decisive leader that satisfies the active must-haves does not gain
        # anything from displaying weaker alternatives.
        if constraints_satisfied and margin >= self.high_margin and entropy <= self.low_entropy:
            return 1

        # Preserve a narrow answer while another useful customer answer can
        # still change the ranking. Near the turn limit, prefer recall instead.
        if question_is_valuable and remaining > self.clarification_horizon:
            return 1

        if ambiguous or incomplete_constraints:
            return requested

        return min(requested, self.moderate_width)

    @staticmethod
    def _relative_margin(scores: tuple[float, ...]) -> float:
        top, runner_up = scores[:2]
        return max(0.0, (top - runner_up) / max(abs(top), abs(runner_up), 1.0))

    @staticmethod
    def _normalized_entropy(scores: tuple[float, ...]) -> float:
        sample = scores[:10]
        if len(sample) < 2:
            return 0.0
        scale = max(abs(sample[0]), abs(sample[-1]), 1.0) * 0.05
        weights = [math.exp(max(-50.0, min(0.0, (score - sample[0]) / scale))) for score in sample]
        total = sum(weights)
        entropy = -sum((weight / total) * math.log(weight / total) for weight in weights)
        return entropy / math.log(len(weights))


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Runtime feature policy; the evaluated default is deterministic and offline."""

    enable_vector_reranker: bool = False
    recommendation_policy: RecommendationPolicy = RecommendationPolicy()
    full_width: bool = False


DEFAULT_AGENT_CONFIG = AgentConfig()
FULL_BREADTH_POLICY = RecommendationPolicy(adaptive=False)
FULL_WIDTH_CONFIG = AgentConfig(
    recommendation_policy=FULL_BREADTH_POLICY,
    full_width=True,
)
