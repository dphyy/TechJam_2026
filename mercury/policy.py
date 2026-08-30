"""Ablatable question selection and conservative, independent slate controls."""

from __future__ import annotations

import math
from collections import defaultdict

from mercury.config import Config
from mercury.ranking import rank_candidates
from mercury.state import SessionState
from mercury.types import Candidate, IntentDecision, PolicyDecision, Preference


_ATTRIBUTES = ("category", "material", "color", "size", "style", "brand",
               "budget", "feature", "use_case")
_QUESTIONS = {
    "category": "What type of item are you looking for?",
    "material": "Do you have a preferred material or fabric?",
    "color": "Do you have a preferred color?",
    "size": "What size should I look for?",
    "style": "Is there a particular style you prefer?",
    "brand": "Do you have a brand preference?",
    "budget": "What budget would you like me to work within?",
    "feature": "Which feature matters most to you?",
    "use_case": "What would you mainly use the item for?",
    "other": "What other detail matters most for finding the right item?",
}
_OTHER_QUESTIONS = (
    ("open_detail", _QUESTIONS["other"]),
    ("open_detail", "What else would help narrow down the right options?"),
    ("must_have_or_dealbreaker", "Is there another must-have feature or deal-breaker?"),
    ("priority", "What should I prioritize next: fit, use, or design?"),
    ("use_case", "Is there anything else about how you will use it?"),
    ("material_style_feature", "Would another material, style, or feature help decide?"),
    ("decisive_detail", "What remaining detail would make one option clearly better?"),
    ("constraint", "Is there another constraint I should keep in mind?"),
    ("final_preference", "What final preference should I use to refine these choices?"),
)
_MAX_POOL = 40
_MAX_ANSWERS = 4
_OUTSIDE_POOL_WEIGHT = 0.20
_NO_PREFERENCE_WEIGHT = 0.15
_ANSWER_WEIGHT = 1.0 - _OUTSIDE_POOL_WEIGHT - _NO_PREFERENCE_WEIGHT


def _eligible_attributes(state: SessionState) -> list[str]:
    known = {preference.attribute for preference in state.active_preferences()}
    return [attribute for attribute in _ATTRIBUTES
            if attribute not in known and attribute not in state.unproductive_attributes
            and not state.asked_counts.get(attribute, 0)]


def _other_entry(state: SessionState, config: Config) -> tuple[str, str] | None:
    limit = min(config.other_question_limit, len(_OTHER_QUESTIONS))
    if "other" in state.unproductive_attributes or state.asked_counts.get("other", 0) >= limit:
        return None
    if config.semantic_question_goals:
        return next((entry for entry in _OTHER_QUESTIONS[:limit]
                     if f"other:{entry[0]}" not in state.asked_question_goals), None)
    return _OTHER_QUESTIONS[min(state.asked_counts.get("other", 0), limit - 1)] if limit else None


def _question_goal(question: str | None, state: SessionState, config: Config) -> str | None:
    if question is None:
        return None
    if question != "other":
        return f"facet:{question}"
    entry = _other_entry(state, config)
    return f"other:{entry[0]}" if entry else None


def _question_message(question: str | None, state: SessionState, config: Config) -> str | None:
    if question != "other":
        return _QUESTIONS.get(question) if question is not None else None
    entry = _other_entry(state, config)
    return entry[1] if entry else None


def _uninformative_answer(state: SessionState) -> bool:
    return state.last_question is not None and (
        state.last_question in state.unproductive_attributes
        or (bool(state.history) and not state.last_update_informative)
    )


def _fallback_question(state: SessionState, eligible: list[str], config: Config) -> str | None:
    if _other_entry(state, config) is not None and not (
            state.last_question == "other" and _uninformative_answer(state)):
        return "other"
    return eligible[0] if eligible else None


def _answers(candidate: Candidate, attribute: str) -> tuple[str, ...]:
    product = candidate.product
    if attribute == "budget":
        # A 'from' price is not an exact price or proof of affordability.
        if product.price is None or product.price_lower_bound or not math.isfinite(product.price):
            return ()
        return (f"<= {product.price:g}",)
    return tuple(sorted({value.strip().lower() for value in product.facets.get(attribute, ())
                         if value.strip()}))


def _entropy(pool: list[Candidate], attribute: str) -> tuple[float, float]:
    if not pool:
        return 0.0, 1.0
    mass: dict[str | None, float] = defaultdict(float)
    for candidate in pool:
        answers = _answers(candidate, attribute)
        if not answers:
            mass[None] += 1.0 / len(pool)
        else:
            for answer in answers:
                mass[answer] += 1.0 / len(pool) / len(answers)
    entropy = -sum(weight * math.log2(weight) for weight in mass.values() if weight > 0)
    return max(0.0, entropy), mass.get(None, 0.0)


def _rank_weights(pool: list[Candidate]) -> list[float]:
    """A rank-only heuristic, never a calibrated probability of correctness."""
    raw = [1.0 / math.sqrt(index + 1) for index in range(len(pool))]
    total = sum(raw)
    return [weight / total for weight in raw]


def _outcome_model(pool: list[Candidate], attribute: str, reference_size: int, turn: int) -> dict:
    weights = _rank_weights(pool)
    answer_targets: dict[str, list[tuple[int, float]]] = defaultdict(list)
    unknown_mass = 0.0
    for index, (candidate, weight) in enumerate(zip(pool, weights)):
        answers = _answers(candidate, attribute)
        if not answers:
            unknown_mass += weight
        else:
            for answer in answers:
                answer_targets[answer].append((index, weight / len(answers)))

    def answer_priority(answer: str) -> tuple[float, float, str]:
        targets = answer_targets[answer]
        # Spend bounded simulations on answers that could recover an item not
        # shown this turn. The prior itself never conditions on earlier misses.
        residual = sum(weight for index, weight in targets if index >= reference_size)
        return -residual, -sum(weight for _, weight in targets), answer

    selected = sorted(answer_targets, key=answer_priority)[:_MAX_ANSWERS]
    modeled_mass = sum(weight for answer in selected for _, weight in answer_targets[answer])
    unmodeled_mass = max(0.0, 1.0 - unknown_mass - modeled_mass) if pool else 0.0
    stable_prefixes = set(range(3, min(reference_size, len(pool))))
    original_ids = [candidate.product.parent_asin for candidate in pool]
    outcomes = []
    for answer in selected:
        hypothetical = Preference(attribute, answer, turn, "catalog-grounded hypothetical answer")
        reranked = rank_candidates(pool, [hypothetical])
        ranks = {candidate.product.parent_asin: rank for rank, candidate in enumerate(reranked, 1)}
        reranked_ids = [candidate.product.parent_asin for candidate in reranked]
        stable_prefixes = {size for size in stable_prefixes
                           if set(original_ids[:size]) == set(reranked_ids[:size])}
        rr_gain = 0.0
        recovery_gain = 0.0
        for index, weight in answer_targets[answer]:
            # Recommendations and the question share the current turn. Items
            # already in the reference slate need no later answer to be found.
            if index < reference_size:
                continue
            rank = ranks[original_ids[index]]
            if rank <= reference_size:
                rr_gain += _ANSWER_WEIGHT * weight / rank
                recovery_gain += _ANSWER_WEIGHT * weight
        outcomes.append({
            "kind": "value", "answer": answer,
            "weight": _ANSWER_WEIGHT * sum(weight for _, weight in answer_targets[answer]),
            "expected_rr_gain": rr_gain, "expected_recovery_gain": recovery_gain,
        })
    # Unknown/no-preference/unmodeled answers preserve the existing ranking.
    # Outside-pool targets have explicitly unresolved recovery, not an invented
    # benefit from asking. All masses below are modeling assumptions.
    for kind, mass in (("unknown", unknown_mass * _ANSWER_WEIGHT),
                       ("unmodeled_value", unmodeled_mass * _ANSWER_WEIGHT),
                       ("no_preference", _NO_PREFERENCE_WEIGHT),
                       ("outside_pool", _OUTSIDE_POOL_WEIGHT)):
        outcomes.append({"kind": kind, "weight": mass,
                         "expected_rr_gain": 0.0, "expected_recovery_gain": 0.0})
    if not pool:
        outcomes[0]["weight"] = _ANSWER_WEIGHT
    rr_gain = sum(outcome["expected_rr_gain"] for outcome in outcomes)
    recovery_gain = sum(outcome["expected_recovery_gain"] for outcome in outcomes)
    return {
        "outcomes": outcomes,
        "expected_rr_gain": rr_gain,
        "expected_recovery_gain": recovery_gain,
        "value": 0.30 * rr_gain + 0.50 * recovery_gain,
        "modeled_answer_mass": modeled_mass * _ANSWER_WEIGHT,
        "unmodeled_answer_mass": unmodeled_mass * _ANSWER_WEIGHT,
        "stable_prefixes": sorted(stable_prefixes) if selected else [],
    }


def _gap_boundary(candidates: list[Candidate], size: int) -> int | None:
    scores = [candidate.score for candidate in candidates[:size]]
    if len(scores) < 4 or not all(math.isfinite(score) for score in scores):
        return None
    if any(left < right for left, right in zip(scores, scores[1:])):
        return None
    spread = scores[0] - scores[-1]
    if spread <= 0:
        return None
    for boundary in range(3, len(scores)):
        if (scores[boundary - 1] - scores[boundary]) / spread >= 0.70:
            return boundary
    return None


def _slate_size(state: SessionState, candidates: list[Candidate], config: Config,
                turn: int, capacity: int, previous_abstentions: int,
                question: str | None, diagnostics: dict) -> tuple[int, str]:
    if capacity == 0:
        return 0, "no_available_recommendation_capacity"
    if turn >= 10:
        return capacity, "final_turn_full_slate"
    if previous_abstentions >= 1:
        return capacity, "consecutive_abstention_guard"
    if _uninformative_answer(state):
        return capacity, "uninformative_answer_recovery"
    size = min(capacity, config.slate_size)
    if config.slate_policy == "fixed" or size == 0:
        return size, "configured_fixed_size"
    boundary = _gap_boundary(candidates, size)
    if boundary is None:
        return size, "no_decisive_gap"
    if config.slate_policy == "gap":
        return boundary, "heuristic_score_gap"
    if question not in _ATTRIBUTES:
        return size, "no_modeled_question"
    models = diagnostics.setdefault("outcome_models", {})
    if question not in models:
        models[question] = _outcome_model(candidates[:_MAX_POOL], question, capacity, turn)
    model = models[question]
    if (boundary in model["stable_prefixes"] and model["modeled_answer_mass"] >= 0.50
            and model["unmodeled_answer_mass"] <= 1e-12):
        return boundary, "stable_answer_prefix"
    return size, "answer_uncertainty_keeps_full_slate"


def choose_policy(state: SessionState, candidates: list[Candidate], config: Config,
                  turn: int, top_k: int, previous_abstentions: int = 0,
                  intent: IntentDecision | None = None) -> PolicyDecision:
    """Choose one allowed question and a slate size without mutating session state.

    Scores model incremental recovery after the current full reference slate,
    not information entropy or a cost for asking. They are heuristic estimates
    over a bounded retrieved pool, not probabilities or expected public scores.
    Adaptive shortening is opt-in; only explicit configuration can abstain.
    """
    capacity = min(10, max(0, top_k), len(candidates))
    eligible = _eligible_attributes(state)
    pool = candidates[:_MAX_POOL]
    diagnostics = {
        "question_policy": config.question_policy,
        "slate_policy": config.slate_policy,
        "pool_size": len(pool),
        "unmodeled_candidates": max(0, len(candidates) - len(pool)),
        "reference_slate_size": capacity,
        "ask_turn_cost": config.question_turn_cost if config.question_policy == "intent" else 0.0,
        "weight_model": "heuristic_rank_prior_not_calibrated",
        "uncertainty_assumptions": {"outside_pool": _OUTSIDE_POOL_WEIGHT,
                                    "no_preference": _NO_PREFERENCE_WEIGHT},
        "already_asked": sorted(attribute for attribute, count in state.asked_counts.items() if count),
    }
    question = None
    if turn < 10 and config.question_policy != "none":
        if config.question_policy == "schedule":
            question = eligible[0] if eligible else _fallback_question(state, eligible, config)
        elif config.question_policy == "entropy":
            entropy = {attribute: _entropy(pool, attribute) for attribute in eligible}
            diagnostics["facet_entropy"] = {attribute: value[0] for attribute, value in entropy.items()}
            diagnostics["unknown_mass"] = {attribute: value[1] for attribute, value in entropy.items()}
            best = max(eligible, key=lambda attribute: entropy[attribute][0], default=None)
            question = best if best is not None and entropy[best][0] > 1e-12 else _fallback_question(state, eligible, config)
        elif config.question_policy == "rank_value":
            models = {attribute: _outcome_model(pool, attribute, capacity, turn) for attribute in eligible}
            diagnostics["outcome_models"] = models
            diagnostics["question_scores"] = {attribute: model["value"] for attribute, model in models.items()}
            diagnostics["objective"] = "0.30 * expected_rr_gain + 0.50 * expected_recovery_gain"
            best = max(eligible, key=lambda attribute: models[attribute]["value"], default=None)
            question = best if best is not None and models[best]["value"] > 1e-12 else _fallback_question(state, eligible, config)
        elif config.question_policy == "intent":
            mode = intent.effective_mode if intent is not None else "mixed"
            diagnostics["intent_mode"] = mode
            diagnostics["over_general"] = bool(intent and intent.over_general)
            models = {attribute: _outcome_model(pool, attribute, capacity, turn) for attribute in eligible}
            scores = {attribute: model["value"] - config.question_turn_cost
                      for attribute, model in models.items()}
            diagnostics["outcome_models"] = models
            diagnostics["question_scores"] = scores
            diagnostics["objective"] = "expected rank/recovery value minus explicit turn cost"
            if mode == "browsing" and "category" in eligible:
                question = "category"
                diagnostics["decision"] = "browsing_product_type"
            elif mode == "browsing" and "use_case" in eligible:
                question = "use_case"
                diagnostics["decision"] = "browsing_use_case"
            else:
                best = max(eligible, key=lambda attribute: scores[attribute], default=None)
                if best is not None and scores[best] > 0:
                    question = best
                    diagnostics["decision"] = "highest_expected_value"
                else:
                    entropy = {attribute: _entropy(pool, attribute)[0] for attribute in eligible}
                    best = max(eligible, key=lambda attribute: entropy[attribute], default=None)
                    question = best if best is not None and entropy[best] > 0 else _fallback_question(
                        state, eligible, config,
                    )
                    diagnostics["decision"] = "facet_uncertainty" if best and entropy[best] > 0 else "bounded_fallback"
            if config.require_positive_question_value and (
                    question == "other" or question is None or scores.get(question, 0.0) <= 0):
                question = None
                diagnostics["decision"] = "question_value_below_turn_cost"
        else:
            question = _fallback_question(state, eligible, config)
    goal = _question_goal(question, state, config)
    if question is not None and goal is None:
        question = None
    size, reason = _slate_size(state, candidates, config, turn, capacity,
                               previous_abstentions, question, diagnostics)
    diagnostics["slate_reason"] = reason
    diagnostics["question_goal"] = goal
    diagnostics["previous_answer_productivity"] = state.last_answer_productivity
    message = _question_message(question, state, config) if question else (
        "Here are the current options." if size else "I don't have a recommendation to show on this turn."
    )
    return PolicyDecision(question, message, size, diagnostics, goal)
