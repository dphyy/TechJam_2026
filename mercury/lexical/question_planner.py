from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from .dialogue import SessionState
from .product_features import FACET_ORDER, ProductFeatures, ProductFeatureStore


# Additional Information for agent to filter by.
ANSWERABILITY_PRIORS = {
    "feature": 1.00,
    "material": 0.95,
    "color": 0.90,
    "budget": 0.80,
    "size": 0.70,
    "style": 0.65,
    "use_case": 0.60,
    "category": 0.30,
    "brand": 0.20,
}
EARLY_OPEN_QUESTION_LIMIT = 2
DEFAULT_SCORE_TEMPERATURE = 5.0


@dataclass(frozen=True)
class FacetScore:
    attribute: str
    information_gain: float
    examples: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class QuestionPlan:
    """A clarification question and the value the planner expects from it."""

    attribute: str | None
    message: str
    information_gain: float
    answerability: float
    expected_value: float


class AdaptiveQuestionPlanner:
    """Select clarification facets from candidate-pool information gain."""

    def __init__(
        self,
        feature_store: ProductFeatureStore,
        score_temperature: float = DEFAULT_SCORE_TEMPERATURE,
    ) -> None:
        if not math.isfinite(score_temperature) or score_temperature <= 0.0:
            raise ValueError("score_temperature must be finite and positive")
        self.feature_store = feature_store
        self.score_temperature = score_temperature

    def choose(
        self,
        state: SessionState,
        candidates: list[dict],
        turn: int,
    ) -> QuestionPlan:
        if turn >= 10 or not candidates:
            return self._no_question()

        facet_scores = self._score_facets(candidates)
        available = [
            facet
            for facet in facet_scores
            if facet.attribute not in state.no_preference_attributes
        ]

        # Cap to avoid repetition and honor an explicit no-preference reply.
        if (
            turn <= 3
            and "other" not in state.no_preference_attributes
            and state.asked_attributes.count("other") < EARLY_OPEN_QUESTION_LIMIT
        ):
            proxy = max(available, key=lambda facet: facet.information_gain, default=None)
            information_gain = proxy.information_gain if proxy else 0.0
            answerability = self._answerability("other", state)
            state.record_question("other")
            return QuestionPlan(
                "other",
                self._word_question("other", ()),
                information_gain,
                answerability,
                information_gain * answerability,
            )

        if not available:
            return self._no_question()

        adjusted = [
            FacetScore(
                facet.attribute,
                facet.information_gain
                * ANSWERABILITY_PRIORS.get(facet.attribute, 0.50)
                / (1.0 + 0.85 * state.asked_attributes.count(facet.attribute)),
                facet.examples,
            )
            for facet in available
        ]
        adjusted.sort(key=lambda facet: (-facet.information_gain, facet.attribute))

        attribute = adjusted[0].attribute
        top_facets = adjusted[:3]
        if self._needs_open_question(candidates, top_facets, state):
            attribute = "other"
            examples = tuple(facet.attribute.replace("_", " ") for facet in top_facets[:2])
            raw_facet = next(
                facet for facet in available if facet.attribute == adjusted[0].attribute
            )
        else:
            examples = adjusted[0].examples
            raw_facet = next(facet for facet in available if facet.attribute == attribute)

        answerability = self._answerability(attribute, state)
        state.record_question(attribute)
        return QuestionPlan(
            attribute,
            self._word_question(attribute, examples),
            raw_facet.information_gain,
            answerability,
            raw_facet.information_gain * answerability,
        )

    @staticmethod
    def _answerability(attribute: str, state: SessionState) -> float:
        return ANSWERABILITY_PRIORS.get(attribute, 0.50) / (
            1.0 + 0.85 * state.asked_attributes.count(attribute)
        )

    @staticmethod
    def _no_question() -> QuestionPlan:
        return QuestionPlan(
            None,
            "These are my best matches based on everything you've shared.",
            0.0,
            0.0,
            0.0,
        )

    def _score_facets(self, candidates: list[dict]) -> list[FacetScore]:
        observations: dict[str, list[tuple[str, ...]]] = {
            attribute: []
            for attribute in (*FACET_ORDER, "budget", "brand", "category", "feature")
        }

        documents = [self._features(product) for product in candidates]
        feature_frequency = Counter(
            token
            for document in documents
            for token in document.feature_tokens
        )

        prices = sorted(
            document.price
            for document in documents
            if document.price is not None
        )
        price_cuts = self._quartiles(prices)

        for product, document in zip(candidates, documents):
            question_features = self.feature_store.question_features(product)
            for attribute in FACET_ORDER:
                observations[attribute].append(
                    question_features.facet_values(attribute)
                )

            observations["budget"].append(
                self._budget_bucket(document.price, price_cuts)
            )
            observations["brand"].append(
                (document.brand,) if document.brand else ()
            )

            category = " ".join(
                document.category_tokens[-3:]
            )
            observations["category"].append((category,) if category else ())

            feature_values = sorted(
                document.feature_tokens,
                key=lambda token: (feature_frequency[token], token),
            )[:2]
            observations["feature"].append(
                tuple(feature_values)
            )

        scores = [float(product.get("_rank_score") or 0.0) for product in candidates]
        probabilities = self._score_probabilities(scores)
        return [
            self._information_gain(attribute, values, probabilities)
            for attribute, values in observations.items()
        ]

    @staticmethod
    def _features(product: dict) -> ProductFeatures:
        features = product.get("_features")
        if not isinstance(features, ProductFeatures):
            raise TypeError("candidate is missing precomputed ProductFeatures")
        return features

    def _information_gain(
        self,
        attribute: str,
        observations: list[tuple[str, ...]],
        probabilities: list[float] | None = None,
    ) -> FacetScore:
        if not observations:
            return FacetScore(attribute, 0.0, ())
        if probabilities is None:
            probabilities = [1.0 / len(observations)] * len(observations)
        if len(probabilities) != len(observations):
            raise ValueError("each observation must have a candidate probability")

        signatures = [" / ".join(values) if values else "<unknown>" for values in observations]
        grouped_probabilities: dict[str, list[float]] = {}
        for signature, probability in zip(signatures, probabilities):
            grouped_probabilities.setdefault(signature, []).append(probability)

        # A known facet answer deterministically partitions the candidate
        # distribution. Missing catalog values are not useful customer answers,
        # so they contribute no reduction rather than becoming a fake answer.
        prior_entropy = self._entropy(probabilities)
        signature_masses: dict[str, float] = {}
        for signature, group in grouped_probabilities.items():
            mass = sum(group)
            signature_masses[signature] = mass
        known_masses = [
            mass
            for signature, mass in signature_masses.items()
            if signature != "<unknown>" and mass > 0.0
        ]
        coverage = sum(known_masses)
        entropy_reduction = (
            coverage * self._entropy([mass / coverage for mass in known_masses])
            if coverage > 0.0
            else 0.0
        )
        information_gain = min(prior_entropy, entropy_reduction)
        examples = tuple(
            value
            for value, _ in sorted(
                signature_masses.items(), key=lambda item: -item[1]
            )
            if value != "<unknown>"
        )[:3]
        return FacetScore(attribute, information_gain, examples)

    def _score_probabilities(self, scores: list[float]) -> list[float]:
        if not scores:
            return []
        maximum = max(scores)
        weights = [
            math.exp(max(-700.0, (score - maximum) / self.score_temperature))
            for score in scores
        ]
        total = sum(weights)
        return [weight / total for weight in weights]

    @staticmethod
    def _entropy(probabilities: list[float]) -> float:
        return -sum(
            probability * math.log(probability)
            for probability in probabilities
            if probability > 0.0
        )

    @staticmethod
    def _needs_open_question(
        candidates: list[dict], facets: list[FacetScore], state: SessionState
    ) -> bool:
        if len(facets) < 2:
            return False
        scores = [float(product.get("_rank_score") or 0.0) for product in candidates[:10]]
        relevance_spread = (
            (scores[0] - scores[-1]) / max(abs(scores[0]), 1.0)
            if len(scores) >= 2
            else 1.0
        )
        facet_competition = facets[1].information_gain / max(facets[0].information_gain, 1e-9)
        broad_uncertainty = relevance_spread < 0.20 and facet_competition > 0.72
        repeated_penalty = 1.0 + 0.70 * state.asked_attributes.count("other")
        return broad_uncertainty and facet_competition / repeated_penalty > 0.40

    @staticmethod
    def _word_question(attribute: str, examples: tuple[str, ...]) -> str:
        if attribute == "other":
            dimensions = " and ".join(examples) if examples else "several details"
            return (
                f"The closest matches vary across {dimensions}. "
                "What must-have detail should I prioritize to narrow them down?"
            )
        label = attribute.replace("_", " ")
        usable_examples = [value for value in examples if len(value) <= 28][:3]
        example_text = f"—for example, {', '.join(usable_examples)}" if usable_examples else ""
        return (
            f"The closest matches differ by {label}{example_text}. "
            f"Which {label} best fits what you need?"
        )

    @staticmethod
    def _quartiles(values: list[float]) -> tuple[float, float, float] | None:
        if len(values) < 4:
            return None
        return (
            values[len(values) // 4],
            values[len(values) // 2],
            values[(3 * len(values)) // 4],
        )

    @staticmethod
    def _budget_bucket(
        value: object, cuts: tuple[float, float, float] | None
    ) -> tuple[str, ...]:
        if cuts is None or value is None:
            return ()
        bucket = sum(value > cut for cut in cuts) + 1
        return (f"price group {bucket}",)
