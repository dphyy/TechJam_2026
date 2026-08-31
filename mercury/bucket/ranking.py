from __future__ import annotations

import math
import re
from dataclasses import dataclass

from mercury.lexical.dialogue import SessionState

from .index import CatalogIndex, FILLER, Product, normalized, tokens


MATERIALS = frozenset({
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon",
    "linen", "suede", "denim", "rubber", "canvas", "viscose", "acrylic", "elastane",
})
COLORS = frozenset({
    "black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey",
    "purple", "yellow", "orange", "navy", "beige", "cream", "gold", "silver",
})
OWNERS = frozenset({"shell", "lining", "upper", "sole", "outsole", "insole", "strap", "footbed"})
QUERY_FILLER = FILLER | {"that", "this", "have", "has", "be", "been", "also", "really",
                         "must", "should", "ideally", "ideal", "option", "options"}
PREFIX = re.compile(
    r"^(?:(?:for that|actually)[,:]?\s*)?(?:i\s+)?(?:would\s+)?"
    r"(?:prefer|need|want|require|am looking for|have a preference for)\s+", re.I,
)
AMOUNT = r"([0-9]+(?:\.[0-9]+)?)"


@dataclass(frozen=True, slots=True)
class Budget:
    lower: float | None = None
    upper: float | None = None
    lower_inclusive: bool = True
    upper_inclusive: bool = True
    target: float | None = None

    def match(self, price: float | None) -> tuple[int, float, bool]:
        if price is None:
            return 0, 0.0, False
        violation = (
            self.lower is not None
            and (price < self.lower or (price == self.lower and not self.lower_inclusive))
        ) or (
            self.upper is not None
            and (price > self.upper or (price == self.upper and not self.upper_inclusive))
        )
        if violation:
            return 0, 0.0, True
        if self.target is not None:
            distance = abs(price - self.target)
            return (4 if distance == 0 else 1), 1 / (1 + distance / max(self.target, 1)), False
        return 4, 1.0, False


def parse_budget(text: str) -> Budget | None:
    value = text.casefold().replace(",", "")
    if match := re.search(r"\bbetween\s*\$?" + AMOUNT + r"\s*(?:and|to|-)\s*\$?" + AMOUNT, value):
        low, high = map(float, match.groups())
        return Budget(low, high) if math.isfinite(low) and math.isfinite(high) else None
    pattern = (r"(under|below|less than|at most|up to|maximum|max|budget(?: of)?|"
               r"around|about|over|above|more than|at least|minimum|min|<=|>=|<|>)"
               r"\s*(?::|is|of)?\s*\$?" + AMOUNT)
    if not (match := re.search(pattern, value)):
        return None
    qualifier, amount = match.groups()
    number = float(amount)
    if not math.isfinite(number):
        return None
    if qualifier in {"around", "about"}:
        return Budget(target=number)
    if qualifier in {"over", "above", "more than", "at least", "minimum", "min", ">=", ">"}:
        return Budget(lower=number, lower_inclusive=qualifier in {"at least", "minimum", "min", ">="})
    return Budget(upper=number, upper_inclusive=qualifier not in {"under", "below", "less than", "<"})


@dataclass(frozen=True, slots=True)
class Branch:
    phrase: str
    words: frozenset[str]
    owner: str | None
    budget: Budget | None


@dataclass(frozen=True, slots=True)
class Requirement:
    branches: tuple[Branch, ...]
    hard: bool
    exclude: bool


def compile_requirements(state: SessionState) -> tuple[Requirement, ...]:
    requirements = []
    seen = set()
    for evidence in state.evidence:
        if evidence.source == "category":
            continue
        branches = []
        for raw in re.split(r"\s+or\s+", evidence.text, flags=re.I):
            raw = PREFIX.sub("", raw.strip())
            raw = re.sub(r"^either\s+", "", raw, flags=re.I)
            phrase = normalized(raw)
            words = frozenset(word for word in tokens(raw) if word not in QUERY_FILLER)
            if not words:
                continue
            owners = words & OWNERS
            branches.append(Branch(phrase, words, next(iter(owners)) if len(owners) == 1 else None,
                                   parse_budget(raw)))
        if not branches:
            continue
        hard = evidence.source in {"hard_constraint", "override", "exclusion"}
        requirement = Requirement(tuple(branches), hard, evidence.source == "exclusion")
        identity = (requirement.branches, requirement.exclude)
        if identity not in seen:
            seen.add(identity)
            requirements.append(requirement)
    return tuple(requirements)


def _scoped_fields(product: Product, owner: str | None) -> list[tuple[str, frozenset[str]]]:
    fields = []
    for (_, field), affirmed in zip(product.fields, product.affirmed_fields, strict=True):
        if owner is None:
            fields.append((field, affirmed))
            continue
        words = tokens(field)
        if owner not in words:
            continue
        # Keep each component's local span rather than joining attributes across
        # components. Both "lining cotton" and "cotton lining" are supported.
        owner_positions = [index for index, word in enumerate(words) if word in OWNERS]
        for index, position in enumerate(owner_positions):
            if words[position] != owner:
                continue
            stop = owner_positions[index + 1] if index + 1 < len(owner_positions) else len(words)
            local = words[position:stop]
            if position and (index == 0 or words[position - 1] in MATERIALS | COLORS):
                start = owner_positions[index - 1] + 1 if index else 0
                before = words[start:position]
                # A preceding property belongs here only for suffix-shaped
                # atoms. "shell cotton lining polyester" uses prefix labels.
                if stop == len(words) and position == len(words) - 1:
                    local = before + local
            text = " ".join(local)
            fields.append((text, frozenset(local) & affirmed))
    return fields


def match_branch(product: Product, branch: Branch) -> tuple[int, float, bool]:
    if branch.budget is not None:
        return branch.budget.match(product.price)
    fields = _scoped_fields(product, branch.owner)
    if not fields:
        return 0, 0.0, False
    all_positive = frozenset(word for _, affirmed in fields for word in affirmed)
    coverage = len(branch.words & all_positive) / len(branch.words)
    contradiction = False
    for family in (MATERIALS, COLORS):
        expected, observed = branch.words & family, all_positive & family
        if expected and observed and not expected <= observed:
            contradiction = True
    complete_fields = [field for field, affirmed in fields if branch.words <= affirmed]
    if branch.phrase in complete_fields:
        return 4, 1.0, contradiction
    if any(" " + branch.phrase + " " in " " + field + " " for field in complete_fields):
        return 3, 1.0, contradiction
    if complete_fields:
        return 2, 1.0, contradiction
    return (1 if coverage else 0), coverage, contradiction


@dataclass(frozen=True, slots=True)
class RankedProduct:
    identifier: str
    key: tuple[float, ...]
    evidence: dict


def rank_candidates(index: CatalogIndex, candidate_ids: list[str], category_ids: set[str],
                    requirements: tuple[Requirement, ...]) -> list[RankedProduct]:
    ranked = []
    for identifier in candidate_ids:
        product = index.products[identifier]
        violations = exact = phrases = complete = hard_exact = hard_phrases = hard_complete = 0
        coverage = hard_coverage = 0.0
        details = []
        for requirement in requirements:
            matches = [match_branch(product, branch) for branch in requirement.branches]
            if requirement.exclude:
                # Excluding an alternative disallows any matching branch.
                violated = any(level >= 2 for level, _, _ in matches)
                violations += int(violated)
                details.append({"excluded": True, "violation": violated})
                continue
            # A requested alternative is one requirement, satisfied by its
            # strongest noncontradictory branch.
            level, fraction, contradiction = max(matches, key=lambda item: (not item[2], item[0], item[1]))
            violations += int(contradiction)
            exact += int(level == 4)
            phrases += int(level >= 3)
            complete += int(level >= 2)
            coverage += fraction
            if requirement.hard:
                hard_exact += int(level == 4)
                hard_phrases += int(level >= 3)
                hard_complete += int(level >= 2)
                hard_coverage += fraction
            details.append({"level": level, "coverage": round(fraction, 6), "violation": contradiction})
        key = (float(identifier in category_ids), -float(violations), float(hard_complete),
               float(hard_exact), float(hard_phrases), hard_coverage, float(exact),
               float(phrases), float(complete), coverage, *product.quality)
        ranked.append(RankedProduct(identifier, key, {
            "category_match": identifier in category_ids, "violations": violations,
            "exact": exact, "phrase": phrases, "complete": complete,
            "coverage": round(coverage, 6), "requirements": details,
        }))
    return sorted(ranked, key=lambda item: (tuple(-part for part in item.key), item.identifier))
