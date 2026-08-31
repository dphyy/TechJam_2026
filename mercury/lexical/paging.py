"""Bounded, deterministic paging of the lexical candidate context."""
from __future__ import annotations

from dataclasses import dataclass
import math
import re

from .diagnostics import signature
from .feedback import HYPOTHETICAL, QUOTED, _semantic_value
from .product_features import component_scope, terms

@dataclass(frozen=True)
class ContextItem:
    identifier: str
    score: float
    violation: bool


OVERRIDE_PREFIX = re.compile(
    r"^(?:actually\b|instead\b|correction\b|(?:i\s+)?changed my mind\b|"
    r"(?:please\s+)?(?:make that|change(?: that| it)? to|ignore|no longer)\b|"
    r"(?:let me\s+)?correct that\b)", re.I,
)


def semantic_signature(state) -> str:
    values = {(component_scope(item.text), _semantic_value(item.text),
               item.source == 'exclusion', item.source in {'hard_constraint', 'override', 'exclusion'})
              for item in state.evidence if item.source != 'category'}
    return signature({'category': terms(state.category_text),
                      'values': sorted(values, key=repr)})


def explicit_override(message: str, state, turn: int) -> bool:
    # State events catch parsed replacements; wording also catches deduplicated
    # same-value overrides, which must replay potentially pre-override products.
    if any(item.turn == turn and item.source == 'override' for item in state.evidence):
        return True
    for clause in re.split(r'[.!?;\n]', QUOTED.sub('', message[:8000])):
        if not HYPOTHETICAL.search(clause) and OVERRIDE_PREFIX.search(clause.strip()):
            return True
    return False


@dataclass(frozen=True)
class PagingState:
    head: frozenset[str] = frozenset()
    semantic: str = ''
    seen: frozenset[str] = frozenset()
    shown: tuple[str, ...] = ()
    advances: int = 0


def select_page(raw: tuple[ContextItem, ...], base_ids: tuple[str, ...],
                semantic: str, override: bool, previous: PagingState | None,
                *, enabled: bool) -> tuple[tuple[str, ...], PagingState, dict]:
    """Pure selection: no query rewriting, scoring, hidden labels, or side effects."""
    identifiers = tuple(item.identifier for item in raw)
    by_id = {item.identifier: item for item in raw}
    if (len(raw) > 100 or len(by_id) != len(raw) or len(base_ids) > 10
            or len(set(base_ids)) != len(base_ids) or not set(base_ids) <= set(identifiers)
            or any(not isinstance(item.identifier, str) or not item.identifier
                   or type(item.score) not in (int, float) or not math.isfinite(item.score)
                   or type(item.violation) is not bool for item in raw)):
        raise ValueError('Invalid bounded candidate context')
    head = frozenset(identifiers[:10])
    reset = ('initial' if previous is None else 'explicit_override' if override else
             'semantic_change' if semantic != previous.semantic else None)
    seen = set() if reset else set(previous.seen)
    stable = bool(previous and head and head == previous.head)
    triggered = bool(enabled and not reset and stable and previous.shown and base_ids)
    chosen = base_ids
    reason = ('reset_base' if reset else 'unchanged_control' if not enabled else
              'empty_shortlist' if not base_ids else 'ranking_changed' if not stable else 'base_head')
    advances = 0 if reset or not stable else previous.advances
    if triggered:
        selected = []
        exhausted = False
        for violation in (False, True):
            quota = sum(by_id[key].violation == violation for key in base_ids)
            tier = [item.identifier for item in raw if item.violation == violation]
            unseen = [key for key in tier if key not in seen]
            exhausted |= quota > len(unseen)
            selected.extend((unseen + [key for key in tier if key in seen])[:quota])
        chosen = tuple(selected)
        advances += 1
        reason = 'unseen_exhausted_ranked_fill' if exhausted else 'highest_ranked_unseen'
    if (len(chosen) != len(base_ids) or len(set(chosen)) != len(chosen)
            or sum(by_id[key].violation for key in chosen) != sum(by_id[key].violation for key in base_ids)):
        raise ValueError('Paging changed width or known-violation quota')
    prior_seen = len(seen)
    repeated = sum(key in seen for key in chosen)
    seen.update(chosen)
    if len(seen) > 100:
        raise ValueError('Exposure exceeded the API session bound')
    next_state = PagingState(head, semantic, frozenset(seen), chosen, advances)
    receipt = {'enabled': enabled, 'triggered': triggered, 'reset': reset,
               'stable_head': stable, 'reason': reason, 'advances': advances,
               'prior_seen': prior_seen, 'repeated_exposures': repeated,
               'new_exposures': len(chosen) - repeated,
               'previous_ids': list(previous.shown) if previous else [],
               'base_ids': list(base_ids), 'returned_ids': list(chosen),
               'semantic_sha256': semantic, 'raw_context_sha256': signature(identifiers),
               'width_preserved': True, 'violation_quota_preserved': True,
               'reset_replayed_base': not reset or chosen == base_ids}
    return chosen, next_state, receipt

