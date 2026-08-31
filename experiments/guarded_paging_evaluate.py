"""Opt-in, presentation-only guarded paging for the frozen lexical agent."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
from pathlib import Path
import re
import socket
import statistics
import time
from unittest.mock import patch

from evaluator import local_evaluator as official
from experiments.presentation_evaluate import (
    ContextItem, HYPOTHETICAL, QUOTED, PresentationAgent, _semantic_value,
    explicit_slate_rejection,
)
from experiments.submission_evaluate import (
    EVALUATOR_SHA256, ObservedAgent, _aggregate, source_receipt,
)
from mercury.lexical.config import DEFAULT_AGENT_CONFIG
from mercury.lexical.diagnostics import constraint_receipts, signature, stage_receipt
from mercury.lexical.product_features import component_scope, terms
from mercury.model_assets import file_sha256

ROOT = Path(__file__).resolve().parents[1]
DATASETS = (
    ('public', 'data/public_set.jsonl'),
    ('cycle5-screening', 'artifacts/cycle5/synthetic-targets/screening.jsonl'),
    ('cycle5-confirmation', 'artifacts/cycle5/synthetic-targets/confirmation.jsonl'),
    ('cycle5-validation', 'artifacts/cycle5/synthetic-targets/validation.jsonl'),
    ('cycle3-screening', 'artifacts/cycle3/synthetic-targets/screening.jsonl'),
    ('robustness-v1-screening', 'artifacts/robustness-matrix-v1/screening.jsonl'),
)
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
            or len(set(base_ids)) != len(base_ids) or not set(base_ids) <= set(identifiers)):
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


class PagingAgent:
    """Experimental adapter; the shipped Agent and its state remain unchanged."""
    def __init__(self, catalog: Path, *, enabled: bool, inner=None):
        self.base = inner or PresentationAgent(catalog)
        self.enabled = enabled
        self._pages = {}
        self._cache = {}
        self._pending = {}
        self.last_diagnostics = {}
        self.startup_fallbacks = self.base.inner.startup_fallbacks

    def reset(self, session_id, profile):
        self.base.reset(session_id, profile)
        for mapping in (self._pages, self._cache, self._pending):
            mapping.pop(session_id, None)
            for key in list(mapping):
                if key not in self.base.inner._sessions:
                    del mapping[key]
        self.last_diagnostics = {}

    def respond(self, session_id, message, turn, top_k):
        request = (turn, message, top_k)
        cached = self._cache.get(session_id)
        if cached is not None and cached[0] == request:
            # Let the ordinary adapter validate lifecycle and mark a real retry.
            self.base.respond(session_id, message, turn, top_k)
            self.last_diagnostics = deepcopy(cached[2])
            self.last_diagnostics['cache_hit'] = True
            return deepcopy(cached[1])
        pending = self._pending.get(session_id)
        if pending is not None and pending[0] == request:
            _, base, diagnostic, raw, products, semantic, override = pending
        else:
            base = self.base.respond(session_id, message, turn, top_k)
            diagnostic = deepcopy(self.base.last_diagnostics)
            raw = self.base.observer.context
            products = self.base.observer.products
            state = self.base.inner._sessions[session_id]
            semantic = semantic_signature(state)
            override = explicit_override(message, state, turn)
            self._pending[session_id] = (request, base, diagnostic, raw, products, semantic, override)
        base_ids = tuple(item['parent_asin'] for item in base['recommendations'])
        selected, next_state, receipt = select_page(raw, base_ids, semantic, override,
                                                   self._pages.get(session_id), enabled=self.enabled)
        response = deepcopy(base)
        by_id = {item.identifier: item for item in raw}
        response['recommendations'] = [{'parent_asin': key, 'score': round(by_id[key].score, 6)}
                                       for key in selected]
        self.base.inner._validate_response(response, top_k)
        if response['message'] != base['message'] or response['ask_attribute'] != base['ask_attribute']:
            raise ValueError('Presentation changed the question')
        diagnostic = deepcopy(diagnostic)
        diagnostic['paging'] = receipt
        diagnostic['base_response'] = deepcopy(base)
        diagnostic['returned_ids'] = list(selected)
        diagnostic['raw_context_ids'] = [item.identifier for item in raw]
        diagnostic['known_violation_ids'] = [item.identifier for item in raw if item.violation]
        diagnostic['stage_receipts']['returned'] = stage_receipt(selected)
        diagnostic['stage_ids']['returned'] = list(selected)
        diagnostic['stage_counts']['returned'] = len(selected)
        diagnostic['output_width']['returned'] = len(selected)
        diagnostic['question_unchanged'] = True
        if selected != base_ids:
            product_map = {item['parent_asin']: item for item in products}
            diagnostic['constraint_checks'] = constraint_receipts(
                [product_map[key] for key in selected], self.base.inner._sessions[session_id].evidence)
            diagnostic['constraint_checks_origin'] = 'paging_shown_products'
        self._pages[session_id] = next_state
        self._cache[session_id] = (request, deepcopy(response), deepcopy(diagnostic))
        self._pending.pop(session_id, None)
        self.last_diagnostics = diagnostic
        return response

    def close(self):
        self.base.close()
        self._pages.clear()
        self._cache.clear()
        self._pending.clear()


def behavior_summary(traces: list[list[dict]], arm: str | None = None) -> dict:
    totals = Counter()
    unique_counts, longest = [], 0
    for session in traces:
        previous = ()
        streak = 0
        unique = set()
        for turn in session:
            if arm:
                current = turn[arm]
                ids, paging = tuple(current['ids']), current['paging']
                message = turn['message']
            else:
                if 'response' not in turn:
                    continue
                ids = tuple(row['parent_asin'] for row in turn['response']['recommendations'])
                paging = turn['diagnostics']['paging']
                message = turn['message']
            reset = paging['reset'] not in (None, 'initial')
            totals['turns'] += 1
            totals['shown_products'] += len(ids)
            totals['repeated_product_exposures'] += paging['repeated_exposures']
            totals['paging_triggers'] += int(paging['triggered'])
            totals['reset_turns'] += int(reset)
            totals['reset_replays_correct'] += int(reset and paging['reset_replayed_base'])
            unique.update(ids)
            if previous and ids:
                totals['comparable_adjacent_pairs'] += 1
                exact, same_set = ids == previous, set(ids) == set(previous)
                totals['exact_adjacent_repeats'] += int(exact)
                totals['set_adjacent_repeats'] += int(same_set)
                totals['nonreset_exact_repeats'] += int(exact and not reset)
                totals['reset_exact_repeats'] += int(exact and reset)
                if not reset:
                    totals['nonreset_adjacent_pairs'] += 1
                    if explicit_slate_rejection(message):
                        totals['explicit_rejection_turns'] += 1
                        totals['rejected_products_reshown'] += len(set(ids) & set(previous))
            streak = streak + 1 if ids and ids == previous and not reset else int(bool(ids))
            longest = max(longest, streak)
            previous = ids
        unique_counts.append(len(unique))
    return {**dict(totals), 'sessions': len(traces),
            'mean_unique_products_per_session': statistics.fmean(unique_counts) if unique_counts else 0,
            'longest_nonreset_exact_slate_streak': longest,
            'exact_repeat_rate': totals['exact_adjacent_repeats'] / max(totals['comparable_adjacent_pairs'], 1),
            'repeated_exposure_rate': totals['repeated_product_exposures'] / max(totals['shown_products'], 1)}


class CompactObserved(ObservedAgent):
    def __init__(self, agent, identifiers):
        super().__init__(agent, identifiers, False)

    def respond(self, session_id, message, turn, top_k):
        response = super().respond(session_id, message, turn, top_k)
        row = self.traces[-1][-1]
        diagnostic = row['diagnostics']
        row['diagnostics'] = {key: deepcopy(diagnostic[key]) for key in (
            'paging', 'base_response', 'raw_context_ids', 'known_violation_ids', 'fallbacks',
            'question_unchanged', 'stage_receipts', 'latency_seconds')}
        return response


def experiment_sources() -> dict:
    return {**source_receipt(), **{name: file_sha256(ROOT / name) for name in (
        'experiments/guarded_paging_evaluate.py', 'experiments/presentation_evaluate.py',
        'tests/test_guarded_paging_evaluate.py', 'docs/ADAPTIVE_GUARDED_PAGING_PROTOCOL.md')}}


def write(path: Path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def deny_network(*args, **kwargs):
    raise RuntimeError('Network disabled for paging evaluation')


@patch.object(socket, 'create_connection', deny_network)
@patch.object(socket.socket, 'connect', deny_network)
def run(catalog: Path, dataset: Path, output: Path, enabled: bool):
    if file_sha256(ROOT / 'evaluator/local_evaluator.py') != EVALUATOR_SHA256:
        raise ValueError('Evaluator changed')
    before = experiment_sources()
    output.mkdir(parents=True, exist_ok=False)
    registration = {'sources': before, 'config': asdict(DEFAULT_AGENT_CONFIG), 'paging_enabled': enabled,
                    'catalog_sha256': file_sha256(catalog), 'dataset_sha256': file_sha256(dataset),
                    'evaluator_sha256': EVALUATOR_SHA256, 'scope': 'consumed local synthetic/public evidence'}
    write(output / 'registration.json', registration)
    identifiers, categories, products = official.catalog_index(catalog)
    started = time.perf_counter()
    agent = PagingAgent(catalog, enabled=enabled)
    cold = time.perf_counter() - started
    observed = CompactObserved(agent, identifiers)
    try:
        started = time.perf_counter()
        result = official.evaluate(observed, official.load_jsonl(dataset), identifiers, categories, products)
        elapsed = time.perf_counter() - started
        if _aggregate(result['sessions']) != {key: result[key] for key in _aggregate(result['sessions'])}:
            raise ValueError('Aggregate mismatch')
        measurement = {'errors': observed.errors, 'startup_fallbacks': list(agent.startup_fallbacks),
                       'fallback_turns': sum(bool(t.get('diagnostics', {}).get('fallbacks'))
                                             for s in observed.traces for t in s),
                       'source_changed': before != experiment_sources(),
                       'catalog_changed': registration['catalog_sha256'] != file_sha256(catalog),
                       'dataset_changed': registration['dataset_sha256'] != file_sha256(dataset),
                       'cold_start_seconds': cold, 'evaluation_seconds': elapsed,
                       'p50_seconds': statistics.median(observed.latencies),
                       'p95_seconds': sorted(observed.latencies)[int((len(observed.latencies)-1)*.95)],
                       'widths': dict(observed.widths)}
        valid = not any(measurement[key] for key in ('errors', 'startup_fallbacks', 'fallback_turns',
                                                    'source_changed', 'catalog_changed', 'dataset_changed'))
        report = {**registration, 'valid': valid, 'result': result, 'measurement': measurement,
                  'behavior': behavior_summary(observed.traces)}
        write(output / 'report.json', report)
        with gzip.open(output / 'traces.json.gz', 'wt') as handle:
            json.dump(observed.traces, handle)
        if not valid:
            raise RuntimeError('Evaluation failed integrity checks')
        return report
    finally:
        agent.close()


@patch.object(socket, 'create_connection', deny_network)
@patch.object(socket.socket, 'connect', deny_network)
def replay(catalog: Path, dataset: Path, output: Path):
    samples = official.load_jsonl(dataset)
    selected = []
    for scenario, count in [('buying', 16), ('browsing', 16), ('intent_override', 6), ('boundary', 2)]:
        group = sorted((s for s in samples if s['scenario_type'] == scenario),
                       key=lambda s: hashlib.sha256(s['sample_id'].encode()).hexdigest())
        if len(group) < count:
            raise ValueError('Insufficient samples for the registered replay quota')
        selected.extend(group[:count])
    before = experiment_sources()
    output.mkdir(parents=True, exist_ok=False)
    registration = {'sources': before, 'catalog_sha256': file_sha256(catalog),
                    'dataset_sha256': file_sha256(dataset), 'sample_ids': [s['sample_id'] for s in selected],
                    'turns_per_session': 10, 'purpose': 'equal-history behavior audit, not official score'}
    write(output / 'registration.json', registration)
    _, categories, products = official.catalog_index(catalog)
    agent = PresentationAgent(catalog)
    traces = []
    fallback_turns = 0
    try:
        for sample in selected:
            card, behavior = official.materialize_hidden_fields(sample, products)
            effective = {**sample, 'intent_card': card, 'behavior': behavior}
            session_id = 'replay_' + hashlib.sha256(sample['sample_id'].encode()).hexdigest()
            agent.reset(session_id, sample['user_profile'])
            disclosed, boundary = set(), False
            message = official.initial_message(effective, official.coarse_category(
                categories[sample['ground_truth']['parent_asin']]), disclosed)
            states = {'control': None, 'paging': None}
            session = []
            for turn in range(1, 11):
                base = agent.respond(session_id, message, turn, 10)
                fallback_turns += bool(agent.last_diagnostics.get('fallbacks'))
                state = agent.inner._sessions[session_id]
                raw = agent.observer.context
                base_ids = tuple(item['parent_asin'] for item in base['recommendations'])
                row = {'turn': turn, 'message': message, 'question': base['ask_attribute'], 'base_ids': base_ids}
                for name, enabled in [('control', False), ('paging', True)]:
                    ids, states[name], receipt = select_page(raw, base_ids, semantic_signature(state),
                        explicit_override(message, state, turn), states[name], enabled=enabled)
                    row[name] = {'ids': list(ids), 'paging': receipt}
                if row['control']['ids'] != list(base_ids) or len(row['paging']['ids']) != len(base_ids):
                    raise ValueError('Replay changed control output or candidate width')
                session.append(row)
                override = behavior.get('override', {})
                if turn + 1 == override.get('turn'):
                    disclosed.add(str(override.get('new_value', '')))
                    message = override['message']
                else:
                    message, boundary = official.customer_reply(effective, base['ask_attribute'], disclosed, boundary)
            traces.append(session)
        if before != experiment_sources():
            raise ValueError('Sources changed during replay')
        if (registration['catalog_sha256'] != file_sha256(catalog)
                or registration['dataset_sha256'] != file_sha256(dataset)
                or agent.inner.startup_fallbacks or fallback_turns):
            raise ValueError('Replay failed data/fallback integrity checks')
        report = {**registration, 'control': behavior_summary(traces, 'control'),
                  'paging': behavior_summary(traces, 'paging'), 'traces': traces,
                  'fallback_turns': fallback_turns, 'valid': True,
                  'paired_result_width_and_question_identity': True}
        write(output / 'report.json', report)
        return report
    finally:
        agent.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=ROOT / 'data/catalog.jsonl')
    parser.add_argument('--dataset', type=Path, default=ROOT / 'data/public_set.jsonl')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=('control', 'paging', 'replay'), required=True)
    args = parser.parse_args()
    report = (replay(args.catalog, args.dataset, args.output) if args.arm == 'replay' else
              run(args.catalog, args.dataset, args.output, args.arm == 'paging'))
    print(json.dumps(report.get('result', report.get('paging', {})) if args.arm == 'replay' else
                     {k: report['result'][k] for k in ('recommended_technical_score', 'hit_rate_at_10', 'mrr', 'mttc')}))


if __name__ == '__main__':
    main()
