from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.guarded_paging_evaluate import (
    PagingAgent, explicit_override, select_page, semantic_signature,
)
from experiments.presentation_evaluate import ContextItem, PresentationAgent
from mercury.lexical.agent import Agent
from mercury.lexical.config import AgentConfig, RecommendationPolicy
from tests.test_presentation_evaluate import FixturePlanner, FixtureSearch, candidate


def context(count=20, unsafe=()):
    return tuple(ContextItem(str(i), float(count-i), i in unsafe) for i in range(count))


class PagingSelectionTest(unittest.TestCase):
    def first(self, raw=None, ids=('0', '1')):
        return select_page(raw or context(), ids, 'same', False, None, enabled=True)[1]

    def test_stalled_dialogue_advances_without_changing_width(self):
        raw = context()
        state = None
        pages = []
        for _ in range(10):
            ids, state, receipt = select_page(raw, ('0', '1'), 'same', False, state, enabled=True)
            pages.extend(ids)
            self.assertEqual(len(ids), 2)
            self.assertTrue(receipt['width_preserved'])
        self.assertEqual(len(set(pages)), 20)
        self.assertEqual(state.advances, 9)

    def test_control_never_pages_and_counts_repeated_exposures(self):
        ids, _, receipt = select_page(context(), ('0', '1'), 'same', False,
                                      self.first(), enabled=False)
        self.assertEqual(ids, ('0', '1'))
        self.assertEqual(receipt['repeated_exposures'], 2)

    def test_shuffle_within_same_head_still_pages(self):
        raw = context()
        shuffled = tuple(reversed(raw[:10])) + raw[10:]
        ids, _, receipt = select_page(shuffled, ('9', '8'), 'same', False,
                                      self.first(), enabled=True)
        self.assertEqual(ids, ('9', '8'))
        self.assertTrue(receipt['triggered'])

    def test_changed_head_replays_best_matches_and_retains_exposure(self):
        raw = context()
        changed = raw[1:] + raw[:1]
        ids, state, receipt = select_page(changed, ('1', '2'), 'same', False,
                                          self.first(), enabled=True)
        self.assertEqual(ids, ('1', '2'))
        self.assertFalse(receipt['triggered'])
        self.assertIn('0', state.seen)

    def test_semantic_change_and_same_value_override_reset_exposure(self):
        for semantic, override, reason in [('changed', False, 'semantic_change'),
                                            ('same', True, 'explicit_override')]:
            with self.subTest(reason=reason):
                state = self.first()
                _, state, _ = select_page(context(), ('0', '1'), 'same', False, state, enabled=True)
                ids, state, receipt = select_page(context(), ('0', '1'), semantic, override,
                                                  state, enabled=True)
                self.assertEqual(ids, ('0', '1'))
                self.assertEqual(state.seen, frozenset(('0', '1')))
                self.assertEqual(receipt['reset'], reason)

    def test_exhaustion_repeats_compatible_products_instead_of_violations(self):
        raw = context(4, unsafe=(2, 3))
        ids, _, receipt = select_page(raw, ('0', '1'), 'same', False,
                                      self.first(raw), enabled=True)
        self.assertEqual(ids, ('0', '1'))
        self.assertEqual(receipt['reason'], 'unseen_exhausted_ranked_fill')

    def test_partial_exhaustion_and_mixed_safety_quota(self):
        raw = context(5, unsafe=(3, 4))
        ids, _, _ = select_page(raw, ('0', '1', '3'), 'same', False,
                                self.first(raw, ('0', '1', '3')), enabled=True)
        self.assertEqual(ids, ('2', '0', '4'))
        self.assertEqual(sum(raw[int(key)].violation for key in ids), 1)

    def test_empty_context_or_shortlist_does_not_invent_recommendations(self):
        for raw in ((), context()):
            ids, _, _ = select_page(raw, (), 'same', False, None, enabled=True)
            self.assertEqual(ids, ())

    def test_widening_shortlist_uses_current_width(self):
        ids, _, _ = select_page(context(), ('0', '1', '2'), 'same', False,
                                self.first(ids=('0',)), enabled=True)
        self.assertEqual(ids, ('1', '2', '3'))

    def test_invalid_or_unbounded_context_is_rejected(self):
        for raw, base in [(context(101), ('0',)), (context() + context(1), ('0',)),
                          (context(), ('missing',)), (context(), ('0', '0'))]:
            with self.assertRaises(ValueError):
                select_page(raw, base, 'same', False, None, enabled=True)

    def test_signature_ignores_repetition_but_tracks_polarity_relaxation_and_category(self):
        positive = SimpleNamespace(text='cotton', source='hard_constraint', turn=1)
        negative = SimpleNamespace(text='cotton', source='exclusion', turn=2)
        def state(evidence, category='Shirts'):
            return SimpleNamespace(evidence=evidence, category_text=category)
        initial = semantic_signature(state([positive]))
        self.assertEqual(initial, semantic_signature(state([positive, positive])))
        for changed in (state([negative]), state([]), state([positive], 'Shoes')):
            self.assertNotEqual(initial, semantic_signature(changed))

    def test_explicit_override_excludes_quotes_and_hypotheticals(self):
        state = SimpleNamespace(evidence=[])
        for message in ('Actually, what I need is: cotton.', 'Correction: cotton.',
                        'I changed my mind. Blue please.'):
            self.assertTrue(explicit_override(message, state, 3))
        for message in ('Still browsing.', 'You said "Actually, cotton."',
                        'If I changed my mind, what would happen?'):
            self.assertFalse(explicit_override(message, state, 3))


class PagingAdapterTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.catalog = Path(directory.name) / 'catalog.jsonl'
        self.catalog.write_text('\n'.join(json.dumps({
            'parent_asin': str(i), 'title': 'Cotton shirt', 'categories': ['Shirts'],
            'features': ['cotton'], 'details': {'Color': 'blue'}, 'price': 10,
            'average_rating': 4, 'rating_number': 10,
        }) for i in range(20)))

    def fixture(self, *, enabled=True, adaptive=False, max_sessions=256, real_search=False):
        inner = Agent(self.catalog, max_sessions=max_sessions, config=AgentConfig(
            recommendation_policy=RecommendationPolicy(adaptive=adaptive)))
        if not real_search:
            inner.search.close()
            inner.search = FixtureSearch([candidate(str(i), i) for i in range(20)], inner.search)
            inner.question_planner = FixturePlanner()
        agent = PagingAgent(self.catalog, enabled=enabled, inner=PresentationAgent(self.catalog, inner=inner))
        self.addCleanup(agent.close)
        agent.reset('s', {})
        return agent

    def respond(self, agent, turn, message="I'm looking for Shirts.", sid='s'):
        return agent.respond(sid, message, turn, 2)

    def test_retry_returns_identical_page_without_an_extra_search(self):
        agent = self.fixture()
        self.respond(agent, 1)
        second = self.respond(agent, 2)
        calls = agent.base.observer.calls
        self.assertEqual(second, self.respond(agent, 2))
        self.assertEqual(calls, agent.base.observer.calls)
        self.assertEqual(agent._pages['s'].advances, 1)
        second['recommendations'].clear()
        self.assertEqual(len(self.respond(agent, 2)['recommendations']), 2)

    def test_failed_presentation_retry_does_not_skip_a_page(self):
        agent = self.fixture()
        self.respond(agent, 1)
        with patch('experiments.guarded_paging_evaluate.select_page', side_effect=RuntimeError('transient')):
            with self.assertRaises(RuntimeError):
                self.respond(agent, 2)
        self.assertEqual(agent._pages['s'].advances, 0)
        result = self.respond(agent, 2)
        self.assertEqual([p['parent_asin'] for p in result['recommendations']], ['2', '3'])
        self.assertEqual(agent.base.observer.calls, 2)

    def test_same_value_override_replays_top_after_paging(self):
        agent = self.fixture()
        self.respond(agent, 1, 'I prefer cotton.')
        self.respond(agent, 2, 'I prefer cotton.')
        result = self.respond(agent, 3, 'Actually, what I need is: cotton.')
        self.assertEqual([p['parent_asin'] for p in result['recommendations']], ['0', '1'])
        self.assertEqual(agent.last_diagnostics['paging']['reset'], 'explicit_override')

    def test_explicit_rejection_does_not_immediately_reshow_the_slate(self):
        agent = self.fixture()
        first = self.respond(agent, 1)
        second = self.respond(agent, 2, 'The options are not quite right.')
        a = {p['parent_asin'] for p in first['recommendations']}
        b = {p['parent_asin'] for p in second['recommendations']}
        self.assertFalse(a & b)

    def test_rejection_paraphrase_pages_without_false_preference_evidence(self):
        agent = self.fixture()
        first = self.respond(agent, 1)
        second = self.respond(agent, 2, "Those options aren't right.")
        self.assertNotEqual(first['recommendations'], second['recommendations'])
        self.assertIsNone(agent.last_diagnostics['paging']['reset'])
        self.assertFalse(any("aren't" in item.text for item in agent.base.inner._sessions['s'].evidence))
        third = self.respond(agent, 3, "Those options aren't right.")
        self.assertNotEqual(second['recommendations'], third['recommendations'])

    def test_sessions_are_isolated_resettable_and_evicted(self):
        agent = self.fixture(max_sessions=2)
        first = self.respond(agent, 1)
        self.respond(agent, 2)
        agent.reset('other', {})
        self.assertEqual(first, self.respond(agent, 1, sid='other'))
        agent.reset('s', {})
        self.assertEqual(first, self.respond(agent, 1))
        agent.reset('third', {})
        self.assertNotIn('other', agent._pages)
        self.assertLessEqual(len(agent._pages), 2)

    def test_stale_retry_and_closed_agent_are_rejected(self):
        agent = self.fixture()
        self.respond(agent, 1)
        self.respond(agent, 2)
        with self.assertRaises(ValueError):
            self.respond(agent, 1)
        agent.close()
        with self.assertRaises(RuntimeError):
            self.respond(agent, 2)

    def test_real_adaptive_search_preserves_control_questions_context_and_width(self):
        control = self.fixture(enabled=False, adaptive=True, real_search=True)
        paging = self.fixture(adaptive=True, real_search=True)
        for turn, message in enumerate([
            "I'm looking for Shirts.", 'A key requirement is: cotton.',
            'A key requirement is: cotton.', 'Actually, what I need is: blue.',
            'I have no preference for color.', 'No leather.',
            "Actually, I'm looking for Shoes instead.",
        ], 1):
            a, b = self.respond(control, turn, message), self.respond(paging, turn, message)
            for key in ('message', 'ask_attribute', 'usage'):
                self.assertEqual(a[key], b[key])
            self.assertEqual(len(a['recommendations']), len(b['recommendations']))
            self.assertEqual(control.last_diagnostics['base_response'], a)
            self.assertEqual(control.base.observer.context, paging.base.observer.context)
            receipt = paging.last_diagnostics['paging']
            self.assertTrue(receipt['violation_quota_preserved'])
            if receipt['reset']:
                self.assertEqual(a, b)


if __name__ == '__main__':
    unittest.main()
