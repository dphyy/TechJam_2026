"""Authored runtime regressions; no evaluation targets or held-out labels."""
from dataclasses import replace
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent import Agent
from mercury.lexical.config import DEFAULT_AGENT_CONFIG, FULL_WIDTH_CONFIG, RecommendationPolicy
from mercury.lexical.dialogue import SessionState
from mercury.lexical.feedback import explicit_slate_rejection, preference_content
from mercury.lexical.paging import semantic_signature
from tests.test_presentation_evaluate import FixturePlanner, FixtureSearch, candidate


class LexicalPagingTest(unittest.TestCase):
    def catalog(self, rows):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / 'catalog.jsonl'
        path.write_text('\n'.join(json.dumps(row) for row in rows))
        return path

    def agent(self, *, fixture=False, **kwargs):
        path = self.catalog([{'parent_asin': str(i), 'title': 'Cotton shirt',
                             'categories': ['Shirts'], 'features': ['cotton'],
                             'price': 10, 'average_rating': 4, 'rating_number': 10}
                            for i in range(20)])
        agent = Agent(path, **kwargs)
        self.addCleanup(agent.close)
        if fixture:
            agent.search.close()
            agent.search = FixtureSearch([candidate(str(i), i) for i in range(20)], agent.search)
            agent.question_planner = FixturePlanner()
        agent.reset('s', {})
        return agent

    @staticmethod
    def ids(response):
        return [row['parent_asin'] for row in response['recommendations']]

    def test_public_entry_pages_and_retry_does_not_advance(self):
        agent = self.agent()
        self.assertTrue(agent.config.guarded_paging)
        first = agent.respond('s', "I'm looking for Shirts.", 1, 2)
        second = agent.respond('s', 'Those options aren\'t right.', 2, 2)
        self.assertFalse(set(self.ids(first)) & set(self.ids(second)))
        before = agent._pages['s']
        self.assertEqual(second, agent.respond('s', "Those options aren't right.", 2, 2))
        self.assertEqual(before, agent._pages['s'])
        self.assertTrue(agent.last_diagnostics['cache_hit'])
        self.assertEqual(self.ids(second), agent.last_diagnostics['stage_ids']['returned'])

    def test_failed_turn_does_not_commit_page_dialogue_or_profile(self):
        agent = self.agent(fixture=True, config=replace(DEFAULT_AGENT_CONFIG,
                           recommendation_policy=RecommendationPolicy(adaptive=False)))
        agent.respond('s', "I'm looking for Shirts.", 1, 2)
        state = deepcopy(agent._sessions['s'])
        page = agent._pages['s']
        with patch('mercury.lexical.agent.turn_receipt', side_effect=RuntimeError('transient')):
            with self.assertRaises(RuntimeError):
                agent.respond('s', "Those options aren't right.", 2, 2)
        self.assertEqual(state, agent._sessions['s'])
        self.assertEqual(page, agent._pages['s'])
        response = agent.respond('s', "Those options aren't right.", 2, 2)
        self.assertEqual(self.ids(response), ['2', '3'])
        self.assertEqual(agent._pages['s'].advances, 1)

    def test_ten_turns_unique_until_compatible_pool_exhausted(self):
        agent = self.agent(fixture=True, config=replace(DEFAULT_AGENT_CONFIG,
                           recommendation_policy=RecommendationPolicy(adaptive=False)))
        seen = []
        for turn in range(1, 11):
            seen.extend(self.ids(agent.respond('s', "I'm looking for Shirts.", turn, 2)))
        self.assertEqual(len(set(seen)), 20)

    def test_explicit_same_value_override_resets_and_replays_best(self):
        agent = self.agent(fixture=True, config=replace(DEFAULT_AGENT_CONFIG,
                           recommendation_policy=RecommendationPolicy(adaptive=False)))
        first = agent.respond('s', 'I prefer cotton.', 1, 2)
        agent.respond('s', 'I prefer cotton.', 2, 2)
        self.assertEqual(self.ids(first), self.ids(agent.respond('s', 'Correction: cotton.', 3, 2)))
        self.assertEqual(agent.last_diagnostics['paging']['reset'], 'explicit_override')

    def test_reset_forget_eviction_and_close_clear_paging(self):
        agent = self.agent(max_sessions=1)
        agent.respond('s', "I'm looking for Shirts.", 1, 2)
        agent.forget_profile('s')
        self.assertNotIn('s', agent._pages)
        agent.respond('s', "I'm looking for Shirts.", 2, 2)
        agent.reset('next', {})
        self.assertNotIn('s', agent._pages)
        agent.close()
        self.assertEqual(agent._pages, {})
        with self.assertRaises(RuntimeError):
            agent.reset('closed', {})
        with self.assertRaises(RuntimeError):
            agent.respond('next', 'cotton', 1, 2)

    def test_full_width_control_is_unpaged(self):
        agent = self.agent(config=FULL_WIDTH_CONFIG)
        a = agent.respond('s', "I'm looking for Shirts.", 1, 10)
        b = agent.respond('s', "I'm looking for Shirts.", 2, 10)
        self.assertEqual(a['recommendations'], b['recommendations'])
        self.assertFalse(agent.last_diagnostics['paging']['enabled'])
        self.assertFalse(agent._pages)

    def test_declined_open_question_is_not_reopened_on_balanced_facets(self):
        rows = [{'parent_asin': str(i), 'title': 'Everyday shirt', 'categories': ['Clothing', 'Shirts'],
                 'price': 10, 'average_rating': 4, 'rating_number': 10,
                 'details': {'Material': ('cotton', 'linen', 'wool', 'silk')[i % 4],
                             'Color': ('red', 'blue', 'green', 'black')[i % 4]}} for i in range(16)]
        agent = Agent(self.catalog(rows))
        self.addCleanup(agent.close)
        agent.reset('s', {})
        self.assertEqual(agent.respond('s', "I'm looking for Shirts.", 1, 10)['ask_attribute'], 'other')
        for turn in range(2, 10):
            response = agent.respond('s', "I don't have a preference for other; please use your judgment.", turn, 10)
            self.assertNotEqual(response['ask_attribute'], 'other')

    def test_rejections_do_not_become_preferences_and_mixed_needs_survive(self):
        for message in ("Those options aren't right.", 'None of these work for me.', 'Skip those items.'):
            state = SessionState({})
            state.observe("I'm looking for Shirts.", 1)
            before = semantic_signature(state)
            state.observe(message, 2)
            self.assertEqual(before, semantic_signature(state), message)
        for message in ("Those options aren't right, I need cotton.",
                        "None of these work, but I need cotton.",
                        "Skip those items; I need cotton.",
                        "Those options aren't right, cotton please."):
            state = SessionState({})
            state.observe("I'm looking for Shirts.", 1)
            state.observe(message, 2)
            self.assertTrue(any('cotton' in item.text for item in state.evidence), message)
            self.assertFalse(any('those' in item.text.lower() for item in state.evidence), message)
        for message in ('None of these are bad.', 'If those options aren\'t right, what comes next?',
                        'You said "Those options aren\'t right".'):
            self.assertFalse(explicit_slate_rejection(message))

    def test_mixed_rejection_preserves_decimal_and_thousands_budget(self):
        for amount in ('10.50', '1,250.75'):
            content = preference_content(f"Those options aren't right. I need a budget under ${amount}.")
            self.assertEqual(content, f'I need a budget under ${amount}')

    def test_category_guard_prevents_feature_match_and_paging_drift(self):
        rows = [{'parent_asin': f'bag{i}', 'title': 'Canvas shoulder bag',
                 'categories': ['Clothing, Shoes & Jewelry', 'Women', 'Bags'],
                 'features': ['canvas', 'adjustable strap']} for i in range(6)]
        rows += [{'parent_asin': 'shoe', 'title': 'Canvas adjustable strap sneakers',
                  'categories': ['Clothing, Shoes & Jewelry', 'Women', 'Shoes', 'Sneakers'],
                  'features': ['canvas', 'adjustable strap']},
                 {'parent_asin': 'cap', 'title': 'Canvas adjustable strap cap',
                  'categories': ['Clothing, Shoes & Jewelry', 'Women', 'Hats & Caps'],
                  'features': ['canvas', 'adjustable strap']}]
        rows += [{'parent_asin': 'replacement', 'title': 'Adjustable strap replacement for bags',
                  'categories': ['Bags'], 'features': ['canvas', 'adjustable strap']},
                 {'parent_asin': 'bag-with-spare', 'title': 'Canvas bag with replacement strap included',
                  'categories': ['Bags'], 'features': ['canvas', 'adjustable strap']}]
        agent = Agent(self.catalog(rows))
        self.addCleanup(agent.close)
        for category in ('bags', 'bag', 'blue waterproof bags with an adjustable strap'):
            agent.reset('s', {})
            for turn, message in enumerate((f"I'm looking for {category}. A key requirement is: canvas; adjustable strap.",
                                           'I have no preference for color.', "Those options aren't right."), 1):
                result = agent.respond('s', message, turn, 10)
                self.assertTrue(result['recommendations'], category)
                self.assertTrue(all(key.startswith('bag') for key in self.ids(result)))
        agent.reset('s', {})
        shoes = agent.respond('s', "I'm looking for shoes. A key requirement is: canvas.", 1, 10)
        self.assertEqual(self.ids(shoes), ['shoe'])

    def test_unknown_category_metadata_remains_unknown_and_empty_match_is_honest(self):
        path = self.catalog([{'parent_asin': 'unknown', 'title': 'Canvas bag'},
                             {'parent_asin': 'known', 'title': 'Canvas bag', 'categories': ['Bags']},
                             {'parent_asin': 'shoe', 'title': 'Canvas sneakers', 'categories': ['Shoes']}])
        agent = Agent(path)
        self.addCleanup(agent.close)
        agent.reset('s', {})
        response = agent.respond('s', "I'm looking for bags.", 1, 10)
        self.assertEqual(self.ids(response)[0], 'known')
        self.assertNotIn('shoe', agent.last_diagnostics['stage_ids']['question_context'])
        agent.reset('s', {})
        response = agent.respond('s', 'qzxvunmatchable', 1, 10)
        self.assertEqual(response['recommendations'], [])
        self.assertIn("couldn't find", response['message'])

    def test_category_short_tokens_and_combined_departments_normalize_symmetrically(self):
        categories = ('Shirts T-Shirts', 'Card & ID Cases Card Cases', 'Socks No Show & Liner Socks')
        rows = [{'parent_asin': str(i), 'title': category, 'categories': [category],
                 'features': ['soft fabric']} for i, category in enumerate(categories)]
        rows.append({'parent_asin': 'man', 'title': 'Mens shirt',
                     'categories': ['Clothing, Shoes & Jewelry', 'Men', 'Clothing', 'Shirts']})
        agent = Agent(self.catalog(rows))
        self.addCleanup(agent.close)
        for i, category in enumerate(categories):
            agent.reset('s', {})
            response = agent.respond('s', f"I'm looking for {category}.", 1, 10)
            self.assertIn(str(i), self.ids(response), category)
        agent.reset('s', {})
        response = agent.respond('s', "I'm looking for Shoes & Jewelry Men.", 1, 10)
        self.assertEqual(self.ids(response), ['man'])


if __name__ == '__main__':
    unittest.main()
