from __future__ import annotations

from dataclasses import replace
import itertools
import json
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
import unittest
from unittest.mock import patch

from mercury.lexical import Agent, FULL_WIDTH_CONFIG
from mercury.lexical.dialogue import Evidence
from mercury.lexical import product_features as features


def reference_polarity(value: str):
    value = re.sub(r"\bnot (?:only|just|exclusively)\b", "", value, flags=re.I)
    value = re.sub(r"\b(?:not (?:necessarily|always)|may not|might not|"
                   r"not (?:sure|clear|known|certain)(?: whether| if)?)\b[^,;.!?]*",
                   "", value, flags=re.I)
    negative = re.compile(r"\b(?:no(?![- ]show\b)|not|without|free of|doesn't contain|does not contain)\b"
                          r"[^,;.!?]*?(?=$|[,;.!?]|\b(?:but|however|whereas|while)\b|"
                          r"\band\s+(?:the|its|a|an|with)\b)", re.I)
    free = re.compile(r"\b(?P<value>\w+)[- ]free\b", re.I)
    denied = [tuple(features.terms(match.group("value"))) for match in free.finditer(value)]
    after_free = free.sub(" ", value)
    for match in negative.finditer(after_free):
        wording = re.sub(r"^(?:no|not|without|free of|doesn't contain|does not contain)\b",
                         "", match.group(), flags=re.I)
        if tokens := tuple(features.terms(wording)):
            denied.append(tokens)
    return tuple(features.terms(negative.sub(" ", after_free))), tuple(denied)


def uncached_view(product, evidence):
    if evidence.scope:
        values = product.component_fields.get(evidence.scope, ("",) * len(features.FIELD_ORDER))
        sequences = tuple(tuple(features.terms(value)) if evidence.literal_absence else features.affirmed_terms(value)
                          for value in values)
        denied = tuple(sequence for value in values for sequence in features.denied_terms(value))
    else:
        sequences = product.field_sequences if evidence.literal_absence else product.affirmed_sequences
        if not sequences or sequences == product.field_sequences:
            return product
        denied = product.denied_sequences
    weights = {}
    for name, sequence in zip(features.FIELD_ORDER, sequences):
        for token in sequence:
            weights[token] = max(weights.get(token, 0.0), features.FIELD_WEIGHTS[name])
    return replace(product, token_weights=MappingProxyType(weights),
                   normalized_text=features.FIELD_SEPARATOR.join(" ".join(sequence) for sequence in sequences),
                   field_sequences=sequences, affirmed_sequences=sequences, denied_sequences=denied,
                   feature_tokens=frozenset(sequences[features.FIELD_ORDER.index("features")]))


def reference_sequence_match(tokens, sequences):
    return bool(tokens) and any(sequence[start:start + len(tokens)] == tokens
                               for sequence in sequences for start in range(len(sequence) - len(tokens) + 1))


class LexicalPerformanceTest(unittest.TestCase):
    def setUp(self):
        self.store = features.ProductFeatureStore(max_size=4)
        self.product = self.store.add("item", {
            "title": "Blue item", "features": "cotton; no polyester; linen; fragrance free",
            "details": "lining: cotton; upper: leather; sole: rubber; pocket: zippered; sleeve: wool; collar: silk; hood: fleece",
        }, price=25)

    def evidence(self, value, source="hard_constraint"):
        return self.store.compile_query([Evidence(value, 3.8, source, 1)]).evidence[0]

    def test_polarity_fast_path_matches_the_unconditional_transform(self):
        prefixes = ("", "no ", "not ", "without ", "free of ", "doesn't contain ",
                    "does not contain ", "not only ", "not exclusively ", "may not contain ",
                    "might not ", "not necessarily ", "not sure whether ")
        values = ("cotton", "cotton-free", "cotton free", "no-show socks", "notary freezer", "İ ı ſ K")
        boundaries = ("", "; linen", ", linen", " but linen", " and the lining is linen", "\nlinen")
        for prefix, value, boundary in itertools.product(prefixes, values, boundaries):
            for wording in (prefix + value + boundary, (prefix + value + boundary).upper()):
                with self.subTest(wording=wording):
                    affirmed, denied = reference_polarity(wording)
                    self.assertEqual(features.affirmed_terms(wording), affirmed)
                    self.assertEqual(features.denied_terms(wording), denied)

    def test_lowercase_once_preserves_tokenization_including_unicode(self):
        values = (None, 0, 123, True, "", "Size M 2 mm", "Men's shirts", "İ ı ſ K", "A_I-x", ["Cotton", "Blue"])
        for value, minimum in itertools.product(values, (0, 1, 2, 3)):
            expected = [token.lower() for token in features.TOKEN_RE.findall(str(value or ""))
                        if len(token) >= minimum and token.lower() not in features.STOPWORDS]
            self.assertEqual(features.terms(value, min_length=minimum), expected)

    def test_phrase_search_preserves_arbitrary_tokens_and_field_boundaries(self):
        alphabet = ("", "a", "aa", "a b", "\x1f", "İ")
        sequences = [value for width in range(4) for value in itertools.product(alphabet, repeat=width)]
        for sequence, phrase in itertools.product(sequences, repeat=2):
            for fields in ((sequence,), (sequence[:1], sequence[1:])):
                self.assertEqual(features._sequence_match(phrase, fields),
                                 reference_sequence_match(phrase, fields))

    def test_views_match_uncached_negative_scoped_or_and_exclusive_semantics(self):
        for wording, source in (("cotton", "hard_constraint"), ("polyester", "exclusion"),
                                ("lining: cotton", "hard_constraint"), ("upper: cotton", "hard_constraint"),
                                ("leather lining", "exclusion"), ("cotton or linen", "hard_constraint"),
                                ("blue only", "override"), ("only red or blue", "override"),
                                ("fragrance free", "hard_constraint")):
            evidence = self.evidence(wording, source)
            for branch in evidence.alternatives or (evidence,):
                expected = uncached_view(self.product, branch)
                self.assertEqual(features.evidence_product(self.product, branch), expected)
                self.assertEqual(features.evidence_product(self.product, branch), expected)

    def test_memo_reuses_only_catalog_scope_and_does_not_retain_unknown_scopes(self):
        evidence = self.evidence("lining: cotton")
        with patch.object(features, "affirmed_terms", wraps=features.affirmed_terms) as transform:
            first = features.evidence_product(self.product, evidence)
            count = transform.call_count
            second = features.evidence_product(self.product, replace(evidence, tokens=("private-query-marker",)))
            self.assertIs(first, second)
            self.assertEqual(transform.call_count, count)
        before = dict(self.product._evidence_views)
        for index in range(100):
            features.evidence_product(self.product, replace(evidence, scope=f"private-scope-{index}"))
        self.assertEqual(self.product._evidence_views, before)
        self.assertNotIn("private-query-marker", repr(self.product._evidence_views))
        self.assertNotIn("private-scope", repr(self.product._evidence_views))
        self.assertIsNone(first._evidence_views)
        nested = features.evidence_product(first, self.evidence("upper: leather"))
        self.assertIsNone(nested._evidence_views)
        self.assertIsNone(first._evidence_views)

    def test_memo_is_bounded_and_replacement_has_fresh_views(self):
        for scope in self.product.component_fields:
            for literal in (False, True):
                item = replace(self.evidence(f"{scope}: cotton"), literal_absence=literal)
                self.assertEqual(features.evidence_product(self.product, item), uncached_view(self.product, item))
                self.assertLessEqual(len(self.product._evidence_views), features.EVIDENCE_VIEW_CACHE_SIZE)
        clone = replace(self.product)
        self.assertEqual(self.product, clone)
        self.assertEqual(repr(self.product), repr(clone))
        self.assertEqual(clone._evidence_views, {})
        item = self.evidence("upper: leather")
        old_view = features.evidence_product(self.product, item)
        replacement = replace(self.product, price=0)
        self.assertEqual(replacement._evidence_views, {})
        new_view = features.evidence_product(replacement, item)
        self.assertEqual(old_view.price, 25)
        self.assertEqual(new_view.price, 0)
        self.assertIsNot(old_view, new_view)
        component_values = [""] * len(features.FIELD_ORDER)
        component_values[features.FIELD_ORDER.index("details")] = "upper cotton"
        rebound = replace(self.product, component_fields=MappingProxyType({"upper": tuple(component_values)}))
        rebound_view = features.evidence_product(rebound, item)
        self.assertIn("cotton", rebound_view.token_weights)
        self.assertNotIn("leather", rebound_view.token_weights)
        self.assertEqual(rebound_view, uncached_view(rebound, item))

    def test_actual_responses_and_all_stage_fingerprints_match_without_memo(self):
        messages = ("I'm looking for jackets.", "A key requirement is: cotton or linen.",
                    "No polyester.", "I also need lining: cotton.", "Actually, what I need is: blue only.")
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.jsonl"
            rows = [{"parent_asin": str(index), "categories": ["jackets"], "title": "Jacket",
                     "features": ["no polyester" if index % 3 else "polyester", "cotton" if index % 2 else "linen"],
                     "details": {"lining": "cotton" if index % 4 else "leather",
                                 "Color": "Blue" if index % 5 else "Red"}, "price": 25}
                    for index in range(24)]
            catalog.write_text("".join(json.dumps(row) + "\n" for row in rows))

            def replay():
                agent = Agent(catalog, config=FULL_WIDTH_CONFIG)
                try:
                    agent.reset("s", {})
                    result = []
                    for turn, message in enumerate(messages, 1):
                        response = agent.respond("s", message, turn, 10)
                        result.append((response, agent.last_diagnostics["stage_receipts"]))
                    return result
                finally:
                    agent.close()

            cached = replay()
            with patch("mercury.lexical.product_features.evidence_product", uncached_view), \
                    patch("mercury.lexical.retrieval.evidence_product", uncached_view), \
                    patch("mercury.lexical.diagnostics.evidence_product", uncached_view), \
                    patch("mercury.lexical.product_features._sequence_match", reference_sequence_match), \
                    patch("mercury.lexical.retrieval._sequence_match", reference_sequence_match):
                self.assertEqual(replay(), cached)


if __name__ == "__main__":
    unittest.main()
