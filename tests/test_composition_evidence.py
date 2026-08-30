from __future__ import annotations

import math
import unittest

from mercury.catalog import product_from_dict
from mercury.composition_evidence import CompositionEvidenceWitness, composition_evidence
from mercury.ranking import rank_composition_evidence
from mercury.types import Candidate, Preference


def material(value: str, **changes: object) -> Preference:
    return Preference("material", value, 1, value, **changes)


def composition(value: str, material_value: str = "cotton", **changes: object) -> Preference:
    return Preference("other", value, 1, value, depends_on=("material", material_value), **changes)


class CompositionEvidenceTest(unittest.TestCase):
    def test_direct_same_field_composition_has_one_bounded_witness(self):
        product = product_from_dict({
            "parent_asin": "direct",
            "description": "A shirt woven from 80 % Cotton for breathable comfort.",
        })

        result = composition_evidence(product, [material("cotton"), composition("80% cotton")])

        self.assertTrue(math.isfinite(result.score))
        self.assertEqual(result.score, 0.010)
        self.assertEqual(result.witnesses, (CompositionEvidenceWitness(
            preference="80% cotton", material="cotton", source="description", span="80 % Cotton", start=19, end=30,
        ),))

    def test_cross_field_material_only_and_unknown_compositions_remain_unknown(self):
        products = [
            product_from_dict({"parent_asin": "cross", "title": "80% shirt", "description": "Cotton fabric."}),
            product_from_dict({"parent_asin": "material", "description": "Cotton fabric."}),
            product_from_dict({"parent_asin": "unknown", "description": "100% synthetic fabric."}),
        ]
        preferences = [
            [material("cotton"), composition("80% cotton")],
            [material("cotton"), composition("80% cotton")],
            [Preference("material", "synthetic", 1, "synthetic"), composition("100% synthetic", "synthetic")],
        ]

        for product, request in zip(products, preferences, strict=True):
            with self.subTest(product=product.parent_asin):
                result = composition_evidence(product, request)
                self.assertEqual(result.score, 0.0)
                self.assertEqual(result.witnesses, ())

    def test_stale_or_mismatched_composition_facts_do_not_score(self):
        product = product_from_dict({"parent_asin": "direct", "description": "80% cotton shell."})
        requests = [
            [composition("80% cotton")],
            [material("canvas"), composition("80% cotton")],
            [material("cotton", polarity=0), composition("80% cotton")],
            [material("cotton", polarity=-1), composition("80% cotton")],
            [material("cotton"), composition("80% cotton", active=False)],
            [material("cotton"), composition("80% cotton", polarity=-1)],
            [material("cotton"), composition("80% cotton", material_value="canvas")],
        ]

        for request in requests:
            with self.subTest(request=request):
                result = composition_evidence(product, request)
                self.assertEqual(result.score, 0.0)
                self.assertEqual(result.witnesses, ())

    def test_multiple_supporting_facts_keep_the_single_registered_cap(self):
        product = product_from_dict({"parent_asin": "blend", "description": "80% cotton and 20% nylon."})
        preferences = [material("cotton"), material("nylon"), composition("80% cotton"), composition("20% nylon", "nylon")]

        first = composition_evidence(product, preferences)
        second = composition_evidence(product, preferences)

        self.assertEqual(first, second)
        self.assertEqual(first.score, 0.010)
        self.assertEqual([item.preference for item in first.witnesses], ["80% cotton", "20% nylon"])

    def test_post_reranker_ranker_preserves_unknown_rows_and_deterministic_order(self):
        supported = product_from_dict({"parent_asin": "supported", "description": "80% cotton jacket."})
        unknown = product_from_dict({"parent_asin": "unknown", "description": "Cotton jacket."})
        candidates = [Candidate(supported, 0.900), Candidate(unknown, 0.905)]

        ranked, diagnostics = rank_composition_evidence(
            candidates, [material("cotton"), composition("80% cotton")],
        )

        self.assertEqual([item.product.parent_asin for item in ranked], ["supported", "unknown"])
        self.assertEqual(ranked[0].route_scores["composition_evidence"], 0.010)
        self.assertNotIn("composition_evidence", ranked[1].route_scores)
        self.assertEqual(set(diagnostics), {"supported"})


if __name__ == "__main__":
    unittest.main()
