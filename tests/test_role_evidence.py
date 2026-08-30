from __future__ import annotations

import math
import unittest

from mercury.catalog import product_from_dict
from mercury.role_evidence import (
    COMPONENT_ROLES,
    WHOLE_PRODUCT_ROLES,
    RoleEvidenceWitness,
    role_evidence,
)
from mercury.types import Preference


def other(value: str, **changes: object) -> Preference:
    return Preference("other", value, 1, value, **changes)


def material(value: str, **changes: object) -> Preference:
    return Preference("material", value, 1, value, **changes)


class RoleEvidenceTest(unittest.TestCase):
    def test_direct_whole_role_phrase_has_a_local_witness(self):
        for role in WHOLE_PRODUCT_ROLES:
            with self.subTest(role=role):
                product = product_from_dict({
                    "parent_asin": role,
                    "description": f"A full-grain leather {role} with a canvas lining.",
                })

                result = role_evidence(product, [material("leather"), other(f"leather {role}")])

                self.assertTrue(math.isfinite(result.score))
                self.assertGreater(result.score, 0.0)
                self.assertLess(result.score, 1.0)
                self.assertEqual(len(result.witnesses), 1)
                witness = result.witnesses[0]
                self.assertEqual(witness.material, "leather")
                self.assertEqual(witness.role, role)
                self.assertEqual(witness.source, "description")
                self.assertEqual(witness.span.lower(), f"leather {role}")

    def test_component_only_phrases_never_support_a_whole_role(self):
        for component in COMPONENT_ROLES:
            with self.subTest(component=component):
                product = product_from_dict({
                    "parent_asin": component,
                    "description": f"Tote with a leather {component}.",
                })

                result = role_evidence(product, [material("leather"), other("leather body")])

                self.assertEqual(result.score, 0.0)
                self.assertEqual(result.witnesses, ())

    def test_cross_field_and_ambiguous_text_remain_unknown(self):
        cross_field = product_from_dict({
            "parent_asin": "cross-field",
            "title": "Soft leather tote",
            "description": "The body is generously sized.",
        })
        ambiguous = product_from_dict({
            "parent_asin": "ambiguous",
            "description": "Leather lining with a cotton body.",
        })

        for product in (cross_field, ambiguous):
            with self.subTest(product=product.parent_asin):
                result = role_evidence(product, [material("leather"), other("leather body")])
                self.assertEqual(result.score, 0.0)
                self.assertEqual(result.witnesses, ())

    def test_inactive_or_nonpositive_or_noncanonical_preferences_do_not_score(self):
        product = product_from_dict({
            "parent_asin": "body",
            "title": "Leather body tote",
        })
        preferences = [
            material("leather"),
            other("leather body", active=False),
            other("leather body", polarity=0),
            other("leather body", polarity=-1),
            other("vegan leather body"),
        ]

        result = role_evidence(product, preferences)

        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.witnesses, ())

    def test_role_support_retracts_when_the_matching_material_is_not_active(self):
        product = product_from_dict({
            "parent_asin": "body",
            "title": "Leather body tote",
        })
        requests = [
            [other("leather body")],
            [material("canvas"), other("leather body")],
            [material("leather", polarity=0), other("leather body")],
            [material("leather", polarity=-1), other("leather body")],
        ]

        for preferences in requests:
            with self.subTest(preferences=preferences):
                result = role_evidence(product, preferences)
                self.assertEqual(result.score, 0.0)
                self.assertEqual(result.witnesses, ())

    def test_witness_order_and_structure_are_deterministic(self):
        product = product_from_dict({
            "parent_asin": "deterministic",
            "title": "Leather body tote",
            "description": "Leather body with a cotton lining.",
        })
        preferences = [material("leather"), other("leather body"), other("leather body")]

        first = role_evidence(product, preferences)
        second = role_evidence(product, preferences)

        expected = RoleEvidenceWitness(
            preference="leather body",
            material="leather",
            role="body",
            source="title",
            span="Leather body",
            start=0,
            end=12,
        )
        self.assertEqual(first, second)
        self.assertEqual(first.witnesses, (expected,))
        self.assertGreater(first.score, 0.0)


if __name__ == "__main__":
    unittest.main()
