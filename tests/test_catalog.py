import json
import tempfile
import unittest
from pathlib import Path

from mercury.catalog import Catalog, flatten, negated_match, product_from_dict


class CatalogTest(unittest.TestCase):
    def test_recursive_flatten_and_unknown_price(self):
        product = product_from_dict({
            "parent_asin": "A", "title": "Blue cotton shirt",
            "details": {"Rank": {"Shirts": "42"}}, "price": "—",
        })
        self.assertIn("42", product.fields["details"])
        self.assertIsNone(product.price)
        self.assertIn("cotton", product.facets["material"])
        self.assertTrue(any(e.source == "title" for e in product.evidence))
        self.assertEqual(flatten(None), "")

    def test_price_bounds_and_untrusted_values(self):
        product = product_from_dict({"parent_asin": "A", "price": "from 12.99"})
        self.assertEqual(product.price, 12.99)
        self.assertTrue(product.price_lower_bound)
        for bad in (True, float("nan"), float("inf"), -1, "not a price"):
            self.assertIsNone(product_from_dict({"parent_asin": "A", "price": bad}).price)

    def test_negated_material_does_not_become_supported_fact(self):
        product = product_from_dict({"parent_asin": "A", "title": "Leather-free vegan shoes"})
        self.assertNotIn("leather", product.facets.get("material", ()))

    def test_negated_features_do_not_become_supported_facts(self):
        for title, feature in (("Not waterproof jacket", "waterproof"),
                               ("Jacket without pockets", "pockets")):
            with self.subTest(title=title):
                product = product_from_dict({"parent_asin": "A", "title": title})
                self.assertNotIn(feature, product.facets.get("feature", ()))

    def test_direct_avoidance_negates_the_matched_span(self):
        for prefix in ("Please avoid", "Avoiding", "Avoidance of", "The design avoids"):
            with self.subTest(prefix=prefix):
                text = prefix + " soaking the strap."
                start = text.index("soaking")
                self.assertTrue(negated_match(text, start, start + len("soaking")))
                product = product_from_dict({"parent_asin": "A", "features": [prefix + " leather"]})
                self.assertNotIn("leather", product.facets.get("material", ()))

    def test_preserves_full_structured_material_without_unqualified_evidence(self):
        for material in ("Faux leather", "Leather-free"):
            with self.subTest(material=material):
                product = product_from_dict({"parent_asin": "A", "details": {"Material": material}})
                self.assertIn(material.lower(), product.facets["material"])
                self.assertNotIn("leather", product.facets["material"])
                self.assertTrue(any(item.value == material.lower() and item.source == "details.Material"
                                    for item in product.evidence))

    def test_preserves_indistinguishable_ids_and_rejects_duplicate_id(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "catalog.jsonl"
            rows = [{"parent_asin": x, "title": "Same cotton shirt"} for x in ("A", "B")]
            file.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
            catalog = Catalog(file)
            self.assertEqual(len(catalog.products), 2)
            self.assertEqual(set(catalog.by_id), {"A", "B"})
            rows.append(rows[0])
            file.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
            with self.assertRaises(ValueError):
                Catalog(file)

    def test_taxonomy_is_preserved_not_assumed_to_be_gender(self):
        product = product_from_dict({"parent_asin": "A", "categories": ["Clothing", "Boot Shop", "Women", "Snow Boots"]})
        self.assertIn("Boot Shop", product.fields["categories"])
        self.assertIn("boots", product.facets.get("category", ()))
