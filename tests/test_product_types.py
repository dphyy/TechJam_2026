import unittest

from mercury.catalog import product_from_dict
from mercury.product_types import accessory_mismatch, classify_product, scoped_value_evidence


class ProductTypesTest(unittest.TestCase):
    def product(self, **row):
        return product_from_dict({"parent_asin": "p", **row})

    def test_primary_object_and_replacement_component_are_distinct(self):
        shoes = self.product(title="Running shoes", categories=["Shoes"])
        laces = self.product(title="Replacement shoe laces", categories=["Shoe Accessories"])
        self.assertEqual(classify_product(shoes).role, "object")
        self.assertEqual(classify_product(laces).role, "component")
        self.assertFalse(accessory_mismatch(shoes, ("sneakers",)))
        self.assertTrue(accessory_mismatch(laces, ("sneakers",)))

    def test_unknown_product_type_is_not_a_mismatch(self):
        product = self.product(title="Handmade daily essential")
        self.assertEqual(classify_product(product).role, "unknown")
        self.assertFalse(accessory_mismatch(product, ("bags",)))

    def test_component_evidence_distinguishes_body_from_handles(self):
        body = self.product(title="Leather body bag with cotton lining")
        handles = self.product(title="Cotton body bag with leather handles")
        self.assertGreater(scoped_value_evidence(body, "leather", "body"), 0)
        self.assertLess(scoped_value_evidence(handles, "leather", "body"), 0)
        self.assertGreater(scoped_value_evidence(handles, "leather", "handles"), 0)


if __name__ == "__main__":
    unittest.main()
