import unittest

from mercury.state import SessionState


class SessionStateTest(unittest.TestCase):
    def test_newer_override_phrasing_retracts_only_changed_preferences(self):
        state = SessionState({}, alternatives_mode="grouped")
        state.update("I need a black leather shoulder bag with an adjustable strap.", 1)
        state.update("I've changed my mind; make that blue canvas, but keep the adjustable strap.", 2)
        active = {(item.attribute, item.value, item.polarity) for item in state.active_preferences()}
        self.assertIn(("material", "canvas", 1), active)
        self.assertIn(("color", "blue", 1), active)
        self.assertIn(("feature", "adjustable", 1), active)
        self.assertIn(("other", "strap", 1), active)
        self.assertNotIn(("material", "leather", 1), active)
        self.assertNotIn(("color", "black", 1), active)

    def test_direct_scratch_and_excess_feedback_are_explicit_rejections(self):
        state = SessionState({})
        state.update("A formal leather jacket.", 1)
        state.update("Scratch leather; go with canvas.", 2)
        active = {(item.attribute, item.value, item.polarity) for item in state.active_preferences()}
        self.assertIn(("material", "leather", -1), active)
        self.assertIn(("material", "canvas", 1), active)
        self.assertNotIn(("other", "go", 1), active)
        state.update("These look too formal; make that casual.", 3)
        active = {(item.attribute, item.value, item.polarity) for item in state.active_preferences()}
        self.assertIn(("style", "formal", -1), active)
        self.assertIn(("style", "casual", 1), active)

    def test_generic_new_data_feedback_does_not_pollute_the_query(self):
        state = SessionState({})
        state.update("A formal leather jacket.", 1)
        before = state.query()
        for turn, message in enumerate((
            "None of these work; show another set.",
            "Those aren't right; keep looking.",
        ), 2):
            state.update(message, turn)
            self.assertEqual(state.last_feedback.scope, "item")
            self.assertEqual(state.query(), before)

    def test_negative_feedback_scope_is_narrow_and_explicit(self):
        state = SessionState({})
        state.update("I need a blue canvas bag.", 1)
        state.update("Not this item.", 2)
        self.assertEqual(state.last_feedback.scope, "item")
        self.assertTrue(any(item.attribute == "category" and item.active for item in state.preferences))
        state.update("Not this product type.", 3)
        self.assertEqual(state.last_feedback.scope, "product_type")
        self.assertFalse(any(item.attribute == "category" and item.active and item.polarity == 1
                             for item in state.preferences))
        self.assertFalse(any(item.attribute == "category" and item.active and item.polarity == -1
                             for item in state.preferences))
        state.update("No leather, please.", 4)
        self.assertEqual(state.last_feedback.scope, "attribute_value")
        self.assertEqual(state.last_feedback.attribute, "material")

    def test_question_goal_and_answer_productivity_are_recorded(self):
        state = SessionState({})
        state.record_question("material", "facet:material")
        state.update("Cotton, please.", 1)
        self.assertEqual(state.last_answer_productivity, "productive")
        self.assertIn("facet:material", state.asked_question_goals)
        state.record_question("color", "facet:color")
        state.update("I do not have a color preference.", 2)
        self.assertEqual(state.last_answer_productivity, "neutral")

    def test_conversation_management_is_not_a_product_feature(self):
        state = SessionState({})
        state.update("I am exploring tunics. My primary requirement is cotton.", 1)
        self.assertIn("tunics", state.query())
        self.assertIn("cotton", state.query())
        for word in ("exploring", "primary", "requirement"):
            self.assertNotIn(word, state.query())
        before = state.query()
        state.update("Please ask about another attribute.", 2)
        self.assertEqual(state.query(), before)
        state.update("Ignore the earlier color preference; blue would be good.", 3)
        self.assertNotIn("earlier", state.query())
        self.assertNotIn("ignore", state.query())

    def test_metadiscourse_cannot_become_or_replace_a_color(self):
        for initial, followup, color in (
            ("Color matters to me: navy. A jacket.", "What matters now is a cotton lining.", "navy"),
            ("The color that matters most is teal on a shirt.", "Material matters too: linen.", "teal"),
        ):
            with self.subTest(initial=initial):
                state = SessionState({})
                state.update(initial, 1)
                self.assertEqual(self.values(state, "color"), {color})
                self.assertNotIn("matters", state.query())
                state.update(followup, 2)
                self.assertEqual(self.values(state, "color"), {color})
                self.assertNotIn("matters", state.query())
                self.assertTrue(self.values(state, "material"))

    def values(self, state, attribute, polarity=1):
        return {
            preference.value
            for preference in state.active_preferences()
            if preference.attribute == attribute and preference.polarity == polarity
        }

    def test_accumulates_independent_facts_with_source_evidence(self):
        state = SessionState({})
        state.update("I'm after a navy cotton shirt for the office.", 1)
        state.update("It also needs to be breathable.", 2)
        self.assertEqual(self.values(state, "category"), {"shirts"})
        self.assertEqual(self.values(state, "color"), {"navy"})
        self.assertEqual(self.values(state, "material"), {"cotton"})
        self.assertEqual(self.values(state, "use_case"), {"work"})
        self.assertEqual(self.values(state, "feature"), {"breathable"})
        cotton = next(p for p in state.active_preferences() if p.value == "cotton")
        self.assertEqual(cotton.source_turn, 1)
        self.assertIn("navy cotton shirt", cotton.source_text)
        self.assertEqual(state.turn, 2)
        self.assertEqual(len(state.history), 2)

    def test_factual_material_components_accumulate_without_correction(self):
        state = SessionState({})
        state.update("Polyester basketball shorts.", 1)
        state.update("Mesh panels.", 2)
        self.assertEqual(self.values(state, "material"), {"polyester", "mesh"})
        for term in ("polyester", "mesh", "basketball", "shorts", "panels"):
            self.assertIn(term, state.query())

    def test_explicit_material_change_still_replaces_previous_material(self):
        for correction in ("I prefer linen.", "Actually, linen.", "Use linen instead."):
            with self.subTest(correction=correction):
                state = SessionState({})
                state.update("A cotton tunic.", 1)
                state.update(correction, 2)
                self.assertEqual(self.values(state, "material"), {"linen"})
                self.assertNotIn("cotton", state.query())
                self.assertIn("tunic", state.query())

    def test_paraphrases_produce_equivalent_preferences(self):
        for message in (
            "I want grey trainers for jogging.",
            "Could you find gray sneakers for running?",
            "Running shoes in grey, please.",
        ):
            with self.subTest(message=message):
                state = SessionState({})
                state.update(message, 1)
                self.assertIn("grey", self.values(state, "color"))
                self.assertIn("sneakers", self.values(state, "category"))
                self.assertIn("running", self.values(state, "use_case"))

    def test_source_alias_query_keeps_only_current_direct_parser_aliases(self):
        state = SessionState({}, alternatives_mode="grouped")
        state.update("I need grey trainers made from vegan leather.", 1)
        self.assertEqual(state.query(), "faux leather grey sneakers")
        self.assertEqual(state.source_alias_query(), "vegan leather trainers")

        state.update("Actually, grey sneakers in faux leather.", 2)
        self.assertEqual(state.source_alias_query(), "")

        state.update("No trainers after all.", 3)
        self.assertEqual(state.source_alias_query(), "")

    def test_correction_removes_superseded_source_terms_only(self):
        state = SessionState({})
        state.update("A red cotton dress for a wedding.", 1)
        state.update("Actually, make it blue instead of red.", 2)
        self.assertEqual(self.values(state, "color"), {"blue"})
        self.assertEqual(self.values(state, "category"), {"dresses"})
        self.assertEqual(self.values(state, "material"), {"cotton"})
        self.assertEqual(self.values(state, "use_case"), {"wedding"})
        self.assertNotIn("red", state.query().split())
        self.assertIn("cotton", state.query())
        self.assertTrue(any(p.value == "red" and not p.active for p in state.preferences))

    def test_same_message_corrections_apply_in_clause_order(self):
        for message, expected in (
            ("A red shirt. Actually, blue.", {"blue"}),
            ("A red shirt; blue instead.", {"blue"}),
            ("A red shirt, but I prefer blue.", {"blue"}),
            ("A red shirt. Blue. Actually, green.", {"green"}),
        ):
            with self.subTest(message=message):
                state = SessionState({})
                state.update(message, 1)
                self.assertEqual(self.values(state, "color"), expected)
                self.assertEqual(self.values(state, "category"), {"shirts"})
                self.assertNotIn("red", state.query())
                self.assertEqual(state.revision, 1)
                self.assertEqual(len(state.history), 1)
                self.assertEqual(state.history[0].text, message)
                red = next(p for p in state.history[0].preferences if p.value == "red")
                self.assertFalse(red.active)
                self.assertEqual(red.source_text, message)

    def test_same_message_correction_preserves_coordinated_alternatives(self):
        state = SessionState({})
        state.update("A red shirt. Actually, blue or green.", 1)
        self.assertEqual(self.values(state, "color"), {"blue", "green"})
        self.assertEqual(self.values(state, "category"), {"shirts"})
        self.assertNotIn("red", state.query())

    def test_same_message_material_components_remain_additive(self):
        state = SessionState({})
        state.update("Polyester basketball shorts. Mesh panels.", 1)
        self.assertEqual(self.values(state, "material"), {"polyester", "mesh"})
        for term in ("polyester", "mesh", "basketball", "shorts", "panels"):
            self.assertIn(term, state.query())

    def test_same_message_polarity_reversal_retracts_earlier_assertion(self):
        for message, positives, negatives in (
            ("No leather. Actually, leather is fine.", {"leather"}, set()),
            ("Leather is fine. Actually, no leather.", set(), {"leather"}),
        ):
            with self.subTest(message=message):
                state = SessionState({})
                state.update("A leather bag.", 1)
                state.update(message, 2)
                self.assertEqual(self.values(state, "material"), positives)
                self.assertEqual(self.values(state, "material", -1), negatives)
                self.assertEqual(self.values(state, "category"), {"bags"})
                self.assertEqual("leather" in state.query(), bool(positives))
                leather = next(p for p in state.active_preferences() if p.value == "leather")
                self.assertEqual(leather.hard, bool(negatives))

    def test_negative_material_correction_is_not_positive_query_evidence(self):
        state = SessionState({})
        state.update("I'd like a leather backpack.", 1)
        state.update("No leather after all; please use canvas instead.", 2)
        self.assertEqual(self.values(state, "material"), {"canvas"})
        self.assertEqual(self.values(state, "material", -1), {"leather"})
        self.assertEqual(self.values(state, "category"), {"backpacks"})
        self.assertNotIn("leather", state.query())
        self.assertTrue(next(p for p in state.active_preferences() if p.polarity == -1).hard)

    def test_local_negation_does_not_apply_to_following_clause(self):
        state = SessionState({})
        state.update("Not wool, but cotton would be nice. It must be waterproof.", 1)
        self.assertEqual(self.values(state, "material", -1), {"wool"})
        self.assertEqual(self.values(state, "material"), {"cotton"})
        self.assertEqual(self.values(state, "feature"), {"waterproof"})
        cotton = next(p for p in state.active_preferences() if p.value == "cotton")
        waterproof = next(p for p in state.active_preferences() if p.value == "waterproof")
        self.assertFalse(cotton.hard)
        self.assertLess(cotton.confidence, waterproof.confidence)
        self.assertTrue(waterproof.hard)

    def test_additive_constraint_does_not_erase_prior_constraint(self):
        state = SessionState({})
        state.update("I need a jacket that's waterproof.", 1)
        state.update("It should also have a hood and zippered pockets.", 2)
        self.assertEqual(self.values(state, "feature"), {"waterproof", "hood", "zippered pockets"})
        self.assertEqual(self.values(state, "category"), {"jackets"})

    def test_explicit_alternative_adds_without_replacing(self):
        state = SessionState({})
        state.update("I prefer blue shirts.", 1)
        state.update("Green is also fine.", 2)
        self.assertEqual(self.values(state, "color"), {"blue", "green"})

    def test_new_color_replaces_color_but_preserves_category(self):
        state = SessionState({})
        state.update("A black wool coat.", 1)
        state.update("I'd prefer beige.", 2)
        self.assertEqual(self.values(state, "color"), {"beige"})
        self.assertEqual(self.values(state, "category"), {"coats"})
        self.assertEqual(self.values(state, "material"), {"wool"})
        self.assertNotIn("black", state.query())

    def test_unknown_answer_retains_existing_fact_without_query_noise(self):
        state = SessionState({})
        state.update("A cotton shirt.", 1)
        before = state.query()
        revision = state.revision
        state.record_question("size")
        state.update("I'm not sure, I don't know yet.", 2)
        self.assertEqual(state.query(), before)
        self.assertEqual(state.revision, revision)
        self.assertFalse(state.last_update_informative)
        self.assertIn("size", state.unproductive_attributes)
        self.assertEqual(state.asked_counts, {"size": 1})

    def test_no_preference_clears_attribute_but_can_be_changed_later(self):
        state = SessionState({})
        state.update("A red silk scarf.", 1)
        state.record_question("color")
        state.update("Any color is fine with me.", 2)
        self.assertEqual(self.values(state, "color"), set())
        self.assertEqual(self.values(state, "color", 0), {"any"})
        self.assertNotIn("red", state.query())
        self.assertIn("color", state.unproductive_attributes)
        state.update("On second thought, I'd like green.", 3)
        self.assertEqual(self.values(state, "color"), {"green"})
        self.assertEqual(self.values(state, "color", 0), set())
        self.assertNotIn("color", state.unproductive_attributes)
        self.assertEqual(self.values(state, "category"), {"scarves"})

    def test_no_preference_for_asked_field_is_contextual(self):
        state = SessionState({})
        state.update("A dress in linen.", 1)
        state.record_question("style")
        state.update("I don't have a preference.", 2)
        self.assertEqual(self.values(state, "style", 0), {"any"})
        self.assertEqual(self.values(state, "material"), {"linen"})
        self.assertNotIn("preference", state.query())

    def test_no_other_preference_retains_explicit_open_vocabulary_facts(self):
        for message in (
            "I'm flexible about other preferences.",
            "Any other preference is fine. Use your judgement.",
            "No preference on other. Please use your judgment.",
            "I have no preference.",
            "Anything is fine.",
        ):
            with self.subTest(message=message):
                state = SessionState({})
                state.update("Trail shoes with gusseted tongues.", 1)
                state.record_question("other")
                before, revision = state.query(), state.revision
                state.update(message, 2)
                self.assertEqual(state.query(), before)
                self.assertEqual(state.revision, revision)
                self.assertEqual(self.values(state, "other", 0), set())
                self.assertIn("other", state.unproductive_attributes)
                self.assertFalse(state.last_update_informative)

    def test_named_no_preference_still_clears_when_other_was_asked(self):
        for attribute in ("color", "material"):
            with self.subTest(attribute=attribute):
                state = SessionState({})
                state.update("A teal cotton shirt with gusseted cuffs.", 1)
                state.record_question("other")
                details = self.values(state, "other")
                state.update(f"I'm flexible on {attribute}.", 2)
                self.assertEqual(self.values(state, attribute), set())
                self.assertEqual(self.values(state, attribute, 0), {"any"})
                self.assertEqual(self.values(state, "other"), details)
                self.assertEqual(self.values(state, "category"), {"shirts"})

    def test_no_new_information_preserves_preferences_and_revision(self):
        for message in (
            "I have no additional preference.",
            "I don't have an additional preference.",
            "No further specific material preferences.",
            "I haven't any more details.",
            "Nothing further to add.",
        ):
            with self.subTest(message=message):
                state = SessionState({})
                state.update("A red cotton shirt.", 1)
                state.record_question("feature")
                before, revision = state.query(), state.revision
                state.update(message, 2)
                self.assertEqual(state.query(), before)
                self.assertEqual(state.revision, revision)
                self.assertEqual(self.values(state, "material"), {"cotton"})
                self.assertEqual(self.values(state, "color"), {"red"})
                self.assertFalse(state.last_update_informative)
                self.assertIn("feature", state.unproductive_attributes)

    def test_no_new_information_does_not_suppress_a_following_fact(self):
        state = SessionState({})
        state.update("A cotton shirt.", 1)
        state.record_question("feature")
        state.update("No additional preferences, but it needs to be breathable.", 2)
        self.assertEqual(self.values(state, "material"), {"cotton"})
        self.assertEqual(self.values(state, "feature"), {"breathable"})
        self.assertNotIn("feature", state.unproductive_attributes)

    def test_refusal_does_not_permanently_block_later_explicit_answer(self):
        state = SessionState({})
        state.update("I'm looking for a sweater.", 1)
        state.record_question("material")
        state.update("I'd rather not answer that. Please just show me some choices.", 2)
        self.assertIn("material", state.unproductive_attributes)
        self.assertFalse(state.last_update_informative)
        self.assertEqual(state.query(), "sweaters")
        state.update("Actually, merino wool is what I want.", 3)
        self.assertEqual(self.values(state, "material"), {"merino wool"})
        self.assertNotIn("material", state.unproductive_attributes)
        self.assertTrue(state.last_update_informative)

    def test_generic_rejection_and_no_new_details_never_pollute_query(self):
        state = SessionState({})
        state.update("A green handbag.", 1)
        before = state.query()
        for turn, message in enumerate((
            "None of those are right.",
            "I don't have any more preferences to add.",
            "Please show different options.",
            "No, not those. Keep looking.",
        ), 2):
            state.update(message, turn)
            self.assertEqual(state.query(), before)
            self.assertFalse(state.last_update_informative)

    def test_budget_normalizes_ceiling_and_replacement(self):
        state = SessionState({})
        state.update("A pair of boots, under $80 please.", 1)
        self.assertEqual(self.values(state, "budget"), {"<= 80"})
        self.assertTrue(next(p for p in state.active_preferences() if p.attribute == "budget").hard)
        state.update("I can stretch to 120 dollars instead.", 2)
        self.assertEqual(self.values(state, "budget"), {"<= 120"})
        self.assertEqual(self.values(state, "category"), {"boots"})

    def test_hedged_budget_is_soft_while_a_stated_limit_stays_hard(self):
        for message, expected in (
            ("For that, what matters is: budget around $22.99.", False),
            ("Budget of about 50 dollars.", False),
            ("My budget is roughly $75.", False),
            ("Under $80 please.", True),
            ("No more than $60.", True),
            ("Maximum budget of $100.", True),
            ("My budget is between $50 and $100.", True),
            # A hedge belonging to another attribute must not soften the budget.
            ("A stylish jacket under $40.", True),
            ("About 12 inches long, with a maximum budget of $90.", True),
        ):
            with self.subTest(message=message):
                state = SessionState({})
                state.update(message, 1)
                budgets = [p for p in state.active_preferences() if p.attribute == "budget"]
                self.assertEqual(len(budgets), 1)
                self.assertIs(budgets[0].hard, expected)

    def test_an_explicit_requirement_overrides_a_hedge(self):
        state = SessionState({})
        state.update("I must stay under about $40.", 1)
        budget = next(p for p in state.active_preferences() if p.attribute == "budget")
        self.assertTrue(budget.hard)

    def test_hedged_budget_never_demotes_a_product_over_the_figure(self):
        from mercury.ranking import rank_constraints
        from mercury.types import Candidate, Product

        fields = {name: "" for name in ("title", "categories", "features", "details", "store", "description")}
        state = SessionState({})
        state.update("My budget is around $22.99.", 1)
        candidates = [
            Candidate(Product("UNDER", "t", fields, price=22.99), 1.0),
            Candidate(Product("OVER", "t", fields, price=23.00), 0.9),
        ]
        ranked = rank_constraints(candidates, state.active_preferences())
        self.assertEqual([item.product.parent_asin for item in ranked], ["UNDER", "OVER"])
        for item in ranked:
            self.assertNotIn("constraint_penalty", item.route_scores)

        firm = SessionState({})
        firm.update("My budget is under $22.99.", 1)
        demoted = rank_constraints(candidates, firm.active_preferences())
        over = next(item for item in demoted if item.product.parent_asin == "OVER")
        self.assertIn("constraint_penalty", over.route_scores)

    def test_budget_range_and_answer_to_budget_question(self):
        state = SessionState({})
        state.update("My budget is between $50 and $100.", 1)
        self.assertEqual(self.values(state, "budget"), {"50-100"})
        state.record_question("budget")
        state.update("75", 2)
        self.assertEqual(self.values(state, "budget"), {"<= 75"})

    def test_size_question_allows_numeric_and_letter_answers(self):
        for answer, expected in (("US 8.5", "us 8.5"), ("M", "m"), ("extra large", "xl")):
            with self.subTest(answer=answer):
                state = SessionState({})
                state.record_question("size")
                state.update(answer, 1)
                self.assertEqual(self.values(state, "size"), {expected})

    def test_size_does_not_capture_budget_number(self):
        state = SessionState({})
        state.update("Size 8 shoes with a maximum budget of $100.", 1)
        self.assertEqual(self.values(state, "size"), {"8"})
        self.assertEqual(self.values(state, "budget"), {"<= 100"})

    def test_percent_composition_preserves_number_and_semantic_owner(self):
        state = SessionState({})
        message = "A shirt made of 80% cotton and 20% nylon."
        state.update(message, 1)
        self.assertIn("80% cotton", state.query())
        self.assertIn("20% nylon", state.query())
        composition = next(p for p in state.active_preferences() if p.value == "80% cotton")
        self.assertEqual(composition.depends_on, ("material", "cotton"))
        self.assertEqual(composition.source_turn, 1)
        self.assertEqual(composition.source_text, message)

    def test_material_correction_retracts_only_owned_composition(self):
        state = SessionState({})
        state.update("A tunic made of 80% cotton and 20% nylon.", 1)
        state.update("Linen instead of cotton.", 2)
        self.assertEqual(self.values(state, "material"), {"linen", "nylon"})
        self.assertIn("20% nylon", state.query())
        self.assertIn("tunic", state.query())
        self.assertNotIn("80", state.query())
        self.assertNotIn("cotton", state.query())

    def test_new_composition_replaces_prior_number_for_same_owner(self):
        state = SessionState({})
        state.update("A shirt with 80% cotton.", 1)
        state.update("Actually, 100% cotton.", 2)
        self.assertIn("100% cotton", state.query())
        self.assertNotIn("80", state.query())

    def test_same_message_composition_correction_keeps_only_latest_amount(self):
        state = SessionState({})
        state.update("A shirt with 80% cotton. Actually, 100% cotton.", 1)
        self.assertEqual(self.values(state, "material"), {"cotton"})
        self.assertIn("100% cotton", state.query())
        self.assertNotIn("80", state.query())

    def test_same_message_material_correction_preserves_other_composition(self):
        state = SessionState({})
        state.update("A tunic made of 80% cotton and 20% nylon. Linen instead of cotton.", 1)
        self.assertEqual(self.values(state, "material"), {"linen", "nylon"})
        self.assertIn("20% nylon", state.query())
        self.assertIn("tunic", state.query())
        self.assertNotIn("80", state.query())
        self.assertNotIn("cotton", state.query())

    def test_same_message_reintroduced_owner_does_not_revive_old_quantity(self):
        for message, owner, quantity in (
            ("An 80% cotton tunic. Linen instead of cotton. I prefer cotton after all.", "cotton", "80"),
            ("A jacket with 3 pockets. No pockets after all. Pockets are fine.", "pockets", "3"),
        ):
            with self.subTest(message=message):
                state = SessionState({})
                state.update(message, 1)
                self.assertIn(owner, state.query())
                self.assertNotIn(quantity, state.query())
                self.assertFalse(any(p.polarity == -1 and p.value == owner for p in state.active_preferences()))

    def test_old_composition_does_not_revive_when_material_returns(self):
        state = SessionState({})
        state.update("An 80% cotton tunic.", 1)
        state.update("Linen instead of cotton.", 2)
        state.update("I prefer cotton after all.", 3)
        self.assertIn("cotton", state.query())
        self.assertNotIn("80", state.query())

    def test_repeating_material_does_not_detach_its_composition(self):
        state = SessionState({})
        state.update("An 80% cotton tunic.", 1)
        revision = state.revision
        state.update("Cotton, yes.", 2)
        self.assertIn("80% cotton", state.query())
        self.assertEqual(state.revision, revision)

    def test_numbered_count_and_measurement_features_survive(self):
        for message, expected in (
            ("A jacket with 3 button closure.", "3 button closure"),
            ("Shoes with a 10 mm heel.", "10 mm heel"),
            ("Shoes with a 2.5 inch heel.", "2.5 inch heel"),
            ("A jacket with 3 pockets.", "3 pockets"),
        ):
            with self.subTest(message=message):
                state = SessionState({})
                state.update(message, 1)
                self.assertIn(expected, state.query())
                numeric = next(p for p in state.active_preferences() if p.value == expected)
                self.assertIsNotNone(numeric.depends_on)

    def test_prompted_budget_does_not_steal_composition_or_measurement(self):
        for message, expected in (("95% cotton", "95% cotton"), ("10 mm heel", "10 mm heel"), ("3 button closure", "3 button closure")):
            with self.subTest(message=message):
                state = SessionState({})
                state.record_question("budget")
                state.update(message, 1)
                self.assertEqual(self.values(state, "budget"), set())
                self.assertIn(expected, state.query())

    def test_unqualified_budget_anchor_does_not_capture_size_number(self):
        state = SessionState({})
        state.update("Size 8 shoes with a budget of 100.", 1)
        self.assertEqual(self.values(state, "size"), {"8"})
        self.assertEqual(self.values(state, "budget"), {"<= 100"})

    def test_negative_numbered_feature_does_not_enter_positive_query(self):
        state = SessionState({})
        state.update("A jacket, but no 3 button closure.", 1)
        self.assertNotIn("3", state.query())
        self.assertNotIn("button", state.query())
        self.assertIn("jackets", state.query())

    def test_retracted_feature_also_retracts_its_count(self):
        state = SessionState({})
        state.update("A jacket with 3 pockets.", 1)
        state.update("No pockets after all.", 2)
        self.assertNotIn("3", state.query())
        self.assertNotIn("pockets", state.query())
        self.assertIn("jackets", state.query())

    def test_explicit_value_retracts_same_value_negative(self):
        state = SessionState({})
        state.update("No polyester, please.", 1)
        state.update("Polyester is fine after all.", 2)
        self.assertEqual(self.values(state, "material"), {"polyester"})
        self.assertEqual(self.values(state, "material", -1), set())

    def test_negative_new_value_does_not_retract_unrelated_positive(self):
        state = SessionState({})
        state.update("I prefer cotton.", 1)
        state.update("And no wool, please.", 2)
        self.assertEqual(self.values(state, "material"), {"cotton"})
        self.assertEqual(self.values(state, "material", -1), {"wool"})

    def test_new_positive_clause_ends_prior_negation_scope(self):
        state = SessionState({})
        state.update("I don't want wool and I need cotton.", 1)
        self.assertEqual(self.values(state, "material", -1), {"wool"})
        self.assertEqual(self.values(state, "material"), {"cotton"})

    def test_same_turn_explicit_value_overrides_no_preference(self):
        state = SessionState({})
        state.update("Any color is fine, but I would prefer blue.", 1)
        self.assertEqual(self.values(state, "color"), {"blue"})
        self.assertEqual(self.values(state, "color", 0), set())

    def test_same_message_neutral_answer_obeys_clause_order(self):
        for message, positives, neutral in (
            ("A red shirt. Any color is fine.", set(), {"any"}),
            ("Any color is fine. Actually, a blue shirt.", {"blue"}, set()),
        ):
            with self.subTest(message=message):
                state = SessionState({})
                state.update(message, 1)
                self.assertEqual(self.values(state, "color"), positives)
                self.assertEqual(self.values(state, "color", 0), neutral)
                self.assertEqual("color" in state.unproductive_attributes, bool(neutral))
                self.assertEqual(self.values(state, "category"), {"shirts"})

    def test_no_preference_for_one_field_does_not_hide_another_field(self):
        state = SessionState({})
        state.update("I don't care about color and I need a cotton shirt.", 1)
        self.assertEqual(self.values(state, "color", 0), {"any"})
        self.assertEqual(self.values(state, "material"), {"cotton"})
        self.assertEqual(self.values(state, "category"), {"shirts"})

    def test_negated_preference_does_not_negate_product_category(self):
        state = SessionState({})
        state.update("I want a jacket without wool.", 1)
        self.assertEqual(self.values(state, "category"), {"jackets"})
        self.assertEqual(self.values(state, "material", -1), {"wool"})

    def test_no_show_compound_is_a_positive_product_description(self):
        for message, category in (
            ("No Show & Liner Socks", "socks"),
            ("No-show socks, please.", "socks"),
            ("I am looking for socks in the No Show & Liner Socks category.", "socks"),
            ("I want a no-show shirt.", "shirts"),
        ):
            with self.subTest(message=message):
                state = SessionState({})
                state.update(message, 1)
                self.assertIn(category, self.values(state, "category"))
                self.assertEqual(self.values(state, "feature"), {"no show"})
                self.assertFalse(any(p.polarity == -1 for p in state.active_preferences()))
                self.assertIn("no show", state.query())
                if "Liner" in message:
                    self.assertIn("liner", state.query())

    def test_no_show_compound_does_not_hide_real_negation(self):
        for message in (
            "No socks, please.",
            "No no-show socks, please.",
            "No no show socks, please.",
            "I do not want no-show socks.",
        ):
            with self.subTest(message=message):
                state = SessionState({})
                state.update(message, 1)
                self.assertEqual(self.values(state, "category"), set())
                self.assertEqual(self.values(state, "category", -1), {"socks"})
                self.assertNotIn("socks", state.query())
                if "show" in message:
                    self.assertEqual(self.values(state, "feature", -1), {"no show"})
                self.assertTrue(all(p.hard for p in state.active_preferences() if p.polarity == -1))

    def test_no_show_compound_preserves_separate_material_exclusions(self):
        for message in (
            "No-show socks without wool.",
            "No wool, but no show socks are fine.",
        ):
            with self.subTest(message=message):
                state = SessionState({})
                state.update(message, 1)
                self.assertEqual(self.values(state, "category"), {"socks"})
                self.assertEqual(self.values(state, "feature"), {"no show"})
                self.assertEqual(self.values(state, "material", -1), {"wool"})
                self.assertNotIn("wool", state.query())

    def test_no_matter_concessive_does_not_create_false_exclusions(self):
        for message in (
            "No matter if your hair is thick or thin, I want a beanie.",
            "No matter whether it is leather or canvas, I want a jacket.",
        ):
            with self.subTest(message=message):
                state = SessionState({})
                state.update(message, 1)
                self.assertFalse(any(preference.polarity == -1 for preference in state.active_preferences()))
                self.assertNotIn("whether", state.query())

        state = SessionState({})
        state.update("I want a jacket without wool.", 1)
        self.assertEqual(self.values(state, "material", -1), {"wool"})

    def test_no_show_compound_preserves_clause_order_corrections(self):
        state = SessionState({})
        state.update("A red pair of no-show socks. Actually, blue.", 1)
        self.assertEqual(self.values(state, "color"), {"blue"})
        self.assertEqual(self.values(state, "category"), {"socks"})
        self.assertEqual(self.values(state, "feature"), {"no show"})
        self.assertNotIn("red", state.query())
        state.update("No socks after all.", 2)
        self.assertEqual(self.values(state, "category"), set())
        self.assertEqual(self.values(state, "category", -1), {"socks"})
        state.update("Actually, no-show socks are fine.", 3)
        self.assertEqual(self.values(state, "category"), {"socks"})
        self.assertEqual(self.values(state, "category", -1), set())

    def test_no_show_description_does_not_demote_matching_invented_product(self):
        from mercury.catalog import product_from_dict
        from mercury.ranking import rank_constraints
        from mercury.types import Candidate

        socks = product_from_dict({"parent_asin": "invented-socks", "title": "Blue No Show Liner Socks"})
        shirt = product_from_dict({"parent_asin": "invented-shirt", "title": "Blue Cotton Shirt"})
        state = SessionState({})
        state.update("I am looking for socks in the No Show & Liner Socks category.", 1)
        ranked = rank_constraints([Candidate(socks, 2.0), Candidate(shirt, 1.0)], state.active_preferences())
        self.assertEqual(ranked[0].product.parent_asin, "invented-socks")
        self.assertEqual(ranked[0].score, 2.0)
        self.assertNotIn("constraint_penalty", ranked[0].route_scores)

    def test_explicit_brand_is_preserved_with_other_facts(self):
        state = SessionState({})
        state.update("I want blue shoes from New Balance.", 1)
        self.assertEqual(self.values(state, "brand"), {"new balance"})
        self.assertEqual(self.values(state, "category"), {"shoes"})

    def test_supplementary_prompt_answer_does_not_capture_discourse(self):
        state = SessionState({})
        state.record_question("material")
        state.update("Thanks, actually cotton.", 1)
        self.assertEqual(self.values(state, "material"), {"cotton"})

    def test_unknown_open_answer_can_supply_unlisted_brand(self):
        state = SessionState({})
        state.record_question("brand")
        state.update("Lark & Finch", 1)
        self.assertEqual(self.values(state, "brand"), {"lark & finch"})

    def test_prompted_fallback_rejects_pure_conversation_scaffolding(self):
        for message in ("I am still looking", "I would like you to keep searching", "That would be fine"):
            with self.subTest(message=message):
                state = SessionState({})
                state.update("A blue jacket.", 1)
                state.record_question("feature")
                before, revision = state.query(), state.revision
                state.update(message, 2)
                self.assertEqual(state.query(), before)
                self.assertEqual(state.revision, revision)
                self.assertIn("feature", state.unproductive_attributes)

    def test_prompted_fallback_cleans_courtesy_around_real_description(self):
        for attribute, message, expected in (
            ("feature", "Imported please", "imported"),
            ("feature", "I would like snap closure please", "snap closure"),
            ("brand", "I prefer Lark & Finch please", "lark & finch"),
        ):
            with self.subTest(message=message):
                state = SessionState({})
                state.record_question(attribute)
                state.update(message, 1)
                self.assertEqual(self.values(state, attribute), {expected})
                self.assertEqual(state.query(), expected)

    def test_unknown_units_do_not_become_budgets_from_a_prior_question(self):
        for message in ("Under 20 denier", "Between 3 and 5 buttons"):
            with self.subTest(message=message):
                state = SessionState({})
                state.record_question("budget")
                state.update(message, 1)
                self.assertEqual(self.values(state, "budget"), set())

    def test_bare_price_is_not_negative_in_no_more_than_budget(self):
        state = SessionState({})
        state.update("No more than $80 for shoes.", 1)
        self.assertEqual(self.values(state, "budget"), {"<= 80"})
        self.assertEqual(self.values(state, "category"), {"shoes"})

    def test_optional_feature_can_be_retracted_without_forbidding_it(self):
        state = SessionState({})
        state.update("I need a waterproof jacket with pockets.", 1)
        state.update("Waterproof isn't necessary anymore.", 2)
        self.assertEqual(self.values(state, "feature"), {"pockets"})
        self.assertEqual(self.values(state, "feature", -1), set())

    def test_open_vocabulary_suffix_relaxation_retracts_owned_quantity(self):
        for suffix in ("are no longer needed", "are not required", "aren't necessary"):
            with self.subTest(suffix=suffix):
                state = SessionState({}, alternatives_mode="grouped")
                state.update("A jacket with 3 snap closures.", 1)
                state.update(f"Snap closures {suffix}.", 2)
                self.assertEqual(state.query(), "jackets")
                self.assertFalse(any(
                    preference.active and preference.polarity != 0 and "snap closure" in preference.value
                    for preference in state.preferences
                ))
                self.assertFalse(any(
                    preference.active and preference.value == "longer"
                    for preference in state.preferences
                ))

    def test_neutral_language_does_not_invent_a_brand(self):
        state = SessionState({})
        state.record_question("brand")
        state.update("Just pick something for me.", 1)
        self.assertEqual(self.values(state, "brand"), set())
        self.assertEqual(state.query(), "")

    def test_no_preference_paraphrases_clear_only_named_attribute(self):
        for message in (
            "I have no material preference.",
            "I don't have a particular preference for fabric.",
            "I'm flexible on material.",
            "I'm not fussed about the fabric.",
        ):
            with self.subTest(message=message):
                state = SessionState({})
                state.update("A red wool coat.", 1)
                state.update(message, 2)
                self.assertEqual(self.values(state, "material"), set())
                self.assertEqual(self.values(state, "material", 0), {"any"})
                self.assertEqual(self.values(state, "category"), {"coats"})
                self.assertEqual(self.values(state, "color"), {"red"})

    def test_do_not_have_named_color_preference_clears_color_only(self):
        state = SessionState({})
        state.update("A red wool coat.", 1)
        state.update("I do not have a color preference.", 2)
        self.assertEqual(self.values(state, "color"), set())
        self.assertEqual(self.values(state, "color", 0), {"any"})
        self.assertEqual(self.values(state, "material"), {"wool"})
        self.assertEqual(self.values(state, "category"), {"coats"})

    def test_negative_feedback_with_specific_value_records_exclusion(self):
        state = SessionState({})
        state.update("I need a blue tote bag with a zipper.", 1)
        state.update("Not those leather ones; show me canvas instead.", 2)
        self.assertEqual(self.values(state, "material", -1), {"leather"})
        self.assertEqual(self.values(state, "material"), {"canvas"})
        self.assertEqual(self.values(state, "color"), {"blue"})
        self.assertEqual(self.values(state, "category"), {"bags"})

    def test_not_just_component_keeps_the_component_eligible(self):
        state = SessionState({})
        state.update("I need a bag with a leather body, not just leather handles.", 1)
        self.assertIn("leather", self.values(state, "material"))
        self.assertIn("body", self.values(state, "other"))
        self.assertIn("leather body", self.values(state, "other"))
        self.assertNotIn("handles", self.values(state, "other"))
        self.assertNotIn("handles", self.values(state, "other", -1))

    def test_not_merely_component_keeps_the_whole_product_requirement(self):
        state = SessionState({})
        state.update("I need a leather outer shell, not merely leather elbow patches.", 1)
        self.assertEqual(self.values(state, "material"), {"leather"})
        self.assertIn("outer shell", self.values(state, "other"))
        self.assertIn("leather outer shell", self.values(state, "other"))
        self.assertNotIn("elbow patches", self.values(state, "other"))
        self.assertNotIn("elbow patches", self.values(state, "other", -1))

    def test_uncertainty_does_not_create_a_named_exclusion(self):
        for message in ("I am not sure about blue.", "I do not know about leather."):
            with self.subTest(message=message):
                state = SessionState({})
                state.update(message, 1)
                self.assertEqual(self.values(state, "color", -1), set())
                self.assertEqual(self.values(state, "material", -1), set())

    def test_only_inferred_soft_preferences_decay(self):
        state = SessionState({})
        state.update("I need a shirt with subtle geometric detailing.", 1)
        state.update("Keep looking.", 4)
        active = state.active_preferences()
        effective = state.effective_preferences(decay_turns=3)
        inferred = next(p for p in effective if p.source_kind == "inferred")
        original = next(p for p in active if p.source_kind == "inferred")
        self.assertLess(inferred.confidence, original.confidence)
        explicit = next(p for p in effective if p.attribute == "category")
        self.assertEqual(explicit.confidence, 1.0)

    def test_intent_override_retires_incompatible_inferred_residual(self):
        state = SessionState({})
        state.update("I want a bag with geometric detailing.", 1)
        state.update("Actually, no geometric detailing; show me floral instead.", 2)
        inferred = [p.value for p in state.active_preferences()
                    if p.source_kind == "inferred" and p.polarity == 1]
        self.assertNotIn("geometric detailing", inferred)

    def test_feature_replacement_does_not_clear_unrelated_features(self):
        state = SessionState({})
        state.update("A waterproof jacket with pockets.", 1)
        state.update("Make it breathable rather than waterproof.", 2)
        self.assertEqual(self.values(state, "feature"), {"breathable", "pockets"})
        self.assertEqual(self.values(state, "feature", -1), {"waterproof"})

    def test_material_description_does_not_invent_brand(self):
        state = SessionState({})
        state.update("A shirt made from cotton.", 1)
        self.assertEqual(self.values(state, "brand"), set())
        self.assertEqual(self.values(state, "material"), {"cotton"})

    def test_exception_phrase_excludes_material(self):
        state = SessionState({})
        state.update("Anything but wool.", 1)
        self.assertEqual(self.values(state, "material", -1), {"wool"})
        self.assertEqual(self.values(state, "material"), set())

    def test_other_than_and_except_create_exclusions_without_spurious_values(self):
        state = SessionState({})
        state.update("I want blue, anything other than red.", 1)
        self.assertEqual(self.values(state, "color"), {"blue"})
        self.assertEqual(self.values(state, "color", -1), {"red"})
        state = SessionState({})
        state.update("Any color except red.", 1)
        self.assertEqual(self.values(state, "color"), set())
        self.assertEqual(self.values(state, "color", -1), {"red"})

    def test_refusal_paraphrase_is_unproductive_but_not_persistent(self):
        state = SessionState({})
        state.record_question("budget")
        state.update("I'm not comfortable answering questions about my budget.", 1)
        self.assertFalse(state.last_update_informative)
        self.assertIn("budget", state.unproductive_attributes)
        state.update("My budget is 60 dollars.", 2)
        self.assertEqual(self.values(state, "budget"), {"<= 60"})
        self.assertNotIn("budget", state.unproductive_attributes)

    def test_surprise_me_does_not_become_an_attribute_value(self):
        state = SessionState({})
        state.record_question("brand")
        state.update("Surprise me", 1)
        self.assertEqual(state.query(), "")
        self.assertFalse(state.last_update_informative)

    def test_open_vocabulary_retains_subcategory_and_feature_with_known_color(self):
        state = SessionState({})
        state.update("A purple tunic with bell sleeves.", 1)
        self.assertEqual(self.values(state, "color"), {"purple"})
        for term in ("purple", "tunic", "bell sleeves"):
            self.assertIn(term, state.query())

    def test_open_vocabulary_retains_gender_and_sport_alongside_category(self):
        state = SessionState({})
        state.update("Men’s basketball footwear.", 1)
        self.assertEqual(self.values(state, "category"), {"shoes"})
        self.assertIn("men's", state.query())
        self.assertIn("basketball", state.query())
        self.assertIn("shoes", state.query())

    def test_open_vocabulary_retains_unknown_feature_and_source_evidence(self):
        state = SessionState({})
        message = "Need a leather belt with a ratchet buckle."
        state.update(message, 1)
        self.assertIn("ratchet buckle", state.query())
        residual = next(p for p in state.active_preferences() if "ratchet" in p.value)
        self.assertEqual(residual.source_turn, 1)
        self.assertEqual(residual.source_text, message)
        self.assertGreater(residual.confidence, 0)
        self.assertLess(residual.confidence, 1)

    def test_open_vocabulary_answer_is_retained_after_known_facts(self):
        for answer in ("Imported", "Snap closure"):
            with self.subTest(answer=answer):
                state = SessionState({})
                state.update("A blue jacket.", 1)
                state.record_question("feature")
                revision = state.revision
                state.update(answer, 2)
                self.assertIn(answer.lower(), state.query())
                self.assertIn("jackets", state.query())
                self.assertGreater(state.revision, revision)

    def test_material_correction_preserves_open_vocabulary_without_old_material(self):
        state = SessionState({})
        state.update("A purple leather tunic with bell sleeves.", 1)
        state.update("Actually, canvas instead of leather.", 2)
        for term in ("purple", "canvas", "tunic", "bell sleeves"):
            self.assertIn(term, state.query())
        self.assertNotIn("leather", state.query())

    def test_open_vocabulary_negative_does_not_enter_positive_query(self):
        state = SessionState({})
        state.update("A leather belt without a ratchet buckle.", 1)
        self.assertNotIn("ratchet", state.query())
        self.assertNotIn("buckle", state.query())
        self.assertIn("belts", state.query())
        self.assertTrue(any("ratchet" in p.value and p.polarity == -1 for p in state.active_preferences()))

    def test_open_vocabulary_feature_replacement_keeps_other_description(self):
        state = SessionState({})
        state.update("A purple tunic with bell sleeves.", 1)
        state.update("Raglan sleeves instead of bell sleeves.", 2)
        self.assertIn("tunic", state.query())
        self.assertIn("purple", state.query())
        self.assertIn("raglan sleeves", state.query())
        self.assertNotIn("bell", state.query())

    def test_open_vocabulary_does_not_capture_generic_reaction_words(self):
        state = SessionState({})
        state.update("A red coat.", 1)
        before = state.query()
        for turn, message in enumerate((
            "That isn't the one I meant.",
            "I would rather see some other choices.",
            "Please continue searching, thanks.",
            "I have nothing else to say about my preferences.",
        ), 2):
            state.update(message, turn)
            self.assertEqual(state.query(), before)
            self.assertFalse(state.last_update_informative)

    def test_rare_product_name_survives_without_any_known_facet(self):
        state = SessionState({})
        state.update("I'm looking for a dirndl.", 1)
        self.assertIn("dirndl", state.query())

    def test_explicit_negative_known_material_does_not_leak_through_residual(self):
        state = SessionState({})
        state.update("A tunic, but please avoid leather trim.", 1)
        self.assertIn("tunic", state.query())
        self.assertNotIn("leather", state.query())
        self.assertNotIn("trim", state.query())
        self.assertNotIn("avoid", state.query())

    def test_contextual_open_vocabulary_fact_can_be_retracted_later(self):
        state = SessionState({})
        state.update("A blue jacket.", 1)
        state.record_question("feature")
        state.update("Snap closure", 2)
        state.record_question(None)
        state.update("No snap closure after all.", 3)
        self.assertNotIn("snap", state.query())
        self.assertNotIn("closure", state.query())
        self.assertIn("jackets", state.query())

    def test_open_vocabulary_correction_does_not_duplicate_unknown_material(self):
        state = SessionState({})
        state.update("A tunic. The material should be jute.", 1)
        self.assertIn("jute", state.query())
        state.update("Actually use cotton instead of jute.", 2)
        self.assertIn("cotton", state.query())
        self.assertIn("tunic", state.query())
        self.assertNotIn("jute", state.query())

    def test_repeated_open_vocabulary_fact_keeps_query_revision_stable(self):
        state = SessionState({})
        state.update("A purple tunic with bell sleeves.", 1)
        revision = state.revision
        state.update("Bell sleeves, please.", 2)
        self.assertEqual(state.revision, revision)

    def test_feature_retraction_preserves_other_features(self):
        state = SessionState({})
        state.update("A waterproof jacket with a hood and pockets.", 1)
        state.update("Actually, no hood.", 2)
        self.assertEqual(self.values(state, "feature"), {"waterproof", "pockets"})
        self.assertEqual(self.values(state, "feature", -1), {"hood"})
        self.assertNotIn("hood", state.query())

    def test_modes_have_deliberately_different_memory(self):
        states = {mode: SessionState({}, mode=mode) for mode in ("ledger", "latest", "history")}
        for state in states.values():
            state.update("A red cotton shirt.", 1)
            state.update("Actually blue, and also waterproof.", 2)
        self.assertNotIn("red", states["ledger"].query())
        self.assertNotIn("red", states["latest"].query())
        self.assertIn("red", states["history"].query())
        for state in states.values():
            self.assertIn("shirts", state.query())
            self.assertIn("blue", state.query())

    def test_latest_is_simple_slots_while_ledger_accumulates_features(self):
        ledger = SessionState({})
        latest = SessionState({}, mode="latest")
        for state in (ledger, latest):
            state.update("A waterproof jacket.", 1)
            state.update("Also breathable.", 2)
        self.assertEqual(self.values(ledger, "feature"), {"waterproof", "breathable"})
        self.assertEqual(self.values(latest, "feature"), {"breathable"})

    def test_control_modes_keep_whole_message_attribute_groups(self):
        for mode, expected in (
            ("latest", {"red", "blue"}),
            ("history", {"yellow", "red", "blue"}),
        ):
            with self.subTest(mode=mode):
                state = SessionState({}, mode=mode)
                state.update("A yellow cotton shirt.", 1)
                state.update("A red shirt. Actually, blue.", 2)
                self.assertEqual(self.values(state, "color"), expected)
                self.assertEqual(self.values(state, "material"), {"cotton"})
                self.assertEqual(self.values(state, "category"), {"shirts"})

    def test_duplicate_fact_does_not_change_revision(self):
        state = SessionState({})
        state.update("A cotton shirt.", 1)
        revision = state.revision
        state.update("Cotton, yes.", 2)
        self.assertEqual(state.revision, revision)
        self.assertFalse(state.last_update_informative)

    def test_new_instance_resets_all_session_state_and_copies_profile(self):
        profile = {"preference_tags": ["casual"], "summary": "A frequent shopper"}
        first = SessionState(profile)
        first.update("A red dress.", 1)
        first.record_question("brand")
        first.update("I don't know.", 2)
        second = SessionState(profile)
        self.assertEqual(second.history, [])
        self.assertEqual(second.preferences, [])
        self.assertEqual(second.query(), "")
        self.assertEqual(second.asked_counts, {})
        self.assertEqual(second.unproductive_attributes, set())
        self.assertIsNone(second.last_question)
        self.assertEqual(second.revision, 0)
        self.assertEqual(second.turn, 0)
        first.profile["preference_tags"].append("formal")
        self.assertEqual(profile["preference_tags"], ["casual"])
        self.assertEqual(second.profile, profile)

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            SessionState({}, mode="oracle")


class ExplicitAlternativesTest(unittest.TestCase):
    def positives(self, state, attribute):
        return [p for p in state.active_preferences() if p.attribute == attribute and p.polarity == 1]

    def values(self, state, attribute, polarity=1):
        return {p.value for p in state.active_preferences() if p.attribute == attribute and p.polarity == polarity}

    def group(self, state, attribute):
        groups = {p.alternative_group for p in self.positives(state, attribute)}
        self.assertEqual(len(groups), 1)
        self.assertNotIn(None, groups)
        return next(iter(groups))

    def test_off_retains_frozen_explicit_list_behavior(self):
        state = SessionState({})
        self.assertEqual(state.alternatives_mode, "off")
        state.update("Either a brown or black cotton belt works.", 1)
        self.assertEqual(state.active_preferences(), [])
        self.assertEqual(state.unsupported_alternatives, [])

    def test_list_is_preserved_in_both_new_modes(self):
        for mode in ("parse", "grouped"):
            for message in (
                "Either a brown or black cotton belt works.",
                "Either brown or black cotton belt is fine.",
                "Either brown or black cotton belt will do.",
            ):
                with self.subTest(mode=mode, message=message):
                    state = SessionState({}, "ledger", mode)
                    state.update(message, 1)
                    self.assertEqual(self.values(state, "color"), {"brown", "black"})
                    self.assertEqual(self.values(state, "material"), {"cotton"})
                    self.assertEqual(self.values(state, "category"), {"belts"})
                    self.assertNotIn("works", state.query())
                    if mode == "grouped":
                        self.group(state, "color")
                    else:
                        self.assertTrue(all(p.alternative_group is None for p in state.active_preferences()))

    def test_material_and_feature_lists_are_known_value_groups(self):
        for attribute, message, values in (
            ("material", "Must be cotton or linen or wool.", {"cotton", "linen", "wool"}),
            ("feature", "It must be waterproof or insulated.", {"waterproof", "insulated"}),
        ):
            with self.subTest(attribute=attribute):
                state = SessionState({}, alternatives_mode="grouped")
                state.update(message, 1)
                self.assertEqual(self.values(state, attribute), values)
                self.group(state, attribute)
                self.assertTrue(all(p.hard for p in self.positives(state, attribute)))

    def test_unrestricted_answer_clears_only_named_choice_group(self):
        for mode in ("parse", "grouped"):
            with self.subTest(mode=mode):
                state = SessionState({}, alternatives_mode=mode)
                state.update("Either black or brown cotton belt works.", 1)
                state.update("Any color works.", 2)
                self.assertEqual(self.values(state, "color"), set())
                self.assertEqual(self.values(state, "color", 0), {"any"})
                self.assertEqual(self.values(state, "material"), {"cotton"})
                self.assertEqual(self.values(state, "category"), {"belts"})
                self.assertTrue(all(p.alternative_group is None for p in state.active_preferences()))

    def test_ordinary_conjunction_remains_ungrouped(self):
        for mode in ("parse", "grouped"):
            with self.subTest(mode=mode):
                state = SessionState({}, alternatives_mode=mode)
                state.update("Must be waterproof and insulated. No leather.", 1)
                self.assertEqual(self.values(state, "feature"), {"waterproof", "insulated"})
                self.assertEqual(self.values(state, "material", -1), {"leather"})
                self.assertTrue(all(p.alternative_group is None for p in state.active_preferences()))

    def test_neither_nor_are_independent_exclusions_in_new_modes(self):
        for mode in ("parse", "grouped"):
            with self.subTest(mode=mode):
                state = SessionState({}, alternatives_mode=mode)
                state.update("A belt. Neither leather nor wool.", 1)
                self.assertEqual(self.values(state, "material"), set())
                self.assertEqual(self.values(state, "material", -1), {"leather", "wool"})
                self.assertEqual(self.values(state, "category"), {"belts"})
                self.assertTrue(all(p.hard for p in state.active_preferences() if p.polarity == -1))
                self.assertTrue(all(p.alternative_group is None for p in state.active_preferences()))

    def test_off_neither_nor_behavior_is_not_changed(self):
        state = SessionState({})
        state.update("Neither leather nor wool.", 1)
        self.assertEqual(self.values(state, "material"), {"leather", "wool"})

    def test_negated_or_does_not_create_a_positive_group(self):
        state = SessionState({}, alternatives_mode="grouped")
        state.update("No cotton or linen.", 1)
        self.assertEqual(self.values(state, "material"), set())
        self.assertEqual(self.values(state, "material", -1), {"cotton", "linen"})
        self.assertTrue(all(p.alternative_group is None for p in state.active_preferences()))

    def test_unsupported_boolean_constructions_do_not_claim_a_group(self):
        for message in (
            "Must be cotton and linen or wool.",
            "Must be cotton or linen and wool.",
            "Must be cotton or linen and wool or silk.",
            "Must be cotton or black.",
            "Must be cotton or (linen and wool).",
            "Must be (cotton or linen).",
            "Either black and brown or blue works.",
        ):
            with self.subTest(message=message):
                state = SessionState({}, alternatives_mode="grouped")
                state.update(message, 1)
                self.assertTrue(all(p.alternative_group is None for p in state.active_preferences()))
                self.assertTrue(state.unsupported_alternatives)
                self.assertLessEqual(len(state.unsupported_alternatives), 8)

    def test_repeating_one_option_preserves_group_and_other_requirements(self):
        state = SessionState({}, alternatives_mode="grouped")
        state.update("Must be black or brown. A cotton belt.", 1)
        group, revision = self.group(state, "color"), state.revision
        state.update("Black, please.", 2)
        self.assertEqual(self.values(state, "color"), {"black", "brown"})
        self.assertEqual(self.group(state, "color"), group)
        self.assertEqual(state.revision, revision)
        self.assertTrue(all(p.hard for p in self.positives(state, "color")))
        self.assertEqual(self.values(state, "material"), {"cotton"})

    def test_repeating_whole_list_keeps_identity_and_revision(self):
        state = SessionState({}, alternatives_mode="grouped")
        state.update("Cotton or linen.", 1)
        group, revision = self.group(state, "material"), state.revision
        state.update("Linen or cotton.", 2)
        self.assertEqual(self.group(state, "material"), group)
        self.assertEqual(state.revision, revision)
        self.assertEqual(len(self.positives(state, "material")), 2)
        self.assertFalse(state.unsupported_alternatives)

    def test_explicit_selection_retires_other_members_and_keeps_force(self):
        for correction in ("Actually, cotton.", "Cotton instead."):
            with self.subTest(correction=correction):
                state = SessionState({}, alternatives_mode="grouped")
                state.update("Must be cotton or linen. A blue shirt.", 1)
                state.update(correction, 2)
                self.assertEqual(self.values(state, "material"), {"cotton"})
                self.assertTrue(self.positives(state, "material")[0].hard)
                self.assertNotIn("linen", state.query())
                self.assertEqual(self.values(state, "category"), {"shirts"})
                self.assertEqual(self.values(state, "color"), {"blue"})

    def test_explicit_replacement_retires_old_option_group(self):
        for first, correction, attribute, expected in (
            ("Must be cotton or linen.", "Actually, linen or wool.", "material", {"linen", "wool"}),
            ("Must be black or brown.", "Actually blue or green, still cotton.", "color", {"blue", "green"}),
            ("Must be waterproof or insulated.", "Actually padded or lightweight.", "feature", {"padded", "lightweight"}),
        ):
            with self.subTest(attribute=attribute):
                state = SessionState({}, alternatives_mode="grouped")
                state.update(first, 1)
                old_group = self.group(state, attribute)
                state.update(correction, 2)
                self.assertEqual(self.values(state, attribute), expected)
                self.assertNotEqual(self.group(state, attribute), old_group)
                self.assertFalse(any(p.alternative_group == old_group for p in state.active_preferences()))

    def test_negative_option_rejection_preserves_singleton_group_force(self):
        state = SessionState({}, alternatives_mode="grouped")
        state.update("Must be cotton or linen. A blue shirt.", 1)
        group = self.group(state, "material")
        state.update("Not linen.", 2)
        self.assertEqual(self.values(state, "material"), {"cotton"})
        self.assertEqual(self.group(state, "material"), group)
        self.assertTrue(self.positives(state, "material")[0].hard)
        negative = next(p for p in state.active_preferences() if p.value == "linen")
        self.assertEqual(negative.polarity, -1)
        self.assertIsNone(negative.alternative_group)
        self.assertEqual(self.values(state, "category"), {"shirts"})

    def test_rejection_preserves_force_after_hard_group_restatement(self):
        state = SessionState({}, alternatives_mode="grouped")
        state.update("Cotton or linen.", 1)
        state.update("Must be cotton.", 2)
        state.update("Not cotton.", 3)
        self.assertEqual(self.values(state, "material"), {"linen"})
        self.assertTrue(self.positives(state, "material")[0].hard)

    def test_feature_group_replacement_keeps_independent_features(self):
        state = SessionState({}, alternatives_mode="grouped")
        state.update("Must be waterproof or insulated. It needs pockets.", 1)
        state.update("Actually padded or lightweight.", 2)
        self.assertEqual(self.values(state, "feature"), {"padded", "lightweight", "pockets"})
        pockets = next(p for p in self.positives(state, "feature") if p.value == "pockets")
        self.assertIsNone(pockets.alternative_group)
        self.assertTrue(pockets.hard)

    def test_additive_feature_keeps_an_unrelated_choice_group(self):
        state = SessionState({}, alternatives_mode="grouped")
        state.update("I need a waterproof or insulated jacket.", 1)
        group, revision = self.group(state, "feature"), state.revision
        state.update("Actually, I also need pockets.", 2)
        self.assertEqual(self.values(state, "feature"), {"waterproof", "insulated", "pockets"})
        weather = [p for p in self.positives(state, "feature") if p.value != "pockets"]
        self.assertEqual({p.alternative_group for p in weather}, {group})
        self.assertTrue(all(p.hard for p in weather))
        pockets = next(p for p in self.positives(state, "feature") if p.value == "pockets")
        self.assertIsNone(pockets.alternative_group)
        self.assertTrue(pockets.hard)
        self.assertEqual(state.revision, revision + 1)
        self.assertEqual(self.values(state, "category"), {"jackets"})

    def test_overlapping_new_list_preserves_previous_attribute_atomically(self):
        state = SessionState({}, alternatives_mode="grouped")
        state.update("Must be cotton or linen. A red shirt.", 1)
        group = self.group(state, "material")
        state.update("Linen or wool. Blue.", 2)
        self.assertEqual(self.values(state, "material"), {"cotton", "linen"})
        self.assertEqual(self.group(state, "material"), group)
        self.assertTrue(all(p.hard for p in self.positives(state, "material")))
        self.assertEqual(self.values(state, "color"), {"blue"})
        self.assertEqual(self.values(state, "category"), {"shirts"})
        self.assertIn({"attribute": "material", "reason": "overlapping alternatives require an explicit replacement"},
                      state.unsupported_alternatives)
        state.update("Cotton, please.", 3)
        self.assertEqual(state.unsupported_alternatives, [])

    def test_overlapping_subset_does_not_silently_narrow_a_group(self):
        state = SessionState({}, alternatives_mode="grouped")
        state.update("Cotton or linen or wool.", 1)
        revision = state.revision
        state.update("Cotton or linen.", 2)
        self.assertEqual(self.values(state, "material"), {"cotton", "linen", "wool"})
        self.assertEqual(state.revision, revision)
        self.assertTrue(state.unsupported_alternatives)

    def test_rejected_overlapping_list_keeps_an_independent_exclusion(self):
        state = SessionState({}, alternatives_mode="grouped")
        state.update("Must be cotton or linen. A red shirt.", 1)
        group, revision = self.group(state, "material"), state.revision
        state.update("Linen or wool and no leather. Blue.", 2)
        self.assertEqual(self.values(state, "material"), {"cotton", "linen"})
        self.assertEqual(self.group(state, "material"), group)
        self.assertTrue(all(p.hard for p in self.positives(state, "material")))
        self.assertEqual(self.values(state, "material", -1), {"leather"})
        leather = next(p for p in state.active_preferences() if p.value == "leather")
        self.assertTrue(leather.hard)
        self.assertIsNone(leather.alternative_group)
        self.assertEqual(state.revision, revision + 1)
        self.assertEqual(self.values(state, "color"), {"blue"})
        self.assertEqual(self.values(state, "category"), {"shirts"})
        self.assertEqual(state.unsupported_alternatives, [
            {"attribute": "material", "reason": "overlapping alternatives require an explicit replacement"},
        ])
        rejected = [p for p in state.history[-1].preferences if p.value in {"linen", "wool"}]
        self.assertEqual(len(rejected), 2)
        self.assertTrue(all(not p.active for p in rejected))

    def test_group_only_change_invalidates_revision_with_identical_query(self):
        state = SessionState({}, alternatives_mode="grouped")
        state.update("Must be cotton and linen.", 1)
        query, revision = state.query(), state.revision
        self.assertTrue(all(p.alternative_group is None for p in state.active_preferences()))
        state.update("Must be cotton or linen.", 2)
        self.assertEqual(state.query(), query)
        self.assertEqual(state.revision, revision + 1)
        self.group(state, "material")

    def test_only_selection_retires_the_other_option(self):
        for message, selected in (
            ("Cotton only.", "cotton"), ("Only cotton.", "cotton"),
            ("Only linen.", "linen"), ("It must be only linen.", "linen"),
        ):
            with self.subTest(message=message):
                state = SessionState({}, alternatives_mode="grouped")
                state.update("Must be cotton or linen. A blue shirt.", 1)
                revision = state.revision
                state.update(message, 2)
                self.assertEqual(self.values(state, "material"), {selected})
                self.assertTrue(self.positives(state, "material")[0].hard)
                self.assertIsNone(self.positives(state, "material")[0].alternative_group)
                self.assertGreater(state.revision, revision)
                self.assertEqual(self.values(state, "color"), {"blue"})
                self.assertEqual(self.values(state, "category"), {"shirts"})

    def test_choice_corrections_resolve_the_group_created_in_the_same_message(self):
        for correction, expected in (
            ("Cotton only.", {"cotton"}), ("Only linen.", {"linen"}),
            ("It must be cotton and linen.", {"cotton", "linen"}),
            ("It must be both cotton and linen.", {"cotton", "linen"}),
        ):
            with self.subTest(correction=correction):
                state = SessionState({}, alternatives_mode="grouped")
                state.update("Must be cotton or linen. A blue shirt. " + correction, 1)
                self.assertEqual(self.values(state, "material"), expected)
                self.assertTrue(all(p.alternative_group is None for p in state.active_preferences()))
                self.assertTrue(all(p.hard for p in self.positives(state, "material")))
                self.assertEqual(self.values(state, "color"), {"blue"})
                self.assertEqual(self.values(state, "category"), {"shirts"})
                self.assertEqual(state.revision, 1)

    def test_choice_syntax_without_a_live_group_keeps_ungrouped_behavior(self):
        for mode in ("off", "parse", "grouped"):
            for correction, expected in (
                ("Cotton only.", {"silk", "cotton"}),
                ("Only cotton.", {"silk", "cotton"}),
                ("It must be cotton and linen.", {"silk", "cotton", "linen"}),
            ):
                with self.subTest(mode=mode, correction=correction):
                    state = SessionState({}, alternatives_mode=mode)
                    state.update("A silk shirt.", 1)
                    state.update(correction, 2)
                    self.assertEqual(self.values(state, "material"), expected)
                    self.assertTrue(all(p.alternative_group is None for p in state.active_preferences()))

    def test_not_only_is_not_an_exclusive_selection(self):
        state = SessionState({}, alternatives_mode="grouped")
        state.update("Must be cotton or linen.", 1)
        group, revision = self.group(state, "material"), state.revision
        state.update("Not only cotton.", 2)
        self.assertEqual(self.values(state, "material"), {"cotton", "linen"})
        self.assertEqual(self.group(state, "material"), group)
        self.assertEqual(state.revision, revision)

    def test_unrestricted_color_and_separate_material_list_keep_their_scope(self):
        for mode in ("off", "parse", "grouped"):
            with self.subTest(mode=mode):
                state = SessionState({}, alternatives_mode=mode)
                state.update("I need a shirt. It must be black or brown.", 1)
                revision = state.revision
                state.update("Any color works and cotton or linen is fine.", 2)
                self.assertEqual(self.values(state, "color"), set())
                self.assertEqual(self.values(state, "color", 0), {"any"})
                self.assertEqual(self.values(state, "category"), {"shirts"})
                self.assertIn("color", state.unproductive_attributes)
                self.assertGreater(state.revision, revision)
                self.assertEqual(self.values(state, "material"),
                                 set() if mode == "off" else {"cotton", "linen"})
                if mode == "grouped":
                    self.group(state, "material")
                else:
                    self.assertTrue(all(p.alternative_group is None for p in state.active_preferences()))

    def test_explicit_conjunction_replaces_an_earlier_alternative_group(self):
        for message in ("It must be both cotton and linen.", "It must be cotton and linen."):
            with self.subTest(message=message):
                state = SessionState({}, alternatives_mode="grouped")
                state.update("Must be cotton or linen.", 1)
                revision = state.revision
                state.update(message, 2)
                self.assertEqual(self.values(state, "material"), {"cotton", "linen"})
                self.assertTrue(all(p.hard for p in self.positives(state, "material")))
                self.assertTrue(all(p.alternative_group is None for p in state.active_preferences()))
                self.assertGreater(state.revision, revision)

    def test_only_selection_is_scoped_to_its_attribute(self):
        state = SessionState({}, alternatives_mode="grouped")
        state.update("Must be cotton or linen. Must be black or brown.", 1)
        color_group = self.group(state, "color")
        state.update("Cotton only. Black or brown.", 2)
        self.assertEqual(self.values(state, "material"), {"cotton"})
        self.assertEqual(self.values(state, "color"), {"black", "brown"})
        self.assertEqual(self.group(state, "color"), color_group)

    def test_invalid_alternatives_mode_is_rejected(self):
        for mode in ("automatic", None, False, 1):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                SessionState({}, alternatives_mode=mode)

    def test_grouped_alternatives_require_ledger_state(self):
        for mode in ("latest", "history"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                SessionState({}, mode=mode, alternatives_mode="grouped")
            for alternatives_mode in ("off", "parse"):
                state = SessionState({}, mode=mode, alternatives_mode=alternatives_mode)
                self.assertEqual(state.mode, mode)


if __name__ == "__main__":
    unittest.main()
