import json
import socket
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import agent as public_entrypoint
import starter.agent as starter_entrypoint
from mercury.agent import Agent
from mercury.config import Config
from mercury.contrast import CONTRAST_VERSION
from mercury.model_assets import MODELS, file_sha256
from mercury.neural import DOCUMENT_VERSION
from mercury.retrieval import terms
from mercury.types import Candidate


CONTRACT = json.loads((Path(__file__).resolve().parents[1] / "docs" / "agent_api_contract.json").read_text())


class AgentTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "catalog.jsonl"
        rows = [{"parent_asin": str(index), "title": "Blue cotton shirt", "categories": ["Shirts"]}
                for index in range(12)]
        self.path.write_text("\n".join(map(json.dumps, rows)))
        self.agent = Agent(self.path)

    def tearDown(self):
        self.agent.close()
        self.temp.cleanup()

    def assert_matches_schema(self, value, schema):
        types = {"object": (dict,), "array": (list,), "string": (str,), "integer": (int,),
                 "number": (int, float), "null": (type(None),)}
        if "type" in schema:
            allowed = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
            self.assertIn(type(value), tuple(kind for name in allowed for kind in types[name]))
        if "enum" in schema:
            self.assertIn(value, schema["enum"])
        if "const" in schema:
            self.assertEqual(value, schema["const"])
        for keyword, operation in (("minimum", self.assertGreaterEqual), ("maximum", self.assertLessEqual)):
            if keyword in schema:
                operation(value, schema[keyword])
        if isinstance(value, str) and "minLength" in schema:
            self.assertGreaterEqual(len(value), schema["minLength"])
        if isinstance(value, list):
            if "maxItems" in schema:
                self.assertLessEqual(len(value), schema["maxItems"])
            for item in value:
                self.assert_matches_schema(item, schema["items"])
        if isinstance(value, dict):
            self.assertTrue(set(schema.get("required", [])) <= value.keys())
            if schema.get("additionalProperties") is False:
                self.assertTrue(value.keys() <= schema["properties"].keys())
            for name, item in value.items():
                self.assert_matches_schema(item, schema["properties"][name])

    def assert_legal_response(self, response, agent=None, count=10):
        agent = agent or self.agent
        self.assert_matches_schema(response, CONTRACT["turn_response"])
        ids = [item["parent_asin"] for item in response["recommendations"]]
        self.assertEqual(len(ids), count)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(set(ids) <= agent.catalog.by_id.keys())
        self.assertEqual(response["usage"]["completion_tokens"], 0)
        return ids

    def test_official_contract_and_unique_ids(self):
        self.agent.reset("a", {})
        response = self.agent.respond("a", "A cotton shirt", 1, 10)
        self.assert_legal_response(response)
        self.assertEqual(response["ask_attribute"], "other")

    def test_public_entrypoints_load_selected_config_and_match_official_contract(self):
        root = Path(self.temp.name)
        (root / "configs").mkdir()
        (root / "configs" / "selected.json").write_text(json.dumps({
            "question_policy": "none", "slate_size": 3, "evidence_ranking": False,
            "artifact_dir": "packaged-assets",
        }))
        profile = {"purchase_frequency": "occasionally", "average_prior_rating": None,
                   "rating_style": "balanced", "preference_tags": [], "summary": ""}
        self.assertIs(starter_entrypoint.Agent, public_entrypoint.Agent)
        for entrypoint in (public_entrypoint, starter_entrypoint):
            with self.subTest(entrypoint=entrypoint.__name__), \
                    patch.object(public_entrypoint, "__file__", str(root / "agent.py")):
                agent = entrypoint.Agent(self.path)
                try:
                    self.assertEqual(agent.config.artifact_dir, str(root.resolve() / "packaged-assets"))
                    self.assertFalse(agent.config.evidence_ranking)
                    reset = {"session_id": "public", "user_profile": profile}
                    request = {"session_id": "public", "user_message": "A cotton shirt", "turn": 1, "top_k": 10}
                    self.assert_matches_schema(reset, CONTRACT["reset_request"])
                    self.assert_matches_schema(request, CONTRACT["turn_request"])
                    agent.reset(**reset)
                    response = agent.respond(**request)
                    self.assert_legal_response(response, agent, count=3)
                    self.assertIsNone(response["ask_attribute"])
                    self.assert_legal_response(agent.respond(**{**request, "turn": 10}), agent)
                finally:
                    agent.close()

    def test_reset_isolates_and_clears_sessions(self):
        self.agent.reset("a", {})
        self.agent.reset("b", {})
        self.agent.respond("a", "Blue cotton shirt", 1, 10)
        self.assertEqual(self.agent.sessions["b"].query(), "")
        self.agent.reset("a", {})
        self.assertEqual(self.agent.sessions["a"].query(), "")
        with self.assertRaises(RuntimeError):
            self.agent.respond("missing", "shirt", 1, 10)

    def test_empty_query_fallback_uses_the_same_bounded_message(self):
        self.agent.reset("a", {})
        message = " " * 8000 + "recognizablesuffix shirt"
        with patch("mercury.agent.terms", wraps=terms) as tokenize:
            self.agent.respond("a", message, 1, 10)
        tokenize.assert_called_once_with(message[:8000])
        self.assertEqual(self.agent.sessions["a"].history[-1].text, message[:8000])
        self.assertEqual(self.agent.last_diagnostics["query"], "")
        self.assertNotIn("recognizablesuffix", self.agent.last_diagnostics["query"])

    def test_bounded_session_eviction(self):
        agent = Agent(self.path, Config(max_sessions=2))
        try:
            for session in ("a", "b", "c"):
                agent.reset(session, {})
            self.assertEqual(list(agent.sessions), ["b", "c"])
        finally:
            agent.close()

    def test_ranking_failure_returns_sparse_legal_response(self):
        self.agent.reset("a", {})
        with patch("mercury.agent.rank_candidates", side_effect=RuntimeError("bad ranker")):
            result = self.agent.respond("a", "Blue cotton shirt", 1, 10)
        self.assert_legal_response(result)
        self.assertIn("ranking", self.agent.last_diagnostics["fallbacks"])

    def test_absent_model_assets_fall_back_without_network_or_key(self):
        agent = Agent(self.path, Config(dense=True, neural_rerank=True, contrast=True, artifact_dir=self.temp.name))
        try:
            agent.reset("a", {})
            result = agent.respond("a", "Blue cotton shirt", 1, 10)
            self.assert_legal_response(result, agent)
            self.assertIsNone(agent.dense)
            self.assertIsNone(agent.reranker)
            self.assertIsNone(agent.contrast)
            self.assertEqual(set(agent.startup_fallbacks), {"dense", "neural_rerank", "contrast"})
            self.assertEqual(set(agent.last_diagnostics["fallbacks"]), set(agent.startup_fallbacks))
        finally:
            agent.close()

    def test_malformed_present_assets_fall_back_without_network_or_model_loading(self):
        root = Path(self.temp.name)
        for component in ("dense", "contrast"):
            (root / component).mkdir()
            (root / component / "manifest.json").write_text("[]")
        for relative in MODELS["reranker"]["required"]:
            path = root / "models" / "reranker" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
        with patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden")), \
                patch("socket.create_connection", side_effect=AssertionError("Network forbidden")), \
                patch("mercury.neural.load_encoder", side_effect=AssertionError("Model loading forbidden")), \
                patch("mercury.neural._model_options", side_effect=AssertionError("Model loading forbidden")):
            agent = Agent(self.path, Config(dense=True, neural_rerank=True, contrast=True, artifact_dir=str(root)))
            try:
                agent.reset("a", {})
                self.assert_legal_response(agent.respond("a", "Blue cotton shirt", 1, 10), agent)
                self.assertEqual(set(agent.startup_fallbacks), {"dense", "neural_rerank", "contrast"})
                self.assertEqual(set(agent.last_diagnostics["fallbacks"]), set(agent.startup_fallbacks))
                self.assertNotIn("dense", agent.last_diagnostics["routes"])
                self.assertTrue(agent.startup_fallbacks["dense"].startswith("AttributeError:"))
                self.assertTrue(agent.startup_fallbacks["contrast"].startswith("AttributeError:"))
                self.assertTrue(agent.startup_fallbacks["neural_rerank"].startswith("ValueError:"))
            finally:
                agent.close()

    def test_dense_search_failures_return_legal_sparse_routes(self):
        for error in (TimeoutError("local dense timeout"), ValueError("invalid dense vector")):
            with self.subTest(error=type(error).__name__):
                self.agent.reset("a", {})
                self.agent.dense = SimpleNamespace(prompt_tokens=0, search=Mock(side_effect=error))
                self.assert_legal_response(self.agent.respond("a", "Blue cotton shirt", 1, 10))
                self.assertIn("dense", self.agent.last_diagnostics["fallbacks"])
                self.assertIn("sparse", self.agent.last_diagnostics["routes"])
                self.assertNotIn("dense", self.agent.last_diagnostics["routes"])

    def test_malformed_index_payloads_fall_back_after_manifest_checks(self):
        root = Path(self.temp.name)
        dense = root / "dense"
        dense.mkdir()
        (dense / "ids.json").write_text(json.dumps([product.parent_asin for product in self.agent.catalog.products]))
        (dense / "vectors.npy").write_text("not a numpy array")
        (dense / "manifest.json").write_text(json.dumps({
            "catalog_sha256": self.agent.catalog.sha256, "model_revision": MODELS["embedding"]["revision"],
            "document_version": DOCUMENT_VERSION, "count": len(self.agent.catalog.products), "dimensions": 384,
            "sha256": {name: file_sha256(dense / name) for name in ("ids.json", "vectors.npy")},
        }))
        contrast = root / "contrast"
        contrast.mkdir()
        data = contrast / "contrasts.json"
        data.write_text(json.dumps({identifier: {"differences": None} for identifier in self.agent.catalog.by_id}))
        (contrast / "manifest.json").write_text(json.dumps({
            "version": CONTRAST_VERSION, "catalog_sha256": self.agent.catalog.sha256, "sha256": file_sha256(data),
        }))
        with patch("mercury.neural.load_encoder", side_effect=AssertionError("Model loading forbidden")):
            agent = Agent(self.path, Config(dense=True, contrast=True, artifact_dir=str(root)))
        try:
            agent.reset("a", {})
            self.assert_legal_response(agent.respond("a", "Blue cotton shirt", 1, 10), agent)
            self.assertTrue(agent.startup_fallbacks["dense"].startswith("ValueError:"))
            self.assertTrue(agent.startup_fallbacks["contrast"].startswith("TypeError:"))
            self.assertEqual(set(agent.last_diagnostics["fallbacks"]), {"dense", "contrast"})
            self.assertNotIn("dense", agent.last_diagnostics["routes"])
        finally:
            agent.close()

    def test_contrast_failure_returns_legal_sparse_response(self):
        self.agent.reset("a", {})
        self.agent.contrast = SimpleNamespace(rank=Mock(side_effect=RuntimeError("bad contrast ranker")))
        self.assert_legal_response(self.agent.respond("a", "Blue cotton shirt", 1, 10))
        self.assertIn("contrast", self.agent.last_diagnostics["fallbacks"])

    def test_malformed_contrast_returns_cannot_change_catalog_ids(self):
        products = self.agent.catalog.products
        malformed = {"missing": None, "duplicates": [Candidate(products[0], 1.0)] * len(products),
                     "unknown_id": [Candidate(replace(products[0], parent_asin="invented"), 1.0)]
                                   + [Candidate(product, 1.0) for product in products[1:]],
                     "nonfinite": [Candidate(product, float("nan")) for product in products],
                     "nonnumeric": [Candidate(product, "invalid") for product in products]}
        for name, result in malformed.items():
            with self.subTest(result=name):
                self.agent.reset("a", {})
                self.agent.contrast = SimpleNamespace(rank=Mock(return_value=result))
                self.assert_legal_response(self.agent.respond("a", "Blue cotton shirt", 1, 10))
                self.assertIn("contrast", self.agent.last_diagnostics["fallbacks"])

    def test_cached_results_preserve_runtime_fallback_provenance(self):
        self.agent.reset("a", {})
        self.agent.dense = SimpleNamespace(prompt_tokens=0, search=Mock(side_effect=TimeoutError("dense timeout")))
        self.agent.contrast = SimpleNamespace(rank=Mock(side_effect=RuntimeError("contrast failure")))
        first = self.agent.respond("a", "Blue cotton shirt", 1, 10)
        original_fallbacks = list(self.agent.last_diagnostics["fallbacks"])
        self.assertEqual(set(original_fallbacks), {"dense", "contrast"})
        second = self.agent.respond("a", "Keep looking.", 2, 10)
        self.assert_legal_response(second)
        self.assertTrue(self.agent.last_diagnostics["cache_hit"])
        self.assertEqual(first["recommendations"], second["recommendations"])
        self.assertEqual(self.agent.last_diagnostics["fallbacks"], original_fallbacks)
        self.agent.dense.search.assert_called_once()
        self.agent.contrast.rank.assert_called_once()

    def material_agent(self, candidate_limit):
        path = Path(self.temp.name) / "materials.jsonl"
        rows = [{"parent_asin": "leather", "title": "Bag", "details": {"Material": "Genuine leather"}},
                {"parent_asin": "unknown", "title": "Bag"},
                {"parent_asin": "faux", "title": "Bag", "details": {"Material": "Faux leather"}}]
        path.write_text("\n".join(map(json.dumps, rows)))
        return Agent(path, Config(evidence_ranking=False, candidate_limit=candidate_limit))

    def test_exclusion_guard_precedes_candidate_limit_when_evidence_ranking_is_off(self):
        agent = self.material_agent(candidate_limit=2)
        try:
            agent.reset("a", {})
            with patch.object(agent.sparse, "search", return_value=["leather", "unknown", "faux"]), \
                    patch("mercury.agent.rank_candidates", side_effect=AssertionError("Evidence ranking disabled")):
                response = agent.respond("a", "A bag. No leather.", 1, 10)
            self.assertEqual(self.assert_legal_response(response, agent, count=2), ["unknown", "faux"])
            self.assertEqual(set(agent.last_diagnostics["retrieved_ids"]), {"leather", "unknown", "faux"})
            self.assertEqual(agent.last_diagnostics["fallbacks"], [])
        finally:
            agent.close()

    def test_exclusion_guard_survives_optional_reranking_without_hard_filtering(self):
        def promote(candidates, replace_scores):
            ranked = []
            for candidate in candidates:
                routes = dict(candidate.route_scores)
                if replace_scores:
                    routes.pop("constraint_penalty", None)
                score = 100.0 if candidate.product.parent_asin == "leather" else 1.0
                if not replace_scores:
                    score += candidate.score
                ranked.append(Candidate(candidate.product, score, routes))
            return sorted(ranked, key=lambda candidate: -candidate.score)

        for component in ("contrast", "reranker"):
            with self.subTest(component=component):
                agent = self.material_agent(candidate_limit=3)
                try:
                    agent.reset("a", {})
                    if component == "contrast":
                        model = SimpleNamespace(rank=Mock(side_effect=lambda candidates, *_: promote(candidates, False)))
                    else:
                        model = SimpleNamespace(prompt_tokens=0,
                                                rank=Mock(side_effect=lambda query, candidates, *_: promote(candidates, True)))
                    setattr(agent, component, model)
                    with patch.object(agent.sparse, "search", return_value=["leather", "unknown", "faux"]):
                        response = agent.respond("a", "A bag. No leather.", 1, 10)
                    ids = self.assert_legal_response(response, agent, count=3)
                    self.assertEqual(set(ids[:2]), {"unknown", "faux"})
                    self.assertEqual(ids[-1], "leather")
                    self.assertEqual(agent.last_diagnostics["fallbacks"], [])
                    model.rank.assert_called_once()
                finally:
                    agent.close()

    def test_constraint_failures_and_malformed_results_keep_legal_fallbacks(self):
        failures = {"error": Mock(side_effect=RuntimeError("constraint failure")),
                    "malformed": Mock(return_value=None)}
        for name, guard in failures.items():
            with self.subTest(result=name):
                self.agent.reset("a", {})
                with patch("mercury.agent.rank_constraints", guard):
                    response = self.agent.respond("a", "Blue cotton shirt", 1, 10)
                self.assert_legal_response(response)
                self.assertEqual(self.agent.last_diagnostics["fallbacks"], ["constraints"])
                self.assertEqual(guard.call_count, 2)

    def test_inference_timeout_and_malformed_ranks_fall_back(self):
        class BrokenRanker:
            prompt_tokens = 0

            def rank(self, *args):
                raise TimeoutError("local inference timeout")

        self.agent.reranker = BrokenRanker()
        self.agent.reset("a", {})
        result = self.agent.respond("a", "Blue cotton shirt", 1, 10)
        self.assert_legal_response(result)
        self.assertIn("neural_rerank", self.agent.last_diagnostics["fallbacks"])

    def test_malformed_ranker_return_cannot_break_legal_fallback(self):
        class BrokenRanker:
            prompt_tokens = 0

            def rank(self, *args):
                return None

        self.agent.reranker = BrokenRanker()
        self.agent.reset("a", {})
        response = self.agent.respond("a", "cotton shirt", 1, 10)
        self.assert_legal_response(response)
        self.assertIn("neural_rerank", self.agent.last_diagnostics["fallbacks"])

    def test_final_turn_and_user_top_k_limits(self):
        agent = Agent(self.path, Config(slate_size=1))
        try:
            agent.reset("a", {})
            self.assertEqual(len(agent.respond("a", "shirt", 1, 10)["recommendations"]), 1)
            self.assertEqual(len(agent.respond("a", "shirt", 10, 10)["recommendations"]), 10)
            self.assertIsNone(agent.respond("a", "shirt", 10, 10)["ask_attribute"])
            self.assertEqual(len(agent.respond("a", "shirt", 10, 3)["recommendations"]), 3)
        finally:
            agent.close()

    def test_previous_recommendations_are_not_eliminated_on_nonconversion(self):
        self.agent.reset("a", {})
        first = self.agent.respond("a", "Blue cotton shirt", 1, 10)
        second = self.agent.respond("a", "Keep looking.", 2, 10)
        self.assertEqual(first["recommendations"], second["recommendations"])
        self.assertTrue(self.agent.last_diagnostics["cache_hit"])

    def choice_agent(self, mode, candidate_limit=120, extra_rows=()):
        path = Path(self.temp.name) / f"choices-{mode}.jsonl"
        rows = [
            {"parent_asin": "cotton", "title": "Shirt", "features": ["Cotton fabric.", "Linen-free."]},
            {"parent_asin": "linen", "title": "Shirt", "features": ["Linen fabric.", "Cotton-free."]},
            {"parent_asin": "neither", "title": "Shirt", "features": ["Cotton-free.", "Linen-free."]},
            {"parent_asin": "unknown", "title": "Shirt"},
        ]
        rows.extend(extra_rows)
        path.write_text("\n".join(map(json.dumps, rows)))
        return Agent(path, Config(evidence_ranking=False, alternatives_mode=mode,
                                  candidate_limit=candidate_limit))

    def test_alternative_modes_reach_session_state_and_expose_actual_penalties(self):
        for mode in ("off", "parse", "grouped"):
            with self.subTest(mode=mode):
                agent = self.choice_agent(mode)
                try:
                    agent.reset("choice", {})
                    self.assertEqual(agent.sessions["choice"].alternatives_mode, mode)
                    response = agent.respond("choice", "The shirt must be cotton or linen.", 1, 10)
                    ids = self.assert_legal_response(response, agent, count=4)
                    diagnostics = agent.last_diagnostics
                    penalties = diagnostics["constraint_penalties"]
                    self.assertEqual(set(penalties), set(diagnostics["ranked_ids"]))
                    self.assertEqual(penalties["unknown"], 0.0)
                    self.assertGreater(penalties["neither"], 0.0)
                    self.assertEqual(diagnostics["unsupported_alternatives"], [])
                    materials = [item for item in diagnostics["preferences"] if item["attribute"] == "material"]
                    if mode == "grouped":
                        self.assertEqual(penalties["cotton"], 0.0)
                        self.assertEqual(penalties["linen"], 0.0)
                        self.assertLess(ids.index("cotton"), ids.index("unknown"))
                        self.assertEqual(len({item["alternative_group"] for item in materials}), 1)
                    else:
                        self.assertGreater(penalties["cotton"], 0.0)
                        self.assertGreater(penalties["linen"], 0.0)
                        self.assertEqual(ids[0], "unknown")
                        self.assertTrue(all("alternative_group" not in item for item in materials))
                    self.assertEqual(diagnostics["fallbacks"], [])
                finally:
                    agent.close()

    def test_grouped_selection_changes_mini_catalog_ranking_and_then_caches(self):
        agent = self.choice_agent("grouped")
        try:
            agent.reset("choice", {})
            agent.respond("choice", "The shirt must be cotton or linen.", 1, 10)
            before_revision = agent.last_diagnostics["revision"]
            response = agent.respond("choice", "Actually, cotton only.", 2, 10)
            ids = self.assert_legal_response(response, agent, count=4)
            diagnostics = agent.last_diagnostics
            self.assertFalse(diagnostics["cache_hit"])
            self.assertGreater(diagnostics["revision"], before_revision)
            self.assertLess(ids.index("cotton"), ids.index("linen"))
            self.assertEqual(diagnostics["constraint_penalties"]["cotton"], 0.0)
            self.assertGreater(diagnostics["constraint_penalties"]["linen"], 0.0)
            positives = {(item["attribute"], item["value"]) for item in diagnostics["preferences"]
                         if item["polarity"] == 1}
            self.assertIn(("category", "shirts"), positives)
            self.assertIn(("material", "cotton"), positives)
            self.assertNotIn(("material", "linen"), positives)
            penalties = dict(diagnostics["constraint_penalties"])
            cached = agent.respond("choice", "Keep looking.", 3, 10)
            self.assertTrue(agent.last_diagnostics["cache_hit"])
            self.assertEqual(cached["recommendations"], response["recommendations"])
            self.assertEqual(agent.last_diagnostics["constraint_penalties"], penalties)
        finally:
            agent.close()

    def test_same_message_choice_corrections_reach_the_real_constraint_guard(self):
        for correction, rejected in (
            ("Cotton only.", "linen"), ("Only linen.", "cotton"),
            ("It must be cotton and linen.", "linen"),
        ):
            with self.subTest(correction=correction):
                agent = self.choice_agent("grouped")
                try:
                    agent.reset("choice", {})
                    response = agent.respond("choice", "The shirt must be cotton or linen. " + correction, 1, 10)
                    self.assert_legal_response(response, agent, count=4)
                    self.assertGreater(agent.last_diagnostics["constraint_penalties"][rejected], 0.0)
                    self.assertTrue(all("alternative_group" not in item
                                        for item in agent.last_diagnostics["preferences"]))
                    self.assertEqual(agent.last_diagnostics["fallbacks"], [])
                finally:
                    agent.close()

    def test_prefix_only_selection_invalidates_cached_ranking(self):
        for correction, selected, rejected in (
            ("Only cotton.", "cotton", "linen"),
            ("Only linen.", "linen", "cotton"),
            ("It must be only linen.", "linen", "cotton"),
        ):
            with self.subTest(correction=correction):
                agent = self.choice_agent("grouped")
                try:
                    agent.reset("choice", {})
                    agent.respond("choice", "A shirt. Must be cotton or linen.", 1, 10)
                    revision = agent.last_diagnostics["revision"]
                    self.assertEqual(agent.last_diagnostics["constraint_penalties"][rejected], 0.0)
                    response = agent.respond("choice", correction, 2, 10)
                    ids = self.assert_legal_response(response, agent, count=4)
                    self.assertFalse(agent.last_diagnostics["cache_hit"])
                    self.assertEqual(agent.last_diagnostics["revision"], revision + 1)
                    self.assertGreater(agent.last_diagnostics["constraint_penalties"][rejected], 0.0)
                    self.assertEqual(agent.last_diagnostics["constraint_penalties"][selected], 0.0)
                    self.assertLess(ids.index(selected), ids.index(rejected))
                    cached = agent.respond("choice", "Keep looking.", 3, 10)
                    self.assertTrue(agent.last_diagnostics["cache_hit"])
                    self.assertEqual(cached["recommendations"], response["recommendations"])
                finally:
                    agent.close()

    def test_rejected_overlap_still_invalidates_cache_for_independent_exclusion(self):
        agent = self.choice_agent("grouped", extra_rows=[{"parent_asin": "leather", "title": "Leather shirt"}])
        try:
            agent.reset("choice", {})
            agent.respond("choice", "The shirt must be cotton or linen.", 1, 10)
            query, revision = agent.last_diagnostics["query"], agent.last_diagnostics["revision"]
            self.assertEqual(agent.last_diagnostics["constraint_penalties"]["leather"], 0.0)
            group = {p.alternative_group for p in agent.sessions["choice"].active_preferences()
                     if p.attribute == "material"}
            response = agent.respond("choice", "Linen or wool and no leather.", 2, 10)
            self.assert_legal_response(response, agent, count=5)
            self.assertFalse(agent.last_diagnostics["cache_hit"])
            self.assertEqual(agent.last_diagnostics["query"], query)
            self.assertEqual(agent.last_diagnostics["revision"], revision + 1)
            self.assertGreater(agent.last_diagnostics["constraint_penalties"]["leather"], 0.0)
            materials = [p for p in agent.sessions["choice"].active_preferences() if p.attribute == "material"]
            self.assertEqual({(p.value, p.polarity) for p in materials},
                             {("cotton", 1), ("linen", 1), ("leather", -1)})
            self.assertEqual({p.alternative_group for p in materials if p.polarity == 1}, group)
            self.assertEqual(agent.last_diagnostics["unsupported_alternatives"], [
                {"attribute": "material", "reason": "overlapping alternatives require an explicit replacement"},
            ])
        finally:
            agent.close()

    def test_neutral_color_beside_material_list_removes_the_color_penalty(self):
        for mode in ("parse", "grouped"):
            with self.subTest(mode=mode):
                agent = self.choice_agent(mode, extra_rows=[{
                    "parent_asin": "red", "title": "Red cotton shirt",
                    "features": ["Black-free.", "Brown-free."],
                }])
                try:
                    agent.reset("choice", {})
                    agent.respond("choice", "The shirt must be black or brown.", 1, 10)
                    self.assertGreater(agent.last_diagnostics["constraint_penalties"]["red"], 0.0)
                    revision = agent.last_diagnostics["revision"]
                    agent.respond("choice", "Any color works and cotton or linen is fine.", 2, 10)
                    self.assertFalse(agent.last_diagnostics["cache_hit"])
                    self.assertEqual(agent.last_diagnostics["revision"], revision + 1)
                    self.assertEqual(agent.last_diagnostics["constraint_penalties"]["red"], 0.0)
                    self.assertEqual(agent.last_diagnostics["fallbacks"], [])
                finally:
                    agent.close()

    def test_additive_feature_retains_weather_group_penalty(self):
        path = Path(self.temp.name) / "weather.jsonl"
        rows = [
            {"parent_asin": "protected", "title": "Waterproof jacket", "features": ["Pockets."]},
            {"parent_asin": "exposed", "title": "Jacket",
             "features": ["Not waterproof.", "Not insulated.", "Pockets."]},
        ]
        path.write_text("\n".join(map(json.dumps, rows)))
        agent = Agent(path, Config(evidence_ranking=False, alternatives_mode="grouped"))
        try:
            agent.reset("weather", {})
            agent.respond("weather", "I need a waterproof or insulated jacket.", 1, 10)
            self.assertGreater(agent.last_diagnostics["constraint_penalties"]["exposed"], 0.0)
            revision = agent.last_diagnostics["revision"]
            response = agent.respond("weather", "Actually, I also need pockets.", 2, 10)
            self.assertEqual(self.assert_legal_response(response, agent, count=2), ["protected", "exposed"])
            self.assertFalse(agent.last_diagnostics["cache_hit"])
            self.assertEqual(agent.last_diagnostics["revision"], revision + 1)
            self.assertGreater(agent.last_diagnostics["constraint_penalties"]["exposed"], 0.0)
            self.assertEqual(agent.last_diagnostics["constraint_penalties"]["protected"], 0.0)
            self.assertEqual(agent.last_diagnostics["fallbacks"], [])
        finally:
            agent.close()

    def test_rejecting_one_choice_retains_group_force_and_independent_exclusion(self):
        agent = self.choice_agent("grouped")
        try:
            agent.reset("choice", {})
            agent.respond("choice", "The shirt must be cotton or linen.", 1, 10)
            agent.respond("choice", "Actually, no linen.", 2, 10)
            material = [item for item in agent.last_diagnostics["preferences"] if item["attribute"] == "material"]
            self.assertEqual({(item["value"], item["polarity"]) for item in material},
                             {("cotton", 1), ("linen", -1)})
            self.assertTrue(next(item for item in material if item["value"] == "cotton")["alternative_group"])
            self.assertNotIn("alternative_group", next(item for item in material if item["value"] == "linen"))
            self.assertEqual(agent.last_diagnostics["constraint_penalties"]["cotton"], 0.0)
            self.assertGreater(agent.last_diagnostics["constraint_penalties"]["neither"], 0.0)
            self.assertGreater(agent.last_diagnostics["constraint_penalties"]["linen"], 0.0)
        finally:
            agent.close()

    def test_group_change_invalidates_cache_even_when_query_words_are_unchanged(self):
        agent = self.choice_agent("grouped")
        try:
            agent.reset("choice", {})
            agent.respond("choice", "I need a shirt. It must be cotton or linen.", 1, 10)
            original_query = agent.last_diagnostics["query"]
            original_revision = agent.last_diagnostics["revision"]
            response = agent.respond("choice", "It must be cotton and linen.", 2, 10)
            self.assertEqual(agent.last_diagnostics["query"], original_query)
            self.assertGreater(agent.last_diagnostics["revision"], original_revision)
            self.assertFalse(agent.last_diagnostics["cache_hit"])
            self.assertEqual(response["recommendations"][0]["parent_asin"], "unknown")
            self.assertTrue(all("alternative_group" not in item for item in agent.last_diagnostics["preferences"]))
            self.assertGreater(agent.last_diagnostics["constraint_penalties"]["cotton"], 0.0)
        finally:
            agent.close()

    def test_unsupported_alternative_diagnostics_are_bounded_and_refreshed(self):
        agent = self.choice_agent("grouped")
        try:
            agent.reset("choice", {})
            agent.respond("choice", "I need cotton or linen.", 1, 10)
            agent.respond("choice", "Cotton or wool. Blue or cotton.", 2, 10)
            unsupported = agent.last_diagnostics["unsupported_alternatives"]
            self.assertGreater(len(unsupported), 0)
            self.assertLessEqual(len(unsupported), 8)
            self.assertTrue(all(set(item) == {"attribute", "reason"} for item in unsupported))
            agent.respond("choice", "Keep looking.", 3, 10)
            self.assertEqual(agent.last_diagnostics["unsupported_alternatives"], [])
        finally:
            agent.close()

    def test_group_guard_runs_before_truncation_and_after_replaced_neural_scores(self):
        agent = self.choice_agent("grouped", candidate_limit=3)

        def replace_scores(query, candidates, *_):
            return [Candidate(item.product, 100.0 if item.product.parent_asin == "linen" else 1.0,
                              {"neural": 1.0}) for item in candidates]

        try:
            agent.reset("choice", {})
            agent.reranker = SimpleNamespace(prompt_tokens=0, rank=Mock(side_effect=replace_scores))
            with patch.object(agent.sparse, "search", return_value=["neither", "linen", "cotton", "unknown"]):
                response = agent.respond("choice", "The shirt must be cotton or linen.", 1, 10)
            self.assertNotIn("neither", agent.last_diagnostics["ranked_ids"])
            self.assertEqual(self.assert_legal_response(response, agent, count=3)[0], "linen")
            self.assertEqual(set(agent.last_diagnostics["constraint_penalties"].values()), {0.0})
            with patch.object(agent.sparse, "search", return_value=["linen", "cotton", "unknown", "neither"]):
                response = agent.respond("choice", "Actually, cotton only.", 2, 10)
            ids = self.assert_legal_response(response, agent, count=3)
            self.assertEqual(ids[-1], "linen")
            self.assertGreater(agent.last_diagnostics["constraint_penalties"]["linen"], 0.0)
            self.assertEqual(agent.last_diagnostics["fallbacks"], [])
        finally:
            agent.close()

    def test_off_mode_and_default_preserve_characterized_responses(self):
        explicit = Agent(self.path, Config(alternatives_mode="off"))
        try:
            self.agent.reset("a", {})
            explicit.reset("a", {})
            for turn, message in enumerate(("Either blue or red shirt works.", "A cotton shirt.",
                                            "No wool.", "Keep looking."), 1):
                default_response = self.agent.respond("a", message, turn, 10)
                off_response = explicit.respond("a", message, turn, 10)
                self.assertEqual(default_response, off_response)
                for field in ("query", "revision", "cache_hit", "preferences", "routes", "ranked_ids"):
                    self.assertEqual(self.agent.last_diagnostics[field], explicit.last_diagnostics[field])
                self.assertTrue(all("alternative_group" not in item for item in explicit.last_diagnostics["preferences"]))
        finally:
            explicit.close()

    def test_failed_or_malformed_neural_ranks_preserve_grouped_sparse_results(self):
        agent = self.choice_agent("grouped")
        try:
            agent.reset("choice", {})
            expected = agent.respond("choice", "The shirt must be cotton or linen.", 1, 10)
            expected_penalties = dict(agent.last_diagnostics["constraint_penalties"])
            malformed = [Candidate(agent.catalog.products[0], 100.0)] * len(agent.catalog.products)
            for failure in (Mock(side_effect=TimeoutError("inference timeout")),
                            Mock(return_value=None), Mock(return_value=malformed)):
                with self.subTest(failure=failure):
                    agent.reset("choice", {})
                    agent.reranker = SimpleNamespace(prompt_tokens=0, rank=failure)
                    response = agent.respond("choice", "The shirt must be cotton or linen.", 1, 10)
                    self.assertEqual(response, expected)
                    self.assertEqual(agent.last_diagnostics["constraint_penalties"], expected_penalties)
                    self.assertEqual(agent.last_diagnostics["fallbacks"], ["neural_rerank"])
                    failure.assert_called_once()
        finally:
            agent.close()

    def test_cycle2_configs_only_change_alternatives_mode(self):
        root = Path(__file__).resolve().parents[1] / "configs"
        selected = Config.load(root / "selected.json")
        for name, mode in (("frozen", "off"), ("parse", "parse"), ("grouped", "grouped")):
            with self.subTest(name=name):
                config = Config.load(root / f"cycle2_{name}.json")
                self.assertEqual(config, replace(selected, alternatives_mode=mode))


if __name__ == "__main__":
    unittest.main()
