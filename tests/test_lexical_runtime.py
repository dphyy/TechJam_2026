import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from mercury.lexical.agent import Agent
from mercury.lexical.config import FULL_WIDTH_CONFIG
from mercury.lexical.vector_index import (
    CatalogVectorIndex,
    catalog_row_identity,
    catalog_sha256,
    file_sha256,
    np,
)


class AgentRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.catalog = Path(self.temp.name) / "catalog.jsonl"
        self.catalog.write_text("\n".join(json.dumps({
            "parent_asin": str(index), "title": "Cotton shirt",
            "categories": ["Shirts"], "features": ["cotton"], "price": 25,
        }) for index in range(12)))

    def agent(self, **kwargs):
        agent = Agent(self.catalog, config=FULL_WIDTH_CONFIG, **kwargs)
        self.addCleanup(agent.close)
        return agent

    def test_identical_retry_is_detached_and_does_not_repeat_work(self):
        agent = self.agent()
        agent.reset("session", {})
        message = "I'm looking for Shirts, but I'm still exploring."
        first = agent.respond("session", message, 1, 10)
        expected = deepcopy(first)
        first["recommendations"][0]["parent_asin"] = "changed"
        state = agent._sessions["session"]
        questions = list(state.asked_attributes)
        with patch.object(agent.search, "search_with_context", side_effect=AssertionError("retried search")):
            for _ in range(5):
                replay = agent.respond("session", message, 1, 10)
                self.assertEqual(replay, expected)
                replay["recommendations"].clear()
        self.assertEqual(state.asked_attributes, questions)
        self.assertEqual(state.messages, [message.rstrip(".")])

    def test_conflicting_stale_and_invalid_turns_do_not_mutate_state(self):
        agent = self.agent()
        agent.reset("session", {})
        agent.respond("session", "A cotton shirt", 1, 10)
        agent.respond("session", "For that, what matters is: blue.", 2, 10)
        state = agent._sessions["session"]
        expected = deepcopy((state.messages, state.evidence, state.asked_attributes, state.last_turn))
        requests = [
            ("A cotton shirt", 1, 10), ("changed", 2, 10),
            ("For that, what matters is: blue.", 2, 5),
            ("new", 0, 10), ("new", 11, 10), ("new", True, 10),
            (None, 3, 10), ("new", 3, 0),
        ]
        for message, turn, top_k in requests:
            with self.subTest(message=message, turn=turn, top_k=top_k):
                with self.assertRaises(ValueError):
                    agent.respond("session", message, turn, top_k)
                self.assertEqual(
                    (state.messages, state.evidence, state.asked_attributes, state.last_turn), expected,
                )

    def test_response_retention_follows_reset_eviction_and_close(self):
        agent = self.agent(max_sessions=1)
        for session in ("first", "second"):
            agent.reset(session, {})
            agent.respond(session, "A cotton shirt", 1, 10)
        self.assertEqual(set(agent._responses), {"second"})
        agent.reset("second", {})
        self.assertEqual(agent._responses, {})
        agent.respond("second", "A cotton shirt", 1, 10)
        agent.close()
        self.assertEqual(agent._responses, {})

    def test_invalid_reset_preserves_the_live_session(self):
        agent = self.agent()
        agent.reset("session", {})
        state = agent._sessions["session"]
        with self.assertRaises(ValueError):
            agent.reset("session", None)
        self.assertIs(agent._sessions["session"], state)

    def test_forgetting_removes_active_shared_and_isolated_profile_influence(self):
        for shared in (False, True):
            with self.subTest(shared=shared):
                agent = self.agent(share_profile_memory=shared)
                profiles = []
                for session in ("first", "second"):
                    agent.reset(session, {"profile_id": "profile", "preference_tags": ["cotton"]})
                    agent.respond(session, "I'm looking for Shirts, but I'm still exploring.", 1, 10)
                    state = agent._sessions[session]
                    profiles.append(state.long_term_profile)
                    self.assertTrue(any(product["_profile_bonus"] > 0 for product in
                                        agent.search.search_with_context(state).candidates))
                agent.forget_profile("profile")
                self.assertIsNone(agent.export_profile("profile"))
                self.assertFalse(agent._responses)
                for profile in profiles:
                    self.assertFalse(profile.learned)
                    self.assertFalse(profile._observations)
                for state in agent._sessions.values():
                    self.assertIsNone(state.long_term_profile)
                    self.assertEqual(state.user_profile, {})
                    self.assertTrue(all(product["_profile_bonus"] == 0 for product in
                                        agent.search.search_with_context(state).candidates))

    def test_reset_detaches_the_request_profile(self):
        agent = self.agent()
        profile = {"preference_tags": ["cotton"]}
        agent.reset("session", profile)
        profile["preference_tags"].append("leather")
        self.assertEqual(agent._sessions["session"].user_profile["preference_tags"], ["cotton"])


@unittest.skipIf(np is None, "optional array library unavailable")
class VectorRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.catalog = self.root / "catalog.jsonl"
        self.vectors = self.root / "vectors.npy"
        self.metadata = self.root / "vectors.json"
        self.write_catalog(2)
        np.save(self.vectors, np.eye(2, dtype=np.float32))
        self.write_metadata()

    def write_catalog(self, count):
        self.catalog.write_text("\n".join(json.dumps({
            "parent_asin": str(index), "title": "Cotton shirt",
        }) for index in range(count)))

    def write_metadata(self, **overrides):
        count, identities = catalog_row_identity(self.catalog)
        metadata = {
            "model": "fixture", "dimensions": 2, "row_count": count,
            "catalog_sha256": catalog_sha256(self.catalog), "normalized": True,
            "vectors_sha256": file_sha256(self.vectors), "product_ids_sha256": identities,
        }
        metadata.update(overrides)
        self.metadata.write_text(json.dumps(metadata))

    @staticmethod
    def response(vector=(1.0, 0.0), index=0):
        return SimpleNamespace(
            data=[SimpleNamespace(index=index, embedding=list(vector))],
            usage=SimpleNamespace(prompt_tokens=7),
        )

    def index(self, client=None, **kwargs):
        result = CatalogVectorIndex(
            self.catalog, vectors_path=self.vectors, metadata_path=self.metadata,
            client=client or SimpleNamespace(embeddings=SimpleNamespace(create=Mock(
                return_value=self.response(),
            ))), **kwargs,
        )
        self.addCleanup(result.close)
        return result

    def test_transient_request_failure_recovers_on_later_call(self):
        request = Mock(side_effect=[TimeoutError("temporary"), self.response()])
        index = self.index(SimpleNamespace(embeddings=SimpleNamespace(create=request)))
        self.assertEqual(index.search("cotton").rows, [])
        self.assertTrue(index.enabled)
        recovered = index.search("cotton")
        self.assertEqual(recovered.rows[0], (1, 1.0))
        self.assertEqual(recovered.prompt_tokens, 7)
        self.assertEqual(index.search("cotton").prompt_tokens, 0)
        self.assertEqual(request.call_count, 2)

    def test_unavailable_client_can_become_available(self):
        client = SimpleNamespace(embeddings=SimpleNamespace(create=Mock(return_value=self.response())))
        index = self.index()
        index.client = None
        with patch("mercury.lexical.vector_index.load_openai_api_key", side_effect=[False, True]), \
                patch("mercury.lexical.vector_index.create_openai_client", return_value=client):
            self.assertEqual(index.search("cotton").rows, [])
            self.assertEqual(index.search("cotton").rows[0], (1, 1.0))

    def test_lru_eviction_and_close_release_query_data(self):
        index = self.index(cache_capacity=2)
        for query in ("first", "second", "first", "third"):
            index.search(query)
        self.assertEqual(list(index._cache), ["first", "third"])
        self.assertEqual(index.client.embeddings.create.call_count, 3)
        index.search("second")
        self.assertEqual(index.client.embeddings.create.call_count, 4)
        index.close()
        self.assertEqual(index._cache, {})
        self.assertFalse(index.enabled)
        self.assertEqual(index.search("first").rows, [])

    def test_malformed_embedding_does_not_poison_future_calls(self):
        for response in (self.response(index=1), self.response(vector=(float("nan"), 0)),
                         self.response(vector=(0, 0)), self.response(vector=(1,))):
            with self.subTest(response=response):
                request = Mock(side_effect=[response, self.response()])
                index = self.index(SimpleNamespace(embeddings=SimpleNamespace(create=request)))
                self.assertEqual(index.search("cotton").rows, [])
                self.assertEqual(index._cache, {})
                self.assertEqual(index.search("cotton").rows[0], (1, 1.0))

    def test_partial_batch_response_does_not_publish_partial_cache(self):
        response = SimpleNamespace(data=[
            SimpleNamespace(index=0, embedding=[1, 0]),
            SimpleNamespace(index=1, embedding=[0]),
        ], usage=SimpleNamespace(prompt_tokens=9))
        index = self.index(SimpleNamespace(embeddings=SimpleNamespace(create=Mock(return_value=response))))
        self.assertEqual(index._embed_missing(["first", "second"]), 9)
        self.assertEqual(index._cache, {})

    def test_changed_vector_bytes_are_rejected_without_updating_metadata(self):
        np.save(self.vectors, np.asarray([[0, 1], [1, 0]], dtype=np.float32))
        index = self.index()
        self.assertFalse(index.enabled)
        self.assertEqual(index.search("cotton").rows, [])
        index.client.embeddings.create.assert_not_called()

    def test_catalog_order_digest_and_actual_row_count_are_checked(self):
        for overrides in ({"product_ids_sha256": "0" * 64}, {"row_count": 1}):
            with self.subTest(overrides=overrides):
                self.write_metadata(**overrides)
                self.assertFalse(self.index().enabled)
        rows = self.catalog.read_text().splitlines()
        old_metadata = json.loads(self.metadata.read_text())
        self.catalog.write_text("\n".join(reversed(rows)))
        self.write_metadata(product_ids_sha256=old_metadata["product_ids_sha256"])
        self.assertFalse(self.index().enabled)

    def test_every_vector_row_is_checked_even_with_matching_digests(self):
        count = 1025
        self.write_catalog(count)
        omitted = next(iter(set(range(count)) - set(np.linspace(0, count - 1, 1024, dtype=np.int64))))
        for invalid in ((float("nan"), 0), (float("inf"), 0), (0, 0), (2, 0)):
            with self.subTest(invalid=invalid):
                matrix = np.tile(np.asarray([[1, 0]], dtype=np.float32), (count, 1))
                matrix[omitted] = invalid
                np.save(self.vectors, matrix)
                self.write_metadata()
                self.assertFalse(self.index().enabled)

    def test_legacy_or_malformed_metadata_falls_back_before_client_use(self):
        for changes in ({"normalized": "true"}, {"dimensions": True}, {"row_count": "2"},
                        {"model": ""}, {"vectors_sha256": None}):
            with self.subTest(changes=changes):
                self.write_metadata(**changes)
                self.assertFalse(self.index().enabled)
        self.write_metadata()
        metadata = json.loads(self.metadata.read_text())
        del metadata["vectors_sha256"]
        self.metadata.write_text(json.dumps(metadata))
        self.assertFalse(self.index().enabled)


if __name__ == "__main__":
    unittest.main()
