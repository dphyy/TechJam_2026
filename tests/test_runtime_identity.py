import json
import unittest
from types import SimpleNamespace

from mercury.runtime_identity import RuntimeIdentity


class RuntimeIdentityTest(unittest.TestCase):
    def test_identity_changes_only_for_runtime_dependencies(self):
        identity = RuntimeIdentity()
        component = SimpleNamespace(asset_identity="a" * 64, backend_identity="b" * 64)
        config = {"enabled": True, "asset_dir": "/private/local/assets"}
        components = {"dense": (True, component)}
        first = identity.snapshot(config, "c" * 64, components, {})
        self.assertEqual(first.identity, identity.snapshot(config, "c" * 64, components, {}).identity)
        component.backend_identity = "d" * 64
        changed = identity.snapshot(config, "c" * 64, components, {})
        self.assertNotEqual(first.identity, changed.identity)
        diagnostics = changed.diagnostics([])
        self.assertNotIn("/private/local/assets", json.dumps(diagnostics))
        self.assertEqual(diagnostics["components"]["dense"]["asset_sha256"], "a" * 64)
        self.assertEqual(len(identity._components), 1)

    def test_unavailable_mode_is_distinct_from_transient_failure(self):
        identity = RuntimeIdentity()
        unavailable = identity.snapshot({}, "c" * 64, {"dense": (True, None)},
                                        {"dense": "FileNotFoundError: /private/path"})
        self.assertEqual(unavailable.blocking_reasons(["dense", "no_matches"]), ())
        row = unavailable.diagnostics(["dense"])["components"]["dense"]
        self.assertEqual(row["reason"], "unavailable")
        self.assertFalse(row["effective"])
        loaded = identity.snapshot({}, "c" * 64, {"dense": (True, object())},
                                   {"dense": "old failure"})
        self.assertEqual(loaded.startup_unavailable, ())
        self.assertEqual(loaded.blocking_reasons(["dense"]), ("dense",))
        row = loaded.diagnostics(["dense"])["components"]["dense"]
        self.assertTrue(row["loaded"])
        self.assertFalse(row["effective"])
        self.assertEqual(row["reason"], "runtime_failure")

    def test_invalid_identity_fails_closed_without_exposing_its_value(self):
        component = SimpleNamespace(asset_identity="/private/untrusted/value")
        snapshot = RuntimeIdentity().snapshot({}, "c" * 64, {"dense": (True, component)}, {})
        self.assertFalse(snapshot.identity_valid)
        self.assertEqual(snapshot.blocking_reasons([]), ("invalid_identity",))
        self.assertNotIn("/private/", json.dumps(snapshot.diagnostics([])))

    def test_disabled_component_does_not_retain_old_startup_failure(self):
        snapshot = RuntimeIdentity().snapshot({}, "c" * 64, {"dense": (False, None)},
                                               {"dense": "old startup failure"})
        self.assertEqual(snapshot.startup_unavailable, ())
        self.assertEqual(snapshot.diagnostics([])["components"]["dense"]["reason"], "disabled")

    def test_unknown_fallback_reasons_remain_bounded(self):
        snapshot = RuntimeIdentity().snapshot({}, "c" * 64, {"dense": (False, None)}, {})
        self.assertEqual(snapshot.blocking_reasons(["/private/failure/details"]), ("runtime_failure",))
        self.assertNotIn("/private/", json.dumps(snapshot.diagnostics(["/private/failure/details"])))

    def test_capability_diagnostics_do_not_mutate_identity(self):
        component = SimpleNamespace(asset_identity="a" * 64)
        identity = RuntimeIdentity()
        snapshot = identity.snapshot({}, "c" * 64, {"neural_rerank": (True, component)}, {})
        faulted = snapshot.diagnostics(["latency_budget"])
        self.assertEqual(faulted["components"]["neural_rerank"]["reason"], "budget_deferred")
        self.assertTrue(snapshot.diagnostics([])["components"]["neural_rerank"]["effective"])
        self.assertEqual(snapshot.identity,
                         identity.snapshot({}, "c" * 64, {"neural_rerank": (True, component)}, {}).identity)


if __name__ == "__main__":
    unittest.main()
