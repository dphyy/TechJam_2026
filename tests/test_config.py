import unittest

from mercury.config import Config


class ConfigTest(unittest.TestCase):
    def test_defaults_are_offline_and_conservative(self):
        config = Config()
        self.assertFalse(config.dense)
        self.assertFalse(config.neural_rerank)
        self.assertEqual(config.slate_policy, "fixed")
        self.assertEqual(config.slate_size, 10)

    def test_rejects_unknown_keys_and_unsafe_bounds(self):
        for value in ({"scenario_type": "buying"}, {"slate_size": 11},
                      {"candidate_limit": 0}, {"question_policy": "oracle"}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Config.from_dict(value)

    def test_round_trip(self):
        config = Config.from_dict({"question_policy": "rank_value", "contrast": True})
        self.assertEqual(Config.from_dict(config.to_dict()), config)

    def test_alternatives_are_explicitly_opt_in(self):
        self.assertEqual(Config().alternatives_mode, "off")
        for mode in ("off", "parse", "grouped"):
            with self.subTest(mode=mode):
                config = Config.from_dict({"alternatives_mode": mode})
                self.assertEqual(config.alternatives_mode, mode)
                self.assertEqual(Config.from_dict(config.to_dict()), config)

    def test_rejects_invalid_alternatives_mode(self):
        for mode in ("automatic", "or", "", None, False, 1):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                Config.from_dict({"alternatives_mode": mode})

    def test_grouped_alternatives_require_ledger_state(self):
        for state_mode in ("latest", "history"):
            with self.subTest(state_mode=state_mode), self.assertRaises(ValueError):
                Config(state_mode=state_mode, alternatives_mode="grouped")
            for alternatives_mode in ("off", "parse"):
                self.assertEqual(Config(state_mode=state_mode, alternatives_mode=alternatives_mode).state_mode,
                                 state_mode)
