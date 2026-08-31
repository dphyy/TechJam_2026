import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.review_prior_tuning import (
    breakdowns, build_fresh_pack, eligible, grid, paired_interval, PredictionMemo,
    reserve, selection_key, verify, write_json,
)
from mercury.config import Config


def products():
    return [{"parent_asin": f"id{band}{member}", "title": f"Family{band}item{member} shirt",
             "categories": ["Clothing"], "rating_number": count}
            for band, count in enumerate((0, 20, 200, 2000)) for member in range(10)]


class FakeRanker:
    def __init__(self):
        self.prompt_tokens = 0
        self.calls = []
        self._last_prediction_token_lengths = []

    def _predict_logits(self, query, documents):
        self.calls.append((query, documents))
        self._last_prediction_token_lengths = [len(query) + len(doc) for doc in documents]
        self.prompt_tokens += sum(self._last_prediction_token_lengths)
        return [float(sum(map(ord, query + doc))) for doc in documents]


def run(score=.8, hit=.9, contradictions=0, ratio=.5, pre=.3, post=.02):
    return {"result": {"recommended_technical_score": score, "hit_rate_at_10": hit},
            "diagnostics": {"constraint_audit": {"returned_contradictions": contradictions}},
            "config": {"review_prior_count_fraction": ratio, "review_prior_pre_weight": pre,
                       "review_prior_post_weight": post}}


class ReviewPriorTuningTest(unittest.TestCase):
    def test_fresh_pack_is_deterministic_and_excludes_whole_families(self):
        rows = products()
        rows.append({**rows[0], "parent_asin": "relative", "title": rows[0]["title"] + " blue 42"})
        first = build_fresh_pack(rows, {rows[0]["parent_asin"]}, per_band=2)
        self.assertEqual(first, build_fresh_pack(list(reversed(rows)), {rows[0]["parent_asin"]}, per_band=2))
        self.assertFalse(any(first["audit"]["overlap"].values()))
        self.assertEqual(first["audit"]["counts"], {"development": 8, "reserved": 8})
        for split, selected in first["datasets"].items():
            self.assertNotIn("relative", [row["ground_truth"]["parent_asin"] for row in selected])
            self.assertEqual(set(first["audit"]["bands"][split].values()), {2})

    def test_insufficient_families_refuse_rebalancing(self):
        with self.assertRaisesRegex(ValueError, "Insufficient fresh families"):
            build_fresh_pack(products(), set(), per_band=6)
        for count in (0, -1, True, .5):
            with self.assertRaises(ValueError):
                build_fresh_pack(products(), set(), per_band=count)

    def test_grid_covers_45_points_and_changes_only_prior_controls(self):
        base = Config.load("configs/selected.json")
        arms = grid(base)
        self.assertEqual(len(arms), 45)
        self.assertEqual(len({name for name, _ in arms}), 45)
        self.assertEqual(sum(config == base for _, config in arms), 1)
        self.assertEqual({config.review_prior_count_fraction for _, config in arms}, {0, .25, .5, .75, 1})
        allowed = {"review_prior_count_fraction", "review_prior_pre_weight", "review_prior_post_weight"}
        for _, config in arms:
            changed = {key for key, value in config.to_dict().items() if value != base.to_dict()[key]}
            self.assertLessEqual(changed, allowed)

    def test_memo_keeps_full_logical_tokens_and_deduplicates_physical_work(self):
        ranker = FakeRanker()
        memo = PredictionMemo(ranker)
        expected = [float(sum(map(ord, "q" + doc))) for doc in ("one", "one", "two")]
        self.assertEqual(memo.predict("q", ["one", "one", "two"]), expected)
        self.assertEqual(ranker.prompt_tokens, 12)
        self.assertEqual(memo.physical_tokens, 8)
        self.assertEqual(memo.physical_pairs, 2)
        self.assertEqual(memo.predict("q", ["one", "one", "two"]), expected)
        self.assertEqual(ranker.prompt_tokens, 24)
        self.assertEqual(len(ranker.calls), 1)
        memo.predict("q", ["one", "three"])
        self.assertEqual(ranker.prompt_tokens, 34)
        self.assertEqual(memo.physical_tokens, 14)
        self.assertEqual(ranker._last_prediction_token_lengths, [4, 6])
        memo.predict("different query", ["one"])
        self.assertEqual(len(ranker.calls), 3)

    def test_memo_handles_hit_eviction_and_empty_requests(self):
        ranker = FakeRanker()
        memo = PredictionMemo(ranker, capacity=1)
        memo.predict("q", ["old"])
        result = memo.predict("q", ["new", "old", "new"])
        self.assertEqual(result[0], result[2])
        self.assertEqual(memo.physical_pairs, 2)
        self.assertEqual(ranker.prompt_tokens, 16)
        self.assertEqual(memo.predict("q", []), [])
        self.assertEqual(ranker.prompt_tokens, 16)
        self.assertEqual(ranker._last_prediction_token_lengths, [])
        for capacity in (0, True, 1.5):
            with self.assertRaises(ValueError):
                PredictionMemo(ranker, capacity)

    def test_selector_uses_preservation_and_registered_tiebreaks(self):
        baseline = run()
        self.assertFalse(eligible(run(score=.95, hit=.8), baseline))
        self.assertFalse(eligible(run(contradictions=1), baseline))
        self.assertTrue(eligible(run(score=.81), baseline))
        self.assertLess(selection_key(run(score=.81)), selection_key(baseline))
        self.assertLess(selection_key(run(post=0)), selection_key(run(post=.01)))
        self.assertLess(selection_key(run(pre=.1)), selection_key(run(pre=.2)))
        self.assertLess(selection_key(run(ratio=.5)), selection_key(run(ratio=.25)))
        self.assertLess(selection_key(run(ratio=.25)), selection_key(run(ratio=.75)))

    def test_paired_interval_and_breakdowns(self):
        left = [{"sample_id": "a", "scenario_type": "buying", "hit": False,
                 "reciprocal_rank": 0, "first_hit_turn": None}]
        right = [{**left[0], "hit": True, "reciprocal_rank": 1, "first_hit_turn": 1}]
        baseline, candidate = {"result": {"sessions": left}}, {"result": {"sessions": right}}
        interval = paired_interval(baseline, candidate)
        self.assertEqual(interval, {"mean": 1, "ci95": [1, 1]})
        with self.assertRaises(ValueError):
            paired_interval(baseline, {"result": {"sessions": [{**right[0], "sample_id": "b"}]}})
        samples = [{"sample_id": "a", "ground_truth": {"parent_asin": "p"}}]
        result = breakdowns(samples, right, {"p": {"rating_number": 2000}})
        self.assertEqual(result["review_band"]["1000-5000"]["recommended_technical_score"], 1)
        self.assertEqual(result["scenario"]["buying"]["sample_count"], 1)

    def test_evidence_is_append_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            write_json(path, {"a": 1})
            with self.assertRaises(FileExistsError):
                write_json(path, {"a": 2})
            self.assertEqual(json.loads(path.read_text()), {"a": 1})

    def test_reserve_second_open_fails_before_data_or_agent_access(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            write_json(path / "finalist-freeze.json", {
                "source_hashes": {}, "manifest_sha256": "hash", "development_report_sha256": "hash"})
            write_json(path / "reserved-consumed.json", {"already_opened": True})
            with (patch("experiments.review_prior_tuning.verify", return_value={"file_sha256": {"reserved": "hash"}}),
                  patch("experiments.review_prior_tuning.source_hashes", return_value={}),
                  patch("experiments.review_prior_tuning.file_sha256", return_value="hash"),
                  patch("experiments.review_prior_tuning.load_jsonl") as load,
                  patch("mercury.agent.Agent") as agent):
                with self.assertRaises(FileExistsError):
                    reserve(path, path)
                load.assert_not_called()
                agent.assert_not_called()

    def test_verify_rejects_source_and_config_drift(self):
        from experiments.review_prior_tuning import SEED

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            manifest = {"schema": SEED, "protocol_sha256": "hash", "file_sha256": {},
                        "inventory": {}, "catalog": "catalog", "catalog_sha256": "hash",
                        "source_hashes": {"source": "hash"}, "baseline_config_sha256": "hash"}
            write_json(path / "manifest.json", manifest)
            with (patch("experiments.review_prior_tuning.file_sha256", return_value="hash"),
                  patch("experiments.review_prior_tuning.source_hashes", return_value={})):
                with self.assertRaisesRegex(ValueError, "Source changed"):
                    verify(path)
            with (patch("experiments.review_prior_tuning.file_sha256", side_effect=lambda p: "new" if p.name == "selected.json" else "hash"),
                  patch("experiments.review_prior_tuning.source_hashes", return_value=manifest["source_hashes"])):
                with self.assertRaisesRegex(ValueError, "Production baseline changed"):
                    verify(path)


if __name__ == "__main__":
    unittest.main()
