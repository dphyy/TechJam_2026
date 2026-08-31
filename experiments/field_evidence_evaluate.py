"""Isolated, opt-in phrase evidence ablations on the unchanged suite evaluator."""
from __future__ import annotations

import argparse
import datetime
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from experiments import evaluate_suite as suite
from experiments.run import source_hashes
from mercury.agent import Agent
from mercury.config import Config
from mercury.field_evidence import Arm, FieldEvidenceConfig, FieldEvidenceResult, field_phrase_evidence
from mercury.model_assets import file_sha256
from mercury.planning import RetrievalPlan
from mercury.state import SessionState
from mercury.types import Candidate, Preference


ARMS = ("off", "scoring_only", "admission_only", "admission_and_scoring")
EVIDENCE_CONFIG = FieldEvidenceConfig(score_cap=0.08)
ADJUSTMENT_KEY = "field_phrase_evidence"


def validate_control(config: Config) -> None:
    required = {"neural_rerank": True, "candidate_limit": 120, "rerank_limit": 30,
                "rerank_admission": "prefix", "slate_policy": "fixed", "slate_size": 10,
                "constraint_check_stage": "both", "turn_budget_seconds": 0.0}
    disabled = ("routed_retrieval", "retrieval_sufficiency_gate", "compute_cascade",
                "over_general_cutoff", "adaptive_rerank_depth", "progressive_frontier_rerank",
                "page_local_rerank")
    if any(getattr(config, key) != value for key, value in required.items()) \
            or any(getattr(config, key) for key in disabled):
        raise ValueError("This experiment requires the fixed 120-candidate, 30-prefix, full-width-10 control")


def apply_score_deltas(candidates: list[Candidate], evidence: FieldEvidenceResult) -> list[Candidate]:
    """Replace this arm's own bounded adjustment; never change pool membership."""
    ranked = []
    for candidate in candidates:
        identifier = candidate.product.parent_asin
        delta = evidence.score_deltas.get(identifier, 0.0)
        if (type(delta) not in (int, float) or not math.isfinite(delta)
                or not 0 <= delta <= EVIDENCE_CONFIG.score_cap
                or (delta and identifier not in evidence.witnesses)):
            raise ValueError("Phrase adjustment requires finite bounded raw evidence")
        parts = dict(candidate.route_scores)
        previous = parts.pop(ADJUSTMENT_KEY, 0.0)
        if type(previous) not in (int, float) or not math.isfinite(previous):
            raise ValueError("Invalid existing phrase adjustment")
        if delta:
            parts[ADJUSTMENT_KEY] = delta
        ranked.append(Candidate(candidate.product, candidate.score - previous + delta, parts))
    return sorted(ranked, key=lambda item: -item.score)


class PhraseRanker:
    """Delegate model identity and ranking, then attach an optional score signal."""

    def __init__(self, inner: object, owner: FieldEvidenceAgent) -> None:
        self.inner = inner
        self.owner = owner

    def __getattr__(self, name: str):
        return getattr(self.inner, name)

    def rank(self, query: str, candidates: list[Candidate], limit: int, weight: float,
             *args, **kwargs) -> list[Candidate]:
        identifiers = [item.product.parent_asin for item in candidates]
        ranked = self.inner.rank(query, candidates, limit, weight, *args, **kwargs)
        self.owner._field_turn.update({
            "neural_input_ids": identifiers, "neural_prefix_ids": identifiers[:limit],
            "neural_output_ids_before_evidence": [item.product.parent_asin for item in ranked],
            "score_application_count": 0,
        })
        if self.owner.field_arm not in {"scoring_only", "admission_and_scoring"}:
            return ranked
        started = time.perf_counter()
        evidence = self.owner._retrieval_evidence
        if evidence is None:
            evidence = field_phrase_evidence(
                self.owner.catalog, self.owner.sparse, identifiers, self.owner._field_preferences,
                arm="scoring_only", config=EVIDENCE_CONFIG,
            )
        result = apply_score_deltas(ranked, evidence)
        present = set(identifiers)
        self.owner._field_turn.update({
            "score_application_count": 1,
            "score_deltas": {key: value for key, value in evidence.score_deltas.items() if key in present},
            "witnesses": {key: asdict(value) for key, value in evidence.witnesses.items() if key in present},
            "evidence_diagnostics": evidence.diagnostics,
            "score_seconds": time.perf_counter() - started,
        })
        return result


class FieldEvidenceAgent(Agent):
    """The ordinary turn API with one isolated retrieval/ranking experiment."""

    def __init__(self, catalog_path: str | Path, config: Config, arm: Arm) -> None:
        validate_control(config)
        if arm not in ARMS:
            raise ValueError("Unknown phrase evidence arm")
        self.field_arm = arm
        self._field_preferences: list[Preference] = []
        self._retrieval_evidence: FieldEvidenceResult | None = None
        self._field_turn: dict = {}
        self.field_traces: list[list[dict]] = []
        super().__init__(catalog_path, config)
        if self.reranker is not None:
            self.reranker = PhraseRanker(self.reranker, self)

    def reset(self, session_id: str, user_profile: dict) -> None:
        super().reset(session_id, user_profile)
        self.field_traces.append([])

    def _retrieve(self, plan: RetrievalPlan, state: SessionState, fallbacks: list[str],
                  source_alias_query: str = "", vocabulary_expansion_query: str = ""
                  ) -> tuple[list[Candidate], dict[str, list[str]], dict[str, float]]:
        candidates, routes, weights = super()._retrieve(
            plan, state, fallbacks, source_alias_query, vocabulary_expansion_query,
        )
        self._field_preferences = state.effective_preferences(
            self.config.soft_decay_turns if self.config.soft_preference_decay else 0,
        )
        self._field_turn["base_retrieved_ids"] = [item.product.parent_asin for item in candidates]
        if self.field_arm in {"admission_only", "admission_and_scoring"}:
            started = time.perf_counter()
            evidence = field_phrase_evidence(
                self.catalog, self.sparse, self._field_turn["base_retrieved_ids"],
                self._field_preferences, arm=self.field_arm, config=EVIDENCE_CONFIG,
            )
            self._retrieval_evidence = evidence
            candidates = candidates + [Candidate(self.catalog.by_id[identifier], 0.0,
                                                {"field_phrase_admission": 0.0})
                                       for identifier in evidence.admitted_ids]
            routes = {**routes, "field_phrase_admission": list(evidence.admitted_ids)}
            weights = {**weights, "field_phrase_admission": 0.0}
            self._field_turn.update({"admission_seconds": time.perf_counter() - started,
                                     "admission_diagnostics": evidence.diagnostics})
        return candidates, routes, weights

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        self._field_turn = {"score_application_count": 0}
        self._retrieval_evidence = None
        response = super().respond(session_id, user_message, turn, top_k)
        diagnostic = self.last_diagnostics
        stages = diagnostic["stage_ids"]
        admitted = diagnostic["routes"].get("field_phrase_admission", [])
        receipt = {
            **self._field_turn, "arm": self.field_arm, "score_cap": EVIDENCE_CONFIG.score_cap,
            "cache_reused": diagnostic["cache_hit"], "admitted_ids": list(admitted),
            "admitted_stage_survival": {
                stage: [identifier for identifier in admitted if identifier in values]
                for stage, values in stages.items()
            },
            "stage_ids": stages, "returned_count": len(response["recommendations"]),
            "returned_ids": [item["parent_asin"] for item in response["recommendations"]],
            "configured_candidate_limit": self.config.candidate_limit,
            "configured_neural_prefix": self.config.rerank_limit,
            "neural_available": self.reranker is not None,
            "runtime_components": diagnostic["effective_capabilities"]["components"],
        }
        self.last_diagnostics["field_evidence"] = receipt
        self.field_traces[-1].append({"turn": turn, "query": diagnostic["query"], **receipt})
        return response


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_experiment(config_path: Path, catalog: Path, dataset: Path, output: Path,
                   arms: tuple[str, ...] = ("off", "scoring_only")) -> dict:
    """Reserve immutable artifacts, then invoke the existing serial suite seam."""
    if not arms or len(set(arms)) != len(arms) or any(arm not in ARMS for arm in arms):
        raise ValueError("Choose unique registered experiment arms")
    config = Config.load(config_path)
    validate_control(config)
    registration = {
        "schema": "field-evidence-registration-v1",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "arms": list(arms), "evidence_config": asdict(EVIDENCE_CONFIG),
        "config": config.to_dict(), "config_sha256": file_sha256(config_path),
        "catalog_sha256": file_sha256(catalog), "dataset_sha256": file_sha256(dataset),
        "source_hashes": source_hashes(),
        "application_stage": "after neural score replacement, before existing final constraints",
        "admission_initial_score": 0.0,
        "control": "unchanged 120-member cap, 30-prefix neural call, fixed full-width 10 policy",
        "caveats": [
            "Admission does not bypass truncation; inspect per-stage survival before claiming recovery.",
            "Missing or failing neural assets yield no scoring-only adjustment and remain fallback runs.",
            "Cache hits retain an already adjusted ranking and apply no new delta.",
            "Scoring can alter later observed conversations; cross-arm membership equality is asserted only for identical input histories.",
            "The factory seam is process-local and serial; run one harness per process.",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    agents: dict[str, FieldEvidenceAgent] = {}

    def factory(spec: suite.SuiteSpec, catalog_path: Path):
        inner = FieldEvidenceAgent(catalog_path, config, spec.name)
        agents[spec.name] = inner
        return inner, config

    try:
        _write_json(output / "registration.json", registration)
        specs = [suite.SuiteSpec(arm, "config", config_path) for arm in arms]
        with patch.object(suite, "_agent_for", side_effect=factory):
            report = suite.evaluate_suite(specs, catalog, dataset)
        if (file_sha256(config_path) != registration["config_sha256"]
                or report["source_changed_during_run"]
                or report["source_hashes"] != registration["source_hashes"]):
            raise RuntimeError("Configuration or implementation changed during experiment")
        report["field_evidence_registration"] = registration
        suite.write_report(report, output / "suite")
        _write_json(output / "field_traces.json", {arm: agent.field_traces for arm, agent in agents.items()})
        return report
    except BaseException as error:
        try:
            _write_json(output / "error.json", {"status": "failed", "error_type": type(error).__name__,
                                                 "registration_remains_consumed": True})
        except OSError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the isolated raw phrase evidence ablations.")
    parser.add_argument("--config", type=Path, default=Path("configs/selected.json"))
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--dataset", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, action="append")
    args = parser.parse_args()
    report = run_experiment(args.config, args.catalog, args.dataset, args.output,
                            tuple(args.arm or ("off", "scoring_only")))
    print(json.dumps({run["name"]: run["metrics"]["technical_score"] for run in report["runs"]}))


if __name__ == "__main__":
    main()
