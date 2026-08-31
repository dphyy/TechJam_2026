"""Opt-in presentation experiments on unchanged lexical retrieval and questions."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
import time
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from unittest.mock import patch

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from experiments.run import ObservedAgent, peak_rss_bytes, source_hashes
from experiments.synthesis_evaluate import EVALUATOR_SHA256
from mercury.lexical.feedback import explicit_slate_rejection, _semantic_value
from mercury.lexical.paging import ContextItem
from mercury.lexical.agent import Agent
from mercury.lexical.config import DEFAULT_AGENT_CONFIG, FULL_WIDTH_CONFIG
from mercury.lexical.dialogue import PreferenceOperation
from mercury.lexical.diagnostics import constraint_receipts, signature, stage_receipt
from mercury.lexical.product_features import component_scope, terms
from mercury.model_assets import file_sha256


POLICIES = ("existing", "tentative_top1", "explicit_rejection", "raw10")
MAX_CONTEXT = 100
MAX_REJECTED = 100
@dataclass(frozen=True)
class PresentationConfig:
    policy: str = "existing"

    def __post_init__(self) -> None:
        if self.policy not in POLICIES:
            raise ValueError("Unknown presentation policy")


PRESETS = {name: PresentationConfig(name) for name in POLICIES}


def _positive_context(state) -> frozenset[tuple[str | None, tuple[str, ...]]]:
    return frozenset((component_scope(item.text), value)
                     for item in state.evidence if item.source not in {"category", "exclusion"}
                     if (value := _semantic_value(item.text)))


def _replacement_reason(before_category: tuple[str, ...], before_values: frozenset,
                        state, turn: int) -> str | None:
    after_category = tuple(terms(state.category_text))
    if before_category and after_category and before_category != after_category:
        return "category_replaced"
    active = _positive_context(state)
    has_replacement = any(item.turn == turn and item.operation == PreferenceOperation.REPLACE
                          and item.source != "category" and _semantic_value(item.text)
                          for item in state.evidence)
    if has_replacement and before_values - active:
        return "active_preference_replaced"
    return None


class ContextObserver:
    """Observe the returned search context without changing any search result."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.context: tuple[ContextItem, ...] = ()
        self.products: tuple[dict, ...] = ()
        self.calls = 0

    def __getattr__(self, name: str):
        return getattr(self.inner, name)

    def search_with_context(self, state, limit: int = 10):
        result = self.inner.search_with_context(state, limit)
        if len(result.candidates) > MAX_CONTEXT:
            raise ValueError("Search context exceeds the bounded presentation contract")
        context = []
        seen = set()
        for product in result.candidates:
            identifier, score = product.get("parent_asin"), product.get("_rank_score")
            if (not isinstance(identifier, str) or not identifier or identifier in seen
                    or type(score) not in (int, float) or not math.isfinite(score)):
                raise ValueError("Search context must contain unique IDs and finite scores")
            seen.add(identifier)
            context.append(ContextItem(identifier, float(score), bool(product.get("_semantic_violation"))))
        if any(identifier not in seen for identifier, _ in result.recommendations):
            raise ValueError("Search recommendations must belong to their context")
        self.context = tuple(context)
        self.products = tuple(dict(product) for product in result.candidates)
        self.calls += 1
        return result


@dataclass(frozen=True)
class PresentationReceipt:
    request: tuple[int, str, int]
    shown: tuple[str, ...]
    rejected: frozenset[str]
    response: dict
    diagnostics: dict


class PresentationAgent:
    def __init__(self, catalog_path: str | Path, config: PresentationConfig = PRESETS["existing"],
                 *, max_sessions: int = 256, inner: Agent | None = None) -> None:
        self.config = config
        self.inner = inner if inner is not None else Agent(
            catalog_path, max_sessions=max_sessions,
            config=FULL_WIDTH_CONFIG if config.policy == "raw10" else replace(DEFAULT_AGENT_CONFIG, guarded_paging=False),
        )
        if config.policy == "raw10" and not self.inner.config.full_width:
            raise ValueError("Raw10 requires the underlying full-width control")
        self.observer = ContextObserver(self.inner.search)
        self.inner.search = self.observer
        self._receipts: dict[str, PresentationReceipt] = {}
        self.last_diagnostics: dict = {}
        self.presentation_identity = {
            "implementation_sha256": file_sha256(Path(__file__)),
            "config_sha256": signature(asdict(config)),
            "scope": "Presentation implementation and policy only; identity retains the underlying agent binding.",
        }
        self._closed = False

    @property
    def last_diagnostics(self) -> dict:
        return deepcopy(self._last_diagnostics)

    @last_diagnostics.setter
    def last_diagnostics(self, value: dict) -> None:
        self._last_diagnostics = deepcopy(value)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Presentation agent is closed")

    def _prune(self) -> None:
        for session_id in tuple(self._receipts):
            if session_id not in self.inner._sessions:
                del self._receipts[session_id]

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._require_open()
        self.inner.reset(session_id, user_profile)
        self._receipts.pop(session_id, None)
        self._prune()
        self.last_diagnostics = {}

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        started = time.perf_counter()
        self._require_open()
        if not isinstance(session_id, str) or not session_id or not isinstance(user_message, str):
            raise ValueError("A session ID and text message are required")
        if type(turn) is not int or not 1 <= turn <= 10 or type(top_k) is not int or not 1 <= top_k <= 10:
            raise ValueError("Turn and top_k must be integers in [1, 10]")
        state = self.inner._sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before respond")
        request = (turn, hashlib.sha256(user_message.encode("utf-8")).hexdigest(), top_k)
        previous = self._receipts.get(session_id)
        if previous is not None and previous.request == request:
            diagnostic = deepcopy(previous.diagnostics)
            diagnostic.update(cache_hit=True, latency_seconds=time.perf_counter() - started,
                              current_call={"search_executed": False, "inference_executed": False,
                                            "presentation_executed": False})
            origin = diagnostic.get("vector_stage", {})
            diagnostic["vector_stage"] = {"attempted": False, "inference_attempted": False,
                                          "status": "cached_response", "returned_count": 0,
                                          "contribution_count": 0, "origin_receipt": origin}
            components = diagnostic.get("effective_capabilities", {}).get("components", {})
            if "vector_rerank" in components:
                components["vector_rerank"].update(attempted=False, status="cached_response")
            if "neural_rerank" in components:
                components["neural_rerank"].update(score_called=False)
            self.last_diagnostics = diagnostic
            self.inner._sessions.move_to_end(session_id)
            return deepcopy(previous.response)
        if turn <= state.last_turn:
            raise ValueError("Turn must advance; conflicting or stale retries are not accepted")
        before_category, before_values = tuple(terms(state.category_text)), _positive_context(state)
        deferred_before = session_id in self.inner._ambiguity_deferred
        calls_before = self.observer.calls
        try:
            base_response = self.inner.respond(session_id, user_message, turn, top_k)
        except Exception:
            diagnostic = deepcopy(self.inner.last_diagnostics)
            diagnostic.update(presentation=asdict(self.config), presentation_identity=deepcopy(self.presentation_identity))
            self.last_diagnostics = diagnostic
            raise
        if self.observer.calls == calls_before:
            raise RuntimeError("A fresh response must expose its own search context")
        state = self.inner._sessions[session_id]
        raw = self.observer.context
        raw_ids = [item.identifier for item in raw]
        base_ids = [item["parent_asin"] for item in base_response["recommendations"]]
        if len(base_ids) != len(set(base_ids)) or not set(base_ids) <= set(raw_ids):
            raise RuntimeError("Base response changed candidate membership")
        ambiguity = (not base_ids and not deferred_before
                     and session_id in self.inner._ambiguity_deferred)
        rejected = set(previous.rejected) if previous is not None else set()
        reset_reason = _replacement_reason(before_category, before_values, state, turn)
        forgotten = sorted(rejected) if reset_reason else []
        if reset_reason:
            rejected.clear()
        rejection = explicit_slate_rejection(user_message)
        newly_rejected = []
        if (self.config.policy == "explicit_rejection" and rejection and previous is not None
                and not reset_reason):
            newly_rejected = sorted(set(previous.shown) - rejected)
            rejected.update(previous.shown)
        if len(rejected) > MAX_REJECTED:
            raise RuntimeError("Rejection memory exceeded the turn and slate bounds")
        ordered = raw
        reasons = []
        if self.config.policy == "explicit_rejection" and rejected:
            # A shopper rejection is negative preference evidence, not a reason
            # to promote a known semantic violation above compatible records.
            ordered = tuple(sorted(raw, key=lambda item: (item.violation, item.identifier in rejected)))
            if ordered != raw:
                reasons.append("explicit_rejection_stable_partition")
        response = deepcopy(base_response)
        if self.config.policy == "tentative_top1" and ambiguity and raw and not raw[0].violation:
            response["recommendations"] = [{"parent_asin": raw[0].identifier,
                                            "score": round(raw[0].score, 6)}]
            reasons.append("tentative_top1_after_ambiguity_deferral")
        elif self.config.policy == "tentative_top1" and ambiguity and raw and raw[0].violation:
            reasons.append("tentative_top1_blocked_known_violation")
        elif self.config.policy == "explicit_rejection" and ordered != raw and base_ids:
            response["recommendations"] = [{"parent_asin": item.identifier, "score": round(item.score, 6)}
                                           for item in ordered[:len(base_ids)]]
        elif self.config.policy == "raw10":
            if base_ids != raw_ids[:min(top_k, 10)]:
                raise RuntimeError("Raw10 control did not return the true ranked prefix")
            reasons.append("raw10_control")
        returned = tuple(item["parent_asin"] for item in response["recommendations"])
        if len(returned) != len(set(returned)) or not set(returned) <= set(raw_ids):
            raise RuntimeError("Presentation changed candidate membership")
        diagnostic = deepcopy(self.inner.last_diagnostics)
        base_width = deepcopy(diagnostic.get("output_width", {}))
        width = {**base_width, "before": len(base_ids), "after": len(returned),
                 "delta": len(returned) - len(base_ids), "returned": len(returned),
                 "base_ambiguity_deferred": ambiguity, "ambiguity_deferred": ambiguity and not returned}
        if list(returned) != base_ids:
            width.update(reason=reasons[0], policy_limit=len(returned))
        diagnostic.update({
            "presentation": asdict(self.config), "cache_hit": False,
            "presentation_identity": deepcopy(self.presentation_identity),
            "base_output_width": base_width,
            "candidate_context_ids": raw_ids, "ranked_context_ids": raw_ids,
            "presentation_context_ids": [item.identifier for item in ordered],
            "base_returned_ids": base_ids, "returned_ids": list(returned),
            "reasons": reasons or ["unchanged"], "explicit_slate_rejection": rejection,
            "previously_shown_ids": list(previous.shown) if previous else [],
            "newly_rejected_ids": newly_rejected, "rejected_ids": sorted(rejected),
            "rejection_memory_reset": reset_reason, "forgotten_rejected_ids": forgotten,
            "output_width": width,
            "ordering_changed": [item.identifier for item in ordered] != raw_ids,
            "question_unchanged": response["message"] == base_response["message"]
                                  and response["ask_attribute"] == base_response["ask_attribute"],
            "known_violation_tier_preserved": True,
            "latency_seconds": time.perf_counter() - started,
        })
        for name, identifiers in (("presentation_context", [item.identifier for item in ordered]),
                                  ("presentation_prefix", list(returned)), ("returned", list(returned))):
            receipt = stage_receipt(identifiers)
            diagnostic.setdefault("stage_receipts", {})[name] = receipt
            diagnostic.setdefault("stage_ids", {})[name] = receipt["ids"]
            diagnostic.setdefault("stage_counts", {})[name] = receipt["count"]
        diagnostic["stage_relationships"] = {
            "presentation_context_parent": "question_context",
            "presentation_prefix_parent": "presentation_context",
            "returned_parent": "presentation_prefix",
            "ranked_prefix": "Unchanged underlying ranking before presentation; rejection may select beyond this prefix.",
        }
        diagnostic.setdefault("current_call", {})["presentation_executed"] = True
        if list(returned) != base_ids:
            products = {product["parent_asin"]: product for product in self.observer.products}
            diagnostic["constraint_checks"] = constraint_receipts([products[key] for key in returned], state.evidence)
        diagnostic["constraint_checks_origin"] = "presentation_shown_products" if list(returned) != base_ids else "base_response"
        self._receipts[session_id] = PresentationReceipt(request, returned, frozenset(rejected),
                                                         deepcopy(response), deepcopy(diagnostic))
        self._prune()
        self.last_diagnostics = diagnostic
        return response

    def forget_profile(self, profile_id: str) -> None:
        self._require_open()
        affected = [session for session, current in self.inner._profile_ids.items() if current == profile_id]
        self.inner.forget_profile(profile_id)
        for session_id in affected:
            self._receipts.pop(session_id, None)
        self.last_diagnostics = {}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._receipts.clear()
        self.last_diagnostics = {}
        self.observer.context = ()
        self.observer.products = ()
        self.inner.close()


def make_agent(catalog_path: str | Path, policy: str = "existing", *, max_sessions: int = 256) -> PresentationAgent:
    return PresentationAgent(catalog_path, PresentationConfig(policy), max_sessions=max_sessions)


def _write(path: Path, payload: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _deny_network(*args, **kwargs):
    raise RuntimeError("Network connections are disabled for this experiment")


@patch.object(socket, "create_connection", _deny_network)
@patch.object(socket.socket, "connect", _deny_network)
def run(catalog: Path, dataset: Path, output: Path, policy: str) -> dict:
    config = PresentationConfig(policy)
    if file_sha256(Path("evaluator/local_evaluator.py")) != EVALUATOR_SHA256:
        raise ValueError("Evaluator identity does not match")
    output.mkdir(parents=True, exist_ok=False)
    registration = {
        "config": asdict(config), "runtime": source_hashes(),
        "catalog_sha256": file_sha256(catalog), "dataset_sha256": file_sha256(dataset),
        "evaluator_sha256": EVALUATOR_SHA256,
        "hypothesis": "Isolate tentative display and explicit rejection memory from retrieval and questions",
        "scope": "One named presentation preset; unchanged ranking and question planner",
        "network": "Python socket connection denial",
    }
    _write(output / "registration.json", registration)
    started = time.perf_counter()
    agent = make_agent(catalog, policy)
    cold = time.perf_counter() - started
    observed = ObservedAgent(agent)
    try:
        ids, categories, products = catalog_index(catalog)
        started = time.perf_counter()
        report = evaluate(observed, load_jsonl(dataset), ids, categories, products)
        elapsed = time.perf_counter() - started
        traces = [turn for session in observed.traces for turn in session]
        latencies = sorted(turn["latency_seconds"] for turn in traces)
        diagnostics = [turn.get("diagnostics", {}) for turn in traces]
        report["measurement"] = {
            "config": asdict(config), "cold_start_seconds": cold, "evaluation_seconds": elapsed,
            "p50_seconds": latencies[int((len(latencies) - 1) * .5)] if latencies else 0,
            "p95_seconds": latencies[int((len(latencies) - 1) * .95)] if latencies else 0,
            "peak_rss_bytes": peak_rss_bytes(), "errors": sum("error" in turn for turn in traces),
            "widths": dict(Counter(len(turn.get("response", {}).get("recommendations", [])) for turn in traces)),
            "width_delta": sum(item.get("output_width", {}).get("delta", 0) for item in diagnostics),
            "reason_counts": dict(Counter(reason for item in diagnostics for reason in item.get("reasons", []))),
            "rejection_events": sum(bool(item.get("newly_rejected_ids")) for item in diagnostics),
            "rejection_resets": sum(bool(item.get("rejection_memory_reset")) for item in diagnostics),
            "source_changed": source_hashes() != registration["runtime"],
            "catalog_changed": file_sha256(catalog) != registration["catalog_sha256"],
            "dataset_changed": file_sha256(dataset) != registration["dataset_sha256"],
        }
        _write(output / "report.json", report)
        _write(output / "traces.json", observed.traces)
        if any(report["measurement"][key] for key in ("errors", "source_changed", "catalog_changed", "dataset_changed")):
            raise RuntimeError("Experiment failed integrity checks; inspect its receipt")
        return report
    finally:
        agent.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--dataset", type=Path, default=Path("data/public_set.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    arguments = parser.parse_args()
    report = run(arguments.catalog, arguments.dataset, arguments.output, arguments.policy)
    print({key: report[key] for key in ("recommended_technical_score", "hit_rate_at_10", "mrr", "mttc")})


if __name__ == "__main__":
    main()
