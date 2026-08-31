"""Bounded receipts derived from visible state, catalog evidence and executed stages."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import asdict, is_dataclass
from functools import partial
from pathlib import Path
import sqlite3
import sys

from .dialogue import Evidence, _infer_attribute
from .product_features import (
    FACET_PATTERNS,
    BUDGET_RE,
    FIELD_ORDER,
    ProductFeatures,
    ProductFeatureStore,
    evidence_product,
    component_scope,
    component_value,
    terms,
)
from .vector_index import CatalogVectorIndex, catalog_sha256, file_sha256


MAX_EVIDENCE = 128
MAX_STAGE_IDS = 2048
MAX_WITNESS_TEXT = 512


def signature(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def stage_receipt(identifiers) -> dict:
    if identifiers is None:
        return {"ids": [], "count": None, "sha256": None, "complete": False, "available": False}
    values = tuple(str(value) for value in identifiers)
    return {"ids": list(values[:MAX_STAGE_IDS]), "count": len(values),
            "sha256": signature(values), "complete": len(values) <= MAX_STAGE_IDS, "available": True}


def _component_source(component) -> dict:
    while isinstance(component, partial):
        component = component.func
    target = component if inspect.isclass(component) or inspect.isroutine(component) else type(component)
    try:
        source_path = Path(inspect.getfile(target)).resolve()
        digest = file_sha256(source_path)
    except (OSError, TypeError):
        source_path, digest = None, None
    return {"sha256": signature((target.__module__, target.__qualname__, digest)),
            "source_available": digest is not None,
            "local_component": source_path is not None and source_path.parent == Path(__file__).parent.resolve()}


def _factory_binding(factory) -> dict:
    """Hash explicit construction arguments without exporting arbitrary values."""
    def value(item, depth=0):
        if depth > 8:
            raise ValueError("binding depth")
        if item is None or type(item) in {bool, int, str}:
            return item
        if type(item) is float and math.isfinite(item):
            return item
        if isinstance(item, Path):
            return str(item)
        if is_dataclass(item) and not isinstance(item, type):
            return value(asdict(item), depth + 1)
        if isinstance(item, (list, tuple)) and len(item) <= 256:
            return [value(child, depth + 1) for child in item]
        if isinstance(item, dict) and len(item) <= 256 and all(isinstance(key, str) for key in item):
            return {key: value(child, depth + 1) for key, child in item.items()}
        raise ValueError("opaque binding")

    source = _component_source(factory)
    try:
        arguments = value({"args": factory.args, "kwargs": factory.keywords}) if isinstance(factory, partial) else {}
        argument_hash, complete = signature(arguments), True
    except (ValueError, TypeError):
        argument_hash, complete = None, False
    target = factory.func if isinstance(factory, partial) else factory
    return {"implementation_sha256": source["sha256"], "arguments_sha256": argument_hash,
            "arguments_complete": complete and inspect.isclass(target)}


def runtime_identity(catalog_path: Path, config, policies, search, *, agent, planner, search_factory,
                     feature_cache_size: int, max_sessions: int, share_profile_memory: bool) -> dict:
    sources = {path.name: file_sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
    components = {"agent": _component_source(agent), "search": _component_source(search),
                  "search_factory": _component_source(search_factory),
                  "question_planner": _component_source(planner),
                  "intent_router": _component_source(search.intent_router)}
    factory_binding = _factory_binding(search_factory)
    settings = {"agent": asdict(config), "ranking": asdict(policies),
                "feature_cache_size": feature_cache_size, "max_sessions": max_sessions,
                "share_profile_memory": share_profile_memory, "search_factory": factory_binding,
                "effective_ranking": asdict(search.ranking_policies),
                "effective_feature_cache_size": search.feature_store.max_size}
    limitations = []
    if any(not receipt["local_component"] for receipt in components.values()):
        limitations.append("custom_component_configuration")
    if not factory_binding["arguments_complete"]:
        limitations.append("opaque_factory_context")
    requested_digest = catalog_sha256(catalog_path)
    effective_path = getattr(search, "catalog_path", None)
    effective_digest = None
    if isinstance(effective_path, (str, Path)):
        effective_path = Path(effective_path)
        effective_digest = (requested_digest if effective_path.resolve() == catalog_path.resolve()
                            else catalog_sha256(effective_path))
        if effective_digest != requested_digest:
            raise ValueError("effective catalog does not match requested catalog")
    else:
        limitations.append("effective_catalog_unavailable")
    vector = search.vector_index
    vector_assets = {}
    if vector is not None:
        components["vector_index"] = _component_source(vector)
        if type(vector) is CatalogVectorIndex:
            settings["vector"] = {"model_sha256": signature(vector.model), "dimensions": vector.dimensions,
                                  "query_cache_capacity": vector._cache_capacity}
            if vector.enabled:
                vector_assets = {"vectors_sha256": file_sha256(vector.vectors_path),
                                 "metadata_sha256": file_sha256(vector.metadata_path)}
                limitations.append("remote_inference_identity")
        else:
            limitations.append("external_vector_identity")
    return {"catalog_sha256": effective_digest, "requested_catalog_sha256": requested_digest,
            "catalog_count": len(search._row_id_by_asin),
            "config_sha256": signature(settings), "configuration": settings,
            "runtime_source_sha256": signature({"package": sources, "components": components}),
            "runtime_hashes": sources, "implementations": components,
            "binding": {"complete": not limitations, "limitations": limitations,
                        "scope": "agent_and_dependencies_at_construction"},
            "environment": {"python": sys.version.split()[0], "sqlite": sqlite3.sqlite_version},
            "vector_assets": vector_assets,
            "catalog_index": {"prebuilt_loaded": search.using_prebuilt_index,
                              "artifact_present": search.catalog_index_path.is_file(),
                              "effective_backend": "persisted" if search.using_prebuilt_index else "memory"}}


def _evidence_id(item: Evidence) -> str:
    return signature((item.text, item.weight, item.source, item.turn, item.attribute, item.operation.value))


def _record(item: Evidence, raw_chunk: str | None = None) -> dict:
    return {"evidence_id": _evidence_id(item), "attribute": _infer_attribute(item.text, item.attribute),
            "value": item.text, "raw_chunk": item.text if raw_chunk is None else raw_chunk,
            "source_turn": item.turn, "source_kind": item.source, "weight": item.weight,
            "operation": item.operation.value, "scope": component_scope(item.text),
            "polarity": -1 if item.source == "exclusion" else 1,
            "hard": item.source in {"hard_constraint", "override", "exclusion"}, "active": True}


def evidence_receipt(before: list[Evidence], after: list[Evidence], previous: dict, turn: int) -> dict:
    """Record actual removals, including the old chunk when a partial chunk survives."""
    old_rows = {row["evidence_id"]: row for row in previous.get("active", [])}
    before_set, after_set = set(before), set(after)
    removed = [item for item in before if item not in after_set]
    active = []
    for item in after[:MAX_EVIDENCE]:
        prior = old_rows.get(_evidence_id(item))
        raw_chunk = prior["raw_chunk"] if prior else None
        if prior is None:
            origins = [old for old in removed
                       if (old.turn, old.source, old.attribute, old.operation, old.weight)
                       == (item.turn, item.source, item.attribute, item.operation, item.weight)
                       and item.text in old.text]
            if len(origins) == 1:
                old = origins[0]
                raw_chunk = old_rows.get(_evidence_id(old), {}).get("raw_chunk", old.text)
        active.append(_record(item, raw_chunk))
    retired = list(previous.get("retired", []))
    for item in removed:
        row = _record(item, old_rows.get(_evidence_id(item), {}).get("raw_chunk", item.text))
        row.update(active=False, retired_turn=turn, retirement_reason="removed_from_active_state")
        retired.append(row)
    return {"active": active, "active_count": len(after), "active_complete": len(after) <= MAX_EVIDENCE,
            "retired": retired[-MAX_EVIDENCE:],
            "retired_count": previous.get("retired_count", 0) + len(removed),
            "retired_complete": previous.get("retired_count", 0) + len(removed) <= MAX_EVIDENCE,
            "added_ids": [_evidence_id(item) for item in after if item not in before_set][:MAX_EVIDENCE],
            "removed_ids": [_evidence_id(item) for item in removed][:MAX_EVIDENCE]}


def _contains(sequence: tuple[str, ...], phrase: tuple[str, ...]) -> bool:
    return bool(phrase) and any(sequence[start:start + len(phrase)] == phrase
                               for start in range(len(sequence) - len(phrase) + 1))


def constraint_receipts(products: list[dict], evidence: list[Evidence]) -> list[dict]:
    query = ProductFeatureStore(max_size=1).compile_query(evidence[:MAX_EVIDENCE])
    result = []
    for product in products[:10]:
        features = product.get("_features")
        checks = []
        for raw, item in zip(evidence[:MAX_EVIDENCE], query.evidence):
            check = {"evidence_id": _evidence_id(raw), "value": raw.text,
                     "source_turn": raw.turn, "status": "unknown", "witnesses": []}
            if isinstance(features, ProductFeatures):
                budget = BUDGET_RE.search(raw.text)
                if item.is_budget and budget and item.source != "exclusion" and features.price is not None:
                    amount = float(budget.group("amount"))
                    mode = (budget.group("mode") or "around").lower()
                    if math.isfinite(amount):
                        check["witnesses"].append({"field": "price", "match_kind": "numeric_value",
                                                  "catalog_value": features.price, "requested_value": amount,
                                                  "mode": mode})
                        if mode in {"under", "below", "maximum", "max"}:
                            check["status"] = "supported" if features.price <= amount else "contradicted"
                        checks.append(check)
                        continue
                for branch in item.alternatives or (item,):
                    view = evidence_product(features, branch)
                    sequences = (view.field_sequences if branch.literal_absence and item.source != "exclusion"
                                 else view.affirmed_sequences)
                    phrase = tuple(terms(component_value(branch.normalized_query))) if branch.scope else branch.tokens
                    for field, sequence in zip(FIELD_ORDER, sequences):
                        if _contains(sequence, phrase):
                            value = str(product.get(field) or "")
                            check["witnesses"].append({"field": field,
                                "match_kind": "scoped_value" if branch.scope else "normalized_phrase",
                                "scope": branch.scope, "normalized_phrase": " ".join(phrase),
                                "raw_value": value[:MAX_WITNESS_TEXT],
                                "raw_value_complete": len(value) <= MAX_WITNESS_TEXT})
                    if check["witnesses"]:
                        check["status"] = "contradicted" if item.source == "exclusion" else "supported"
                        break
                if check["status"] == "unknown" and item.source == "exclusion":
                    fields = features.component_fields.get(item.scope, ()) if item.scope else tuple(
                        str(product.get(field) or "") for field in FIELD_ORDER)
                    negative_phrases = (tuple(terms(f"no {item.normalized_query}")),
                                        tuple(terms(f"without {item.normalized_query}")),
                                        tuple(terms(f"{item.normalized_query} free")))
                    for field, value in zip(FIELD_ORDER, fields):
                        if any(_contains(tuple(terms(value)), phrase) for phrase in negative_phrases):
                            check["status"] = "supported"
                            check["witnesses"].append({"field": field, "match_kind": "explicit_absence",
                                "scope": item.scope, "raw_value": value[:MAX_WITNESS_TEXT],
                                "raw_value_complete": len(value) <= MAX_WITNESS_TEXT})
                if check["status"] == "unknown" and item.scope and item.source != "exclusion":
                    # Every alternative must have a known contradiction before
                    # an OR group can be called contradicted.
                    contradictions = []
                    for branch in item.alternatives or (item,):
                        view = evidence_product(features, branch)
                        contradictions.append(any(
                            (actual := set(FACET_PATTERNS[attribute].findall(view.normalized_text)))
                            and actual.isdisjoint(expected) for attribute, expected in branch.facets
                        ))
                    if contradictions and all(contradictions):
                        check["status"] = "contradicted"
            checks.append(check)
        result.append({"parent_asin": str(product["parent_asin"]), "evidence": checks})
    return result


def capability_receipt(identity: dict, config, search, vector_stage: dict) -> dict:
    vector = search.vector_index
    known_enabled = getattr(vector, "enabled", None)
    loaded = bool(vector is not None and (known_enabled is None or known_enabled is True))
    requested = bool(config.enable_vector_reranker or vector is not None)
    attempted = vector_stage.get("attempted", False)
    status = vector_stage.get("status", "not_attempted")
    faults = []
    if attempted and status in {"inference_failed", "client_unavailable", "backend_unavailable", "invalid_similarity"}:
        faults.append("vector_rerank")
    fallback = bool(requested and not loaded) or bool(faults)
    return {"catalog_sha256": identity["catalog_sha256"],
            "config_sha256": identity["config_sha256"],
            "runtime_source_sha256": identity["runtime_source_sha256"],
            "components": {
                "sparse_retrieval": {"requested": True, "loaded": True, "effective": True},
                "exact_constraint_index": {"requested": True, "loaded": True, "effective": True},
                "vector_rerank": {"requested": requested, "loaded": loaded,
                    "effective": bool(attempted and vector_stage.get("contribution_count", 0) > 0),
                    "returned_results": vector_stage.get("returned_count", 0) > 0,
                    "attempted": attempted, "status": status,
                    "contributed": vector_stage.get("contribution_count", 0) > 0},
                "neural_rerank": {"requested": False, "loaded": False, "effective": False}},
            "ranking_faults": faults, "fallbacks": ["vector_rerank"] if fallback else []}


def turn_receipt(*, before, state, previous, sources, turn, top_k, result, response,
                 identity, config, search, latency_seconds: float, deferred: bool) -> dict:
    evidence = evidence_receipt(before, state.evidence, previous.get("evidence", {}), turn)
    stage_values = {
        "retrieval_union": result.candidate_ids,
        "question_context": [row["parent_asin"] for row in result.candidates],
        "ranked_prefix": [key for key, _ in result.recommendations],
        "returned": [row["parent_asin"] for row in response["recommendations"]],
    }
    stages = {name: stage_receipt(values) for name, values in stage_values.items()}
    capabilities = capability_receipt(identity, config, search, result.vector_stage)
    returned = set(stage_values["returned"])
    checks = constraint_receipts([row for row in result.candidates if row["parent_asin"] in returned], state.evidence)
    fallbacks = list(capabilities["fallbacks"])
    if identity["catalog_index"]["artifact_present"] and not identity["catalog_index"]["prebuilt_loaded"]:
        fallbacks.append("catalog_index_rebuilt")
    return {"turn": turn, "cache_hit": False, "request_succeeded": True,
            "effective_capabilities": capabilities, "identity": identity, "fallbacks": fallbacks,
            "latency_seconds": latency_seconds, "evidence": evidence,
            "preferences": evidence["active"], "retired_preferences": evidence["retired"],
            "evidence_sources": [{"source_turn": source_turn, **source} for source_turn, source in sorted(sources.items())],
            "stage_ids": {name: value["ids"] for name, value in stages.items()},
            "stage_counts": {name: value["count"] for name, value in stages.items()},
            "stage_receipts": stages, "retrieved_ids": stages["retrieval_union"]["ids"],
            "constraint_checks": checks, "vector_stage": result.vector_stage,
            "output_width": {"requested": top_k, "returned": len(response["recommendations"]),
                             "full_width": config.full_width, "ambiguity_deferred": deferred},
            "usage": response["usage"]}
