"""Bounded loaded-runtime identity without per-turn artifact I/O."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


_DETERMINISTIC_FALLBACKS = frozenset({
    "no_matches", "minimal_probe_no_matches", "no_intent_hypothesis",
})
_COMPONENT_FAULTS = {
    "dense": "dense", "contrast": "contrast", "neural_rerank": "neural_rerank",
    "frontier_rerank": "neural_rerank", "frontier_page_rerank": "neural_rerank",
    "latency_budget": "neural_rerank", "admission_model": "admission_model",
}
_FAULT_CODES = frozenset({
    *_COMPONENT_FAULTS, "ranking", "constraints", "product_guard",
})


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _asset_digest(component: object, attribute: str) -> tuple[str | None, bool]:
    try:
        value = getattr(component, attribute, None)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None, False
    if value is None:
        return None, True
    if isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value):
        return value, True
    return None, False


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    identity: str
    config_sha256: str
    catalog_sha256: str
    components: dict[str, dict]
    startup_unavailable: tuple[str, ...]
    identity_valid: bool

    def blocking_reasons(self, fallbacks: list[str] | tuple[str, ...]) -> tuple[str, ...]:
        allowed = _DETERMINISTIC_FALLBACKS | set(self.startup_unavailable)
        faults = {
            reason if reason in _FAULT_CODES else "runtime_failure"
            for reason in fallbacks if reason not in allowed
        }
        if not self.identity_valid:
            faults.add("invalid_identity")
        return tuple(sorted(faults))

    def diagnostics(self, fallbacks: list[str]) -> dict:
        components = {name: dict(value) for name, value in self.components.items()}
        faults = self.blocking_reasons(fallbacks)
        for fault in faults:
            name = _COMPONENT_FAULTS.get(fault)
            if name in components:
                components[name]["effective"] = False
                components[name]["reason"] = "budget_deferred" if fault == "latency_budget" else "runtime_failure"
        return {
            "identity_sha256": self.identity,
            "config_sha256": self.config_sha256,
            "catalog_sha256": self.catalog_sha256,
            "components": components,
            "ranking_faults": list(faults),
        }


class RuntimeIdentity:
    """Retain only current component references; replacement invalidates all old keys."""

    def __init__(self) -> None:
        self._components: tuple[object, ...] = ()
        self._generation = 0

    def snapshot(self, config: dict, catalog_sha256: str,
                 components: dict[str, tuple[bool, object | None]],
                 startup_fallbacks: dict[str, str]) -> RuntimeSnapshot:
        ordered = sorted(components)
        current = tuple(components[name][1] for name in ordered)
        if len(current) != len(self._components) or any(
            before is not after for before, after in zip(self._components, current)
        ):
            self._generation += 1
            self._components = current
        rows = {}
        valid = True
        startup = []
        for name in ordered:
            requested, component = components[name]
            loaded = component is not None
            asset, asset_valid = _asset_digest(component, "asset_identity")
            backend, backend_valid = _asset_digest(component, "backend_identity")
            identity_valid = asset_valid and backend_valid
            valid = valid and identity_valid
            if requested and not loaded and name in startup_fallbacks:
                startup.append(name)
            reason = ("invalid_identity" if not identity_valid else "loaded" if loaded
                      else "unavailable" if requested else "disabled")
            rows[name] = {
                "requested": requested, "loaded": loaded,
                "effective": loaded and identity_valid, "reason": reason,
                "asset_sha256": asset, "backend_sha256": backend,
            }
        config_sha256 = _digest(config)
        identity = _digest({
            "version": 1, "config_sha256": config_sha256,
            "catalog_sha256": catalog_sha256, "components": rows,
            "generation": self._generation,
        })
        return RuntimeSnapshot(identity, config_sha256, catalog_sha256, rows, tuple(startup), valid)
