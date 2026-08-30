from __future__ import annotations

import json
import heapq
import math
import time
from collections import defaultdict
from pathlib import Path

from mercury.catalog import Catalog
from mercury.model_assets import file_sha256
from mercury.ranking import value_matches
from mercury.retrieval import terms
from mercury.types import Candidate, Preference


CONTRAST_VERSION = "lexical-neighbors-v2-negation"


def compile_contrasts(catalog: Catalog, neighbor_limit: int = 8, pool_limit: int = 256) -> dict:
    """Uniform, bounded title-neighbor comparisons; never generate new facts."""
    token_sets = [set(terms(product.title)) for product in catalog.products]
    postings: dict[str, list[int]] = defaultdict(list)
    for index, tokens in enumerate(token_sets):
        for token in tokens:
            postings[token].append(index)
    count = len(catalog.products)
    idf = {token: math.log1p(count / len(indices)) for token, indices in postings.items()}
    records = {}
    for index, product in enumerate(catalog.products):
        tokens = token_sets[index]
        rare_terms = sorted((token for token in tokens if 1 < len(postings[token]) <= 1000),
                            key=lambda token: (len(postings[token]), token))[:6]
        pool: set[int] = set()
        # The pool cap is per product, independent of target labels or profiles.
        for token in rare_terms:
            pool.update(postings[token][:pool_limit])
        pool.discard(index)

        def similarity(other: int) -> tuple[float, str]:
            shared = tokens & token_sets[other]
            union = tokens | token_sets[other]
            numerator = sum(idf[token] for token in shared)
            denominator = sum(idf[token] for token in union) or 1.0
            return (-numerator / denominator, catalog.products[other].parent_asin)

        # At most six capped posting lists; final evidence uses only nearest eight.
        neighbors = heapq.nsmallest(neighbor_limit, pool, key=similarity)
        differences = []
        if neighbors:
            unique_evidence = {}
            for item in product.evidence:
                if len(item.value) > 80:
                    continue
                key = (item.attribute, item.value)
                if key not in unique_evidence or item.confidence > unique_evidence[key].confidence:
                    unique_evidence[key] = item
            for (attribute, value), evidence in sorted(unique_evidence.items()):
                support = sum(any(value_matches(value, other_value)
                                  for other_value in catalog.products[other].facets.get(attribute, ()))
                              for other in neighbors)
                if support < len(neighbors):
                    differences.append({"attribute": attribute, "value": value,
                                        "source": evidence.source, "confidence": evidence.confidence,
                                        "neighbor_support": support, "neighbor_count": len(neighbors),
                                        "weight": (1.0 - support / len(neighbors)) * evidence.confidence})
        records[product.parent_asin] = {"neighbors": [catalog.products[i].parent_asin for i in neighbors],
                                       "differences": differences}
    return records


def write_contrasts(catalog: Catalog, destination: Path) -> dict:
    if destination.exists():
        ContrastIndex(catalog, destination)
        return json.loads((destination / "manifest.json").read_text())
    started = time.perf_counter()
    records = compile_contrasts(catalog)
    destination.mkdir(parents=True)
    data = destination / "contrasts.json"
    data.write_text(json.dumps(records, separators=(",", ":")))
    manifest = {"version": CONTRAST_VERSION, "catalog_sha256": catalog.sha256,
                "count": len(catalog.products), "neighbor_limit": 8, "pool_per_term_limit": 256,
                "posting_terms_limit": 6, "sha256": file_sha256(data),
                "build_seconds": time.perf_counter() - started, "bytes": data.stat().st_size}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


class ContrastIndex:
    def __init__(self, catalog: Catalog, root: Path):
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("version") != CONTRAST_VERSION or manifest.get("catalog_sha256") != catalog.sha256:
            raise ValueError("Contrast index version/catalog mismatch")
        if file_sha256(root / "contrasts.json") != manifest.get("sha256"):
            raise ValueError("Contrast data checksum mismatch")
        self.records = json.loads((root / "contrasts.json").read_text())
        if set(self.records) != set(catalog.by_id):
            raise ValueError("Contrast index product IDs do not match catalog")
        self._by_attribute = {}
        for identifier, record in self.records.items():
            grouped = defaultdict(list)
            for item in record["differences"]:
                grouped[item["attribute"]].append(item)
            self._by_attribute[identifier] = {key: tuple(items) for key, items in grouped.items()}

    def rank(self, candidates: list[Candidate], preferences: list[Preference], weight: float) -> list[Candidate]:
        result = []
        for candidate in candidates:
            attributes = self._by_attribute[candidate.product.parent_asin]
            support = sum(max((item["weight"] for item in attributes.get(preference.attribute, ())
                               if value_matches(preference.value, item["value"])), default=0.0)
                          * preference.polarity * preference.confidence
                          for preference in preferences if preference.active and preference.polarity)
            result.append(Candidate(candidate.product, candidate.score + weight * support,
                                    {**candidate.route_scores, "contrast": support}))
        return sorted(result, key=lambda item: (-item.score, item.product.parent_asin))
