from __future__ import annotations

import argparse
import html
import json
import re
import time
from pathlib import Path
from typing import Iterable

from mercury.agent import Agent
from mercury.config import Config
from mercury.types import Preference, Product


MESSAGES = (
    "I need a black leather shoulder bag with an adjustable strap.",
    "I don't want leather anymore. I've changed my mind; make that blue canvas, but keep the adjustable strap.",
    "I don't have another preference to add.",
    "No more details from me.",
    "None of these work; show another set.",
)


def _preference(preference: Preference) -> dict:
    return {
        "attribute": preference.attribute,
        "value": preference.value,
        "polarity": preference.polarity,
        "hard": preference.hard,
        "active": preference.active,
        "source_turn": preference.source_turn,
        "source_kind": preference.source_kind,
    }


def _mentions(product: Product, attribute: str, value: str) -> bool:
    normalized_value = re.sub(r"\s+", " ", value.lower()).strip()
    if normalized_value == "gray":
        normalized_value = "grey"
    if attribute in {"material", "color", "style", "use_case", "feature", "category"}:
        return normalized_value in product.facets.get(attribute, ())
    normalized = re.sub(r"\s+", " ", product.text.lower())
    return bool(normalized_value and re.search(
        rf"(?<!\w){re.escape(normalized_value)}(?!\w)", normalized,
    ))


def _catalog_evidence(product: Product, preferences: Iterable[Preference]) -> list[dict]:
    evidence = []
    for preference in preferences:
        mentioned = _mentions(product, preference.attribute, preference.value)
        if preference.polarity > 0 and mentioned:
            status = "supported"
            explanation = "The requested value appears in the catalog record."
        elif preference.polarity < 0 and mentioned:
            status = "contradicted"
            explanation = "The excluded value appears in the catalog record."
        else:
            status = "unknown"
            explanation = "The catalog record does not prove or disprove this preference."
        evidence.append({
            "attribute": preference.attribute,
            "value": preference.value,
            "polarity": preference.polarity,
            "status": status,
            "explanation": explanation,
        })
    return evidence


def _product_record(product: Product, preferences: Iterable[Preference], rank: int) -> dict:
    categories = product.fields.get("categories", "")
    return {
        "rank": rank,
        "parent_asin": product.parent_asin,
        "title": product.title or "Untitled catalog item",
        "price": product.price,
        "price_lower_bound": product.price_lower_bound,
        "categories": categories,
        "evidence": _catalog_evidence(product, preferences),
    }


def build_showcase(output: str | Path, agent: Agent, messages: Iterable[str] = MESSAGES,
                   results: dict | None = None, session_id: str = "judge-showcase") -> dict:
    """Run a real correction flow and write a portable, inspectable judge report."""
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    agent.reset(session_id, {})
    started = time.perf_counter()
    turns = []
    for turn, message in enumerate(messages, 1):
        response = agent.respond(session_id, message, turn, 10)
        diagnostics = agent.last_diagnostics
        state = agent.sessions[session_id]
        active = state.active_preferences()
        active_signatures = {(item.attribute, item.value, item.polarity) for item in active}
        identifiers = [item["parent_asin"] for item in response["recommendations"]]
        products = [
            _product_record(agent.catalog.by_id[identifier], active, rank)
            for rank, identifier in enumerate(identifiers, 1)
        ]
        turns.append({
            "turn": turn,
            "user_message": message,
            "agent_message": response["message"],
            "ask_attribute": response["ask_attribute"],
            "active_preferences": [_preference(item) for item in active],
            "retracted_preferences": [
                _preference(item) for item in state.preferences
                if not item.active and (item.attribute, item.value, item.polarity) not in active_signatures
            ],
            "intent": diagnostics["intent"],
            "query": diagnostics["query"],
            "negative_feedback": diagnostics["negative_feedback"],
            "slate_page": diagnostics["slate_page"],
            "slate_page_reset": diagnostics.get("slate_page_reset"),
            "fallbacks": diagnostics["fallbacks"],
            "latency_seconds": diagnostics["latency_seconds"],
            "products": products,
        })
    report = {
        "title": "Mercury judge showcase",
        "generated_from_real_agent": True,
        "catalog_product_count": len(agent.catalog.products),
        "catalog_sha256": agent.catalog.sha256,
        "config": agent.config.to_dict(),
        "elapsed_seconds": time.perf_counter() - started,
        "evidence_policy": (
            "Supported requires a literal catalog mention. Contradicted requires an excluded value "
            "to appear. Missing evidence remains unknown."
        ),
        "benchmark_results": results,
        "turns": turns,
    }
    (destination / "evidence.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (destination / "index.html").write_text(_render_html(report), encoding="utf-8")
    return report


def _chip(text: str, kind: str = "") -> str:
    return f'<span class="chip {kind}">{html.escape(text)}</span>'


def _render_html(report: dict) -> str:
    metrics = report.get("benchmark_results") or {}
    metric_cards = ""
    for key, label in (("technical_score", "Technical score"), ("hit_rate_at_10", "Hit rate @ 10"),
                       ("mrr", "MRR"), ("mttc", "MTTC")):
        if key in metrics:
            value = metrics[key]
            shown = f"{value:.3f}" if isinstance(value, float) else str(value)
            metric_cards += f'<div class="metric"><b>{html.escape(shown)}</b><span>{label}</span></div>'
    turn_sections = []
    for turn in report["turns"]:
        active = "".join(_chip(
            ("Avoid " if item["polarity"] < 0 else "") + f'{item["attribute"]}: {item["value"]}',
            "negative" if item["polarity"] < 0 else "positive",
        ) for item in turn["active_preferences"]) or _chip("No active preferences")
        retracted = "".join(_chip(
            f'{item["attribute"]}: {item["value"]}', "retracted",
        ) for item in turn["retracted_preferences"]) or '<span class="muted">None</span>'
        products = []
        for product in turn["products"][:5]:
            evidence = "".join(
                f'<li><span class="status {item["status"]}">{item["status"]}</span> '
                f'<b>{html.escape(item["attribute"])}:</b> {html.escape(item["value"])}'
                f'<small>{html.escape(item["explanation"])}</small></li>'
                for item in product["evidence"]
            ) or '<li><span class="muted">No extracted preferences to check.</span></li>'
            price = "Not listed" if product["price"] is None else (
                ("From " if product["price_lower_bound"] else "") + f'${product["price"]:.2f}'
            )
            products.append(
                f'<article class="product"><div class="rank">{product["rank"]}</div><div>'
                f'<h4>{html.escape(product["title"])}</h4>'
                f'<p class="meta">ID {html.escape(product["parent_asin"])} · {html.escape(price)}'
                f' · {html.escape(product["categories"] or "Uncategorised")}</p>'
                f'<ul class="evidence">{evidence}</ul></div></article>'
            )
        reasons = ", ".join(turn["intent"].get("reasons", [])) or "No additional routing reason"
        fallback = ", ".join(turn["fallbacks"]) or "None"
        page_reset = turn.get("slate_page_reset") or "None"
        turn_sections.append(
            f'<section class="turn"><header><span>Turn {turn["turn"]}</span>'
            f'<strong>{html.escape(turn["intent"]["mode"].title())} intent</strong></header>'
            f'<div class="conversation"><p><b>Shopper</b>{html.escape(turn["user_message"])}</p>'
            f'<p><b>Mercury</b>{html.escape(turn["agent_message"])}</p></div>'
            f'<div class="ledger"><div><h3>Active preference ledger</h3>{active}</div>'
            f'<div><h3>Retracted evidence</h3>{retracted}</div></div>'
            f'<details><summary>Why this response?</summary><dl>'
            f'<dt>Retrieval query</dt><dd>{html.escape(turn["query"] or "(empty)")}</dd>'
            f'<dt>Routing rationale</dt><dd>{html.escape(reasons)}</dd>'
            f'<dt>Negative feedback</dt><dd>{html.escape(turn["negative_feedback"]["scope"])}</dd>'
            f'<dt>Result page</dt><dd>{turn["slate_page"] + 1}</dd>'
            f'<dt>Page reset</dt><dd>{html.escape(page_reset)}</dd>'
            f'<dt>Fallbacks</dt><dd>{html.escape(fallback)}</dd></dl></details>'
            f'<h3>Top catalog results</h3>{"".join(products)}</section>'
        )
    data = json.dumps(report, separators=(",", ":")).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mercury judge showcase</title><style>
:root{{--ink:#172033;--muted:#697386;--line:#dfe4ea;--surface:#f6f8fb;--brand:#4b45d6;--good:#18794e;--bad:#c52a2a;--unknown:#7a5d00}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--surface);color:var(--ink);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1080px;margin:auto;padding:40px 20px 80px}}.hero,.turn{{background:white;border:1px solid var(--line);border-radius:18px;padding:26px;margin-bottom:22px;box-shadow:0 8px 24px #1720330b}}
h1{{font-size:38px;line-height:1.1;margin:8px 0}}h2,h3,h4,p{{margin-top:0}}.eyebrow,.turn>header span{{color:var(--brand);font-weight:750;text-transform:uppercase;letter-spacing:.08em}}
.metrics{{display:flex;gap:10px;flex-wrap:wrap;margin-top:22px}}.metric{{background:#eeedff;border-radius:12px;padding:12px 18px;display:grid}}.metric b{{font-size:23px}}.metric span,.muted,.meta,small{{color:var(--muted)}}
.turn>header{{display:flex;justify-content:space-between;margin-bottom:18px}}.conversation{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}.conversation p{{padding:16px;background:var(--surface);border-radius:12px}}.conversation b{{display:block;margin-bottom:5px}}
.ledger{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:18px 0}}.ledger>div{{border-left:3px solid var(--line);padding-left:15px}}.ledger h3{{font-size:14px;margin-bottom:8px}}
.chip{{display:inline-block;padding:4px 9px;border-radius:999px;background:#edf0f4;margin:2px 4px 2px 0;font-size:13px}}.positive{{background:#e5f6ed;color:var(--good)}}.negative{{background:#ffe9e8;color:var(--bad)}}.retracted{{text-decoration:line-through;color:var(--muted)}}
details{{border:1px solid var(--line);border-radius:10px;padding:10px 14px;margin:18px 0}}summary{{cursor:pointer;font-weight:700}}dl{{display:grid;grid-template-columns:145px 1fr;gap:5px 12px}}dt{{font-weight:700}}dd{{margin:0}}
.product{{display:grid;grid-template-columns:38px 1fr;gap:12px;padding:16px 0;border-top:1px solid var(--line)}}.rank{{width:30px;height:30px;border-radius:50%;display:grid;place-items:center;background:#eeedff;color:var(--brand);font-weight:800}}.product h4{{margin-bottom:3px}}.meta{{font-size:13px}}
.evidence{{padding:0;margin:10px 0 0;list-style:none;display:flex;flex-wrap:wrap;gap:7px}}.evidence li{{background:var(--surface);padding:7px 9px;border-radius:8px;font-size:12px}}.evidence small{{display:block}}.status{{font-weight:800;text-transform:uppercase;font-size:10px;margin-right:5px}}.supported{{color:var(--good)}}.contradicted{{color:var(--bad)}}.unknown{{color:var(--unknown)}}
.proof{{border-left:4px solid var(--brand);padding-left:14px}}@media(max-width:700px){{.conversation,.ledger{{grid-template-columns:1fr}}h1{{font-size:30px}}dl{{grid-template-columns:1fr}}}}
</style></head><body><main><section class="hero"><div class="eyebrow">Inspectable, offline shopping search</div>
<h1>See what Mercury remembered, changed, and proved.</h1>
<p class="proof">This report was generated by the real agent over {report["catalog_product_count"]:,} catalog products. It exposes the state transition behind every recommendation and never treats missing catalog data as evidence.</p>
<div class="metrics">{metric_cards}</div></section>{''.join(turn_sections)}
<section class="hero"><h2>Evidence boundary</h2><p>{html.escape(report["evidence_policy"])}</p><p class="muted">Open <code>evidence.json</code> beside this page to inspect the complete machine-readable trace.</p></section>
<script type="application/json" id="showcase-data">{data}</script></main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a portable judge-facing report from real agent turns.")
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("configs/selected.json"))
    parser.add_argument("--results", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/judge-showcase"))
    args = parser.parse_args()
    results = json.loads(args.results.read_text(encoding="utf-8")) if args.results else None
    agent = Agent(args.catalog, Config.load(args.config))
    try:
        build_showcase(args.output, agent, results=results)
    finally:
        agent.close()
    print(f"Judge showcase: {args.output / 'index.html'}")


if __name__ == "__main__":
    main()
