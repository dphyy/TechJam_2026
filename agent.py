"""Portable submission entry point; the evaluator remains unchanged."""

from dataclasses import replace
from pathlib import Path

from mercury.agent import Agent as SearchAgent
from mercury.config import Config


class Agent(SearchAgent):
    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        root = Path(__file__).resolve().parent
        selected = root / "configs" / "selected.json"
        config = Config.load(selected) if selected.exists() else Config()
        artifacts = Path(config.artifact_dir)
        if not artifacts.is_absolute():
            config = replace(config, artifact_dir=str(root / artifacts))
        super().__init__(catalog_path, config)
