from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Config:
    dense: bool = False
    neural_rerank: bool = False
    contrast: bool = False
    evidence_ranking: bool = True
    state_mode: str = "ledger"
    alternatives_mode: str = "off"
    question_policy: str = "other"
    slate_policy: str = "fixed"
    slate_size: int = 10
    other_question_limit: int = 2
    candidate_limit: int = 120
    sparse_limit: int = 180
    dense_limit: int = 100
    rerank_limit: int = 30
    max_sessions: int = 256
    dense_weight: float = 0.35
    neural_weight: float = 0.25
    contrast_weight: float = 0.12
    artifact_dir: str = "artifacts"
    device: str = "cpu"
    threads: int = 4

    def __post_init__(self) -> None:
        choices = {
            "state_mode": {"ledger", "latest", "history"},
            "alternatives_mode": {"off", "parse", "grouped"},
            "question_policy": {"other", "schedule", "entropy", "rank_value", "none"},
            "slate_policy": {"fixed", "gap", "lookahead"},
            "device": {"cpu", "mps", "cuda"},
        }
        for key, allowed in choices.items():
            if getattr(self, key) not in allowed:
                raise ValueError(f"Invalid {key}: {getattr(self, key)!r}")
        if self.alternatives_mode == "grouped" and self.state_mode != "ledger":
            raise ValueError("Grouped alternatives require ledger state")
        for key in ("dense", "neural_rerank", "contrast", "evidence_ranking"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be a boolean")
        for key in ("candidate_limit", "sparse_limit", "dense_limit", "rerank_limit",
                    "max_sessions", "threads"):
            value = getattr(self, key)
            if type(value) is not int or not 1 <= value <= 10000:
                raise ValueError(f"{key} must be an integer in [1, 10000]")
        if type(self.slate_size) is not int or not 0 <= self.slate_size <= 10:
            raise ValueError("slate_size must be an integer in [0, 10]")
        if type(self.other_question_limit) is not int or not 0 <= self.other_question_limit <= 9:
            raise ValueError("other_question_limit must be an integer in [0, 9]")
        for key in ("dense_weight", "neural_weight", "contrast_weight"):
            value = getattr(self, key)
            if type(value) not in (int, float) or not 0 <= value <= 1:
                raise ValueError(f"{key} must be a finite number in [0, 1]")
        if not isinstance(self.artifact_dir, str) or not self.artifact_dir:
            raise ValueError("artifact_dir must be a nonempty path")

    @classmethod
    def from_dict(cls, values: dict) -> Config:
        if not isinstance(values, dict):
            raise ValueError("Configuration must be an object")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        return cls(**values)

    @classmethod
    def load(cls, path: str | Path) -> Config:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict:
        return asdict(self)
