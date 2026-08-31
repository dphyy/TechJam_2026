"""Local, process-lifetime preference memory with no external dependencies."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


def _attribute_for(value: str) -> str:
    lowered = value.casefold()
    if any(term in lowered.split() for term in {"leather", "cotton", "wool", "nylon", "silk", "linen", "suede", "denim"}):
        return "material"
    if any(term in lowered.split() for term in {"black", "white", "blue", "red", "green", "brown", "gray", "grey", "navy"}):
        return "color"
    if any(term in lowered.split() for term in {"fit", "casual", "formal", "classic", "modern", "vintage", "slim", "relaxed"}):
        return "style"
    return "other"


@dataclass(frozen=True, slots=True)
class PreferenceRecord:
    attribute: str
    value: str
    confidence: float
    confirmations: int
    first_turn: int
    latest_turn: int


@dataclass
class LongTermUserProfile:
    profile_id: str
    learned: dict[tuple[str, str], PreferenceRecord] = field(default_factory=dict)
    _observations: dict[tuple[str, str], int] = field(default_factory=dict)

    def seed(self, tags: object) -> None:
        if not isinstance(tags, list):
            return
        for tag in tags:
            value = str(tag).strip()
            if value:
                attribute = _attribute_for(value)
                key = (attribute, value.casefold())
                self.learned.setdefault(key, PreferenceRecord(attribute, value, 0.65, 1, 0, 0))

    def observe(self, attribute: str, value: str, turn: int, *, durable: bool, replacement: bool) -> None:
        value = value.strip()
        if not value or attribute == "category":
            return
        if replacement:
            for key in [key for key in self.learned if key[0] == attribute and key[1] != value.casefold()]:
                self.learned.pop(key, None)
        key = (attribute, value.casefold())
        count = self._observations.get(key, 0) + 1
        self._observations[key] = count
        if not durable and not replacement and count < 2:
            return
        prior = self.learned.get(key)
        confidence = min(0.90, (prior.confidence if prior else (0.75 if durable or replacement else 0.70)) + (0.10 if prior else 0.0))
        self.learned[key] = PreferenceRecord(attribute, value, confidence, count, prior.first_turn if prior else turn, turn)

    def reject(self, attribute: str, value: str) -> None:
        needle = value.casefold()
        for key, record in list(self.learned.items()):
            if record.attribute == attribute and (needle in record.value.casefold() or record.value.casefold() in needle):
                self.learned.pop(key, None)

    def snapshot(self) -> dict:
        return {"profile_id": self.profile_id, "preferences": [asdict(record) for record in self.learned.values()]}


@dataclass
class UserProfileStore:
    profiles: dict[str, LongTermUserProfile] = field(default_factory=dict)

    def get(self, profile_id: str, user_profile: dict) -> LongTermUserProfile:
        profile = self.profiles.get(profile_id)
        if profile is None:
            profile = LongTermUserProfile(profile_id)
            profile.seed(user_profile.get("preference_tags") if isinstance(user_profile, dict) else None)
            self.profiles[profile_id] = profile
        return profile

    def forget(self, profile_id: str) -> None:
        self.profiles.pop(profile_id, None)
