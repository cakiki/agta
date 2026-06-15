from dataclasses import dataclass, field
from agta.models import TripRecord


class EpisodicMemory:
    def __init__(self):
        self.records: list[TripRecord] = []

    def add(self, record: TripRecord):
        self.records.append(record)

    def retrieve(self, to_activity: str, k: int = 3) -> list[TripRecord]:
        relevant = [r for r in self.records if r.to_activity == to_activity]
        return relevant[-k:]

    def retrieve_by_day(self, day: int) -> list[TripRecord]:
        return [r for r in self.records if r.day == day]

    def to_prompt_string(self, to_activity: str, k: int = 3) -> str:
        retrieved = self.retrieve(to_activity, k)
        if not retrieved:
            return ""
        lines = []
        for r in retrieved:
            lines.append(f"  Day {r.day}, {r.time}: {r.from_activity} -> {r.to_activity}: {r.mode} ({r.reasoning})")
        return "\n".join(lines)
    