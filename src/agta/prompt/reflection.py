from agta.memory.memory_manager import MemoryManager


def build_reflection_prompt(persona: str, memory: MemoryManager, day: int) -> str:
    trips = memory.episodic.retrieve_by_day(day)
    if not trips:
        return ""
    lines = []
    lines.append(f"You are:\n{persona}")
    lines.append(f"\nToday's trips:")
    for t in trips:
        lines.append(f"  {t.time} {t.from_activity} -> {t.to_activity}: {t.mode} ({t.reasoning})")
    lines.append(f"\nBased on today's travel, what general beliefs or preferences about your transport habits can you extract?")
    lines.append('Return JSON: {"beliefs": ["belief 1", "belief 2", ...]}')
    return "\n".join(lines)