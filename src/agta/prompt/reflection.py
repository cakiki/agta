import calendar
from pathlib import Path
from jinja2 import Template
from agta.memory.memory_manager import MemoryManager

_reflection = Template((Path(__file__).parent / "templates" / "reflection.jinja").read_text())
_procedural = Template((Path(__file__).parent / "templates" / "procedural_reflection.jinja").read_text())
_consolidation = Template((Path(__file__).parent / "templates" / "consolidation.jinja").read_text())
_evaluation = Template((Path(__file__).parent / "templates" / "evaluation.jinja").read_text())
_rules_consolidation = Template((Path(__file__).parent / "templates" / "rules_consolidation.jinja").read_text())

def build_reflection_prompt(persona: str, memory: MemoryManager, day: int, evaluations: list | None = None) -> str:
    trips = memory.episodic.retrieve_by_day(day)
    if not trips:
        return ""
    return _reflection.render(persona=persona, trips=trips, evaluations=evaluations or [], weekday=calendar.day_name[day % 7])

def build_procedural_reflection_prompt(persona: str, memory: MemoryManager, day: int) -> str:
    trips = memory.episodic.retrieve_by_day(day)
    if not trips:
        return ""
    return _procedural.render(persona=persona, trips=trips, weekday=calendar.day_name[day % 7])

def build_consolidation_prompt(beliefs: list[str]) -> str:
    return _consolidation.render(beliefs=beliefs)

def build_evaluation_prompt(persona: str, memory: MemoryManager, day: int) -> str:
    trips = memory.episodic.retrieve_by_day(day)
    if not trips:
        return ""
    return _evaluation.render(persona=persona, trips=trips, weekday=calendar.day_name[day % 7])

def build_rules_consolidation_prompt(rules: list[str]) -> str:
    return _rules_consolidation.render(rules=rules)