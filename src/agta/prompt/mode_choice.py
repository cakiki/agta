from pathlib import Path
from jinja2 import Template
from agta.models import TripContext
from agta.memory.memory_manager import MemoryManager

_template = Template((Path(__file__).parent / "templates" / "mode_choice.jinja").read_text())

def build_trip_prompt(persona: str, memory: MemoryManager, trip: TripContext) -> str:
    return _template.render(
        persona=persona,
        memory_state=memory.get_prompt_ready(),
        episodic_context=memory.get_episodic_context(trip.to_activity),
        from_time=trip.from_time,
        from_activity=trip.from_activity,
        to_activity=trip.to_activity,
        options=trip.route_options,
        weather=trip.weather,
    )