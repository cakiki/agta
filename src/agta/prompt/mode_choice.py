from agta.models import TripContext
from agta.memory.memory_manager import MemoryManager


def build_trip_prompt(persona: str, memory: MemoryManager, trip: TripContext) -> str:
    lines = []
    lines.append(f"You are:\n{persona}")
    lines.append(f"\nCurrent state:\n{memory.get_prompt_ready()}")
    lines.append(f"\nYour next trip:")
    lines.append(f"  {trip.from_time} {trip.from_activity} -> {trip.to_activity}")
    lines.append(f"\nAvailable options:")
    for option in trip.route_options:
        parts = [f"  {option.mode}: {option.distance_km}km, {option.duration_min}min"]
        if option.cost_eur is not None:
            parts.append(f"{option.cost_eur}EUR")
        if option.transfers is not None:
            parts.append(f"{option.transfers} transfers")
        lines.append(", ".join(parts))
    if trip.weather:
        lines.append(f"\nWeather: {trip.weather}")
    lines.append('\nChoose your transport mode. Explain your reasoning in one sentence.')
    lines.append('Return JSON: {"mode": "...", "reasoning": "..."}')
    return "\n".join(lines)