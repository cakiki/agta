import random


BERLIN_WEATHER = [
    ("sunny, 22°C", 0.25),
    ("partly cloudy, 18°C", 0.20),
    ("overcast, 14°C", 0.15),
    ("overcast, 10°C", 0.10),
    ("light rain, 12°C", 0.15),
    ("rain, 9°C", 0.10),
    ("heavy rain, 8°C", 0.05),
]


def resolve_weather(config: dict, day: int, num_days: int) -> str | None:
    weather_config = config.get("simulation", {}).get("weather")
    if not weather_config:
        return None

    source = weather_config.get("source", "config")

    if source == "config":
        values = weather_config.get("values", [])
        if day < len(values):
            return values[day]
        return None

    if source == "random":
        seed = weather_config.get("seed", 42)
        rng = random.Random(seed + day)
        conditions, weights = zip(*BERLIN_WEATHER)
        return rng.choices(conditions, weights=weights, k=1)[0]

    return None