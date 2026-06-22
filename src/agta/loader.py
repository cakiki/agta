import json
from agta.models import TripContext, RouteOption
import random

MODE_MAP = {"pedestrian": "walk", "passenger": "car", "bicycle": "bicycle", "public transport": "public transport"}


def load_from_json(path: str, limit: int | None = None, seed=None):
    with open(path) as f:
        raw_data = json.load(f)
    if limit:
        if seed is not None:
            rng = random.Random(seed)
            raw_data = rng.sample(raw_data, min(limit, len(raw_data)))
        else:
            raw_data = raw_data[:limit]

    agents_data = {}
    routes_data = {}

    for entry in raw_data:
        agent = json.loads(entry) if isinstance(entry, str) else entry
        agent_id = agent["id"]

        agents_data[agent_id] = {
            "persona": agent["description"],
            "seed": agent["seed"],
            "schedule": agent.get("day_schedule"),
            "attitudes": None,
        }

        trips = []
        for lc in agent.get("location_changes") or []:
            options = []
            for r in lc.get("possible_routes") or []:
                options.append(RouteOption(
                    mode=MODE_MAP[r["means_of_transport"]],
                    distance_km=round(r["distance"] / 1000, 2),
                    duration_min=round(r["travel_time"] / 60, 1),
                ))

            trips.append(TripContext(
                route_id=str(lc["route_id"]),
                from_activity=lc["from"]["task"]["building_type"],
                to_activity=lc["to"]["task"]["building_type"],
                from_time=lc["from"]["task"]["time"],
                to_time=lc["to"]["task"]["time"],
                from_location=(lc["from"]["building"]["location"]["x"], lc["from"]["building"]["location"]["y"]),
                to_location=(lc["to"]["building"]["location"]["x"], lc["to"]["building"]["location"]["y"]),
                route_options=options,
            ))

        routes_data[agent_id] = trips

    return agents_data, routes_data


def generate(**kwargs):
    raise NotImplementedError("On-the-fly generation not yet implemented")