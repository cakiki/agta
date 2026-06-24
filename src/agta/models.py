from dataclasses import dataclass


@dataclass
class RouteOption:
    mode: str
    distance_km: float
    duration_min: float
    cost_eur: float | None = None
    co2_grams: float | None = None
    transfers: int | None = None
    walk_distance_km: float | None = None
    wait_time_min: float | None = None
    nearby_pois: list[str] | None = None


@dataclass
class TripContext:
    route_id: str
    from_activity: str
    to_activity: str
    from_time: str
    to_time: str
    from_location: tuple
    to_location: tuple
    route_options: list[RouteOption]
    weather: str | None = None

@dataclass
class TripDecision:
    route_id: str
    mode: str
    reasoning: str
    vehicle_state: dict


@dataclass
class TripRecord:
    day: int
    time: str
    from_activity: str
    to_activity: str
    from_location: tuple
    to_location: tuple
    mode: str
    reasoning: str
    distance_km: float
    duration_min: float
    vehicle_state_before: dict
    vehicle_state_after: dict
    available_options: list[dict] | None = None
    episodic_retrievals: list[str] | None = None
