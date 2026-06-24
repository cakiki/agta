from dataclasses import dataclass, field
from agta.models import TripContext, TripDecision, TripRecord


@dataclass
class WorkingMemory:
    home_type: str = "home"
    car_location: str = "home"
    bicycle_location: str = "home"
    current_location: str = "home"
    trips_today: list = field(default_factory=list)

    def reset_day(self):
        self.current_location = self.home_type
        self.trips_today = []

    def available_modes(self, trip: TripContext) -> list[str]:
        available = []
        for option in trip.route_options:
            if option.mode == "car" and self.car_location != trip.from_activity:
                continue
            if option.mode == "bicycle" and self.bicycle_location != trip.from_activity:
                continue
            available.append(option.mode)
        return available

    def update_after_trip(self, decision: TripDecision, trip: TripContext):
        if decision.mode == "car":
            self.car_location = trip.to_activity
        elif decision.mode == "bicycle":
            self.bicycle_location = trip.to_activity
        self.current_location = trip.to_activity