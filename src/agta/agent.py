import json
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.reasoning import Reasoning
from agta.models import TripContext, TripDecision, TripRecord, RouteOption
from agta.memory.memory_manager import MemoryManager
from agta.prompt.mode_choice import build_trip_prompt


class MobilityAgent(LLMAgent):
    def __init__(self, model, agent_id, persona, seed, schedule, attitudes=None, **kwargs):
        super().__init__(model=model, reasoning=Reasoning, **kwargs)
        self.agent_id = agent_id
        self.persona = persona
        self.seed = seed
        self.schedule = schedule
        self.attitudes = attitudes
        self.memory = MemoryManager(agent=self)

    def decide_trip(self, trip: TripContext, day: int = 0) -> TripDecision:
        available = [o for o in trip.route_options if o.mode in self.memory.working.available_modes(trip)]
        if not available:
            available = [RouteOption(mode="walk", distance_km=0.0, duration_min=0.0)]

        filtered_trip = TripContext(
            route_id=trip.route_id,
            from_activity=trip.from_activity,
            to_activity=trip.to_activity,
            from_time=trip.from_time,
            to_time=trip.to_time,
            from_location=trip.from_location,
            to_location=trip.to_location,
            route_options=available,
            weather=trip.weather,
        )

        prompt = build_trip_prompt(
            persona=self.persona,
            memory=self.memory,
            trip=filtered_trip,
        )

        response = self.llm.generate(prompt)
        text = response.choices[0].message.content
        parsed = json.loads(text)

        vehicle_state_before = {
            "car": self.memory.working.car_location,
            "bicycle": self.memory.working.bicycle_location,
        }

        decision = TripDecision(
            route_id=trip.route_id,
            mode=parsed["mode"],
            reasoning=parsed["reasoning"],
            vehicle_state=vehicle_state_before,
        )

        self.memory.working.update_after_trip(decision, trip)

        chosen_option = next((o for o in available if o.mode == decision.mode), available[0])
        record = TripRecord(
            day=day,
            time=trip.from_time,
            from_activity=trip.from_activity,
            to_activity=trip.to_activity,
            from_location=trip.from_location,
            to_location=trip.to_location,
            mode=decision.mode,
            reasoning=decision.reasoning,
            distance_km=chosen_option.distance_km,
            duration_min=chosen_option.duration_min,
            vehicle_state_before=vehicle_state_before,
            vehicle_state_after={
                "car": self.memory.working.car_location,
                "bicycle": self.memory.working.bicycle_location,
            },
        )
        self.memory.working.trips_today.append(record)

        return decision