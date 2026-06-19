from agta.util import extract_json_from
from mesa_llm.llm_agent import LLMAgent
from agta.models import TripContext, TripDecision, TripRecord, RouteOption
from agta.memory.memory_manager import MemoryManager
from agta.prompt.mode_choice import build_trip_prompt
from mesa_llm.reasoning.reasoning import Reasoning as BaseReasoning, Plan

class NoOpReasoning(BaseReasoning):
    def plan(self, prompt=None, obs=None, ttl=1, selected_tools=None, tool_calls="auto"):
        return Plan(step=0, llm_plan=None, ttl=ttl)

class MobilityAgent(LLMAgent):
    def __init__(self, model, agent_id, persona, seed, schedule, attitudes=None, **kwargs):
        super().__init__(model=model, reasoning=NoOpReasoning, **kwargs)
        self.agent_id = agent_id
        self.persona = persona
        self.seed = seed
        self.schedule = schedule
        self.attitudes = attitudes
        self.memory = MemoryManager(agent=self)
        
    def reflect(self, day: int):
        from agta.prompt.reflection import build_reflection_prompt
        prompt = build_reflection_prompt(self.persona, self.memory, day)
        if not prompt:
            return
        response = self.llm.generate(prompt)
        text = response.choices[0].message.content
        parsed = extract_json_from(text)
        for belief in parsed.get("beliefs", []):
            self.memory.semantic.add_belief(belief)
    

    def consolidate_beliefs(self):
        if len(self.memory.semantic.beliefs) < 5:
            return
        prompt = "Here are transport beliefs about a person:\n"
        for b in self.memory.semantic.beliefs:
            prompt += f"  - {b}\n"
        prompt += "\nMerge duplicates and remove redundancies. Return a consolidated list.\n"
        prompt += 'Return JSON: {"beliefs": ["belief 1", "belief 2", ...]}'
        response = self.llm.generate(prompt)
        text = response.choices[0].message.content
        parsed = extract_json_from(text)
        self.memory.semantic.beliefs = parsed.get("beliefs", self.memory.semantic.beliefs)

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

        if self.model.verbose:
            print("=== PROMPT FOR TRIP", len(self.memory.working.trips_today) + 1, "===")
            print(prompt)
            print("=== END ===")
    
        response = self.llm.generate(prompt)
        text = response.choices[0].message.content
        parsed = extract_json_from(text)

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
            available_options=[{"mode": o.mode, "distance_km": o.distance_km, "duration_min": o.duration_min} for o in available],
        )
        self.memory.working.trips_today.append(record)
        self.memory.episodic.add(record)
        return decision
    