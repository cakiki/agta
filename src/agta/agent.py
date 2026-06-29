from agta.util import extract_json_from
from mesa_llm.llm_agent import LLMAgent
from agta.models import TripContext, TripDecision, TripRecord, RouteOption
from agta.memory.memory_manager import MemoryManager
from agta.prompt.mode_choice import build_trip_prompt
from mesa_llm.reasoning.reasoning import Reasoning as BaseReasoning, Plan
import logging
from functools import wraps

from litellm.exceptions import APIError, BadRequestError, APIConnectionError

def try_llm_call(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except (APIError, BadRequestError, APIConnectionError) as e:
            logging.warning(f"{func.__name__} failed for agent {self.agent_id}: {e}. Skipping.")
    return wrapper

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
        self.all_evaluations = []

    def reset_day(self):
        self.memory.working.reset_day()

    def process_day_trips(self, day, routes_data):
        for trip in routes_data[self.agent_id]:
            self.decide_trip(trip, day=day)

    def end_of_day(self, day):
        self.evaluate_day(day=day)
        self.reflect(day=day)
        self.reflect_procedural(day=day)
        if len(self.memory.semantic.beliefs) >= self.model.belief_consolidation_threshold:
            self.consolidate_beliefs()  

    @try_llm_call
    def reflect(self, day: int):
        from agta.prompt.reflection import build_reflection_prompt
        evaluations = getattr(self, '_day_evaluations', [])
        prompt = build_reflection_prompt(self.persona, self.memory, day, evaluations=evaluations)
        if not prompt:
            return
        response = self.llm.generate(prompt)
        text = response.choices[0].message.content
        parsed = extract_json_from(text)
        for belief in parsed.get("beliefs", []):
            self.memory.semantic.add_belief(belief)
    
    @try_llm_call
    def reflect_procedural(self, day: int):
        from agta.prompt.reflection import build_procedural_reflection_prompt
        prompt = build_procedural_reflection_prompt(self.persona, self.memory, day)
        if not prompt:
            return
        response = self.llm.generate(prompt)
        text = response.choices[0].message.content
        parsed = extract_json_from(text)
        for rule in parsed.get("rules", []):
            self.memory.procedural.add_rule(rule)

    @try_llm_call
    def consolidate_beliefs(self):
        if len(self.memory.semantic.beliefs) < 5:
            return
        from agta.prompt.reflection import build_consolidation_prompt
        prompt = build_consolidation_prompt(self.memory.semantic.beliefs)
        response = self.llm.generate(prompt)
        text = response.choices[0].message.content
        parsed = extract_json_from(text)
        new_beliefs = parsed.get("beliefs", [])
        if new_beliefs:
            self.memory.semantic.beliefs = new_beliefs

    @try_llm_call
    def evaluate_day(self, day):
        from agta.prompt.reflection import build_evaluation_prompt
        prompt = build_evaluation_prompt(self.persona, self.memory, day)
        if not prompt:
            return
        response = self.llm.generate(prompt)
        text = response.choices[0].message.content
        parsed = extract_json_from(text)
        self._day_evaluations = parsed.get("evaluations", [])
        self.all_evaluations.extend(self._day_evaluations)

    def decide_trip(self, trip: TripContext, day: int = 0) -> TripDecision:
        available = [o for o in trip.route_options if o.mode in self.memory.working.available_modes(trip)]
        if not available:
            available = [RouteOption(mode="walk", distance_km=0.0, duration_min=0.0)]
        fastest = min(available, key=lambda o: o.duration_min)
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
    
        try:
            response = self.llm.generate(prompt)
            text = response.choices[0].message.content
            parsed = extract_json_from(text)
            mode = parsed["mode"].lower().strip()
            reasoning = parsed["reasoning"]
        except Exception as e:
            logging.warning(f"LLM call failed for agent {self.agent_id}: {e}. Using fallback.")
            fallback = min(available, key=lambda o: o.duration_min)
            mode = fallback.mode
            reasoning = "fallback: LLM unavailable"

        valid_modes = [o.mode for o in available]
        if mode not in valid_modes:
            fallback = min(available, key=lambda o: o.duration_min)
            reasoning = f"Fallback: LLM chose '{mode}' (unavailable). Using {fallback.mode}. Original: {reasoning}"
            mode = fallback.mode

        vehicle_state_before = {
            "car": self.memory.working.car_location,
            "bicycle": self.memory.working.bicycle_location,
        }

        decision = TripDecision(
            route_id=trip.route_id,
            mode=mode,
            reasoning=reasoning,
            vehicle_state=vehicle_state_before,
        )

        self.memory.working.update_after_trip(decision, trip)
        episodic_context = self.memory.get_episodic_context(filtered_trip.to_activity)
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
            episodic_retrievals=episodic_context.split("\n") if episodic_context else [],
            picked_fastest=mode == fastest.mode,
        )
        self.memory.working.trips_today.append(record)
        self.memory.episodic.add(record)
        return decision
    