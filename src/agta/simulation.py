import mesa
from agta.agent import MobilityAgent


class MobilityModel(mesa.Model):
    def __init__(self, agents_data, routes_data, llm_model, num_days=1, **kwargs):
        self.verbose = kwargs.pop("verbose", False)
        self.belief_consolidation_threshold = kwargs.pop("belief_consolidation_threshold", 10)
        self.retrieval_config = kwargs.pop("retrieval_config", {})
        super().__init__(**kwargs)
        self.current_day = 0
        self.num_days = num_days
        self.routes_data = routes_data
        self.results = []
        self.llm_model = ""
        self.prompt_log_path = ""
        self.output_path = ""
        

        for agent_id, data in agents_data.items():
            agent = MobilityAgent(
                model=self,
                agent_id=agent_id,
                persona=data["persona"],
                seed=data["seed"],
                schedule=data["schedule"],
                attitudes=data.get("attitudes"),
                llm_model=llm_model,
            )
            home_type = data.get("home_type", "home")
            agent.memory.working.home_type = home_type
            agent.memory.working.car_location = home_type
            agent.memory.working.bicycle_location = home_type
            agent.memory.working.current_location = home_type
            
        self.datacollector = mesa.DataCollector(
        model_reporters={
            "day": lambda m: m.current_day,
            "walk_pct": lambda m: MobilityModel._modal_pct(m, "walk"),
            "bicycle_pct": lambda m: MobilityModel._modal_pct(m, "bicycle"),
            "car_pct": lambda m: MobilityModel._modal_pct(m, "car"),
            "pt_pct": lambda m: MobilityModel._modal_pct(m, "public transport"),
            "total_trips": lambda m: sum(len(a.memory.working.trips_today) for a in m.agents),
            "distinct_modes": lambda m: len({t.mode for a in m.agents for t in a.memory.working.trips_today}),
            "avg_trip_distance": lambda m: round(sum(t.distance_km for a in m.agents for t in a.memory.working.trips_today) / max(1, sum(len(a.memory.working.trips_today) for a in m.agents)), 2),
        },
        agent_reporters={
            "agent_id": lambda a: a.agent_id,
            "trips_today": lambda a: len(a.memory.working.trips_today),
            "dominant_mode": lambda a: max(set(t.mode for t in a.memory.working.trips_today), key=lambda mode: sum(1 for t in a.memory.working.trips_today if t.mode == mode)) if a.memory.working.trips_today else None,
            "total_distance_km": lambda a: round(sum(t.distance_km for t in a.memory.working.trips_today), 2),
            "total_duration_min": lambda a: round(sum(t.duration_min for t in a.memory.working.trips_today), 1),
            "beliefs_count": lambda a: len(a.memory.semantic.beliefs),
            "rules_count": lambda a: len(a.memory.procedural.rules),
            "episodic_count": lambda a: len(a.memory.episodic.records),
        },
    )
    @staticmethod
    def _modal_pct(model, mode):
        total = sum(len(a.memory.working.trips_today) for a in model.agents)
        if total == 0:
            return 0.0
        count = sum(1 for a in model.agents for t in a.memory.working.trips_today if t.mode == mode)
        return round(count / total * 100, 1)
    
    def step(self):
        self.agents.do("reset_day")
        self.agents.do("process_day_trips", day=self.current_day, routes_data=self.routes_data)
        if self.verbose:
            for agent in self.agents:
                print(f"\nAgent {agent.agent_id}:")
                for t in agent.memory.working.trips_today:
                    print(f"  {t.time} {t.from_activity} -> {t.to_activity}: {t.mode} ({t.reasoning})")
        self.datacollector.collect(self)
        self.agents.do("end_of_day", day=self.current_day)
        self.current_day += 1