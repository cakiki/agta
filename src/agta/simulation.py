import mesa
from agta.agent import MobilityAgent


class MobilityModel(mesa.Model):
    def __init__(self, agents_data, routes_data, llm_model, num_days=1, **kwargs):
        self.verbose = kwargs.pop("verbose", False)
        self.belief_consolidation_threshold = kwargs.pop("belief_consolidation_threshold", 10)
        super().__init__(**kwargs)
        self.current_day = 0
        self.num_days = num_days
        self.routes_data = routes_data
        self.results = []

        for agent_id, data in agents_data.items():
            MobilityAgent(
                model=self,
                agent_id=agent_id,
                persona=data["persona"],
                seed=data["seed"],
                schedule=data["schedule"],
                attitudes=data.get("attitudes"),
                llm_model=llm_model,
            )