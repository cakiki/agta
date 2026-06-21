import logging
logging.getLogger("mesa_llm.module_llm").setLevel(logging.ERROR)

import yaml
import sys
from tqdm import tqdm
from agta.loader import load_from_json
from agta.simulation import MobilityModel
from agta.eval import evaluate
from agta.output import save_results
from agta.retrieval.dense import set_embedding_model



def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run(config: dict):
    agents_data, routes_data = load_from_json(
        config["data"]["path"],
        limit=config["data"].get("limit"),
    )
    model = MobilityModel(
        agents_data, routes_data,
        config["simulation"]["llm_model"],
        config["simulation"]["num_days"],
        verbose=config["simulation"].get("verbose", False),
        belief_consolidation_threshold=config["simulation"].get("belief_consolidation_threshold", 10),
        retrieval_config=config.get("retrieval", {}),
    )

    for day in tqdm(range(model.num_days), desc="Days"):
        model.current_day = day

        for agent in model.agents:
            agent.memory.working.reset_day()

        for agent in tqdm(list(model.agents), desc=f"Day {day} trips", leave=True):
            for trip in routes_data[agent.agent_id]:
                agent.decide_trip(trip, day=day)

        if model.verbose:
            for agent in model.agents:
                print(f"\nAgent {agent.agent_id}:")
                for t in agent.memory.working.trips_today:
                    print(f"  {t.time} {t.from_activity} -> {t.to_activity}: {t.mode} ({t.reasoning})")

        for agent in tqdm(list(model.agents), desc=f"Day {day} reflection", leave=True):
            agent.reflect(day=day)
            agent.reflect_procedural(day=day)
            if len(agent.memory.semantic.beliefs) >= model.belief_consolidation_threshold:
                agent.consolidate_beliefs()

    return model


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    config = load_config(config_path)
    set_embedding_model(config["retrieval"]["embedding_model"])
    model = run(config)
    evaluate(model.agents)
    save_results(model, config=config, path=config["output"]["path"])