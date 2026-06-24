import logging
logging.getLogger("mesa_llm.module_llm").setLevel(logging.ERROR)
logging.getLogger("searcharray").setLevel(logging.ERROR)
logging.getLogger("searcharray.indexing").setLevel(logging.ERROR)
logging.getLogger("searcharray.indexing").handlers = []

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
        seed=config["data"].get("seed"),
    )
    model = MobilityModel(
        agents_data, routes_data,
        config["simulation"]["llm_model"],
        config["simulation"]["num_days"],
        verbose=config["simulation"].get("verbose", False),
        belief_consolidation_threshold=config["simulation"].get("belief_consolidation_threshold", 10),
        retrieval_config=config.get("retrieval", {}),
    )

    for _ in tqdm(range(model.num_days), desc="Days"):
        model.step()

    return model


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    config = load_config(config_path)
    set_embedding_model(config["retrieval"]["embedding_model"])
    model = run(config)
    evaluate(model.agents)
    print(model.datacollector.get_model_vars_dataframe())
    print(model.datacollector.get_agent_vars_dataframe())
    save_results(model, config=config, path=config["output"]["path"])