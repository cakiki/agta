import logging
logging.getLogger("mesa_llm.module_llm").setLevel(logging.ERROR)
logging.getLogger("searcharray").setLevel(logging.ERROR)
logging.getLogger("searcharray.indexing").setLevel(logging.ERROR)
logging.getLogger("searcharray.indexing").handlers = []
logging.getLogger("searcharray.indexing").propagate = False
logging.getLogger("searcharray").disabled = True

import yaml
import sys
from tqdm import tqdm
from agta.loader import load_from_json
from agta.simulation import MobilityModel
from agta.eval import evaluate
from agta.output import save_results
from agta.retrieval.dense import set_embedding_model
import os
from datetime import datetime

def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run(config: dict, output_dir: str):
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

    model.llm_model = config["simulation"]["llm_model"]
    model.prompt_log_path = os.path.join(output_dir, "llm_log.jsonl")
    model.output_path = os.path.join(output_dir, "simulation_output.json")

    for _ in tqdm(range(model.num_days), desc="Days"):
        model.step()

    return model


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    config = load_config(config_path)
    output_dir = config["output"].get("dir")
    if not output_dir:
        output_dir = f"results/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    set_embedding_model(config["retrieval"]["embedding_model"])
    model = run(config, output_dir)
    evaluate(model.agents)
    print(model.datacollector.get_model_vars_dataframe())
    print(model.datacollector.get_agent_vars_dataframe())
    save_results(model, config=config, path=model.output_path)