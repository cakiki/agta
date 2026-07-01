import json
import logging

from agta.models import TripRecord
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

    start_day = 0
    resumed_day = load_checkpoint(model, output_dir)
    if resumed_day is not None:
        start_day = resumed_day
        model.current_day = start_day
        print(f"Resumed from checkpoint (day {resumed_day}). Starting at day {start_day}.")

    for day in tqdm(range(start_day, model.num_days), desc="Days"):
        model.step()
        save_checkpoint(model, output_dir)

    checkpoint_path = os.path.join(output_dir, "checkpoint.json")
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    return model

def save_checkpoint(model, output_dir: str):
    checkpoint = {
        "current_day": model.current_day,
        "agents": {}
    }
    for agent in model.agents:
        records = []
        for r in agent.memory.episodic.records:
            records.append({
                "day": r.day, "time": r.time,
                "from_activity": r.from_activity, "to_activity": r.to_activity,
                "from_location": list(r.from_location) if r.from_location else None,
                "to_location": list(r.to_location) if r.to_location else None,
                "mode": r.mode, "reasoning": r.reasoning,
                "distance_km": r.distance_km, "duration_min": r.duration_min,
                "vehicle_state_before": r.vehicle_state_before,
                "vehicle_state_after": r.vehicle_state_after,
                "available_options": r.available_options,
                "episodic_retrievals": r.episodic_retrievals,
                "picked_fastest": r.picked_fastest,
                "picked_shortest": r.picked_shortest,
            })
        checkpoint["agents"][agent.agent_id] = {
            "beliefs": agent.memory.semantic.beliefs,
            "rules": agent.memory.procedural.rules,
            "evaluations": agent.all_evaluations,
            "episodic_records": records,
            "car_location": agent.memory.working.car_location,
            "bicycle_location": agent.memory.working.bicycle_location,
        }
    path = os.path.join(output_dir, "checkpoint.json")
    with open(path, "w") as f:
        json.dump(checkpoint, f)

def load_checkpoint(model, output_dir: str) -> int | None:
    path = os.path.join(output_dir, "checkpoint.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        checkpoint = json.load(f)
    for agent in model.agents:
        state = checkpoint["agents"].get(str(agent.agent_id))
        if not state:
            continue
        agent.memory.semantic.beliefs = state["beliefs"]
        agent.memory.procedural.rules = state["rules"]
        agent.all_evaluations = state["evaluations"]
        agent.memory.working.car_location = state["car_location"]
        agent.memory.working.bicycle_location = state["bicycle_location"]
        for r in state["episodic_records"]:
            record = TripRecord(
                day=r["day"], time=r["time"],
                from_activity=r["from_activity"], to_activity=r["to_activity"],
                from_location=tuple(r["from_location"]) if r["from_location"] else (),
                to_location=tuple(r["to_location"]) if r["to_location"] else (),
                mode=r["mode"], reasoning=r["reasoning"],
                distance_km=r["distance_km"], duration_min=r["duration_min"],
                vehicle_state_before=r["vehicle_state_before"],
                vehicle_state_after=r["vehicle_state_after"],
                available_options=r["available_options"],
                episodic_retrievals=r["episodic_retrievals"],
                picked_fastest=r.get("picked_fastest", False),
                picked_shortest=r.get("picked_shortest", False),
            )
            agent.memory.episodic.add(record)
    return checkpoint["current_day"]

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