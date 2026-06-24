import json
from datetime import datetime


def save_results(model, config=None, path="results/simulation_output.json"):
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_agents": len(model.agents),
            "num_days": model.num_days,
            "config": config,
        },
        "agents": {}
    }

    for agent in model.agents:
        output["agents"][str(agent.agent_id)] = {
            "persona": agent.persona,
            "transport_attitudes": agent.memory.semantic.attitudes,
            "learned_beliefs": agent.memory.semantic.beliefs,
            "procedural_rules": agent.memory.procedural.rules,
            "evaluations": agent.all_evaluations,
            "trips": [
                {
                    "day": r.day,
                    "time": r.time,
                    "origin": r.from_activity,
                    "destination": r.to_activity,
                    "origin_coordinates": r.from_location,
                    "destination_coordinates": r.to_location,
                    "chosen_mode": r.mode,
                    "reasoning": r.reasoning,
                    "distance_km": r.distance_km,
                    "duration_min": r.duration_min,
                    "available_options": r.available_options,
                    "vehicle_locations_before": r.vehicle_state_before,
                    "vehicle_locations_after": r.vehicle_state_after,
                    "episodic_retrievals": r.episodic_retrievals or [],
                }
                for r in agent.memory.episodic.records
            ],
        }

    with open(path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {path}")
