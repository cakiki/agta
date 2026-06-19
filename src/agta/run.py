from agta.loader import load_from_json
from agta.simulation import MobilityModel
from agta.eval import evaluate
from agta.output import save_results

def run(data_path, llm_model, num_days=1, limit=None):
    agents_data, routes_data = load_from_json(data_path, limit=limit)
    model = MobilityModel(agents_data, routes_data, llm_model, num_days)

    for day in range(model.num_days):
        model.current_day = day

        for agent in model.agents:
            agent.memory.working.reset_day()

        for agent in model.agents:
            for trip in routes_data[agent.agent_id]:
                agent.decide_trip(trip, day=day)
        for agent in model.agents:
            print(f"\nAgent {agent.agent_id}:")
            for t in agent.memory.working.trips_today:
                print(f"  {t.time} {t.from_activity} -> {t.to_activity}: {t.mode} ({t.reasoning})")
        for agent in model.agents:
            agent.reflect(day=day)
            if len(agent.memory.semantic.beliefs) >= model.belief_consolidation_threshold:
                agent.consolidate_beliefs()
    return model


if __name__ == "__main__":
    model = run(
        data_path="data/agents_4a_route_options.json",
        llm_model="huggingface/Qwen/Qwen2.5-7B-Instruct",
        limit=1,
        num_days=2,
    )
    for agent in model.agents:
        print(f"\nAgent {agent.agent_id}:")
        print(f"  Episodic records: {len(agent.memory.episodic.records)}")
        print(f"  Beliefs: {agent.memory.semantic.beliefs}")
        for r in agent.memory.episodic.records:
            print(f"  Day {r.day}, {r.time}: {r.from_activity} -> {r.to_activity}: {r.mode}")
    save_results(model)
