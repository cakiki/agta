from agta.loader import load_from_json
from agta.simulation import BerlinMobilityModel


def run(data_path, llm_model, num_days=1, limit=None):
    agents_data, routes_data = load_from_json(data_path, limit=limit)
    model = BerlinMobilityModel(agents_data, routes_data, llm_model, num_days)

    for day in range(model.num_days):
        model.current_day = day

        for agent in model.agents:
            agent.memory.working.reset_day()

        for agent in model.agents:
            for trip in routes_data[agent.agent_id]:
                agent.decide_trip(trip, day=day)

    return model


if __name__ == "__main__":
    model = run(
        data_path="data/agents_4a_route_options.json",
        llm_model="huggingface/openai/gpt-oss-120b",
        limit=10,
    )