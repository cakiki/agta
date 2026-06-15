import numpy as np


MID_BERLIN_MODAL_SPLIT = {
    "walk": 27.1,
    "bicycle": 15.2,
    "car": 31.6,
    "public transport": 26.0,
}


def compute_rmse(dict_a, dict_b):
    diffs = [(dict_a[k] - dict_b.get(k, 0)) ** 2 for k in dict_a]
    return round(np.sqrt(np.mean(diffs)), 2)


def modal_split(agents):
    counts = {}
    total = 0
    for agent in agents:
        for trip in agent.memory.working.trips_today:
            counts[trip.mode] = counts.get(trip.mode, 0) + 1
            total += 1
    if total == 0:
        return {}
    return {mode: round(count / total * 100, 1) for mode, count in counts.items()}


def evaluate(agents):
    sim_split = modal_split(agents)
    rmse = compute_rmse(MID_BERLIN_MODAL_SPLIT, sim_split)
    print(f"MiD:  {MID_BERLIN_MODAL_SPLIT}")
    print(f"Sim:  {sim_split}")
    print(f"RMSE: {rmse}")
    return rmse
