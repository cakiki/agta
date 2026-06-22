import logging
logging.getLogger("mesa_llm.module_llm").setLevel(logging.ERROR)
logging.getLogger("searcharray").setLevel(logging.ERROR)

from mesa.visualization import SolaraViz, make_plot_component
from agta.loader import load_from_json
from agta.simulation import MobilityModel
from agta.run import load_config
from agta.retrieval.dense import set_embedding_model

config = load_config()
set_embedding_model(config["retrieval"]["embedding_model"])
agents_data, routes_data = load_from_json(config["data"]["path"], limit=config["data"].get("limit"))

model = MobilityModel(
    agents_data, routes_data,
    config["simulation"]["llm_model"],
    config["simulation"]["num_days"],
    verbose=False,
    belief_consolidation_threshold=config["simulation"].get("belief_consolidation_threshold", 10),
    retrieval_config=config.get("retrieval", {}),
)

page = SolaraViz(
    model,
    components=[
        make_plot_component(["walk_pct", "bicycle_pct", "car_pct", "pt_pct"]),
        make_plot_component(["total_trips", "distinct_modes"]),
    ],
    name="AGTA Mobility Simulation",
)