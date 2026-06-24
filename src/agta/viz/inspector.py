import streamlit as st
import json
import pandas as pd
from pyproj import Transformer

st.set_page_config(page_title="AGTA Inspector", layout="wide")

utm_to_latlon = Transformer.from_crs("EPSG:25833", "EPSG:4326", always_xy=True)

def to_latlon(coords):
    if not coords or len(coords) != 2:
        return None, None
    lon, lat = utm_to_latlon.transform(coords[0], coords[1])
    return lat, lon

uploaded = st.sidebar.file_uploader("Load simulation JSON", type="json")
if not uploaded:
    st.title("AGTA Inspector")
    st.info("Upload a simulation output JSON to begin.")
    st.stop()

data = json.load(uploaded)
meta = data.get("metadata", {})
config = meta.get("config", {})
agents = data.get("agents", {})

# Build dataframes
agent_rows = []
trip_rows = []
for agent_id, agent in agents.items():
    trips = agent.get("trips", [])
    modes = [t["chosen_mode"] for t in trips]
    dominant = max(set(modes), key=modes.count) if modes else None
    total_dist = sum(t.get("distance_km", 0) for t in trips)
    total_dur = sum(t.get("duration_min", 0) for t in trips)
    fallback_count = sum(1 for t in trips if "fallback" in t.get("reasoning", "").lower())
    agent_rows.append({
        "agent_id": agent_id,
        "trips": len(trips),
        "dominant_mode": dominant,
        "total_km": round(total_dist, 1),
        "total_min": round(total_dur, 1),
        "beliefs": len(agent.get("learned_beliefs", [])),
        "rules": len(agent.get("procedural_rules", [])),
        "fallbacks": fallback_count,
    })
    for t in trips:
        fastest = min(t.get("available_options", []), key=lambda o: o["duration_min"], default=None)
        shortest = min(t.get("available_options", []), key=lambda o: o["distance_km"], default=None)
        trip_rows.append({
            "agent_id": agent_id,
            "day": t.get("day", 0),
            "time": t.get("time", ""),
            "origin": t.get("origin", ""),
            "destination": t.get("destination", ""),
            "mode": t.get("chosen_mode", ""),
            "reasoning": t.get("reasoning", ""),
            "distance_km": t.get("distance_km", 0),
            "duration_min": t.get("duration_min", 0),
            "options_count": len(t.get("available_options", [])),
            "picked_fastest": t.get("chosen_mode") == fastest["mode"] if fastest else None,
            "picked_shortest": t.get("chosen_mode") == shortest["mode"] if shortest else None,
            "is_fallback": "fallback" in t.get("reasoning", "").lower(),
            "car_before": t.get("vehicle_locations_before", {}).get("car", ""),
            "bicycle_before": t.get("vehicle_locations_before", {}).get("bicycle", ""),
            "available_options": t.get("available_options", []),
            "episodic_retrievals": t.get("episodic_retrievals", []),
        })

agents_df = pd.DataFrame(agent_rows)
trips_df = pd.DataFrame(trip_rows)

# Sidebar nav
view = st.sidebar.radio("View", ["Overview", "Agents", "Agent Detail", "Trip Detail", "Comparison", "Map"])

# ── Overview ──
if view == "Overview":
    st.title("Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Agents", meta.get("num_agents", len(agents)))
    col2.metric("Days", meta.get("num_days", "?"))
    col3.metric("Total Trips", len(trips_df))
    col4.metric("Fallbacks", trips_df["is_fallback"].sum())

    st.subheader("Config")
    st.json(config)

    st.subheader("Modal Split")
    modal = trips_df["mode"].value_counts(normalize=True).mul(100).round(1)
    st.bar_chart(modal)

    st.subheader("MiD Comparison")
    mid = {"walk": 27.1, "bicycle": 15.2, "car": 31.6, "public transport": 26.0}
    compare = pd.DataFrame({
        "MiD": mid,
        "Simulation": {m: modal.get(m, 0) for m in mid},
    })
    st.dataframe(compare)

    rmse = ((compare["MiD"] - compare["Simulation"]) ** 2).mean() ** 0.5
    st.metric("RMSE", round(rmse, 2))

# ── Agents ──
elif view == "Agents":
    st.title("Agents")
    st.dataframe(
        agents_df.sort_values("agent_id"),
        use_container_width=True,
        hide_index=True,
    )

# ── Agent Detail ──
elif view == "Agent Detail":
    st.title("Agent Detail")
    agent_id = st.sidebar.selectbox("Select Agent", list(agents.keys()))
    agent = agents[agent_id]

    st.subheader(f"Agent {agent_id}")

    with st.expander("Persona", expanded=True):
        st.write(agent.get("persona", ""))

    col1, col2 = st.columns(2)
    with col1:
        with st.expander(f"Beliefs ({len(agent.get('learned_beliefs', []))})"):
            for b in agent.get("learned_beliefs", []):
                st.write(f"- {b}")
    with col2:
        with st.expander(f"Rules ({len(agent.get('procedural_rules', []))})"):
            for r in agent.get("procedural_rules", []):
                st.write(f"- {r}")

    st.subheader("Mode Breakdown")
    agent_trips = trips_df[trips_df["agent_id"] == agent_id]
    mode_counts = agent_trips["mode"].value_counts()
    st.bar_chart(mode_counts)

    st.subheader("Trips")
    for _, trip in agent_trips.iterrows():
        fastest_label = " ⚡" if trip["picked_fastest"] else ""
        fallback_label = " ⚠️ FALLBACK" if trip["is_fallback"] else ""
        with st.expander(f"Day {trip['day']} {trip['time']} | {trip['origin']} → {trip['destination']} | {trip['mode']}{fastest_label}{fallback_label}"):
            st.write(f"**Reasoning:** {trip['reasoning']}")
            st.write(f"**Distance:** {trip['distance_km']}km | **Duration:** {trip['duration_min']}min")
            st.write(f"**Picked fastest:** {trip['picked_fastest']} | **Picked shortest:** {trip['picked_shortest']}")
            st.write(f"**Vehicles before:** car={trip['car_before']}, bicycle={trip['bicycle_before']}")
            st.write("**Available options:**")
            for opt in trip["available_options"]:
                marker = "→ " if opt["mode"] == trip["mode"] else "  "
                st.write(f"{marker}{opt['mode']}: {opt['distance_km']}km, {opt['duration_min']}min")
            if trip["episodic_retrievals"]:
                st.write("**Episodic retrievals:**")
                for r in trip["episodic_retrievals"]:
                    if r.strip():
                        st.write(f"  - {r.strip()}")

# ── Trip Detail ──
elif view == "Trip Detail":
    st.title("All Trips")

    mode_filter = st.sidebar.multiselect("Filter by mode", trips_df["mode"].unique().tolist(), default=trips_df["mode"].unique().tolist())
    fallback_only = st.sidebar.checkbox("Fallback trips only")
    non_fastest = st.sidebar.checkbox("Non-fastest picks only")

    filtered = trips_df[trips_df["mode"].isin(mode_filter)]
    if fallback_only:
        filtered = filtered[filtered["is_fallback"]]
    if non_fastest:
        filtered = filtered[filtered["picked_fastest"] == False]

    st.write(f"{len(filtered)} trips")
    st.dataframe(
        filtered[["agent_id", "day", "time", "origin", "destination", "mode", "distance_km", "duration_min", "picked_fastest", "is_fallback", "reasoning"]],
        use_container_width=True,
        hide_index=True,
    )

# ── Comparison ──
elif view == "Comparison":
    st.title("Agent Comparison")

    st.subheader("Modal Split per Agent")
    pivot = trips_df.groupby(["agent_id", "mode"]).size().unstack(fill_value=0)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0).mul(100).round(1)
    st.bar_chart(pivot_pct)

    st.subheader("Distance per Agent")
    dist_by_agent = trips_df.groupby("agent_id")["distance_km"].sum().round(1)
    st.bar_chart(dist_by_agent)

    st.subheader("Monoculture Detection")
    for agent_id in agents:
        agent_trips = trips_df[trips_df["agent_id"] == agent_id]
        modes_used = agent_trips["mode"].nunique()
        dominant = agent_trips["mode"].value_counts().iloc[0] / len(agent_trips) * 100
        if dominant > 90:
            st.warning(f"Agent {agent_id}: {dominant:.0f}% {agent_trips['mode'].value_counts().index[0]} ({modes_used} modes used)")

# ── Map ──
elif view == "Map":
    st.title("Trip Map")
    import folium
    from streamlit_folium import st_folium

    MODE_COLORS = {
        "walk": "#2ecc71",
        "bicycle": "#f39c12",
        "car": "#e74c3c",
        "public transport": "#3498db",
    }

    agent_filter = st.sidebar.selectbox("Agent", ["All"] + list(agents.keys()))
    day_filter = st.sidebar.selectbox("Day", ["All"] + sorted(trips_df["day"].unique().tolist()))

    filtered = trips_df.copy()
    if agent_filter != "All":
        filtered = filtered[filtered["agent_id"] == agent_filter]
    if day_filter != "All":
        filtered = filtered[filtered["day"] == day_filter]

    # Collect map points
    points = []
    for _, row in filtered.iterrows():
        agent = agents[row["agent_id"]]
        agent_trips = agent.get("trips", [])
        trip = next((t for t in agent_trips if t.get("time") == row["time"] and t.get("day") == row["day"]), None)
        if not trip:
            continue
        orig_lat, orig_lon = to_latlon(trip.get("origin_coordinates"))
        dest_lat, dest_lon = to_latlon(trip.get("destination_coordinates"))
        if orig_lat and dest_lat:
            points.append({
                "agent_id": row["agent_id"],
                "time": row["time"],
                "origin": row["origin"],
                "destination": row["destination"],
                "mode": row["mode"],
                "reasoning": row["reasoning"],
                "distance_km": row["distance_km"],
                "duration_min": row["duration_min"],
                "orig_lat": orig_lat,
                "orig_lon": orig_lon,
                "dest_lat": dest_lat,
                "dest_lon": dest_lon,
            })

    if not points:
        st.warning("No trips with coordinates found.")
        st.stop()

    center_lat = sum(p["orig_lat"] for p in points) / len(points)
    center_lon = sum(p["orig_lon"] for p in points) / len(points)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

    for p in points:
        color = MODE_COLORS.get(p["mode"], "#999999")
        folium.PolyLine(
            [[p["orig_lat"], p["orig_lon"]], [p["dest_lat"], p["dest_lon"]]],
            color=color,
            weight=3,
            opacity=0.7,
            popup=f"Agent {p['agent_id']}<br>{p['time']} {p['origin']} → {p['destination']}<br>{p['mode']} ({p['distance_km']}km, {p['duration_min']}min)<br>{p['reasoning']}",
        ).add_to(m)
        folium.CircleMarker(
            [p["orig_lat"], p["orig_lon"]],
            radius=5,
            color=color,
            fill=True,
            popup=f"{p['origin']}",
        ).add_to(m)
        folium.CircleMarker(
            [p["dest_lat"], p["dest_lon"]],
            radius=5,
            color=color,
            fill=True,
            popup=f"{p['destination']}",
        ).add_to(m)

    st.write(f"{len(points)} trips displayed")
    legend_cols = st.columns(len(MODE_COLORS))
    for col, (mode, color) in zip(legend_cols, MODE_COLORS.items()):
        col.markdown(f"<span style='color:{color}'>●</span> {mode}", unsafe_allow_html=True)

    st_folium(m, width=None, height=600)