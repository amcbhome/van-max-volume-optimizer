import pandas as pd
import pulp
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Max Volume Fleet Optimizer", page_icon="📦", layout="wide"
)

st.title("📦 Fleet Volume Maximization Optimizer")
st.write(
    "Select the vehicle with the largest cargo volume capacity while satisfying your required payload weight constraint."
)


# Fleet Dataset
@st.cache_data
def get_fleet_data():
    data = [
        {
            "model": "Ford Transit Custom",
            "cat": "M",
            "vol": 6.8,
            "weight": 1327,
            "cost": 0.173,
        },
        {
            "model": "Vauxhall Vivaro",
            "cat": "M",
            "vol": 6.6,
            "weight": 1400,
            "cost": 0.159,
        },
        {
            "model": "Volkswagen Transporter",
            "cat": "M",
            "vol": 6.7,
            "weight": 1280,
            "cost": 0.176,
        },
        {
            "model": "Renault Trafic",
            "cat": "M",
            "vol": 6.7,
            "weight": 1251,
            "cost": 0.164,
        },
        {
            "model": "Toyota Proace",
            "cat": "M",
            "vol": 6.6,
            "weight": 1333,
            "cost": 0.160,
        },
        {
            "model": "Mercedes-Benz Sprinter",
            "cat": "L",
            "vol": 14.0,
            "weight": 1380,
            "cost": 0.213,
        },
        {
            "model": "Ford Transit Full-Size",
            "cat": "L",
            "vol": 15.1,
            "weight": 1450,
            "cost": 0.203,
        },
        {
            "model": "Peugeot Boxer",
            "cat": "L",
            "vol": 13.0,
            "weight": 1570,
            "cost": 0.193,
        },
        {
            "model": "Volkswagen Crafter",
            "cat": "L",
            "vol": 14.4,
            "weight": 1354,
            "cost": 0.206,
        },
        {
            "model": "Iveco Daily",
            "cat": "L",
            "vol": 16.0,
            "weight": 1750,
            "cost": 0.231,
        },
        {
            "model": "Citroën Berlingo",
            "cat": "S",
            "vol": 3.8,
            "weight": 1000,
            "cost": 0.126,
        },
        {
            "model": "Peugeot Partner",
            "cat": "S",
            "vol": 3.8,
            "weight": 987,
            "cost": 0.127,
        },
        {
            "model": "Volkswagen Caddy",
            "cat": "S",
            "vol": 3.7,
            "weight": 780,
            "cost": 0.135,
        },
        {
            "model": "Ford Transit Connect",
            "cat": "S",
            "vol": 3.6,
            "weight": 865,
            "cost": 0.132,
        },
        {
            "model": "Renault Kangoo",
            "cat": "S",
            "vol": 3.9,
            "weight": 980,
            "cost": 0.128,
        },
        {
            "model": "Ford E-Transit LWB",
            "cat": "E",
            "vol": 11.0,
            "weight": 1758,
            "cost": 0.119,
        },
        {
            "model": "Vauxhall Vivaro Electric",
            "cat": "E",
            "vol": 6.6,
            "weight": 1263,
            "cost": 0.102,
        },
        {
            "model": "Mercedes-Benz eSprinter",
            "cat": "E",
            "vol": 11.0,
            "weight": 1045,
            "cost": 0.128,
        },
        {
            "model": "Maxus eDeliver 9",
            "cat": "E",
            "vol": 11.0,
            "weight": 1200,
            "cost": 0.134,
        },
        {
            "model": "Nissan Townstar EV",
            "cat": "E",
            "vol": 3.9,
            "weight": 800,
            "cost": 0.069,
        },
    ]
    return pd.DataFrame(data)


df = get_fleet_data()

# User Input Controls
st.sidebar.header("Payload Requirements")
req_vol = st.sidebar.number_input(
    "Required Minimum Volume (m³)",
    min_value=0.1,
    max_value=20.0,
    value=5.0,
    step=0.1,
)
req_weight = st.sidebar.number_input(
    "Required Minimum Payload Weight (kg)",
    min_value=1,
    max_value=2000,
    value=1000,
    step=25,
)


# Volume Maximization Solver
def solve_max_volume(data, target_vol, target_weight):
    # Set objective to LpMaximize
    model = pulp.LpProblem("MaxVolumeSelection", pulp.LpMaximize)

    y = {
        row["model"]: pulp.LpVariable(f"y_{i}", cat=pulp.LpBinary)
        for i, row in data.iterrows()
    }

    # Objective Function: Maximize vehicle volume capacity
    model += pulp.lpSum(
        [row["vol"] * y[row["model"]] for _, row in data.iterrows()]
    )

    # Constraint 1: Select exactly ONE van
    model += pulp.lpSum([y[m] for m in y]) == 1, "SingleSelection"

    # Constraint 2: Lower bound weight requirement
    model += (
        pulp.lpSum(
            [row["weight"] * y[row["model"]] for _, row in data.iterrows()]
        )
        >= target_weight,
        "WeightCap",
    )

    # Constraint 3: Lower bound volume requirement
    model += (
        pulp.lpSum(
            [row["vol"] * y[row["model"]] for _, row in data.iterrows()]
        )
        >= target_vol,
        "VolCap",
    )

    model.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[model.status] == "Optimal":
        for _, row in data.iterrows():
            if y[row["model"]].varValue == 1:
                return row
    return None


# Execute Optimization
selected = solve_max_volume(df, req_vol, req_weight)

# Results Display
if selected is not None:
    st.success(f"**Max Volume Vehicle Selected:** {selected['model']}")

    # Utilization Metrics
    vol_util = (req_vol / selected["vol"]) * 100
    weight_util = (req_weight / selected["weight"]) * 100

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Category", selected["cat"])
    col2.metric("Volume Capacity", f"{selected['vol']} m³")
    col3.metric("Volume Utilisation", f"{vol_util:.1f}%")
    col4.metric("Payload Capacity", f"{selected['weight']} kg")
    col5.metric("Payload Utilisation", f"{weight_util:.1f}%")
    col6.metric("Cost Rate", f"£{selected['cost']:.3f} / mi")

    st.subheader("Fleet Capacity Breakdown")
    st.dataframe(df, use_container_width=True)

else:
    st.error(
        "No single vehicle in the fleet meets both the requested weight and volume parameters."
    )
