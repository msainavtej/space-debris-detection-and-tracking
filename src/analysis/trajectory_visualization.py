import pandas as pd
import plotly.graph_objects as go
import os

# =========================
# Configuration
# =========================

INPUT_CSV = "outputs/tracking/tracking_results.csv"

OUTPUT_DIR = "outputs/analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_HTML = os.path.join(
    OUTPUT_DIR,
    "trajectory_plot.html"
)

# =========================
# Load Data
# =========================

df = pd.read_csv(INPUT_CSV)

# =========================
# Plot Trajectories
# =========================

fig = go.Figure()

for object_id in sorted(df["object_id"].unique()):

    obj_df = df[df["object_id"] == object_id]

    fig.add_trace(
        go.Scatter(
            x=obj_df["x_center"],
            y=obj_df["y_center"],
            mode="lines+markers",
            name=f"Object {object_id}"
        )
    )

fig.update_layout(
    title="Space Debris Trajectories",
    xaxis_title="X Position",
    yaxis_title="Y Position",
    height=700
)

# Make origin feel image-like
fig.update_yaxes(autorange="reversed")

fig.write_html(OUTPUT_HTML)

print(f"Saved: {OUTPUT_HTML}")
