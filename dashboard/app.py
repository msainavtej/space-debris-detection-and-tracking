import os
import subprocess
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import gradio as gr

# =========================
# Paths & Setup
# =========================
TEMP_INPUT_PATH = "data/processed/temp_dashboard_input.mp4"
OUTPUT_DIR = "reports/dashboard_outputs"
TRACKED_VIDEO_PATH = os.path.join(OUTPUT_DIR, "tracked_video.mp4")
TRACKING_CSV_PATH = os.path.join(OUTPUT_DIR, "tracking_results.csv")
VELOCITY_CSV_PATH = os.path.join(OUTPUT_DIR, "velocity_analysis.csv")

os.makedirs("data/processed", exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# Helper Functions
# =========================

def generate_plotly_trajectory(tracking_csv):
    """Parses tracking CSV data and builds an interactive Plotly trajectory map."""
    if not os.path.exists(tracking_csv) or os.path.getsize(tracking_csv) == 0:
        # Return blank placeholder figure if empty
        fig = go.Figure()
        fig.update_layout(title="No Trajectory Data Found", template="plotly_dark")
        return fig

    df = pd.read_csv(tracking_csv)
    
    # Ensure standard expected schema
    # Expected columns: frame, id, x, y (and optionally velocity)
    if not all(
        col in df.columns
        for col in [
            "frame",
            "object_id",
            "x_center",
            "y_center"
        ]
    ):
        fig = go.Figure()
        fig.update_layout(title="Invalid CSV Schema", template="plotly_dark")
        return fig

    # Build interactive trajectory lines plot
    fig = px.line(
        df,
        x="x_center",
        y="y_center",
        color="object_id",
        line_group="object_id",
        hover_data=["frame", "object_id"],
        markers=True,
        title="Orbital Debris Trajectories (Interactive Spatial Map)",
        labels={"x": "X Coordinate (px)", "y": "Y Coordinate (px)", "id": "Debris ID"}
    )

    # Invert Y-axis to match Computer Vision camera pixel conventions (0,0 at top-left)
    fig.update_yaxes(autorange="reversed")
    
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#111827",
        paper_bgcolor="#111827",
        font=dict(color="#F3F4F6"),
        height=500,
        legend_title_text="Tracked Objects"
    )
    
    return fig


def process_video_pipeline(video_file):
    """Orchestrates pipeline script execution upon user upload."""
    if video_file is None:
        return None, "0", "0", "0.0 px/f", "0.0 px/f", go.Figure()

    # 1. Save uploaded file to deterministic location for scripts to consume
    # Handle Gradio file object path
    input_src = video_file if isinstance(video_file, str) else video_file.name
    
    # Copy or re-save input video
    with open(input_src, "rb") as f_in:
        with open(TEMP_INPUT_PATH, "wb") as f_out:
            f_out.write(f_in.read())

    print(f"📥 Received video input: {TEMP_INPUT_PATH}")

    # 2. Sequential Orchestration Subprocess Execution
    try:
        print("⚙️ Running ByteTrack tracking engine...")
        subprocess.run([
            "python", "src/tracking/track_and_export.py",
            "--input", TEMP_INPUT_PATH,
            "--output_video", TRACKED_VIDEO_PATH,
            "--output_csv", TRACKING_CSV_PATH
        ], check=True)

        print("⚡ Running velocity & kinematics analysis...")
        subprocess.run([
            "python", "src/analysis/velocity_analysis.py",
            "--input_csv", TRACKING_CSV_PATH,
            "--output_csv", VELOCITY_CSV_PATH
        ], check=True)

    except subprocess.CalledProcessError as e:
        print(f"❌ Error during script execution: {e}")
        # Return fallback state if pipeline fails on custom inputs
        pass

    # 3. Calculate Telemetry Statistics from CSV Data
    total_objects = "0"
    total_frames = "0"
    avg_velocity = "0.0 px/f"
    max_velocity = "0.0 px/f"

    if os.path.exists(VELOCITY_CSV_PATH) and os.path.getsize(VELOCITY_CSV_PATH) > 0:
        v_df = pd.read_csv(VELOCITY_CSV_PATH)
        
        if "object_id" in v_df.columns:
            total_objects = str(
            v_df["object_id"].nunique()
            )
        
        if "velocity_px_per_frame" in v_df.columns:
            avg_v = v_df["velocity_px_per_frame"].mean()
            max_v = v_df["velocity_px_per_frame"].max()
            avg_velocity = f"{avg_v:.2f} px/f"
            max_velocity = f"{max_v:.2f} px/f"

    if os.path.exists(TRACKING_CSV_PATH) and os.path.getsize(TRACKING_CSV_PATH) > 0:
        t_df = pd.read_csv(TRACKING_CSV_PATH)
        if "frame" in t_df.columns:
            total_frames = str(t_df["frame"].nunique())

    # 4. Generate In-Memory Plotly Trajectory Graph
    fig = generate_plotly_trajectory(TRACKING_CSV_PATH)

    # Return elements matching Gradio UI outputs
    return TRACKED_VIDEO_PATH, total_objects, total_frames, avg_velocity, max_velocity, fig


# =========================
# Gradio UI Layout construction
# =========================

theme = gr.themes.Soft(
    primary_hue="cyan",
    neutral_hue="slate"
)

with gr.Blocks(theme=theme, title="Space Debris Tracking Dashboard") as demo:
    
    gr.Markdown(
        """
        # 🛰️ Orbital Space Debris Detection & Tracking Dashboard
        Upload a space domain video stream to run detection, Multi-Object Tracking (ByteTrack), kinematic analysis, and trajectory mapping.
        """
    )
    
    with gr.Row():
        # --- Left Column: Controls & Input ---
        with gr.Column(scale=1):
            gr.Markdown("### 1. Source Video Selection")
            video_input = gr.Video(label="Upload Raw Video (.mp4)", sources=["upload"])
            process_btn = gr.Button("🚀 Process Video Stream", variant="primary", size="lg")
            
            gr.Markdown("---")
            gr.Markdown("### 2. Live Telemetry & Kinematics")
            with gr.Row():
                stat_objects = gr.Textbox(label="Total Objects Tracked", value="0", interactive=False)
                stat_frames = gr.Textbox(label="Total Frames", value="0", interactive=False)
            with gr.Row():
                stat_avg_vel = gr.Textbox(label="Average Velocity", value="0.0 px/f", interactive=False)
                stat_max_vel = gr.Textbox(label="Maximum Velocity", value="0.0 px/f", interactive=False)

        # --- Right Column: Primary Visual Output ---
        with gr.Column(scale=2):
            gr.Markdown("### 3. Visual Detection & Tracking Stream")
            video_output = gr.Video(label="Annotated Tracking Output (YOLO + ByteTrack)", interactive=False)

    gr.Markdown("---")
    
    # --- Bottom Row: Interactive Plots ---
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 4. Interactive Orbital Trajectory Inspection")
            trajectory_plot = gr.Plot(label="Interactive Trajectory Map")

    # Wire event listener button click
    process_btn.click(
        fn=process_video_pipeline,
        inputs=[video_input],
        outputs=[
            video_output,
            stat_objects,
            stat_frames,
            stat_avg_vel,
            stat_max_vel,
            trajectory_plot
        ]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
