import gradio as gr
import matplotlib.pyplot as plt
from analysis_utils import analyze_results
from clipping_analysis import (
    run_experiment,
    run_clip_range_sweep,
    run_combined_sweep,
)

def run_and_plot(
    experiment_type,
    env,
    epochs,
    steps,
    clip_range,
    clip_ranges,
    epochs_list,
):
    # Parse comma-separated inputs
    clip_ranges_list = [float(x.strip()) for x in str(clip_ranges).split(",") if str(x).strip()]
    epochs_list_parsed = [int(x.strip()) for x in str(epochs_list).split(",") if str(x).strip()]

    if experiment_type == "Standard (vary n_epochs)":
        results = run_experiment(
            env_name=env,
            n_epochs_list=[epochs],
            total_timesteps=int(steps),
            n_seeds=1,
            clip_range=float(clip_range),
        )
    elif experiment_type == "Clip Range Sweep":
        results = run_clip_range_sweep(
            env_name=env,
            clip_ranges=clip_ranges_list,
            n_epochs=int(epochs),
            total_timesteps=int(steps),
            n_seeds=1,
        )
    elif experiment_type == "Combined Sweep (clip_range & n_epochs)":
        results = run_combined_sweep(
            env_name=env,
            clip_ranges=clip_ranges_list,
            n_epochs_list=epochs_list_parsed,
            total_timesteps=int(steps),
            n_seeds=1,
        )
    else:
        raise ValueError("Unknown experiment type selected.")

    fig = analyze_results(results)
    fig_path = "ppo_analysis_result.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    return fig_path

def toggle_inputs(exp_type):
    """
    Control visibility of inputs depending on the selected experiment.
    - Standard: show clip_range + epochs; hide clip_ranges list + epochs_list
    - Clip Range Sweep: show clip_ranges list + epochs; hide clip_range + epochs_list
    - Combined: show clip_ranges list + epochs_list; hide clip_range + epochs
    """
    if exp_type == "Standard (vary n_epochs)":
        return (
            gr.update(visible=True),   # clip_range
            gr.update(visible=False),  # clip_ranges
            gr.update(visible=False),  # epochs_list
            gr.update(visible=True),   # epochs
        )
    if exp_type == "Clip Range Sweep":
        return (
            gr.update(visible=False),  # clip_range
            gr.update(visible=True),   # clip_ranges
            gr.update(visible=False),  # epochs_list
            gr.update(visible=True),   # epochs
        )
    # Combined
    return (
        gr.update(visible=False),  # clip_range
        gr.update(visible=True),   # clip_ranges
        gr.update(visible=True),   # epochs_list
        gr.update(visible=False),  # epochs
    )

with gr.Blocks(title="PPO Clipping Analysis Experiment Suite") as demo:
    gr.Markdown("# PPO Clipping Analysis Experiment Suite")
    gr.Markdown(
        "Run PPO with different configurations to analyze clipping behavior.\n\n"
        "**Modes:**\n"
        "- **Standard:** Vary `n_epochs` (uses a single `clip_range`).\n"
        "- **Clip Range Sweep:** Compare multiple `clip_range` values (uses a single `n_epochs`).\n"
        "- **Combined Sweep:** Explore interactions between multiple `clip_range` values **and** multiple `n_epochs`."
    )

    with gr.Row():
        exp_type = gr.Dropdown(
            ["Standard (vary n_epochs)", "Clip Range Sweep", "Combined Sweep (clip_range & n_epochs)"],
            label="Experiment Type",
            value="Standard (vary n_epochs)",
        )
        env = gr.Textbox(label="Environment", value="CartPole-v1", scale=2)

    with gr.Row():
        epochs = gr.Slider(1, 20, value=5, step=1, label="Epochs (used in Standard & Clip Range Sweep)")
        steps = gr.Number(label="Total Timesteps", value=20000)

    with gr.Row():
        clip_range = gr.Number(label="Clip Range (used in Standard only)", value=0.2)
        clip_ranges = gr.Textbox(
            label="Clip Ranges (comma-separated, used in sweeps)",
            value="0.1,0.2,0.3",
            visible=False,
        )

    epochs_list = gr.Textbox(
        label="Epochs List (comma-separated, used in Combined Sweep)",
        value="3,10,20",
        visible=False,
    )

    run_btn = gr.Button("Run Experiment", variant="primary")
    output_img = gr.Image(label="Analysis Plot")

    # Wire visibility toggling
    exp_type.change(
        fn=toggle_inputs,
        inputs=[exp_type],
        outputs=[clip_range, clip_ranges, epochs_list, epochs],
    )

    # Run
    run_btn.click(
        fn=run_and_plot,
        inputs=[exp_type, env, epochs, steps, clip_range, clip_ranges, epochs_list],
        outputs=[output_img],
    )

if __name__ == "__main__":
    demo.launch()
