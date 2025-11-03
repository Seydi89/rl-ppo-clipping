import gradio as gr
import matplotlib.pyplot as plt
from analysis_utils import analyze_results
from clipping_analysis import run_experiment

def run_and_plot(env="CartPole-v1", epochs=5, steps=20000):
    results = run_experiment(env_name=env, n_epochs_list=[epochs], total_timesteps=steps, n_seeds=1)
    fig = analyze_results(results)
    # Save the figure temporarily
    fig_path = "ppo_analysis_result.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    return fig_path

demo = gr.Interface(
    fn=run_and_plot,
    inputs=[
        gr.Textbox(label="Environment", value="CartPole-v1"),
        gr.Slider(1, 20, value=5, step=1, label="Epochs"),
        gr.Number(label="Total Timesteps", value=20000),
    ],
    outputs=gr.Image(label="Analysis Plot"),
    title="PPO Clipping Analysis Experiment",
    description="Run PPO on a selected environment and visualize clipping behavior."
)

if __name__ == "__main__":
    demo.launch()
