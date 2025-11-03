
from matplotlib import pyplot as plt
from analysis_utils import analyze_results, print_summary_results
from clipping_analysis import run_experiment


if __name__ == "__main__":
    print("Starting PPO Clipping Analysis Experiment...")

    # Run the experiment with a few settings for demonstration
    experiment_results = run_experiment(
        env_name="CartPole-v1",
        n_epochs_list=[1, 5, 10, 20],
        total_timesteps=100,
        n_seeds=3,
        clip_range=0.2
    )

    # Analyze and visualize the results
    print("\nExperiment complete. Generating analysis plots...")
    analysis_figure = analyze_results(experiment_results)
    print_summary_results(experiment_results)
    # Save the figure and show it
    figure_path = 'ppo_clipping_analysis.png'
    analysis_figure.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nAnalysis complete! Results saved to '{figure_path}'")