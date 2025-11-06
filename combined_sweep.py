"""
EXPERIMENT 3: Varying Both Clipping Range AND Epochs
Analyzes the interaction between clip window width and data reuse.
"""

from matplotlib import pyplot as plt
from clipping_analysis import run_combined_sweep
from analysis_utils import analyze_combined_sweep_results, print_combined_sweep_summary

if __name__ == "__main__":
    print("="*80)
    print("EXPERIMENT 3: Combined Sweep (Clip Range × Epochs)")
    print("="*80)
    print("\nThis experiment will:")
    print("  - Test 3 clip ranges: [0.1, 0.2, 0.3]")
    print("  - Test 3 epoch values: [3, 10, 20]")
    print("  - Run 3 seeds per configuration")
    print("  - Total: 27 training runs on CartPole-v1")
    print("\nEstimated time: ~30-40 minutes")
    print("="*80)
    
    # Run the experiment
    results = run_combined_sweep(
        env_name="CartPole-v1",
        clip_ranges=[0.1, 0.2, 0.3],
        n_epochs_list=[3, 10, 20],
        total_timesteps=50000,
        n_seeds=3
    )
    
    print("\nGenerating comprehensive analysis plots...")
    fig, df, summary = analyze_combined_sweep_results(results)
    
    print_combined_sweep_summary(results)
    
    figure_path = 'experiment3_combined_sweep.png'
    fig.savefig(figure_path, dpi=300, bbox_inches='tight')
    
    # summary.to_csv('experiment3_summary.csv', index=False)
    # print(f"\n✓ Summary table saved to 'experiment3_summary.csv'")
    
    plt.show()
    
    print(f"\n✓ Analysis complete! Results saved to '{figure_path}'")