"""
EXPERIMENT 2: Varying Clipping Range, Fixed Epochs
Analyzes how the width of the clipping window affects performance.
"""

from matplotlib import pyplot as plt
from clipping_analysis import run_clip_range_sweep
from analysis_utils import analyze_clip_sweep_results_detailed, print_clip_sweep_summary

if __name__ == "__main__":
    print("="*80)
    print("EXPERIMENT 2: Varying Clipping Range, Fixed Epochs (10)")
    print("="*80)
    print("\nThis experiment will:")
    print("  - Test 3 clip ranges: [0.1, 0.2, 0.3]")
    print("  - Use fixed n_epochs = 10")
    print("  - Run 3 seeds per configuration")
    print("  - Total: 9 training runs on CartPole-v1")
    print("="*80)
    
    results = run_clip_range_sweep(
        env_name="CartPole-v1",
        clip_ranges=[0.1, 0.2, 0.3],
        n_epochs=10,
        total_timesteps=50000,
        n_seeds=3
    )
    
    print("\nGenerating comprehensive analysis plots...")
    fig = analyze_clip_sweep_results_detailed(results)
    
    print_clip_sweep_summary(results)
    
    figure_path = 'experiment2_clip_range_sweep_detailed.png'
    # fig.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ Analysis complete! Results saved to '{figure_path}'")