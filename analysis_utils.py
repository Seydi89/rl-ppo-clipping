"""
analysis_utils.py
Contains result analysis and plotting utilities.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict


def analyze_results(results: List[Dict]):
    """
    Analyzes and visualizes the results from the PPO clipping frequency experiment.
    Produces a 2x3 grid of plots summarizing clipping dynamics and performance.
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('PPO Clipping Analysis: Impact of Number of Epochs', fontsize=18, y=1.03)

    # 1. Clipping Frequency vs. Epoch (within an update)
    ax = axes[0, 0]
    df_data = []
    for res in results:
        n_epochs = res['n_epochs']
        clip_fractions = res['callback'].update_stats['clip_fractions_per_epoch']
        for update_idx, epoch_clips in enumerate(clip_fractions):
            for epoch_idx, clip_val in enumerate(epoch_clips):
                df_data.append({
                    'n_epochs': n_epochs,
                    'Epoch within Update': epoch_idx + 1,
                    'Clipping Fraction': clip_val
                })

    if df_data:
        df = pd.DataFrame(df_data)
        sns.lineplot(
            data=df,
            x='Epoch within Update',
            y='Clipping Fraction',
            hue='n_epochs',
            marker='o',
            ax=ax,
            palette='viridis',
            ci='sd'
        )

    ax.set_title('Clipping Frequency Increases with Each Epoch', fontsize=14)
    ax.set_xlabel('Epoch within Update', fontsize=12)
    ax.set_ylabel('Mean Clipping Fraction', fontsize=12)
    ax.legend(title='n_epochs')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 2. Final Performance vs. n_epochs
    ax = axes[0, 1]
    perf_data = [{'n_epochs': r['n_epochs'], 'Final Return': r['final_return']} for r in results]
    if perf_data:
        df_perf = pd.DataFrame(perf_data)
        sns.lineplot(
            data=df_perf,
            x='n_epochs',
            y='Final Return',
            marker='o',
            ax=ax,
            ci='sd',
            err_style="band"
        )

    ax.set_title('Performance vs. Number of Epochs', fontsize=14)
    ax.set_xlabel('Number of Epochs (n_epochs)', fontsize=12)
    ax.set_ylabel('Average Return (Last 10 Updates)', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 3. Total Clipping vs. Performance
    ax = axes[0, 2]
    scatter_data = []
    for result in results:
        callback = result['callback']
        if callback.update_stats['clip_fractions_per_epoch']:
            total_clip = np.mean([
                np.mean(epoch_clips)
                for epoch_clips in callback.update_stats['clip_fractions_per_epoch']
            ])
            scatter_data.append({
                'total_clipping': total_clip,
                'final_return': result['final_return'],
                'n_epochs': result['n_epochs']
            })

    if scatter_data:
        df_scatter = pd.DataFrame(scatter_data)
        scatter = sns.scatterplot(
            data=df_scatter,
            x='total_clipping',
            y='final_return',
            hue='n_epochs',
            palette='viridis',
            s=100,
            alpha=0.7,
            ax=ax
        )
        scatter.legend(title='n_epochs')

    ax.set_title('Performance vs. Average Clipping Frequency', fontsize=14)
    ax.set_xlabel('Average Clipping Fraction Across Training', fontsize=12)
    ax.set_ylabel('Final Return', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 4. KL Divergence Evolution
    ax = axes[1, 0]
    kl_data = []
    for res in results:
        n_epochs = res['n_epochs']
        kl_divs = res['callback'].update_stats['kl_divs_per_epoch']
        for update_idx, epoch_kls in enumerate(kl_divs):
            for epoch_idx, kl_val in enumerate(epoch_kls):
                kl_data.append({
                    'n_epochs': n_epochs,
                    'Epoch within Update': epoch_idx + 1,
                    'KL Divergence': kl_val
                })

    if kl_data:
        df_kl = pd.DataFrame(kl_data)
        sns.lineplot(
            data=df_kl,
            x='Epoch within Update',
            y='KL Divergence',
            hue='n_epochs',
            marker='o',
            ax=ax,
            palette='viridis',
            ci='sd'
        )
        ax.set_yscale('log')

    ax.set_title('KL Divergence Rises Across Epochs', fontsize=14)
    ax.set_xlabel('Epoch within Update', fontsize=12)
    ax.set_ylabel('KL Divergence (log scale)', fontsize=12)
    ax.legend(title='n_epochs')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 5. Learning Curves
    ax = axes[1, 1]
    lc_data = []
    for res in results:
        n_epochs = res['n_epochs']
        seed = res['seed']
        returns = res['callback'].update_stats['returns']
        timesteps = res['callback'].update_stats['timesteps'][:len(returns)]
        for t, r in zip(timesteps, returns):
            lc_data.append({
                'Timesteps': t,
                'Episode Return': r,
                'n_epochs': n_epochs,
                'seed': seed
            })

    if lc_data:
        df_lc = pd.DataFrame(lc_data)
        sns.lineplot(
            data=df_lc,
            x='Timesteps',
            y='Episode Return',
            hue='n_epochs',
            palette='viridis',
            ax=ax,
            ci='sd'
        )

    ax.set_title('Learning Curves', fontsize=14)
    ax.set_xlabel('Timesteps', fontsize=12)
    ax.set_ylabel('Mean Episode Return', fontsize=12)
    ax.legend(title='n_epochs')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 6. Ratio Distribution
    ax = axes[1, 2]
    unique_n_epochs = sorted(list(set(r['n_epochs'] for r in results)))
    sampled_results = [
        next(r for r in results if r['n_epochs'] == n)
        for n in unique_n_epochs
    ]

    for result in sampled_results:
        callback = result['callback']
        if callback.update_stats['ratio_distributions_per_epoch']:
            last_ratios = callback.update_stats['ratio_distributions_per_epoch'][-1][0]
            sns.histplot(
                last_ratios,
                bins=50,
                alpha=0.5,
                label=f"n_epochs={result['n_epochs']}",
                ax=ax,
                stat="density",
                kde=True
            )

    ax.axvline(x=0.8, color='r', linestyle='--', label='Clip Bounds (0.2)', linewidth=2)
    ax.axvline(x=1.2, color='r', linestyle='--', linewidth=2)
    ax.set_title('Ratio Distribution (Last Update, 1st Epoch)', fontsize=14)
    ax.set_xlabel('Importance Sampling Ratio', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig

def print_summary_results(results: List[Dict]):
    """
    Prints a concise text summary of experiment outcomes:
    - Average final returns for each n_epochs setting
    - Average clipping fractions per epoch (if available)
    - Average KL divergences per epoch (if available)
    """
    import numpy as np
    from collections import defaultdict

    print("\n" + "=" * 100)
    print(" " * 30 + "PPO CLIPPING ANALYSIS SUMMARY")
    print("=" * 100)

    results_by_epochs = defaultdict(list)
    for res in results:
        results_by_epochs[res['n_epochs']].append(res)

    for n_epochs, seed_results in sorted(results_by_epochs.items()):
        print(f"\n{'#' * 40}  n_epochs = {n_epochs}  {'#' * 40}")
        final_returns = [r['final_return'] for r in seed_results]
        print(f"  Average Final Return: {np.mean(final_returns):.2f} ± {np.std(final_returns):.2f}")

        # Aggregate clipping fractions
        try:
            all_clips = np.array([
                res['callback'].update_stats['clip_fractions_per_epoch']
                for res in seed_results
                if res['callback'].update_stats['clip_fractions_per_epoch']
            ], dtype=object)
            if all_clips.size > 0:
                mean_clips_per_epoch = np.mean(
                    [np.mean(epoch_clips) for res in seed_results
                     for epoch_clips in res['callback'].update_stats['clip_fractions_per_epoch']]
                )
                print(f"  Avg Clipping Fraction (across all updates): {mean_clips_per_epoch:.4f}")
        except Exception as e:
            print(f"  [!] Could not compute clipping fraction summary: {e}")

        # Aggregate KL divergences (if available)
        try:
            all_kls = np.array([
                res['callback'].update_stats['kl_divs_per_epoch']
                for res in seed_results
                if res['callback'].update_stats['kl_divs_per_epoch']
            ], dtype=object)
            if all_kls.size > 0:
                mean_kls_per_epoch = np.mean(
                    [np.mean(epoch_kls) for res in seed_results
                     for epoch_kls in res['callback'].update_stats['kl_divs_per_epoch']]
                )
                print(f"  Avg KL Divergence (across all updates): {mean_kls_per_epoch:.5f}")
        except Exception as e:
            print(f"  [!] Could not compute KL summary: {e}")

    print("\n" + "=" * 100)
    print("Summary complete.\n")

#---------Enhanced Analysis for Clip Range Sweep Experiment---------#
def analyze_clip_sweep_results_detailed(results: List[Dict]):
    """
    Comprehensive analysis for clip_range sweep - mirrors analyze_results() structure.
    Produces a 2x3 grid of plots analyzing clipping dynamics across different clip ranges.
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('PPO Clipping Analysis: Impact of Clip Range', fontsize=18, y=1.03)

    # 1. Clipping Frequency vs. Epoch (within an update)
    ax = axes[0, 0]
    df_data = []
    for res in results:
        clip_range = res['clip_range']
        clip_fractions = res['callback'].update_stats['clip_fractions_per_epoch']
        for update_idx, epoch_clips in enumerate(clip_fractions):
            for epoch_idx, clip_val in enumerate(epoch_clips):
                df_data.append({
                    'clip_range': clip_range,
                    'Epoch within Update': epoch_idx + 1,
                    'Clipping Fraction': clip_val
                })

    if df_data:
        df = pd.DataFrame(df_data)
        sns.lineplot(
            data=df,
            x='Epoch within Update',
            y='Clipping Fraction',
            hue='clip_range',
            marker='o',
            ax=ax,
            palette='viridis',
            ci='sd'
        )

    ax.set_title('Clipping Frequency vs. Epoch (Different Clip Ranges)', fontsize=14)
    ax.set_xlabel('Epoch within Update', fontsize=12)
    ax.set_ylabel('Mean Clipping Fraction', fontsize=12)
    ax.legend(title='clip_range (ε)')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 2. Final Performance vs. Clip Range
    ax = axes[0, 1]
    perf_data = [{'clip_range': r['clip_range'], 'Final Return': r['final_return']} for r in results]
    if perf_data:
        df_perf = pd.DataFrame(perf_data)
        sns.lineplot(
            data=df_perf,
            x='clip_range',
            y='Final Return',
            marker='o',
            ax=ax,
            ci='sd',
            err_style="band",
            color='steelblue'
        )

    ax.set_title('Performance vs. Clip Range', fontsize=14)
    ax.set_xlabel('Clip Range (ε)', fontsize=12)
    ax.set_ylabel('Average Return (Last 10 Updates)', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 3. Total Clipping vs. Performance
    ax = axes[0, 2]
    scatter_data = []
    for result in results:
        callback = result['callback']
        if callback.update_stats['clip_fractions_per_epoch']:
            total_clip = np.mean([
                np.mean(epoch_clips)
                for epoch_clips in callback.update_stats['clip_fractions_per_epoch']
            ])
            scatter_data.append({
                'total_clipping': total_clip,
                'final_return': result['final_return'],
                'clip_range': result['clip_range']
            })

    if scatter_data:
        df_scatter = pd.DataFrame(scatter_data)
        scatter = sns.scatterplot(
            data=df_scatter,
            x='total_clipping',
            y='final_return',
            hue='clip_range',
            palette='viridis',
            s=100,
            alpha=0.7,
            ax=ax
        )
        scatter.legend(title='clip_range (ε)')

    ax.set_title('Performance vs. Average Clipping Frequency', fontsize=14)
    ax.set_xlabel('Average Clipping Fraction Across Training', fontsize=12)
    ax.set_ylabel('Final Return', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 4. KL Divergence Evolution
    ax = axes[1, 0]
    kl_data = []
    for res in results:
        clip_range = res['clip_range']
        kl_divs = res['callback'].update_stats['kl_divs_per_epoch']
        for update_idx, epoch_kls in enumerate(kl_divs):
            for epoch_idx, kl_val in enumerate(epoch_kls):
                kl_data.append({
                    'clip_range': clip_range,
                    'Epoch within Update': epoch_idx + 1,
                    'KL Divergence': kl_val
                })

    if kl_data:
        df_kl = pd.DataFrame(kl_data)
        sns.lineplot(
            data=df_kl,
            x='Epoch within Update',
            y='KL Divergence',
            hue='clip_range',
            marker='o',
            ax=ax,
            palette='viridis',
            ci='sd'
        )
        ax.set_yscale('log')

    ax.set_title('KL Divergence Across Epochs (Different Clip Ranges)', fontsize=14)
    ax.set_xlabel('Epoch within Update', fontsize=12)
    ax.set_ylabel('KL Divergence (log scale)', fontsize=12)
    ax.legend(title='clip_range (ε)')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 5. Learning Curves
    ax = axes[1, 1]
    lc_data = []
    for res in results:
        clip_range = res['clip_range']
        seed = res['seed']
        returns = res['callback'].update_stats['returns']
        timesteps = res['callback'].update_stats['timesteps'][:len(returns)]
        for t, r in zip(timesteps, returns):
            lc_data.append({
                'Timesteps': t,
                'Episode Return': r,
                'clip_range': clip_range,
                'seed': seed
            })

    if lc_data:
        df_lc = pd.DataFrame(lc_data)
        sns.lineplot(
            data=df_lc,
            x='Timesteps',
            y='Episode Return',
            hue='clip_range',
            palette='viridis',
            ax=ax,
            ci='sd'
        )

    ax.set_title('Learning Curves', fontsize=14)
    ax.set_xlabel('Timesteps', fontsize=12)
    ax.set_ylabel('Mean Episode Return', fontsize=12)
    ax.legend(title='clip_range (ε)')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 6. Ratio Distribution
    ax = axes[1, 2]
    unique_clip_ranges = sorted(list(set(r['clip_range'] for r in results)))
    sampled_results = [
        next(r for r in results if r['clip_range'] == cr)
        for cr in unique_clip_ranges
    ]

    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_clip_ranges)))
    for idx, result in enumerate(sampled_results):
        callback = result['callback']
        clip_range = result['clip_range']
        if callback.update_stats['ratio_distributions_per_epoch']:
            last_ratios = callback.update_stats['ratio_distributions_per_epoch'][-1][0]
            sns.histplot(
                last_ratios,
                bins=50,
                alpha=0.5,
                label=f"ε={clip_range}",
                ax=ax,
                stat="density",
                kde=True,
                color=colors[idx]
            )
        
        # Add vertical lines for THIS clip range's bounds
        ax.axvline(x=1 - clip_range, color=colors[idx], linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axvline(x=1 + clip_range, color=colors[idx], linestyle='--', linewidth=1.5, alpha=0.7)

    ax.set_title('Ratio Distribution (Last Update, 1st Epoch)', fontsize=14)
    ax.set_xlabel('Importance Sampling Ratio', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def print_clip_sweep_summary(results: List[Dict]):
    """
    Prints a text summary for clip range sweep experiments.
    """
    print("\n" + "=" * 100)
    print(" " * 25 + "PPO CLIP RANGE SWEEP ANALYSIS SUMMARY")
    print("=" * 100)

    results_by_clip = defaultdict(list)
    for res in results:
        results_by_clip[res['clip_range']].append(res)

    for clip_range, seed_results in sorted(results_by_clip.items()):
        print(f"\n{'#' * 40}  clip_range (ε) = {clip_range}  {'#' * 40}")
        final_returns = [r['final_return'] for r in seed_results]
        print(f"  Average Final Return: {np.mean(final_returns):.2f} ± {np.std(final_returns):.2f}")

        # Aggregate clipping fractions
        try:
            mean_clips_per_epoch = np.mean([
                np.mean(epoch_clips)
                for res in seed_results
                for epoch_clips in res['callback'].update_stats['clip_fractions_per_epoch']
            ])
            print(f"  Avg Clipping Fraction (across all updates): {mean_clips_per_epoch:.4f}")
        except Exception as e:
            print(f"  [!] Could not compute clipping fraction summary: {e}")

        # Aggregate KL divergences
        try:
            mean_kls_per_epoch = np.mean([
                np.mean(epoch_kls)
                for res in seed_results
                for epoch_kls in res['callback'].update_stats['kl_divs_per_epoch']
            ])
            print(f"  Avg KL Divergence (across all updates): {mean_kls_per_epoch:.5f}")
        except Exception as e:
            print(f"  [!] Could not compute KL summary: {e}")

    print("\n" + "=" * 100)
    print("Summary complete.\n")


#---------Analysis for Combined Sweep Experiment---------#

def analyze_combined_sweep_results(results: List[Dict]):
    """
    Comprehensive analysis for combined clip_range + n_epochs sweep.
    Creates a 2x3 grid with heatmaps and 3D visualizations.
    """
    # Prepare data
    rows = []
    for r in results:
        us = r["callback"].update_stats
        clip_fractions = us.get("clip_fractions_per_epoch", [])
        mean_clip_fraction = float(np.nanmean(clip_fractions)) if clip_fractions else np.nan

        rows.append({
            "clip_range": r["clip_range"],
            "n_epochs": r["n_epochs"],
            "seed": r["seed"],
            "final_return": r["final_return"],
            "mean_clip_fraction": mean_clip_fraction
        })

    df = pd.DataFrame(rows)
    
    # Calculate summary statistics
    summary = (
        df.groupby(["clip_range", "n_epochs"])
        .agg(
            mean_return=("final_return", "mean"),
            std_return=("final_return", "std"),
            mean_clip_fraction=("mean_clip_fraction", "mean")
        )
        .reset_index()
    )

    # Create figure with 2x3 subplots
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('PPO Combined Sweep: Clip Range × Epochs Interaction', fontsize=18, y=0.98)

    # 1. Heatmap: Mean Final Return
    ax1 = plt.subplot(2, 3, 1)
    pivot_perf = summary.pivot(index="n_epochs", columns="clip_range", values="mean_return")
    sns.heatmap(pivot_perf, annot=True, fmt=".1f", cmap="viridis", ax=ax1, cbar_kws={'label': 'Mean Return'})
    ax1.set_title("Mean Final Return (Higher is Better)", fontsize=14)
    ax1.set_ylabel("Number of Epochs", fontsize=12)
    ax1.set_xlabel("Clip Range (ε)", fontsize=12)

    # 2. Heatmap: Std of Final Return (variability)
    ax2 = plt.subplot(2, 3, 2)
    pivot_std = summary.pivot(index="n_epochs", columns="clip_range", values="std_return")
    sns.heatmap(pivot_std, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax2, cbar_kws={'label': 'Std Dev'})
    ax2.set_title("Return Variability (Lower is Better)", fontsize=14)
    ax2.set_ylabel("Number of Epochs", fontsize=12)
    ax2.set_xlabel("Clip Range (ε)", fontsize=12)

    # 3. Heatmap: Mean Clip Fraction
    ax3 = plt.subplot(2, 3, 3)
    pivot_clip = summary.pivot(index="n_epochs", columns="clip_range", values="mean_clip_fraction")
    sns.heatmap(pivot_clip, annot=True, fmt=".3f", cmap="coolwarm", ax=ax3, cbar_kws={'label': 'Clip Fraction'})
    ax3.set_title("Mean Clipping Fraction", fontsize=14)
    ax3.set_ylabel("Number of Epochs", fontsize=12)
    ax3.set_xlabel("Clip Range (ε)", fontsize=12)

    # 4. 3D Surface Plot: Performance
    ax4 = plt.subplot(2, 3, 4, projection='3d')
    X = summary['clip_range'].values
    Y = summary['n_epochs'].values
    Z = summary['mean_return'].values
    
    # Create meshgrid for surface
    from scipy.interpolate import griddata
    xi = np.linspace(X.min(), X.max(), 20)
    yi = np.linspace(Y.min(), Y.max(), 20)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((X, Y), Z, (Xi, Yi), method='cubic')
    
    surf = ax4.plot_surface(Xi, Yi, Zi, cmap='viridis', alpha=0.8)
    ax4.scatter(X, Y, Z, c='red', s=50, marker='o')
    ax4.set_xlabel('Clip Range (ε)', fontsize=10)
    ax4.set_ylabel('n_epochs', fontsize=10)
    ax4.set_zlabel('Mean Return', fontsize=10)
    ax4.set_title('Performance Surface', fontsize=14)
    fig.colorbar(surf, ax=ax4, shrink=0.5)

    # 5. Line plot: Performance vs. n_epochs for each clip_range
    ax5 = plt.subplot(2, 3, 5)
    for cr in sorted(summary['clip_range'].unique()):
        subset = summary[summary['clip_range'] == cr]
        ax5.plot(subset['n_epochs'], subset['mean_return'], marker='o', label=f"ε={cr}", linewidth=2)
        ax5.fill_between(
            subset['n_epochs'],
            subset['mean_return'] - subset['std_return'],
            subset['mean_return'] + subset['std_return'],
            alpha=0.2
        )
    ax5.set_xlabel('Number of Epochs', fontsize=12)
    ax5.set_ylabel('Mean Return', fontsize=12)
    ax5.set_title('Performance vs. Epochs (per Clip Range)', fontsize=14)
    ax5.legend(title='clip_range')
    ax5.grid(True, alpha=0.3)

    # 6. Line plot: Performance vs. clip_range for each n_epochs
    ax6 = plt.subplot(2, 3, 6)
    for ne in sorted(summary['n_epochs'].unique()):
        subset = summary[summary['n_epochs'] == ne]
        ax6.plot(subset['clip_range'], subset['mean_return'], marker='o', label=f"epochs={ne}", linewidth=2)
        ax6.fill_between(
            subset['clip_range'],
            subset['mean_return'] - subset['std_return'],
            subset['mean_return'] + subset['std_return'],
            alpha=0.2
        )
    ax6.set_xlabel('Clip Range (ε)', fontsize=12)
    ax6.set_ylabel('Mean Return', fontsize=12)
    ax6.set_title('Performance vs. Clip Range (per n_epochs)', fontsize=14)
    ax6.legend(title='n_epochs')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, df, summary


def print_combined_sweep_summary(results: List[Dict]):
    """
    Prints a detailed text summary for combined sweep experiments.
    """
    print("\n" + "=" * 100)
    print(" " * 20 + "PPO COMBINED SWEEP ANALYSIS SUMMARY")
    print(" " * 15 + "(Varying Both Clip Range AND Number of Epochs)")
    print("=" * 100)

    # Group by both parameters
    config_results = defaultdict(list)
    for res in results:
        key = (res['clip_range'], res['n_epochs'])
        config_results[key].append(res)

    # Print results organized by configuration
    for (clip_range, n_epochs), seed_results in sorted(config_results.items()):
        print(f"\n{'#' * 35}  ε={clip_range}, epochs={n_epochs}  {'#' * 35}")
        
        final_returns = [r['final_return'] for r in seed_results]
        print(f"  Average Final Return: {np.mean(final_returns):.2f} ± {np.std(final_returns):.2f}")
        
        try:
            mean_clip = np.mean([
                np.mean(res['callback'].update_stats['clip_fractions_per_epoch'])
                for res in seed_results
            ])
            print(f"  Avg Clipping Fraction: {mean_clip:.4f}")
        except:
            pass

    # Find best configuration
    print("\n" + "=" * 100)
    print("OPTIMAL CONFIGURATION:")
    print("=" * 100)
    
    config_means = {}
    for (clip_range, n_epochs), seed_results in config_results.items():
        config_means[(clip_range, n_epochs)] = np.mean([r['final_return'] for r in seed_results])
    
    best_config = max(config_means, key=config_means.get)
    best_mean = config_means[best_config]
    
    print(f"  ✓ Best Configuration: clip_range={best_config[0]}, n_epochs={best_config[1]}")
    print(f"    - Mean Return: {best_mean:.2f}")
    print("=" * 100 + "\n")