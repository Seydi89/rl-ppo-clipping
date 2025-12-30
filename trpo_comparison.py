import gymnasium as gym
from stable_baselines3 import PPO
from sb3_contrib import TRPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def run_training(algo_class, algo_name, env_name, total_timesteps, seed, **kwargs):
    """
    Trains an agent and returns rewards and lengths.
    """
    print(f"--- Training {algo_name} on {env_name} (Seed {seed}) ---")
    
    def make_env():
        return Monitor(gym.make(env_name))
    
    env = DummyVecEnv([make_env])
    
    # Initialize Model with the specific seed
    model = algo_class("MlpPolicy", env, verbose=0, seed=seed, **kwargs)
    
    # Train
    model.learn(total_timesteps=total_timesteps)
    
    rewards = env.envs[0].get_episode_rewards()
    lengths = env.envs[0].get_episode_lengths()
    
    env.close()
    return rewards, lengths

# Comparison & Data Collection
def collect_data(env_name, steps, seeds):
    """
    Runs PPO vs TRPO multiple times and returns a DataFrame.
    """
    results = []
    print(f"\nStarting comparison on {env_name} for {steps} steps ({seeds} seeds)...")

    for seed in range(seeds):
        ppo_rewards, ppo_lengths = run_training(
            PPO, "PPO", env_name, steps, seed,
            learning_rate=3e-4, 
            n_epochs=10, 
            clip_range=0.2
        )
        
        # Run TRPO 
        trpo_rewards, trpo_lengths = run_training(
            TRPO, "TRPO", env_name, steps, seed,
            learning_rate=1e-3
        )

        ppo_timesteps = np.cumsum(ppo_lengths)
        for r, t in zip(ppo_rewards, ppo_timesteps):
            results.append({"Algorithm": "PPO", "Timesteps": t, "Return": r, "Seed": seed})

        trpo_timesteps = np.cumsum(trpo_lengths)
        for r, t in zip(trpo_rewards, trpo_timesteps):
            results.append({"Algorithm": "TRPO", "Timesteps": t, "Return": r, "Seed": seed})

    return pd.DataFrame(results)

# Plotting Function 
def plot_comparison(df, env_name, steps):
    """
    Bins the data and plots Mean ± SD.
    """

    bin_size = max(1, steps // 100) 
    df["Timesteps_Binned"] = (df["Timesteps"] // bin_size) * bin_size

    plt.figure(figsize=(10, 6))
    
    sns.lineplot(
        data=df, 
        x="Timesteps_Binned", 
        y="Return", 
        hue="Algorithm", 
        errorbar='sd'
    )
    
    plt.title(f"PPO vs. TRPO on {env_name}")
    plt.xlabel("Total Timesteps")
    plt.ylabel("Score (Mean ± SD)")
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    filename = f"comparison_{env_name}.png"
    plt.savefig(filename)
    print(f"Plot saved as {filename}")
    plt.show()

if __name__ == "__main__":
    TOTAL_STEPS = 1_000_000  
    SEEDS = 5               

    # Run CartPole-v1 
    df_cart = collect_data("CartPole-v1", TOTAL_STEPS, SEEDS)
    plot_comparison(df_cart, "CartPole-v1", TOTAL_STEPS)

    #  Run HalfCheetah-v5 
    try:
        df_cheetah = collect_data("HalfCheetah-v5", TOTAL_STEPS, SEEDS)
        plot_comparison(df_cheetah, "HalfCheetah-v5", TOTAL_STEPS)
    except Exception as e:
        print(f"\nCould not run HalfCheetah-v5. Error: {e}")
