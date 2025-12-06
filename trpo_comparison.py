import gymnasium as gym
from stable_baselines3 import PPO
from sb3_contrib import TRPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def run_training(algo_class, algo_name, env_name, total_timesteps, **kwargs):
    """
    Generic trainer for both PPO and TRPO
    """
    print(f"--- Training {algo_name} ---")
    
    # Setup Environment
    def make_env():
        return Monitor(gym.make(env_name))
    env = DummyVecEnv([make_env])
    
    # Initialize Model
    model = algo_class("MlpPolicy", env, verbose=0, **kwargs)
    
    # Train
    model.learn(total_timesteps=total_timesteps)
    
    # Extract Logged Returns
    rewards = env.envs[0].get_episode_rewards()
    env.close()
    return rewards

def run_trpo_comparison(env_name="CartPole-v1", steps=20000, seeds=1):
    """
    Runs PPO vs TRPO and returns the Matplotlib Figure object.
    """
    results = []

    for seed in range(seeds):
        # Run PPO (Standard Paper Settings)
        ppo_rewards = run_training(
            PPO, "PPO", env_name, steps, 
            learning_rate=3e-4, 
            n_epochs=10, 
            clip_range=0.2, 
            seed=seed
        )
        
        # Run TRPO (The Baseline)
        trpo_rewards = run_training(
            TRPO, "TRPO", env_name, steps, 
            learning_rate=1e-3, 
            seed=seed
        )

        # Store Data
        for i, r in enumerate(ppo_rewards):
            results.append({"Algorithm": "PPO", "Episode": i, "Return": r, "Seed": seed})
        for i, r in enumerate(trpo_rewards):
            results.append({"Algorithm": "TRPO", "Episode": i, "Return": r, "Seed": seed})

    # Plotting
    df = pd.DataFrame(results)
    
    # Create a new figure explicitly
    fig = plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Episode", y="Return", hue="Algorithm", errorbar='sd')
    plt.title(f"PPO (Clipping) vs. TRPO (Trust Region) on {env_name}")
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.grid(True, alpha=0.3)
    
    # Return the figure object so Gradio can display it
    return fig

if __name__ == "__main__":
    # Test run
    fig = run_trpo_comparison()
    plt.show()