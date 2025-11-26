"""
PPO Clipping Frequency Analysis Experiment
Measures how clipping frequency changes across epochs and its impact on performance.
Includes real-time printing of clipping statistics during training.
"""

import numpy as np
import torch as th
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from typing import Dict, List
from collections import defaultdict


class ClippingAnalysisCallback(BaseCallback):
    """
    Custom callback to track and print clipping statistics per epoch during training.

    This callback hooks into the PPO `train` method to capture detailed,
    per-epoch statistics that are not available through standard logging.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.current_update = 0

        # Comprehensive dictionary to store all relevant statistics for later analysis
        self.update_stats = {
            'clip_fractions_per_epoch': [],
            'ratio_distributions_per_epoch': [],
            'advantages_per_epoch': [],
            'policy_losses_per_epoch': [],
            'kl_divs_per_epoch': [],
            'returns': [],
            'episode_lengths': [],
            'timesteps': []
        }

    def _on_step(self) -> bool:
        # This is called at each environment step.
        # We don't need to do anything here for this analysis.
        return True

    def _on_rollout_end(self) -> None:
        """
        This is called at the end of each rollout collection (before the update).
        We use it to log the mean return and episode length from the latest episodes.
        """
        if self.model.ep_info_buffer:
            new_returns = [ep_info['r'] for ep_info in self.model.ep_info_buffer]
            mean_return = np.mean(new_returns)
            self.update_stats['returns'].append(mean_return)
            self.update_stats['timesteps'].append(self.num_timesteps)
            self.model.ep_info_buffer.clear()

        pass

    def _on_training_start(self) -> None:
        """
        This method is called once before the first rollout starts.
        We "monkey-patch" the model's `train` method to insert our data collection logic.
        """
        # Store the original train method
        self.original_train = self.model.train
        # Replace it with our wrapper
        self.model.train = self._wrapped_train

    def _wrapped_train(self) -> None:
        """
        A wrapper around the original `train` method to collect epoch-wise statistics.
        This code is an adaptation of the SB3 PPO `train` method.
        """
        self.model.policy.set_training_mode(True)
        self.model._update_learning_rate(self.model.policy.optimizer)

        clip_range = self.model.clip_range(self.model._current_progress_remaining)
        clip_range_vf = None
        if self.model.clip_range_vf is not None:
            clip_range_vf = self.model.clip_range_vf(self.model._current_progress_remaining)

        # Statistics per update (across all epochs)
        update_clip_fractions = []
        update_ratios = []
        update_advantages = []
        update_kl_divs = []

        continue_training = True

        # Train for n_epochs
        for epoch in range(self.model.n_epochs):
            epoch_clip_fractions = []
            epoch_ratios = []
            epoch_advantages = []
            epoch_kl_divs = []

            # Iterate through minibatches
            for rollout_data in self.model.rollout_buffer.get(self.model.batch_size):
                actions = rollout_data.actions
                if isinstance(self.model.action_space, gym.spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.model.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()

                # Normalize advantages
                advantages = rollout_data.advantages
                if self.model.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Ratio between old and new policy
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # Store statistics for detailed analysis
                epoch_ratios.append(ratio.detach().cpu().numpy())
                epoch_advantages.append(advantages.detach().cpu().numpy())

                # --- CORE CLIPPING CALCULATION ---
                # Identify which importance weights are outside the clipping range
                clipped = th.abs(ratio - 1) > clip_range
                # Calculate the fraction of clipped weights in the minibatch
                clip_fraction = th.mean(clipped.float()).item()
                epoch_clip_fractions.append(clip_fraction)

                # Policy loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Value loss
                if self.model.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = th.nn.functional.mse_loss(rollout_data.returns, values_pred)

                # Entropy loss
                entropy_loss = -th.mean(entropy) if entropy is not None else -th.mean(-log_prob)

                loss = policy_loss + self.model.ent_coef * entropy_loss + self.model.vf_coef * value_loss

                # KL divergence for early stopping
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    epoch_kl_divs.append(approx_kl_div)

                if self.model.target_kl is not None and approx_kl_div > 1.5 * self.model.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {self.num_timesteps} due to reaching max KL divergence.")
                    break

                # Optimization step
                self.model.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.model.policy.parameters(), self.model.max_grad_norm)
                self.model.policy.optimizer.step()

            self.model._n_updates += 1

            # Aggregate stats for the completed epoch
            update_clip_fractions.append(np.mean(epoch_clip_fractions))
            update_ratios.append(np.concatenate(epoch_ratios))
            update_advantages.append(np.concatenate(epoch_advantages))
            update_kl_divs.append(np.mean(epoch_kl_divs))

            if not continue_training:
                break

        # --- IMPROVEMENT: PRINTING STATS DURING TRAINING ---
        # After all epochs in this update are finished, print a summary
        if self.update_stats['returns']:
            latest_return = self.update_stats['returns'][-1]
            clip_fractions_str = ", ".join([f"{cf:.3f}" for cf in update_clip_fractions])
            print(
                f"Update: {self.current_update:<4} | "
                f"Timesteps: {self.num_timesteps:<7} | "
                f"Avg Return: {latest_return:<8.2f} | "
                f"Clip Fractions per Epoch: [{clip_fractions_str}]"
            )
        # --- END OF IMPROVEMENT ---

        # Store update-level statistics for final plotting
        self.update_stats['clip_fractions_per_epoch'].append(update_clip_fractions)
        self.update_stats['ratio_distributions_per_epoch'].append(update_ratios)
        self.update_stats['advantages_per_epoch'].append(update_advantages)
        self.update_stats['kl_divs_per_epoch'].append(update_kl_divs)

        self.current_update += 1


def run_experiment(
    env_name: str = "CartPole-v1",
    n_epochs_list: List[int] = [1, 5, 10, 20],
    total_timesteps: int = 50000,
    n_seeds: int = 3,
    clip_range: float = 0.2
):
    """
    Runs the PPO experiment with different `n_epochs` values to analyze clipping.
    """
    results = []

    for n_epochs in n_epochs_list:
        print(f"\n{'='*80}")
        print(f"EXPERIMENT: Training with n_epochs = {n_epochs}")
        print(f"{'='*80}")

        for seed in range(n_seeds):
            print(f"\n--- Running Seed {seed+1}/{n_seeds} ---")

            def make_monitor_env():
                return Monitor(gym.make(env_name))

            env = DummyVecEnv([make_monitor_env])
            callback = ClippingAnalysisCallback(verbose=1)

            model = PPO(
                "MlpPolicy",
                env,
                n_epochs=n_epochs,
                clip_range=clip_range,
                n_steps=2048,
                batch_size=64,
                verbose=0,
                seed=seed
            )

            model.learn(total_timesteps=total_timesteps, callback=callback)

            results.append({
                'n_epochs': n_epochs,
                'seed': seed,
                'callback': callback,
                'final_return': np.mean(callback.update_stats['returns'][-10:]) if callback.update_stats['returns'] else 0
            })
            env.close()

    return results

#------------Experiment for varying clip_range------------#

def run_clip_range_sweep(
    env_name: str = "CartPole-v1",
    clip_ranges: List[float] = [0.1, 0.2, 0.3],
    n_epochs: int = 10,
    total_timesteps: int = 50000,
    n_seeds: int = 3
):
    """
    Runs the PPO experiment with different `clip_range` values to analyze clipping.
    """
    results = []

    for clip_range in clip_ranges:
        print(f"\n{'='*80}")
        print(f"EXPERIMENT: Training with clip_range = {clip_range}")
        print(f"{'='*80}")

        for seed in range(n_seeds):
            print(f"\n--- Running Seed {seed+1}/{n_seeds} ---")

            def make_monitor_env():
                return Monitor(gym.make(env_name))

            env = DummyVecEnv([make_monitor_env])
            callback = ClippingAnalysisCallback(verbose=1)

            model = PPO(
                "MlpPolicy",
                env,
                n_epochs=n_epochs,
                clip_range=clip_range,
                n_steps=2048,
                batch_size=64,
                verbose=0,
                seed=seed
            )

            model.learn(total_timesteps=total_timesteps, callback=callback)

            results.append({
                'clip_range': clip_range,
                'n_epochs': n_epochs,
                'seed': seed,
                'callback': callback,
                'final_return': np.mean(callback.update_stats['returns'][-10:]) if callback.update_stats['returns'] else 0
            })
            env.close()

    return results

#------------Experiment 3: Varying both clip_range AND n_epochs------------#

def run_combined_sweep(
    env_name: str = "CartPole-v1",
    clip_ranges: List[float] = [0.1, 0.2, 0.3],
    n_epochs_list: List[int] = [3, 10, 20],
    total_timesteps: int = 50000,
    n_seeds: int = 3
):
    """
    Runs the PPO experiment varying BOTH `clip_range` and `n_epochs` to analyze their interaction.
    """
    results = []

    for clip_range in clip_ranges:
        for n_epochs in n_epochs_list:
            print(f"\n{'='*80}")
            print(f"EXPERIMENT 3: clip_range = {clip_range}, n_epochs = {n_epochs}")
            print(f"{'='*80}")

            for seed in range(n_seeds):
                print(f"\n--- Running Seed {seed+1}/{n_seeds} ---")

                def make_monitor_env():
                    return Monitor(gym.make(env_name))

                env = DummyVecEnv([make_monitor_env])
                callback = ClippingAnalysisCallback(verbose=1)

                model = PPO(
                    "MlpPolicy",
                    env,
                    n_epochs=n_epochs,
                    clip_range=clip_range,
                    n_steps=2048,
                    batch_size=64,
                    verbose=0,
                    seed=seed
                )

                model.learn(total_timesteps=total_timesteps, callback=callback)

                results.append({
                    'clip_range': clip_range,
                    'n_epochs': n_epochs,
                    'seed': seed,
                    'callback': callback,
                    'final_return': np.mean(callback.update_stats['returns'][-10:]) if callback.update_stats['returns'] else 0
                })
                env.close()

    return results