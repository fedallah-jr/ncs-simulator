"""
OpenAI-ES training for the NCS environment using evosax and JAX.

Compatible with TPU/GPU acceleration for the evolution strategy updates.
Evaluates the population in parallel on CPU.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import multiprocessing
from pathlib import Path
from typing import Optional, List

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import gymnasium as gym
from evosax import OpenAES, ParameterReshaper

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ncs_env.env import NCS_Env
from utils import SingleAgentWrapper
from utils.run_utils import prepare_run_directory, save_config_with_hyperparameters

# -----------------------------------------------------------------------------
# Policy Network
# -----------------------------------------------------------------------------

class PolicyNet(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.action_dim)(x)
        return x

# -----------------------------------------------------------------------------
# Worker / Environment Helper
# -----------------------------------------------------------------------------

# Global variables for worker process
_worker_env: Optional[gym.Env] = None
_worker_model: Optional[PolicyNet] = None
_worker_reshaper: Optional[ParameterReshaper] = None

def _init_worker(
    config_path: Optional[str], 
    episode_length: int, 
    seed: Optional[int],
    obs_dim: int,
    action_dim: int
):
    """Initialize the environment in the worker process."""
    global _worker_env, _worker_model, _worker_reshaper
    
    def factory():
        return NCS_Env(
            n_agents=1,
            episode_length=episode_length,
            config_path=config_path,
            seed=seed,
        )
    
    _worker_env = SingleAgentWrapper(factory)
    _worker_model = PolicyNet(action_dim=action_dim)
    
    # Initialize reshaper structure
    dummy_obs = jnp.zeros((1, obs_dim))
    dummy_params = _worker_model.init(jax.random.PRNGKey(0), dummy_obs)
    _worker_reshaper = ParameterReshaper(dummy_params)


def _evaluate_params(flat_params: np.ndarray) -> float:
    """Run one episode with the given flattened parameters."""
    global _worker_env, _worker_model, _worker_reshaper
    
    # Reconstruct Flax params from flat array
    params = _worker_reshaper.reshape_single(flat_params)
    
    obs, _ = _worker_env.reset()
    total_reward = 0.0
    truncated = False
    terminated = False
    
    while not (truncated or terminated):
        # Simple forward pass
        # We use numpy for the observation to avoid excessive device transfer overhead
        # for single-step inference in a CPU worker.
        logits = _worker_model.apply(params, obs)
        action = int(np.argmax(logits))
        
        obs, reward, terminated, truncated, _ = _worker_env.step(action)
        total_reward += reward
        
    return total_reward

# -----------------------------------------------------------------------------
# Main Training Loop
# -----------------------------------------------------------------------------

def train(args):
    # 1. Setup Dummy Environment to get shapes
    config_path_str = str(args.config) if args.config is not None else None
    dummy_env = NCS_Env(n_agents=1, config_path=config_path_str)
    obs_dim = dummy_env.observation_space["agent_0"].shape[0]
    action_dim = dummy_env.action_space["agent_0"].n
    dummy_env.close()
    
    print(f"Observation Dim: {obs_dim}, Action Dim: {action_dim}")

    # 2. Setup Flax Model & Reshaper
    rng = jax.random.PRNGKey(args.seed if args.seed is not None else 0)
    model = PolicyNet(action_dim=action_dim)
    dummy_obs = jnp.zeros((1, obs_dim))
    params = model.init(rng, dummy_obs)
    
    param_reshaper = ParameterReshaper(params)
    print(f"Total Parameters: {param_reshaper.total_params}")

    # 3. Setup OpenAES Strategy
    strategy = OpenAES(
        popsize=args.popsize,
        num_dims=param_reshaper.total_params,
        opt_name="adam",
        lrate_init=args.learning_rate,
        lrate_decay=args.lrate_decay,
        sigma_init=args.sigma_init,
        sigma_decay=args.sigma_decay,
    )
    
    es_params = strategy.default_params
    state = strategy.initialize(rng, es_params)

    # JIT the ASK and TELL steps for acceleration
    @jax.jit
    def ask_step(rng, state, params):
        x, state = strategy.ask(rng, state, params)
        return x, state

    @jax.jit
    def tell_step(x, fitness, state, params):
        return strategy.tell(x, fitness, state, params)

    # 4. Setup Parallel Workers
    # 'spawn' is safer for JAX+Multiprocessing compatibility
    ctx = multiprocessing.get_context("spawn")
    pool = ctx.Pool(
        processes=args.n_workers,
        initializer=_init_worker,
        initargs=(config_path_str, args.episode_length, args.seed, obs_dim, action_dim)
    )

    # 5. Logging Setup
    run_dir, metadata = prepare_run_directory("openai_es", args.config, args.output_root)
    rewards_file = run_dir / "training_rewards.csv"
    with rewards_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "mean_reward", "max_reward", "time_elapsed"])

    print(f"Starting training for {args.generations} generations...")
    start_time = time.time()
    
    best_fitness_all_time = -float("inf")

    # 6. Training Loop
    for gen in range(1, args.generations + 1):
        rng, rng_ask = jax.random.split(rng)
        
        # ASK (Accelerator)
        x, state = ask_step(rng_ask, state, es_params)
        
        # EVALUATE (CPU Parallel)
        # Convert JAX array to numpy for pickling to workers
        population_params = np.array(x) 
        fitness_values = pool.map(_evaluate_params, list(population_params))
        
        # Move fitness back to JAX array
        fitness_array = jnp.array(fitness_values)
        
        # TELL (Accelerator)
        state = tell_step(x, fitness_array, state, es_params)
        
        # Logging
        mean_fit = float(jnp.mean(fitness_array))
        max_fit = float(jnp.max(fitness_array))
        elapsed = time.time() - start_time
        
        if gen % 10 == 0 or gen == 1:
            print(f"Gen {gen}/{args.generations} | Mean: {mean_fit:.2f} | Max: {max_fit:.2f} | Time: {elapsed:.1f}s")
            
        with rewards_file.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([gen, mean_fit, max_fit, elapsed])

        # Save Best Model
        if max_fit > best_fitness_all_time:
            best_fitness_all_time = max_fit
            best_idx = int(jnp.argmax(fitness_array))
            best_params_flat = population_params[best_idx]
            np.savez(run_dir / "best_model.npz", flat_params=best_params_flat)
            np.savez(run_dir / "latest_model.npz", flat_params=best_params_flat)

        # Periodic Checkpoint (Mean Policy)
        if gen % args.eval_freq == 0:
            mean_params = np.array(state.mean)
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            np.savez(checkpoint_dir / f"gen_{gen}.npz", flat_params=mean_params)

    # Cleanup
    pool.close()
    pool.join()
    
    # Save final hyperparameters
    hyperparams = {
        "generations": args.generations,
        "popsize": args.popsize,
        "episode_length": args.episode_length,
        "learning_rate": args.learning_rate,
        "sigma_init": args.sigma_init,
        "seed": args.seed,
        "n_workers": args.n_workers
    }
    save_config_with_hyperparameters(run_dir, args.config, "openai_es", hyperparams)
    
    print(f"Training complete. Artifacts saved to {run_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train OpenAI-ES on NCS env.")
    parser.add_argument("--config", type=Path, default=None, help="Config JSON path.")
    parser.add_argument("--generations", type=int, default=100, help="Number of generations.")
    parser.add_argument("--popsize", type=int, default=128, help="Population size.")
    parser.add_argument("--episode-length", type=int, default=1000, help="Episode length.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate.")
    parser.add_argument("--lrate-decay", type=float, default=0.999, help="Learning rate decay.")
    parser.add_argument("--sigma-init", type=float, default=0.1, help="Initial sigma.")
    parser.add_argument("--sigma-decay", type=float, default=0.999, help="Sigma decay.")
    parser.add_argument("--n-workers", type=int, default=multiprocessing.cpu_count(), help="Number of parallel workers.")
    parser.add_argument("--eval-freq", type=int, default=50, help="Checkpoint frequency.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="Base directory where run artifacts will be stored.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main_args = parse_args()
    train(main_args)
