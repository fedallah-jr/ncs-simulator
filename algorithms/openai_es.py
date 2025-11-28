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
from typing import Optional, List, Any

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import gymnasium as gym
import optax
from evosax.algorithms.distribution_based import Open_ES

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

def _init_worker(
    config_path: Optional[str], 
    episode_length: int, 
    seed: Optional[int],
    action_dim: int
):
    jax.config.update("jax_platform_name", "cpu")
    """Initialize the environment in the worker process."""
    global _worker_env, _worker_model
    
    def factory():
        return NCS_Env(
            n_agents=1,
            episode_length=episode_length,
            config_path=config_path,
            seed=seed,
        )
    
    _worker_env = SingleAgentWrapper(factory)
    _worker_model = PolicyNet(action_dim=action_dim)


def _evaluate_params(params: Any) -> float:
    """Run one episode with the given structured parameters."""
    global _worker_env, _worker_model
    
    obs, _ = _worker_env.reset()
    total_reward = 0.0
    truncated = False
    terminated = False
    
    while not (truncated or terminated):
        # Simple forward pass
        # Params are already structured (PyTree of numpy arrays or jax arrays)
        # We pass them directly to apply.
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

    # 2. Setup Flax Model & Initial Params
    rng = jax.random.PRNGKey(args.seed if args.seed is not None else 0)
    model = PolicyNet(action_dim=action_dim)
    dummy_obs = jnp.zeros((1, obs_dim))
    params = model.init(rng, dummy_obs)
    
    # Create flattener for saving/logging purposes
    flat_params_dummy, unravel_fn = jax.flatten_util.ravel_pytree(params)
    print(f"Total Parameters: {flat_params_dummy.size}")

    # 3. Setup Open_ES Strategy
    
    # Create schedules for learning rate and sigma
    # Note: transition_steps=1 means decay is applied every step (generation)
    lrate_schedule = optax.exponential_decay(
        init_value=args.learning_rate, 
        transition_steps=1, 
        decay_rate=args.lrate_decay
    )
    
    std_schedule = optax.exponential_decay(
        init_value=args.sigma_init, 
        transition_steps=1, 
        decay_rate=args.sigma_decay
    )
    
    optimizer = optax.adam(learning_rate=lrate_schedule)

    strategy = Open_ES(
        population_size=args.popsize,
        solution=params, # Template params
        optimizer=optimizer,
        std_schedule=std_schedule
    )
    
    es_params = strategy.default_params
    es_params = strategy.default_params
    state = strategy.init(rng, params, es_params)

    # JIT the ASK and TELL steps for acceleration
    @jax.jit
    def ask_step(rng, state, params):
        x, state = strategy.ask(rng, state, params)
        return x, state

    @jax.jit
    def tell_step(rng, x, fitness, state, params):
        return strategy.tell(rng, x, fitness, state, params)

    # 4. Setup Parallel Workers
    ctx = multiprocessing.get_context("spawn")
    pool = ctx.Pool(
        processes=args.n_workers,
        initializer=_init_worker,
        initargs=(config_path_str, args.episode_length, args.seed, action_dim)
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

    # Helper to slice pytree of batches into list of pytrees
    def slice_population(pop_tree, size):
        # pop_tree leaves are (size, ...)
        return [jax.tree_util.tree_map(lambda x: x[i], pop_tree) for i in range(size)]

    # 6. Training Loop
    for gen in range(1, args.generations + 1):
        rng, rng_ask = jax.random.split(rng)
        
        # ASK (Accelerator)
        # x is structured population (PyTree where leaves have leading batch dim)
        x, state = ask_step(rng_ask, state, es_params)
        
        # EVALUATE (CPU Parallel)
        # Move to CPU and convert to numpy
        x_cpu = jax.tree_util.tree_map(lambda arr: np.array(arr), x)
        
        # Slice into individual parameters
        population_params_list = slice_population(x_cpu, args.popsize)
        
        fitness_values = pool.map(_evaluate_params, population_params_list)
        
        # Move fitness back to JAX array
        fitness_array = jnp.array(fitness_values)
        
        # TELL (Accelerator)
        rng, rng_tell = jax.random.split(rng)
        state, metrics = tell_step(rng_tell, x, fitness_array, state, es_params)
        
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
            
            # Get best params (structured)
            best_params = population_params_list[best_idx]
            
            # Flatten for saving (compatibility)
            flat_best, _ = jax.flatten_util.ravel_pytree(best_params)
            np.savez(run_dir / "best_model.npz", flat_params=flat_best)
            np.savez(run_dir / "latest_model.npz", flat_params=flat_best)

        # Periodic Checkpoint (Mean Policy)
        if gen % args.eval_freq == 0:
            # Mean params from state
            mean_params = state.mean # Wait, state.mean is FLAT in evosax? 
            # Let's check State definition in open_es.py
            # class State(BaseState): mean: jax.Array
            # Yes, mean is a flat array! 
            # Because Open_ES manages flat mean internally but ask/tell handle ravel/unravel.
            
            # So state.mean is ALREADY flat!
            # We can save it directly.
            # But wait, ask returns UNRAVELED population.
            # So state.mean corresponds to flattened params.
            
            np.savez(run_dir / "checkpoints" / f"gen_{gen}.npz", flat_params=np.array(state.mean))

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