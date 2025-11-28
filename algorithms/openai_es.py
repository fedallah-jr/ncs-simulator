"""
OpenAI-ES training for the NCS environment using evosax and JAX.

Compatible with TPU/GPU acceleration for the evolution strategy updates.
Evaluates the population in parallel on CPU workers.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import multiprocessing
from pathlib import Path
from typing import Optional, List, Any, Dict

import numpy as np
import gymnasium as gym

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ncs_env.env import NCS_Env
from utils import SingleAgentWrapper
from utils.run_utils import prepare_run_directory, save_config_with_hyperparameters

# -----------------------------------------------------------------------------
# NOTE: JAX/Flax imports are DEFERRED to avoid TPU initialization issues
# in worker processes. They are imported lazily inside functions that need them.
# -----------------------------------------------------------------------------

# Global variables for worker process (initialized lazily)
_worker_env: Optional[gym.Env] = None
_worker_model: Any = None  # Will be PolicyNet instance
_worker_params: Any = None  # Cached params structure for unraveling


def _create_policy_net(action_dim: int):
    """
    Create a PolicyNet instance. Imports Flax lazily.
    
    This function should only be called AFTER setting JAX platform to CPU
    in worker processes.
    """
    import flax.linen as nn
    
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
    
    return PolicyNet(action_dim=action_dim)


def _init_worker(
    config_path: Optional[str], 
    episode_length: int, 
    seed: Optional[int],
    action_dim: int,
    obs_dim: int
):
    """
    Initialize the environment and model in the worker process.
    
    CRITICAL: Must set JAX platform to CPU before importing JAX to avoid
    TPU conflicts when the main process is using the TPU.
    """
    global _worker_env, _worker_model, _worker_params
    
    # Set JAX to use CPU BEFORE importing JAX
    # This must happen before any JAX import to take effect
    os.environ["JAX_PLATFORMS"] = "cpu"
    
    # Now we can safely import JAX
    import jax
    import jax.numpy as jnp
    
    # Also explicitly set via config (belt and suspenders)
    jax.config.update("jax_platform_name", "cpu")
    
    # Create environment
    def factory():
        return NCS_Env(
            n_agents=1,
            episode_length=episode_length,
            config_path=config_path,
            seed=seed,
        )
    
    _worker_env = SingleAgentWrapper(factory)
    
    # Create model (imports Flax lazily)
    _worker_model = _create_policy_net(action_dim)
    
    # Initialize dummy params to get the structure for unraveling
    rng = jax.random.PRNGKey(0)
    dummy_obs = jnp.zeros((1, obs_dim))
    _worker_params = _worker_model.init(rng, dummy_obs)


def _evaluate_params(flat_params: np.ndarray) -> float:
    """
    Run one episode with the given flattened parameters.
    
    Args:
        flat_params: Flattened numpy array of network parameters
        
    Returns:
        Total episode reward
    """
    global _worker_env, _worker_model, _worker_params
    
    # Import JAX (already configured for CPU in _init_worker)
    import jax
    import jax.numpy as jnp
    from jax import flatten_util
    
    # Unravel flat params to pytree structure
    _, unravel_fn = flatten_util.ravel_pytree(_worker_params)
    params = unravel_fn(flat_params)
    
    obs, _ = _worker_env.reset()
    total_reward = 0.0
    truncated = False
    terminated = False
    
    while not (truncated or terminated):
        # Forward pass - ensure obs is properly shaped
        obs_jax = jnp.array(obs[np.newaxis, :])
        logits = _worker_model.apply(params, obs_jax)
        action = int(jnp.argmax(logits))
        
        obs, reward, terminated, truncated, _ = _worker_env.step(action)
        total_reward += reward
        
    return total_reward


def train(args):
    """Main training loop."""
    # Import JAX/Flax for main process (uses TPU/GPU if available)
    import jax
    import jax.numpy as jnp
    from jax import flatten_util
    import flax.linen as nn
    import optax
    from evosax.algorithms.distribution_based import Open_ES
    
    # 1. Setup Dummy Environment to get shapes
    config_path_str = str(args.config) if args.config is not None else None
    dummy_env = NCS_Env(n_agents=1, config_path=config_path_str)
    obs_dim = dummy_env.observation_space["agent_0"].shape[0]
    action_dim = dummy_env.action_space["agent_0"].n
    dummy_env.close()
    
    print(f"Observation Dim: {obs_dim}, Action Dim: {action_dim}")
    
    # Log hardware acceleration status
    devices = jax.devices()
    print(f"JAX devices available: {devices}")
    default_backend = jax.default_backend()
    print(f"Default backend: {default_backend}")
    if default_backend == "tpu":
        print("✓ TPU acceleration ENABLED for ES updates (ASK/TELL)")
    elif default_backend == "gpu":
        print("✓ GPU acceleration ENABLED for ES updates (ASK/TELL)")
    else:
        print("⚠ Running on CPU only (no TPU/GPU detected)")
    print(f"Fitness evaluations will run on {args.n_workers} CPU workers in parallel")

    # 2. Setup Flax Model & Initial Params
    rng = jax.random.PRNGKey(args.seed if args.seed is not None else 0)
    model = _create_policy_net(action_dim)
    dummy_obs = jnp.zeros((1, obs_dim))
    params = model.init(rng, dummy_obs)
    
    # Create flattener for parameter handling
    flat_params_template, unravel_fn = flatten_util.ravel_pytree(params)
    num_params = flat_params_template.size
    print(f"Total Parameters: {num_params}")

    # 3. Setup Open_ES Strategy
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
        solution=params,
        optimizer=optimizer,
        std_schedule=std_schedule
    )
    
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

    # 4. Setup Parallel Workers (using spawn to ensure clean processes)
    ctx = multiprocessing.get_context("spawn")
    pool = ctx.Pool(
        processes=args.n_workers,
        initializer=_init_worker,
        initargs=(config_path_str, args.episode_length, args.seed, action_dim, obs_dim)
    )

    # 5. Logging Setup
    run_dir, metadata = prepare_run_directory("openai_es", args.config, args.output_root)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    
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
        
        # ASK (on TPU/GPU) - get population of params as pytree
        x, state = ask_step(rng_ask, state, es_params)
        
        # Convert population to list of flat numpy arrays for workers
        # x is a pytree where each leaf has shape (popsize, ...)
        population_flat = []
        for i in range(args.popsize):
            # Extract individual i from the population pytree
            individual = jax.tree_util.tree_map(lambda arr: arr[i], x)
            # Flatten to 1D array
            flat_individual, _ = flatten_util.ravel_pytree(individual)
            # Convert to numpy for worker processes
            population_flat.append(np.array(flat_individual))
        
        # EVALUATE (on CPU workers in parallel)
        fitness_values = pool.map(_evaluate_params, population_flat)
        
        # Convert fitness back to JAX array
        fitness_array = jnp.array(fitness_values)
        
        # TELL (on TPU/GPU) - update strategy
        rng, rng_tell = jax.random.split(rng)
        state, metrics = tell_step(rng_tell, x, fitness_array, state, es_params)
        
        # Logging
        mean_fit = float(jnp.mean(fitness_array))
        max_fit = float(jnp.max(fitness_array))
        elapsed = time.time() - start_time
        
        if gen % 3 == 0 or gen == 1:
            print(f"Gen {gen}/{args.generations} | Mean: {mean_fit:.2f} | Max: {max_fit:.2f} | Time: {elapsed:.1f}s")
            
        with rewards_file.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([gen, mean_fit, max_fit, elapsed])

        # Save Best Model
        if max_fit > best_fitness_all_time:
            best_fitness_all_time = max_fit
            best_idx = int(jnp.argmax(fitness_array))
            best_flat = population_flat[best_idx]
            np.savez(run_dir / "best_model.npz", flat_params=best_flat)
            
        # Always save latest (mean of distribution)
        # state.mean is the flat mean vector in Open_ES
        np.savez(run_dir / "latest_model.npz", flat_params=np.array(state.mean))

        # Periodic Checkpoint
        if gen % args.eval_freq == 0:
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
        "lrate_decay": args.lrate_decay,
        "sigma_init": args.sigma_init,
        "sigma_decay": args.sigma_decay,
        "seed": args.seed,
        "n_workers": args.n_workers,
        "eval_freq": args.eval_freq,
    }
    save_config_with_hyperparameters(run_dir, args.config, "openai_es", hyperparams)
    
    print(f"\nTraining complete. Artifacts saved to {run_dir}")
    print(f"Best fitness achieved: {best_fitness_all_time:.2f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train OpenAI-ES on NCS env.")
    parser.add_argument("--config", type=Path, default=None, help="Config JSON path.")
    parser.add_argument("--generations", type=int, default=100, help="Number of generations.")
    parser.add_argument("--popsize", type=int, default=128, help="Population size.")
    parser.add_argument("--episode-length", type=int, default=750, help="Episode length.")
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