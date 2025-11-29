"""
OpenAI-ES training for the NCS environment using evosax and JAX.

Compatible with TPU/GPU acceleration for the evolution strategy updates.
Evaluates the population in parallel on CPU workers.

Key features:
1. Anti-overfitting: Each individual is evaluated on multiple episodes
   with different seeds, and the mean fitness is used.
2. Fitness shaping: Uses evosax's built-in fitness shaping (centered_rank by default).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import multiprocessing
from pathlib import Path
from typing import Optional, List, Any, Tuple

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


def create_policy_net(action_dim: int, hidden_size: int = 64, use_layer_norm: bool = False):
    """
    Create a PolicyNet instance. Imports Flax lazily.
    
    Args:
        action_dim: Number of output actions
        hidden_size: Number of units in hidden layers
        use_layer_norm: Whether to use Layer Normalization
    """
    import flax.linen as nn
    
    class PolicyNet(nn.Module):
        action_dim: int
        hidden_size: int = 64
        use_layer_norm: bool = False

        @nn.compact
        def __call__(self, x):
            x = nn.Dense(self.hidden_size)(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.tanh(x)
            
            x = nn.Dense(self.hidden_size)(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.tanh(x)
            
            x = nn.Dense(self.action_dim)(x)
            return x
    
    return PolicyNet(action_dim=action_dim, hidden_size=hidden_size, use_layer_norm=use_layer_norm)


def _init_worker(
    config_path: Optional[str], 
    episode_length: int, 
    seed: Optional[int],
    action_dim: int,
    obs_dim: int,
    hidden_size: int = 64,
    use_layer_norm: bool = False
):
    """Initialize the environment and model in the worker process."""
    global _worker_env, _worker_model, _worker_params
    
    os.environ["JAX_PLATFORMS"] = "cpu"
    
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_platform_name", "cpu")
    
    def factory():
        return NCS_Env(
            n_agents=1,
            episode_length=episode_length,
            config_path=config_path,
            seed=seed,
        )
    
    _worker_env = SingleAgentWrapper(factory)
    _worker_model = create_policy_net(action_dim, hidden_size=hidden_size, use_layer_norm=use_layer_norm)
    
    rng = jax.random.PRNGKey(0)
    dummy_obs = jnp.zeros((1, obs_dim))
    _worker_params = _worker_model.init(rng, dummy_obs)


def _run_single_episode(flat_params: np.ndarray, episode_seed: int) -> float:
    """Run one episode with the given flattened parameters and seed."""
    global _worker_env, _worker_model, _worker_params
    
    import jax.numpy as jnp
    from jax import flatten_util
    
    _, unravel_fn = flatten_util.ravel_pytree(_worker_params)
    params = unravel_fn(flat_params)
    
    obs, _ = _worker_env.reset(seed=episode_seed)
    total_reward = 0.0
    truncated = False
    terminated = False
    
    while not (truncated or terminated):
        obs_jax = jnp.array(obs[np.newaxis, :])
        logits = _worker_model.apply(params, obs_jax)
        action = int(jnp.argmax(logits))
        
        obs, reward, terminated, truncated, _ = _worker_env.step(action)
        total_reward += reward
        
    return total_reward


def _evaluate_params_multi_episode(args_tuple: Tuple[np.ndarray, List[int]]) -> float:
    """Run multiple episodes and return mean reward (anti-overfitting)."""
    flat_params, episode_seeds = args_tuple
    rewards = [_run_single_episode(flat_params, seed) for seed in episode_seeds]
    return float(np.mean(rewards))


def get_fitness_shaping_fn(method: str):
    """
    Get a fitness shaping function from evosax.
    
    Args:
        method: Shaping method. Options:
            - "centered_rank": Rank-based, maps to [-0.5, 0.5] (default, recommended)
            - "z_score": Standardize to mean=0, std=1  
            - "normalize": Normalize to [0, 1] range
            - "none": No shaping (identity)
            - "weights": CMA-ES style weighted recombination
        
    Returns:
        Fitness shaping function compatible with evosax strategies
    """
    from evosax.core.fitness_shaping import (
        centered_rank_fitness_shaping_fn,
        standardize_fitness_shaping_fn,
        normalize_fitness_shaping_fn,
        identity_fitness_shaping_fn,
        weights_fitness_shaping_fn,
    )
    
    mapping = {
        "centered_rank": centered_rank_fitness_shaping_fn,
        "z_score": standardize_fitness_shaping_fn,
        "normalize": normalize_fitness_shaping_fn,
        "none": identity_fitness_shaping_fn,
        "weights": weights_fitness_shaping_fn,
    }
    
    if method not in mapping:
        raise ValueError(f"Unknown method: {method}. Options: {list(mapping.keys())}")
    
    return mapping[method]


def train(args):
    """Main training loop."""
    import jax
    import jax.numpy as jnp
    from jax import flatten_util
    import optax
    from evosax.algorithms.distribution_based import Open_ES
    
    # 1. Setup environment to get shapes
    config_path_str = str(args.config) if args.config is not None else None
    dummy_env = NCS_Env(n_agents=1, config_path=config_path_str)
    obs_dim = dummy_env.observation_space["agent_0"].shape[0]
    action_dim = dummy_env.action_space["agent_0"].n
    dummy_env.close()
    
    print(f"Observation Dim: {obs_dim}, Action Dim: {action_dim}")
    print(f"Hidden Size: {args.hidden_size}, Layer Norm: {args.use_layer_norm}")
    print(f"Evaluation episodes per individual: {args.eval_episodes}")
    print(f"Fitness shaping: {args.fitness_shaping} (evosax built-in)")
    
    # Log hardware
    devices = jax.devices()
    print(f"JAX devices: {devices}")
    default_backend = jax.default_backend()
    print(f"Default backend: {default_backend}")
    if default_backend in ("tpu", "gpu"):
        print(f"✓ {default_backend.upper()} acceleration ENABLED")
    else:
        print("⚠ Running on CPU only")
    print(f"Workers: {args.n_workers}")

    # 2. Setup model
    rng = jax.random.PRNGKey(args.seed if args.seed is not None else 0)
    model = create_policy_net(action_dim, hidden_size=args.hidden_size, use_layer_norm=args.use_layer_norm)
    dummy_obs = jnp.zeros((1, obs_dim))
    params = model.init(rng, dummy_obs)
    
    flat_params_template, unravel_fn = flatten_util.ravel_pytree(params)
    print(f"Total Parameters: {flat_params_template.size}")

    # 3. Setup Open_ES with evosax's built-in fitness shaping
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
    
    # Get fitness shaping function from evosax
    fitness_shaping_fn = get_fitness_shaping_fn(args.fitness_shaping)

    # Create strategy with fitness shaping built-in
    strategy = Open_ES(
        population_size=args.popsize,
        solution=params,
        optimizer=optimizer,
        std_schedule=std_schedule,
        fitness_shaping_fn=fitness_shaping_fn,  # evosax handles this internally
    )
    
    es_params = strategy.default_params
    state = strategy.init(rng, params, es_params)

    # JIT the ASK and TELL steps
    @jax.jit
    def ask_step(rng, state, params):
        return strategy.ask(rng, state, params)

    @jax.jit
    def tell_step(rng, x, fitness, state, params):
        return strategy.tell(rng, x, fitness, state, params)

    # 4. Setup parallel workers
    ctx = multiprocessing.get_context("spawn")
    pool = ctx.Pool(
        processes=args.n_workers,
        initializer=_init_worker,
        initargs=(config_path_str, args.episode_length, args.seed, action_dim, obs_dim,
                  args.hidden_size, args.use_layer_norm)
    )

    # 5. Logging setup
    run_dir, metadata = prepare_run_directory("openai_es", args.config, args.output_root)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    
    rewards_file = run_dir / "training_rewards.csv"
    with rewards_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "mean_reward", "max_reward", "min_reward", "std_reward", "time_elapsed"])

    print(f"Starting training for {args.generations} generations...")
    start_time = time.time()
    best_fitness_all_time = -float("inf")

    # 6. Training loop
    for gen in range(1, args.generations + 1):
        rng, rng_seeds = jax.random.split(rng)
        
        # Generate seeds for multi-episode evaluation (CRN within generation)
        gen_episode_seeds = [
            int(jax.random.randint(
                jax.random.fold_in(rng_seeds, i), 
                (), minval=0, maxval=2**31 - 1, dtype=jnp.int32
            ))
            for i in range(args.eval_episodes)
        ]

        rng, rng_ask = jax.random.split(rng)
        
        # ASK - get population
        x, state = ask_step(rng_ask, state, es_params)
        
        # Convert to flat arrays for workers
        population_flat = []
        for i in range(args.popsize):
            individual = jax.tree_util.tree_map(lambda arr: arr[i], x)
            flat_individual, _ = flatten_util.ravel_pytree(individual)
            population_flat.append(np.array(flat_individual))
        
        # EVALUATE in parallel
        eval_payloads = [(fp, gen_episode_seeds) for fp in population_flat]
        fitness_values = pool.map(_evaluate_params_multi_episode, eval_payloads)
        fitness_array = jnp.array(fitness_values)
        
        # TELL - evosax applies fitness shaping internally
        rng, rng_tell = jax.random.split(rng)
        state, metrics = tell_step(rng_tell, x, fitness_array, state, es_params)
        
        # Logging
        mean_fit = float(jnp.mean(fitness_array))
        max_fit = float(jnp.max(fitness_array))
        min_fit = float(jnp.min(fitness_array))
        std_fit = float(jnp.std(fitness_array))
        elapsed = time.time() - start_time
        
        if gen % 1 == 0 or gen == 1:
            print(f"Gen {gen}/{args.generations} | Mean: {mean_fit:.1f} | Max: {max_fit:.1f} | Std: {std_fit:.1f} | Time: {elapsed:.1f}s")
            
        with rewards_file.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([gen, mean_fit, max_fit, min_fit, std_fit, elapsed])

        # Save best model
        if max_fit > best_fitness_all_time:
            best_fitness_all_time = max_fit
            best_idx = int(jnp.argmax(fitness_array))
            np.savez(
                run_dir / "best_model.npz",
                flat_params=population_flat[best_idx],
                hidden_size=args.hidden_size,
                use_layer_norm=args.use_layer_norm
            )
            
        # Save latest
        np.savez(
            run_dir / "latest_model.npz",
            flat_params=np.array(state.mean),
            hidden_size=args.hidden_size,
            use_layer_norm=args.use_layer_norm
        )

        # Checkpoint
        if gen % args.checkpoint_freq == 0:
            np.savez(
                run_dir / "checkpoints" / f"gen_{gen}.npz",
                flat_params=np.array(state.mean),
                hidden_size=args.hidden_size,
                use_layer_norm=args.use_layer_norm
            )

    pool.close()
    pool.join()
    
    # Save hyperparameters
    hyperparams = {
        "generations": args.generations,
        "popsize": args.popsize,
        "episode_length": args.episode_length,
        "eval_episodes": args.eval_episodes,
        "fitness_shaping": args.fitness_shaping,
        "hidden_size": args.hidden_size,
        "use_layer_norm": args.use_layer_norm,
        "learning_rate": args.learning_rate,
        "lrate_decay": args.lrate_decay,
        "sigma_init": args.sigma_init,
        "sigma_decay": args.sigma_decay,
        "seed": args.seed,
        "n_workers": args.n_workers,
        "checkpoint_freq": args.checkpoint_freq,
    }
    save_config_with_hyperparameters(run_dir, args.config, "openai_es", hyperparams)
    
    print(f"\nTraining complete. Artifacts saved to {run_dir}")
    print(f"Best fitness: {best_fitness_all_time:.2f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train OpenAI-ES on NCS env.")
    parser.add_argument("--config", type=Path, default=None, help="Config JSON path.")
    parser.add_argument("--generations", type=int, default=100, help="Number of generations.")
    parser.add_argument("--popsize", type=int, default=132, help="Population size.")
    parser.add_argument("--episode-length", type=int, default=500, help="Episode length.")
    parser.add_argument("--eval-episodes", type=int, default=3, 
                        help="Episodes per individual (anti-overfitting). Default: 3")
    parser.add_argument("--fitness-shaping", type=str, default="weights",
                        choices=["centered_rank", "z_score", "normalize", "none", "weights"],
                        help="Fitness shaping (evosax built-in). Default: centered_rank")
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden layer size.")
    parser.add_argument("--use-layer-norm", action="store_true", help="Use LayerNorm.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate.")
    parser.add_argument("--lrate-decay", type=float, default=0.999, help="LR decay.")
    parser.add_argument("--sigma-init", type=float, default=0.25, help="Initial sigma.")
    parser.add_argument("--sigma-decay", type=float, default=0.99, help="Sigma decay.")
    parser.add_argument("--n-workers", type=int, default=multiprocessing.cpu_count(), help="Workers.")
    parser.add_argument("--checkpoint-freq", type=int, default=5, help="Checkpoint frequency.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"), help="Output directory.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())