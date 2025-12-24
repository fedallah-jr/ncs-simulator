"""
OpenAI-ES training for the NCS environment using evosax and JAX.

Compatible with TPU/GPU acceleration for the evolution strategy updates.
Evaluates the population in parallel on CPU workers.

Key features:
1. Anti-overfitting: Each individual is evaluated on multiple episodes
   with different seeds, and the mean fitness is used.
2. Fitness shaping: Uses evosax built-in fitness shaping functions
   (applied automatically in the tell() method).
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ncs_env.env import NCS_Env
from ncs_env.config import load_config
from utils.run_utils import prepare_run_directory, save_config_with_hyperparameters

# -----------------------------------------------------------------------------
# NOTE: JAX/Flax imports are DEFERRED to avoid TPU initialization issues
# in worker processes. They are imported lazily inside functions that need them.
# -----------------------------------------------------------------------------

# Global variables for worker process (initialized lazily)
_worker_env: Optional[gym.Env] = None
_worker_model: Any = None
_worker_params: Any = None
_worker_config_path: Optional[str] = None
_worker_episode_length: int = 500
_worker_n_agents: int = 1
_worker_use_agent_id: bool = False
_worker_agent_id_eye: Optional[np.ndarray] = None


def _build_obs_batch(obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
    obs_batch = np.stack(
        [np.asarray(obs_dict[f"agent_{i}"], dtype=np.float32) for i in range(_worker_n_agents)],
        axis=0,
    )
    if _worker_use_agent_id:
        if _worker_agent_id_eye is None:
            raise RuntimeError("Agent id eye matrix not initialized.")
        obs_batch = np.concatenate([obs_batch, _worker_agent_id_eye], axis=1)
    return obs_batch


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
    n_agents: int,
    use_agent_id: bool,
    hidden_size: int = 64,
    use_layer_norm: bool = False
):
    """Initialize the environment and model in the worker process."""
    global _worker_env, _worker_model, _worker_params, _worker_config_path, _worker_episode_length
    global _worker_n_agents, _worker_use_agent_id, _worker_agent_id_eye
    
    # Set JAX to use CPU BEFORE importing JAX
    os.environ["JAX_PLATFORMS"] = "cpu"
    
    import jax
    import jax.numpy as jnp
    
    jax.config.update("jax_platform_name", "cpu")
    
    _worker_config_path = config_path
    _worker_episode_length = episode_length
    _worker_n_agents = int(n_agents)
    _worker_use_agent_id = bool(use_agent_id)
    _worker_agent_id_eye = np.eye(_worker_n_agents, dtype=np.float32) if _worker_use_agent_id else None
    
    _worker_env = NCS_Env(
        n_agents=_worker_n_agents,
        episode_length=episode_length,
        config_path=config_path,
        seed=seed,
    )
    _worker_model = create_policy_net(action_dim, hidden_size=hidden_size, use_layer_norm=use_layer_norm)
    
    rng = jax.random.PRNGKey(0)
    input_dim = obs_dim + (_worker_n_agents if _worker_use_agent_id else 0)
    dummy_obs = jnp.zeros((1, input_dim))
    _worker_params = _worker_model.init(rng, dummy_obs)


def _run_single_episode(flat_params: np.ndarray, episode_seed: int) -> float:
    """Run one episode with the given flattened parameters and seed."""
    global _worker_env, _worker_model, _worker_params, _worker_n_agents
    
    import jax.numpy as jnp
    from jax import flatten_util
    
    _, unravel_fn = flatten_util.ravel_pytree(_worker_params)
    params = unravel_fn(flat_params)
    
    obs_dict, _ = _worker_env.reset(seed=episode_seed)
    total_reward = 0.0
    done = False

    while not done:
        obs_batch = _build_obs_batch(obs_dict)
        obs_jax = jnp.asarray(obs_batch)
        logits = _worker_model.apply(params, obs_jax)
        actions = np.asarray(jnp.argmax(logits, axis=1))
        action_dict = {f"agent_{i}": int(actions[i]) for i in range(_worker_n_agents)}

        obs_dict, rewards, terminated, truncated, _ = _worker_env.step(action_dict)
        step_reward = float(np.sum([rewards[f"agent_{i}"] for i in range(_worker_n_agents)]))
        total_reward += step_reward
        done = any(terminated[f"agent_{i}"] or truncated[f"agent_{i}"] for i in range(_worker_n_agents))
        
    return total_reward


def _evaluate_params_multi_episode(args_tuple: Tuple[np.ndarray, List[int]]) -> float:
    """Run multiple episodes with different seeds and return mean reward."""
    flat_params, episode_seeds = args_tuple
    
    rewards = []
    for seed in episode_seeds:
        reward = _run_single_episode(flat_params, seed)
        rewards.append(reward)
    
    return float(np.mean(rewards))


def get_fitness_shaping_fn(method: str = "centered_rank"):
    """
    Get a fitness shaping function from evosax.
    
    Args:
        method: Shaping method. Options:
            - "centered_rank": Rank-based, maps to [-0.5, 0.5] (default, recommended)
            - "z_score": Standardize to mean=0, std=1
            - "normalize": Normalize to [0, 1] range
            - "none": No shaping (identity)
        
    Returns:
        Fitness shaping function compatible with evosax
    """
    from evosax.core.fitness_shaping import (
        centered_rank_fitness_shaping_fn,
        standardize_fitness_shaping_fn,
        normalize_fitness_shaping_fn,
        identity_fitness_shaping_fn,
    )
    
    shaping_fns = {
        "centered_rank": centered_rank_fitness_shaping_fn,
        "z_score": standardize_fitness_shaping_fn,
        "normalize": normalize_fitness_shaping_fn,
        "none": identity_fitness_shaping_fn,
    }
    
    if method not in shaping_fns:
        available = ", ".join(shaping_fns.keys())
        raise ValueError(f"Unknown fitness shaping method: {method}. Available: {available}")
    
    return shaping_fns[method]


def train(args):
    """Main training loop."""
    import jax
    import jax.numpy as jnp
    from jax import flatten_util
    import optax
    from evosax.algorithms.distribution_based import Open_ES
    
    if args.meta_population_size < 1:
        raise ValueError("meta_population_size must be at least 1.")
    if not 0.0 <= args.truncation_percentage <= 0.5:
        raise ValueError("truncation_percentage must be between 0.0 and 0.5.")
    if args.pbt_interval < 1:
        raise ValueError("pbt_interval must be at least 1.")

    config_path_str = str(args.config) if args.config is not None else None
    cfg = load_config(config_path_str)
    system_cfg = cfg.get("system", {})
    n_agents = int(system_cfg.get("n_agents", 1))
    use_agent_id = n_agents > 1

    # 1. Setup Dummy Environment to get shapes
    dummy_env = NCS_Env(n_agents=n_agents, config_path=config_path_str)
    obs_dim = dummy_env.observation_space["agent_0"].shape[0]
    action_dim = dummy_env.action_space["agent_0"].n
    dummy_env.close()
    input_dim = obs_dim + (n_agents if use_agent_id else 0)
    
    print(f"Observation Dim: {obs_dim}, Input Dim: {input_dim}, Action Dim: {action_dim}")
    print(f"N Agents: {n_agents}, Use Agent Id: {use_agent_id}")
    print(f"Hidden Size: {args.hidden_size}, Layer Norm: {args.use_layer_norm}")
    print(f"Evaluation episodes per individual: {args.eval_episodes}")
    print(f"Fitness shaping: {args.fitness_shaping} (evosax built-in)")
    if args.meta_population_size > 1:
        print(
            f"Meta population: {args.meta_population_size} strategies | "
            f"Truncation: {args.truncation_percentage:.2f} every {args.pbt_interval} generations"
        )
    
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
    master_rng = jax.random.PRNGKey(args.seed if args.seed is not None else 0)
    model = create_policy_net(action_dim, hidden_size=args.hidden_size, use_layer_norm=args.use_layer_norm)
    dummy_obs = jnp.zeros((1, input_dim))
    master_rng, template_rng = jax.random.split(master_rng)
    params_template = model.init(template_rng, dummy_obs)
    
    flat_params_template, _ = flatten_util.ravel_pytree(params_template)
    num_params = flat_params_template.size
    print(f"Total Parameters: {num_params}")

    # 3. Setup Open_ES Strategy with built-in fitness shaping
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
    fitness_shaping_fn = get_fitness_shaping_fn(args.fitness_shaping)

    def build_strategy_context(init_key: Any, param_key: Any) -> Dict[str, Any]:
        """Construct an independent Open_ES strategy with its own initialization."""
        initial_params = model.init(param_key, dummy_obs)
        strategy = Open_ES(
            population_size=args.popsize,
            solution=initial_params,
            optimizer=optimizer,
            std_schedule=std_schedule,
            fitness_shaping_fn=fitness_shaping_fn,
        )
        es_params = strategy.default_params
        state = strategy.init(init_key, initial_params, es_params)

        @jax.jit
        def ask_step(rng_key, es_state):
            x, es_state = strategy.ask(rng_key, es_state, es_params)
            return x, es_state

        @jax.jit
        def tell_step(rng_key, x, fitness, es_state):
            return strategy.tell(rng_key, x, fitness, es_state, es_params)

        return {
            "strategy": strategy,
            "es_params": es_params,
            "state": state,
            "ask_step": ask_step,
            "tell_step": tell_step,
            "mean_fitness": -float("inf"),
        }

    meta_strategies: List[Dict[str, Any]] = []
    strategy_rngs: List[Any] = []
    for strat_idx in range(args.meta_population_size):
        master_rng, init_key, param_key, loop_key = jax.random.split(master_rng, 4)
        meta_strategies.append(build_strategy_context(init_key, param_key))
        strategy_rngs.append(loop_key)

    # 4. Setup Parallel Workers
    pool = None
    map_fn = None
    if args.n_workers and args.n_workers > 1:
        ctx = multiprocessing.get_context("spawn")
        pool = ctx.Pool(
            processes=args.n_workers,
            initializer=_init_worker,
            initargs=(
                config_path_str,
                args.episode_length,
                args.seed,
                action_dim,
                obs_dim,
                n_agents,
                use_agent_id,
                args.hidden_size,
                args.use_layer_norm,
            ),
        )
        map_fn = pool.map
    else:
        def map_fn(func, iterable):
            return list(map(func, iterable))
        _init_worker(
            config_path_str,
            args.episode_length,
            args.seed,
            action_dim,
            obs_dim,
            n_agents,
            use_agent_id,
            args.hidden_size,
            args.use_layer_norm,
        )

    # 5. Logging Setup
    run_dir = prepare_run_directory("openai_es", args.config, args.output_root)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    
    rewards_file = run_dir / "training_rewards.csv"
    with rewards_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "mean_reward", "max_reward", "min_reward", "std_reward", "time_elapsed"])

    print(f"Starting training for {args.generations} generations...")
    start_time = time.time()
    best_fitness_all_time = -float("inf")

    def copy_tree(tree: Any) -> Any:
        """Deep copy a pytree so PBT resets do not alias state."""
        return jax.tree_util.tree_map(
            lambda v: jnp.array(v, copy=True) if isinstance(v, (jnp.ndarray, np.ndarray)) else (v.copy() if hasattr(v, "copy") else v),
            tree,
        )

    # 6. Training Loop
    for gen in range(1, args.generations + 1):
        master_rng, rng_seeds = jax.random.split(master_rng)
        
        # Generate seeds for multi-episode evaluation (CRN within generation)
        gen_episode_seeds = [
            int(jax.random.randint(
                jax.random.fold_in(rng_seeds, i), 
                (), minval=0, maxval=2**31 - 1, dtype=jnp.int32
            ))
            for i in range(args.eval_episodes)
        ]

        all_fitness_values: List[float] = []

        for strat_idx, context in enumerate(meta_strategies):
            strategy_rngs[strat_idx], rng_ask = jax.random.split(strategy_rngs[strat_idx])
            x, context_state = context["ask_step"](rng_ask, context["state"])
            
            population_flat = []
            for i in range(args.popsize):
                individual = jax.tree_util.tree_map(lambda arr: arr[i], x)
                flat_individual, _ = flatten_util.ravel_pytree(individual)
                population_flat.append(np.array(flat_individual))
            
            eval_payloads = [(flat_params, gen_episode_seeds) for flat_params in population_flat]
            fitness_values = map_fn(_evaluate_params_multi_episode, eval_payloads)
            fitness_array = jnp.array(fitness_values)
            
            strategy_rngs[strat_idx], rng_tell = jax.random.split(strategy_rngs[strat_idx])
            context_state, _ = context["tell_step"](rng_tell, x, fitness_array, context_state)
            context["state"] = context_state
            
            mean_fit = float(jnp.mean(fitness_array))
            max_fit = float(jnp.max(fitness_array))
            context["mean_fitness"] = mean_fit

            all_fitness_values.extend(fitness_values)

            if max_fit > best_fitness_all_time:
                best_fitness_all_time = max_fit
                best_idx = int(jnp.argmax(fitness_array))
                np.savez(
                    run_dir / "best_model.npz",
                    flat_params=population_flat[best_idx],
                    hidden_size=args.hidden_size,
                    use_layer_norm=args.use_layer_norm,
                    n_agents=n_agents,
                    obs_dim=obs_dim,
                    use_agent_id=use_agent_id,
                )

        all_fitness_array = jnp.array(all_fitness_values)
        mean_fit = float(jnp.mean(all_fitness_array))
        max_fit = float(jnp.max(all_fitness_array))
        min_fit = float(jnp.min(all_fitness_array))
        std_fit = float(jnp.std(all_fitness_array))
        elapsed = time.time() - start_time

        best_strategy_idx = int(np.argmax([ctx["mean_fitness"] for ctx in meta_strategies]))

        if gen % 3 == 0 or gen == 1:
            best_strategy_mean = meta_strategies[best_strategy_idx]["mean_fitness"]
            print(
                f"Gen {gen}/{args.generations} | Global Mean: {mean_fit:.1f} | "
                f"Global Max: {max_fit:.1f} | Min: {min_fit:.1f} | Std: {std_fit:.1f} | "
                f"Best Strategy {best_strategy_idx} Mean: {best_strategy_mean:.1f} | "
                f"Time: {elapsed:.1f}s"
            )
            
        with rewards_file.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([gen, mean_fit, max_fit, min_fit, std_fit, elapsed])

        best_strategy_state = meta_strategies[best_strategy_idx]["state"]
        np.savez(
            run_dir / "latest_model.npz",
            flat_params=np.array(best_strategy_state.mean),
            hidden_size=args.hidden_size,
            use_layer_norm=args.use_layer_norm,
            n_agents=n_agents,
            obs_dim=obs_dim,
            use_agent_id=use_agent_id,
        )

        if gen % args.checkpoint_freq == 0:
            np.savez(
                run_dir / "checkpoints" / f"gen_{gen}.npz",
                flat_params=np.array(best_strategy_state.mean),
                hidden_size=args.hidden_size,
                use_layer_norm=args.use_layer_norm,
                n_agents=n_agents,
                obs_dim=obs_dim,
                use_agent_id=use_agent_id,
            )

        if args.meta_population_size > 1 and args.truncation_percentage > 0.0 and gen % args.pbt_interval == 0:
            n_replace = int(args.meta_population_size * args.truncation_percentage)
            if n_replace > 0:
                ranked_indices = sorted(
                    range(args.meta_population_size),
                    key=lambda idx: meta_strategies[idx]["mean_fitness"]
                )
                bottom_indices = ranked_indices[:n_replace]
                top_indices = ranked_indices[-n_replace:]
                top_indices_array = jnp.array(top_indices)

                for dest_idx in bottom_indices:
                    master_rng, sample_key = jax.random.split(master_rng)
                    source_idx = int(jax.random.choice(sample_key, top_indices_array))
                    source_state = meta_strategies[source_idx]["state"]
                    dest_state = meta_strategies[dest_idx]["state"]
                    meta_strategies[dest_idx]["state"] = dest_state.replace(
                        mean=jnp.array(source_state.mean, copy=True),
                        opt_state=copy_tree(source_state.opt_state),
                        std=jnp.array(source_state.std, copy=True),
                        best_solution=jnp.array(source_state.best_solution, copy=True),
                        best_fitness=jnp.array(source_state.best_fitness, copy=True),
                    )
                    meta_strategies[dest_idx]["mean_fitness"] = meta_strategies[source_idx]["mean_fitness"]
                    master_rng, strategy_rngs[dest_idx] = jax.random.split(master_rng)

    if pool is not None:
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
        "n_agents": n_agents,
        "use_agent_id": use_agent_id,
        "checkpoint_freq": args.checkpoint_freq,
        "meta_population_size": args.meta_population_size,
        "truncation_percentage": args.truncation_percentage,
        "pbt_interval": args.pbt_interval,
    }
    save_config_with_hyperparameters(run_dir, args.config, "openai_es", hyperparams)
    
    print(f"\nTraining complete. Artifacts saved to {run_dir}")
    print(f"Best fitness achieved: {best_fitness_all_time:.2f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train OpenAI-ES on NCS env.")
    parser.add_argument("--config", type=Path, default=None, help="Config JSON path.")
    parser.add_argument("--generations", type=int, default=100, help="Number of generations.")
    parser.add_argument("--popsize", type=int, default=132, help="Population size.")
    parser.add_argument("--episode-length", type=int, default=500, help="Episode length.")
    parser.add_argument("--eval-episodes", type=int, default=3, 
                        help="Episodes per individual (anti-overfitting). Default: 3")
    parser.add_argument("--fitness-shaping", type=str, default="centered_rank",
                        choices=["centered_rank", "z_score", "normalize", "none"],
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
    parser.add_argument("--meta-population-size", type=int, default=1,
                        help="Number of independent Open_ES strategies to evolve in parallel.")
    parser.add_argument("--truncation-percentage", type=float, default=0.2,
                        help="Fraction (0.0-0.5) of strategies replaced during PBT exploitation.")
    parser.add_argument("--pbt-interval", type=int, default=10,
                        help="Generations between truncation selections.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
