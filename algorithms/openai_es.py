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
from utils.reward_normalization import (
    configure_shared_running_normalizers,
    reset_shared_running_normalizers,
)
from utils.bc import BCPretrainConfig, JaxActorBCAdapter, load_bc_dataset, pretrain_actor
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
    use_layer_norm: bool = False,
    shared_normalizer_store: Optional[Any] = None,
    shared_normalizer_lock: Optional[Any] = None,
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

    configure_shared_running_normalizers(shared_normalizer_store, shared_normalizer_lock)
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
    reset_shared_running_normalizers()
    import jax
    import jax.numpy as jnp
    from jax import flatten_util
    import optax
    from evosax.algorithms.distribution_based import Open_ES
    
    if args.bc_epochs > 0 and args.bc_dataset is None:
        raise ValueError("--bc-epochs requires --bc-dataset")
    if args.bc_dataset is not None and args.bc_epochs <= 0:
        raise ValueError("--bc-dataset requires --bc-epochs > 0")
    if args.bc_batch_size <= 0:
        raise ValueError("--bc-batch-size must be positive")
    if args.bc_init_std < 0.0:
        raise ValueError("--bc-init-std must be >= 0")

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
    
    flat_params_template, unravel_fn = flatten_util.ravel_pytree(params_template)
    num_params = flat_params_template.size
    print(f"Total Parameters: {num_params}")

    bc_lr: Optional[float] = None
    pretrained_params: Optional[Any] = None
    pretrained_flat: Optional[np.ndarray] = None
    if args.bc_dataset is not None:
        bc_dataset = load_bc_dataset(args.bc_dataset)
        if bc_dataset.num_steps <= 0:
            raise ValueError("BC dataset is empty.")
        if bc_dataset.obs.shape[2] != obs_dim:
            raise ValueError(
                f"BC obs_dim {bc_dataset.obs.shape[2]} does not match env obs_dim {obs_dim}."
            )
        if bc_dataset.n_agents != n_agents:
            raise ValueError(
                f"BC dataset n_agents {bc_dataset.n_agents} does not match env n_agents {n_agents}."
            )
        meta_n_agents = bc_dataset.metadata.get("n_agents")
        if meta_n_agents is not None and int(meta_n_agents) != n_agents:
            raise ValueError(
                f"BC dataset n_agents {meta_n_agents} does not match env n_agents {n_agents}."
            )
        meta_use_agent_id = bc_dataset.metadata.get("use_agent_id")
        if meta_use_agent_id is not None and bool(meta_use_agent_id) != use_agent_id:
            raise ValueError(
                f"BC dataset use_agent_id {meta_use_agent_id} does not match {use_agent_id}."
            )
        min_action = int(bc_dataset.actions.min())
        max_action = int(bc_dataset.actions.max())
        if min_action < 0 or max_action >= action_dim:
            raise ValueError("BC dataset actions are outside the valid action range.")

        bc_lr = float(args.learning_rate) if args.bc_lr is None else float(args.bc_lr)
        bc_config = BCPretrainConfig(
            epochs=int(args.bc_epochs),
            batch_size=int(args.bc_batch_size),
            learning_rate=bc_lr,
            use_agent_id=use_agent_id,
            n_agents=n_agents,
        )
        bc_rng = np.random.default_rng(args.seed)
        bc_adapter = JaxActorBCAdapter(model.apply, n_agents=n_agents, use_agent_id=use_agent_id)
        bc_result = pretrain_actor(bc_adapter, params_template, bc_dataset, bc_config, bc_rng)
        pretrained_params = bc_result.params
        pretrained_flat = np.array(flatten_util.ravel_pytree(pretrained_params)[0])
        print(f"[BC] Pretrained actor with avg loss {bc_result.metrics.get('loss', 0.0):.6f}")

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
        if pretrained_params is None:
            initial_params = model.init(param_key, dummy_obs)
        else:
            if args.bc_init_std > 0.0:
                if pretrained_flat is None:
                    raise RuntimeError("Pretrained flat params missing.")
                noise = jax.random.normal(param_key, shape=pretrained_flat.shape) * args.bc_init_std
                initial_params = unravel_fn(pretrained_flat + noise)
            else:
                initial_params = pretrained_params
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
        }

    master_rng, init_key, param_key, strategy_rng = jax.random.split(master_rng, 4)
    strategy_context = build_strategy_context(init_key, param_key)

    # 4. Setup Parallel Workers
    pool = None
    manager = None
    shared_normalizer_store = None
    shared_normalizer_lock = None
    map_fn = None
    if args.n_workers and args.n_workers > 1:
        ctx = multiprocessing.get_context("spawn")
        manager = ctx.Manager()
        shared_normalizer_store = manager.dict()
        shared_normalizer_lock = manager.Lock()
        configure_shared_running_normalizers(shared_normalizer_store, shared_normalizer_lock)
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
                shared_normalizer_store,
                shared_normalizer_lock,
            ),
        )
        map_fn = pool.map
    else:
        configure_shared_running_normalizers(None, None)

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
    if pretrained_flat is not None:
        np.savez(
            run_dir / "bc_pretrain.npz",
            flat_params=pretrained_flat,
            hidden_size=args.hidden_size,
            use_layer_norm=args.use_layer_norm,
            n_agents=n_agents,
            obs_dim=obs_dim,
            use_agent_id=use_agent_id,
            bc_epochs=args.bc_epochs,
            bc_batch_size=args.bc_batch_size,
            bc_lr=bc_lr,
        )

    rewards_file = run_dir / "training_rewards.csv"
    with rewards_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "mean_reward", "max_reward", "min_reward", "std_reward", "time_elapsed"])

    print(f"Starting training for {args.generations} generations...")
    start_time = time.time()
    best_fitness_all_time = -float("inf")

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

        strategy_rng, rng_ask = jax.random.split(strategy_rng)
        x, context_state = strategy_context["ask_step"](rng_ask, strategy_context["state"])

        population_flat = []
        for i in range(args.popsize):
            individual = jax.tree_util.tree_map(lambda arr: arr[i], x)
            flat_individual, _ = flatten_util.ravel_pytree(individual)
            population_flat.append(np.array(flat_individual))

        eval_payloads = [(flat_params, gen_episode_seeds) for flat_params in population_flat]
        fitness_values = map_fn(_evaluate_params_multi_episode, eval_payloads)
        fitness_array = jnp.array(fitness_values)

        strategy_rng, rng_tell = jax.random.split(strategy_rng)
        context_state, _ = strategy_context["tell_step"](rng_tell, x, fitness_array, context_state)
        strategy_context["state"] = context_state

        mean_fit = float(jnp.mean(fitness_array))
        max_fit = float(jnp.max(fitness_array))
        min_fit = float(jnp.min(fitness_array))
        std_fit = float(jnp.std(fitness_array))
        elapsed = time.time() - start_time

        if gen % 3 == 0 or gen == 1:
            print(
                f"Gen {gen}/{args.generations} | Mean: {mean_fit:.1f} | "
                f"Max: {max_fit:.1f} | Min: {min_fit:.1f} | Std: {std_fit:.1f} | "
                f"Time: {elapsed:.1f}s"
            )
            
        with rewards_file.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([gen, mean_fit, max_fit, min_fit, std_fit, elapsed])

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

        best_strategy_state = strategy_context["state"]
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

    if pool is not None:
        pool.close()
        pool.join()
    if manager is not None:
        manager.shutdown()
        configure_shared_running_normalizers(None, None)
    
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
        "bc_dataset": str(args.bc_dataset) if args.bc_dataset is not None else None,
        "bc_epochs": args.bc_epochs,
        "bc_batch_size": args.bc_batch_size,
        "bc_lr": bc_lr,
        "bc_init_std": args.bc_init_std,
    }
    save_config_with_hyperparameters(run_dir, args.config, "openai_es", hyperparams)
    
    print(f"\nTraining complete. Artifacts saved to {run_dir}")
    print(f"Best fitness achieved: {best_fitness_all_time:.2f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train OpenAI-ES on NCS env.")
    parser.add_argument("--config", type=Path, default=None, help="Config JSON path.")
    parser.add_argument("--generations", type=int, default=100, help="Number of generations.")
    parser.add_argument("--popsize", type=int, default=1000, help="Population size.")
    parser.add_argument("--episode-length", type=int, default=500, help="Episode length.")
    parser.add_argument("--eval-episodes", type=int, default=3, 
                        help="Episodes per individual (anti-overfitting). Default: 3")
    parser.add_argument("--fitness-shaping", type=str, default="centered_rank",
                        choices=["centered_rank", "z_score", "normalize", "none"],
                        help="Fitness shaping (evosax built-in). Default: centered_rank")
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden layer size.")
    parser.add_argument("--use-layer-norm", action="store_true", help="Use LayerNorm.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--learning-rate", type=float, default=0.005, help="Learning rate.")
    parser.add_argument("--lrate-decay", type=float, default=0.999, help="LR decay.")
    parser.add_argument("--sigma-init", type=float, default=0.25, help="Initial sigma.")
    parser.add_argument("--sigma-decay", type=float, default=0.99, help="Sigma decay.")
    parser.add_argument(
        "--bc-dataset",
        type=Path,
        default=None,
        help="Path to behavioral cloning dataset (.npz).",
    )
    parser.add_argument(
        "--bc-epochs",
        type=int,
        default=0,
        help="Number of BC epochs for actor pretraining (0 disables).",
    )
    parser.add_argument(
        "--bc-batch-size",
        type=int,
        default=1024,
        help="Mini-batch size for BC pretraining.",
    )
    parser.add_argument(
        "--bc-lr",
        type=float,
        default=None,
        help="BC learning rate (defaults to --learning-rate).",
    )
    parser.add_argument(
        "--bc-init-std",
        type=float,
        default=0.0,
        help="Std dev of Gaussian noise added to BC init params.",
    )
    parser.add_argument("--n-workers", type=int, default=multiprocessing.cpu_count(), help="Workers.")
    parser.add_argument("--checkpoint-freq", type=int, default=5, help="Checkpoint frequency.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"), help="Output directory.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
