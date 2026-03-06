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
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
import torch

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ncs_env.env import NCS_Env
from utils.marl.args_builder import _add_set_override_argument
from utils.marl.common import run_evaluation_vectorized_seeded
from utils.marl.vector_env import create_eval_async_vector_env
from utils.marl_training import load_config_with_overrides, resolve_training_eval_baseline
from utils.reward_normalization import (
    configure_shared_running_normalizers,
    reset_shared_running_normalizers,
)
from utils.bc import BCPretrainConfig, JaxActorBCAdapter, load_bc_dataset, pretrain_actor
from utils.marl.obs_normalization import RunningObsNormalizer
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
_worker_episode_length: int = 250
_worker_n_agents: int = 1
_worker_use_agent_id: bool = False
_worker_agent_id_eye: Optional[np.ndarray] = None
_worker_obs_normalizer: Optional[RunningObsNormalizer] = None
_worker_obs_dim: int = 0
_worker_predict: Any = None
_worker_unravel_fn: Any = None


def _build_obs_batch(obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
    return np.stack(
        [np.asarray(obs_dict[f"agent_{i}"], dtype=np.float32) for i in range(_worker_n_agents)],
        axis=0,
    )


def _append_agent_id(obs_batch: np.ndarray) -> np.ndarray:
    if not _worker_use_agent_id:
        return obs_batch
    if _worker_agent_id_eye is None:
        raise RuntimeError("Agent id eye matrix not initialized.")
    return np.concatenate([obs_batch, _worker_agent_id_eye], axis=1)


def _set_obs_normalizer_state(state: Optional[Tuple[np.ndarray, np.ndarray, int]]) -> None:
    if _worker_obs_normalizer is None or state is None:
        return
    mean, m2, count = state
    _worker_obs_normalizer.set_state(mean, m2, count)


def create_policy_net(
    action_dim: int,
    hidden_dims: Tuple[int, ...] = (64, 64),
    activation: str = "tanh",
):
    """Create a PolicyNet instance. Imports Flax lazily."""
    import flax.linen as nn
    import jax.numpy as jnp

    _activations = {"tanh": nn.tanh, "relu": nn.relu, "elu": nn.elu}

    class PolicyNet(nn.Module):
        action_dim: int
        hidden_dims: tuple = (64, 64)
        activation: str = "tanh"

        @nn.compact
        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            act_fn = _activations[self.activation]
            for dim in self.hidden_dims:
                x = nn.Dense(dim)(x)
                x = act_fn(x)
            x = nn.Dense(self.action_dim)(x)
            return x

    return PolicyNet(
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        activation=activation,
    )


def _init_worker(
    config_path: Optional[str],
    episode_length: int,
    seed: Optional[int],
    action_dim: int,
    obs_dim: int,
    n_agents: int,
    use_agent_id: bool,
    hidden_dims: Tuple[int, ...] = (64, 64),
    activation: str = "tanh",
    normalize_obs: bool = True,
    obs_norm_clip: float = 5.0,
    obs_norm_eps: float = 1e-8,
    params_template: Optional[Any] = None,
    reward_override: Optional[Dict[str, Any]] = None,
    network_override: Optional[Dict[str, Any]] = None,
    termination_override: Optional[Dict[str, Any]] = None,
):
    """Initialize the environment and model in the worker process."""
    global _worker_env, _worker_model, _worker_params, _worker_config_path, _worker_episode_length
    global _worker_n_agents, _worker_use_agent_id, _worker_agent_id_eye, _worker_obs_dim
    global _worker_obs_normalizer, _worker_predict, _worker_unravel_fn
    
    # Set JAX to use CPU BEFORE importing JAX
    os.environ["JAX_PLATFORMS"] = "cpu"
    warnings.filterwarnings(
        "ignore",
        message=r"Transparent hugepages are not enabled\..*",
        category=UserWarning,
        module=r"jax\._src\.cloud_tpu_init",
    )
    
    import jax
    import jax.numpy as jnp
    
    jax.config.update("jax_platform_name", "cpu")
    
    _worker_config_path = config_path
    _worker_episode_length = episode_length
    _worker_n_agents = int(n_agents)
    _worker_use_agent_id = bool(use_agent_id)
    _worker_agent_id_eye = np.eye(_worker_n_agents, dtype=np.float32) if _worker_use_agent_id else None
    _worker_obs_dim = int(obs_dim)
    if normalize_obs:
        clip_value = None if obs_norm_clip <= 0 else float(obs_norm_clip)
        _worker_obs_normalizer = RunningObsNormalizer.create(
            obs_dim, clip=clip_value, eps=float(obs_norm_eps)
        )
    else:
        _worker_obs_normalizer = None

    configure_shared_running_normalizers(None, None)
    _worker_env = NCS_Env(
        n_agents=_worker_n_agents,
        episode_length=episode_length,
        config_path=config_path,
        seed=seed,
        reward_override=reward_override,
        network_override=network_override,
        termination_override=termination_override,
    )
    _worker_model = create_policy_net(
        action_dim,
        hidden_dims=hidden_dims,
        activation=activation,
    )

    if params_template is not None:
        _worker_params = jax.tree_util.tree_map(jnp.asarray, params_template)
    else:
        rng = jax.random.PRNGKey(0)
        input_dim = obs_dim + (_worker_n_agents if _worker_use_agent_id else 0)
        dummy_obs = jnp.zeros((1, input_dim))
        _worker_params = _worker_model.init(rng, dummy_obs)

    from jax import flatten_util
    _worker_unravel_fn = flatten_util.ravel_pytree(_worker_params)[1]

    @jax.jit
    def _predict(params, obs):
        return jnp.argmax(_worker_model.apply(params, obs), axis=1)

    _worker_predict = _predict


def _run_single_episode(
    flat_params: np.ndarray,
    episode_seed: int,
) -> Tuple[float, np.ndarray, np.ndarray, int]:
    """Run one episode with the given flattened parameters and seed."""
    global _worker_env, _worker_model, _worker_params, _worker_n_agents
    
    params = _worker_unravel_fn(flat_params)

    obs_dict, _ = _worker_env.reset(seed=episode_seed)
    total_reward = 0.0
    done = False
    obs_sum = np.zeros((_worker_obs_dim,), dtype=np.float64)
    obs_sumsq = np.zeros((_worker_obs_dim,), dtype=np.float64)
    obs_count = 0

    while not done:
        obs_batch = _build_obs_batch(obs_dict)
        if _worker_obs_normalizer is not None:
            obs_sum += obs_batch.sum(axis=0)
            obs_sumsq += np.square(obs_batch).sum(axis=0)
            obs_count += int(obs_batch.shape[0])
            obs_batch = _worker_obs_normalizer.normalize(obs_batch, update=False)
        obs_batch = _append_agent_id(obs_batch)
        actions = np.asarray(_worker_predict(params, obs_batch))
        action_dict = {f"agent_{i}": int(actions[i]) for i in range(_worker_n_agents)}

        obs_dict, rewards, terminated, truncated, _ = _worker_env.step(action_dict)
        step_reward = float(np.sum([rewards[f"agent_{i}"] for i in range(_worker_n_agents)]))
        total_reward += step_reward
        done = any(terminated[f"agent_{i}"] or truncated[f"agent_{i}"] for i in range(_worker_n_agents))
        
    return total_reward, obs_sum, obs_sumsq, obs_count


def _evaluate_params_multi_episode(
    args_tuple: Tuple[np.ndarray, List[int], Optional[Tuple[np.ndarray, np.ndarray, int]]]
) -> Tuple[float, np.ndarray, np.ndarray, int]:
    """Run multiple episodes with different seeds and return mean reward plus obs stats."""
    flat_params, episode_seeds, obs_state = args_tuple

    _set_obs_normalizer_state(obs_state)
    rewards = []
    obs_sum = np.zeros((_worker_obs_dim,), dtype=np.float64)
    obs_sumsq = np.zeros((_worker_obs_dim,), dtype=np.float64)
    obs_count = 0
    for seed in episode_seeds:
        reward, ep_sum, ep_sumsq, ep_count = _run_single_episode(flat_params, seed)
        rewards.append(reward)
        if ep_count > 0:
            obs_sum += ep_sum
            obs_sumsq += ep_sumsq
            obs_count += ep_count
    
    return float(np.mean(rewards)), obs_sum, obs_sumsq, obs_count


def _make_vectorized_action_selector(
    *,
    params: Any,
    predict_fn: Any,
    n_agents: int,
    use_agent_id: bool,
) -> Any:
    agent_id_eye = np.eye(n_agents, dtype=np.float32) if use_agent_id else None

    def _select_actions(obs: np.ndarray) -> np.ndarray:
        obs_batch = np.asarray(obs, dtype=np.float32)
        if obs_batch.ndim != 3 or obs_batch.shape[1] != n_agents:
            raise ValueError("obs must have shape (n_envs, n_agents, obs_dim)")
        flat_obs = obs_batch.reshape(obs_batch.shape[0] * n_agents, obs_batch.shape[-1])
        if agent_id_eye is not None:
            tiled_agent_ids = np.broadcast_to(
                agent_id_eye, (obs_batch.shape[0], n_agents, n_agents)
            ).reshape(obs_batch.shape[0] * n_agents, n_agents)
            flat_obs = np.concatenate([flat_obs, tiled_agent_ids], axis=1)
        flat_actions = np.asarray(predict_fn(params, flat_obs), dtype=np.int64)
        return flat_actions.reshape(obs_batch.shape[0], n_agents)

    return _select_actions


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


def _load_pt_checkpoint(
    pt_path: Path,
    obs_dim: int,
    n_agents: int,
    use_agent_id: bool,
) -> Dict[str, Any]:
    """Load a PyTorch MARL checkpoint and return architecture info + state dict."""
    import torch

    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    algorithm = ckpt.get("algorithm", "unknown")
    print(f"[PT] Loading {algorithm} checkpoint from {pt_path}")

    if ckpt.get("dueling", False):
        raise ValueError("Dueling architecture checkpoints cannot be converted to ES PolicyNet")
    if "agent_state_dict" not in ckpt:
        if "agent_state_dicts" in ckpt:
            raise ValueError("Per-agent parameters not supported, only parameter sharing")
        raise ValueError("Checkpoint missing 'agent_state_dict'")
    if ckpt.get("n_agents", 1) != n_agents:
        raise ValueError(f"Checkpoint n_agents={ckpt['n_agents']} != env n_agents={n_agents}")
    if ckpt.get("obs_dim", 0) != obs_dim:
        raise ValueError(f"Checkpoint obs_dim={ckpt['obs_dim']} != env obs_dim={obs_dim}")
    if ckpt.get("use_agent_id", False) != use_agent_id:
        raise ValueError(f"Checkpoint use_agent_id={ckpt['use_agent_id']} != {use_agent_id}")
    if ckpt.get("feature_norm", False) or ckpt.get("layer_norm", False):
        raise ValueError("LayerNorm checkpoints are no longer supported by openai_es.")

    return {
        "hidden_dims": tuple(ckpt["agent_hidden_dims"]),
        "activation": ckpt["agent_activation"],
        "state_dict": ckpt["agent_state_dict"],
        "obs_norm_state": ckpt.get("obs_normalization", {}),
    }


def _pt_state_dict_to_flax(
    state_dict: Dict[str, Any],
    params_template: Any,
    hidden_dims: Tuple[int, ...],
) -> Any:
    """Convert a PyTorch MLPAgent state_dict to Flax PolicyNet params."""
    import jax
    import jax.numpy as jnp

    n_hidden = len(hidden_dims)
    stride = 2
    params_dict: Dict[str, Dict[str, Any]] = {}

    for i in range(n_hidden):
        pt_idx = i * stride
        w = state_dict[f"net.{pt_idx}.weight"].numpy()
        b = state_dict[f"net.{pt_idx}.bias"].numpy()
        params_dict[f"Dense_{i}"] = {
            "kernel": jnp.array(w.T, dtype=jnp.float32),
            "bias": jnp.array(b, dtype=jnp.float32),
        }

    pt_out_idx = n_hidden * stride
    w = state_dict[f"net.{pt_out_idx}.weight"].numpy()
    b = state_dict[f"net.{pt_out_idx}.bias"].numpy()
    params_dict[f"Dense_{n_hidden}"] = {
        "kernel": jnp.array(w.T, dtype=jnp.float32),
        "bias": jnp.array(b, dtype=jnp.float32),
    }

    flax_params = {"params": params_dict}
    for t, c in zip(jax.tree_util.tree_leaves(params_template), jax.tree_util.tree_leaves(flax_params)):
        if t.shape != c.shape:
            raise ValueError(f"Shape mismatch: template {t.shape} != converted {c.shape}")
    return flax_params


def train(args):
    """Main training loop."""
    reset_shared_running_normalizers()
    warnings.filterwarnings(
        "ignore",
        message=r"Transparent hugepages are not enabled\..*",
        category=UserWarning,
        module=r"jax\._src\.cloud_tpu_init",
    )
    import jax
    import jax.numpy as jnp
    from jax import flatten_util
    import optax
    from evosax.algorithms.distribution_based import Open_ES
    
    if args.init_checkpoint is not None and args.bc_dataset is not None:
        raise ValueError("--init-checkpoint and --bc-dataset are mutually exclusive")
    if args.bc_epochs > 0 and args.bc_dataset is None:
        raise ValueError("--bc-epochs requires --bc-dataset")
    if args.bc_dataset is not None and args.bc_epochs <= 0:
        raise ValueError("--bc-dataset requires --bc-epochs > 0")
    if args.bc_batch_size <= 0:
        raise ValueError("--bc-batch-size must be positive")
    if args.bc_init_std < 0.0:
        raise ValueError("--bc-init-std must be >= 0")
    if args.eval_freq_generations <= 0:
        raise ValueError("--eval-freq-generations must be positive")
    if args.n_eval_episodes <= 0:
        raise ValueError("--n-eval-episodes must be positive")
    if args.n_eval_envs <= 0:
        raise ValueError("--n-eval-envs must be positive")
    
    cfg, config_path_str, n_agents, use_agent_id, eval_reward_override, eval_termination_override, network_override, training_reward_override = (
        load_config_with_overrides(args.config, args.n_agents, not args.no_agent_id, args.set_overrides)
    )
    eval_baseline = resolve_training_eval_baseline(cfg, n_agents)

    # 1. Setup Dummy Environment to get shapes
    dummy_env = NCS_Env(
        n_agents=n_agents, 
        config_path=config_path_str,
        reward_override=training_reward_override,
        network_override=network_override,
    )
    obs_dim = dummy_env.observation_space["agent_0"].shape[0]
    action_dim = dummy_env.action_space["agent_0"].n
    dummy_env.close()
    input_dim = obs_dim + (n_agents if use_agent_id else 0)
    # obs_normalizer is initialized later - either from BC dataset or fresh
    obs_normalizer: Optional[RunningObsNormalizer] = None

    # Determine architecture (may be overridden by compatible checkpoints)
    hidden_dims = tuple(args.hidden_dims)
    activation = args.activation
    _pt_state_dict = None
    _pt_obs_norm_state: Optional[Dict[str, Any]] = None

    if args.init_checkpoint is not None and args.init_checkpoint.suffix == ".pt":
        pt_info = _load_pt_checkpoint(args.init_checkpoint, obs_dim, n_agents, use_agent_id)
        hidden_dims = pt_info["hidden_dims"]
        activation = pt_info["activation"]
        _pt_state_dict = pt_info["state_dict"]
        _pt_obs_norm_state = pt_info["obs_norm_state"]
    elif args.init_checkpoint is not None and args.init_checkpoint.suffix == ".npz":
        with np.load(args.init_checkpoint, allow_pickle=False) as ckpt:
            if bool(ckpt["feature_norm"]) if "feature_norm" in ckpt else False:
                raise ValueError("LayerNorm ES checkpoints are no longer supported.")

    print(f"Observation Dim: {obs_dim}, Input Dim: {input_dim}, Action Dim: {action_dim}")
    print(f"N Agents: {n_agents}, Use Agent Id: {use_agent_id}")
    print(f"Hidden Dims: {hidden_dims}, Activation: {activation}")
    print(
        "Obs normalization: "
        f"{'enabled' if args.normalize_obs else 'disabled'} "
        f"(clip={args.obs_norm_clip}, eps={args.obs_norm_eps})"
    )
    print(f"Evaluation episodes per individual: {args.eval_episodes}")
    print(
        f"Model-selection eval: every {args.eval_freq_generations} generation(s), "
        f"{args.n_eval_episodes} episodes over {args.n_eval_envs} envs"
    )
    print(f"Fitness shaping: {args.fitness_shaping} (evosax built-in)")
    print(f"L2 weight decay: {args.weight_decay}")
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
    model_selection_eval_episodes = int(args.n_eval_episodes)

    # 2. Setup Flax Model & Initial Params
    master_rng = jax.random.PRNGKey(args.seed if args.seed is not None else 0)
    master_rng, eval_rng = jax.random.split(master_rng)
    model = create_policy_net(
        action_dim,
        hidden_dims=hidden_dims,
        activation=activation,
    )
    dummy_obs = jnp.zeros((1, input_dim))
    master_rng, template_rng = jax.random.split(master_rng)
    params_template = model.init(template_rng, dummy_obs)

    flat_params_template, unravel_fn = flatten_util.ravel_pytree(params_template)
    num_params = flat_params_template.size
    print(f"Total Parameters: {num_params}")
    params_template_for_workers = jax.tree_util.tree_map(np.asarray, params_template)

    @jax.jit
    def eval_predict(params, obs):
        return jnp.argmax(model.apply(params, obs), axis=1)

    bc_lr: Optional[float] = None
    pretrained_params: Optional[Any] = None
    pretrained_flat: Optional[np.ndarray] = None

    # Load .pt checkpoint weights (architecture was already overridden above)
    if _pt_state_dict is not None:
        pretrained_params = _pt_state_dict_to_flax(
            _pt_state_dict, params_template, hidden_dims,
        )
        pretrained_flat = np.array(flatten_util.ravel_pytree(pretrained_params)[0])
        if args.normalize_obs and _pt_obs_norm_state and _pt_obs_norm_state.get("enabled", False):
            clip_value = None if args.obs_norm_clip <= 0 else float(args.obs_norm_clip)
            obs_normalizer = RunningObsNormalizer.from_state_dict(_pt_obs_norm_state)
            obs_normalizer.clip = clip_value
            obs_normalizer.eps = float(args.obs_norm_eps)
        print(f"[PT] Initialized {num_params} params from {args.init_checkpoint}")

    bc_dataset = None
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

        # Initialize obs normalizer from BC dataset if normalization is enabled
        if args.normalize_obs:
            clip_value = None if args.obs_norm_clip <= 0 else float(args.obs_norm_clip)
            obs_normalizer = bc_dataset.compute_obs_normalizer(
                clip=clip_value, eps=float(args.obs_norm_eps)
            )
            print(
                f"[BC] Initialized obs normalizer from dataset "
                f"(count={obs_normalizer.count}, mean_norm={np.linalg.norm(obs_normalizer.mean):.4f})"
            )

        bc_lr = float(args.learning_rate) if args.bc_lr is None else float(args.bc_lr)
        bc_config = BCPretrainConfig(
            epochs=int(args.bc_epochs),
            batch_size=int(args.bc_batch_size),
            learning_rate=bc_lr,
            use_agent_id=use_agent_id,
            n_agents=n_agents,
            obs_normalizer=obs_normalizer,
        )
        bc_rng = np.random.default_rng(args.seed)
        bc_adapter = JaxActorBCAdapter(
            model.apply,
            n_agents=n_agents,
            use_agent_id=use_agent_id,
        )
        bc_result = pretrain_actor(bc_adapter, params_template, bc_dataset, bc_config, bc_rng)
        pretrained_params = bc_result.params
        pretrained_flat = np.array(flatten_util.ravel_pytree(pretrained_params)[0])
        print(f"[BC] Pretrained actor with avg loss {bc_result.metrics.get('loss', 0.0):.6f}")

    # Load from .npz checkpoint if provided
    if args.init_checkpoint is not None and args.init_checkpoint.suffix == ".npz":
        ckpt = np.load(args.init_checkpoint, allow_pickle=False)
        pretrained_flat = np.asarray(ckpt["flat_params"])
        if pretrained_flat.shape[0] != num_params:
            raise ValueError(
                f"Checkpoint has {pretrained_flat.shape[0]} params, model expects {num_params}"
            )
        pretrained_params = unravel_fn(pretrained_flat)
        if args.normalize_obs and "obs_norm_mean" in ckpt:
            clip_value = None if args.obs_norm_clip <= 0 else float(args.obs_norm_clip)
            obs_normalizer = RunningObsNormalizer(
                mean=np.asarray(ckpt["obs_norm_mean"]),
                m2=np.asarray(ckpt["obs_norm_m2"]),
                count=int(ckpt["obs_norm_count"]),
                clip=clip_value,
                eps=float(args.obs_norm_eps),
            )
        print(f"[Checkpoint] Loaded from {args.init_checkpoint} ({num_params} params)")

    # Create fresh obs normalizer if not initialized from BC dataset
    if obs_normalizer is None and args.normalize_obs:
        clip_value = None if args.obs_norm_clip <= 0 else float(args.obs_norm_clip)
        obs_normalizer = RunningObsNormalizer.create(
            obs_dim, clip=clip_value, eps=float(args.obs_norm_eps)
        )

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
            initial_params = params_template
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
                hidden_dims,
                activation,
                args.normalize_obs,
                args.obs_norm_clip,
                args.obs_norm_eps,
                params_template_for_workers,
                training_reward_override,
                network_override,
                eval_termination_override,
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
            hidden_dims,
            activation,
            args.normalize_obs,
            args.obs_norm_clip,
            args.obs_norm_eps,
            params_template_for_workers,
            training_reward_override,
            network_override,
            eval_termination_override,
        )

    # 5. Logging Setup
    run_dir = prepare_run_directory("openai_es", args.config, args.output_root)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    # JAX/TPU is already initialized in the main process; eval env workers must use spawn.
    eval_env = create_eval_async_vector_env(
        n_eval_envs=args.n_eval_envs,
        n_agents=n_agents,
        episode_length=args.episode_length,
        config_path_str=config_path_str,
        seed=args.seed,
        reward_override=eval_reward_override,
        termination_override=eval_termination_override,
        context="spawn",
    )
    eval_device = torch.device("cpu")

    def build_checkpoint_payload(flat_params: np.ndarray) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "flat_params": flat_params,
            "hidden_dims": list(hidden_dims),
            "activation": activation,
            "hidden_size": int(hidden_dims[0]),
            "normalize_obs": args.normalize_obs,
            "obs_norm_clip": args.obs_norm_clip,
            "obs_norm_eps": args.obs_norm_eps,
            "n_agents": n_agents,
            "obs_dim": obs_dim,
            "use_agent_id": use_agent_id,
        }
        if obs_normalizer is not None:
            payload.update(
                {
                    "obs_norm_mean": obs_normalizer.mean.astype(np.float64),
                    "obs_norm_m2": obs_normalizer.m2.astype(np.float64),
                    "obs_norm_count": int(obs_normalizer.count),
                }
            )
        return payload

    if pretrained_flat is not None:
        bc_payload = build_checkpoint_payload(pretrained_flat)
        bc_payload.update(
            {
                "bc_epochs": args.bc_epochs,
                "bc_batch_size": args.bc_batch_size,
                "bc_lr": bc_lr,
            }
        )
        np.savez(run_dir / "bc_pretrain.npz", **bc_payload)

    rewards_file = run_dir / "training_rewards.csv"
    with rewards_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "mean_reward", "max_reward", "min_reward", "std_reward", "time_elapsed"])

    eval_rewards_file = run_dir / "evaluation_rewards.csv"
    with eval_rewards_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "mean_reward", "std_reward"])

    eval_drop_stats_file = run_dir / "evaluation_drop_stats.csv"
    with eval_drop_stats_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "baseline_policy",
                "baseline_perfect_communication",
                "policy_mean_reward",
                "policy_std_reward",
                "baseline_mean_reward",
                "baseline_std_reward",
                "drop_ratio_mean",
                "drop_ratio_std",
                "num_episodes",
            ]
        )

    try:
        print(f"Starting training for {args.generations} generations...")
        start_time = time.time()
        best_drop_ratio_all_time = float("inf")
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

            # Vectorized flatten: reshape each leaf to (popsize, -1) and concatenate
            leaves = jax.tree_util.tree_leaves(x)
            flat_population = np.asarray(
                jnp.concatenate([leaf.reshape(args.popsize, -1) for leaf in leaves], axis=1)
            )  # (popsize, num_params) — single device-to-host transfer
            population_flat = [flat_population[i] for i in range(args.popsize)]

            obs_state = None
            if obs_normalizer is not None:
                obs_state = (
                    obs_normalizer.mean.copy(),
                    obs_normalizer.m2.copy(),
                    int(obs_normalizer.count),
                )

            eval_payloads = [(flat_params, gen_episode_seeds, obs_state) for flat_params in population_flat]
            eval_results = map_fn(_evaluate_params_multi_episode, eval_payloads)
            fitness_values = [result[0] for result in eval_results]

            # Apply L2 regularization to fitness (penalize large parameter norms)
            if args.weight_decay > 0:
                l2_penalties = [
                    args.weight_decay * float(np.sum(np.square(flat_params)))
                    for flat_params in population_flat
                ]
                fitness_values = [f - p for f, p in zip(fitness_values, l2_penalties)]

            fitness_array = jnp.array(fitness_values)
            best_idx_generation = int(jnp.argmax(fitness_array))
            candidate_flat_params = population_flat[best_idx_generation]

            if obs_normalizer is not None:
                obs_sum_total = np.zeros((obs_dim,), dtype=np.float64)
                obs_sumsq_total = np.zeros((obs_dim,), dtype=np.float64)
                obs_count_total = 0
                for _mean_reward, obs_sum, obs_sumsq, obs_count in eval_results:
                    if obs_count > 0:
                        obs_sum_total += obs_sum
                        obs_sumsq_total += obs_sumsq
                        obs_count_total += int(obs_count)
                if obs_count_total > 0:
                    batch_mean = obs_sum_total / float(obs_count_total)
                    batch_var = obs_sumsq_total / float(obs_count_total) - np.square(batch_mean)
                    batch_var = np.maximum(batch_var, 0.0)
                    obs_normalizer.update_from_moments(batch_mean, batch_var, obs_count_total)

            strategy_rng, rng_tell = jax.random.split(strategy_rng)
            context_state, _ = strategy_context["tell_step"](rng_tell, x, -fitness_array, context_state)
            strategy_context["state"] = context_state
            best_strategy_state = strategy_context["state"]
            mean_flat_params = np.array(best_strategy_state.mean)

            mean_fit = float(jnp.mean(fitness_array))
            max_fit = float(jnp.max(fitness_array))
            min_fit = float(jnp.min(fitness_array))
            std_fit = float(jnp.std(fitness_array))
            elapsed = time.time() - start_time

            run_model_selection_eval = (
                gen % args.eval_freq_generations == 0 or gen == args.generations
            )
            if run_model_selection_eval:
                eval_rng, rng_model_selection_eval = jax.random.split(eval_rng)
                model_selection_eval_seeds = [
                    int(jax.random.randint(
                        jax.random.fold_in(rng_model_selection_eval, i),
                        (), minval=0, maxval=2**31 - 1, dtype=jnp.int32
                    ))
                    for i in range(model_selection_eval_episodes)
                ]
                policy_params = unravel_fn(mean_flat_params)
                policy_action_selector = _make_vectorized_action_selector(
                    params=policy_params,
                    predict_fn=eval_predict,
                    n_agents=n_agents,
                    use_agent_id=use_agent_id,
                )

                fixed_eval_mean, fixed_eval_std, fixed_eval_rewards = run_evaluation_vectorized_seeded(
                    eval_env=eval_env,
                    agent=None,
                    n_eval_envs=args.n_eval_envs,
                    n_agents=n_agents,
                    n_actions=action_dim,
                    use_agent_id=use_agent_id,
                    device=eval_device,
                    episode_seeds=model_selection_eval_seeds,
                    obs_normalizer=obs_normalizer,
                    action_selector=policy_action_selector,
                )

                baseline_label = str(eval_baseline.get("label", "perfect_comm"))
                baseline_policy = str(eval_baseline.get("heuristic_policy", "always_send"))
                baseline_deterministic = bool(eval_baseline.get("deterministic", True))
                baseline_perfect_comm = bool(eval_baseline.get("use_perfect_communication", False))
                heuristic_name = None if baseline_policy == "always_send" else baseline_policy
                baseline_fixed_action = 1 if baseline_policy == "always_send" else None

                current_pc_states = eval_env.call("get_perfect_communication")
                current_pc = bool(current_pc_states[0]) if current_pc_states else False
                if baseline_perfect_comm != current_pc:
                    eval_env.call("set_perfect_communication", baseline_perfect_comm)
                try:
                    baseline_mean, baseline_std, baseline_rewards = run_evaluation_vectorized_seeded(
                        eval_env=eval_env,
                        agent=None,
                        n_eval_envs=args.n_eval_envs,
                        n_agents=n_agents,
                        n_actions=action_dim,
                        use_agent_id=use_agent_id,
                        device=eval_device,
                        episode_seeds=model_selection_eval_seeds,
                        obs_normalizer=obs_normalizer,
                        heuristic_policy_name=heuristic_name,
                        heuristic_deterministic=baseline_deterministic,
                        fixed_action=baseline_fixed_action,
                    )
                finally:
                    if baseline_perfect_comm != current_pc:
                        eval_env.call("set_perfect_communication", current_pc)

                policy_arr = np.asarray(fixed_eval_rewards, dtype=np.float64)
                baseline_arr = np.asarray(baseline_rewards, dtype=np.float64)
                denom = np.maximum(np.abs(baseline_arr), 1e-8)
                drop_ratios = (baseline_arr - policy_arr) / denom
                mean_drop_ratio = float(np.mean(drop_ratios))
                std_drop_ratio = float(np.std(drop_ratios))

                print(
                    f"Gen {gen}/{args.generations} | Mean: {mean_fit:.1f} | "
                    f"Max: {max_fit:.1f} | Min: {min_fit:.1f} | Std: {std_fit:.1f} | "
                    f"Eval({model_selection_eval_episodes} eps): {fixed_eval_mean:.1f} ± {fixed_eval_std:.1f} | "
                    f"Baseline({baseline_label}): {baseline_mean:.1f} ± {baseline_std:.1f} | "
                    f"Drop: {mean_drop_ratio:.6f} ± {std_drop_ratio:.6f} | "
                    f"Time: {elapsed:.1f}s"
                )
            else:
                print(
                    f"Gen {gen}/{args.generations} | Mean: {mean_fit:.1f} | "
                    f"Max: {max_fit:.1f} | Min: {min_fit:.1f} | Std: {std_fit:.1f} | "
                    f"Time: {elapsed:.1f}s"
                )

            with rewards_file.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([gen, mean_fit, max_fit, min_fit, std_fit, elapsed])

            if run_model_selection_eval:
                with eval_rewards_file.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([gen, fixed_eval_mean, fixed_eval_std])

                with eval_drop_stats_file.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            gen,
                            baseline_label,
                            int(baseline_perfect_comm),
                            fixed_eval_mean,
                            fixed_eval_std,
                            baseline_mean,
                            baseline_std,
                            mean_drop_ratio,
                            std_drop_ratio,
                            len(drop_ratios),
                        ]
                    )

                if mean_drop_ratio < best_drop_ratio_all_time:
                    best_drop_ratio_all_time = mean_drop_ratio
                    np.savez(
                        run_dir / "best_model.npz",
                        **build_checkpoint_payload(mean_flat_params),
                    )

            if max_fit > best_fitness_all_time:
                best_fitness_all_time = max_fit
                np.savez(
                    run_dir / "best_fitness_model.npz",
                    **build_checkpoint_payload(candidate_flat_params),
                )

            np.savez(
                run_dir / "latest_model.npz",
                **build_checkpoint_payload(mean_flat_params),
            )

            if gen % args.checkpoint_freq == 0:
                np.savez(
                    run_dir / "checkpoints" / f"gen_{gen}.npz",
                    **build_checkpoint_payload(np.array(best_strategy_state.mean)),
                )
    finally:
        if pool is not None:
            if sys.exc_info()[0] is None:
                pool.close()
            else:
                pool.terminate()
            pool.join()
        eval_env.close()
        configure_shared_running_normalizers(None, None)
    
    # Save hyperparameters
    hyperparams = {
        "generations": args.generations,
        "popsize": args.popsize,
        "episode_length": args.episode_length,
        "eval_episodes": args.eval_episodes,
        "eval_freq_generations": args.eval_freq_generations,
        "model_selection_eval_episodes": model_selection_eval_episodes,
        "model_selection_eval_seed_mode": "advancing_paired",
        "n_eval_envs": args.n_eval_envs,
        "fitness_shaping": args.fitness_shaping,
        "hidden_dims": list(hidden_dims),
        "activation": activation,
        "normalize_obs": args.normalize_obs,
        "obs_norm_clip": args.obs_norm_clip,
        "obs_norm_eps": args.obs_norm_eps,
        "learning_rate": args.learning_rate,
        "lrate_decay": args.lrate_decay,
        "sigma_init": args.sigma_init,
        "sigma_decay": args.sigma_decay,
        "weight_decay": args.weight_decay,
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
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint is not None else None,
    }
    save_config_with_hyperparameters(
        run_dir, 
        args.config, 
        "openai_es", 
        hyperparams,
        resolved_config=cfg,
        set_overrides=args.set_overrides,
    )
    
    print(f"\nTraining complete. Artifacts saved to {run_dir}")
    print(f"Best fixed-seed drop ratio achieved: {best_drop_ratio_all_time:.6f}")
    print(f"Best fitness achieved: {best_fitness_all_time:.2f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train OpenAI-ES on NCS env.")
    parser.add_argument("--config", type=Path, default=None, help="Config JSON path.")
    parser.add_argument("--n-agents", type=int, default=1, help="Number of agents (if not in config).")
    parser.add_argument(
        "--no-agent-id",
        action="store_true",
        help="Disable appending one-hot agent ID to observations.",
    )
    parser.add_argument("--generations", type=int, default=100, help="Number of generations.")
    parser.add_argument("--popsize", type=int, default=1000, help="Population size.")
    parser.add_argument("--episode-length", type=int, default=250, help="Episode length.")
    parser.add_argument("--eval-episodes", type=int, default=3, 
                        help="Episodes per individual (anti-overfitting). Default: 3")
    parser.add_argument(
        "--eval-freq-generations",
        type=int,
        default=1,
        help="Run model-selection evaluation every N generations (final generation always evaluates).",
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=80,
        help="Number of paired fixed-seed episodes for model-selection evaluation.",
    )
    parser.add_argument(
        "--n-eval-envs",
        type=int,
        default=8,
        help="Number of parallel async environments used for model-selection evaluation.",
    )
    parser.add_argument("--fitness-shaping", type=str, default="centered_rank",
                        choices=["centered_rank", "z_score", "normalize", "none"],
                        help="Fitness shaping (evosax built-in). Default: centered_rank")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 64],
                        help="Hidden layer dimensions (e.g., --hidden-dims 128 128).")
    parser.add_argument("--activation", type=str, default="tanh",
                        choices=["tanh", "relu", "elu"], help="Activation function.")
    obs_norm_group = parser.add_mutually_exclusive_group()
    obs_norm_group.add_argument(
        "--normalize-obs",
        action="store_true",
        help="Normalize observations with running mean/std (default).",
    )
    obs_norm_group.add_argument(
        "--no-normalize-obs",
        action="store_false",
        dest="normalize_obs",
        help="Disable observation normalization.",
    )
    parser.add_argument(
        "--obs-norm-clip",
        type=float,
        default=5.0,
        help="Clip normalized observations to +/- this value (<=0 disables).",
    )
    parser.add_argument(
        "--obs-norm-eps",
        type=float,
        default=1e-8,
        help="Epsilon for observation normalization.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--learning-rate", type=float, default=0.005, help="Learning rate.")
    parser.add_argument("--lrate-decay", type=float, default=0.999, help="LR decay.")
    parser.add_argument("--sigma-init", type=float, default=0.02, help="Initial sigma.")
    parser.add_argument("--sigma-decay", type=float, default=1, help="Sigma decay.")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.005,
        help="L2 regularization coefficient applied to fitness (default: 0.005).",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Path to .npz checkpoint to initialize from (must contain flat_params).",
    )
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
    _add_set_override_argument(parser)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"), help="Output directory.")
    parser.set_defaults(normalize_obs=True)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
