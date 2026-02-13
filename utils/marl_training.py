"""
Shared utilities for MARL training scripts.

This module contains common setup and utility functions used across
all MARL training algorithms (IQL, VDN, QMIX, QPLEX, MAPPO).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING

import multiprocessing
from multiprocessing.managers import SyncManager

import numpy as np
import torch

if TYPE_CHECKING:
    from ncs_env.env import NCS_Env
    from utils.marl import RunningObsNormalizer
    from utils.run_utils import BestModelTracker


def setup_device_and_rng(
    device_str: str,
    seed: Optional[int],
) -> Tuple[torch.device, np.random.Generator]:
    """
    Initialize torch device and numpy random generator.

    Args:
        device_str: Device selection ("auto", "cpu", or "cuda")
        seed: Random seed (None for no seeding)

    Returns:
        Tuple of (torch device, numpy random generator)
    """
    from utils.marl.common import select_device

    device = select_device(device_str)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed if seed is not None else 0)
    return device, rng


def load_config_with_overrides(
    config_path: Optional[Path],
    default_n_agents: int,
    use_agent_id_flag: bool,
) -> Tuple[Dict[str, Any], str, int, bool, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Load config and extract evaluation overrides.

    Args:
        config_path: Path to config JSON (or None for default)
        default_n_agents: Default number of agents if not in config
        use_agent_id_flag: Value of --no-agent-id flag (True if NOT set)

    Returns:
        Tuple of:
            - Full config dict
            - Config path as string (or None)
            - Number of agents
            - Whether to use agent ID
            - Evaluation reward override dict (or None)
            - Evaluation termination override dict (or None)
    """
    from ncs_env.config import load_config

    config_path_str = str(config_path) if config_path is not None else None
    cfg = load_config(config_path_str)
    system_cfg = cfg.get("system", {})
    n_agents = int(system_cfg.get("n_agents", default_n_agents))
    use_agent_id = use_agent_id_flag

    # Load evaluation reward config if present
    eval_reward_override: Optional[Dict[str, Any]] = None
    reward_cfg = cfg.get("reward", {})
    eval_reward_cfg = reward_cfg.get("evaluation", None)
    if isinstance(eval_reward_cfg, dict):
        eval_reward_override = eval_reward_cfg

    # Load evaluation termination config if present
    eval_termination_override: Optional[Dict[str, Any]] = None
    termination_cfg = cfg.get("termination", {})
    eval_termination_cfg = termination_cfg.get("evaluation", None)
    if isinstance(eval_termination_cfg, dict):
        eval_termination_override = eval_termination_cfg

    return (
        cfg,
        config_path_str,
        n_agents,
        use_agent_id,
        eval_reward_override,
        eval_termination_override,
    )


def create_environments(
    n_agents: int,
    episode_length: int,
    config_path_str: Optional[str],
    seed: Optional[int],
    eval_reward_override: Optional[Dict[str, Any]],
    eval_termination_override: Optional[Dict[str, Any]],
) -> Tuple["NCS_Env", "NCS_Env"]:
    """
    Create training and evaluation environments.

    Args:
        n_agents: Number of agents
        episode_length: Maximum episode length
        config_path_str: Path to config JSON as string (or None)
        seed: Random seed
        eval_reward_override: Evaluation reward config override
        eval_termination_override: Evaluation termination config override

    Returns:
        Tuple of (training environment, evaluation environment)
    """
    from ncs_env.env import NCS_Env

    env = NCS_Env(
        n_agents=n_agents,
        episode_length=episode_length,
        config_path=config_path_str,
        seed=seed,
    )

    eval_env = NCS_Env(
        n_agents=n_agents,
        episode_length=episode_length,
        config_path=config_path_str,
        seed=seed,
        reward_override=eval_reward_override,
        termination_override=eval_termination_override,
        freeze_running_normalization=True,
    )

    return env, eval_env


def create_obs_normalizer(
    obs_dim: int,
    normalize_obs: bool,
    obs_norm_clip: float,
    obs_norm_eps: float,
) -> Optional["RunningObsNormalizer"]:
    """
    Create observation normalizer if enabled.

    Args:
        obs_dim: Observation dimension
        normalize_obs: Whether to enable normalization
        obs_norm_clip: Clip value for normalized observations (<=0 disables)
        obs_norm_eps: Epsilon for normalization

    Returns:
        RunningObsNormalizer instance or None if disabled
    """
    from utils.marl import RunningObsNormalizer

    if not normalize_obs:
        return None

    clip_value = None if obs_norm_clip <= 0 else float(obs_norm_clip)
    return RunningObsNormalizer.create(
        obs_dim, clip=clip_value, eps=float(obs_norm_eps)
    )


def print_run_summary(
    run_dir: Path,
    latest_path: Path,
    rewards_csv_path: Path,
    eval_csv_path: Path,
) -> None:
    """
    Print summary of run artifacts.

    Args:
        run_dir: Run output directory
        latest_path: Path to latest model checkpoint
        rewards_csv_path: Path to training rewards CSV
        eval_csv_path: Path to evaluation rewards CSV
    """
    print(f"Run artifacts stored in {run_dir}")
    print(f"  - Latest model: {latest_path}")
    print(f"  - Best eval model: {run_dir / 'best_model.pt'}")
    print(f"  - Best train model: {run_dir / 'best_train_model.pt'}")
    print(f"  - Training rewards: {rewards_csv_path}")
    print(f"  - Evaluation rewards: {eval_csv_path}")
    print(f"  - Config with hyperparameters: {run_dir / 'config.json'}")


def setup_shared_reward_normalizer(
    reward_cfg: Dict[str, Any],
    run_dir: Path,
    *,
    sync_interval: int = 32,
) -> Tuple[Optional["SharedRewardNormalizerConfig"], Optional[SyncManager]]:
    """
    Configure shared running reward normalization for async vector env workers.

    Returns:
        Tuple of (shared normalizer config, manager). Manager is kept alive by caller.
    """
    if not bool(reward_cfg.get("normalize", True)):
        return None, None

    from utils.marl.vector_env import SharedRewardNormalizerConfig
    from utils.reward_normalization import configure_shared_running_normalizers

    manager = multiprocessing.Manager()
    store = manager.dict()
    lock = manager.Lock()
    namespace = f"reward_norm:{run_dir.name}"

    configure_shared_running_normalizers(
        store,
        lock,
        sync_interval=sync_interval,
        namespace=namespace,
        reset_store=True,
    )

    config = SharedRewardNormalizerConfig(
        store=store,
        lock=lock,
        namespace=namespace,
        sync_interval=sync_interval,
        reset_store=False,
    )
    return config, manager


def evaluate_and_log(
    *,
    eval_env: Any,
    agent: Any,
    n_eval_envs: int,
    n_agents: int,
    n_actions: int,
    use_agent_id: bool,
    device: torch.device,
    n_episodes: int,
    seed: Optional[int],
    obs_normalizer: Optional["RunningObsNormalizer"],
    eval_writer: csv.writer,
    eval_f: Any,
    best_model_tracker: "BestModelTracker",
    run_dir: Path,
    save_checkpoint: Callable[[Path], None],
    global_step: int,
    algo_name: str,
) -> None:
    """Run vectorized evaluation, write CSV row, update best model, and print."""
    from utils.marl.common import run_evaluation_vectorized

    mean_eval_reward, std_eval_reward, _ = run_evaluation_vectorized(
        eval_env=eval_env,
        agent=agent,
        n_eval_envs=n_eval_envs,
        n_agents=n_agents,
        n_actions=n_actions,
        use_agent_id=use_agent_id,
        device=device,
        n_episodes=n_episodes,
        seed=seed,
        obs_normalizer=obs_normalizer,
    )
    eval_writer.writerow([global_step, mean_eval_reward, std_eval_reward])
    eval_f.flush()

    best_model_tracker.update(
        "eval", mean_eval_reward, run_dir / "best_model.pt", save_checkpoint
    )

    print(
        f"[{algo_name}] Eval at step {global_step}: "
        f"mean_reward={mean_eval_reward:.3f} std={std_eval_reward:.3f}"
    )


def log_completed_episodes(
    *,
    done_reset: np.ndarray,
    episode_reward_sums: np.ndarray,
    global_step: int,
    episode: int,
    train_writer: csv.writer,
    train_f: Any,
    best_model_tracker: "BestModelTracker",
    run_dir: Path,
    save_checkpoint: Callable[[Path], None],
    log_interval: int,
    algo_name: str,
    extra_csv_values: Any = None,
    extra_log_str: str = "",
    episode_lengths: Optional[np.ndarray] = None,
) -> int:
    """Log completed episodes to CSV, update best train model, and print.

    Args:
        extra_csv_values: For Q-learning, a scalar (epsilon) appended to each row.
            For MAPPO, pass None and use *episode_lengths* instead.
        extra_log_str: Extra string appended to the print line (e.g. " eps=0.100").
        episode_lengths: If provided, the per-env episode length array.
            Values for done envs are written to CSV and then reset to 0.

    Returns:
        Updated episode counter.
    """
    if not np.any(done_reset):
        return episode

    done_indices = np.where(done_reset)[0]
    for env_idx in done_indices:
        row = [episode, float(episode_reward_sums[env_idx])]
        if episode_lengths is not None:
            row.append(int(episode_lengths[env_idx]))
        if extra_csv_values is not None:
            row.append(float(extra_csv_values))
        row.append(global_step)
        train_writer.writerow(row)
        train_f.flush()

        best_model_tracker.update(
            "train",
            float(episode_reward_sums[env_idx]),
            run_dir / "best_train_model.pt",
            save_checkpoint,
        )

        if episode % log_interval == 0:
            print(
                f"[{algo_name}] episode={episode} steps={global_step} "
                f"reward_sum={episode_reward_sums[env_idx]:.3f}{extra_log_str}"
            )
        episode += 1
        episode_reward_sums[env_idx] = 0.0
        if episode_lengths is not None:
            episode_lengths[env_idx] = 0

    return episode
