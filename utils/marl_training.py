"""
Shared utilities for MARL training scripts.

This module contains common setup and utility functions used across
all MARL training algorithms (IQL, VDN, QMIX, QPLEX, MAPPO).
"""

from __future__ import annotations

import csv
from pathlib import Path
import copy
import json
import os
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

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
    set_overrides: Optional[list] = None,
) -> Tuple[Dict[str, Any], str, int, bool, Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Load config and extract evaluation overrides.

    Args:
        config_path: Path to config JSON (or None for default)
        default_n_agents: Default number of agents if not in config
        use_agent_id_flag: Value of --no-agent-id flag (True if NOT set)
        set_overrides: List of "key=value" strings from ``--set`` CLI args

    Returns:
        Tuple of:
            - Full config dict
            - Config path as string (or None)
            - Number of agents
            - Whether to use agent ID
            - Evaluation reward override dict (or None)
            - Evaluation termination override dict (or None)
            - Network override dict for training envs (cfg["network"] with --set applied, or None)
            - Reward override dict for training envs (cfg["reward"] minus "evaluation" key, or None)
    """
    from ncs_env.config import load_config

    config_path_str = str(config_path) if config_path is not None else None
    cfg = load_config(config_path_str)
    cfg_base = copy.deepcopy(cfg)

    if set_overrides:
        from tools._common import parse_set_overrides, deep_merge
        overrides = parse_set_overrides(set_overrides)
        if overrides:
            deep_merge(cfg, overrides)
            # Persist merged config so non-reward/non-network overrides (e.g. controller/system)
            # are honored by environment constructors that load from config_path.
            # Keep eval network semantics unchanged by preserving the original network section;
            # training env still receives overridden network via network_override below.
            cfg_for_file = copy.deepcopy(cfg)
            if "network" in cfg_base:
                cfg_for_file["network"] = copy.deepcopy(cfg_base["network"])
            else:
                cfg_for_file.pop("network", None)
            fd, tmp_path = tempfile.mkstemp(prefix="ncs_cfg_merged_", suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
                json.dump(cfg_for_file, tmp_file)
            config_path_str = tmp_path
    system_cfg = cfg.get("system", {})
    n_agents = int(system_cfg.get("n_agents", default_n_agents))
    use_agent_id = use_agent_id_flag

    # Load evaluation reward config if present
    eval_reward_override: Optional[Dict[str, Any]] = None
    reward_cfg = cfg.get("reward", {})
    eval_reward_cfg = reward_cfg.get("evaluation", None)
    if isinstance(eval_reward_cfg, dict):
        eval_reward_override = dict(eval_reward_cfg)
        eval_reward_override.setdefault("reward_clip_min", None)
        eval_reward_override.setdefault("reward_clip_max", None)

    # Load evaluation termination config if present
    eval_termination_override: Optional[Dict[str, Any]] = None
    termination_cfg = cfg.get("termination", {})
    eval_termination_cfg = termination_cfg.get("evaluation", None)
    if isinstance(eval_termination_cfg, dict):
        eval_termination_override = eval_termination_cfg

    # Network override for training envs: in-memory cfg["network"] with --set overrides applied.
    # Eval envs load the config from the file directly (no network_override), so they always
    # use the original dropout rates and network settings from the config file.
    network_override: Optional[Dict[str, Any]] = cfg.get("network") or None

    # Reward override for training envs: in-memory cfg["reward"] (minus the "evaluation" sub-dict)
    # with --set overrides applied. Eval envs use eval_reward_override instead.
    training_reward_cfg = {k: v for k, v in cfg.get("reward", {}).items() if k != "evaluation"}
    training_reward_override: Optional[Dict[str, Any]] = training_reward_cfg or None

    return (
        cfg,
        config_path_str,
        n_agents,
        use_agent_id,
        eval_reward_override,
        eval_termination_override,
        network_override,
        training_reward_override,
    )


def create_environments(
    n_agents: int,
    episode_length: int,
    config_path_str: Optional[str],
    seed: Optional[int],
    eval_reward_override: Optional[Dict[str, Any]],
    eval_termination_override: Optional[Dict[str, Any]],
    network_override: Optional[Dict[str, Any]] = None,
    training_reward_override: Optional[Dict[str, Any]] = None,
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
        network_override: Network config override for the training env (--set network.* values).
            Eval env always loads network settings from the config file directly.
        training_reward_override: Reward config override for the training env (--set reward.* values).
            Eval env uses eval_reward_override instead.

    Returns:
        Tuple of (training environment, evaluation environment)
    """
    from ncs_env.env import NCS_Env

    env = NCS_Env(
        n_agents=n_agents,
        episode_length=episode_length,
        config_path=config_path_str,
        seed=seed,
        reward_override=training_reward_override,
        network_override=network_override,
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


def _heuristic_is_stochastic(policy_name: str) -> bool:
    return str(policy_name).strip().startswith("random_")


def resolve_training_eval_baseline(
    cfg: Dict[str, Any],
    n_agents: int,
) -> Dict[str, Any]:
    """Resolve training-time evaluation baseline policy from config.

    Config path:
      training_evaluation.baseline_policy

    Supported values:
      - "perfect_comm" (alias for always_send with perfect_communication=true)
      - Any heuristic name from tools/heuristic_policies.py (e.g., "random_20")
    """
    training_eval_cfg = cfg.get("training_evaluation", {})
    baseline_policy = "perfect_comm"
    if isinstance(training_eval_cfg, dict):
        raw_policy = training_eval_cfg.get("baseline_policy", baseline_policy)
        if raw_policy is not None:
            baseline_policy = str(raw_policy).strip()
    if not baseline_policy:
        baseline_policy = "perfect_comm"

    if baseline_policy == "perfect_comm":
        return {
            "label": "perfect_comm",
            "heuristic_policy": "always_send",
            "use_perfect_communication": True,
            "deterministic": True,
        }

    from tools.heuristic_policies import get_heuristic_policy

    try:
        get_heuristic_policy(
            baseline_policy,
            n_agents=max(1, int(n_agents)),
            seed=0,
            agent_index=0,
        )
    except Exception as exc:
        raise ValueError(
            "Invalid training_evaluation.baseline_policy="
            f"{baseline_policy!r}. Use 'perfect_comm' or a valid heuristic policy name."
        ) from exc

    return {
        "label": baseline_policy,
        "heuristic_policy": baseline_policy,
        "use_perfect_communication": False,
        "deterministic": not _heuristic_is_stochastic(baseline_policy),
    }


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
    if not bool(reward_cfg.get("normalize", False)):
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
    eval_baseline: Dict[str, Any],
) -> None:
    """Run paired-seed evaluation, write CSV row, update best model, and print."""
    from utils.marl.common import run_evaluation_vectorized_seeded

    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive")

    episode_seeds: List[Optional[int]]
    if seed is None:
        episode_seeds = [None for _ in range(int(n_episodes))]
    else:
        episode_seeds = [int(seed) + ep for ep in range(int(n_episodes))]

    mean_eval_reward, std_eval_reward, policy_rewards = run_evaluation_vectorized_seeded(
        eval_env=eval_env,
        agent=agent,
        n_eval_envs=n_eval_envs,
        n_agents=n_agents,
        n_actions=n_actions,
        use_agent_id=use_agent_id,
        device=device,
        episode_seeds=episode_seeds,
        obs_normalizer=obs_normalizer,
    )

    baseline_label = str(eval_baseline.get("label", "perfect_comm"))
    baseline_policy = str(eval_baseline.get("heuristic_policy", "always_send"))
    baseline_deterministic = bool(eval_baseline.get("deterministic", True))
    baseline_perfect_comm = bool(eval_baseline.get("use_perfect_communication", False))

    current_pc_states = eval_env.call("get_perfect_communication")
    current_pc = bool(current_pc_states[0]) if current_pc_states else False
    if baseline_perfect_comm != current_pc:
        eval_env.call("set_perfect_communication", baseline_perfect_comm)

    try:
        heuristic_name = None if baseline_policy == "always_send" else baseline_policy
        mean_baseline_reward, std_baseline_reward, baseline_rewards = run_evaluation_vectorized_seeded(
            eval_env=eval_env,
            agent=agent,
            n_eval_envs=n_eval_envs,
            n_agents=n_agents,
            n_actions=n_actions,
            use_agent_id=use_agent_id,
            device=device,
            episode_seeds=episode_seeds,
            obs_normalizer=obs_normalizer,
            heuristic_policy_name=heuristic_name,
            heuristic_deterministic=baseline_deterministic,
            fixed_action=1 if baseline_policy == "always_send" else None,
        )
    finally:
        if baseline_perfect_comm != current_pc:
            eval_env.call("set_perfect_communication", current_pc)

    if len(policy_rewards) != len(baseline_rewards):
        raise RuntimeError("Policy and baseline evaluation episode counts do not match")

    policy_arr = np.asarray(policy_rewards, dtype=np.float64)
    baseline_arr = np.asarray(baseline_rewards, dtype=np.float64)
    denom = np.maximum(np.abs(baseline_arr), 1e-8)
    drop_ratios = (baseline_arr - policy_arr) / denom
    mean_drop_ratio = float(np.mean(drop_ratios))
    std_drop_ratio = float(np.std(drop_ratios))

    drop_csv_path = run_dir / "evaluation_drop_stats.csv"
    write_drop_header = (not drop_csv_path.exists()) or drop_csv_path.stat().st_size == 0
    with drop_csv_path.open("a", newline="", encoding="utf-8") as drop_f:
        drop_writer = csv.writer(drop_f)
        if write_drop_header:
            drop_writer.writerow(
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
        drop_writer.writerow(
            [
                global_step,
                baseline_label,
                int(baseline_perfect_comm),
                mean_eval_reward,
                std_eval_reward,
                mean_baseline_reward,
                std_baseline_reward,
                mean_drop_ratio,
                std_drop_ratio,
                len(drop_ratios),
            ]
        )
        drop_f.flush()

    eval_writer.writerow([global_step, mean_eval_reward, std_eval_reward])
    eval_f.flush()

    best_model_tracker.update(
        "eval_drop_ratio", -mean_drop_ratio, run_dir / "best_model.pt", save_checkpoint
    )

    print(
        f"[{algo_name}] Eval at step {global_step}: "
        f"mean_reward={mean_eval_reward:.3f} std={std_eval_reward:.3f} | "
        f"baseline={baseline_label} "
        f"mean={mean_baseline_reward:.3f} std={std_baseline_reward:.3f} | "
        f"drop_ratio_mean={mean_drop_ratio:.6f} drop_ratio_std={std_drop_ratio:.6f}"
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
