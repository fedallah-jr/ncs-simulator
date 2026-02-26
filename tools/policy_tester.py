"""
Policy testing tool for the NCS simulator.

Evaluates a target policy against a fixed set of heuristic baselines across multiple seeds.
By default uses raw absolute reward without normalization; enable normalization via CLI.

Replay Testing:
    Use --test-replay to test policy determinism by recording actions on a recording seed
    (default 424242), then comparing original vs replay policies on test seeds.

    Works in both single-policy and batch (--models-root) modes:
    - Single mode: Creates replay_test/ subdirectory with comparison results
    - Batch mode: Creates separate replayboard.csv with original vs replay for each model
      (not included in main leaderboard)

    Reports for each policy:
    - How many environments yielded the same reward (exact matches)
    - How many times replay wins (replay reward > original reward)
    - How many times original wins (original reward > replay reward)
    - Statistical analysis of reward differences
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os

# Prevent thread thrashing when using ProcessPoolExecutor with PyTorch
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ncs_env.config import load_config
from ncs_env.env import NCS_Env
from tools.heuristic_policies import get_heuristic_policy
from tools._common import (
    MultiAgentHeuristicPolicy,
    load_es_policy,
    load_marl_torch_multi_agent_policy,
    infer_policy_n_agents,
    read_marl_torch_n_agents,
    read_es_n_agents,
    resolve_n_agents,
    sanitize_filename as _sanitize_filename,
    parse_set_overrides,
    deep_merge,
    apply_config_overrides,
    add_set_override_argument,
)

# Heuristic policies to compare against (edit as needed).
HEURISTIC_POLICY_NAMES: Sequence[str] = (
    "zero_wait",
    "perfect_sync",
    "perfect_sync_n2",
    "always_send",
    "never_send",
    "random_50",
)

# Names for stochastic heuristics that should use non-deterministic actions.
STOCHASTIC_HEURISTICS: Sequence[str] = ("random_50",)

# Reward override for evaluation: absolute reward, no normalization by default.
# Reward clipping is disabled by default but can be enabled via --use-reward-clipping.
EVAL_REWARD_OVERRIDE: Dict[str, Any] = {
    "state_error_reward": "absolute",
    "normalize": False,
    "reward_clip_min": None,
    "reward_clip_max": None,
}

# Keep evaluation seeds disjoint from training-time evaluation (0-10).
TRAINING_EVAL_SEED_COUNT = 11
DEFAULT_EVAL_SEED_START = 100

REWARD_COMPARISON_KEYS: Sequence[str] = (
    "comm_penalty_alpha",
    "simple_comm_penalty_alpha",
    "simple_freshness_decay",
    "comm_recent_window",
    "comm_throughput_window",
    "comm_throughput_floor",
)

NON_DYNAMICS_CONFIG_KEYS: Sequence[str] = (
    "use_agent_id",
    "include_agent_id",
    "no_agent_id",
    "agent_id",
)

# ---------------------------------------------------------------------------
# CSV fieldname constants (used in multiple places)
# ---------------------------------------------------------------------------
_PER_SEED_FIELDS_CORE: Tuple[str, ...] = (
    "policy_label",
    "policy_type",
    "seed",
    "total_reward",
    "mean_reward",
    "mean_state_error",
    "final_state_error",
    "total_lqr_cost",
    "mean_lqr_cost",
    "send_rate",
    "mean_true_goodput_kbps",
    "steps",
    "n_agents",
    "episode_length",
)

_PER_SEED_FIELDS_NETWORK: Tuple[str, ...] = (
    "tx_attempts",
    "tx_acked",
    "tx_dropped",
    "tx_rewrites",
    "tx_collisions",
    "data_delivered",
    "mac_ack_sent",
    "mac_ack_collisions",
    "ack_timeouts",
    "tx_success_rate",
    "data_delivery_rate",
    "collision_rate",
    "drop_rate",
    "rewrite_rate",
    "ack_collision_rate",
    "ack_timeout_rate",
)

_PER_SEED_FIELDS: Tuple[str, ...] = _PER_SEED_FIELDS_CORE + _PER_SEED_FIELDS_NETWORK

_SUMMARY_FIELDS_CORE: Tuple[str, ...] = (
    "policy_label",
    "policy_type",
    "num_seeds",
    "mean_total_reward",
    "std_total_reward",
    "mean_state_error",
    "std_state_error",
    "mean_final_error",
    "std_final_error",
    "mean_total_lqr_cost",
    "std_total_lqr_cost",
    "mean_lqr_cost",
    "std_lqr_cost",
    "mean_send_rate",
    "std_send_rate",
    "mean_true_goodput_kbps",
    "std_true_goodput_kbps",
    "mean_steps",
    "std_steps",
    "mean_reward_per_step",
    "std_reward_per_step",
    "completion_rate",
)

_SUMMARY_FIELDS_NETWORK: Tuple[str, ...] = (
    "mean_tx_attempts",
    "std_tx_attempts",
    "mean_tx_acked",
    "std_tx_acked",
    "mean_tx_dropped",
    "std_tx_dropped",
    "mean_tx_rewrites",
    "std_tx_rewrites",
    "mean_tx_collisions",
    "std_tx_collisions",
    "mean_data_delivered",
    "std_data_delivered",
    "mean_mac_ack_sent",
    "std_mac_ack_sent",
    "mean_mac_ack_collisions",
    "std_mac_ack_collisions",
    "mean_ack_timeouts",
    "std_ack_timeouts",
    "mean_tx_success_rate",
    "std_tx_success_rate",
    "mean_data_delivery_rate",
    "std_data_delivery_rate",
    "mean_collision_rate",
    "std_collision_rate",
    "mean_drop_rate",
    "std_drop_rate",
    "mean_rewrite_rate",
    "std_rewrite_rate",
    "mean_ack_collision_rate",
    "std_ack_collision_rate",
    "mean_ack_timeout_rate",
    "std_ack_timeout_rate",
)

_SUMMARY_FIELDS: Tuple[str, ...] = _SUMMARY_FIELDS_CORE + _SUMMARY_FIELDS_NETWORK

_LEADERBOARD_FIELDS: Tuple[str, ...] = (
    "rank",
    "model_name",
    "checkpoint",
    "policy_label",
    "policy_type",
    "num_seeds",
    "mean_total_reward",
    "std_total_reward",
    "mean_state_error",
    "std_state_error",
    "mean_final_error",
    "std_final_error",
    "mean_lqr_cost",
    "std_lqr_cost",
    "mean_send_rate",
    "std_send_rate",
    "mean_steps",
    "std_steps",
    "mean_reward_per_step",
    "std_reward_per_step",
    "completion_rate",
)

_NETWORK_LEADERBOARD_FIELDS: Tuple[str, ...] = (
    "rank",
    "model_name",
    "checkpoint",
    "policy_label",
    "policy_type",
    "num_seeds",
) + _SUMMARY_FIELDS_NETWORK

_REPLAYBOARD_FIELDS: Tuple[str, ...] = (
    "model_name",
    "checkpoint",
    "policy_type",
    "num_seeds",
    "mean_total_reward",
    "std_total_reward",
    "mean_state_error",
    "std_state_error",
    "mean_final_error",
    "std_final_error",
    "mean_lqr_cost",
    "std_lqr_cost",
    "mean_send_rate",
    "std_send_rate",
    "mean_steps",
    "std_steps",
    "mean_reward_per_step",
    "std_reward_per_step",
    "completion_rate",
)


@dataclass(frozen=True)
class PolicySpec:
    label: str
    policy_type: str
    policy_path: str


@dataclass
class EpisodeResult:
    policy_label: str
    policy_type: str
    seed: int
    total_reward: float
    mean_reward: float
    mean_state_error: float
    final_state_error: float
    total_lqr_cost: float
    mean_lqr_cost: float
    send_rate: float
    mean_true_goodput_kbps: float
    steps: int
    n_agents: int
    episode_length: int
    tx_attempts: int
    tx_acked: int
    tx_dropped: int
    tx_rewrites: int
    tx_collisions: int
    data_delivered: int
    mac_ack_sent: int
    mac_ack_collisions: int
    ack_timeouts: int


@dataclass(frozen=True)
class ModelRun:
    name: str
    path: Path
    config_path: Path
    best_model: Optional[Path]
    best_train_model: Optional[Path]
    latest_model: Optional[Path]


class ReplayPolicy:
    """Policy that replays a pre-recorded action sequence."""
    def __init__(self, recorded_actions: List[Dict[str, int]], n_agents: int) -> None:
        self.recorded_actions = recorded_actions
        self.n_agents = int(n_agents)
        self.step_index = 0

    def reset(self) -> None:
        self.step_index = 0

    def act(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, int]:
        if self.step_index >= len(self.recorded_actions):
            # Return zero actions if we've exhausted the recording
            return {f"agent_{i}": 0 for i in range(self.n_agents)}
        actions = dict(self.recorded_actions[self.step_index])
        self.step_index += 1
        return actions


def _episode_result_to_seed_row(
    result: EpisodeResult,
    *,
    include_network_stats: bool = False,
) -> Dict[str, Any]:
    """Build a per-seed CSV row from an EpisodeResult."""
    row: Dict[str, Any] = {
        "policy_label": result.policy_label,
        "policy_type": result.policy_type,
        "seed": result.seed,
        "total_reward": result.total_reward,
        "mean_reward": result.mean_reward,
        "mean_state_error": result.mean_state_error,
        "final_state_error": result.final_state_error,
        "total_lqr_cost": result.total_lqr_cost,
        "mean_lqr_cost": result.mean_lqr_cost,
        "send_rate": result.send_rate,
        "mean_true_goodput_kbps": result.mean_true_goodput_kbps,
        "steps": result.steps,
        "n_agents": result.n_agents,
        "episode_length": result.episode_length,
    }
    if include_network_stats:
        row.update({
            "tx_attempts": result.tx_attempts,
            "tx_acked": result.tx_acked,
            "tx_dropped": result.tx_dropped,
            "tx_rewrites": result.tx_rewrites,
            "tx_collisions": result.tx_collisions,
            "data_delivered": result.data_delivered,
            "mac_ack_sent": result.mac_ack_sent,
            "mac_ack_collisions": result.mac_ack_collisions,
            "ack_timeouts": result.ack_timeouts,
            "tx_success_rate": _safe_rate(result.tx_acked, result.tx_attempts),
            "data_delivery_rate": _safe_rate(result.data_delivered, result.tx_attempts),
            "collision_rate": _safe_rate(result.tx_collisions, result.tx_attempts),
            "drop_rate": _safe_rate(result.tx_dropped, result.tx_attempts),
            "rewrite_rate": _safe_rate(result.tx_rewrites, result.tx_attempts),
            "ack_collision_rate": _safe_rate(result.mac_ack_collisions, result.mac_ack_sent),
            "ack_timeout_rate": _safe_rate(result.ack_timeouts, result.tx_attempts),
        })
    return row


def _iter_seeds(seed_start: int, num_seeds: int) -> List[int]:
    return list(range(int(seed_start), int(seed_start) + int(num_seeds)))


def _is_stochastic_heuristic(policy_name: str) -> bool:
    return policy_name in STOCHASTIC_HEURISTICS or policy_name.startswith("random_")


def _log(message: str, *, indent: int = 0) -> None:
    print(f"{' ' * indent}{message}")


def _format_seed_range(seeds: Sequence[int]) -> str:
    if not seeds:
        return "none"
    if len(seeds) == 1:
        return str(seeds[0])
    return f"{seeds[0]}..{seeds[-1]}"


def _sum_network_stat(network_stats: Dict[str, Any], key: str) -> int:
    values = network_stats.get(key)
    if isinstance(values, list):
        return int(sum(int(x) for x in values))
    if values is None:
        return 0
    try:
        return int(values)
    except (TypeError, ValueError):
        return 0


def _extract_network_totals(info: Dict[str, Any]) -> Dict[str, int]:
    network_stats = info.get("network_stats", {})
    return {
        "tx_attempts": _sum_network_stat(network_stats, "tx_attempts"),
        "tx_acked": _sum_network_stat(network_stats, "tx_acked"),
        "tx_dropped": _sum_network_stat(network_stats, "tx_dropped"),
        "tx_rewrites": _sum_network_stat(network_stats, "tx_rewrites"),
        "tx_collisions": _sum_network_stat(network_stats, "tx_collisions"),
        "data_delivered": _sum_network_stat(network_stats, "data_delivered"),
        "mac_ack_sent": _sum_network_stat(network_stats, "mac_ack_sent"),
        "mac_ack_collisions": _sum_network_stat(network_stats, "mac_ack_collisions"),
        "ack_timeouts": _sum_network_stat(network_stats, "ack_timeouts"),
    }


def _safe_rate(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _record_policy_actions(
    policy: Any,
    config_path: Path,
    episode_length: int,
    n_agents: int,
    recording_seed: int,
    termination_override: Optional[Dict[str, Any]],
    reward_override: Optional[Dict[str, Any]],
) -> List[Dict[str, int]]:
    """Record actions from a policy for one episode."""
    env = _build_env(
        config_path,
        episode_length,
        n_agents,
        recording_seed,
        termination_override,
        reward_override,
    )
    try:
        if hasattr(policy, "reset"):
            policy.reset()
        obs_dict, _ = env.reset(seed=recording_seed)
        recorded_actions: List[Dict[str, int]] = []

        for _ in range(episode_length):
            action_dict = policy.act(obs_dict)
            recorded_actions.append(dict(action_dict))
            obs_dict, _, terminated, truncated, _ = env.step(action_dict)

            done = any(
                bool(terminated[f"agent_{i}"]) or bool(truncated[f"agent_{i}"])
                for i in range(n_agents)
            )
            if done:
                break

        return recorded_actions
    finally:
        if hasattr(env, "close"):
            env.close()


@dataclass
class ReplayComparison:
    """Results of comparing replay vs original policy."""
    seed: int
    original_reward: float
    replay_reward: float
    reward_diff: float
    match: bool  # True if rewards are exactly equal
    original_wins: bool  # True if original > replay
    replay_wins: bool  # True if replay > original


def _compare_replay_vs_original(
    original_results: List[EpisodeResult],
    replay_results: List[EpisodeResult],
    tolerance: float = 1e-6,
) -> Tuple[List[ReplayComparison], Dict[str, Any]]:
    """Compare original policy vs replay policy results."""
    if len(original_results) != len(replay_results):
        raise ValueError("Result lists must have the same length")

    comparisons: List[ReplayComparison] = []
    for orig, replay in zip(original_results, replay_results):
        if orig.seed != replay.seed:
            raise ValueError(f"Seed mismatch: {orig.seed} vs {replay.seed}")

        reward_diff = float(orig.total_reward - replay.total_reward)
        match = abs(reward_diff) < tolerance
        original_wins = reward_diff > tolerance
        replay_wins = reward_diff < -tolerance

        comparisons.append(
            ReplayComparison(
                seed=orig.seed,
                original_reward=orig.total_reward,
                replay_reward=replay.total_reward,
                reward_diff=reward_diff,
                match=match,
                original_wins=original_wins,
                replay_wins=replay_wins,
            )
        )

    # Compute summary statistics
    n_total = len(comparisons)
    n_matches = sum(1 for c in comparisons if c.match)
    n_original_wins = sum(1 for c in comparisons if c.original_wins)
    n_replay_wins = sum(1 for c in comparisons if c.replay_wins)

    avg_reward_diff = float(np.mean([c.reward_diff for c in comparisons]))
    std_reward_diff = float(np.std([c.reward_diff for c in comparisons]))
    max_reward_diff = float(max(abs(c.reward_diff) for c in comparisons))

    summary = {
        "n_total": n_total,
        "n_matches": n_matches,
        "n_original_wins": n_original_wins,
        "n_replay_wins": n_replay_wins,
        "match_rate": float(n_matches) / float(n_total) if n_total > 0 else 0.0,
        "original_win_rate": float(n_original_wins) / float(n_total) if n_total > 0 else 0.0,
        "replay_win_rate": float(n_replay_wins) / float(n_total) if n_total > 0 else 0.0,
        "avg_reward_diff": avg_reward_diff,
        "std_reward_diff": std_reward_diff,
        "max_reward_diff": max_reward_diff,
        "avg_original_reward": float(np.mean([c.original_reward for c in comparisons])),
        "avg_replay_reward": float(np.mean([c.replay_reward for c in comparisons])),
    }

    return comparisons, summary


def _build_reward_override(
    use_reward_normalization: bool,
    use_reward_clipping: bool = False,
) -> Dict[str, Any]:
    reward_override = dict(EVAL_REWARD_OVERRIDE)
    if use_reward_normalization:
        reward_override["normalize"] = True
    # Disable reward clipping during evaluation unless explicitly enabled
    if not use_reward_clipping:
        reward_override["reward_clip_min"] = None
        reward_override["reward_clip_max"] = None
    else:
        # When clipping is enabled, remove these keys so config values are used
        reward_override.pop("reward_clip_min", None)
        reward_override.pop("reward_clip_max", None)
    return reward_override


def _write_perfect_comm_config(config: Dict[str, Any], output_path: Path) -> Path:
    config_copy = copy.deepcopy(config)
    network_cfg = config_copy.setdefault("network", {})
    network_cfg["perfect_communication"] = True
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(config_copy, handle, indent=2, sort_keys=True, ensure_ascii=True)
    return output_path


def _write_perfect_control_config(
    config: Dict[str, Any],
    output_path: Path,
) -> Path:
    config_copy = copy.deepcopy(config)
    network_cfg = config_copy.setdefault("network", {})
    network_cfg["perfect_communication"] = True
    controller_cfg = config_copy.setdefault("controller", {})
    controller_cfg["use_true_state_control"] = True
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(config_copy, handle, indent=2, sort_keys=True, ensure_ascii=True)
    return output_path


def _resolve_n_agents(
    config: Dict[str, Any],
    specs: Sequence[PolicySpec],
    explicit_n_agents: Optional[int],
) -> int:
    """Thin wrapper that adapts PolicySpec objects into (path, type) tuples."""
    return resolve_n_agents(
        config,
        [(spec.policy_path, spec.policy_type) for spec in specs],
        explicit_n_agents,
    )


def _build_env(
    config_path: Path,
    episode_length: int,
    n_agents: int,
    seed: int,
    termination_override: Optional[Dict[str, Any]],
    reward_override: Optional[Dict[str, Any]],
) -> NCS_Env:
    return NCS_Env(
        n_agents=n_agents,
        episode_length=episode_length,
        config_path=str(config_path),
        seed=seed,
        reward_override=reward_override,
        termination_override=termination_override,
        track_true_goodput=True,
        track_lqr_cost=True,
    )


def _load_policy(
    spec: PolicySpec,
    env: Any,
    *,
    seed: int,
) -> Any:
    policy_type = spec.policy_type.lower()
    if policy_type == "heuristic":
        deterministic = not _is_stochastic_heuristic(spec.policy_path)
        return MultiAgentHeuristicPolicy(
            spec.policy_path,
            n_agents=getattr(env, "n_agents", 1),
            seed=seed,
            deterministic=deterministic,
        )
    if policy_type in {"es", "openai_es"}:
        return load_es_policy(spec.policy_path, env)
    if policy_type == "marl_torch":
        return load_marl_torch_multi_agent_policy(spec.policy_path, env)
    raise ValueError(f"Unknown policy type: {spec.policy_type}")


def _run_multi_agent_episode(
    env: NCS_Env,
    policy: Any,
    *,
    seed: int,
    episode_length: int,
) -> EpisodeResult:
    if hasattr(policy, "reset"):
        policy.reset()
    obs_dict, info = env.reset(seed=seed)
    n_agents = int(env.n_agents)

    total_reward = 0.0
    total_state_error = 0.0
    total_lqr_cost = 0.0
    send_count = 0
    steps = 0
    true_goodput_sum = 0.0
    true_goodput_steps = 0
    last_info = info
    for _ in range(episode_length):
        action_dict = policy.act(obs_dict)
        obs_dict, rewards, terminated, truncated, info = env.step(action_dict)
        total_reward += float(sum(rewards.values()))
        send_count += int(sum(action_dict.values()))
        if "lqr_cost_total" in info:
            total_lqr_cost += float(info["lqr_cost_total"])

        # Track state error separately from rewards
        states = np.asarray(info.get("states", []), dtype=float)
        if states.size > 0:
            for agent_idx, state in enumerate(states):
                state_error = env._compute_state_error(state)
                total_state_error += float(state_error)
        steps += 1
        last_info = info
        if "true_goodput_kbps_total" in info:
            true_goodput_sum += float(info["true_goodput_kbps_total"])
            true_goodput_steps += 1
        done = any(bool(terminated[f"agent_{i}"]) or bool(truncated[f"agent_{i}"]) for i in range(n_agents))
        if done:
            break

    states = np.asarray(last_info.get("states", []), dtype=float)
    if states.size:
        final_error = float(np.mean(np.linalg.norm(states, axis=1)))
    else:
        final_error = 0.0
    mean_reward = total_reward / float(max(1, steps * n_agents))
    mean_state_error = total_state_error / float(max(1, steps * n_agents))
    mean_lqr_cost = total_lqr_cost / float(max(1, steps * n_agents))
    send_rate = float(send_count) / float(max(1, steps * n_agents))
    mean_true_goodput_kbps = true_goodput_sum / float(max(1, true_goodput_steps))
    network_totals = _extract_network_totals(last_info)
    return EpisodeResult(
        policy_label="",
        policy_type="",
        seed=int(seed),
        total_reward=total_reward,
        mean_reward=mean_reward,
        mean_state_error=mean_state_error,
        final_state_error=final_error,
        total_lqr_cost=total_lqr_cost,
        mean_lqr_cost=mean_lqr_cost,
        send_rate=send_rate,
        mean_true_goodput_kbps=mean_true_goodput_kbps,
        steps=steps,
        n_agents=n_agents,
        episode_length=episode_length,
        **network_totals,
    )


def _mean_std(arr: np.ndarray, has_data: bool) -> Tuple[float, float]:
    if not has_data:
        return 0.0, 0.0
    return float(np.mean(arr)), float(np.std(arr))


def _summarize_results(results: List[EpisodeResult]) -> Dict[str, float]:
    has_data = bool(results)
    steps = np.array([r.steps for r in results], dtype=float)
    episode_length = results[0].episode_length if results else 0
    completion_rate = float(np.mean(steps == episode_length)) if results else 0.0

    # (dict_key_prefix, array) pairs for mean/std computation
    direct_fields: Sequence[Tuple[str, np.ndarray]] = (
        ("total_reward", np.array([r.total_reward for r in results], dtype=float)),
        ("state_error", np.array([r.mean_state_error for r in results], dtype=float)),
        ("final_error", np.array([r.final_state_error for r in results], dtype=float)),
        ("total_lqr_cost", np.array([r.total_lqr_cost for r in results], dtype=float)),
        ("lqr_cost", np.array([r.mean_lqr_cost for r in results], dtype=float)),
        ("send_rate", np.array([r.send_rate for r in results], dtype=float)),
        ("true_goodput_kbps", np.array([r.mean_true_goodput_kbps for r in results], dtype=float)),
        ("steps", steps),
        ("reward_per_step", np.array([r.mean_reward for r in results], dtype=float)),
        ("tx_attempts", np.array([r.tx_attempts for r in results], dtype=float)),
        ("tx_acked", np.array([r.tx_acked for r in results], dtype=float)),
        ("tx_dropped", np.array([r.tx_dropped for r in results], dtype=float)),
        ("tx_rewrites", np.array([r.tx_rewrites for r in results], dtype=float)),
        ("tx_collisions", np.array([r.tx_collisions for r in results], dtype=float)),
        ("data_delivered", np.array([r.data_delivered for r in results], dtype=float)),
        ("mac_ack_sent", np.array([r.mac_ack_sent for r in results], dtype=float)),
        ("mac_ack_collisions", np.array([r.mac_ack_collisions for r in results], dtype=float)),
        ("ack_timeouts", np.array([r.ack_timeouts for r in results], dtype=float)),
        ("tx_success_rate", np.array([_safe_rate(r.tx_acked, r.tx_attempts) for r in results], dtype=float)),
        ("data_delivery_rate", np.array([_safe_rate(r.data_delivered, r.tx_attempts) for r in results], dtype=float)),
        ("collision_rate", np.array([_safe_rate(r.tx_collisions, r.tx_attempts) for r in results], dtype=float)),
        ("drop_rate", np.array([_safe_rate(r.tx_dropped, r.tx_attempts) for r in results], dtype=float)),
        ("rewrite_rate", np.array([_safe_rate(r.tx_rewrites, r.tx_attempts) for r in results], dtype=float)),
        ("ack_collision_rate", np.array([_safe_rate(r.mac_ack_collisions, r.mac_ack_sent) for r in results], dtype=float)),
        ("ack_timeout_rate", np.array([_safe_rate(r.ack_timeouts, r.tx_attempts) for r in results], dtype=float)),
    )

    out: Dict[str, float] = {}
    for key, arr in direct_fields:
        m, s = _mean_std(arr, has_data)
        out[f"mean_{key}"] = m
        out[f"std_{key}"] = s
    out["completion_rate"] = completion_rate
    return out


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _filter_rows(rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> List[Dict[str, Any]]:
    return [{name: row.get(name, "") for name in fieldnames} for row in rows]


def _strip_non_dynamics_config_fields(value: Any) -> Any:
    """Remove agent-representation keys that do not change environment dynamics."""
    if isinstance(value, dict):
        cleaned: Dict[str, Any] = {}
        for key, item in value.items():
            if key in NON_DYNAMICS_CONFIG_KEYS:
                continue
            cleaned[key] = _strip_non_dynamics_config_fields(item)
        return cleaned
    if isinstance(value, list):
        return [_strip_non_dynamics_config_fields(item) for item in value]
    return value


def _extract_env_signature(config: Dict[str, Any]) -> Dict[str, Any]:
    signature: Dict[str, Any] = {}
    for key in ("system", "lqr", "network", "observation", "termination", "controller"):
        if key in config:
            signature[key] = _strip_non_dynamics_config_fields(config.get(key))
    reward_cfg = config.get("reward", {})
    signature["reward"] = _strip_non_dynamics_config_fields(
        {key: reward_cfg.get(key) for key in REWARD_COMPARISON_KEYS if key in reward_cfg}
    )
    return signature


def _config_signature_string(config: Dict[str, Any]) -> str:
    signature = _extract_env_signature(config)
    return json.dumps(signature, sort_keys=True, separators=(",", ":"))


def _discover_model_runs(models_root: Path) -> List[ModelRun]:
    runs: List[ModelRun] = []
    for entry in sorted(models_root.iterdir()):
        if not entry.is_dir():
            continue
        config_path = entry / "config.json"
        if not config_path.exists():
            continue

        def _select_checkpoint(stem: str) -> Optional[Path]:
            for suffix in (".pt", ".npz"):
                candidate = entry / f"{stem}{suffix}"
                if candidate.exists():
                    return candidate
            return None

        best_model = _select_checkpoint("best_model")
        best_train_model = _select_checkpoint("best_train_model")
        latest_model = _select_checkpoint("latest_model")
        if best_model is None and latest_model is None and best_train_model is None:
            continue
        runs.append(
            ModelRun(
                name=entry.name,
                path=entry,
                config_path=config_path,
                best_model=best_model,
                best_train_model=best_train_model,
                latest_model=latest_model,
            )
        )
    return runs


def _infer_policy_type(model_path: Path) -> str:
    suffix = model_path.suffix.lower()
    if suffix == ".pt":
        return "marl_torch"
    if suffix == ".npz":
        return "es"
    raise ValueError(f"Unsupported model extension: {model_path}")


def _chunk_seeds(seeds: Sequence[int], num_chunks: int) -> List[List[int]]:
    if not seeds:
        return []
    if num_chunks <= 1:
        return [list(seeds)]
    total = len(seeds)
    num_chunks = min(num_chunks, total)
    base, extra = divmod(total, num_chunks)
    chunks: List[List[int]] = []
    start = 0
    for idx in range(num_chunks):
        size = base + (1 if idx < extra else 0)
        end = start + size
        chunks.append(list(seeds[start:end]))
        start = end
    return chunks


def _evaluate_policy_for_seeds(
    spec: PolicySpec,
    config_path: Path,
    episode_length: int,
    n_agents: int,
    seeds: Sequence[int],
    termination_override: Optional[Dict[str, Any]],
    reward_override: Optional[Dict[str, Any]],
) -> List[EpisodeResult]:
    if not seeds:
        return []
    results: List[EpisodeResult] = []
    cached_policy: Optional[Any] = None
    env = _build_env(
        config_path,
        episode_length,
        n_agents,
        int(seeds[0]),
        termination_override,
        reward_override,
    )
    try:
        for seed in seeds:
            if spec.policy_type.lower() == "heuristic":
                policy = _load_policy(spec, env, seed=int(seed))
            else:
                if cached_policy is None:
                    policy = _load_policy(spec, env, seed=int(seed))
                    cached_policy = policy
                else:
                    policy = cached_policy

            episode = _run_multi_agent_episode(
                env,
                policy,
                seed=int(seed),
                episode_length=episode_length,
            )
            episode.policy_label = spec.label
            episode.policy_type = spec.policy_type
            results.append(episode)
    finally:
        if hasattr(env, "close"):
            env.close()

    return results


def _evaluate_policy(
    spec: PolicySpec,
    *,
    config_path: Path,
    episode_length: int,
    n_agents: int,
    seeds: Sequence[int],
    termination_override: Optional[Dict[str, Any]],
    reward_override: Optional[Dict[str, Any]],
    num_workers: int = 1,
) -> List[EpisodeResult]:
    if not seeds:
        return []
    worker_count = max(1, int(num_workers))

    chunks = _chunk_seeds(seeds, worker_count)
    if len(chunks) <= 1:
        results = _evaluate_policy_for_seeds(
            spec,
            config_path=config_path,
            episode_length=episode_length,
            n_agents=n_agents,
            seeds=seeds,
            termination_override=termination_override,
            reward_override=reward_override,
        )
        results.sort(key=lambda r: r.seed)
        return results

    with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
        results_by_chunk = list(
            executor.map(
                _evaluate_policy_for_seeds,
                [spec] * len(chunks),
                [config_path] * len(chunks),
                [episode_length] * len(chunks),
                [n_agents] * len(chunks),
                chunks,
                [termination_override] * len(chunks),
                [reward_override] * len(chunks),
            )
        )

    results = [episode for chunk in results_by_chunk for episode in chunk]
    results.sort(key=lambda r: r.seed)
    return results


def _write_policy_results(
    run_dir: Path,
    spec: PolicySpec,
    results: List[EpisodeResult],
) -> Dict[str, Any]:
    per_seed_rows = [_episode_result_to_seed_row(r, include_network_stats=True) for r in results]
    summary = _summarize_results(results)
    summary_row = {
        "policy_label": spec.label,
        "policy_type": spec.policy_type,
        "num_seeds": len(results),
        **summary,
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(run_dir / "per_seed_results.csv", _PER_SEED_FIELDS, per_seed_rows)
    _write_csv(run_dir / "summary_results.csv", _SUMMARY_FIELDS, [summary_row])
    return summary_row


def _write_replay_comparison(
    run_dir: Path,
    comparisons: List[ReplayComparison],
    summary: Dict[str, Any],
    recording_seed: int,
) -> None:
    """Write replay vs original comparison results to files."""
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write per-seed comparison
    comparison_rows: List[Dict[str, Any]] = []
    for comp in comparisons:
        comparison_rows.append(
            {
                "seed": comp.seed,
                "original_reward": comp.original_reward,
                "replay_reward": comp.replay_reward,
                "reward_diff": comp.reward_diff,
                "match": comp.match,
                "original_wins": comp.original_wins,
                "replay_wins": comp.replay_wins,
            }
        )

    _write_csv(
        run_dir / "replay_comparison.csv",
        [
            "seed",
            "original_reward",
            "replay_reward",
            "reward_diff",
            "match",
            "original_wins",
            "replay_wins",
        ],
        comparison_rows,
    )

    # Write summary with recording info
    summary_with_meta = {
        "recording_seed": recording_seed,
        **summary,
    }

    with (run_dir / "replay_summary.json").open("w") as f:
        json.dump(summary_with_meta, f, indent=2, sort_keys=True, ensure_ascii=True)


def _write_ranked_csv(path: Path, fieldnames: Sequence[str], rows: List[Dict[str, Any]]) -> None:
    _write_csv(path, fieldnames, _filter_rows(rows, fieldnames))


def _smooth_rewards(rewards: np.ndarray, window_size: int) -> np.ndarray:
    """Apply moving average smoothing to reward curve."""
    if len(rewards) < window_size:
        return rewards
    smoothed = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
    # Pad the beginning to maintain the same length
    padding = np.full(window_size - 1, smoothed[0])
    return np.concatenate([padding, smoothed])


def _select_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col_name in candidates:
        if col_name in df.columns:
            return col_name
    return None


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _compute_training_stats(model_dir: Path) -> Optional[Dict[str, Union[str, float, int]]]:
    training_csv = model_dir / "training_rewards.csv"
    if not training_csv.exists():
        return None

    try:
        train_df = pd.read_csv(str(training_csv))
        if len(train_df) == 0:
            return None

        episode_col = _select_column(train_df, ["episode", "Episode", "generation", "Generation"])
        reward_col = _select_column(
            train_df,
            ["reward_sum", "reward", "total_reward", "episode_reward", "mean_reward"],
        )
        if reward_col is None:
            raise ValueError(f"Could not find reward column. Available: {list(train_df.columns)}")

        rewards = train_df[reward_col].astype(float)
        max_idx = int(np.argmax(rewards))
        min_idx = int(np.argmin(rewards))
        max_episode = _to_float(train_df[episode_col].iloc[max_idx]) if episode_col else float("nan")
        min_episode = _to_float(train_df[episode_col].iloc[min_idx]) if episode_col else float("nan")

        return {
            "episode_column": episode_col or "",
            "reward_column": reward_col,
            "count": int(len(rewards)),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "max_reward": float(rewards.iloc[max_idx]),
            "max_reward_episode": max_episode,
            "min_reward": float(rewards.iloc[min_idx]),
            "min_reward_episode": min_episode,
        }
    except Exception as exc:
        print(f"Warning: Could not compute training stats: {exc}")
        return None


def _compute_evaluation_stats(model_dir: Path) -> Optional[Dict[str, Union[str, float, int]]]:
    eval_csv = model_dir / "evaluation_rewards.csv"
    if not eval_csv.exists():
        return None

    try:
        eval_df = pd.read_csv(str(eval_csv))
        if len(eval_df) == 0:
            return None

        step_col = _select_column(eval_df, ["step", "Step", "steps", "Steps"])
        mean_reward_col = _select_column(eval_df, ["mean_reward", "reward", "avg_reward"])
        std_reward_col = _select_column(eval_df, ["std_reward", "reward_std", "std"])
        if mean_reward_col is None:
            raise ValueError(f"Could not find mean reward column. Available: {list(eval_df.columns)}")

        mean_rewards = eval_df[mean_reward_col].astype(float)
        max_idx = int(np.argmax(mean_rewards))
        min_idx = int(np.argmin(mean_rewards))
        max_step = _to_float(eval_df[step_col].iloc[max_idx]) if step_col else float("nan")
        min_step = _to_float(eval_df[step_col].iloc[min_idx]) if step_col else float("nan")

        mean_std_reward = None
        if std_reward_col is not None:
            std_rewards = eval_df[std_reward_col].astype(float)
            mean_std_reward = float(np.mean(std_rewards))

        stats: Dict[str, Union[str, float, int]] = {
            "step_column": step_col or "",
            "mean_reward_column": mean_reward_col,
            "std_reward_column": std_reward_col or "",
            "count": int(len(mean_rewards)),
            "mean_of_mean_reward": float(np.mean(mean_rewards)),
            "std_of_mean_reward": float(np.std(mean_rewards)),
            "max_mean_reward": float(mean_rewards.iloc[max_idx]),
            "max_mean_reward_step": max_step,
            "min_mean_reward": float(mean_rewards.iloc[min_idx]),
            "min_mean_reward_step": min_step,
        }
        if mean_std_reward is not None:
            stats["mean_of_std_reward"] = mean_std_reward
        return stats
    except Exception as exc:
        print(f"Warning: Could not compute evaluation stats: {exc}")
        return None


def _write_reward_stats(
    run_dir: Path,
    training_stats: Optional[Dict[str, Union[str, float, int]]],
    evaluation_stats: Optional[Dict[str, Union[str, float, int]]],
) -> None:
    if training_stats is None and evaluation_stats is None:
        return
    payload = {
        "training": training_stats,
        "evaluation": evaluation_stats,
    }
    path = run_dir / "reward_stats.json"
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)


def _reward_normalization_disabled(config_path: Path) -> bool:
    if not config_path.exists():
        return False
    try:
        config = load_config(str(config_path))
    except Exception as exc:
        print(f"Warning: Could not read config for reward normalization: {exc}")
        return False
    reward_cfg = config.get("reward", {})
    if not isinstance(reward_cfg, dict):
        return False
    normalize = reward_cfg.get("normalize", False)
    return not bool(normalize)


def _load_reward_csv(
    path: Path,
    log_warnings: bool = True,
) -> Optional[pd.DataFrame]:
    """Load a reward CSV and return DataFrame, or None if missing/empty."""
    if not path.exists():
        if log_warnings:
            print(f"Warning: Rewards not found at {path}")
        return None
    df = pd.read_csv(str(path))
    if len(df) == 0:
        if log_warnings:
            print(f"Warning: Rewards file is empty at {path}")
        return None
    return df


def _compute_y_limits(*smoothed_arrays: np.ndarray, padding: float = 0.15) -> Tuple[float, float]:
    """Compute padded y-axis limits from one or more smoothed arrays."""
    combined = np.concatenate(smoothed_arrays)
    y_min, y_max = float(np.min(combined)), float(np.max(combined))
    y_range = y_max - y_min
    y_pad = y_range * padding
    return y_min - y_pad, y_max + y_pad


def _save_plot(
    fig: plt.Figure,
    path: Path,
    log_warnings: bool = True,
    description: str = "plot",
) -> None:
    """Save a matplotlib figure, create parent dirs, then close."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    if log_warnings:
        print(f"Saved {description} to {path}")


def _plot_combined_rewards(
    model_dir: Path,
    output_path: Path,
    model_name: str = "Model",
    *,
    log_warnings: bool = True,
) -> bool:
    """Plot training and evaluation rewards on a shared y-axis when scales match."""
    train_df = _load_reward_csv(model_dir / "training_rewards.csv", log_warnings)
    eval_df = _load_reward_csv(model_dir / "evaluation_rewards.csv", log_warnings)
    if train_df is None or eval_df is None:
        return False

    try:
        episode_col = _select_column(train_df, ["episode", "Episode", "generation", "Generation"])
        reward_col = _select_column(train_df, ["reward", "reward_sum", "total_reward", "episode_reward", "mean_reward"])
        if episode_col is None or reward_col is None:
            raise ValueError(f"Could not find episode/reward columns. Available: {list(train_df.columns)}")

        step_col = _select_column(eval_df, ["step", "Step", "steps", "Steps"])
        mean_reward_col = _select_column(eval_df, ["mean_reward", "reward", "avg_reward"])
        std_reward_col = _select_column(eval_df, ["std_reward", "reward_std", "std"])
        if step_col is None or mean_reward_col is None:
            raise ValueError(f"Could not find step/reward columns. Available: {list(eval_df.columns)}")

        episodes = train_df[episode_col].values
        rewards = train_df[reward_col].values
        smoothed_train = _smooth_rewards(rewards, window_size=200)

        steps = eval_df[step_col].values
        mean_rewards = eval_df[mean_reward_col].values
        std_rewards = eval_df[std_reward_col].values if std_reward_col is not None else None
        smoothed_eval = _smooth_rewards(mean_rewards, window_size=20)

        y_lim_min, y_lim_max = _compute_y_limits(smoothed_train, smoothed_eval)

        smoothed_std = _smooth_rewards(std_rewards, window_size=20) if std_rewards is not None else None

        fig, ax_train = plt.subplots(figsize=(7, 5))
        ax_eval = ax_train.twiny()

        ax_train.plot(episodes, rewards, alpha=0.3, linewidth=0.5, color="blue", label="Train Raw")
        ax_train.plot(episodes, smoothed_train, linewidth=2, color="blue", label="Train Smoothed (window=200)")

        ax_eval.plot(steps, mean_rewards, alpha=0.3, linewidth=0.5, color="green", label="Eval Raw")
        ax_eval.plot(steps, smoothed_eval, linewidth=2, color="green", label="Eval Smoothed (window=20)")

        if smoothed_std is not None:
            ax_eval.fill_between(
                steps,
                smoothed_eval - smoothed_std,
                smoothed_eval + smoothed_std,
                alpha=0.2,
                color="green",
                label="Eval +/- 1 std",
            )

        ax_train.set_ylim(y_lim_min, y_lim_max)

        train_xlabel = "Generation" if "generation" in episode_col.lower() else "Episode"
        ax_train.set_xlabel(train_xlabel)
        ax_eval.set_xlabel("Evaluation Steps")
        ax_train.set_ylabel("Reward (shared scale)")
        ax_train.set_title(f"{model_name} - Training vs Evaluation Rewards")
        ax_train.grid(True, alpha=0.3)

        train_handles, train_labels = ax_train.get_legend_handles_labels()
        eval_handles, eval_labels = ax_eval.get_legend_handles_labels()
        ax_train.legend(train_handles + eval_handles, train_labels + eval_labels, loc="best")

        plt.tight_layout()
        _save_plot(fig, output_path, log_warnings, "combined rewards plot")
        return True
    except Exception as exc:
        if log_warnings:
            print(f"Warning: Could not plot combined rewards: {exc}")
        return False


def _plot_training_rewards(
    model_dir: Path,
    output_path: Path,
    model_name: str = "Model",
    *,
    log_warnings: bool = True,
) -> bool:
    """Plot training reward curves from a model directory."""
    train_df = _load_reward_csv(model_dir / "training_rewards.csv", log_warnings)
    if train_df is None:
        return False

    try:
        episode_col = _select_column(train_df, ["episode", "Episode", "generation", "Generation"])
        reward_col = _select_column(train_df, ["reward", "reward_sum", "total_reward", "episode_reward", "mean_reward"])
        if episode_col is None or reward_col is None:
            raise ValueError(f"Could not find episode/reward columns. Available: {list(train_df.columns)}")

        episodes = train_df[episode_col].values
        rewards = train_df[reward_col].values
        smoothed = _smooth_rewards(rewards, window_size=200)

        y_lim_min, y_lim_max = _compute_y_limits(smoothed)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(episodes, rewards, alpha=0.3, linewidth=0.5, color="blue", label="Raw")
        ax.plot(episodes, smoothed, linewidth=2, color="blue", label="Smoothed (window=200)")
        ax.set_ylim(y_lim_min, y_lim_max)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Training Reward")
        ax.set_title(f"{model_name} - Training Rewards")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        _save_plot(fig, output_path, log_warnings, "training rewards plot")
        return True
    except Exception as exc:
        if log_warnings:
            print(f"Warning: Could not plot training rewards: {exc}")
        return False


def _plot_evaluation_rewards(
    model_dir: Path,
    output_path: Path,
    model_name: str = "Model",
    *,
    log_warnings: bool = True,
) -> bool:
    """Plot evaluation reward curves from a model directory."""
    eval_df = _load_reward_csv(model_dir / "evaluation_rewards.csv", log_warnings)
    if eval_df is None:
        return False

    try:
        step_col = _select_column(eval_df, ["step", "Step", "steps", "Steps"])
        mean_reward_col = _select_column(eval_df, ["mean_reward", "reward", "avg_reward"])
        std_reward_col = _select_column(eval_df, ["std_reward", "reward_std", "std"])
        if step_col is None or mean_reward_col is None:
            raise ValueError(f"Could not find step/reward columns. Available: {list(eval_df.columns)}")

        steps = eval_df[step_col].values
        mean_rewards = eval_df[mean_reward_col].values
        std_rewards = eval_df[std_reward_col].values if std_reward_col is not None else None
        smoothed = _smooth_rewards(mean_rewards, window_size=20)

        y_lim_min, y_lim_max = _compute_y_limits(smoothed)
        smoothed_std = _smooth_rewards(std_rewards, window_size=20) if std_rewards is not None else None

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(steps, mean_rewards, alpha=0.3, linewidth=0.5, color="green", label="Raw")
        ax.plot(steps, smoothed, linewidth=2, color="green", label="Smoothed (window=20)")

        if smoothed_std is not None:
            ax.fill_between(
                steps,
                smoothed - smoothed_std,
                smoothed + smoothed_std,
                alpha=0.2,
                color="green",
                label="+/- 1 std",
            )

        ax.set_ylim(y_lim_min, y_lim_max)
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Evaluation Reward (mean)")
        ax.set_title(f"{model_name} - Evaluation Rewards")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        _save_plot(fig, output_path, log_warnings, "evaluation rewards plot")
        return True
    except Exception as exc:
        if log_warnings:
            print(f"Warning: Could not plot evaluation rewards: {exc}")
        return False


def _plot_reward_curves(
    model_dir: Path,
    config_path: Path,
    model_name: str = "Model",
    *,
    log_warnings: bool = True,
) -> List[str]:
    if _reward_normalization_disabled(config_path):
        combined_path = model_dir / "training_evaluation_rewards.png"
        if _plot_combined_rewards(model_dir, combined_path, model_name=model_name, log_warnings=log_warnings):
            return [str(combined_path)]

    plotted_paths: List[str] = []
    training_path = model_dir / "training_rewards.png"
    evaluation_path = model_dir / "evaluation_rewards.png"
    if _plot_training_rewards(model_dir, training_path, model_name=model_name, log_warnings=log_warnings):
        plotted_paths.append(str(training_path))
    if _plot_evaluation_rewards(model_dir, evaluation_path, model_name=model_name, log_warnings=log_warnings):
        plotted_paths.append(str(evaluation_path))
    return plotted_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate policies against heuristic baselines.")
    parser.add_argument("--config", help="Path to config JSON")
    parser.add_argument("--models-root", help="Root folder containing model_* subfolders")
    parser.add_argument("--policy", help="Path to policy file or heuristic name")
    parser.add_argument(
        "--policy-type",
        choices=["es", "openai_es", "heuristic", "marl_torch"],
        help="Policy type for the target policy",
    )
    parser.add_argument("--policy-label", default=None, help="Label for the target policy")
    parser.add_argument("--episode-length", type=int, default=250, help="Episode length to evaluate")
    parser.add_argument(
        "--n-agents",
        type=int,
        default=None,
        help="Optional override for agent count (default: read from checkpoint or config)",
    )
    parser.add_argument("--num-seeds", type=int, default=50, help="Number of seeds to evaluate")
    default_workers = os.cpu_count() or 1
    parser.add_argument(
        "--num-workers",
        type=int,
        default=default_workers,
        help="Number of parallel workers for evaluation (default: CPU count)",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=DEFAULT_EVAL_SEED_START,
        help="First seed value (default avoids training evaluation seeds 0-10)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/policy_tests",
        help="Directory to write results",
    )
    parser.add_argument(
        "--use-reward-normalization",
        action="store_true",
        help="Enable running reward normalization during evaluation (default: disabled)",
    )
    parser.add_argument(
        "--use-reward-clipping",
        action="store_true",
        help="Enable reward clipping during evaluation using config values (default: disabled)",
    )
    parser.add_argument(
        "--no-override",
        action="store_true",
        help="Disable evaluation reward overrides and use the config reward as-is",
    )
    parser.add_argument(
        "--only-heuristics",
        action="store_true",
        help="Evaluate only heuristic baselines (zero_wait, perfect_sync, perfect_sync_n2, always_send, never_send, random_50)",
    )
    parser.add_argument(
        "--test-replay",
        action="store_true",
        help="Test determinism by recording actions and comparing replay vs original policy",
    )
    parser.add_argument(
        "--replay-recording-seed",
        type=int,
        default=424242,
        help="Seed to use for recording actions (default: 424242, unlikely to conflict with eval seeds)",
    )
    parser.add_argument(
        "--replay-tolerance",
        type=float,
        default=1e-6,
        help="Tolerance for considering rewards as matching (default: 1e-6)",
    )
    add_set_override_argument(parser)
    args = parser.parse_args()

    if args.num_seeds < 1:
        raise ValueError("--num-seeds must be >= 1")
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >= 1")
    config_overrides = parse_set_overrides(args.set_overrides)
    # Build reward override based on flags
    if args.no_override:
        # With --no-override, only disable clipping if not explicitly enabled
        if not args.use_reward_clipping:
            reward_override = {
                "reward_clip_min": None,
                "reward_clip_max": None,
            }
        else:
            reward_override = {}
    else:
        reward_override = _build_reward_override(
            bool(args.use_reward_normalization),
            bool(args.use_reward_clipping),
        )
    if args.no_override and args.use_reward_normalization:
        _log("Note: --use-reward-normalization ignored because --no-override is set.")
    if args.no_override and args.use_reward_clipping:
        _log("Note: Using reward clipping from config.")

    if args.test_replay and args.only_heuristics:
        raise ValueError("--test-replay cannot be used with --only-heuristics.")

    if args.only_heuristics:
        if args.models_root or args.policy or args.policy_type:
            raise ValueError("--only-heuristics cannot be used with --models-root, --policy, or --policy-type.")
        if not args.config:
            raise ValueError("--only-heuristics requires --config.")
        # Treat as single-policy mode but skip the target policy
        models_root = None
    elif args.models_root:
        if args.policy or args.policy_type or args.config:
            raise ValueError("Use --models-root alone, or specify --config/--policy/--policy-type for single runs.")
        models_root = Path(args.models_root)
        if not models_root.exists():
            raise FileNotFoundError(f"Models root not found: {models_root}")
    else:
        if not args.config or not args.policy or not args.policy_type:
            raise ValueError("Single-policy evaluation requires --config, --policy, and --policy-type.")
        models_root = None

    if models_root is None:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        config = load_config(str(config_path))

        termination_override = config.get("termination", {}).get("evaluation")
        if not isinstance(termination_override, dict):
            termination_override = None
        seeds = _iter_seeds(args.seed_start, args.num_seeds)
        output_root = Path(args.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)

        # Build policy specs list
        policy_specs = []
        if args.only_heuristics:
            target_label = "heuristics_only"
            policy_type = None
        else:
            policy_type = args.policy_type.lower()
            target_label = args.policy_label or _sanitize_filename(
                Path(args.policy).name if policy_type != "heuristic" else args.policy
            )
            policy_specs.append(PolicySpec(label=target_label, policy_type=policy_type, policy_path=args.policy))

        for heuristic_name in HEURISTIC_POLICY_NAMES:
            policy_specs.append(
                PolicySpec(label=heuristic_name, policy_type="heuristic", policy_path=heuristic_name)
            )

        run_dir = output_root / f"policy_test_{_sanitize_filename(target_label)}"
        run_dir.mkdir(parents=True, exist_ok=True)

        if config_overrides:
            config_path = apply_config_overrides(
                config_path, config_overrides, run_dir / "overridden_config.json"
            )
            config = load_config(str(config_path))
            termination_override = config.get("termination", {}).get("evaluation")
            if not isinstance(termination_override, dict):
                termination_override = None

        resolved_n_agents = _resolve_n_agents(config, policy_specs, args.n_agents)
        policy_type_names = [spec.policy_type.lower() for spec in policy_specs]
        allowed_policy_types = {"marl_torch", "heuristic", "es", "openai_es"}
        if not all(policy in allowed_policy_types for policy in policy_type_names):
            raise ValueError("Supported policy types: marl_torch, es, openai_es, heuristic")
        if resolved_n_agents < 1:
            raise ValueError("Resolved n_agents must be >= 1.")

        _log("Policy tester (heuristics only)" if args.only_heuristics else "Policy tester (single policy)")
        _log(f"Config: {config_path}")
        if config_overrides:
            _log("Config overrides:")
            for arg in args.set_overrides:
                _log(f"  --set {arg}")
        _log(f"Seeds: {_format_seed_range(seeds)} ({len(seeds)})")
        if args.num_workers > 1:
            _log(f"Workers: {args.num_workers}")
        if args.test_replay and not args.only_heuristics:
            _log(f"Replay testing: enabled (recording seed {args.replay_recording_seed})")
        if args.no_override:
            clipping_status = "with clipping from config" if args.use_reward_clipping else "no clipping"
            _log(f"Evaluation: reward from config (no override, {clipping_status}).")
        elif args.use_reward_normalization:
            clipping_status = "with clipping" if args.use_reward_clipping else "no clipping"
            _log(f"Evaluation: absolute reward with running normalization; {clipping_status}; comm/termination from config.")
        else:
            clipping_status = "with clipping" if args.use_reward_clipping else "no clipping"
            _log(f"Evaluation: raw absolute reward; {clipping_status}; comm/termination from config.")
        _log("Policies:")
        for spec in policy_specs:
            _log(f"- {spec.label} ({spec.policy_type})", indent=2)

        per_seed_rows: List[Dict[str, Any]] = []
        summary_rows: List[Dict[str, Any]] = []

        for idx, spec in enumerate(policy_specs, start=1):
            _log(f"[{idx}/{len(policy_specs)}] {spec.label} ({spec.policy_type})")
            results = _evaluate_policy(
                spec,
                config_path=config_path,
                episode_length=int(args.episode_length),
                n_agents=resolved_n_agents,
                seeds=seeds,
                termination_override=termination_override,
                reward_override=reward_override,
                num_workers=int(args.num_workers),
            )
            for result in results:
                per_seed_rows.append(_episode_result_to_seed_row(result))

            summary_rows.append(
                {
                    "policy_label": spec.label,
                    "policy_type": spec.policy_type,
                    "num_seeds": len(results),
                    **_summarize_results(results),
                }
            )

        # Add perfect communication baseline if heuristics-only mode
        if args.only_heuristics:
            perfect_comm_config_path = _write_perfect_comm_config(
                config, run_dir / "perfect_comm_config.json"
            )
            _log("Perfect communication baseline:")
            perfect_comm_spec = PolicySpec(
                label="perfect_comm_always_send",
                policy_type="heuristic",
                policy_path="always_send",
            )
            perfect_comm_results = _evaluate_policy(
                perfect_comm_spec,
                config_path=perfect_comm_config_path,
                episode_length=int(args.episode_length),
                n_agents=resolved_n_agents,
                seeds=seeds,
                termination_override=termination_override,
                reward_override=reward_override,
                num_workers=int(args.num_workers),
            )
            for result in perfect_comm_results:
                per_seed_rows.append(_episode_result_to_seed_row(result))
            summary_rows.append(
                {
                    "policy_label": "perfect_comm_always_send",
                    "policy_type": "heuristic",
                    "num_seeds": len(perfect_comm_results),
                    **_summarize_results(perfect_comm_results),
                }
            )
            perfect_control_config_path = _write_perfect_control_config(
                config, run_dir / "perfect_control_config.json"
            )
            _log("Perfect control baseline:")
            perfect_control_spec = PolicySpec(
                label="perfect_control_always_send",
                policy_type="heuristic",
                policy_path="always_send",
            )
            perfect_control_results = _evaluate_policy(
                perfect_control_spec,
                config_path=perfect_control_config_path,
                episode_length=int(args.episode_length),
                n_agents=resolved_n_agents,
                seeds=seeds,
                termination_override=termination_override,
                reward_override=reward_override,
                num_workers=int(args.num_workers),
            )
            for result in perfect_control_results:
                per_seed_rows.append(_episode_result_to_seed_row(result))
            summary_rows.append(
                {
                    "policy_label": "perfect_control_always_send",
                    "policy_type": "heuristic",
                    "num_seeds": len(perfect_control_results),
                    **_summarize_results(perfect_control_results),
                }
            )

        # Replay test: compare original policy vs replay on test seeds
        if args.test_replay and not args.only_heuristics and policy_type is not None:
            _log("\nReplay determinism test:")
            _log(f"Recording actions on seed {args.replay_recording_seed}")

            # Load the target policy
            env_for_loading = _build_env(
                config_path,
                int(args.episode_length),
                resolved_n_agents,
                args.replay_recording_seed,
                termination_override,
                reward_override,
            )
            try:
                target_spec = PolicySpec(
                    label=target_label,
                    policy_type=policy_type,
                    policy_path=args.policy,
                )
                original_policy = _load_policy(target_spec, env_for_loading, seed=args.replay_recording_seed)
            finally:
                if hasattr(env_for_loading, "close"):
                    env_for_loading.close()

            # Record actions
            _log("Recording policy actions...")
            recorded_actions = _record_policy_actions(
                original_policy,
                config_path,
                int(args.episode_length),
                resolved_n_agents,
                args.replay_recording_seed,
                termination_override,
                reward_override,
            )
            _log(f"Recorded {len(recorded_actions)} timesteps")

            # Evaluate original policy on test seeds
            _log(f"Evaluating original policy on {len(seeds)} test seeds...")
            original_results = _evaluate_policy(
                target_spec,
                config_path=config_path,
                episode_length=int(args.episode_length),
                n_agents=resolved_n_agents,
                seeds=seeds,
                termination_override=termination_override,
                reward_override=reward_override,
                num_workers=int(args.num_workers),
            )

            # Create and evaluate replay policy
            _log(f"Evaluating replay policy on {len(seeds)} test seeds...")
            replay_spec = PolicySpec(
                label=f"{target_label}_replay",
                policy_type="replay",
                policy_path="replay",
            )

            # For replay, we need to manually run episodes since we need the custom policy
            replay_results: List[EpisodeResult] = []
            for seed in seeds:
                replay_policy = ReplayPolicy(recorded_actions, resolved_n_agents)
                env = _build_env(
                    config_path,
                    int(args.episode_length),
                    resolved_n_agents,
                    seed,
                    termination_override,
                    reward_override,
                )
                try:
                    episode = _run_multi_agent_episode(
                        env,
                        replay_policy,
                        seed=seed,
                        episode_length=int(args.episode_length),
                    )
                    episode.policy_label = replay_spec.label
                    episode.policy_type = replay_spec.policy_type
                    replay_results.append(episode)
                finally:
                    if hasattr(env, "close"):
                        env.close()

            # Compare results
            _log("Comparing original vs replay...")
            comparisons, summary = _compare_replay_vs_original(
                original_results,
                replay_results,
                tolerance=args.replay_tolerance,
            )

            # Write comparison results
            replay_dir = run_dir / "replay_test"
            _write_replay_comparison(
                replay_dir,
                comparisons,
                summary,
                args.replay_recording_seed,
            )

            # Log summary
            _log("\nReplay test results:")
            _log(f"  Total episodes: {summary['n_total']}")
            _log(f"  Exact matches: {summary['n_matches']} ({summary['match_rate']:.2%})")
            _log(f"  Original wins: {summary['n_original_wins']} ({summary['original_win_rate']:.2%})")
            _log(f"  Replay wins: {summary['n_replay_wins']} ({summary['replay_win_rate']:.2%})")
            _log(f"  Avg reward diff: {summary['avg_reward_diff']:.6f} ± {summary['std_reward_diff']:.6f}")
            _log(f"  Max reward diff: {summary['max_reward_diff']:.6f}")
            _log(f"  Avg original reward: {summary['avg_original_reward']:.2f}")
            _log(f"  Avg replay reward: {summary['avg_replay_reward']:.2f}")
            _log(f"\nReplay test results saved to: {replay_dir}")
            _log("")

        _write_csv(run_dir / "per_seed_results.csv", _PER_SEED_FIELDS_CORE, per_seed_rows)
        _write_csv(run_dir / "summary_results.csv", _SUMMARY_FIELDS, summary_rows)

        _log("Outputs:")
        _log(str(run_dir / "per_seed_results.csv"), indent=2)
        _log(str(run_dir / "summary_results.csv"), indent=2)

        # Plot training curves and save stats if the policy is from a trained model
        if policy_type and policy_type != "heuristic":
            policy_path = Path(args.policy)
            model_dir = policy_path.parent
            training_stats = _compute_training_stats(model_dir)
            evaluation_stats = _compute_evaluation_stats(model_dir)
            _write_reward_stats(run_dir, training_stats, evaluation_stats)
            plotted_paths = _plot_reward_curves(
                model_dir,
                model_dir / "config.json",
                model_name=target_label,
                log_warnings=False,
            )
            if plotted_paths:
                _log("Plots:")
                for path in plotted_paths:
                    _log(path, indent=2)

        return 0

    runs = _discover_model_runs(models_root)
    if not runs:
        raise ValueError(f"No model folders found under {models_root}")

    config_map: Dict[str, Tuple[Path, str]] = {}
    for run in runs:
        cfg = load_config(str(run.config_path))
        config_map[run.name] = (run.config_path, _config_signature_string(cfg))

    reference_name = runs[0].name
    reference_signature = config_map[reference_name][1]
    mismatched = [
        name
        for name, (_, signature) in config_map.items()
        if signature != reference_signature
    ]
    if mismatched:
        mismatch_paths = ", ".join(sorted(mismatched))
        raise ValueError(
            "Config mismatch across model folders. "
            f"Ensure environment sections match for all runs. Offending runs: {mismatch_paths}"
        )

    reference_config = load_config(str(runs[0].config_path))
    termination_override = reference_config.get("termination", {}).get("evaluation")
    if not isinstance(termination_override, dict):
        termination_override = None
    if config_overrides:
        deep_merge(reference_config, config_overrides)
        eval_config_path: Optional[Path] = models_root / "overridden_config.json"
        eval_config_path.parent.mkdir(parents=True, exist_ok=True)
        with eval_config_path.open("w") as handle:
            json.dump(reference_config, handle, indent=2, sort_keys=True, ensure_ascii=True)
        termination_override = reference_config.get("termination", {}).get("evaluation")
        if not isinstance(termination_override, dict):
            termination_override = None
    else:
        eval_config_path = None
    model_specs_by_run: Dict[str, List[Tuple[str, PolicySpec]]] = {}
    run_lookup = {run.name: run for run in runs}
    policy_types: List[str] = []
    for run in runs:
        specs: List[Tuple[str, PolicySpec]] = []
        for checkpoint_name, model_path in (
            ("best", run.best_model),
            ("best_train", run.best_train_model),
            ("latest", run.latest_model),
        ):
            if model_path is None:
                continue
            policy_type = _infer_policy_type(model_path)
            policy_types.append(policy_type)
            label = f"{run.name}/{checkpoint_name}"
            specs.append(
                (checkpoint_name, PolicySpec(label=label, policy_type=policy_type, policy_path=str(model_path)))
            )
        if specs:
            model_specs_by_run[run.name] = specs

    if not model_specs_by_run:
        raise ValueError("No model checkpoints found under the specified root.")

    unique_policy_types = sorted(set(policy_types))
    allowed_policy_types = {"marl_torch", "es"}
    if not set(unique_policy_types).issubset(allowed_policy_types):
        raise ValueError("Batch mode supports only marl_torch and es checkpoints.")

    resolved_n_agents = _resolve_n_agents(
        reference_config,
        [spec for specs in model_specs_by_run.values() for _, spec in specs],
        args.n_agents,
    )
    if resolved_n_agents < 1:
        raise ValueError("Resolved n_agents must be >= 1.")

    seeds = _iter_seeds(args.seed_start, args.num_seeds)
    leaderboard_rows: List[Dict[str, Any]] = []
    replayboard_rows: List[Dict[str, Any]] = []

    total_models = len(model_specs_by_run)
    total_checkpoints = sum(len(specs) for specs in model_specs_by_run.values())
    _log("Policy tester (batch)")
    _log(f"Models: {total_models}")
    _log(f"Checkpoints: {total_checkpoints}")
    _log(f"Seeds: {_format_seed_range(seeds)} ({len(seeds)})")
    if config_overrides:
        _log("Config overrides:")
        for arg in args.set_overrides:
            _log(f"  --set {arg}")
    if args.num_workers > 1:
        _log(f"Workers: {args.num_workers}")
    if args.test_replay:
        _log(f"Replay testing: enabled (recording seed {args.replay_recording_seed})")
    if args.no_override:
        clipping_status = "with clipping from config" if args.use_reward_clipping else "no clipping"
        _log(f"Evaluation: reward from config (no override, {clipping_status}).")
    elif args.use_reward_normalization:
        clipping_status = "with clipping" if args.use_reward_clipping else "no clipping"
        _log(f"Evaluation: absolute reward with running normalization; {clipping_status}; comm/termination from config.")
    else:
        clipping_status = "with clipping" if args.use_reward_clipping else "no clipping"
        _log(f"Evaluation: raw absolute reward; {clipping_status}; comm/termination from config.")

    for model_idx, (model_name, specs) in enumerate(model_specs_by_run.items(), start=1):
        _log(f"[{model_idx}/{total_models}] {model_name}")
        run = run_lookup[model_name]
        training_stats = _compute_training_stats(run.path)
        evaluation_stats = _compute_evaluation_stats(run.path)
        for checkpoint_name, spec in specs:
            _log(f"  checkpoint: {checkpoint_name} ({spec.policy_type})")
            results = _evaluate_policy(
                spec,
                config_path=eval_config_path or run.config_path,
                episode_length=int(args.episode_length),
                n_agents=resolved_n_agents,
                seeds=seeds,
                termination_override=termination_override,
                reward_override=reward_override,
                num_workers=int(args.num_workers),
            )
            run_dir = models_root / model_name / "policy_tests" / f"{checkpoint_name}_eval"
            summary_row = _write_policy_results(run_dir, spec, results)
            summary_row["model_name"] = model_name
            summary_row["checkpoint"] = checkpoint_name
            leaderboard_rows.append(summary_row)
            _write_reward_stats(run_dir, training_stats, evaluation_stats)
            _log(f"    results: {run_dir / 'summary_results.csv'}")

            # Replay test for this checkpoint
            if args.test_replay:
                _log(f"    replay test: recording seed {args.replay_recording_seed}")

                # Load policy for recording
                env_for_loading = _build_env(
                    eval_config_path or run.config_path,
                    int(args.episode_length),
                    resolved_n_agents,
                    args.replay_recording_seed,
                    termination_override,
                    reward_override,
                )
                try:
                    policy_for_recording = _load_policy(spec, env_for_loading, seed=args.replay_recording_seed)
                finally:
                    if hasattr(env_for_loading, "close"):
                        env_for_loading.close()

                # Record actions
                recorded_actions = _record_policy_actions(
                    policy_for_recording,
                    eval_config_path or run.config_path,
                    int(args.episode_length),
                    resolved_n_agents,
                    args.replay_recording_seed,
                    termination_override,
                    reward_override,
                )

                # Evaluate replay policy
                replay_results: List[EpisodeResult] = []
                for seed in seeds:
                    replay_policy = ReplayPolicy(recorded_actions, resolved_n_agents)
                    env = _build_env(
                        eval_config_path or run.config_path,
                        int(args.episode_length),
                        resolved_n_agents,
                        seed,
                        termination_override,
                        reward_override,
                    )
                    try:
                        episode = _run_multi_agent_episode(
                            env,
                            replay_policy,
                            seed=seed,
                            episode_length=int(args.episode_length),
                        )
                        episode.policy_label = f"{spec.label}_replay"
                        episode.policy_type = "replay"
                        replay_results.append(episode)
                    finally:
                        if hasattr(env, "close"):
                            env.close()

                # Write replay results
                replay_run_dir = models_root / model_name / "policy_tests" / f"{checkpoint_name}_replay_eval"
                replay_summary_row = _write_policy_results(replay_run_dir,
                    PolicySpec(label=f"{spec.label}_replay", policy_type="replay", policy_path="replay"),
                    replay_results)
                replay_summary_row["model_name"] = model_name
                replay_summary_row["checkpoint"] = checkpoint_name

                # Compare and write comparison
                comparisons, comparison_summary = _compare_replay_vs_original(
                    results,
                    replay_results,
                    tolerance=args.replay_tolerance,
                )
                replay_comparison_dir = models_root / model_name / "policy_tests" / f"{checkpoint_name}_replay_comparison"
                _write_replay_comparison(
                    replay_comparison_dir,
                    comparisons,
                    comparison_summary,
                    args.replay_recording_seed,
                )

                # Add to replayboard (both original and replay)
                original_row = dict(summary_row)
                original_row["policy_type"] = "original"
                replayboard_rows.append(original_row)

                replay_row = dict(replay_summary_row)
                replay_row["policy_type"] = "replay"
                replayboard_rows.append(replay_row)

                _log(f"    replay: matches={comparison_summary['n_matches']}/{comparison_summary['n_total']} "
                     f"({comparison_summary['match_rate']:.1%}), "
                     f"orig_wins={comparison_summary['n_original_wins']}, "
                     f"replay_wins={comparison_summary['n_replay_wins']}")

        plotted_paths = _plot_reward_curves(
            run.path,
            run.config_path,
            model_name=model_name,
            log_warnings=False,
        )
        if plotted_paths:
            _log("  plots:")
            for path in plotted_paths:
                _log(path, indent=4)

    heuristic_config_path = eval_config_path or runs[0].config_path
    _log("Heuristics:")
    for heuristic_name in HEURISTIC_POLICY_NAMES:
        _log(f"- {heuristic_name}", indent=2)
        spec = PolicySpec(label=heuristic_name, policy_type="heuristic", policy_path=heuristic_name)
        results = _evaluate_policy(
            spec,
            config_path=heuristic_config_path,
            episode_length=int(args.episode_length),
            n_agents=resolved_n_agents,
            seeds=seeds,
            termination_override=termination_override,
            reward_override=reward_override,
            num_workers=int(args.num_workers),
        )
        run_dir = models_root / "heuristics" / f"{heuristic_name}_eval"
        summary_row = _write_policy_results(run_dir, spec, results)
        summary_row["model_name"] = "heuristics"
        summary_row["checkpoint"] = "n/a"
        leaderboard_rows.append(summary_row)
        _log(f"results: {run_dir / 'summary_results.csv'}", indent=4)

    perfect_comm_root = models_root / "perfect_comm"
    perfect_comm_config_path = _write_perfect_comm_config(
        reference_config, perfect_comm_root / "perfect_comm_config.json"
    )
    _log("Perfect communication baseline:")
    _log(f"config: {perfect_comm_config_path}", indent=2)
    perfect_comm_spec = PolicySpec(
        label="perfect_comm/always_send",
        policy_type="heuristic",
        policy_path="always_send",
    )
    perfect_comm_results = _evaluate_policy(
        perfect_comm_spec,
        config_path=perfect_comm_config_path,
        episode_length=int(args.episode_length),
        n_agents=resolved_n_agents,
        seeds=seeds,
        termination_override=termination_override,
        reward_override=reward_override,
        num_workers=int(args.num_workers),
    )
    perfect_comm_dir = perfect_comm_root / "policy_tests" / "always_send_eval"
    perfect_comm_summary = _write_policy_results(perfect_comm_dir, perfect_comm_spec, perfect_comm_results)
    perfect_comm_summary["model_name"] = "perfect_comm"
    perfect_comm_summary["checkpoint"] = "always_send"
    leaderboard_rows.append(perfect_comm_summary)
    _log(f"results: {perfect_comm_dir / 'summary_results.csv'}", indent=2)

    perfect_control_root = models_root / "perfect_control"
    perfect_control_config_path = _write_perfect_control_config(
        reference_config,
        perfect_control_root / "perfect_control_config.json",
    )
    _log("Perfect control baseline:")
    _log(f"config: {perfect_control_config_path}", indent=2)
    perfect_control_spec = PolicySpec(
        label="perfect_control/always_send",
        policy_type="heuristic",
        policy_path="always_send",
    )
    perfect_control_results = _evaluate_policy(
        perfect_control_spec,
        config_path=perfect_control_config_path,
        episode_length=int(args.episode_length),
        n_agents=resolved_n_agents,
        seeds=seeds,
        termination_override=termination_override,
        reward_override=reward_override,
        num_workers=int(args.num_workers),
    )
    perfect_control_dir = perfect_control_root / "policy_tests" / "always_send_eval"
    perfect_control_summary = _write_policy_results(
        perfect_control_dir,
        perfect_control_spec,
        perfect_control_results,
    )
    perfect_control_summary["model_name"] = "perfect_control"
    perfect_control_summary["checkpoint"] = "always_send"
    leaderboard_rows.append(perfect_control_summary)
    _log(f"results: {perfect_control_dir / 'summary_results.csv'}", indent=2)

    leaderboard_rows.sort(key=lambda row: float(row["mean_total_reward"]), reverse=True)
    ranked_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(leaderboard_rows, start=1):
        ranked = dict(row)
        ranked["rank"] = idx
        ranked_rows.append(ranked)

    leaderboard_path = models_root / "leaderboard.csv"
    _write_ranked_csv(leaderboard_path, _LEADERBOARD_FIELDS, ranked_rows)
    _log(f"Leaderboard: {leaderboard_path}")
    network_stats_path = models_root / "leaderboard_network_stats.csv"
    _write_ranked_csv(network_stats_path, _NETWORK_LEADERBOARD_FIELDS, ranked_rows)
    _log(f"Network stats: {network_stats_path}")

    # Write replayboard if replay testing was enabled
    if args.test_replay and replayboard_rows:
        replayboard_path = models_root / "replayboard.csv"
        _write_ranked_csv(replayboard_path, _REPLAYBOARD_FIELDS, replayboard_rows)
        _log(f"Replayboard (original vs replay): {replayboard_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
