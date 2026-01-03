"""
Policy testing tool for the NCS simulator.

Evaluates a target policy against a fixed set of heuristic baselines across multiple seeds,
using raw absolute reward (no normalization).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

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
from utils.wrapper import SingleAgentWrapper
from tools.visualize_policy import (
    load_es_policy,
    load_marl_torch_multi_agent_policy,
    load_sb3_policy,
)

# Heuristic policies to compare against (edit as needed).
HEURISTIC_POLICY_NAMES: Sequence[str] = ("zero_wait", "always_send", "random_50")

# Names for stochastic heuristics that should use non-deterministic actions.
STOCHASTIC_HEURISTICS: Sequence[str] = ("random_50",)

# Reward override for evaluation: raw absolute reward, no normalization.
EVAL_REWARD_OVERRIDE: Dict[str, Any] = {
    "state_error_reward": "absolute",
    "normalize": False,
}

REWARD_COMPARISON_KEYS: Sequence[str] = (
    "state_cost_matrix",
    "comm_penalty_alpha",
    "simple_comm_penalty_alpha",
    "simple_freshness_decay",
    "comm_recent_window",
    "comm_throughput_window",
    "comm_throughput_floor",
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
    final_state_error: float
    send_rate: float
    steps: int
    n_agents: int
    episode_length: int


@dataclass(frozen=True)
class ModelRun:
    name: str
    path: Path
    config_path: Path
    best_model: Optional[Path]
    latest_model: Optional[Path]


class MultiAgentHeuristicPolicy:
    def __init__(self, policy_name: str, n_agents: int, seed: Optional[int], deterministic: bool) -> None:
        self.policy_name = policy_name
        self.n_agents = int(n_agents)
        self.deterministic = bool(deterministic)
        self._policies = []
        for idx in range(self.n_agents):
            agent_seed = None if seed is None else int(seed) + idx
            self._policies.append(get_heuristic_policy(policy_name, n_agents=1, seed=agent_seed))

    def reset(self) -> None:
        for policy in self._policies:
            if hasattr(policy, "reset"):
                policy.reset()

    def act(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, int]:
        actions: Dict[str, int] = {}
        for idx in range(self.n_agents):
            obs = obs_dict[f"agent_{idx}"]
            action, _ = self._policies[idx].predict(obs, deterministic=self.deterministic)
            actions[f"agent_{idx}"] = int(action)
        return actions


def _sanitize_filename(value: str) -> str:
    keep = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("_.")
    return out if out else "policy"


def _iter_seeds(seed_start: int, num_seeds: int) -> List[int]:
    return list(range(int(seed_start), int(seed_start) + int(num_seeds)))


def _is_stochastic_heuristic(policy_name: str) -> bool:
    return policy_name in STOCHASTIC_HEURISTICS


def _read_marl_torch_n_agents(model_path: Path) -> Optional[int]:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required to read marl_torch checkpoints") from exc

    ckpt = torch.load(str(model_path), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError("MARL torch checkpoint must be a dict")
    if "n_agents" not in ckpt:
        return None
    return int(ckpt["n_agents"])


def _read_es_n_agents(model_path: Path) -> Optional[int]:
    try:
        with np.load(str(model_path)) as data:
            if "n_agents" not in data:
                return None
            return int(data["n_agents"])
    except Exception as exc:
        raise ValueError(f"Could not load numpy data from {model_path}: {exc}") from exc


def _infer_policy_n_agents(spec: PolicySpec) -> Optional[int]:
    policy_type = spec.policy_type.lower()
    if policy_type == "sb3":
        return 1
    if policy_type == "marl_torch":
        return _read_marl_torch_n_agents(Path(spec.policy_path))
    if policy_type in {"es", "openai_es"}:
        return _read_es_n_agents(Path(spec.policy_path))
    return None


def _resolve_n_agents(
    config: Dict[str, Any],
    specs: Sequence[PolicySpec],
    explicit_n_agents: Optional[int],
) -> int:
    inferred_values: List[int] = []
    for spec in specs:
        inferred = _infer_policy_n_agents(spec)
        if inferred is not None:
            inferred_values.append(int(inferred))

    unique_values = sorted(set(inferred_values))
    if len(unique_values) > 1:
        raise ValueError(f"Policies require different n_agents values: {unique_values}")

    if explicit_n_agents is not None:
        if unique_values and int(explicit_n_agents) != unique_values[0]:
            raise ValueError(
                f"--n-agents={explicit_n_agents} does not match checkpoint n_agents={unique_values[0]}"
            )
        return int(explicit_n_agents)

    if unique_values:
        return int(unique_values[0])

    config_n_agents = config.get("system", {}).get("n_agents")
    if config_n_agents is not None:
        return int(config_n_agents)
    return 1


def _build_env(
    config_path: Path,
    episode_length: int,
    n_agents: int,
    seed: int,
    termination_override: Optional[Dict[str, Any]],
) -> NCS_Env:
    return NCS_Env(
        n_agents=n_agents,
        episode_length=episode_length,
        config_path=str(config_path),
        seed=seed,
        reward_override=EVAL_REWARD_OVERRIDE,
        termination_override=termination_override,
    )


def _load_policy(
    spec: PolicySpec,
    env: Any,
    *,
    seed: int,
    use_multi_agent: bool,
) -> Any:
    policy_type = spec.policy_type.lower()
    if policy_type == "heuristic":
        deterministic = not _is_stochastic_heuristic(spec.policy_path)
        if use_multi_agent:
            return MultiAgentHeuristicPolicy(
                spec.policy_path,
                n_agents=getattr(env, "n_agents", 1),
                seed=seed,
                deterministic=deterministic,
            )
        return get_heuristic_policy(spec.policy_path, n_agents=1, seed=seed)
    if policy_type == "sb3":
        return load_sb3_policy(spec.policy_path, env)
    if policy_type in {"es", "openai_es"}:
        return load_es_policy(spec.policy_path, env)
    if policy_type == "marl_torch":
        return load_marl_torch_multi_agent_policy(spec.policy_path, env)
    raise ValueError(f"Unknown policy type: {spec.policy_type}")


def _run_single_agent_episode(
    env: SingleAgentWrapper,
    policy: Any,
    *,
    seed: int,
    episode_length: int,
    deterministic: bool,
) -> EpisodeResult:
    if hasattr(policy, "reset"):
        policy.reset()
    obs, info = env.reset(seed=seed)

    total_reward = 0.0
    send_count = 0
    steps = 0
    last_info = info
    for _ in range(episode_length):
        action, _ = policy.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        send_count += int(action)
        steps += 1
        last_info = info
        if terminated or truncated:
            break

    states = np.asarray(last_info.get("states", []), dtype=float)
    final_error = float(np.linalg.norm(states[0])) if states.size else 0.0
    send_rate = float(send_count) / float(max(1, steps))
    mean_reward = total_reward / float(max(1, steps))
    return EpisodeResult(
        policy_label="",
        policy_type="",
        seed=int(seed),
        total_reward=total_reward,
        mean_reward=mean_reward,
        final_state_error=final_error,
        send_rate=send_rate,
        steps=steps,
        n_agents=1,
        episode_length=episode_length,
    )


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
    send_count = 0
    steps = 0
    last_info = info
    for _ in range(episode_length):
        action_dict = policy.act(obs_dict)
        obs_dict, rewards, terminated, truncated, info = env.step(action_dict)
        total_reward += float(sum(rewards.values()))
        send_count += int(sum(action_dict.values()))
        steps += 1
        last_info = info
        done = any(bool(terminated[f"agent_{i}"]) or bool(truncated[f"agent_{i}"]) for i in range(n_agents))
        if done:
            break

    states = np.asarray(last_info.get("states", []), dtype=float)
    if states.size:
        final_error = float(np.mean(np.linalg.norm(states, axis=1)))
    else:
        final_error = 0.0
    mean_reward = total_reward / float(max(1, steps * n_agents))
    send_rate = float(send_count) / float(max(1, steps * n_agents))
    return EpisodeResult(
        policy_label="",
        policy_type="",
        seed=int(seed),
        total_reward=total_reward,
        mean_reward=mean_reward,
        final_state_error=final_error,
        send_rate=send_rate,
        steps=steps,
        n_agents=n_agents,
        episode_length=episode_length,
    )


def _summarize_results(results: List[EpisodeResult]) -> Dict[str, float]:
    totals = np.array([r.total_reward for r in results], dtype=float)
    final_errors = np.array([r.final_state_error for r in results], dtype=float)
    send_rates = np.array([r.send_rate for r in results], dtype=float)
    steps = np.array([r.steps for r in results], dtype=float)
    mean_rewards = np.array([r.mean_reward for r in results], dtype=float)
    episode_length = results[0].episode_length if results else 0
    completion_rate = float(np.mean(steps == episode_length)) if results else 0.0
    return {
        "mean_total_reward": float(np.mean(totals)) if results else 0.0,
        "std_total_reward": float(np.std(totals)) if results else 0.0,
        "mean_final_error": float(np.mean(final_errors)) if results else 0.0,
        "std_final_error": float(np.std(final_errors)) if results else 0.0,
        "mean_send_rate": float(np.mean(send_rates)) if results else 0.0,
        "std_send_rate": float(np.std(send_rates)) if results else 0.0,
        "mean_steps": float(np.mean(steps)) if results else 0.0,
        "std_steps": float(np.std(steps)) if results else 0.0,
        "mean_reward_per_step": float(np.mean(mean_rewards)) if results else 0.0,
        "std_reward_per_step": float(np.std(mean_rewards)) if results else 0.0,
        "completion_rate": completion_rate,
    }


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _extract_env_signature(config: Dict[str, Any]) -> Dict[str, Any]:
    signature: Dict[str, Any] = {}
    for key in ("system", "lqr", "network", "observation", "termination"):
        if key in config:
            signature[key] = config.get(key)
    reward_cfg = config.get("reward", {})
    signature["reward"] = {key: reward_cfg.get(key) for key in REWARD_COMPARISON_KEYS if key in reward_cfg}
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
        best_model = entry / "best_model.pt"
        latest_model = entry / "latest_model.pt"
        if not best_model.exists() and not latest_model.exists():
            continue
        runs.append(
            ModelRun(
                name=entry.name,
                path=entry,
                config_path=config_path,
                best_model=best_model if best_model.exists() else None,
                latest_model=latest_model if latest_model.exists() else None,
            )
        )
    return runs


def _infer_policy_type(model_path: Path) -> str:
    suffix = model_path.suffix.lower()
    if suffix == ".pt":
        return "marl_torch"
    if suffix == ".zip":
        return "sb3"
    if suffix == ".npz":
        return "es"
    raise ValueError(f"Unsupported model extension: {model_path}")


def _evaluate_policy(
    spec: PolicySpec,
    *,
    config_path: Path,
    episode_length: int,
    n_agents: int,
    seeds: Sequence[int],
    use_multi_agent: bool,
    termination_override: Optional[Dict[str, Any]],
) -> List[EpisodeResult]:
    results: List[EpisodeResult] = []
    cached_policy: Optional[Any] = None

    for seed in seeds:
        if use_multi_agent:
            env = _build_env(config_path, episode_length, n_agents, seed, termination_override)
        else:
            env = SingleAgentWrapper(
                lambda seed=seed: _build_env(
                    config_path, episode_length, n_agents, seed, termination_override
                )
            )
        try:
            if spec.policy_type.lower() == "heuristic":
                policy = _load_policy(spec, env, seed=seed, use_multi_agent=use_multi_agent)
            else:
                if cached_policy is None:
                    policy = _load_policy(
                        spec, env, seed=seed, use_multi_agent=use_multi_agent
                    )
                    cached_policy = policy
                else:
                    policy = cached_policy

            if use_multi_agent:
                episode = _run_multi_agent_episode(
                    env,
                    policy,
                    seed=seed,
                    episode_length=episode_length,
                )
            else:
                deterministic = not _is_stochastic_heuristic(spec.policy_path) if spec.policy_type == "heuristic" else True
                episode = _run_single_agent_episode(
                    env,
                    policy,
                    seed=seed,
                    episode_length=episode_length,
                    deterministic=deterministic,
                )
            episode.policy_label = spec.label
            episode.policy_type = spec.policy_type
            results.append(episode)
        finally:
            if hasattr(env, "close"):
                env.close()

    return results


def _write_policy_results(
    run_dir: Path,
    spec: PolicySpec,
    results: List[EpisodeResult],
) -> Dict[str, Any]:
    per_seed_rows: List[Dict[str, Any]] = []
    for result in results:
        per_seed_rows.append(
            {
                "policy_label": result.policy_label,
                "policy_type": result.policy_type,
                "seed": result.seed,
                "total_reward": result.total_reward,
                "mean_reward": result.mean_reward,
                "final_state_error": result.final_state_error,
                "send_rate": result.send_rate,
                "steps": result.steps,
                "n_agents": result.n_agents,
                "episode_length": result.episode_length,
            }
        )

    summary = _summarize_results(results)
    summary_row = {
        "policy_label": spec.label,
        "policy_type": spec.policy_type,
        "num_seeds": len(results),
        **summary,
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        run_dir / "per_seed_results.csv",
        [
            "policy_label",
            "policy_type",
            "seed",
            "total_reward",
            "mean_reward",
            "final_state_error",
            "send_rate",
            "steps",
            "n_agents",
            "episode_length",
        ],
        per_seed_rows,
    )
    _write_csv(
        run_dir / "summary_results.csv",
        [
            "policy_label",
            "policy_type",
            "num_seeds",
            "mean_total_reward",
            "std_total_reward",
            "mean_final_error",
            "std_final_error",
            "mean_send_rate",
            "std_send_rate",
            "mean_steps",
            "std_steps",
            "mean_reward_per_step",
            "std_reward_per_step",
            "completion_rate",
        ],
        [summary_row],
    )
    return summary_row


def _write_leaderboard(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "rank",
        "model_name",
        "checkpoint",
        "policy_label",
        "policy_type",
        "num_seeds",
        "mean_total_reward",
        "std_total_reward",
        "mean_final_error",
        "std_final_error",
        "mean_send_rate",
        "std_send_rate",
        "mean_steps",
        "std_steps",
        "mean_reward_per_step",
        "std_reward_per_step",
        "completion_rate",
    ]
    _write_csv(path, fieldnames, rows)


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


def _plot_training_rewards(model_dir: Path, output_path: Path, model_name: str = "Model") -> None:
    """Plot training reward curves from a model directory."""
    training_csv = model_dir / "training_rewards.csv"
    if not training_csv.exists():
        print(f"Warning: Training rewards not found at {training_csv}")
        return

    try:
        train_df = pd.read_csv(str(training_csv))
        if len(train_df) == 0:
            print(f"Warning: Training rewards file is empty at {training_csv}")
            return

        episode_col = _select_column(train_df, ["episode", "Episode", "generation", "Generation"])
        reward_col = _select_column(train_df, ["reward", "reward_sum", "total_reward", "episode_reward", "mean_reward"])
        if episode_col is None or reward_col is None:
            raise ValueError(f"Could not find episode/reward columns. Available: {list(train_df.columns)}")

        episodes = train_df[episode_col].values
        rewards = train_df[reward_col].values
        smoothed = _smooth_rewards(rewards, window_size=200)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(episodes, rewards, alpha=0.3, linewidth=0.5, color="blue", label="Raw")
        ax.plot(episodes, smoothed, linewidth=2, color="blue", label="Smoothed (window=200)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Training Reward")
        ax.set_title(f"{model_name} - Training Rewards")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Saved training rewards plot to {output_path}")
    except Exception as exc:
        print(f"Warning: Could not plot training rewards: {exc}")


def _plot_evaluation_rewards(model_dir: Path, output_path: Path, model_name: str = "Model") -> None:
    """Plot evaluation reward curves from a model directory."""
    eval_csv = model_dir / "evaluation_rewards.csv"
    if not eval_csv.exists():
        print(f"Warning: Evaluation rewards not found at {eval_csv}")
        return

    try:
        eval_df = pd.read_csv(str(eval_csv))
        if len(eval_df) == 0:
            print(f"Warning: Evaluation rewards file is empty at {eval_csv}")
            return

        step_col = _select_column(eval_df, ["step", "Step", "steps", "Steps"])
        mean_reward_col = _select_column(eval_df, ["mean_reward", "reward", "avg_reward"])
        std_reward_col = _select_column(eval_df, ["std_reward", "reward_std", "std"])
        if step_col is None or mean_reward_col is None:
            raise ValueError(f"Could not find step/reward columns. Available: {list(eval_df.columns)}")

        steps = eval_df[step_col].values
        mean_rewards = eval_df[mean_reward_col].values
        std_rewards = eval_df[std_reward_col].values if std_reward_col is not None else None
        smoothed = _smooth_rewards(mean_rewards, window_size=20)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(steps, mean_rewards, alpha=0.3, linewidth=0.5, color="green", label="Raw")
        ax.plot(steps, smoothed, linewidth=2, color="green", label="Smoothed (window=20)")

        if std_rewards is not None:
            smoothed_std = _smooth_rewards(std_rewards, window_size=20)
            ax.fill_between(
                steps,
                smoothed - smoothed_std,
                smoothed + smoothed_std,
                alpha=0.2,
                color="green",
                label="+/- 1 std",
            )

        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Evaluation Reward (mean)")
        ax.set_title(f"{model_name} - Evaluation Rewards")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Saved evaluation rewards plot to {output_path}")
    except Exception as exc:
        print(f"Warning: Could not plot evaluation rewards: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate policies against heuristic baselines.")
    parser.add_argument("--config", help="Path to config JSON")
    parser.add_argument("--models-root", help="Root folder containing model_* subfolders")
    parser.add_argument("--policy", help="Path to policy file or heuristic name")
    parser.add_argument(
        "--policy-type",
        choices=["sb3", "es", "openai_es", "heuristic", "marl_torch"],
        help="Policy type for the target policy",
    )
    parser.add_argument("--policy-label", default=None, help="Label for the target policy")
    parser.add_argument("--episode-length", type=int, default=500, help="Episode length to evaluate")
    parser.add_argument(
        "--n-agents",
        type=int,
        default=None,
        help="Optional override for agent count (default: read from checkpoint or config)",
    )
    parser.add_argument("--num-seeds", type=int, default=30, help="Number of seeds to evaluate")
    parser.add_argument("--seed-start", type=int, default=0, help="First seed value")
    parser.add_argument(
        "--output-dir",
        default="outputs/policy_tests",
        help="Directory to write results",
    )
    args = parser.parse_args()

    if args.models_root:
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

        policy_type = args.policy_type.lower()

        target_label = args.policy_label
        if not target_label:
            target_label = _sanitize_filename(Path(args.policy).name if policy_type != "heuristic" else args.policy)

        termination_override = config.get("termination", {}).get("evaluation")
        if not isinstance(termination_override, dict):
            termination_override = None

        seeds = _iter_seeds(args.seed_start, args.num_seeds)
        output_root = Path(args.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        run_dir = output_root / f"policy_test_{_sanitize_filename(target_label)}"
        run_dir.mkdir(parents=True, exist_ok=True)

        policy_specs = [PolicySpec(label=target_label, policy_type=policy_type, policy_path=args.policy)]
        for heuristic_name in HEURISTIC_POLICY_NAMES:
            policy_specs.append(
                PolicySpec(label=heuristic_name, policy_type="heuristic", policy_path=heuristic_name)
            )

        resolved_n_agents = _resolve_n_agents(config, policy_specs, args.n_agents)
        policy_type_names = [spec.policy_type.lower() for spec in policy_specs]
        multi_agent_types = {"marl_torch", "heuristic"}
        use_multi_agent = all(policy in multi_agent_types for policy in policy_type_names) and (
            any(policy == "marl_torch" for policy in policy_type_names) or resolved_n_agents > 1
        )
        if resolved_n_agents < 1:
            raise ValueError("Resolved n_agents must be >= 1.")

        print(f"Evaluating {len(policy_specs)} policies over {len(seeds)} seeds...")
        print("Raw absolute reward enabled; comm/termination settings from config.")

        per_seed_rows: List[Dict[str, Any]] = []
        summary_rows: List[Dict[str, Any]] = []

        for spec in policy_specs:
            print(f"  -> {spec.label} ({spec.policy_type})")
            results = _evaluate_policy(
                spec,
                config_path=config_path,
                episode_length=int(args.episode_length),
                n_agents=resolved_n_agents,
                seeds=seeds,
                use_multi_agent=use_multi_agent,
                termination_override=termination_override,
            )
            for result in results:
                per_seed_rows.append(
                    {
                        "policy_label": result.policy_label,
                        "policy_type": result.policy_type,
                        "seed": result.seed,
                        "total_reward": result.total_reward,
                        "mean_reward": result.mean_reward,
                        "final_state_error": result.final_state_error,
                        "send_rate": result.send_rate,
                        "steps": result.steps,
                        "n_agents": result.n_agents,
                        "episode_length": result.episode_length,
                    }
                )

            summary_rows.append(
                {
                    "policy_label": spec.label,
                    "policy_type": spec.policy_type,
                    "num_seeds": len(results),
                    **_summarize_results(results),
                }
            )

        _write_csv(
            run_dir / "per_seed_results.csv",
            [
                "policy_label",
                "policy_type",
                "seed",
                "total_reward",
                "mean_reward",
                "final_state_error",
                "send_rate",
                "steps",
                "n_agents",
                "episode_length",
            ],
            per_seed_rows,
        )
        _write_csv(
            run_dir / "summary_results.csv",
            [
                "policy_label",
                "policy_type",
                "num_seeds",
                "mean_total_reward",
                "std_total_reward",
                "mean_final_error",
                "std_final_error",
                "mean_send_rate",
                "std_send_rate",
                "mean_steps",
                "std_steps",
                "mean_reward_per_step",
                "std_reward_per_step",
                "completion_rate",
            ],
            summary_rows,
        )

        print(f"✓ Wrote per-seed results to {run_dir / 'per_seed_results.csv'}")
        print(f"✓ Wrote summary results to {run_dir / 'summary_results.csv'}")

        # Plot training curves if the policy is from a trained model
        if policy_type != "heuristic":
            policy_path = Path(args.policy)
            model_dir = policy_path.parent
            _plot_training_rewards(
                model_dir,
                model_dir / "training_rewards.png",
                model_name=target_label,
            )
            _plot_evaluation_rewards(
                model_dir,
                model_dir / "evaluation_rewards.png",
                model_name=target_label,
            )

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

    model_specs: List[Tuple[str, str, PolicySpec]] = []
    run_lookup = {run.name: run for run in runs}
    policy_types: List[str] = []
    for run in runs:
        for checkpoint_name, model_path in (("best", run.best_model), ("latest", run.latest_model)):
            if model_path is None:
                continue
            policy_type = _infer_policy_type(model_path)
            policy_types.append(policy_type)
            label = f"{run.name}/{checkpoint_name}"
            model_specs.append(
                (run.name, checkpoint_name, PolicySpec(label=label, policy_type=policy_type, policy_path=str(model_path)))
            )

    if not model_specs:
        raise ValueError("No model checkpoints found under the specified root.")

    unique_policy_types = sorted(set(policy_types))
    if "marl_torch" in unique_policy_types and len(unique_policy_types) > 1:
        raise ValueError("Batch mode does not support mixing marl_torch with single-agent policies.")
    use_multi_agent = "marl_torch" in unique_policy_types

    resolved_n_agents = _resolve_n_agents(
        reference_config,
        [spec for _, _, spec in model_specs],
        args.n_agents,
    )
    if resolved_n_agents < 1:
        raise ValueError("Resolved n_agents must be >= 1.")

    seeds = _iter_seeds(args.seed_start, args.num_seeds)
    leaderboard_rows: List[Dict[str, Any]] = []
    plotted_models: Set[str] = set()

    print(f"Evaluating {len(model_specs)} model checkpoints over {len(seeds)} seeds...")
    print("Raw absolute reward enabled; comm/termination settings from config.")

    for model_name, checkpoint, spec in model_specs:
        print(f"  -> {spec.label} ({spec.policy_type})")
        run = run_lookup[model_name]
        results = _evaluate_policy(
            spec,
            config_path=run.config_path,
            episode_length=int(args.episode_length),
            n_agents=resolved_n_agents,
            seeds=seeds,
            use_multi_agent=use_multi_agent,
            termination_override=termination_override,
        )
        run_dir = models_root / model_name / "policy_tests" / f"{checkpoint}_eval"
        summary_row = _write_policy_results(run_dir, spec, results)
        summary_row["model_name"] = model_name
        summary_row["checkpoint"] = checkpoint
        leaderboard_rows.append(summary_row)

        if model_name not in plotted_models:
            _plot_training_rewards(
                run.path,
                run.path / "training_rewards.png",
                model_name=model_name,
            )
            _plot_evaluation_rewards(
                run.path,
                run.path / "evaluation_rewards.png",
                model_name=model_name,
            )
            plotted_models.add(model_name)

    heuristic_config_path = runs[0].config_path
    for heuristic_name in HEURISTIC_POLICY_NAMES:
        spec = PolicySpec(label=heuristic_name, policy_type="heuristic", policy_path=heuristic_name)
        results = _evaluate_policy(
            spec,
            config_path=heuristic_config_path,
            episode_length=int(args.episode_length),
            n_agents=resolved_n_agents,
            seeds=seeds,
            use_multi_agent=use_multi_agent,
            termination_override=termination_override,
        )
        run_dir = models_root / "heuristics" / f"{heuristic_name}_eval"
        summary_row = _write_policy_results(run_dir, spec, results)
        summary_row["model_name"] = "heuristics"
        summary_row["checkpoint"] = "n/a"
        leaderboard_rows.append(summary_row)

    leaderboard_rows.sort(key=lambda row: float(row["mean_total_reward"]), reverse=True)
    ranked_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(leaderboard_rows, start=1):
        ranked = dict(row)
        ranked["rank"] = idx
        ranked_rows.append(ranked)

    leaderboard_path = models_root / "leaderboard.csv"
    _write_leaderboard(leaderboard_path, ranked_rows)
    print(f"✓ Wrote leaderboard to {leaderboard_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
