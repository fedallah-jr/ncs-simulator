"""
Lightweight hyperparameter search helper for PPO and DQN runs on the NCS env.

Usage (examples):
    python -m tools.hparam_search --algorithm ppo --config configs/perfect_comm.json --samples 3
    python -m tools.hparam_search --algorithm dqn --config configs/perfect_comm.json --samples 5

This script performs a simple random search over small, built-in hyperparameter
grids and launches training runs inline (no subprocesses). It reuses the SB3
training logic used in `algorithms/ppo_single.py` and `algorithms/deep_q_learning.py`
so output directories and saved configs stay consistent. Each sampled run now
also records training rewards, evaluation summaries (best/mean/worst/stability),
and a run-level search_summary.json to make downstream comparison easier. Use
`--repeats` to run each hyperparameter sample multiple times, compute consistency
statistics (mean/std/min/max across repeats), and emit both individual and
aggregate leaderboards in the final search results file.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ncs_env.env import NCS_Env
from utils import SingleAgentWrapper
from utils.run_utils import prepare_run_directory, save_config_with_hyperparameters


def _to_py_scalar(x):
    return x.item() if hasattr(x, "item") else x


def _extract_training_rewards(vec_env: gym.Env) -> List[float]:
    """Pull per-episode rewards from the Monitor wrapper, handling VecNormalize."""
    monitor_env = vec_env.venv if isinstance(vec_env, VecNormalize) else vec_env
    rewards: List[float] = []
    if hasattr(monitor_env, "envs") and monitor_env.envs:
        env = monitor_env.envs[0]
        if hasattr(env, "get_episode_rewards"):
            rewards = list(env.get_episode_rewards())
    return rewards


def _write_training_rewards_csv(rewards: List[float], output_path: Path) -> None:
    """Persist episode rewards to CSV for later inspection."""
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])
        for idx, rew in enumerate(rewards, start=1):
            writer.writerow([idx, rew])


def summarize_training_rewards(rewards: List[float]) -> Dict[str, float]:
    """Compute basic statistics over training episode rewards."""
    stats = {
        "mean_reward": float("nan"),
        "std_reward": float("nan"),
        "min_reward": float("nan"),
        "max_reward": float("nan"),
        "final_reward": float("nan"),
        "trend_slope": float("nan"),  # linear fit slope over episodes
        "start_end_lift": float("nan"),  # reward_last - reward_first
        "frac_positive_steps": float("nan"),  # share of episodes that improved vs previous
        "num_episodes": 0,
    }
    if not rewards:
        return stats

    arr = np.array(rewards, dtype=float)
    stats.update(
        {
            "mean_reward": float(np.mean(arr)),
            "std_reward": float(np.std(arr)),
            "min_reward": float(np.min(arr)),
            "max_reward": float(np.max(arr)),
            "final_reward": float(arr[-1]),
            "trend_slope": float(np.polyfit(np.arange(arr.shape[0]), arr, 1)[0]) if arr.shape[0] > 1 else float("nan"),
            "start_end_lift": float(arr[-1] - arr[0]) if arr.shape[0] > 0 else float("nan"),
            "frac_positive_steps": float(np.mean(arr[1:] > arr[:-1])) if arr.shape[0] > 1 else float("nan"),
            "num_episodes": int(arr.shape[0]),
        }
    )
    return stats


def summarize_evaluations(eval_path: Path) -> Dict[str, float]:
    """Compute evaluation metrics from the EvalCallback NPZ outputs."""
    metrics = {
        "best_eval_mean": float("nan"),
        "final_eval_mean": float("nan"),
        "mean_eval_mean": float("nan"),
        "worst_eval_mean": float("nan"),
        "eval_stability": float("nan"),
        "num_evaluations": 0,
    }
    if not eval_path.exists():
        return metrics

    try:
        npz = np.load(eval_path)
        results = npz.get("results")
        if results is None or results.size == 0:
            return metrics

        mean_rewards = results.mean(axis=1)
        metrics.update(
            {
                "best_eval_mean": float(np.max(mean_rewards)),
                "final_eval_mean": float(mean_rewards[-1]),
                "mean_eval_mean": float(np.mean(mean_rewards)),
                "worst_eval_mean": float(np.min(mean_rewards)),
                "eval_stability": float(np.std(mean_rewards)),
                "num_evaluations": int(mean_rewards.shape[0]),
            }
        )
    except Exception:
        # Keep NaNs if loading fails
        pass
    return metrics


def _sort_table_value(value: float, higher_is_better: bool) -> Tuple[int, float]:
    """Ensure NaNs sort last while respecting metric direction."""
    is_nan = isinstance(value, float) and math.isnan(value)
    direction_value = 0.0 if is_nan else (-value if higher_is_better else value)
    return (1 if is_nan else 0, direction_value)


def _aggregate_metric_values(values: List[float]) -> Dict[str, float]:
    """Compute mean/std/min/max over a list, skipping NaNs."""
    clean = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    if not clean:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "count": 0,
        }
    arr = np.array(clean, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "count": int(arr.shape[0]),
    }


def _sort_key_with_nan(value: float, higher_is_better: bool) -> Tuple[int, float]:
    """Helper for multi-metric ranking to push NaNs to the end."""
    is_nan = isinstance(value, float) and math.isnan(value)
    direction_value = 0.0 if is_nan else (-value if higher_is_better else value)
    return (1 if is_nan else 0, direction_value)


def build_comparison_table(
    summaries: List[Dict],
    metric_name: str,
    getter: Callable[[Dict[str, object]], float],
    higher_is_better: bool = True,
    label_getter: Callable[[Dict[str, object]], object] = lambda s: s["run_dir"],
    label_key: str = "run_dir",
) -> Dict[str, List[Dict]]:
    """Create a sorted comparison table for a metric across runs."""
    table = []
    for s in summaries:
        value = getter(s)
        table.append(
            {
                label_key: label_getter(s),
                "algorithm": s["algorithm"],
                "value": _to_py_scalar(value),
                "hyperparameters": s["hyperparameters"],
            }
        )

    table.sort(key=lambda row: _sort_table_value(row["value"], higher_is_better))
    return {metric_name: table}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random-search hyperparameters for PPO or DQN on NCS.")
    parser.add_argument("--algorithm", choices=["ppo", "dqn"], required=True, help="SB3 algorithm to train.")
    parser.add_argument("--config", type=Path, required=True, help="Path to env config JSON.")
    parser.add_argument("--samples", type=int, default=15, help="Number of random hyperparameter samples.")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeats per hyperparameter sample.")
    parser.add_argument("--total-timesteps", type=int, default=200_000, help="Training timesteps per trial.")
    parser.add_argument("--episode-length", type=int, default=1000, help="Episode length.")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed for sampling hyperparams.")
    parser.add_argument("--normalize-reward", action="store_true", help="Enable VecNormalize for rewards/obs.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"), help="Root for run artifacts.")
    return parser.parse_args()


def sample_hparams(algo: str, rng: np.random.Generator) -> Dict[str, float]:
    """Sample a hyperparameter dict from small predefined grids."""
    if algo == "ppo":
        grid: Dict[str, List] = {
            "learning_rate": [3e-4, 1e-4, 5e-4],
            "gamma": [0.95, 0.99],
            "batch_size": [64, 128, 256],
            "n_steps": [512, 1024, 2048],
            "ent_coef": [0.0, 0.005, 0.01],
        }
    else:  # dqn
        grid = {
            "learning_rate": [1e-3, 5e-4, 2e-4],
            "gamma": [0.95, 0.99],
            "batch_size": [64, 128, 256],
            "buffer_size": [50_000, 100_000, 200_000],
            "train_freq": [4, 8],
            "target_update_interval": [1_000, 5_000, 10_000],
        }
    return {k: rng.choice(v) for k, v in grid.items()}


def build_envs(config_path: str, episode_length: int, seed: int, normalize_reward: bool) -> Tuple[gym.Env, gym.Env]:
    """Construct SB3 train/eval envs with consistent wrappers."""

    def env_factory():
        def make_env():
            return NCS_Env(
                n_agents=1,
                episode_length=episode_length,
                config_path=config_path,
                seed=seed,
            )

        return SingleAgentWrapper(make_env)

    train_env: gym.Env = DummyVecEnv([lambda: Monitor(env_factory())])
    eval_env: gym.Env = DummyVecEnv([lambda: Monitor(env_factory())])

    if normalize_reward:
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)
        eval_env.training = False

    return train_env, eval_env


def aggregate_repeat_group(
    algo: str, hparams: Dict[str, float], repeats: List[Dict[str, object]]
) -> Dict[str, object]:
    """Aggregate metrics across repeated runs for the same hyperparameters."""

    def collect(metric_group: str, metric_name: str) -> Dict[str, float]:
        values = [r[metric_group].get(metric_name, float("nan")) for r in repeats]
        return _aggregate_metric_values(values)

    aggregate_eval = {
        "best_eval_mean": collect("eval_metrics", "best_eval_mean"),
        "mean_eval_mean": collect("eval_metrics", "mean_eval_mean"),
        "worst_eval_mean": collect("eval_metrics", "worst_eval_mean"),
        "final_eval_mean": collect("eval_metrics", "final_eval_mean"),
        "eval_stability": collect("eval_metrics", "eval_stability"),
    }
    aggregate_training = {
        "mean_reward": collect("training_metrics", "mean_reward"),
        "final_reward": collect("training_metrics", "final_reward"),
        "std_reward": collect("training_metrics", "std_reward"),
        "max_reward": collect("training_metrics", "max_reward"),
        "min_reward": collect("training_metrics", "min_reward"),
        "trend_slope": collect("training_metrics", "trend_slope"),
        "start_end_lift": collect("training_metrics", "start_end_lift"),
        "frac_positive_steps": collect("training_metrics", "frac_positive_steps"),
    }

    return {
        "algorithm": algo,
        "hyperparameters": {k: _to_py_scalar(v) for k, v in hparams.items()},
        "aggregate_eval": aggregate_eval,
        "aggregate_training": aggregate_training,
        "runs": [r["run_dir"] for r in repeats],
        "seeds": [r["training_setup"]["seed"] for r in repeats],
    }


def train_once(
    algo: str,
    config_path: Path,
    hparams: Dict[str, float],
    args: argparse.Namespace,
    run_seed: int,
    repeat_index: int,
) -> Dict[str, object]:
    """Train a single run with sampled hyperparameters."""
    run_dir = prepare_run_directory(algo, config_path, args.output_root)
    cfg_path_str = str(config_path)

    train_env, eval_env = build_envs(cfg_path_str, args.episode_length, run_seed, args.normalize_reward)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir),
        log_path=str(run_dir),
        eval_freq=max(1, args.total_timesteps // 20),
        n_eval_episodes=5,
        deterministic=True,
    )

    if algo == "ppo":
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=hparams["learning_rate"],
            gamma=hparams["gamma"],
            n_steps=int(hparams["n_steps"]),
            batch_size=int(hparams["batch_size"]),
            ent_coef=hparams["ent_coef"],
            seed=run_seed,
            verbose=1,
        )
    else:
        model = DQN(
            "MlpPolicy",
            train_env,
            learning_rate=hparams["learning_rate"],
            gamma=hparams["gamma"],
            batch_size=int(hparams["batch_size"]),
            buffer_size=int(hparams["buffer_size"]),
            train_freq=int(hparams["train_freq"]),
            target_update_interval=int(hparams["target_update_interval"]),
            seed=run_seed,
            verbose=1,
        )

    model.learn(total_timesteps=args.total_timesteps, callback=eval_callback)
    model.save(str(run_dir / "latest_model"))

    training_rewards = _extract_training_rewards(train_env)
    training_rewards_path = run_dir / "training_rewards.csv"
    _write_training_rewards_csv(training_rewards, training_rewards_path)
    training_metrics = summarize_training_rewards(training_rewards)

    eval_path = Path(run_dir) / "evaluations.npz"
    eval_metrics = summarize_evaluations(eval_path)

    # Persist config + hyperparams
    hp_record = {k: _to_py_scalar(v) for k, v in hparams.items()}
    hp_record.update(
        {
            "total_timesteps": args.total_timesteps,
            "episode_length": args.episode_length,
            "normalize_reward": args.normalize_reward,
            "seed": run_seed,
            "algorithm": algo,
            "search_seed": args.seed,
        }
    )
    save_config_with_hyperparameters(run_dir, config_path, algo, hp_record)

    summary = {
        "run_dir": str(run_dir),
        "algorithm": algo,
        "hyperparameters": {k: _to_py_scalar(v) for k, v in hparams.items()},
        "training_setup": {
            "total_timesteps": args.total_timesteps,
            "episode_length": args.episode_length,
            "normalize_reward": args.normalize_reward,
            "seed": run_seed,
        },
        "repeat_index": repeat_index,
        "eval_metrics": {k: _to_py_scalar(v) for k, v in eval_metrics.items()},
        "training_metrics": {k: _to_py_scalar(v) for k, v in training_metrics.items()},
        "artifacts": {
            "config": str(run_dir / "config.json"),
            "latest_model": str(run_dir / "latest_model.zip"),
            "best_model": str(run_dir / "best_model.zip"),
            "evaluations": str(eval_path),
            "training_rewards": str(training_rewards_path),
        },
    }

    with (run_dir / "search_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    train_env.close()
    eval_env.close()
    return summary


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    config_path = args.config.resolve()

    print(f"Running random search for {args.algorithm} | samples={args.samples} | config={config_path}")
    global_summaries = []  # individual runs
    aggregate_summaries = []  # per-hparam repeat aggregates
    for idx in range(args.samples):
        hparams = sample_hparams(args.algorithm, rng)
        print(f"\n[{idx + 1}/{args.samples}] Hyperparameters: {hparams}")
        repeat_runs = []
        for rep in range(args.repeats):
            run_seed = args.seed + rep
            print(f"  Repeat {rep + 1}/{args.repeats} (seed={run_seed})")
            summary = train_once(
                args.algorithm,
                config_path,
                hparams,
                args,
                run_seed=run_seed,
                repeat_index=rep,
            )
            repeat_runs.append(summary)
            global_summaries.append(summary)

            eval_metrics = summary.get("eval_metrics", {})
            best_mean = eval_metrics.get("best_eval_mean", float("nan"))
            mean_eval = eval_metrics.get("mean_eval_mean", float("nan"))
            worst_eval = eval_metrics.get("worst_eval_mean", float("nan"))
            stability = eval_metrics.get("eval_stability", float("nan"))
            print(
                "    Completed run -> {run_dir} | best_eval_mean={best:.2f} | mean_eval_mean={mean:.2f} "
                "| worst_eval_mean={worst:.2f} | eval_stability={stability:.3f}".format(
                    run_dir=summary["run_dir"],
                    best=best_mean,
                    mean=mean_eval,
                    worst=worst_eval,
                    stability=stability,
                )
            )

        group_summary = aggregate_repeat_group(args.algorithm, hparams, repeat_runs)
        aggregate_summaries.append(group_summary)

        agg_best_mean = group_summary["aggregate_eval"]["best_eval_mean"]["mean"]
        agg_stability = group_summary["aggregate_eval"]["eval_stability"]["mean"]
        print(
            "  Aggregate over repeats -> best_eval_mean(mean)={best:.2f} | eval_stability(mean std)={stability:.3f}".format(
                best=agg_best_mean,
                stability=agg_stability,
            )
        )

    comparison_tables_individual = {}
    comparison_tables_individual.update(
        build_comparison_table(global_summaries, "best_eval_mean", lambda s: s["eval_metrics"]["best_eval_mean"])
    )
    comparison_tables_individual.update(
        build_comparison_table(global_summaries, "mean_eval_mean", lambda s: s["eval_metrics"]["mean_eval_mean"])
    )
    comparison_tables_individual.update(
        build_comparison_table(
            global_summaries, "eval_stability", lambda s: s["eval_metrics"]["eval_stability"], higher_is_better=False
        )
    )
    comparison_tables_individual.update(
        build_comparison_table(global_summaries, "worst_eval_mean", lambda s: s["eval_metrics"]["worst_eval_mean"])
    )
    comparison_tables_individual.update(
        build_comparison_table(global_summaries, "final_eval_mean", lambda s: s["eval_metrics"]["final_eval_mean"])
    )
    comparison_tables_individual.update(
        build_comparison_table(
            global_summaries, "training_reward_mean", lambda s: s["training_metrics"]["mean_reward"]
        )
    )
    comparison_tables_individual.update(
        build_comparison_table(
            global_summaries, "training_reward_final", lambda s: s["training_metrics"]["final_reward"]
        )
    )
    comparison_tables_individual.update(
        build_comparison_table(
            global_summaries, "training_trend_slope", lambda s: s["training_metrics"]["trend_slope"]
        )
    )
    comparison_tables_individual.update(
        build_comparison_table(
            global_summaries, "training_start_end_lift", lambda s: s["training_metrics"]["start_end_lift"]
        )
    )
    comparison_tables_individual.update(
        build_comparison_table(
            global_summaries,
            "training_frac_positive_steps",
            lambda s: s["training_metrics"]["frac_positive_steps"],
        )
    )

    comparison_tables_aggregate = {}
    comparison_tables_aggregate.update(
        build_comparison_table(
            aggregate_summaries,
            "best_eval_mean_mean",
            lambda s: s["aggregate_eval"]["best_eval_mean"]["mean"],
            label_getter=lambda s: s["runs"],
            label_key="runs",
        )
    )
    comparison_tables_aggregate.update(
        build_comparison_table(
            aggregate_summaries,
            "mean_eval_mean_mean",
            lambda s: s["aggregate_eval"]["mean_eval_mean"]["mean"],
            label_getter=lambda s: s["runs"],
            label_key="runs",
        )
    )
    comparison_tables_aggregate.update(
        build_comparison_table(
            aggregate_summaries,
            "eval_stability_mean",
            lambda s: s["aggregate_eval"]["eval_stability"]["mean"],
            higher_is_better=False,
            label_getter=lambda s: s["runs"],
            label_key="runs",
        )
    )
    comparison_tables_aggregate.update(
        build_comparison_table(
            aggregate_summaries,
            "worst_eval_mean_mean",
            lambda s: s["aggregate_eval"]["worst_eval_mean"]["mean"],
            label_getter=lambda s: s["runs"],
            label_key="runs",
        )
    )
    comparison_tables_aggregate.update(
        build_comparison_table(
            aggregate_summaries,
            "final_eval_mean_mean",
            lambda s: s["aggregate_eval"]["final_eval_mean"]["mean"],
            label_getter=lambda s: s["runs"],
            label_key="runs",
        )
    )
    comparison_tables_aggregate.update(
        build_comparison_table(
            aggregate_summaries,
            "training_reward_mean_mean",
            lambda s: s["aggregate_training"]["mean_reward"]["mean"],
            label_getter=lambda s: s["runs"],
            label_key="runs",
        )
    )
    comparison_tables_aggregate.update(
        build_comparison_table(
            aggregate_summaries,
            "training_reward_final_mean",
            lambda s: s["aggregate_training"]["final_reward"]["mean"],
            label_getter=lambda s: s["runs"],
            label_key="runs",
        )
    )
    comparison_tables_aggregate.update(
        build_comparison_table(
            aggregate_summaries,
            "training_trend_slope_mean",
            lambda s: s["aggregate_training"]["trend_slope"]["mean"],
            label_getter=lambda s: s["runs"],
            label_key="runs",
        )
    )
    comparison_tables_aggregate.update(
        build_comparison_table(
            aggregate_summaries,
            "training_start_end_lift_mean",
            lambda s: s["aggregate_training"]["start_end_lift"]["mean"],
            label_getter=lambda s: s["runs"],
            label_key="runs",
        )
    )
    comparison_tables_aggregate.update(
        build_comparison_table(
            aggregate_summaries,
            "training_frac_positive_steps_mean",
            lambda s: s["aggregate_training"]["frac_positive_steps"]["mean"],
            label_getter=lambda s: s["runs"],
            label_key="runs",
        )
    )

    search_payload = {
        "algorithm": args.algorithm,
        "config_path": str(config_path),
        "samples": args.samples,
        "repeats": args.repeats,
        "runs": global_summaries,
        "aggregates": aggregate_summaries,
        "comparison_tables": {
            "individual_runs": comparison_tables_individual,
            "aggregate_means": comparison_tables_aggregate,
        },
    }

    # Save global comparison table
    search_table_path = args.output_root / f"search_results_{args.algorithm}.json"
    with search_table_path.open("w", encoding="utf-8") as f:
        json.dump(search_payload, f, indent=2)
        f.write("\n")
    print(f"\nSaved search comparison to {search_table_path}")

    # Build top-3 rankings for performance and stability using aggregate metrics
    def perf_key(entry: Dict[str, object]) -> Tuple:
        agg_eval = entry["aggregate_eval"]
        return (
            _sort_key_with_nan(agg_eval["best_eval_mean"]["mean"], higher_is_better=True),
            _sort_key_with_nan(agg_eval["mean_eval_mean"]["mean"], higher_is_better=True),
            _sort_key_with_nan(agg_eval["worst_eval_mean"]["mean"], higher_is_better=True),
            _sort_key_with_nan(agg_eval["final_eval_mean"]["mean"], higher_is_better=True),
        )

    def stability_key(entry: Dict[str, object]) -> Tuple:
        agg_eval = entry["aggregate_eval"]
        agg_train = entry["aggregate_training"]
        return (
            _sort_key_with_nan(agg_eval["eval_stability"]["mean"], higher_is_better=False),
            _sort_key_with_nan(agg_train["std_reward"]["mean"], higher_is_better=False),
            _sort_key_with_nan(agg_train["trend_slope"]["mean"], higher_is_better=True),
            _sort_key_with_nan(agg_train["start_end_lift"]["mean"], higher_is_better=True),
            _sort_key_with_nan(agg_train["frac_positive_steps"]["mean"], higher_is_better=True),
        )

    # Sort with custom keys
    perf_sorted = sorted(aggregate_summaries, key=lambda e: perf_key(e))[:3]
    stability_sorted = sorted(aggregate_summaries, key=lambda e: stability_key(e))[:3]

    def build_perf_entry(rank: int, entry: Dict[str, object]) -> Dict[str, object]:
        agg_eval = entry["aggregate_eval"]
        return {
            "rank": rank,
            "runs": entry["runs"],
            "hyperparameters": entry["hyperparameters"],
            "best_eval_mean_mean": _to_py_scalar(agg_eval["best_eval_mean"]["mean"]),
            "tie_breakers": {
                "mean_eval_mean_mean": _to_py_scalar(agg_eval["mean_eval_mean"]["mean"]),
                "worst_eval_mean_mean": _to_py_scalar(agg_eval["worst_eval_mean"]["mean"]),
                "final_eval_mean_mean": _to_py_scalar(agg_eval["final_eval_mean"]["mean"]),
            },
        }

    def build_stability_entry(rank: int, entry: Dict[str, object]) -> Dict[str, object]:
        agg_eval = entry["aggregate_eval"]
        agg_train = entry["aggregate_training"]
        return {
            "rank": rank,
            "runs": entry["runs"],
            "hyperparameters": entry["hyperparameters"],
            "eval_stability_mean": _to_py_scalar(agg_eval["eval_stability"]["mean"]),
            "tie_breakers": {
                "std_reward_mean": _to_py_scalar(agg_train["std_reward"]["mean"]),
                "trend_slope_mean": _to_py_scalar(agg_train["trend_slope"]["mean"]),
                "start_end_lift_mean": _to_py_scalar(agg_train["start_end_lift"]["mean"]),
                "frac_positive_steps_mean": _to_py_scalar(agg_train["frac_positive_steps"]["mean"]),
            },
        }

    top3_payload = {
        "algorithm": args.algorithm,
        "config_path": str(config_path),
        "performance_top3": [build_perf_entry(i + 1, e) for i, e in enumerate(perf_sorted)],
        "stability_top3": [build_stability_entry(i + 1, e) for i, e in enumerate(stability_sorted)],
    }

    top3_path = args.output_root / f"top3_{args.algorithm}.json"
    with top3_path.open("w", encoding="utf-8") as f:
        json.dump(top3_payload, f, indent=2)
        f.write("\n")
    print(f"Saved top-3 performance/stability summary to {top3_path}")


if __name__ == "__main__":
    main()
