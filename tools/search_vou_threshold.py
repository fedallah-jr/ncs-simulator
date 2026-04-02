#!/usr/bin/env python3
"""Endpoint-inclusive interval search for a VoU threshold using policy_tester evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ncs_env.config import load_config
from tools.policy_tester import PolicySpec, _evaluate_policy, _summarize_results

SEARCH_REWARD_OVERRIDE: Dict[str, Any] = {
    "state_error_reward": "lqr_cost",
}


def _resolve_n_agents(config: Dict[str, Any]) -> int:
    system_cfg = config.get("system", {})
    raw_n_agents = system_cfg.get("n_agents")
    if raw_n_agents is not None:
        return int(raw_n_agents)
    hetero = system_cfg.get("heterogeneous_plants")
    if isinstance(hetero, list) and hetero:
        return len(hetero)
    raise ValueError("Could not resolve n_agents from config.")


def _threshold_policy_name(threshold: float) -> str:
    return f"value_of_update_{float(threshold):.12g}"


def _dedupe(values: Sequence[float], *, rel_tol: float = 1e-12) -> List[float]:
    unique: List[float] = []
    for value in values:
        if not any(math.isclose(value, seen, rel_tol=rel_tol, abs_tol=rel_tol) for seen in unique):
            unique.append(float(value))
    return unique


def _same_send_rate(rate_a: float, rate_b: float, *, abs_tol: float) -> bool:
    return math.isclose(float(rate_a), float(rate_b), rel_tol=0.0, abs_tol=abs_tol)


def _sample_candidates(
    *,
    lower: float,
    upper: float,
    count: int,
) -> List[float]:
    """Return *count* log-spaced thresholds spanning [lower, upper] (endpoints included)."""
    if count < 2:
        raise ValueError("count must be at least 2 so both interval endpoints are evaluated")
    if math.isclose(lower, upper, rel_tol=1e-12, abs_tol=1e-12):
        return [float(lower)]
    return sorted(_dedupe(np.geomspace(lower, upper, num=count).tolist()))


def _all_send_rates_same(rows: Sequence[Dict[str, Any]], *, send_rate_tol: float) -> bool:
    if len(rows) < 2:
        return True
    ref_send_rate = float(rows[0]["mean_send_rate"])
    return all(
        _same_send_rate(float(row["mean_send_rate"]), ref_send_rate, abs_tol=send_rate_tol)
        for row in rows[1:]
    )


def _select_best_distinct_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    send_rate_tol: float,
) -> Tuple[Dict[str, Any], Dict[str, Any] | None]:
    if not rows:
        raise ValueError("rows must not be empty")
    ordered = sorted(rows, key=lambda row: float(row["mean_total_reward"]), reverse=True)
    best_row = ordered[0]
    best_send_rate = float(best_row["mean_send_rate"])
    second_row = next(
        (
            row
            for row in ordered[1:]
            if not _same_send_rate(float(row["mean_send_rate"]), best_send_rate, abs_tol=send_rate_tol)
        ),
        None,
    )
    return best_row, second_row


def _eval_threshold(
    *,
    threshold: float,
    config_path: Path,
    episode_length: int,
    n_agents: int,
    seeds: Sequence[int],
    num_workers: int,
) -> Dict[str, float]:
    spec = PolicySpec(
        label=f"vou_{float(threshold):.12g}",
        policy_type="heuristic",
        policy_path=_threshold_policy_name(threshold),
    )
    results = _evaluate_policy(
        spec,
        config_path=config_path,
        episode_length=episode_length,
        n_agents=n_agents,
        seeds=seeds,
        termination_override=None,
        reward_override=SEARCH_REWARD_OVERRIDE,
        num_workers=num_workers,
    )
    return _summarize_results(results)


def _eval_candidates(
    candidates: Sequence[float],
    *,
    config_path: Path,
    episode_length: int,
    n_agents: int,
    seeds: Sequence[int],
    num_workers: int,
    cache: Dict[str, Dict[str, float]],
) -> List[Dict[str, float]]:
    """Evaluate *candidates* with caching; parallelize across thresholds."""
    results: Dict[float, Dict[str, float]] = {}
    to_eval: List[float] = []
    for c in candidates:
        key = f"{c:.15g}"
        if key in cache:
            results[c] = cache[key]
        else:
            to_eval.append(c)

    if to_eval:
        eval_kwargs = dict(
            config_path=config_path,
            episode_length=episode_length,
            n_agents=n_agents,
            seeds=seeds,
            num_workers=1,  # avoid nested multiprocessing
        )
        if num_workers <= 1 or len(to_eval) == 1:
            for c in to_eval:
                summary = _eval_threshold(threshold=c, **eval_kwargs)
                cache[f"{c:.15g}"] = summary
                results[c] = summary
        else:
            pool_size = min(num_workers, len(to_eval))
            with ProcessPoolExecutor(max_workers=pool_size) as executor:
                future_to_c = {
                    executor.submit(_eval_threshold, threshold=c, **eval_kwargs): c
                    for c in to_eval
                }
                for future in as_completed(future_to_c):
                    c = future_to_c[future]
                    summary = future.result()
                    cache[f"{c:.15g}"] = summary
                    results[c] = summary

    return [results[c] for c in candidates]


def _collect_scores_for_seed(
    *,
    threshold: float,
    config_path: Path,
    episode_length: int,
    n_agents: int,
    seed: int,
) -> List[float]:
    """Collect VoU scores for a single seed (top-level for pickling)."""
    from ncs_env.env import NCS_Env
    from tools._common import MultiAgentHeuristicPolicy

    env = NCS_Env(
        n_agents=n_agents,
        episode_length=episode_length,
        config_path=str(config_path),
        seed=seed,
        reward_override=dict(SEARCH_REWARD_OVERRIDE),
        track_lqr_cost=True,
    )
    policy_name = _threshold_policy_name(threshold)
    policy = MultiAgentHeuristicPolicy(
        policy_name,
        n_agents=n_agents,
        seed=seed,
        deterministic=True,
        env=env,
    )
    if hasattr(policy, "reset"):
        policy.reset()
    obs_dict, _ = env.reset(seed=seed)

    scores: List[float] = []
    for _ in range(episode_length):
        for agent_idx in range(n_agents):
            obs = obs_dict[f"agent_{agent_idx}"]
            weight_matrix = np.asarray(
                env._get_kf_info_matrix(agent_idx), dtype=np.float64
            )
            gap = obs[2 * env.state_dim : 3 * env.state_dim].astype(np.float64)
            scores.append(float(gap @ weight_matrix @ gap))

        action_dict = policy.act(obs_dict)
        obs_dict, _, terminated, truncated, _ = env.step(action_dict)
        done = any(
            bool(terminated[f"agent_{i}"]) or bool(truncated[f"agent_{i}"])
            for i in range(n_agents)
        )
        if done:
            break

    return scores


def _collect_vou_scores(
    *,
    threshold: float,
    config_path: Path,
    episode_length: int,
    n_agents: int,
    seeds: Sequence[int],
    num_workers: int = 1,
) -> np.ndarray:
    """Run episodes with a VoU threshold policy and collect per-step scores.

    Returns a 1-D array of all per-agent, per-step ``e^T M e`` scores
    collected across the given seeds.  Seeds are evaluated in parallel when
    *num_workers* > 1.
    """
    seed_kwargs = dict(
        threshold=threshold,
        config_path=config_path,
        episode_length=episode_length,
        n_agents=n_agents,
    )
    if num_workers <= 1 or len(seeds) <= 1:
        all_scores: List[float] = []
        for seed in seeds:
            all_scores.extend(
                _collect_scores_for_seed(seed=int(seed), **seed_kwargs)
            )
    else:
        pool_size = min(num_workers, len(seeds))
        all_scores = []
        with ProcessPoolExecutor(max_workers=pool_size) as executor:
            futures = [
                executor.submit(_collect_scores_for_seed, seed=int(s), **seed_kwargs)
                for s in seeds
            ]
            for future in futures:  # preserve seed order
                all_scores.extend(future.result())

    return np.asarray(all_scores, dtype=np.float64)


def _compute_quantile_edges(scores: np.ndarray, *, bits: int) -> np.ndarray:
    """Compute equal-probability bin edges for *bits*-bit encoding.

    For *bits* bits there are ``2^bits`` levels and ``2^bits - 1`` edges,
    placed at the ``1/2^bits, 2/2^bits, …, (2^bits-1)/2^bits`` quantiles
    of *scores*.
    """
    n_levels = 1 << bits
    quantiles = np.linspace(0.0, 1.0, n_levels + 1)[1:-1]  # interior edges
    return np.quantile(scores, quantiles)


def _default_output_dir(config_path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / f"search_vou_threshold_{config_path.stem}_{stamp}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Search a VoU threshold by repeated policy_tester evaluation.")
    parser.add_argument("--config", type=Path, default=Path("configs/marl_absolute_plants_hetero.json"))
    parser.add_argument("--episode-length", type=int, default=250)
    parser.add_argument("--eval-seed-start", type=int, default=100)
    parser.add_argument("--eval-num-seeds", type=int, default=32)
    parser.add_argument("--threshold-min", type=float, default=0.01)
    parser.add_argument("--threshold-max", type=float, default=1e1)
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument(
        "--samples-per-iter",
        type=int,
        default=8,
        help="Total thresholds evaluated per iteration, including the lower and upper bounds.",
    )
    parser.add_argument(
        "--send-rate-tol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for treating two mean send rates as identical.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of thresholds (or seeds for score collection) to evaluate in parallel.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--collect-bits",
        type=int,
        nargs="+",
        default=None,
        help="After search, collect VoU score distribution under the best threshold "
        "and compute equal-probability quantile edges for each given bit width "
        "(e.g. --collect-bits 2 4).",
    )
    args = parser.parse_args()

    if args.threshold_min <= 0 or args.threshold_max <= 0:
        raise ValueError("threshold bounds must be positive")
    if args.threshold_min >= args.threshold_max:
        raise ValueError("threshold_min must be < threshold_max")
    if args.eval_num_seeds <= 0:
        raise ValueError("eval_num_seeds must be positive")
    if args.iters <= 0:
        raise ValueError("iters must be positive")
    if args.samples_per_iter < 2:
        raise ValueError("samples_per_iter must be >= 2")
    if args.send_rate_tol < 0:
        raise ValueError("send_rate_tol must be >= 0")

    config_path = args.config.resolve()
    config = load_config(str(config_path))
    n_agents = _resolve_n_agents(config)
    seeds = list(range(int(args.eval_seed_start), int(args.eval_seed_start) + int(args.eval_num_seeds)))
    lower = float(args.threshold_min)
    upper = float(args.threshold_max)
    initial_lower = float(lower)
    initial_upper = float(upper)
    best_result: Dict[str, Any] | None = None
    history: List[Dict[str, Any]] = []
    stop_reason = "max_iters_reached"
    eval_cache: Dict[str, Dict[str, float]] = {}

    print("VoU threshold search")
    print(f"  config: {config_path}")
    print(f"  eval seeds: {seeds[0]}..{seeds[-1]}")
    print(f"  initial range: [{lower:.12g}, {upper:.12g}]")
    print(f"  samples per iter: {int(args.samples_per_iter)} (including endpoints)")
    print(f"  num workers: {int(args.num_workers)}")

    for iteration in range(int(args.iters)):
        candidates = _sample_candidates(
            lower=lower,
            upper=upper,
            count=int(args.samples_per_iter),
        )
        summaries = _eval_candidates(
            candidates,
            config_path=config_path,
            episode_length=int(args.episode_length),
            n_agents=n_agents,
            seeds=seeds,
            num_workers=int(args.num_workers),
            cache=eval_cache,
        )
        rows: List[Dict[str, Any]] = []
        for sample_index, (candidate, summary) in enumerate(zip(candidates, summaries)):
            row = {
                "iteration": iteration,
                "sample_index": sample_index,
                "range_lower": float(lower),
                "range_upper": float(upper),
                "theta": float(candidate),
                "selected_rank": 0,
                "selected_for_next_interval": 0,
                **summary,
            }
            rows.append(row)

        best_row, second_row = _select_best_distinct_rows(rows, send_rate_tol=float(args.send_rate_tol))
        best_row["selected_rank"] = 1

        all_send_rates_same = _all_send_rates_same(rows, send_rate_tol=float(args.send_rate_tol))
        if all_send_rates_same:
            best_row["selected_for_next_interval"] = 1
            if best_result is None or float(best_row["mean_total_reward"]) > float(best_result["mean_total_reward"]):
                best_result = dict(best_row)
            history.extend(rows)
            print(
                f"  iter={iteration:02d} "
                f"best_theta={float(best_row['theta']):.12g} "
                f"reward={float(best_row['mean_total_reward']):.6f} "
                f"send_rate={float(best_row['mean_send_rate']):.6f} "
                f"range=[{lower:.12g}, {upper:.12g}] "
                f"status=all_send_rates_equal"
            )
            stop_reason = "all_sampled_send_rates_equal"
            break

        if second_row is None:
            best_row["selected_for_next_interval"] = 1
            if best_result is None or float(best_row["mean_total_reward"]) > float(best_result["mean_total_reward"]):
                best_result = dict(best_row)
            history.extend(rows)
            print(
                f"  iter={iteration:02d} "
                f"best_theta={float(best_row['theta']):.12g} "
                f"reward={float(best_row['mean_total_reward']):.6f} "
                f"send_rate={float(best_row['mean_send_rate']):.6f} "
                f"range=[{lower:.12g}, {upper:.12g}] "
                f"status=no_distinct_send_rate_match"
            )
            stop_reason = "no_distinct_send_rate_found"
            break

        second_row["selected_rank"] = 2
        best_row["selected_for_next_interval"] = 1
        second_row["selected_for_next_interval"] = 1
        if best_result is None or float(best_row["mean_total_reward"]) > float(best_result["mean_total_reward"]):
            best_result = dict(best_row)
        history.extend(rows)

        next_lower = min(float(best_row["theta"]), float(second_row["theta"]))
        next_upper = max(float(best_row["theta"]), float(second_row["theta"]))
        print(
            f"  iter={iteration:02d} "
            f"best_theta={float(best_row['theta']):.12g} "
            f"reward={float(best_row['mean_total_reward']):.6f} "
            f"send_rate={float(best_row['mean_send_rate']):.6f} "
            f"second_theta={float(second_row['theta']):.12g} "
            f"second_reward={float(second_row['mean_total_reward']):.6f} "
            f"second_send_rate={float(second_row['mean_send_rate']):.6f} "
            f"next_range=[{next_lower:.12g}, {next_upper:.12g}]"
        )
        lower = next_lower
        upper = next_upper

    output_dir = args.output_dir.resolve() if args.output_dir is not None else _default_output_dir(config_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    history_path = output_dir / "search_history.csv"
    with history_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    summary_path = output_dir / "search_summary.json"
    with summary_path.open("w") as handle:
        json.dump(
            {
                "config_path": str(config_path),
                "episode_length": int(args.episode_length),
                "eval_seeds": seeds,
                "initial_lower": initial_lower,
                "initial_upper": initial_upper,
                "final_lower": lower,
                "final_upper": upper,
                "samples_per_iter": int(args.samples_per_iter),
                "send_rate_tol": float(args.send_rate_tol),
                "stop_reason": stop_reason,
                "num_iterations_completed": len({row["iteration"] for row in history}),
                "best_result": best_result,
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    if best_result is not None:
        print(f"Best threshold: {float(best_result['theta']):.12g}")
        print(
            f"  mean_total_reward={float(best_result['mean_total_reward']):.6f}, "
            f"mean_send_rate={float(best_result['mean_send_rate']):.6f}"
        )
    print(f"Final range: [{lower:.12g}, {upper:.12g}]")
    print(f"Stop reason: {stop_reason}")
    print(f"Artifacts: {history_path}, {summary_path}")

    # ------------------------------------------------------------------
    # Score collection & quantile edge computation
    # ------------------------------------------------------------------
    if args.collect_bits and best_result is not None:
        best_theta = float(best_result["theta"])
        print(f"\nCollecting VoU score distribution under threshold={best_theta:.12g} ...")
        scores = _collect_vou_scores(
            threshold=best_theta,
            config_path=config_path,
            episode_length=int(args.episode_length),
            n_agents=n_agents,
            seeds=seeds,
            num_workers=int(args.num_workers),
        )
        scores_path = output_dir / "vou_scores.npy"
        np.save(str(scores_path), scores)
        print(f"  collected {len(scores)} scores -> {scores_path}")
        print(
            f"  min={float(np.min(scores)):.6g}  median={float(np.median(scores)):.6g}"
            f"  mean={float(np.mean(scores)):.6g}  max={float(np.max(scores)):.6g}"
        )

        edges_info: Dict[str, Any] = {}
        for bits in args.collect_bits:
            if bits <= 0:
                print(f"  skipping bits={bits} (must be positive)")
                continue
            edges = _compute_quantile_edges(scores, bits=bits)
            key = f"bits_{bits}"
            edges_info[key] = {
                "bits": bits,
                "n_levels": 1 << bits,
                "edges": [float(e) for e in edges],
            }
            print(f"  bits={bits}: {1 << bits} levels, edges={[f'{e:.6g}' for e in edges]}")

        edges_path = output_dir / "quantile_edges.json"
        with edges_path.open("w") as handle:
            json.dump(
                {
                    "threshold": best_theta,
                    "n_scores": len(scores),
                    "score_stats": {
                        "min": float(np.min(scores)),
                        "max": float(np.max(scores)),
                        "mean": float(np.mean(scores)),
                        "median": float(np.median(scores)),
                        "std": float(np.std(scores)),
                    },
                    "edges": edges_info,
                },
                handle,
                indent=2,
            )
        print(f"  quantile edges -> {edges_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
