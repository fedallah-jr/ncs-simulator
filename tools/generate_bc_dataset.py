from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ncs_env.config import load_config
from ncs_env.env import NCS_Env
from tools.heuristic_policies import get_heuristic_policy
from utils.marl import stack_obs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an actor-only behavioral cloning dataset from a heuristic policy."
    )
    parser.add_argument("--config", type=Path, default=None, help="Config JSON path.")
    parser.add_argument("--output", type=Path, required=True, help="Output .npz file.")
    parser.add_argument("--policy", type=str, default="zero_wait", help="Heuristic policy name.")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes to roll out.")
    parser.add_argument("--episode-length", type=int, default=500, help="Episode length.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--n-agents",
        type=int,
        default=3,
        help="Number of agents (overridden by config if present).",
    )
    parser.add_argument(
        "--no-agent-id",
        action="store_true",
        help="Store metadata with use_agent_id=False.",
    )
    return parser.parse_args()


def _resolve_n_agents(cfg: Dict[str, object], fallback: int) -> int:
    system_cfg = cfg.get("system", {})
    if isinstance(system_cfg, dict):
        return int(system_cfg.get("n_agents", fallback))
    return int(fallback)


def main() -> None:
    args = parse_args()
    config_path_str = str(args.config) if args.config is not None else None
    cfg = load_config(config_path_str)
    n_agents = _resolve_n_agents(cfg, args.n_agents)
    use_agent_id = not args.no_agent_id

    env = NCS_Env(
        n_agents=n_agents,
        episode_length=args.episode_length,
        config_path=config_path_str,
        seed=args.seed,
    )

    policies = []
    for idx in range(n_agents):
        agent_seed = None if args.seed is None else int(args.seed) + idx
        policies.append(
            get_heuristic_policy(
                args.policy,
                n_agents=n_agents,
                seed=agent_seed,
                agent_index=idx,
            )
        )

    obs_dim = int(env.observation_space.spaces["agent_0"].shape[0])
    n_actions = int(env.action_space.spaces["agent_0"].n)

    step_obs: List[np.ndarray] = []
    step_actions: List[np.ndarray] = []

    for episode in range(int(args.episodes)):
        episode_seed = None if args.seed is None else int(args.seed) + episode
        obs_dict, _info = env.reset(seed=episode_seed)
        for policy in policies:
            policy.reset()

        done = False
        while not done:
            obs = stack_obs(obs_dict, n_agents)
            actions: List[int] = []
            for i in range(n_agents):
                action, _ = policies[i].predict(obs[i], deterministic=True)
                actions.append(int(action))

            action_dict = {f"agent_{i}": actions[i] for i in range(n_agents)}
            next_obs_dict, _rewards_dict, terminated, truncated, _infos = env.step(action_dict)
            done = all(terminated.values()) or all(truncated.values())

            step_obs.append(obs.astype(np.float32, copy=False))
            step_actions.append(np.asarray(actions, dtype=np.int64))

            obs_dict = next_obs_dict

    obs_arr = np.asarray(step_obs, dtype=np.float32)
    actions_arr = np.asarray(step_actions, dtype=np.int64)

    if obs_arr.ndim != 3 or actions_arr.ndim != 2:
        raise RuntimeError("Invalid dataset shapes when generating BC data.")
    if obs_arr.shape[0] != actions_arr.shape[0]:
        raise RuntimeError("BC dataset arrays must align on the first dimension.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        obs=obs_arr,
        actions=actions_arr,
        format_version=np.array(1, dtype=np.int64),
        n_agents=np.array(n_agents, dtype=np.int64),
        obs_dim=np.array(obs_dim, dtype=np.int64),
        n_actions=np.array(n_actions, dtype=np.int64),
        episodes=np.array(int(args.episodes), dtype=np.int64),
        episode_length=np.array(int(args.episode_length), dtype=np.int64),
        seed=np.array(int(args.seed), dtype=np.int64),
        policy_name=np.array(args.policy, dtype=str),
        config_path=np.array(config_path_str or "", dtype=str),
        use_agent_id=np.array(bool(use_agent_id), dtype=bool),
    )

    env.close()
    print(f"Saved {obs_arr.shape[0]} steps to {args.output}")


if __name__ == "__main__":
    main()
