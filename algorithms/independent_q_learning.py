"""
Independent Q-learning baseline for the Networked Control System simulator.

The training loop instantiates :class:`ncs_env.env.NCS_Env` directly and
optimizes a tabular policy for each agent independently by discretizing the
continuous observation vector. The implementation is intentionally simple so
that it can be used as a reference or quick sanity check when experimenting
with new environment settings.
"""

from __future__ import annotations

import argparse
import csv
import copy
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ncs_env.env import NCS_Env
from utils import ObservationDiscretizer, RewardNormalizer, compute_reward_normalizer


class IndependentQLearningAgent:
    """Tabular Q-learning agent with epsilon-greedy exploration."""

    def __init__(
        self,
        n_actions: int,
        discretizer: ObservationDiscretizer,
        learning_rate: float,
        gamma: float,
    ):
        self.n_actions = n_actions
        self.discretizer = discretizer
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.q_table: Dict[Tuple[int, ...], np.ndarray] = {}

    def _q_values(self, key: Tuple[int, ...]) -> np.ndarray:
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.n_actions, dtype=np.float32)
        return self.q_table[key]

    def select_action(self, observation: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
        key = self.discretizer.discretize(observation)
        if rng.random() < epsilon:
            return int(rng.integers(self.n_actions))
        return int(np.argmax(self._q_values(key)))

    def update(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: Optional[np.ndarray],
    ) -> None:
        state_key = self.discretizer.discretize(observation)
        current_q = self._q_values(state_key)

        if next_observation is None:
            td_target = reward
        else:
            next_key = self.discretizer.discretize(next_observation)
            td_target = reward + self.gamma * np.max(self._q_values(next_key))

        td_error = td_target - current_q[action]
        current_q[action] += self.learning_rate * td_error


@dataclass
class TrainingStats:
    episode: int
    total_reward: float
    avg_agent_reward: float
    steps: int
    epsilon: float


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    """Write a list of dictionaries to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class IndependentQLearningTrainer:
    """Manage multi-agent training."""

    def __init__(
        self,
        env: NCS_Env,
        learning_rate: float,
        gamma: float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay: float,
        discretizer: ObservationDiscretizer,
        track_trajectories: bool = False,
        reward_normalizer: Optional[RewardNormalizer] = None,
    ):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.discretizer = discretizer
        self.track_trajectories = track_trajectories
        self.trajectory_records: List[Dict[str, Any]] = []
        self.reward_normalizer = reward_normalizer

        self.agents = {
            agent_id: IndependentQLearningAgent(
                n_actions=int(self.env.action_space[agent_id].n),
                discretizer=self.discretizer,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
            )
            for agent_id in self.env.action_space.spaces.keys()
        }
        self.best_episode_reward: float = -float("inf")
        self.best_q_tables: Optional[Dict[str, Dict[Tuple[int, ...], np.ndarray]]] = None

    @property
    def agent_ids(self) -> List[str]:
        return list(self.agents.keys())

    def train(
        self,
        num_episodes: int,
        max_steps: Optional[int] = None,
        *,
        seed: Optional[int] = None,
        log_interval: int = 10,
    ) -> List[TrainingStats]:
        rng = np.random.default_rng(seed)
        epsilon = self.epsilon_start
        max_steps = max_steps or self.env.episode_length
        history: List[TrainingStats] = []
        if self.track_trajectories:
            self.trajectory_records = []

        for episode in range(1, num_episodes + 1):
            observations, _ = self.env.reset(seed=rng.integers(0, 10_000) if seed is not None else None)
            episode_reward = 0.0
            agent_rewards = {agent_id: 0.0 for agent_id in self.agent_ids}

            steps = 0
            for steps in range(1, max_steps + 1):
                # Select actions for every agent (even if they would otherwise be inactive).
                actions = {
                    agent_id: agent.select_action(observations[agent_id], epsilon, rng)
                    for agent_id, agent in self.agents.items()
                }

                next_obs, rewards, terminated, truncated, _ = self.env.step(actions)
                episode_reward += float(sum(rewards.values()))
                for agent_id in self.agent_ids:
                    agent_rewards[agent_id] += float(rewards[agent_id])
                    done = terminated[agent_id] or truncated[agent_id]
                    next_observation = None if done else next_obs[agent_id]
                    reward_for_update = (
                        self.reward_normalizer(rewards[agent_id])
                        if self.reward_normalizer is not None
                        else rewards[agent_id]
                    )
                    self.agents[agent_id].update(
                        observations[agent_id],
                        actions[agent_id],
                        reward_for_update,
                        next_observation,
                    )
                    if self.track_trajectories:
                        self.trajectory_records.append(
                            {
                                "episode": episode,
                                "step": steps,
                                "agent_id": agent_id,
                                "action": int(actions[agent_id]),
                                "reward": float(rewards[agent_id]),
                                "terminated": bool(terminated[agent_id]),
                                "truncated": bool(truncated[agent_id]),
                            }
                        )

                observations = next_obs
                if all(terminated[agent_id] or truncated[agent_id] for agent_id in self.agent_ids):
                    break

            history.append(
                TrainingStats(
                    episode=episode,
                    total_reward=episode_reward,
                    avg_agent_reward=episode_reward / len(self.agent_ids),
                    steps=steps,
                    epsilon=epsilon,
                )
            )
            if episode_reward > self.best_episode_reward:
                self.best_episode_reward = episode_reward
                self.best_q_tables = {aid: copy.deepcopy(agent.q_table) for aid, agent in self.agents.items()}

            if log_interval and episode % log_interval == 0:
                avg_rewards = np.mean([agent_rewards[aid] for aid in self.agent_ids])
                print(
                    f"Episode {episode:04d} | total_reward={episode_reward:.2f} "
                    f"| avg_agent_reward={avg_rewards:.2f} | epsilon={epsilon:.3f}"
                )

            epsilon = max(self.epsilon_end, epsilon * self.epsilon_decay)

        return history

    def set_q_tables(self, q_tables: Dict[str, Dict[Tuple[int, ...], np.ndarray]]) -> None:
        """Overwrite agent Q-tables."""
        for aid, table in q_tables.items():
            if aid in self.agents:
                self.agents[aid].q_table = copy.deepcopy(table)

    def evaluate(
        self,
        episodes: int,
        max_steps: Optional[int] = None,
        *,
        seed: Optional[int] = None,
        record_actions: bool = False,
    ) -> List[Dict[str, Any]]:
        """Run greedy evaluation and optionally record trajectories."""
        rng = np.random.default_rng(seed)
        max_steps = max_steps or self.env.episode_length
        records: List[Dict[str, Any]] = []

        for episode in range(1, episodes + 1):
            observations, _ = self.env.reset(seed=rng.integers(0, 10_000) if seed is not None else None)
            for step in range(1, max_steps + 1):
                actions = {
                    agent_id: int(
                        np.argmax(
                            self.agents[agent_id]._q_values(
                                self.discretizer.discretize(observations[agent_id])
                            )
                        )
                    )
                    for agent_id in self.agent_ids
                }

                next_obs, rewards, terminated, truncated, _ = self.env.step(actions)
                if record_actions:
                    for agent_id in self.agent_ids:
                        records.append(
                            {
                                "episode": episode,
                                "step": step,
                                "agent_id": agent_id,
                                "action": int(actions[agent_id]),
                                "reward": float(rewards[agent_id]),
                                "terminated": bool(terminated[agent_id]),
                                "truncated": bool(truncated[agent_id]),
                            }
                        )
                observations = next_obs
                if all(terminated[aid] or truncated[aid] for aid in self.agent_ids):
                    break
        return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train independent Q-learning agents.")
    parser.add_argument("--config", type=Path, default=None, help="Path to config JSON file.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum steps per episode (defaults to env episode_length).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--n-agents", type=int, default=3, help="Number of agents.")
    parser.add_argument("--episode-length", type=int, default=500, help="Episode length.")
    parser.add_argument("--comm-cost", type=float, default=0.01, help="Communication penalty.")

    parser.add_argument("--learning-rate", type=float, default=0.1, help="Q-learning step size.")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor.")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial exploration rate.")
    parser.add_argument("--epsilon-end", type=float, default=0.05, help="Minimum exploration rate.")
    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=0.995,
        help="Multiplicative epsilon decay applied after each episode.",
    )

    parser.add_argument("--history-levels", type=int, default=3, help="Bins for history entries.")
    parser.add_argument("--throughput-bins", type=int, default=5, help="Bins for throughput.")
    parser.add_argument(
        "--throughput-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(0.0, 50.0),
        help="Value range (kbps) considered during throughput discretization.",
    )
    parser.add_argument("--state-bins", type=int, default=7, help="Bins per plant state dimension.")
    parser.add_argument(
        "--state-limit",
        type=float,
        default=5.0,
        help="Absolute value used when clipping plant states before binning.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Episode interval for printing training metrics.",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=None,
        help="Path to CSV file where per-episode stats will be saved.",
    )
    parser.add_argument(
        "--trajectory-output",
        type=Path,
        default=None,
        help="Path to CSV file storing per-step agent actions and rewards.",
    )
    parser.add_argument(
        "--pre-normalize-reward",
        action="store_true",
        help="Run a random roll-out to compute reward normalization statistics before training.",
    )
    parser.add_argument(
        "--pre-normalize-episodes",
        type=int,
        default=10,
        help="Episodes used to estimate reward normalization stats when enabled.",
    )
    parser.add_argument(
        "--pre-normalize-seed",
        type=int,
        default=None,
        help="Seed for the reward normalization roll-out.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes to run with the best Q-table (greedy).",
    )
    parser.add_argument(
        "--eval-output",
        type=Path,
        default=None,
        help="CSV path to save evaluation actions and rewards for the best Q-table.",
    )
    parser.add_argument(
        "--best-q-output",
        type=Path,
        default=None,
        help="Path to write the best Q-tables (pickle).",
    )
    return parser.parse_args()


def main(args: Optional[argparse.Namespace] = None) -> List[TrainingStats]:
    args = args or parse_args()
    def make_env():
        return NCS_Env(
            n_agents=args.n_agents,
            episode_length=args.episode_length,
            comm_cost=args.comm_cost,
            config_path=str(args.config) if args.config is not None else None,
            seed=args.seed,
        )

    reward_normalizer: Optional[RewardNormalizer] = None
    if args.pre_normalize_reward:
        reward_normalizer = compute_reward_normalizer(
            make_env,
            episodes=args.pre_normalize_episodes,
            max_steps=args.max_steps,
            seed=args.pre_normalize_seed,
        )
        print(
            f"Reward normalization enabled: mean={reward_normalizer.mean:.4f}, std={reward_normalizer.std:.4f}"
        )

    env = make_env()

    history_window = env.history_window  # type: ignore[attr-defined]
    state_history_window = env.state_history_window  # type: ignore[attr-defined]
    state_dim = env.state_dim  # type: ignore[attr-defined]
    discretizer = ObservationDiscretizer(
        history_window=history_window,
        state_dim=state_dim,
        state_history_window=state_history_window,
        throughput_history_window=history_window,
        history_levels=args.history_levels,
        throughput_bins=args.throughput_bins,
        throughput_range=tuple(args.throughput_range),
        state_bins=args.state_bins,
        state_limits=args.state_limit,
    )

    trainer = IndependentQLearningTrainer(
        env=env,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        discretizer=discretizer,
        track_trajectories=args.trajectory_output is not None,
        reward_normalizer=reward_normalizer,
    )

    history = trainer.train(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        log_interval=args.log_interval,
    )

    if args.stats_output:
        stats_rows = [asdict(record) for record in history]
        _write_csv(
            args.stats_output,
            stats_rows,
            ["episode", "total_reward", "avg_agent_reward", "steps", "epsilon"],
        )

    if args.trajectory_output:
        _write_csv(
            args.trajectory_output,
            trainer.trajectory_records,
            ["episode", "step", "agent_id", "action", "reward", "terminated", "truncated"],
        )

    best_q_tables = trainer.best_q_tables or {aid: agent.q_table for aid, agent in trainer.agents.items()}

    if args.best_q_output:
        args.best_q_output.parent.mkdir(parents=True, exist_ok=True)
        with args.best_q_output.open("wb") as f:
            pickle.dump(best_q_tables, f)
        print(f"Saved best Q-tables to {args.best_q_output}")

    if args.eval_episodes > 0:
        trainer.set_q_tables(best_q_tables)
        eval_records = trainer.evaluate(
            episodes=args.eval_episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            record_actions=args.eval_output is not None,
        )
        if args.eval_output:
            _write_csv(
                args.eval_output,
                eval_records,
                ["episode", "step", "agent_id", "action", "reward", "terminated", "truncated"],
            )
            print(f"Saved evaluation trajectories to {args.eval_output}")

    env.close()
    return history


if __name__ == "__main__":
    main()
