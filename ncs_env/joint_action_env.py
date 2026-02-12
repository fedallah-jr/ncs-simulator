from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .env import NCS_Env


def decode_joint_action(action_index: int, n_agents: int, n_actions: int) -> List[int]:
    """
    Decode a flat joint action index into per-agent discrete actions.

    Agent 0 is the least-significant digit in base ``n_actions``.
    """
    if n_agents <= 0:
        raise ValueError("n_agents must be positive")
    if n_actions <= 0:
        raise ValueError("n_actions must be positive")

    max_joint_actions = n_actions ** n_agents
    if action_index < 0 or action_index >= max_joint_actions:
        raise ValueError(
            f"joint action index {action_index} out of bounds for {max_joint_actions} joint actions"
        )

    remaining = int(action_index)
    actions: List[int] = []
    for _ in range(n_agents):
        actions.append(int(remaining % n_actions))
        remaining //= n_actions
    return actions


def encode_joint_action(actions: Sequence[int], n_actions: int) -> int:
    """Encode per-agent discrete actions into a single flat joint action index."""
    if n_actions <= 0:
        raise ValueError("n_actions must be positive")
    if len(actions) == 0:
        raise ValueError("actions must be non-empty")

    action_index = 0
    base = 1
    for action in actions:
        action_int = int(action)
        if action_int < 0 or action_int >= n_actions:
            raise ValueError(f"action {action_int} out of bounds for n_actions={n_actions}")
        action_index += action_int * base
        base *= n_actions
    return int(action_index)


class CentralizedJointActionEnv(gym.Env):
    """
    Single-agent centralized wrapper around ``NCS_Env``.

    Observation:
        Concatenation of all per-agent observations.
    Action:
        A single discrete joint-action index covering all per-agent actions.
    Reward:
        Sum of per-agent rewards from the wrapped environment.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_agents: int = 3,
        episode_length: int = 1000,
        config_path: Optional[str] = None,
        seed: Optional[int] = None,
        reward_override: Optional[Dict[str, Any]] = None,
        termination_override: Optional[Dict[str, Any]] = None,
        freeze_running_normalization: bool = False,
        minimal_info: bool = False,
    ) -> None:
        super().__init__()
        self.env = NCS_Env(
            n_agents=n_agents,
            episode_length=episode_length,
            config_path=config_path,
            seed=seed,
            reward_override=reward_override,
            termination_override=termination_override,
            freeze_running_normalization=freeze_running_normalization,
            minimal_info=minimal_info,
        )
        self.n_agents = int(self.env.n_agents)
        self.agent_keys = [f"agent_{i}" for i in range(self.n_agents)]

        first_obs_space = self.env.observation_space[self.agent_keys[0]]
        if not isinstance(first_obs_space, spaces.Box):
            raise TypeError("CentralizedJointActionEnv expects per-agent Box observations")
        self.per_agent_obs_dim = int(np.prod(first_obs_space.shape))

        first_action_space = self.env.action_space[self.agent_keys[0]]
        if not isinstance(first_action_space, spaces.Discrete):
            raise TypeError("CentralizedJointActionEnv expects per-agent Discrete actions")
        self.per_agent_n_actions = int(first_action_space.n)
        self.n_joint_actions = int(self.per_agent_n_actions ** self.n_agents)

        obs_low = np.tile(
            np.asarray(first_obs_space.low, dtype=np.float32).reshape(-1),
            self.n_agents,
        )
        obs_high = np.tile(
            np.asarray(first_obs_space.high, dtype=np.float32).reshape(-1),
            self.n_agents,
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_joint_actions)

    def _flatten_obs(self, obs_dict: Dict[str, Any]) -> np.ndarray:
        parts = [
            np.asarray(obs_dict[agent_key], dtype=np.float32).reshape(-1)
            for agent_key in self.agent_keys
        ]
        return np.concatenate(parts, axis=0)

    def _joint_action_to_dict(self, action_index: int) -> Dict[str, int]:
        per_agent_actions = decode_joint_action(
            action_index=action_index,
            n_agents=self.n_agents,
            n_actions=self.per_agent_n_actions,
        )
        return {
            agent_key: int(per_agent_actions[agent_idx])
            for agent_idx, agent_key in enumerate(self.agent_keys)
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs_dict, info = self.env.reset(seed=seed, options=options)
        flat_obs = self._flatten_obs(obs_dict)
        info_out = dict(info) if isinstance(info, dict) else {}
        return flat_obs, info_out

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action_index = int(action)
        action_dict = self._joint_action_to_dict(action_index)
        obs_dict, rewards_dict, terminated_dict, truncated_dict, info = self.env.step(action_dict)

        reward = float(sum(float(rewards_dict[agent_key]) for agent_key in self.agent_keys))
        terminated = any(bool(terminated_dict[agent_key]) for agent_key in self.agent_keys)
        truncated = any(bool(truncated_dict[agent_key]) for agent_key in self.agent_keys)

        info_out = dict(info) if isinstance(info, dict) else {}
        info_out["joint_action"] = int(action_index)
        info_out["agent_actions"] = [int(action_dict[agent_key]) for agent_key in self.agent_keys]
        info_out["agent_rewards"] = [float(rewards_dict[agent_key]) for agent_key in self.agent_keys]

        flat_obs = self._flatten_obs(obs_dict)
        return flat_obs, reward, bool(terminated), bool(truncated), info_out

    def render(self) -> Any:
        return self.env.render()

    def close(self) -> None:
        self.env.close()
