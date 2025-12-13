"""
Environment wrappers and adapters.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import gymnasium as gym


class SingleAgentWrapper(gym.Env):
    """
    Wrap an NCS_Env configured with a single agent into a standard Gym env.

    This strips the Dict observation/action space down to the lone agent entry
    so Stable-Baselines3 algorithms can interact with it directly.
    """

    metadata = {"render_modes": []}

    def __init__(self, make_env: Callable[[], Any], other_agent_action: int = 0):
        super().__init__()
        self._make_env = make_env
        self._other_agent_action = int(other_agent_action)
        self.env: Optional[Any] = None
        tmp_env = make_env()
        self._n_agents = int(getattr(tmp_env, "n_agents", 1))
        self.observation_space = tmp_env.observation_space.spaces["agent_0"]
        self.action_space = tmp_env.action_space.spaces["agent_0"]
        tmp_env.close()

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if self.env is None:
            self.env = self._make_env()
        obs_dict, info = self.env.reset(seed=seed, options=options)
        return obs_dict["agent_0"], info

    def step(self, action):
        full_actions: Dict[str, int] = {
            f"agent_{i}": self._other_agent_action for i in range(self._n_agents)
        }
        full_actions["agent_0"] = int(action)
        obs_dict, rewards, terminated, truncated, info = self.env.step(full_actions)
        return (
            obs_dict["agent_0"],
            float(rewards["agent_0"]),
            bool(terminated["agent_0"]),
            bool(truncated["agent_0"]),
            info,
        )

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def get_reward_mix_weight(self) -> float:
        """
        Expose the underlying environment's reward mixing weight.

        Creates the env on demand to avoid attribute errors.
        """
        if self.env is None:
            self.env = self._make_env()
            self.env.reset()
        if hasattr(self.env, "get_reward_mix_weight"):
            return float(self.env.get_reward_mix_weight())
        return 0.0
