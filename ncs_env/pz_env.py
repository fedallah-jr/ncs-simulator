from __future__ import annotations

from typing import Dict, Optional

from pettingzoo import ParallelEnv

from .env import NCS_Env


class NCSParallelEnv(ParallelEnv):
    """
    PettingZoo Parallel Environment wrapper around NCS_Env.

    Exposes the simultaneous-action API expected by PettingZoo agents while
    reusing the underlying Gymnasium environment implementation.
    """

    metadata = {"name": "ncs_parallel_v0", "render_modes": []}

    def __init__(
        self,
        n_agents: int = 3,
        episode_length: int = 1000,
        config_path: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self._env = NCS_Env(
            n_agents=n_agents,
            episode_length=episode_length,
            config_path=config_path,
            seed=seed,
        )
        self.n_agents = n_agents
        self.possible_agents = [f"agent_{i}" for i in range(n_agents)]
        self.agents = self.possible_agents[:]

        self._observation_spaces = {
            agent: self._env.observation_space[agent] for agent in self.possible_agents
        }
        self._action_spaces = {
            agent: self._env.action_space[agent] for agent in self.possible_agents
        }

    def observation_space(self, agent: str):
        return self._observation_spaces[agent]

    def action_space(self, agent: str):
        return self._action_spaces[agent]

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        observations, info = self._env.reset(seed=seed, options=options)
        self.agents = self.possible_agents[:]
        infos = self._format_info(info)
        return observations, infos

    def step(self, actions: Dict[str, int]):
        if not self.agents:
            raise RuntimeError("step() called after the episode has terminated.")

        full_actions = {}
        for agent in self.possible_agents:
            if agent in self.agents:
                full_actions[agent] = actions.get(agent, 0)
            else:
                full_actions[agent] = 0

        observations, rewards, terminated, truncated, info = self._env.step(full_actions)
        infos = self._format_info(info)

        self.agents = [
            agent for agent in self.possible_agents if not (terminated[agent] or truncated[agent])
        ]
        live_observations = {agent: observations[agent] for agent in self.agents}

        return live_observations, rewards, terminated, truncated, infos

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

    def _format_info(self, info: Dict) -> Dict[str, Dict]:
        """PettingZoo expects an info dict per agent."""
        per_agent_info = {}
        for agent in self.possible_agents:
            per_agent_info[agent] = info.copy() if isinstance(info, dict) else info
        return per_agent_info
