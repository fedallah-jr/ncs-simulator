from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional, Tuple, List

import numpy as np
from gymnasium.vector import AsyncVectorEnv
from gymnasium.vector.async_vector_env import AutoresetMode


def make_env(
    n_agents: int,
    episode_length: int,
    config_path_str: Optional[str],
    seed: Optional[int],
    reward_override: Optional[Dict[str, Any]] = None,
    termination_override: Optional[Dict[str, Any]] = None,
    freeze_running_normalization: bool = False,
) -> "NCS_Env":
    from ncs_env.env import NCS_Env

    return NCS_Env(
        n_agents=n_agents,
        episode_length=episode_length,
        config_path=config_path_str,
        seed=seed,
        reward_override=reward_override,
        termination_override=termination_override,
        freeze_running_normalization=freeze_running_normalization,
    )


def make_vector_env_fn(
    n_agents: int,
    episode_length: int,
    config_path_str: Optional[str],
    seed: Optional[int],
) -> Callable[[], "VectorEnvAdapter"]:
    def _thunk() -> "VectorEnvAdapter":
        env = make_env(n_agents, episode_length, config_path_str, seed)
        return VectorEnvAdapter(env, n_agents)

    return _thunk


def stack_vector_obs(obs_dict: Dict[str, Any], n_agents: int) -> np.ndarray:
    return np.stack(
        [np.asarray(obs_dict[f"agent_{i}"], dtype=np.float32) for i in range(n_agents)],
        axis=1,
    )


class VectorEnvAdapter:
    def __init__(self, env: Any, n_agents: int) -> None:
        self.env = env
        self.n_agents = int(n_agents)
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.metadata = getattr(env, "metadata", {})
        self.render_mode = getattr(env, "render_mode", None)

    def reset(self, **kwargs: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return self.env.reset(**kwargs)

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], np.ndarray, bool, bool, Dict[str, Any]]:
        obs, rewards, terminated, truncated, info = self.env.step(action)
        rewards_arr = np.asarray(
            [rewards[f"agent_{i}"] for i in range(self.n_agents)], dtype=np.float32
        )
        terminated_any = any(terminated[f"agent_{i}"] for i in range(self.n_agents))
        truncated_any = any(truncated[f"agent_{i}"] for i in range(self.n_agents))
        return obs, rewards_arr, bool(terminated_any), bool(truncated_any), info

    def close(self) -> None:
        self.env.close()


def create_async_vector_env(
    n_envs: int,
    n_agents: int,
    episode_length: int,
    config_path_str: Optional[str],
    seed: Optional[int],
) -> Tuple[AsyncVectorEnv, Optional[List[int]]]:
    if n_envs <= 0:
        raise ValueError("n_envs must be positive")

    env_seeds = None
    if seed is not None:
        env_seeds = [int(seed) + env_idx for env_idx in range(n_envs)]

    env_fns = []
    for env_idx in range(n_envs):
        env_seed = None if env_seeds is None else env_seeds[env_idx]
        env_fns.append(
            make_vector_env_fn(
                n_agents=n_agents,
                episode_length=episode_length,
                config_path_str=config_path_str,
                seed=env_seed,
            )
        )

    shared_memory = True
    shared_memory_flag = os.getenv("NCS_SHARED_MEMORY")
    if shared_memory_flag is not None:
        shared_memory = shared_memory_flag.strip().lower() not in {"0", "false", "no", "off"}

    env = AsyncVectorEnv(
        env_fns,
        shared_memory=shared_memory,
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    return env, env_seeds
