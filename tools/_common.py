"""Shared utilities for policy loading and agent resolution.

This module centralizes code used by both ``policy_tester`` and
``visualize_policy`` so that definitions are not duplicated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ncs_env.config import load_config
from ncs_env.env import NCS_Env
from tools.heuristic_policies import get_heuristic_policy
from utils.marl.obs_normalization import RunningObsNormalizer


# ---------------------------------------------------------------------------
# Filename sanitization
# ---------------------------------------------------------------------------

def sanitize_filename(value: str) -> str:
    keep = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("_.")
    return out if out else "policy"


# ---------------------------------------------------------------------------
# Multi-agent heuristic policy wrapper
# ---------------------------------------------------------------------------

class MultiAgentHeuristicPolicy:
    def __init__(
        self,
        policy_name: str,
        n_agents: int,
        seed: Optional[int],
        *,
        deterministic: bool = False,
    ) -> None:
        self.policy_name = policy_name
        self.n_agents = int(n_agents)
        self.deterministic = bool(deterministic)
        self._policies = []
        for idx in range(self.n_agents):
            agent_seed = None if seed is None else int(seed) + idx
            self._policies.append(
                get_heuristic_policy(
                    policy_name,
                    n_agents=self.n_agents,
                    seed=agent_seed,
                    agent_index=idx,
                )
            )

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


# ---------------------------------------------------------------------------
# n_agents resolution helpers
# ---------------------------------------------------------------------------

def read_marl_torch_n_agents(model_path: Path) -> Optional[int]:
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


def read_es_n_agents(model_path: Path) -> Optional[int]:
    try:
        with np.load(str(model_path)) as data:
            if "n_agents" not in data:
                return None
            return int(data["n_agents"])
    except Exception as exc:
        raise ValueError(f"Could not load numpy data from {model_path}: {exc}") from exc


def infer_policy_n_agents(policy_path: str, policy_type: str) -> Optional[int]:
    policy_type_norm = policy_type.lower()
    if policy_type_norm in {"es", "openai_es"}:
        return read_es_n_agents(Path(policy_path))
    if policy_type_norm == "marl_torch":
        return read_marl_torch_n_agents(Path(policy_path))
    return None


def resolve_n_agents(
    config: Dict[str, Any],
    policy_specs: Sequence[Tuple[str, str]],
    explicit_n_agents: Optional[int],
) -> int:
    """Resolve n_agents from checkpoints, config, or explicit override.

    *policy_specs* is a sequence of ``(policy_path, policy_type)`` tuples.
    """
    inferred_values: List[int] = []
    for policy_path, policy_type in policy_specs:
        inferred = infer_policy_n_agents(policy_path, policy_type)
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
    raise ValueError("n_agents could not be resolved; set system.n_agents or pass --n-agents")


# ---------------------------------------------------------------------------
# Policy loaders
# ---------------------------------------------------------------------------

def load_es_policy(model_path: str, env: NCS_Env):
    """Load an OpenAI-ES policy (JAX/Flax).

    Args:
        model_path: Path to the saved model (.npz file)
        env: Multi-agent environment instance

    Returns:
        Policy object with act() method
    """
    model_path_p = Path(model_path)
    if not model_path_p.exists():
        raise FileNotFoundError(f"Model file not found: {model_path_p}")

    try:
        import jax
        import jax.numpy as jnp
        from jax import flatten_util
        from algorithms.openai_es import create_policy_net
    except ImportError:
        raise ImportError(
            "jax, flax, and evosax are required to load ES policies. "
            "Install with: pip install jax jaxlib flax evosax"
        )

    try:
        data = np.load(str(model_path_p))
        flat_params = data['flat_params']

        if 'hidden_dims' in data:
            hidden_dims = tuple(int(x) for x in data['hidden_dims'])
        elif 'hidden_size' in data:
            hidden_dims = (int(data['hidden_size']), int(data['hidden_size']))
        else:
            hidden_dims = (64, 64)
        activation = str(data['activation']) if 'activation' in data else "tanh"
        normalize_obs = bool(data['normalize_obs']) if 'normalize_obs' in data else False
        obs_norm_clip = float(data['obs_norm_clip']) if 'obs_norm_clip' in data else 5.0
        obs_norm_eps = float(data['obs_norm_eps']) if 'obs_norm_eps' in data else 1e-8
        obs_norm_mean = data['obs_norm_mean'] if 'obs_norm_mean' in data else None
        obs_norm_m2 = data['obs_norm_m2'] if 'obs_norm_m2' in data else None
        obs_norm_count = int(data['obs_norm_count']) if 'obs_norm_count' in data else 0
        saved_n_agents = int(data['n_agents']) if 'n_agents' in data else None
        use_agent_id = bool(data['use_agent_id']) if 'use_agent_id' in data else False
        saved_obs_dim = int(data['obs_dim']) if 'obs_dim' in data else None

        print(
            "  Architecture: "
            f"hidden_dims={hidden_dims}, activation={activation}, "
            f"normalize_obs={normalize_obs}, use_agent_id={use_agent_id}"
        )
    except Exception as e:
        raise ValueError(f"Could not load numpy data from {model_path_p}: {e}")

    if hasattr(env, "action_space") and hasattr(env.action_space, "spaces"):
        action_dim = int(env.action_space.spaces["agent_0"].n)
    else:
        raise ValueError("Environment must expose per-agent action spaces.")

    if hasattr(env, "observation_space") and hasattr(env.observation_space, "spaces"):
        obs_dim = int(env.observation_space.spaces["agent_0"].shape[0])
    else:
        raise ValueError("Environment must expose per-agent observation spaces.")
    env_n_agents = int(getattr(env, "n_agents", 0))
    if env_n_agents < 1:
        raise ValueError("Environment must define n_agents for ES policies.")
    n_agents = int(saved_n_agents) if saved_n_agents is not None else env_n_agents

    if saved_n_agents is not None and env_n_agents != n_agents:
        raise ValueError(f"Env n_agents={env_n_agents} does not match checkpoint n_agents={n_agents}")
    if saved_obs_dim is not None and obs_dim != saved_obs_dim:
        raise ValueError(f"Env obs_dim={obs_dim} does not match checkpoint obs_dim={saved_obs_dim}")
    input_dim = obs_dim + (n_agents if use_agent_id else 0)

    model = create_policy_net(
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        activation=activation,
    )

    rng = jax.random.PRNGKey(0)
    dummy_obs = jnp.zeros((1, input_dim))
    dummy_params = model.init(rng, dummy_obs)

    _, unravel_fn = flatten_util.ravel_pytree(dummy_params)
    params = unravel_fn(flat_params)

    obs_normalizer = None
    obs_normalizer_update = False
    if normalize_obs:
        clip_value = None if obs_norm_clip <= 0 else float(obs_norm_clip)
        obs_normalizer = RunningObsNormalizer.create(
            obs_dim, clip=clip_value, eps=float(obs_norm_eps)
        )
        if obs_norm_mean is not None and obs_norm_m2 is not None:
            obs_normalizer.set_state(obs_norm_mean, obs_norm_m2, obs_norm_count)
            obs_normalizer_update = False
        else:
            obs_normalizer_update = True

    class ESMultiAgentPolicy:
        def __init__(
            self,
            model,
            params,
            n_agents: int,
            use_agent_id: bool,
            obs_normalizer: Optional[RunningObsNormalizer],
            obs_normalizer_update: bool,
        ):
            self.model = model
            self.params = params
            self.n_agents = int(n_agents)
            self.use_agent_id = bool(use_agent_id)
            self.obs_normalizer = obs_normalizer
            self.obs_normalizer_update = bool(obs_normalizer_update)

        def reset(self) -> None:
            return None

        def act(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, int]:
            obs_batch = np.stack(
                [np.asarray(obs_dict[f"agent_{i}"], dtype=np.float32) for i in range(self.n_agents)],
                axis=0,
            )
            if self.obs_normalizer is not None:
                obs_batch = self.obs_normalizer.normalize(
                    obs_batch, update=self.obs_normalizer_update
                )
            if self.use_agent_id:
                agent_ids = np.eye(self.n_agents, dtype=obs_batch.dtype)
                obs_batch = np.concatenate([obs_batch, agent_ids], axis=1)
            logits = self.model.apply(self.params, jnp.array(obs_batch))
            actions = np.argmax(np.asarray(logits), axis=1)
            return {f"agent_{i}": int(actions[i]) for i in range(self.n_agents)}

    return ESMultiAgentPolicy(
        model,
        params,
        n_agents=n_agents,
        use_agent_id=use_agent_id,
        obs_normalizer=obs_normalizer,
        obs_normalizer_update=obs_normalizer_update,
    )


def load_marl_torch_multi_agent_policy(model_path: str, env: NCS_Env):
    from utils.marl.torch_policy import MARLTorchMultiAgentPolicy, load_marl_torch_agents_from_checkpoint

    try:
        import torch
    except ImportError as e:
        raise ImportError("torch is required to load MARL torch checkpoints") from e

    agent_or_agents, meta = load_marl_torch_agents_from_checkpoint(Path(model_path))
    if int(getattr(env, "n_agents", 0)) != meta.n_agents:
        raise ValueError(
            f"Env n_agents={getattr(env, 'n_agents', None)} does not match checkpoint n_agents={meta.n_agents}. "
            "Ensure the config and checkpoint describe the same agent count."
        )
    env_obs_dim = int(env.observation_space.spaces["agent_0"].shape[0])
    if env_obs_dim != meta.obs_dim:
        raise ValueError(f"Env obs_dim={env_obs_dim} does not match checkpoint obs_dim={meta.obs_dim}")
    return MARLTorchMultiAgentPolicy(agent_or_agents, meta, device=torch.device("cpu"))


def load_multi_agent_policy(
    policy_path: str,
    policy_type: str,
    env: NCS_Env,
    n_agents: int,
    seed: Optional[int],
    *,
    deterministic: bool = False,
):
    if policy_type == "marl_torch":
        return load_marl_torch_multi_agent_policy(policy_path, env)
    if policy_type in {"es", "openai_es"}:
        return load_es_policy(policy_path, env)
    if policy_type == "heuristic":
        return MultiAgentHeuristicPolicy(
            policy_path,
            n_agents=n_agents,
            seed=seed,
            deterministic=deterministic,
        )
    raise ValueError(
        "Multi-agent visualization supports only 'marl_torch', 'es', and 'heuristic' policies."
    )


# ---------------------------------------------------------------------------
# Config override helpers (--set key=value)
# ---------------------------------------------------------------------------

def parse_set_overrides(set_args: Optional[List[str]]) -> Dict[str, Any]:
    """Parse ``--set key=value`` arguments into a nested dict."""
    if not set_args:
        return {}
    overrides: Dict[str, Any] = {}
    for arg in set_args:
        if "=" not in arg:
            raise ValueError(f"Invalid --set argument (expected key=value): {arg}")
        key, value_str = arg.split("=", 1)
        parts = key.split(".")
        if not all(parts):
            raise ValueError(f"Invalid key in --set argument: {key}")
        try:
            parsed_value = json.loads(value_str)
        except (json.JSONDecodeError, ValueError):
            parsed_value = value_str
        current = overrides
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = parsed_value
    return overrides


def deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *overrides* into *base* (mutates *base*)."""
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def apply_config_overrides(
    config_path: Path,
    overrides: Dict[str, Any],
    output_path: Path,
) -> Path:
    """Load *config_path*, deep-merge *overrides*, write to *output_path*."""
    config = load_config(str(config_path))
    deep_merge(config, overrides)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(config, handle, indent=2, sort_keys=True, ensure_ascii=True)
    return output_path


def add_set_override_argument(parser) -> None:
    """Add the ``--set KEY=VALUE`` argument to an argparse parser."""
    parser.add_argument(
        "--set",
        action="append",
        default=None,
        dest="set_overrides",
        metavar="KEY=VALUE",
        help="Override config values using dot notation (e.g., --set network.data_packet_size=80). Repeatable.",
    )
