"""Shared utilities for policy loading and agent resolution.

This module centralizes code used by both ``policy_tester`` and
``visualize_policy`` so that definitions are not duplicated.
"""

from __future__ import annotations

import json
import os
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


# PyTorch does very small CPU inference batches in policy_tester. On machines
# with many cores, the default intra-op/inter-op thread pools can dominate the
# actual MLP work by orders of magnitude. Keep this helper local to policy
# loading so training code is not affected.
_TORCH_INFERENCE_THREADS_CONFIGURED = False


def _configure_torch_policy_inference_threads(torch_module: Any) -> None:
    global _TORCH_INFERENCE_THREADS_CONFIGURED
    if _TORCH_INFERENCE_THREADS_CONFIGURED:
        return

    thread_count = 1
    os.environ.setdefault("OMP_NUM_THREADS", str(thread_count))
    os.environ.setdefault("MKL_NUM_THREADS", str(thread_count))
    try:
        torch_module.set_num_threads(thread_count)
    except Exception:
        pass
    try:
        torch_module.set_num_interop_threads(thread_count)
    except Exception:
        # PyTorch only allows this before inter-op work has started. In that
        # case, set_num_threads above still prevents the expensive intra-op
        # oversubscription that hurts policy testing most.
        pass
    _TORCH_INFERENCE_THREADS_CONFIGURED = True


def validate_marl_checkpoint_matches_env_spec(
    *,
    env_n_agents: int,
    env_obs_dim: int,
    env_n_actions: int,
    meta: Any,
) -> None:
    """Compare environment dims against checkpoint metadata.

    Public helper so vectorized eval scripts (which expose dims via
    ``single_observation_space``) can reuse the same checks as single-env
    callers (``NCS_Env``).
    """
    if env_n_agents != meta.n_agents:
        raise ValueError(
            f"Env n_agents={env_n_agents} does not match "
            f"checkpoint n_agents={meta.n_agents}."
        )
    if env_obs_dim != meta.obs_dim:
        raise ValueError(
            f"Env obs_dim={env_obs_dim} does not match checkpoint obs_dim={meta.obs_dim}"
        )
    if env_n_actions != meta.n_actions:
        raise ValueError(
            f"Env n_actions={env_n_actions} does not match checkpoint n_actions={meta.n_actions}. "
            "Ensure the config and checkpoint use the same action space."
        )


def _validate_marl_checkpoint_matches_env(env: NCS_Env, meta: Any) -> None:
    validate_marl_checkpoint_matches_env_spec(
        env_n_agents=int(getattr(env, "n_agents", 0)),
        env_obs_dim=int(env.observation_space.spaces["agent_0"].shape[0]),
        env_n_actions=int(env.action_space.spaces["agent_0"].n),
        meta=meta,
    )


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
        env: Optional[NCS_Env] = None,
    ) -> None:
        self.policy_name = policy_name
        self.n_agents = int(n_agents)
        self.deterministic = bool(deterministic)
        self.env = env
        self._policies = []
        for idx in range(self.n_agents):
            agent_seed = None if seed is None else int(seed) + idx
            self._policies.append(
                get_heuristic_policy(
                    policy_name,
                    n_agents=self.n_agents,
                    seed=agent_seed,
                    agent_index=idx,
                    env=self.env,
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
    _configure_torch_policy_inference_threads(torch)

    ckpt = torch.load(str(model_path), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError("MARL torch checkpoint must be a dict")
    if "n_agents" not in ckpt:
        return None
    return int(ckpt["n_agents"])


def infer_policy_n_agents(policy_path: str, policy_type: str) -> Optional[int]:
    policy_type_norm = policy_type.lower()
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

def load_marl_torch_multi_agent_policy(
    model_path: str,
    env: NCS_Env,
    *,
    ndq_cut_mu_thres: float = 0.0,
):
    import torch
    _configure_torch_policy_inference_threads(torch)

    # Check if this is a DIAL checkpoint
    ckpt = torch.load(str(model_path), map_location="cpu")
    if isinstance(ckpt, dict) and ckpt.get("dial", False):
        from utils.marl.torch_policy import MARLDialRNNTorchPolicy, load_dial_rnn_agent_from_checkpoint
        from utils.marl.networks import DRU

        agent, meta, comm_dim, dru_sigma = load_dial_rnn_agent_from_checkpoint(Path(model_path))
        _validate_marl_checkpoint_matches_env(env, meta)
        dru = DRU(sigma=dru_sigma)
        return MARLDialRNNTorchPolicy(agent, meta, dru, comm_dim, device=torch.device("cpu"))
    if isinstance(ckpt, dict) and ckpt.get("ndq", False):
        from utils.marl.torch_policy import MARLNDQTorchPolicy, load_ndq_agent_from_checkpoint

        agent, comm_encoder, meta, comm_embed_dim = load_ndq_agent_from_checkpoint(Path(model_path))
        _validate_marl_checkpoint_matches_env(env, meta)
        return MARLNDQTorchPolicy(
            agent, comm_encoder, meta, comm_embed_dim,
            device=torch.device("cpu"),
            cut_mu_thres=float(ndq_cut_mu_thres),
        )
    if isinstance(ckpt, dict) and ckpt.get("rnn_qmix", False):
        from utils.marl_rnn_qmix import (
            MARLRNNQMIXTorchPolicy,
            load_rnn_qmix_agent_from_checkpoint,
        )

        agent, meta = load_rnn_qmix_agent_from_checkpoint(Path(model_path))
        _validate_marl_checkpoint_matches_env(env, meta)
        return MARLRNNQMIXTorchPolicy(agent, meta, device=torch.device("cpu"))

    from utils.marl.torch_policy import MARLTorchMultiAgentPolicy, load_marl_torch_agents_from_checkpoint

    agent_or_agents, meta = load_marl_torch_agents_from_checkpoint(Path(model_path))
    _validate_marl_checkpoint_matches_env(env, meta)
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
    if policy_type == "heuristic":
        return MultiAgentHeuristicPolicy(
            policy_path,
            n_agents=n_agents,
            seed=seed,
            deterministic=deterministic,
            env=env,
        )
    raise ValueError(
        "Multi-agent visualization supports only 'marl_torch' and 'heuristic' policies."
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
