"""
Policy Visualization Tool for Networked Control Systems

This tool visualizes the state evolution of policies trained or defined in the NCS simulator.
It supports learned policies (OpenAI-ES, MARL torch) and heuristic policies.

Usage:
    # Visualize an ES policy
    python -m tools.visualize_policy --config configs/perfect_comm.json --policy path/to/model.npz --policy-type es

    # Visualize a heuristic policy
    python -m tools.visualize_policy --config configs/perfect_comm.json --policy always_send --policy-type heuristic

    # Visualize an IQL/VDN/QMIX PyTorch checkpoint (all agents act)
    python -m tools.visualize_policy --config configs/marl_mixed_plants.json --policy path/to/latest_model.pt --policy-type marl_torch --generate-video --per-agent-videos

    # Compare MARL vs. always_send (multi-agent)
    python -m tools.visualize_policy --config configs/marl_mixed_plants.json \
        --policies path/to/latest_model.pt always_send \
        --policy-types marl_torch heuristic \
        --labels "MARL" "Always Send" \
        --generate-video --per-agent-videos

    # Compare multiple policies
    python -m tools.visualize_policy --config configs/marl_mixed_plants.json \
        --policies path/to/latest_model.pt always_send send_every_5 \
        --policy-types marl_torch heuristic heuristic \
        --labels "MARL" "Always Send" "Send Every 5"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import json
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ncs_env.env import NCS_Env
from ncs_env.config import load_config
from tools.heuristic_policies import get_heuristic_policy, HEURISTIC_POLICIES


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
            self._policies.append(get_heuristic_policy(policy_name, n_agents=1, seed=agent_seed))

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


def _read_marl_torch_n_agents(model_path: Path) -> Optional[int]:
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


def _read_es_n_agents(model_path: Path) -> Optional[int]:
    try:
        with np.load(str(model_path)) as data:
            if "n_agents" not in data:
                return None
            return int(data["n_agents"])
    except Exception as exc:
        raise ValueError(f"Could not load numpy data from {model_path}: {exc}") from exc


def _infer_policy_n_agents(policy_path: str, policy_type: str) -> Optional[int]:
    policy_type_norm = policy_type.lower()
    if policy_type_norm in {"es", "openai_es"}:
        return _read_es_n_agents(Path(policy_path))
    if policy_type_norm == "marl_torch":
        return _read_marl_torch_n_agents(Path(policy_path))
    return None


def _resolve_run_n_agents(
    policy_paths: List[str],
    policy_types: List[str],
    *,
    config: Dict[str, Any],
    explicit_n_agents: Optional[int],
) -> int:
    inferred_values: List[int] = []
    for policy_path, policy_type in zip(policy_paths, policy_types):
        inferred = _infer_policy_n_agents(policy_path, policy_type)
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


def load_es_policy(model_path: str, env: NCS_Env):
    """
    Load an OpenAI-ES policy (JAX/Flax).

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
        import numpy as np
        from jax import flatten_util
        from algorithms.openai_es import create_policy_net
    except ImportError:
        raise ImportError(
            "jax, flax, and evosax are required to load ES policies. "
            "Install with: pip install jax jaxlib flax evosax"
        )

    # Load saved data including architecture info
    try:
        data = np.load(str(model_path_p))
        flat_params = data['flat_params']
        
        # Load architecture parameters with backward compatibility
        hidden_size = int(data['hidden_size']) if 'hidden_size' in data else 64
        use_layer_norm = bool(data['use_layer_norm']) if 'use_layer_norm' in data else False

        saved_n_agents = int(data['n_agents']) if 'n_agents' in data else None
        use_agent_id = bool(data['use_agent_id']) if 'use_agent_id' in data else False
        saved_obs_dim = int(data['obs_dim']) if 'obs_dim' in data else None

        print(
            "  Architecture: "
            f"hidden_size={hidden_size}, use_layer_norm={use_layer_norm}, "
            f"use_agent_id={use_agent_id}"
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

    # Create model with correct architecture
    model = create_policy_net(action_dim=action_dim, hidden_size=hidden_size, use_layer_norm=use_layer_norm)

    # Initialize dummy to get structure
    rng = jax.random.PRNGKey(0)
    dummy_obs = jnp.zeros((1, input_dim))
    dummy_params = model.init(rng, dummy_obs)

    _, unravel_fn = flatten_util.ravel_pytree(dummy_params)
    params = unravel_fn(flat_params)

    class ESMultiAgentPolicy:
        def __init__(self, model, params, n_agents: int, use_agent_id: bool):
            self.model = model
            self.params = params
            self.n_agents = int(n_agents)
            self.use_agent_id = bool(use_agent_id)

        def reset(self) -> None:
            return None

        def act(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, int]:
            obs_batch = np.stack(
                [np.asarray(obs_dict[f"agent_{i}"], dtype=np.float32) for i in range(self.n_agents)],
                axis=0,
            )
            if self.use_agent_id:
                agent_ids = np.eye(self.n_agents, dtype=obs_batch.dtype)
                obs_batch = np.concatenate([obs_batch, agent_ids], axis=1)
            logits = self.model.apply(self.params, jnp.array(obs_batch))
            actions = np.argmax(np.asarray(logits), axis=1)
            return {f"agent_{i}": int(actions[i]) for i in range(self.n_agents)}

    return ESMultiAgentPolicy(model, params, n_agents=n_agents, use_agent_id=use_agent_id)


def load_marl_torch_multi_agent_policy(model_path: str, env: NCS_Env) -> MARLTorchMultiAgentPolicy:
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


def _sanitize_filename(value: str) -> str:
    keep = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("_.")
    return out if out else "policy"


def run_episode_multi_agent(
    env: NCS_Env,
    policy: Any,
    episode_length: int,
    *,
    seed: Optional[int] = None,
    network_trace: bool = False,
    trace_interval: int = 50,
    trace_start: int = 1,
) -> Dict[str, Any]:
    if hasattr(policy, "reset"):
        policy.reset()

    obs_dict, info = env.reset(seed=seed)
    n_agents = int(env.n_agents)
    state_dim = int(env.state_dim)
    network_tick_traces: List[Dict[str, Any]] = []

    _configure_network_trace_env(
        env,
        enabled=network_trace,
        trace_interval=trace_interval,
        trace_start=trace_start,
    )

    states = np.zeros((episode_length + 1, n_agents, state_dim), dtype=np.float32)
    estimates = np.zeros((episode_length + 1, n_agents, state_dim), dtype=np.float32)
    actions = np.zeros((episode_length, n_agents), dtype=np.int64)
    rewards = np.zeros((episode_length, n_agents), dtype=np.float32)
    state_errors = np.zeros((episode_length + 1, n_agents), dtype=np.float32)
    throughputs = np.zeros(episode_length + 1, dtype=np.float32)
    collided_packets = np.zeros(episode_length + 1, dtype=np.float32)
    network_stats: Dict[str, List[int]] = {
        "tx_attempts": [0 for _ in range(n_agents)],
        "tx_acked": [0 for _ in range(n_agents)],
        "tx_dropped": [0 for _ in range(n_agents)],
        "tx_rewrites": [0 for _ in range(n_agents)],
        "tx_collisions": [0 for _ in range(n_agents)],
    }

    states[0] = np.asarray(info.get("states", states[0]), dtype=np.float32)
    estimates[0] = np.asarray(info.get("estimates", estimates[0]), dtype=np.float32)
    state_errors[0] = np.linalg.norm(states[0], axis=1).astype(np.float32)
    throughputs[0] = float(info.get("true_goodput_kbps_total", info.get("throughput_kbps", 0.0)))
    collided_packets[0] = float(info.get("collided_packets", 0.0))
    if "network_stats" in info:
        network_stats = {k: [int(x) for x in v] for k, v in info["network_stats"].items()}

    for t in range(episode_length):
        action_dict = policy.act(obs_dict)
        actions[t] = np.asarray([action_dict[f"agent_{i}"] for i in range(n_agents)], dtype=np.int64)
        obs_dict, rewards_dict, terminated, truncated, info = env.step(action_dict)
        rewards[t] = np.asarray([rewards_dict[f"agent_{i}"] for i in range(n_agents)], dtype=np.float32)

        states[t + 1] = np.asarray(info.get("states", states[t + 1]), dtype=np.float32)
        estimates[t + 1] = np.asarray(info.get("estimates", estimates[t + 1]), dtype=np.float32)
        state_errors[t + 1] = np.linalg.norm(states[t + 1], axis=1).astype(np.float32)
        throughputs[t + 1] = float(info.get("true_goodput_kbps_total", info.get("throughput_kbps", 0.0)))
        collided_packets[t + 1] = float(info.get("collided_packets", 0.0))
        if "network_stats" in info:
            network_stats = {k: [int(x) for x in v] for k, v in info["network_stats"].items()}
        if network_trace:
            tick_trace = info.get("network_tick_trace")
            if isinstance(tick_trace, dict):
                network_tick_traces.append(tick_trace)

        done = any(bool(terminated[f"agent_{i}"]) or bool(truncated[f"agent_{i}"]) for i in range(n_agents))
        if done:
            end = t + 1
            states = states[: end + 1]
            estimates = estimates[: end + 1]
            actions = actions[:end]
            rewards = rewards[:end]
            state_errors = state_errors[: end + 1]
            throughputs = throughputs[: end + 1]
            collided_packets = collided_packets[: end + 1]
            break

    return {
        "states": states,
        "estimates": estimates,
        "actions": actions,
        "rewards": rewards,
        "state_errors": state_errors,
        "throughput_kbps": throughputs,
        "collided_packets": collided_packets,
        "network_stats": network_stats,
        "timesteps": np.arange(states.shape[0]),
        "network_tick_traces": network_tick_traces,
    }


def _configure_network_trace_env(
    env: NCS_Env,
    *,
    enabled: bool,
    trace_interval: int,
    trace_start: int,
) -> None:
    env.network_trace_enabled = bool(enabled)
    env.network_trace_interval = int(trace_interval)
    env.network_trace_start = int(trace_start)
    env.network.trace_enabled = bool(enabled)


def _write_network_trace_jsonl(traces: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for trace in traces:
            f.write(json.dumps(trace))
            f.write("\n")


def plot_network_tick_timeline(trace: Dict[str, Any], save_path: Path, title: str) -> None:
    slots = trace.get("slots", [])
    if not slots:
        return
    entity_labels = [str(label) for label in trace.get("entity_labels", [])]
    n_entities = len(entity_labels)
    n_slots = len(slots)
    state_map = {
        "IDLE": 0,
        "BACKING_OFF": 1,
        "CCA": 2,
        "TRANSMITTING": 3,
        "WAITING_ACK": 4,
    }
    state_colors = ["#f0f0f0", "#ffd166", "#f4a261", "#2a9d8f", "#457b9d"]
    data = np.zeros((n_entities, n_slots), dtype=np.int64)
    for slot_idx, slot in enumerate(slots):
        states = slot.get("entity_states", [])
        for ent_idx in range(min(n_entities, len(states))):
            state_name = str(states[ent_idx])
            data[ent_idx, slot_idx] = state_map.get(state_name, 0)

    fig, ax = plt.subplots(1, 1, figsize=(max(8, n_slots * 0.35), max(4, n_entities * 0.35)))
    cmap = ListedColormap(state_colors[: len(state_map)])
    ax.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=len(state_map) - 1)
    ax.set_xlabel("Micro-slot", fontsize=11)
    ax.set_ylabel("Entity", fontsize=11)
    ax.set_xticks(np.arange(n_slots))
    ax.set_yticks(np.arange(n_entities))
    ax.set_yticklabels(entity_labels, fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(np.arange(-0.5, n_slots, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_entities, 1), minor=True)
    ax.grid(which="minor", color="#e0e0e0", linestyle="-", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    event_types_present: set[str] = set()
    for slot_idx, slot in enumerate(slots):
        events = slot.get("events", [])
        for event in events:
            event_type = event.get("type")
            if event_type == "collision":
                entities = event.get("entities", [])
                for ent_idx in entities:
                    ax.scatter(slot_idx, ent_idx, marker="x", color="#e63946", s=50, zorder=3)
                event_types_present.add("collision")
                continue
            entity_idx = event.get("entity_idx")
            if entity_idx is None:
                continue
            if event_type == "backoff_draw":
                units = int(event.get("backoff_units", event.get("backoff_slots", 0)))
                ax.scatter(slot_idx, entity_idx, marker="v", color="#111111", s=35, zorder=3)
                ax.text(
                    slot_idx + 0.15,
                    entity_idx + 0.15,
                    f"{units}",
                    fontsize=7,
                    color="#111111",
                    zorder=4,
                )
                event_types_present.add("backoff_draw")
            elif event_type == "retry":
                ax.scatter(slot_idx, entity_idx, marker="o", color="#f4a261", s=45, zorder=3)
                event_types_present.add("retry")
            elif event_type == "drop":
                ax.scatter(slot_idx, entity_idx, marker="X", color="#e63946", s=55, zorder=3)
                event_types_present.add("drop")
            elif event_type == "ack_timeout":
                ax.scatter(slot_idx, entity_idx, marker="D", color="#264653", s=40, zorder=3)
                event_types_present.add("ack_timeout")
            elif event_type == "tx_start" and bool(event.get("is_mac_ack", False)):
                ax.scatter(slot_idx, entity_idx, marker="+", color="#457b9d", s=55, zorder=3)
                event_types_present.add("mac_ack_start")

    legend_handles = [
        Patch(color=state_colors[state_map["IDLE"]], label="IDLE"),
        Patch(color=state_colors[state_map["BACKING_OFF"]], label="BACKING_OFF"),
        Patch(color=state_colors[state_map["CCA"]], label="CCA"),
        Patch(color=state_colors[state_map["TRANSMITTING"]], label="TRANSMITTING"),
        Patch(color=state_colors[state_map["WAITING_ACK"]], label="WAITING_ACK"),
    ]
    event_handles: List[Line2D] = []
    if "backoff_draw" in event_types_present:
        event_handles.append(
            Line2D([0], [0], marker="v", color="none", markerfacecolor="#111111",
                   markeredgecolor="#111111", linestyle="None", label="Backoff draw (units)")
        )
    if "collision" in event_types_present:
        event_handles.append(
            Line2D([0], [0], marker="x", color="#e63946", linestyle="None", label="Collision")
        )
    if "retry" in event_types_present:
        event_handles.append(
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#f4a261",
                   markeredgecolor="#f4a261", linestyle="None", label="Retry")
        )
    if "drop" in event_types_present:
        event_handles.append(
            Line2D([0], [0], marker="X", color="#e63946", linestyle="None", label="Drop")
        )
    if "ack_timeout" in event_types_present:
        event_handles.append(
            Line2D([0], [0], marker="D", color="none", markerfacecolor="#264653",
                   markeredgecolor="#264653", linestyle="None", label="ACK timeout")
        )
    if "mac_ack_start" in event_types_present:
        event_handles.append(
            Line2D([0], [0], marker="+", color="#457b9d", linestyle="None", label="MAC ACK start")
        )

    ax.legend(
        handles=legend_handles + event_handles,
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=False,
    )
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved network tick timeline to {save_path}")


def _slice_marl_agent_trajectory(traj: Dict[str, np.ndarray], agent_index: int) -> Dict[str, np.ndarray]:
    states = traj["states"][:, agent_index, :]
    estimates = traj["estimates"][:, agent_index, :]
    actions = traj["actions"][:, agent_index]
    rewards = traj["rewards"][:, agent_index]
    state_errors = traj["state_errors"][:, agent_index]
    controls = np.zeros((actions.shape[0], 1), dtype=np.float32)
    return {
        "states": states,
        "estimates": estimates,
        "actions": actions,
        "rewards": rewards,
        "controls": controls,
        "state_errors": state_errors,
        "timesteps": np.arange(states.shape[0]),
    }


def plot_marl_action_raster(
    traj: Dict[str, np.ndarray],
    label: str,
    save_path: Path,
    title: str = "Coordination (Actions)",
) -> None:
    actions = np.asarray(traj["actions"], dtype=np.int64)
    n_steps, n_agents = actions.shape

    fig, ax = plt.subplots(1, 1, figsize=(14, 3 + 0.3 * n_agents))
    im = ax.imshow(actions.T, aspect="auto", interpolation="nearest", cmap="Greys", vmin=0, vmax=1)
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Agent", fontsize=12)
    ax.set_yticks(np.arange(n_agents))
    ax.set_yticklabels([f"agent_{i}" for i in range(n_agents)])
    ax.set_title(f"{title} - {label}", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, ticks=[0, 1], label="Action (0/1)")
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved action raster to {save_path}")


def plot_marl_state_space_2d(
    traj: Dict[str, np.ndarray],
    label: str,
    save_path: Path,
    title: str = "Multi-Agent State Space (2D)",
    show_estimates: bool = True,
) -> None:
    states = np.asarray(traj["states"], dtype=np.float32)
    estimates = np.asarray(traj["estimates"], dtype=np.float32)
    n_steps, n_agents, state_dim = states.shape
    if state_dim < 2:
        raise ValueError("2D state space plot requires state_dim >= 2")

    colors = plt.cm.tab10(np.linspace(0, 1, n_agents))
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for agent_idx in range(n_agents):
        ax.plot(
            states[:, agent_idx, 0],
            states[:, agent_idx, 1],
            "-",
            color=colors[agent_idx],
            linewidth=2,
            alpha=0.8,
            label=f"agent_{agent_idx} (state)",
        )
        ax.plot(
            states[0, agent_idx, 0],
            states[0, agent_idx, 1],
            "o",
            color=colors[agent_idx],
            markersize=8,
            alpha=0.9,
        )
        if show_estimates:
            ax.plot(
                estimates[:, agent_idx, 0],
                estimates[:, agent_idx, 1],
                "--",
                color=colors[agent_idx],
                linewidth=1.5,
                alpha=0.5,
                label=f"agent_{agent_idx} (est.)",
            )

    ax.plot(0, 0, "r*", markersize=18, label="Target (origin)")
    ax.set_xlabel("State Dimension 1", fontsize=12)
    ax.set_ylabel("State Dimension 2", fontsize=12)
    ax.set_title(f"{title} - {label}", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    ax.axis("equal")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved multi-agent state space plot to {save_path}")


def plot_marl_comparison_summary(
    trajectories: List[Dict[str, np.ndarray]],
    labels: List[str],
    save_path: Path,
    title: str = "MARL Summary",
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))

    ax1: Axes = axes[0]
    ax2: Axes = axes[1]
    for idx, (traj, label) in enumerate(zip(trajectories, labels)):
        state_errors = np.asarray(traj["state_errors"], dtype=np.float32)
        timesteps = np.asarray(traj["timesteps"], dtype=np.int64)
        mean_error = state_errors.mean(axis=1)
        ax1.plot(timesteps, mean_error, color=colors[idx], linewidth=2, label=label)

        actions = np.asarray(traj["actions"], dtype=np.float32)
        mean_action_rate = actions.mean(axis=1)
        ax2.plot(np.arange(mean_action_rate.shape[0]), mean_action_rate, color=colors[idx], linewidth=2, label=label)

    ax1.set_ylabel("Mean ||x|| across agents", fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc="best")

    ax2.set_xlabel("Timestep", fontsize=12)
    ax2.set_ylabel("Mean action rate", fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved MARL summary plot to {save_path}")


def create_marl_state_evolution_animation(
    traj: Dict[str, np.ndarray],
    label: str,
    save_path: Path,
    title: str = "MARL State Evolution (All Agents)",
    fps: int = 30,
    speedup: int = 1,
    show_estimates: bool = True,
) -> None:
    states = np.asarray(traj["states"], dtype=np.float32)
    estimates = np.asarray(traj["estimates"], dtype=np.float32)
    actions = np.asarray(traj["actions"], dtype=np.int64)
    n_steps, n_agents, state_dim = states.shape
    if state_dim < 2:
        raise ValueError("MARL animation requires state_dim >= 2")

    colors = plt.cm.tab10(np.linspace(0, 1, n_agents))
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    all_points = states.reshape(-1, state_dim)
    if show_estimates:
        all_points = np.vstack([all_points, estimates.reshape(-1, state_dim)])
    x_min, x_max = float(all_points[:, 0].min()), float(all_points[:, 0].max())
    y_min, y_max = float(all_points[:, 1].min()), float(all_points[:, 1].max())
    x_range = max(0.1, x_max - x_min)
    y_range = max(0.1, y_max - y_min)
    margin = 0.15
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    max_range = max(x_range, y_range) * (1 + 2 * margin)
    ax.set_xlim(x_center - max_range / 2, x_center + max_range / 2)
    ax.set_ylim(y_center - max_range / 2, y_center + max_range / 2)

    state_lines: List[Any] = []
    state_points: List[Any] = []
    estimate_lines: List[Any] = []
    estimate_points: List[Any] = []

    for agent_idx in range(n_agents):
        line, = ax.plot([], [], "-", color=colors[agent_idx], linewidth=2.5, alpha=0.85, label=f"agent_{agent_idx}")
        point, = ax.plot([], [], "o", color=colors[agent_idx], markersize=10, zorder=5)
        state_lines.append(line)
        state_points.append(point)
        if show_estimates:
            est_line, = ax.plot([], [], "--", color=colors[agent_idx], linewidth=2, alpha=0.5)
            est_point, = ax.plot([], [], "s", color=colors[agent_idx], markersize=8, alpha=0.7, zorder=5)
            estimate_lines.append(est_line)
            estimate_points.append(est_point)

    ax.plot(0, 0, "r*", markersize=18, label="Target", zorder=10)
    ax.set_xlabel("State Dimension 1", fontsize=14, fontweight="bold")
    ax.set_ylabel("State Dimension 2", fontsize=14, fontweight="bold")
    ax.set_title(f"{title} - {label}", fontsize=16, fontweight="bold", pad=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc="best", framealpha=0.9)
    ax.set_aspect("equal", adjustable="box")

    time_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9),
    )

    def init():
        for line in state_lines + estimate_lines:
            line.set_data([], [])
        for point in state_points + estimate_points:
            point.set_data([], [])
        time_text.set_text("")
        return state_lines + estimate_lines + state_points + estimate_points + [time_text]

    def animate(frame: int):
        idx = min(frame * speedup, n_steps - 1)
        if idx < actions.shape[0]:
            sending = [str(i) for i in np.flatnonzero(actions[idx] > 0)]
        else:
            sending = []
        time_text.set_text(f"Timestep: {idx} | sending: [{', '.join(sending)}]")

        for agent_idx in range(n_agents):
            state_lines[agent_idx].set_data(states[: idx + 1, agent_idx, 0], states[: idx + 1, agent_idx, 1])
            state_points[agent_idx].set_data([states[idx, agent_idx, 0]], [states[idx, agent_idx, 1]])
            if show_estimates:
                estimate_lines[agent_idx].set_data(
                    estimates[: idx + 1, agent_idx, 0], estimates[: idx + 1, agent_idx, 1]
                )
                estimate_points[agent_idx].set_data([estimates[idx, agent_idx, 0]], [estimates[idx, agent_idx, 1]])

        artists: List[Any] = []
        artists.extend(state_lines)
        artists.extend(state_points)
        if show_estimates:
            artists.extend(estimate_lines)
            artists.extend(estimate_points)
        artists.append(time_text)
        return artists

    num_frames = (n_steps - 1) // speedup + 1
    anim = FuncAnimation(fig, animate, init_func=init, frames=num_frames, interval=1000 / fps, blit=True, repeat=True)

    try:
        writer = FFMpegWriter(fps=fps, bitrate=2400, codec="libx264")
        anim.save(str(save_path), writer=writer)
        print(f"  ✓ Saved MARL animation to {save_path}")
    except Exception as e:
        print(f"  ✗ Error saving MARL animation: {e}")
        print(f"  Make sure FFmpeg is installed:")
        print(f"    - Linux: sudo apt-get install ffmpeg")
        print(f"    - Mac: brew install ffmpeg")
    finally:
        plt.close(fig)


def create_marl_agent_state_evolution_animation(
    traj: Dict[str, np.ndarray],
    label: str,
    agent_index: int,
    save_path: Path,
    title: str = "Agent State Evolution",
    fps: int = 30,
    speedup: int = 1,
    show_estimates: bool = True,
) -> None:
    sliced = _slice_marl_agent_trajectory(traj, agent_index)
    states = np.asarray(sliced["states"], dtype=np.float32)
    estimates = np.asarray(sliced["estimates"], dtype=np.float32)
    actions = np.asarray(sliced["actions"], dtype=np.int64)
    n_steps, state_dim = states.shape
    if state_dim < 2:
        raise ValueError("Agent animation requires state_dim >= 2")

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    all_points = states
    if show_estimates:
        all_points = np.vstack([all_points, estimates])
    x_min, x_max = float(all_points[:, 0].min()), float(all_points[:, 0].max())
    y_min, y_max = float(all_points[:, 1].min()), float(all_points[:, 1].max())
    x_range = max(0.1, x_max - x_min)
    y_range = max(0.1, y_max - y_min)
    margin = 0.15
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    max_range = max(x_range, y_range) * (1 + 2 * margin)
    ax.set_xlim(x_center - max_range / 2, x_center + max_range / 2)
    ax.set_ylim(y_center - max_range / 2, y_center + max_range / 2)

    state_line, = ax.plot([], [], "-", color="tab:blue", linewidth=2.5, alpha=0.85, label="state")
    state_point, = ax.plot([], [], "o", color="tab:blue", markersize=10, zorder=5)
    if show_estimates:
        est_line, = ax.plot([], [], "--", color="tab:orange", linewidth=2, alpha=0.6, label="estimate")
        est_point, = ax.plot([], [], "s", color="tab:orange", markersize=8, alpha=0.7, zorder=5)
    else:
        est_line, est_point = None, None

    ax.plot(0, 0, "r*", markersize=18, label="Target", zorder=10)
    ax.set_xlabel("State Dimension 1", fontsize=14, fontweight="bold")
    ax.set_ylabel("State Dimension 2", fontsize=14, fontweight="bold")
    ax.set_title(f"{title} - {label} - agent_{agent_index}", fontsize=16, fontweight="bold", pad=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc="best", framealpha=0.9)
    ax.set_aspect("equal", adjustable="box")

    time_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9),
    )

    def init():
        state_line.set_data([], [])
        state_point.set_data([], [])
        if show_estimates and est_line is not None and est_point is not None:
            est_line.set_data([], [])
            est_point.set_data([], [])
        time_text.set_text("")
        artists: List[Any] = [state_line, state_point, time_text]
        if show_estimates and est_line is not None and est_point is not None:
            artists.extend([est_line, est_point])
        return artists

    def animate(frame: int):
        idx = min(frame * speedup, n_steps - 1)
        action_val = int(actions[idx]) if idx < actions.shape[0] else 0
        time_text.set_text(f"Timestep: {idx} | action: {action_val}")
        state_line.set_data(states[: idx + 1, 0], states[: idx + 1, 1])
        state_point.set_data([states[idx, 0]], [states[idx, 1]])
        artists: List[Any] = [state_line, state_point, time_text]
        if show_estimates and est_line is not None and est_point is not None:
            est_line.set_data(estimates[: idx + 1, 0], estimates[: idx + 1, 1])
            est_point.set_data([estimates[idx, 0]], [estimates[idx, 1]])
            artists.extend([est_line, est_point])
        return artists

    num_frames = (n_steps - 1) // speedup + 1
    anim = FuncAnimation(fig, animate, init_func=init, frames=num_frames, interval=1000 / fps, blit=True, repeat=True)

    try:
        writer = FFMpegWriter(fps=fps, bitrate=2400, codec="libx264")
        anim.save(str(save_path), writer=writer)
        print(f"  ✓ Saved agent animation to {save_path}")
    except Exception as e:
        print(f"  ✗ Error saving agent animation: {e}")
        print(f"  Make sure FFmpeg is installed:")
        print(f"    - Linux: sudo apt-get install ffmpeg")
        print(f"    - Mac: brew install ffmpeg")
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize policy state evolution in NCS simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config JSON file'
    )

    # Single policy mode
    parser.add_argument(
        '--policy',
        type=str,
        help='Path to policy file (ES/MARL torch) or policy name (heuristic)'
    )
    parser.add_argument(
        '--policy-type',
        type=str,
        choices=['es', 'openai_es', 'heuristic', 'marl_torch'],
        help='Type of policy: es, openai_es, heuristic, or marl_torch'
    )

    # Multi-policy comparison mode
    parser.add_argument(
        '--policies',
        type=str,
        nargs='+',
        help='List of policies to compare'
    )
    parser.add_argument(
        '--policy-types',
        type=str,
        nargs='+',
        choices=['es', 'openai_es', 'heuristic', 'marl_torch'],
        help='List of policy types corresponding to --policies'
    )
    parser.add_argument(
        '--per-agent-videos',
        action='store_true',
        help='With --generate-video for multi-agent policies, also write one MP4 per agent'
    )
    parser.add_argument(
        '--labels',
        type=str,
        nargs='+',
        help='Labels for policies in plots (optional)'
    )

    # Simulation parameters
    parser.add_argument(
        '--episode-length',
        type=int,
        default=500,
        help='Length of episode to simulate (default: 500)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--n-agents',
        type=int,
        default=None,
        help='Optional override for agent count (default: read from checkpoint or config)'
    )

    # Visualization parameters
    parser.add_argument(
        '--show-estimates',
        action='store_true',
        default=True,
        help='Show controller estimates in plots (default: True)'
    )
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='Force deterministic policy actions (default: False)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='vis',
        help='Directory to save visualizations (default: vis/)'
    )
    parser.add_argument(
        '--output-prefix',
        type=str,
        default='',
        help='Prefix for output files (default: timestamp)'
    )

    # Video/Animation parameters
    parser.add_argument(
        '--generate-video',
        action='store_true',
        help='Generate MP4 animation video of state evolution (requires FFmpeg)'
    )
    parser.add_argument(
        '--video-fps',
        type=int,
        default=30,
        help='Frames per second for video (default: 30)'
    )
    parser.add_argument(
        '--video-speedup',
        type=int,
        default=1,
        help='Speed multiplier for video (e.g., 2 = 2x speed, default: 1)'
    )

    # Network trace parameters
    parser.add_argument(
        '--network-trace',
        action='store_true',
        help='Trace micro-slot network activity and render per-tick timelines'
    )
    parser.add_argument(
        '--trace-interval',
        type=int,
        default=50,
        help='Tick interval for network traces (default: 50)'
    )
    parser.add_argument(
        '--trace-start',
        type=int,
        default=1,
        help='First tick index to trace (default: 1)'
    )

    # List heuristic policies
    parser.add_argument(
        '--list-heuristics',
        action='store_true',
        help='List available heuristic policies and exit'
    )

    # Check for list-heuristics flag first (before full parsing)
    if '--list-heuristics' in sys.argv:
        print("\nAvailable heuristic policies:")
        print("-" * 50)
        for name in sorted(HEURISTIC_POLICIES.keys()):
            print(f"  - {name}")
        print()
        return

    args = parser.parse_args()

    # Validate arguments
    if args.policy is None and args.policies is None:
        parser.error("Either --policy or --policies must be specified")

    if args.policy is not None and args.policy_type is None:
        parser.error("--policy-type must be specified when using --policy")

    if args.policies is not None:
        if args.policy_types is None:
            parser.error("--policy-types must be specified when using --policies")
        if len(args.policies) != len(args.policy_types):
            parser.error("--policies and --policy-types must have the same length")

    # Setup single or multi-policy mode
    if args.policy is not None:
        policies_to_load = [args.policy]
        policy_types = [args.policy_type]
        policy_labels = args.labels if args.labels else [args.policy]
    else:
        policies_to_load = args.policies
        policy_types = args.policy_types
        policy_labels = args.labels if args.labels else [f"Policy {i+1}" for i in range(len(policies_to_load))]

    if len(policy_labels) != len(policies_to_load):
        parser.error("--labels must have the same length as number of policies")

    policy_types_norm = [policy_type.lower() for policy_type in policy_types]

    print(f"\n{'='*60}")
    print(f"NCS Policy Visualization Tool")
    print(f"{'='*60}\n")

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    reward_override = config.get("reward", {}).get("evaluation")
    if not isinstance(reward_override, dict):
        reward_override = None
    termination_override = config.get("termination", {}).get("evaluation")
    if not isinstance(termination_override, dict):
        termination_override = None
    print(f"✓ Configuration loaded")
    if reward_override is not None or termination_override is not None:
        print("✓ Using evaluation reward/termination overrides")
    print()

    allowed_policy_types = {"marl_torch", "heuristic", "es", "openai_es"}
    if not all(policy_type in allowed_policy_types for policy_type in policy_types_norm):
        parser.error("Supported policy types: marl_torch, es, openai_es, heuristic")

    try:
        resolved_n_agents = _resolve_run_n_agents(
            list(policies_to_load),
            list(policy_types_norm),
            config=config,
            explicit_n_agents=args.n_agents,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if resolved_n_agents < 1:
        parser.error("Resolved n_agents must be >= 1")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output prefix if not provided
    if not args.output_prefix:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_prefix = f"viz_{timestamp}"
    else:
        output_prefix = args.output_prefix

    # Create environment factory function
    def make_env():
        return NCS_Env(
            n_agents=resolved_n_agents,
            episode_length=args.episode_length,
            config_path=args.config,
            seed=args.seed,
            reward_override=reward_override,
            termination_override=termination_override,
            track_true_goodput=True,
        )

    print(f"Creating environment...")
    temp_env = make_env()
    state_dim = int(temp_env.state_dim)
    temp_env.close()
    print(
        f"✓ Environment created (multi-agent, n_agents={resolved_n_agents}, "
        f"state_dim={state_dim}, episode_length={args.episode_length})\n"
    )

    # Load policies and run episodes
    trajectories = []
    print(f"Running simulations for {len(policies_to_load)} policy/policies...\n")

    for i, (policy_path, policy_type, label) in enumerate(zip(policies_to_load, policy_types, policy_labels)):
        print(f"[{i+1}/{len(policies_to_load)}] {label}")
        print(f"  Type: {policy_type}")
        print(f"  Path/Name: {policy_path}")

        env: Any = make_env()
        try:
            policy = load_multi_agent_policy(
                policy_path,
                policy_type.lower(),
                env,
                n_agents=resolved_n_agents,
                seed=args.seed,
                deterministic=args.deterministic,
            )
        except Exception as e:
            env.close()
            print(f"  ✗ Error loading policy: {e}\n")
            continue
        print("  Running multi-agent episode...")
        traj = run_episode_multi_agent(
            env,
            policy,
            args.episode_length,
            seed=args.seed,
            network_trace=args.network_trace,
            trace_interval=args.trace_interval,
            trace_start=args.trace_start,
        )
        env.close()

        # Print summary statistics
        total_reward = float(np.sum(traj["rewards"]))
        tx_count = float(np.sum(traj["actions"]))
        final_error = float(np.mean(traj["state_errors"][-1]))
        print(f"  ✓ Episode complete")
        print(f"    - Total reward: {total_reward:.2f}")
        denom = args.episode_length * resolved_n_agents
        print(f"    - Transmissions: {int(tx_count)}/{denom}")
        print(f"    - Final state error: {final_error:.4f}\n")

        trajectories.append(traj)
        if args.network_trace:
            tick_traces = traj.get("network_tick_traces", [])
            if tick_traces:
                tag = f"p{i+1}_{_sanitize_filename(label)}"
                trace_path = output_dir / f"{output_prefix}_{tag}_network_trace.jsonl"
                _write_network_trace_jsonl(tick_traces, trace_path)
                print(f"    - Saved network trace: {trace_path}")
                for trace in tick_traces:
                    tick = trace.get("tick", 0)
                    plot_path = output_dir / f"{output_prefix}_{tag}_network_tick_{tick}.png"
                    plot_network_tick_timeline(
                        trace,
                        plot_path,
                        title=f"Network Tick {tick} - {label}",
                    )

    if len(trajectories) == 0:
        print("No successful policy runs. Exiting.")
        return

    # Generate plots
    print(f"Generating visualizations...\n")

    summary_plot_path = output_dir / f"{output_prefix}_marl_summary.png"
    plot_marl_comparison_summary(
        trajectories,
        policy_labels[:len(trajectories)],
        summary_plot_path,
        title="MARL Summary (Mean state error + mean action rate)",
    )

    for idx, (label, traj) in enumerate(zip(policy_labels[:len(trajectories)], trajectories)):
        tag = f"p{idx+1}_{_sanitize_filename(label)}"
        raster_path = output_dir / f"{output_prefix}_{tag}_actions.png"
        plot_marl_action_raster(traj, label=label, save_path=raster_path)
        if state_dim >= 2:
            state_space_path = output_dir / f"{output_prefix}_{tag}_state_space.png"
            plot_marl_state_space_2d(
                traj,
                label=label,
                save_path=state_space_path,
                show_estimates=args.show_estimates,
            )

    # Generate video animation if requested
    if args.generate_video:
        print(f"\nGenerating animation...\n")
        for idx, (label, traj) in enumerate(zip(policy_labels[:len(trajectories)], trajectories)):
            tag = f"p{idx+1}_{_sanitize_filename(label)}"
            video_path = output_dir / f"{output_prefix}_{tag}_animation.mp4"
            create_marl_state_evolution_animation(
                traj,
                label=label,
                save_path=video_path,
                title="MARL State Evolution (All Agents)",
                fps=args.video_fps,
                speedup=args.video_speedup,
                show_estimates=args.show_estimates,
            )
            if args.per_agent_videos:
                n_agents = int(traj["states"].shape[1])
                for agent_idx in range(n_agents):
                    agent_video_path = output_dir / f"{output_prefix}_{tag}_agent_{agent_idx}.mp4"
                    create_marl_agent_state_evolution_animation(
                        traj,
                        label=label,
                        agent_index=agent_idx,
                        save_path=agent_video_path,
                        title="Agent State Evolution",
                        fps=args.video_fps,
                        speedup=args.video_speedup,
                        show_estimates=args.show_estimates,
                    )

    # Save summary statistics to JSON
    summary_path = output_dir / f"{output_prefix}_summary.json"
    summary = {
        'config': args.config,
        'episode_length': args.episode_length,
        'seed': args.seed,
        'policies': []
    }

    for label, traj in zip(policy_labels[:len(trajectories)], trajectories):
        rewards = np.asarray(traj["rewards"], dtype=np.float32)
        actions = np.asarray(traj["actions"], dtype=np.float32)
        state_errors = np.asarray(traj["state_errors"], dtype=np.float32)

        network_stats = traj.get("network_stats", {})
        tx_attempts = network_stats.get("tx_attempts", [])
        tx_acked = network_stats.get("tx_acked", [])
        tx_dropped = network_stats.get("tx_dropped", [])
        tx_rewrites = network_stats.get("tx_rewrites", [])
        tx_collisions = network_stats.get("tx_collisions", [])
        policy_summary = {
            "label": label,
            "n_agents": int(state_errors.shape[1]),
            "total_reward": float(rewards.sum()),
            "avg_reward_per_agent_step": float(rewards.mean()),
            "transmission_count": int(actions.sum()),
            "transmission_rate_per_agent_step": float(actions.mean()),
            "network_stats": {
                "tx_attempts": [int(x) for x in tx_attempts],
                "tx_acked": [int(x) for x in tx_acked],
                "tx_dropped": [int(x) for x in tx_dropped],
                "tx_rewrites": [int(x) for x in tx_rewrites],
                "tx_collisions": [int(x) for x in tx_collisions],
                "tx_attempts_total": int(np.sum(tx_attempts)) if tx_attempts else 0,
                "tx_acked_total": int(np.sum(tx_acked)) if tx_acked else 0,
                "tx_dropped_total": int(np.sum(tx_dropped)) if tx_dropped else 0,
                "tx_rewrites_total": int(np.sum(tx_rewrites)) if tx_rewrites else 0,
                "tx_collisions_total": int(np.sum(tx_collisions)) if tx_collisions else 0,
            },
            "initial_state_error_mean": float(state_errors[0].mean()),
            "final_state_error_mean": float(state_errors[-1].mean()),
            "avg_state_error_mean": float(state_errors.mean(axis=1).mean()),
            "initial_state_error_per_agent": [float(x) for x in state_errors[0]],
            "final_state_error_per_agent": [float(x) for x in state_errors[-1]],
        }
        summary['policies'].append(policy_summary)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary statistics to {summary_path}")

    print(f"\n{'='*60}")
    print(f"Visualization complete!")
    print(f"Output files saved to: {output_dir}/")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
