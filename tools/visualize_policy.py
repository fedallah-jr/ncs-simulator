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
from tools.heuristic_policies import HEURISTIC_POLICIES
from tools._common import (
    MultiAgentHeuristicPolicy,
    load_es_policy,
    load_marl_torch_multi_agent_policy,
    load_multi_agent_policy,
    resolve_n_agents,
    sanitize_filename as _sanitize_filename,
    parse_set_overrides,
    deep_merge,
    apply_config_overrides,
    add_set_override_argument,
)

# Modes where the reward metric comes from kf_info_gain (stored as negative,
# negated here so the plot shows a positive "cost" / "uncertainty" quantity).
_INFO_GAIN_MODES: Dict[str, str] = {
    "kf_info_m_noise": r"$e^\top M e$",
    "kf_info_m": r"$\mathrm{tr}(M P)$",
    "kf_info_s": r"$\mathrm{tr}(S P)$",
    "kf_info": r"$\mathrm{tr}((M{+}S) P)$",
    "lqr_cost_immediate_surrogate": r"$e^\top M_{\mathrm{surr}} e$",
    "kf_q": r"$\mathrm{tr}(Q P)$",
    "kf_q_noise": r"$e^\top Q e$",
}

# Modes where the reward metric comes from curr_error (already positive).
_ERROR_MODES: Dict[str, str] = {
    "absolute": r"$\|x\|$",
    "lqr_cost": r"LQR cost",
    "estimation_error": r"Estimation error",
}


def _reward_metric_info(mode: str):
    """Return (component_key, negate, label) for the given reward mode."""
    if mode in _INFO_GAIN_MODES:
        return "kf_info_gain", True, _INFO_GAIN_MODES[mode]
    if mode in _ERROR_MODES:
        return "curr_error", False, _ERROR_MODES[mode]
    # Fallback: try kf_info_gain
    return "kf_info_gain", True, mode


def run_episode_multi_agent(
    env: NCS_Env,
    policy: Any,
    episode_length: int,
    *,
    seed: Optional[int] = None,
    network_trace: bool = False,
    trace_interval: int = 50,
    trace_start: int = 1,
    epsilon: float = 0.0,
    rng: Optional[np.random.Generator] = None,
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

    reward_mode = str(env.reward_definition.mode)
    metric_component_key, metric_negate, reward_metric_label = _reward_metric_info(reward_mode)

    states = np.zeros((episode_length + 1, n_agents, state_dim), dtype=np.float32)
    estimates = np.zeros((episode_length + 1, n_agents, state_dim), dtype=np.float32)
    estimate_covariances = np.zeros((episode_length + 1, n_agents, state_dim, state_dim), dtype=np.float32)
    actions = np.zeros((episode_length, n_agents), dtype=np.int64)
    drop_delay_steps = np.full((episode_length, n_agents), np.nan, dtype=np.float32)
    reward_metric = np.full((episode_length, n_agents), np.nan, dtype=np.float32)
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
    estimate_covariances[0] = np.asarray(info.get("estimate_covariances", estimate_covariances[0]), dtype=np.float32)
    state_errors[0] = np.linalg.norm(states[0], axis=1).astype(np.float32)
    throughputs[0] = float(info.get("true_goodput_kbps_total", info.get("throughput_kbps", 0.0)))
    collided_packets[0] = float(info.get("collided_packets", 0.0))
    if "network_stats" in info:
        network_stats = {k: [int(x) for x in v] for k, v in info["network_stats"].items()}

    for t in range(episode_length):
        action_dict = policy.act(obs_dict)
        if epsilon > 0:
            if rng is None:
                rng = np.random.default_rng()
            for key in action_dict:
                if rng.random() < epsilon:
                    action_dict[key] = rng.integers(0, 2)
        actions[t] = np.asarray([action_dict[f"agent_{i}"] for i in range(n_agents)], dtype=np.int64)
        obs_dict, rewards_dict, terminated, truncated, info = env.step(action_dict)
        rewards[t] = np.asarray([rewards_dict[f"agent_{i}"] for i in range(n_agents)], dtype=np.float32)
        reward_components = info.get("reward_components", {})
        if isinstance(reward_components, dict):
            for i in range(n_agents):
                agent_components = reward_components.get(f"agent_{i}", {})
                if not isinstance(agent_components, dict):
                    continue
                metric_value = agent_components.get(metric_component_key, np.nan)
                try:
                    raw = float(metric_value)
                    reward_metric[t, i] = -raw if metric_negate else raw
                except (TypeError, ValueError):
                    reward_metric[t, i] = np.nan

        states[t + 1] = np.asarray(info.get("states", states[t + 1]), dtype=np.float32)
        estimates[t + 1] = np.asarray(info.get("estimates", estimates[t + 1]), dtype=np.float32)
        estimate_covariances[t + 1] = np.asarray(info.get("estimate_covariances", estimate_covariances[t + 1]), dtype=np.float32)
        state_errors[t + 1] = np.linalg.norm(states[t + 1], axis=1).astype(np.float32)
        throughputs[t + 1] = float(info.get("true_goodput_kbps_total", info.get("throughput_kbps", 0.0)))
        collided_packets[t + 1] = float(info.get("collided_packets", 0.0))
        if "network_stats" in info:
            network_stats = {k: [int(x) for x in v] for k, v in info["network_stats"].items()}
        dropped_data_packets_step = info.get("dropped_data_packets_step", [])
        if isinstance(dropped_data_packets_step, list):
            for event in dropped_data_packets_step:
                if not isinstance(event, dict):
                    continue
                sensor_id = int(event.get("sensor_id", -1))
                measurement_timestamp = int(event.get("measurement_timestamp", -1))
                age_steps = float(event.get("age_steps", 0.0))
                if 0 <= sensor_id < n_agents and 0 <= measurement_timestamp < episode_length:
                    drop_delay_steps[measurement_timestamp, sensor_id] = max(0.0, age_steps)
        if network_trace:
            tick_trace = info.get("network_tick_trace")
            if isinstance(tick_trace, dict):
                network_tick_traces.append(tick_trace)

        done = any(bool(terminated[f"agent_{i}"]) or bool(truncated[f"agent_{i}"]) for i in range(n_agents))
        if done:
            end = t + 1
            states = states[: end + 1]
            estimates = estimates[: end + 1]
            estimate_covariances = estimate_covariances[: end + 1]
            actions = actions[:end]
            drop_delay_steps = drop_delay_steps[:end]
            reward_metric = reward_metric[:end]
            rewards = rewards[:end]
            state_errors = state_errors[: end + 1]
            throughputs = throughputs[: end + 1]
            collided_packets = collided_packets[: end + 1]
            break

    return {
        "states": states,
        "estimates": estimates,
        "estimate_covariances": estimate_covariances,
        "actions": actions,
        "drop_delay_steps": drop_delay_steps,
        "reward_metric": reward_metric,
        "reward_mode": reward_mode,
        "reward_metric_label": reward_metric_label,
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

    # Get drop-delay map and reward metric.
    drop_delay_steps = np.asarray(
        traj.get("drop_delay_steps", np.full(actions.shape, np.nan, dtype=np.float32)),
        dtype=np.float32,
    )
    reward_metric = np.asarray(
        traj.get("reward_metric", np.full(actions.shape, np.nan, dtype=np.float32)),
        dtype=np.float32,
    )
    reward_metric_label = str(traj.get("reward_metric_label", "Reward metric"))

    # Create figure with 2 subplots:
    # 1) Drop outcomes with drop-delay coloring
    # 2) Uncertainty metric growth as row-wise line plots (one row per agent)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14.5, 6.8 + 0.40 * n_agents),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.15]},
    )
    fig.patch.set_facecolor("white")

    # Plot 1: Drop outcomes (non-drops as category + drop delay heatmap)
    ax1 = axes[0]
    ax1.set_facecolor("white")
    sent_mask = actions > 0
    drop_mask = np.isfinite(drop_delay_steps)
    sent_not_dropped_mask = sent_mask & ~drop_mask
    base_outcomes = np.zeros((n_steps, n_agents), dtype=np.int64)
    base_outcomes[sent_not_dropped_mask] = 1
    base_outcomes_vis = base_outcomes.T
    base_cmap = ListedColormap(["#f5f5f5", "#457b9d"])
    ax1.imshow(base_outcomes_vis, aspect="auto", interpolation="nearest", cmap=base_cmap, vmin=0, vmax=1, origin="upper")

    drop_delay_vis = drop_delay_steps.T
    masked_drop_delay = np.ma.masked_invalid(drop_delay_vis)
    if masked_drop_delay.count() > 0:
        max_drop_delay = float(masked_drop_delay.max())
        if max_drop_delay <= 0.0:
            max_drop_delay = 1.0
        im1 = ax1.imshow(
            masked_drop_delay,
            aspect="auto",
            interpolation="nearest",
            cmap="YlOrRd",
            vmin=0.0,
            vmax=max_drop_delay,
            origin="upper",
        )
        cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.02, pad=0.02)
        cbar1.set_label("Drop delay (timesteps)", fontsize=10)
        cbar1.ax.tick_params(labelsize=9)
    else:
        ax1.text(
            0.5,
            0.5,
            "No dropped packets",
            transform=ax1.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="#444444",
        )

    ax1.set_ylabel("Agent", fontsize=12)
    ax1.set_yticks(np.arange(n_agents))
    ax1.set_yticklabels([f"agent_{i}" for i in range(n_agents)])
    ax1.tick_params(axis="both", labelsize=10)
    ax1.set_title(f"Drops (color = delay) - {label}", fontsize=14, fontweight="bold")
    ax1.legend(
        handles=[
            Patch(facecolor="#f5f5f5", edgecolor="#bdbdbd", label="No send"),
            Patch(facecolor="#457b9d", edgecolor="#457b9d", label="Sent, not dropped"),
            Patch(facecolor="#d7301f", edgecolor="#d7301f", label="Dropped (delay heatmap)"),
        ],
        loc="upper right",
        fontsize=9,
        frameon=True,
    )
    ax1.grid(False)

    # Plot 2: reward metric growth as row-wise line plots (one row per agent)
    ax2 = axes[1]
    ax2.set_facecolor("white")
    agent_colors = plt.cm.tab10(np.linspace(0, 1, max(n_agents, 1)))
    row_pad = 0.08
    row_bottoms = np.arange(n_agents, dtype=np.float64) + row_pad
    row_tops = np.arange(n_agents, dtype=np.float64) + (1.0 - row_pad)
    row_centers = 0.5 * (row_bottoms + row_tops)

    metric_matrix = np.where(np.isfinite(reward_metric), reward_metric, np.nan)
    global_scale_low = float("nan")
    global_scale_high = float("nan")
    peak_values = np.full((n_agents,), np.nan, dtype=np.float64)
    peak_timesteps = np.full((n_agents,), -1, dtype=np.int64)
    if np.all(np.isnan(metric_matrix)):
        ax2.text(
            0.5,
            0.5,
            f"No finite {reward_metric_label} values",
            transform=ax2.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="#444444",
        )
    else:
        x = np.arange(n_steps, dtype=np.float64)
        scale_start_idx = 1 if n_steps > 1 else 0
        finite_scale_source = metric_matrix[scale_start_idx:, :][np.isfinite(metric_matrix[scale_start_idx:, :])]
        if finite_scale_source.size == 0:
            finite_scale_source = metric_matrix[np.isfinite(metric_matrix)]
        global_scale_low, global_scale_high = np.percentile(finite_scale_source, [5.0, 95.0])
        if global_scale_high <= global_scale_low:
            global_scale_low = float(np.min(finite_scale_source))
            global_scale_high = float(np.max(finite_scale_source))
        global_scale_span = max(global_scale_high - global_scale_low, 1e-12)

        for i in range(n_agents):
            row_bottom = row_bottoms[i]
            row_top = row_tops[i]
            row_height = row_top - row_bottom
            vals = metric_matrix[:, i].astype(np.float64, copy=False)
            finite_vals = vals[np.isfinite(vals)]
            if finite_vals.size == 0:
                continue

            peak_local_idx = int(np.nanargmax(vals))
            peak_values[i] = float(vals[peak_local_idx])
            peak_timesteps[i] = peak_local_idx

            clipped_vals = np.clip(vals, global_scale_low, global_scale_high)
            scaled = (clipped_vals - global_scale_low) / global_scale_span
            row_curve = row_bottom + row_height * scaled
            row_curve[~np.isfinite(vals)] = np.nan

            ax2.plot(
                x,
                row_curve,
                color=agent_colors[i],
                linewidth=2.0,
                alpha=0.95,
                solid_capstyle="round",
            )
            ax2.fill_between(
                x,
                row_bottom,
                row_curve,
                where=np.isfinite(row_curve),
                color=agent_colors[i],
                alpha=0.16,
                linewidth=0.0,
            )
            if 0 <= peak_local_idx < n_steps and np.isfinite(row_curve[peak_local_idx]):
                ax2.scatter(
                    [float(peak_local_idx)],
                    [float(row_curve[peak_local_idx])],
                    s=26,
                    color=agent_colors[i],
                    edgecolors="#111111",
                    linewidths=0.6,
                    zorder=4,
                )

    for y in range(n_agents + 1):
        ax2.axhline(float(y), color="#d0d4d8", linewidth=0.9, zorder=0)

    if n_steps > 0:
        ax2.set_xlim(0, n_steps - 1)
    ax2.set_ylim(0, n_agents)
    ax2.set_xlabel("Timestep", fontsize=12)
    ax2.set_ylabel("Agent", fontsize=12)
    ax2.set_yticks(row_centers)
    ax2.set_yticklabels([f"agent_{i}" for i in range(n_agents)])
    ax2.tick_params(axis="both", labelsize=10)
    ax2.set_title(f"{reward_metric_label} - {label}", fontsize=15, fontweight="bold")
    ax2.invert_yaxis()

    # Right-side labels: one global scale (shared by all rows) + per-agent peak reports.
    right_scale_transform = ax2.get_yaxis_transform()
    for i in range(n_agents):
        if not np.isfinite(peak_values[i]):
            continue
        ax2.text(
            1.01,
            row_centers[i],
            f"peak {peak_values[i]:.3g} @ t={int(peak_timesteps[i])}",
            transform=right_scale_transform,
            ha="left",
            va="center",
            fontsize=9,
            color=agent_colors[i],
            clip_on=False,
        )
    if np.isfinite(global_scale_low) and np.isfinite(global_scale_high):
        ax2.text(
            1.01,
            float(n_agents) + 0.02,
            f"global clipped scale\n(top p95 {global_scale_high:.3g} / bot p5 {global_scale_low:.3g})",
            transform=right_scale_transform,
            ha="left",
            va="bottom",
            fontsize=9,
            color="#303030",
            fontweight="semibold",
            linespacing=1.1,
            clip_on=False,
        )

    ax2.grid(False)

    plt.tight_layout(rect=[0.0, 0.0, 0.88, 1.0])
    plt.savefig(str(save_path), dpi=400, bbox_inches="tight")
    plt.close(fig)
    peak_report_parts: List[str] = []
    for i in range(n_agents):
        if np.isfinite(peak_values[i]):
            peak_report_parts.append(f"agent_{i}: {peak_values[i]:.6g} @ t={int(peak_timesteps[i])}")
    if peak_report_parts:
        print(f"  Peaks {reward_metric_label}: {', '.join(peak_report_parts)}")
    print(f"✓ Saved coordination raster with drops and {reward_metric_label} to {save_path}")


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


def _resolve_two_policy_layout(
    trajectories: List[Dict[str, np.ndarray]],
    split_t: Optional[int] = None,
) -> tuple[int, int, int]:
    if len(trajectories) != 2:
        raise ValueError("Two-policy comparison requires exactly two trajectories")

    horizons: List[int] = []
    agent_counts: List[int] = []
    for traj in trajectories:
        actions_shape = np.asarray(traj["actions"], dtype=np.int64).shape
        if len(actions_shape) != 2 or actions_shape[0] <= 0:
            raise ValueError("Each trajectory must contain non-empty actions with shape (timesteps, n_agents)")
        horizons.append(int(actions_shape[0]))
        agent_counts.append(int(actions_shape[1]))

    if len(set(agent_counts)) != 1:
        raise ValueError("Two-policy comparison requires both trajectories to have the same number of agents")

    common_steps = min(horizons)
    if common_steps <= 1:
        raise ValueError("Need at least 2 timesteps for two-policy comparison")

    if split_t is None:
        split_t = estimate_two_policy_split_timestep(trajectories)
    split_t = int(np.clip(int(split_t), 1, common_steps - 1))
    return common_steps, agent_counts[0], split_t


def _extract_two_policy_metrics(
    trajectories: List[Dict[str, np.ndarray]],
    common_steps: int,
) -> tuple[List[np.ndarray], str]:
    metric_per_policy: List[np.ndarray] = []
    metric_label = str(trajectories[0].get("reward_metric_label", r"$\mathrm{tr}(M e e^\top)$"))
    for traj in trajectories:
        metric_full = np.asarray(traj.get("reward_metric", []), dtype=np.float32)
        if metric_full.ndim != 2:
            metric_full = np.full((common_steps, 1), np.nan, dtype=np.float32)
        metric_per_policy.append(metric_full[:common_steps])
    return metric_per_policy, metric_label


def plot_marl_two_policy_comparison(
    trajectories: List[Dict[str, np.ndarray]],
    labels: List[str],
    save_path: Path,
    title: str = "Two-Policy Comparison",
    split_t: Optional[int] = None,
) -> None:
    if len(trajectories) != 2 or len(labels) != 2:
        raise ValueError("plot_marl_two_policy_comparison requires exactly two trajectories and two labels")

    common_steps, _, split_t = _resolve_two_policy_layout(trajectories, split_t)

    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor("white")
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 1.35],
        width_ratios=[1.0, 1.0, 0.045],
        hspace=0.24,
        wspace=0.16,
    )
    top_axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    cbar_ax = fig.add_subplot(grid[0, 2])
    cbar_ax.set_visible(False)
    bottom_grid = grid[1, :2].subgridspec(1, 2, wspace=0.16)
    ax_bottom_early = fig.add_subplot(bottom_grid[0, 0])
    ax_bottom_late = fig.add_subplot(bottom_grid[0, 1])
    ax_bottom_early.set_facecolor("white")
    ax_bottom_late.set_facecolor("white")

    max_drop_delay = 1.0
    for traj in trajectories:
        actions = np.asarray(traj["actions"], dtype=np.int64)[:common_steps]
        drop_delay_steps = np.asarray(
            traj.get("drop_delay_steps", np.full(actions.shape, np.nan, dtype=np.float32)),
            dtype=np.float32,
        )[:common_steps]
        masked_drop_delay = np.ma.masked_invalid(drop_delay_steps.T)
        if masked_drop_delay.count() > 0:
            max_drop_delay = max(max_drop_delay, float(masked_drop_delay.max()))

    drop_im = None
    for idx, (ax, traj, label) in enumerate(zip(top_axes, trajectories, labels)):
        ax.set_facecolor("white")
        actions = np.asarray(traj["actions"], dtype=np.int64)[:common_steps]
        n_steps, n_agents = actions.shape
        drop_delay_steps = np.asarray(
            traj.get("drop_delay_steps", np.full(actions.shape, np.nan, dtype=np.float32)),
            dtype=np.float32,
        )[:common_steps]

        sent_mask = actions > 0
        drop_mask = np.isfinite(drop_delay_steps)
        sent_not_dropped_mask = sent_mask & ~drop_mask
        base_outcomes = np.zeros((n_steps, n_agents), dtype=np.int64)
        base_outcomes[sent_not_dropped_mask] = 1
        base_outcomes_vis = base_outcomes.T
        base_cmap = ListedColormap(["#f5f5f5", "#457b9d"])
        extent = (-0.5, common_steps - 0.5, n_agents - 0.5, -0.5)
        ax.imshow(
            base_outcomes_vis,
            aspect="auto",
            interpolation="nearest",
            cmap=base_cmap,
            vmin=0,
            vmax=1,
            origin="upper",
            extent=extent,
        )

        masked_drop_delay = np.ma.masked_invalid(drop_delay_steps.T)
        if masked_drop_delay.count() > 0:
            drop_im = ax.imshow(
                masked_drop_delay,
                aspect="auto",
                interpolation="nearest",
                cmap="YlOrRd",
                vmin=0.0,
                vmax=max_drop_delay,
                origin="upper",
                extent=extent,
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No dropped packets",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="#444444",
            )

        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_xlabel("Timestep", fontsize=11)
        ax.set_yticks(np.arange(n_agents))
        # Keep agent_0 at the top row deterministically.
        ax.set_ylim(n_agents - 0.5, -0.5)
        ax.set_xlim(-0.5, common_steps - 0.5)
        if idx == 0:
            ax.set_ylabel("Agent", fontsize=11)
            ax.set_yticklabels([f"agent_{i}" for i in range(n_agents)])
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])
        ax.tick_params(axis="both", labelsize=9)

    if drop_im is not None:
        cbar_ax.set_visible(True)
        cbar = fig.colorbar(drop_im, cax=cbar_ax)
        cbar.set_label("Drop delay (timesteps)", fontsize=10)
        cbar.ax.tick_params(labelsize=9)

    top_axes[1].legend(
        handles=[
            Patch(facecolor="#f5f5f5", edgecolor="#bdbdbd", label="No send"),
            Patch(facecolor="#457b9d", edgecolor="#457b9d", label="Sent, not dropped"),
            Patch(facecolor="#d7301f", edgecolor="#d7301f", label="Dropped (delay heatmap)"),
        ],
        loc="upper left",
        fontsize=9,
        frameon=True,
    )

    policy_colors = ["#1f77b4", "#d62728"]
    metric_per_policy, metric_label = _extract_two_policy_metrics(trajectories, common_steps)

    def _plot_metric_window(ax: Axes, t0: int, t1: int, subtitle: str) -> None:
        for idx, (metric, label) in enumerate(zip(metric_per_policy, labels)):
            if t1 <= t0 or metric.shape[0] < t1:
                continue
            window_metric = metric[t0:t1]
            if window_metric.size == 0 or np.all(np.isnan(window_metric)):
                continue
            x = np.arange(t0, t1, dtype=np.int64)
            mean_metric = np.nanmean(window_metric, axis=1)
            q25 = np.nanpercentile(window_metric, 25, axis=1)
            q75 = np.nanpercentile(window_metric, 75, axis=1)
            valid = np.isfinite(mean_metric)
            integrate_fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
            if np.count_nonzero(valid) >= 2:
                auc = float(integrate_fn(mean_metric[valid], x=x[valid]))
            elif np.count_nonzero(valid) == 1:
                auc = float(mean_metric[valid][0])
            else:
                auc = float("nan")
            color = policy_colors[idx % len(policy_colors)]
            auc_str = f"{auc:.4g}" if np.isfinite(auc) else "nan"
            ax.plot(x, mean_metric, color=color, linewidth=2.5, label=f"{label} (mean, AUC={auc_str})")
            ax.fill_between(x, q25, q75, color=color, alpha=0.16, linewidth=0)

        ax.set_title(subtitle, fontsize=12, fontweight="bold")
        ax.set_xlabel("Timestep", fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(t0 - 0.5, t1 - 0.5)

    _plot_metric_window(
        ax_bottom_early,
        0,
        split_t,
        f"{metric_label} Early Window (t=0..{split_t - 1})",
    )
    _plot_metric_window(
        ax_bottom_late,
        split_t,
        common_steps,
        f"{metric_label} Late Window (t={split_t}..{common_steps - 1})",
    )

    ax_bottom_early.set_ylabel(metric_label, fontsize=12)
    ax_bottom_early.legend(fontsize=9, loc="upper right")
    ax_bottom_late.legend(fontsize=9, loc="upper right")

    fig.suptitle(f"{title}  |  auto split t={split_t}", fontsize=16, fontweight="bold", y=0.99)
    fig.subplots_adjust(left=0.06, right=0.96, bottom=0.08, top=0.90, hspace=0.24, wspace=0.16)
    plt.savefig(str(save_path), dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved two-policy comparison plot to {save_path} (split: t={split_t})")


def plot_marl_two_policy_per_agent_metric(
    trajectories: List[Dict[str, np.ndarray]],
    labels: List[str],
    save_path: Path,
    title: str = "Two-Policy Per-Agent Metric Comparison",
    split_t: Optional[int] = None,
) -> None:
    if len(trajectories) != 2 or len(labels) != 2:
        raise ValueError("plot_marl_two_policy_per_agent_metric requires exactly two trajectories and two labels")

    common_steps, n_agents, split_t = _resolve_two_policy_layout(trajectories, split_t)
    metric_per_policy, metric_label = _extract_two_policy_metrics(trajectories, common_steps)
    policy_colors = ["#1f77b4", "#d62728"]

    fig_height = max(6.5, 2.35 * n_agents)
    fig, axes = plt.subplots(
        n_agents,
        2,
        figsize=(18, fig_height),
        sharex="col",
        squeeze=False,
    )
    fig.patch.set_facecolor("white")

    def _plot_agent_window(ax: Axes, agent_idx: int, t0: int, t1: int, subtitle: str) -> None:
        if agent_idx == 0:
            ax.set_title(subtitle, fontsize=12, fontweight="bold")

        window_has_data = False
        for idx, (metric, label) in enumerate(zip(metric_per_policy, labels)):
            if t1 <= t0 or metric.shape[0] < t1 or metric.shape[1] <= agent_idx:
                continue
            window_metric = metric[t0:t1, agent_idx]
            if window_metric.size == 0 or np.all(np.isnan(window_metric)):
                continue

            x = np.arange(t0, t1, dtype=np.int64)
            valid = np.isfinite(window_metric)
            integrate_fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
            if np.count_nonzero(valid) >= 2:
                auc = float(integrate_fn(window_metric[valid], x=x[valid]))
            elif np.count_nonzero(valid) == 1:
                auc = float(window_metric[valid][0])
            else:
                auc = float("nan")

            color = policy_colors[idx % len(policy_colors)]
            auc_str = f"{auc:.4g}" if np.isfinite(auc) else "nan"
            ax.plot(
                x,
                window_metric,
                color=color,
                linewidth=2.2,
                label=f"{label} (AUC={auc_str})",
            )
            window_has_data = True

        if not window_has_data:
            ax.text(
                0.5,
                0.5,
                f"No finite {metric_label}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color="#444444",
            )

        ax.grid(True, alpha=0.25)
        ax.set_xlim(t0 - 0.5, t1 - 0.5)
        handles, legend_labels = ax.get_legend_handles_labels()
        if handles and legend_labels:
            ax.legend(fontsize=8, loc="upper right")

    for agent_idx in range(n_agents):
        finite_values: List[np.ndarray] = []
        for metric in metric_per_policy:
            if metric.shape[1] <= agent_idx:
                continue
            finite = metric[:common_steps, agent_idx]
            finite = finite[np.isfinite(finite)]
            if finite.size > 0:
                finite_values.append(finite)

        row_limits = None
        if finite_values:
            agent_values = np.concatenate(finite_values)
            y_min = float(np.min(agent_values))
            y_max = float(np.max(agent_values))
            y_span = y_max - y_min
            if y_span <= 0.0:
                pad = max(1e-6, 0.08 * max(abs(y_max), 1.0))
            else:
                pad = 0.08 * y_span
            row_limits = (y_min - pad, y_max + pad)

        ax_early = axes[agent_idx, 0]
        ax_late = axes[agent_idx, 1]
        ax_early.set_facecolor("white")
        ax_late.set_facecolor("white")

        _plot_agent_window(
            ax_early,
            agent_idx,
            0,
            split_t,
            f"{metric_label} Early Window (t=0..{split_t - 1})",
        )
        _plot_agent_window(
            ax_late,
            agent_idx,
            split_t,
            common_steps,
            f"{metric_label} Late Window (t={split_t}..{common_steps - 1})",
        )

        if row_limits is not None:
            ax_early.set_ylim(*row_limits)
            ax_late.set_ylim(*row_limits)

        ax_early.set_ylabel(f"agent_{agent_idx}", fontsize=11)
        if agent_idx == n_agents - 1:
            ax_early.set_xlabel("Timestep", fontsize=11)
            ax_late.set_xlabel("Timestep", fontsize=11)

    fig.text(0.018, 0.5, metric_label, va="center", rotation="vertical", fontsize=12)
    fig.suptitle(f"{title}  |  auto split t={split_t}", fontsize=16, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.06, top=0.94, hspace=0.34, wspace=0.14)
    plt.savefig(str(save_path), dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved per-agent two-policy {metric_label} plot to {save_path} (split: t={split_t})")


def estimate_two_policy_split_timestep(
    trajectories: List[Dict[str, np.ndarray]],
    metric_key: str = "reward_metric",
) -> int:
    """Estimate the transient/end split time from metric scale decay.

    Strategy:
    - Build a robust per-timestep envelope using the 90th percentile across agents.
    - Combine policies by taking per-timestep max envelope.
    - Find the earliest sustained point where envelope drops near tail scale.
    """
    envelopes: List[np.ndarray] = []
    for traj in trajectories:
        metric = np.asarray(traj.get(metric_key, []), dtype=np.float32)
        if metric.ndim != 2 or metric.shape[0] < 2:
            continue
        q90 = np.nanpercentile(metric, 90, axis=1)
        envelopes.append(np.abs(q90))

    if not envelopes:
        return 1

    n_steps = int(min(env.shape[0] for env in envelopes))
    if n_steps < 4:
        return 1

    stacked = np.stack([env[:n_steps] for env in envelopes], axis=0)
    envelope = np.nanmax(stacked, axis=0)

    head_len = max(3, n_steps // 12)
    tail_len = max(6, n_steps // 5)
    peak = float(np.nanmax(envelope[:head_len]))
    tail_level = float(np.nanmedian(envelope[-tail_len:]))
    if not np.isfinite(peak) or not np.isfinite(tail_level):
        return max(1, n_steps // 8)

    # Threshold closer to tail than peak to mark post-transient region.
    threshold = tail_level + 0.15 * max(0.0, peak - tail_level)
    sustain = max(4, n_steps // 25)
    split = None
    for t in range(head_len, n_steps - sustain):
        if float(np.nanmax(envelope[t:t + sustain])) <= threshold:
            split = t
            break
    if split is None:
        split = max(1, n_steps // 8)

    # Keep both windows meaningful.
    min_split = max(1, int(0.04 * n_steps))
    max_split = max(min_split + 1, int(0.55 * n_steps))
    return int(np.clip(split, min_split, max_split))


def _compute_axis_limits(all_points: np.ndarray, margin: float = 0.15):
    """Return (xlim, ylim) tuples with equal aspect padding."""
    x_min, x_max = float(all_points[:, 0].min()), float(all_points[:, 0].max())
    y_min, y_max = float(all_points[:, 1].min()), float(all_points[:, 1].max())
    x_range = max(0.1, x_max - x_min)
    y_range = max(0.1, y_max - y_min)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    max_range = max(x_range, y_range) * (1 + 2 * margin)
    xlim = (x_center - max_range / 2, x_center + max_range / 2)
    ylim = (y_center - max_range / 2, y_center + max_range / 2)
    return xlim, ylim


def _save_animation(anim, fig, path: Path, fps: int, description: str) -> None:
    """Save an animation with FFMpegWriter and common error handling."""
    try:
        writer = FFMpegWriter(fps=fps, bitrate=2400, codec="libx264")
        anim.save(str(path), writer=writer)
        print(f"  ✓ Saved {description} to {path}")
    except Exception as e:
        print(f"  ✗ Error saving {description}: {e}")
        print(f"  Make sure FFmpeg is installed:")
        print(f"    - Linux: sudo apt-get install ffmpeg")
        print(f"    - Mac: brew install ffmpeg")
    finally:
        plt.close(fig)


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
    xlim, ylim = _compute_axis_limits(all_points)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

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
    _save_animation(anim, fig, save_path, fps, "MARL animation")


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
    xlim, ylim = _compute_axis_limits(all_points)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

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
    _save_animation(anim, fig, save_path, fps, "agent animation")


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
        default=250,
        help='Length of episode to simulate (default: 250)'
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
        '--epsilon-greedy',
        type=float,
        default=0.0,
        help='Probability of replacing each agent\'s action with a random action '
             '(applied before env.step so agents observe the randomized action in their history)'
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
    add_set_override_argument(parser)

    # Check for list-heuristics flag first (before full parsing)
    if '--list-heuristics' in sys.argv:
        print("\nAvailable heuristic policies:")
        print("-" * 50)
        for name in sorted(HEURISTIC_POLICIES.keys()):
            print(f"  - {name}")
        print("\nPattern aliases:")
        print("  - perfect_sync_n<K> (example: perfect_sync_n2)")
        print("  - perfect_sync_<K>  (example: perfect_sync_2)")
        print("  - value_of_update_<threshold> (example: value_of_update_0.05)")
        print("  - value_of_update_threshold_<threshold> (example: value_of_update_threshold_1e-3)")
        print("  - vou_<threshold> (example: vou_0.05)")
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
    config_overrides = parse_set_overrides(args.set_overrides)
    config_path_for_env = args.config
    config = load_config(args.config)
    if config_overrides:
        deep_merge(config, config_overrides)
    reward_override = config.get("reward", {}).get("evaluation")
    if not isinstance(reward_override, dict):
        reward_override = {}
    else:
        reward_override = dict(reward_override)
    # Always disable reward clipping during evaluation
    reward_override["reward_clip_min"] = None
    reward_override["reward_clip_max"] = None
    termination_override = config.get("termination", {}).get("evaluation")
    if not isinstance(termination_override, dict):
        termination_override = None
    print(f"✓ Configuration loaded")
    if config_overrides:
        print("Config overrides:")
        for arg in args.set_overrides:
            print(f"  --set {arg}")
    print("✓ Reward clipping disabled for evaluation")
    if reward_override or termination_override is not None:
        print("✓ Using evaluation reward/termination overrides")
    print()

    allowed_policy_types = {"marl_torch", "heuristic", "es", "openai_es"}
    if not all(policy_type in allowed_policy_types for policy_type in policy_types_norm):
        parser.error("Supported policy types: marl_torch, es, openai_es, heuristic")

    try:
        resolved_n_agents = resolve_n_agents(
            config,
            list(zip(policies_to_load, policy_types_norm)),
            args.n_agents,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if resolved_n_agents < 1:
        parser.error("Resolved n_agents must be >= 1")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config_overrides:
        config_path_for_env = str(
            apply_config_overrides(
                Path(args.config), config_overrides, output_dir / "overridden_config.json"
            )
        )

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
            config_path=config_path_for_env,
            seed=args.seed,
            reward_override=reward_override,
            termination_override=termination_override,
            track_true_goodput=True,
            track_eval_stats=True,
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
        eps_rng = np.random.default_rng(args.seed) if args.epsilon_greedy > 0 else None
        traj = run_episode_multi_agent(
            env,
            policy,
            args.episode_length,
            seed=args.seed,
            network_trace=args.network_trace,
            trace_interval=args.trace_interval,
            trace_start=args.trace_start,
            epsilon=args.epsilon_greedy,
            rng=eps_rng,
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

    if len(trajectories) == 2:
        split_t = estimate_two_policy_split_timestep(trajectories)
        print(f"✓ Auto-selected two-policy split timestep: t={split_t}")
        two_policy_plot_path = output_dir / f"{output_prefix}_two_policy_comparison.png"
        plot_marl_two_policy_comparison(
            trajectories,
            policy_labels[:2],
            two_policy_plot_path,
            title=f"Two-Policy Comparison (Drops + {trajectories[0].get('reward_metric_label', 'metric')})",
            split_t=split_t,
        )
        per_agent_metric_plot_path = output_dir / f"{output_prefix}_two_policy_per_agent_metric.png"
        plot_marl_two_policy_per_agent_metric(
            trajectories,
            policy_labels[:2],
            per_agent_metric_plot_path,
            title=f"Two-Policy Per-Agent {trajectories[0].get('reward_metric_label', 'Metric')}",
            split_t=split_t,
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
