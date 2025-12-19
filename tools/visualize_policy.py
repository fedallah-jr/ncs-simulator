"""
Policy Visualization Tool for Networked Control Systems

This tool visualizes the state evolution of policies trained or defined in the NCS simulator.
It supports both learned policies (Stable-Baselines3, OpenAI-ES) and heuristic policies.

Usage:
    # Visualize a trained PPO policy
    python -m tools.visualize_policy --config configs/perfect_comm.json --policy path/to/model.zip --policy-type sb3

    # Visualize an ES policy
    python -m tools.visualize_policy --config configs/perfect_comm.json --policy path/to/model.npz --policy-type es

    # Visualize a heuristic policy
    python -m tools.visualize_policy --config configs/perfect_comm.json --policy always_send --policy-type heuristic

    # Visualize an IQL/VDN/QMIX PyTorch checkpoint (agent_0)
    python -m tools.visualize_policy --config configs/perfect_comm.json --policy path/to/latest_model.pt --policy-type marl_torch --n-agents 3

    # True multi-agent visualization (all agents act from checkpoint)
    python -m tools.visualize_policy --config configs/marl_mixed_plants.json --policy path/to/latest_model.pt --policy-type marl_torch --n-agents 3 --multi-agent --generate-video --per-agent-videos

    # Compare multiple policies
    python -m tools.visualize_policy --config configs/perfect_comm.json \
        --policies path/to/ppo.zip always_send send_every_5 \
        --policy-types sb3 heuristic heuristic \
        --labels "PPO" "Always Send" "Send Every 5"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.animation import FuncAnimation, FFMpegWriter
import json
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ncs_env.env import NCS_Env
from ncs_env.config import load_config
from utils.wrapper import SingleAgentWrapper
from tools.heuristic_policies import get_heuristic_policy, HEURISTIC_POLICIES


def load_sb3_policy(model_path: str, env: Any):
    """
    Load a Stable-Baselines3 policy.

    Args:
        model_path: Path to the saved model (.zip file)
        env: Environment instance

    Returns:
        Loaded model

    Raises:
        ImportError: If stable-baselines3 is not installed
        FileNotFoundError: If model file doesn't exist
    """
    model_path_p = Path(model_path)
    if not model_path_p.exists():
        raise FileNotFoundError(f"Model file not found: {model_path_p}")

    try:
        # Try to import stable-baselines3
        from stable_baselines3 import PPO, DQN, A2C, SAC, TD3
    except ImportError:
        raise ImportError(
            "stable-baselines3 is required to load trained policies. "
            "Install with: pip install stable-baselines3"
        )

    # Try to load with different algorithms
    for algorithm in [PPO, DQN, A2C, SAC, TD3]:
        try:
            model = algorithm.load(str(model_path_p), env=env)
            print(f"✓ Loaded {algorithm.__name__} model from {model_path_p}")
            return model
        except Exception:
            continue

    raise ValueError(f"Could not load model from {model_path} with any SB3 algorithm")


def load_es_policy(model_path: str, env: Any):
    """
    Load an OpenAI-ES policy (JAX/Flax).

    Args:
        model_path: Path to the saved model (.npz file)
        env: Environment instance

    Returns:
        Policy object with predict() method
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
        
        print(f"  Architecture: hidden_size={hidden_size}, use_layer_norm={use_layer_norm}")
    except Exception as e:
        raise ValueError(f"Could not load numpy data from {model_path_p}: {e}")

    # Reconstruct model structure to get reshaper
    # env is a SingleAgentWrapper, so env.action_space is Discrete(2)
    if hasattr(env, "action_space") and hasattr(env.action_space, "n"):
        action_dim = env.action_space.n
    else:
        # Fallback for unwrapped envs if necessary
        action_dim = 2 

    if hasattr(env, "observation_space") and hasattr(env.observation_space, "shape"):
        obs_dim = env.observation_space.shape[0]
    else:
        raise ValueError("Environment must have a valid observation space.")

    # Create model with correct architecture
    model = create_policy_net(action_dim=action_dim, hidden_size=hidden_size, use_layer_norm=use_layer_norm)

    # Initialize dummy to get structure
    rng = jax.random.PRNGKey(0)
    dummy_obs = jnp.zeros((1, obs_dim))
    dummy_params = model.init(rng, dummy_obs)

    _, unravel_fn = flatten_util.ravel_pytree(dummy_params)
    params = unravel_fn(flat_params)

    class ESPolicyWrapper:
        def __init__(self, model, params):
            self.model = model
            self.params = params

        def predict(self, observation, deterministic=True):
            # Observation is (obs_dim,) numpy array
            # Add batch dim and convert to JAX array
            obs = jnp.array(observation[np.newaxis, :])
            logits = self.model.apply(self.params, obs)
            action = jnp.argmax(logits).item()
            return action, None

    return ESPolicyWrapper(model, params)


def load_marl_torch_policy(model_path: str, env: Any, marl_agent_index: int = 0):
    """
    Load a PyTorch MARL (IQL/VDN/QMIX) policy checkpoint saved by `algorithms/marl_*.py`.

    Returns a lightweight wrapper with a `predict(obs, deterministic=True)` method.
    Note: Visualization uses `SingleAgentWrapper`, so only the selected agent index is
    queried for actions while other agents are held at `SingleAgentWrapper(other_agent_action=...)`.
    """
    model_path_p = Path(model_path)

    try:
        import torch
        from utils.marl.networks import MLPAgent
    except ImportError as e:
        raise ImportError("torch is required to load MARL torch checkpoints") from e

    from utils.marl.torch_policy import load_marl_torch_agents_from_checkpoint

    agent_or_agents, meta = load_marl_torch_agents_from_checkpoint(model_path_p)
    n_agents = meta.n_agents
    obs_dim = meta.obs_dim
    n_actions = meta.n_actions
    if marl_agent_index < 0 or marl_agent_index >= n_agents:
        raise ValueError("marl_agent_index must be within [0, n_agents)")

    env_obs_dim = int(getattr(env.observation_space, "shape", (0,))[0])
    if env_obs_dim != obs_dim:
        raise ValueError(f"Env obs_dim={env_obs_dim} does not match checkpoint obs_dim={obs_dim}")

    if isinstance(agent_or_agents, MLPAgent):
        chosen_agent = agent_or_agents
    else:
        chosen_agent = agent_or_agents[marl_agent_index]

    class MARLTorchPolicyWrapper:
        def __init__(self, agent, n_agents: int, use_agent_id: bool, marl_agent_index: int):
            self.agent = agent
            self.n_agents = n_agents
            self.use_agent_id = use_agent_id
            self.marl_agent_index = marl_agent_index

        def predict(self, observation, deterministic=True):
            obs = np.asarray(observation, dtype=np.float32)
            obs_t = torch.from_numpy(obs).float()
            if self.use_agent_id:
                onehot = torch.zeros(self.n_agents, dtype=torch.float32)
                onehot[self.marl_agent_index] = 1.0
                obs_t = torch.cat([obs_t, onehot], dim=0)
            q = self.agent(obs_t.unsqueeze(0))
            action = int(torch.argmax(q, dim=-1).item())
            return action, None

    return MARLTorchPolicyWrapper(
        chosen_agent,
        n_agents=n_agents,
        use_agent_id=meta.use_agent_id,
        marl_agent_index=marl_agent_index,
    )


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
            "Pass the correct `--n-agents` for the checkpoint."
        )
    env_obs_dim = int(env.observation_space.spaces["agent_0"].shape[0])
    if env_obs_dim != meta.obs_dim:
        raise ValueError(f"Env obs_dim={env_obs_dim} does not match checkpoint obs_dim={meta.obs_dim}")
    return MARLTorchMultiAgentPolicy(agent_or_agents, meta, device=torch.device("cpu"))


def load_policy(
    policy_path: str,
    policy_type: str,
    env: Any,
    n_agents: int = 1,
    seed: Optional[int] = None,
    marl_agent_index: int = 0,
):
    """
    Load a policy (SB3, ES, or heuristic).

    Args:
        policy_path: Path to model file or policy name
        policy_type: 'sb3', 'es', or 'heuristic'
        env: Environment instance
        n_agents: Number of agents (for heuristic policies)
        seed: Random seed

    Returns:
        Policy object with predict() method
    """
    if policy_type.lower() == 'sb3':
        return load_sb3_policy(policy_path, env)
    elif policy_type.lower() == 'es' or policy_type.lower() == 'openai_es':
        return load_es_policy(policy_path, env)
    elif policy_type.lower() == 'heuristic':
        return get_heuristic_policy(policy_path, n_agents=n_agents, seed=seed)
    elif policy_type.lower() in {'marl_torch', 'marl', 'torch_marl'}:
        return load_marl_torch_policy(policy_path, env, marl_agent_index=marl_agent_index)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}. Use 'sb3', 'es', 'heuristic', or 'marl_torch'")


def run_episode(env: Any, policy: Any, episode_length: int, deterministic: bool = True) -> Dict[str, np.ndarray]:
    """
    Run a single episode and collect trajectory data.

    Args:
        env: Environment instance (single-agent wrapped)
        policy: Policy with predict() method
        episode_length: Number of timesteps to run
        deterministic: Whether to use deterministic policy

    Returns:
        Dictionary containing:
            - states: Array of shape (episode_length+1, state_dim) with plant states
            - estimates: Array of shape (episode_length+1, state_dim) with controller estimates
            - actions: Array of shape (episode_length,) with actions taken
            - rewards: Array of shape (episode_length,) with rewards received
            - controls: Array of shape (episode_length, control_dim) with control inputs
            - state_errors: Array of shape (episode_length+1,) with state error magnitudes
    """
    # Reset policy if it has a reset method
    if hasattr(policy, 'reset'):
        policy.reset()

    # Reset environment
    obs, info = env.reset()

    # Get the underlying NCS_Env (SingleAgentWrapper stores it in env.env)
    ncs_env = env.env

    # Get state dimension from environment
    state_dim = ncs_env.state_dim

    # Initialize data storage
    states = np.zeros((episode_length + 1, state_dim))
    estimates = np.zeros((episode_length + 1, state_dim))
    actions = np.zeros(episode_length)
    rewards = np.zeros(episode_length)
    controls = np.zeros((episode_length, ncs_env.control_dim))
    state_errors = np.zeros(episode_length + 1)

    # Store initial state
    states[0] = ncs_env.plants[0].get_state()
    estimates[0] = ncs_env.controllers[0].x_hat
    state_errors[0] = np.linalg.norm(states[0])

    # Run episode
    for t in range(episode_length):
        # Get action from policy
        action, _ = policy.predict(obs, deterministic=deterministic)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Store data
        states[t + 1] = ncs_env.plants[0].get_state()
        estimates[t + 1] = ncs_env.controllers[0].x_hat
        actions[t] = action
        rewards[t] = reward
        controls[t] = ncs_env.controllers[0].last_control if hasattr(
            ncs_env.controllers[0], 'last_control'
        ) else 0.0
        state_errors[t + 1] = np.linalg.norm(states[t + 1])

        # Break if episode ended early (shouldn't happen normally)
        if terminated or truncated:
            # Episode ended before completing all timesteps
            # t is 0-indexed, so if t < episode_length-1, it ended early
            if t < episode_length - 1:
                print(f"Warning: Episode ended early at timestep {t} (expected {episode_length})")
            # Trim arrays
            states = states[:t + 2]
            estimates = estimates[:t + 2]
            actions = actions[:t + 1]
            rewards = rewards[:t + 1]
            controls = controls[:t + 1]
            state_errors = state_errors[:t + 2]
            break

    return {
        'states': states,
        'estimates': estimates,
        'actions': actions,
        'rewards': rewards,
        'controls': controls,
        'state_errors': state_errors,
        'timesteps': np.arange(len(states)),
    }


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
) -> Dict[str, np.ndarray]:
    if hasattr(policy, "reset"):
        policy.reset()

    obs_dict, info = env.reset(seed=seed)
    n_agents = int(env.n_agents)
    state_dim = int(env.state_dim)

    states = np.zeros((episode_length + 1, n_agents, state_dim), dtype=np.float32)
    estimates = np.zeros((episode_length + 1, n_agents, state_dim), dtype=np.float32)
    actions = np.zeros((episode_length, n_agents), dtype=np.int64)
    rewards = np.zeros((episode_length, n_agents), dtype=np.float32)
    state_errors = np.zeros((episode_length + 1, n_agents), dtype=np.float32)
    throughputs = np.zeros(episode_length + 1, dtype=np.float32)
    collided_packets = np.zeros(episode_length + 1, dtype=np.float32)
    reward_mix_weight = np.zeros(episode_length + 1, dtype=np.float32)

    states[0] = np.asarray(info.get("states", states[0]), dtype=np.float32)
    estimates[0] = np.asarray(info.get("estimates", estimates[0]), dtype=np.float32)
    state_errors[0] = np.linalg.norm(states[0], axis=1).astype(np.float32)
    throughputs[0] = float(info.get("throughput_kbps", 0.0))
    collided_packets[0] = float(info.get("collided_packets", 0.0))
    reward_mix_weight[0] = float(info.get("reward_mix_weight", 0.0))

    for t in range(episode_length):
        action_dict = policy.act(obs_dict)
        actions[t] = np.asarray([action_dict[f"agent_{i}"] for i in range(n_agents)], dtype=np.int64)
        obs_dict, rewards_dict, terminated, truncated, info = env.step(action_dict)
        rewards[t] = np.asarray([rewards_dict[f"agent_{i}"] for i in range(n_agents)], dtype=np.float32)

        states[t + 1] = np.asarray(info.get("states", states[t + 1]), dtype=np.float32)
        estimates[t + 1] = np.asarray(info.get("estimates", estimates[t + 1]), dtype=np.float32)
        state_errors[t + 1] = np.linalg.norm(states[t + 1], axis=1).astype(np.float32)
        throughputs[t + 1] = float(info.get("throughput_kbps", 0.0))
        collided_packets[t + 1] = float(info.get("collided_packets", 0.0))
        reward_mix_weight[t + 1] = float(info.get("reward_mix_weight", 0.0))

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
            reward_mix_weight = reward_mix_weight[: end + 1]
            break

    return {
        "states": states,
        "estimates": estimates,
        "actions": actions,
        "rewards": rewards,
        "state_errors": state_errors,
        "throughput_kbps": throughputs,
        "collided_packets": collided_packets,
        "reward_mix_weight": reward_mix_weight,
        "timesteps": np.arange(states.shape[0]),
    }


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


def plot_state_evolution_2d(
    trajectories: List[Dict[str, np.ndarray]],
    labels: List[str],
    save_path: str,
    title: str = "State Evolution (2D)",
    show_estimates: bool = True,
):
    """
    Plot 2D state evolution for multiple policies.

    Args:
        trajectories: List of trajectory dictionaries from run_episode()
        labels: List of labels for each trajectory
        save_path: Path to save the plot
        title: Plot title
        show_estimates: Whether to show controller estimates
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Define colors for different policies
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))

    # Plot 1: State trajectory in state space
    ax1: Axes = axes[0]
    for i, (traj, label) in enumerate(zip(trajectories, labels)):
        states = traj['states']
        # Plot state trajectory
        ax1.plot(states[:, 0], states[:, 1], '-', color=colors[i], alpha=0.7,
                 linewidth=2, label=f"{label} (actual)")
        ax1.plot(states[0, 0], states[0, 1], 'o', color=colors[i],
                 markersize=10, label=f"{label} (start)")
        ax1.plot(states[-1, 0], states[-1, 1], 's', color=colors[i],
                 markersize=10, label=f"{label} (end)")

        # Plot estimates if requested
        if show_estimates:
            estimates = traj['estimates']
            ax1.plot(estimates[:, 0], estimates[:, 1], '--', color=colors[i],
                     alpha=0.5, linewidth=1.5, label=f"{label} (estimate)")

    ax1.plot(0, 0, 'r*', markersize=15, label='Target (origin)')
    ax1.set_xlabel('State Dimension 1', fontsize=12)
    ax1.set_ylabel('State Dimension 2', fontsize=12)
    ax1.set_title('State Space Trajectory', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc='best')
    ax1.axis('equal')

    # Plot 2: State magnitude over time
    ax2: Axes = axes[1]
    for i, (traj, label) in enumerate(zip(trajectories, labels)):
        state_errors = traj['state_errors']
        timesteps = traj['timesteps']
        ax2.plot(timesteps, state_errors, '-', color=colors[i],
                 linewidth=2, label=label)

    ax2.set_xlabel('Timestep', fontsize=12)
    ax2.set_ylabel('State Magnitude ||x||', fontsize=12)
    ax2.set_title('State Magnitude Evolution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc='best')

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved state evolution plot to {save_path}")
    plt.close()


def plot_detailed_analysis(
    trajectories: List[Dict[str, np.ndarray]],
    labels: List[str],
    save_path: str,
    title: str = "Detailed Policy Analysis",
):
    """
    Plot detailed analysis including actions, rewards, and controls.

    Args:
        trajectories: List of trajectory dictionaries from run_episode()
        labels: List of labels for each trajectory
        save_path: Path to save the plot
        title: Plot title
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))

    # Plot 1: Actions over time
    ax1: Axes = axes[0, 0]
    for i, (traj, label) in enumerate(zip(trajectories, labels)):
        actions = traj['actions']
        timesteps = np.arange(len(actions))
        # Plot as step function
        ax1.step(timesteps, actions, where='post', color=colors[i],
                 linewidth=2, alpha=0.7, label=label)

    ax1.set_xlabel('Timestep', fontsize=12)
    ax1.set_ylabel('Action (0=No Send, 1=Send)', fontsize=12)
    ax1.set_title('Transmission Decisions', fontsize=14, fontweight='bold')
    ax1.set_ylim([-0.1, 1.1])
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc='best')

    # Plot 2: Rewards over time
    ax2: Axes = axes[0, 1]
    for i, (traj, label) in enumerate(zip(trajectories, labels)):
        rewards = traj['rewards']
        timesteps = np.arange(len(rewards))
        # Plot cumulative reward
        cumulative_rewards = np.cumsum(rewards)
        ax2.plot(timesteps, cumulative_rewards, '-', color=colors[i],
                 linewidth=2, alpha=0.7, label=label)

    ax2.set_xlabel('Timestep', fontsize=12)
    ax2.set_ylabel('Cumulative Reward', fontsize=12)
    ax2.set_title('Cumulative Reward', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc='best')

    # Plot 3: State components over time
    ax3: Axes = axes[1, 0]
    for i, (traj, label) in enumerate(zip(trajectories, labels)):
        states = traj['states']
        timesteps = traj['timesteps']
        # Plot each state dimension
        ax3.plot(timesteps, states[:, 0], '-', color=colors[i],
                 linewidth=2, alpha=0.7, label=f"{label} (dim 1)")
        if states.shape[1] > 1:
            ax3.plot(timesteps, states[:, 1], '--', color=colors[i],
                     linewidth=2, alpha=0.7, label=f"{label} (dim 2)")

    ax3.set_xlabel('Timestep', fontsize=12)
    ax3.set_ylabel('State Value', fontsize=12)
    ax3.set_title('State Components Over Time', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8, loc='best')

    # Plot 4: Transmission statistics
    ax4: Axes = axes[1, 1]
    tx_counts = []
    avg_rewards = []
    final_errors = []
    for i, (traj, label) in enumerate(zip(trajectories, labels)):
        tx_count = np.sum(traj['actions'])
        avg_reward = np.mean(traj['rewards'])
        final_error = traj['state_errors'][-1]

        tx_counts.append(tx_count)
        avg_rewards.append(avg_reward)
        final_errors.append(final_error)

    x = np.arange(len(labels))
    width = 0.25

    ax4_twin1 = ax4.twinx()
    ax4_twin2 = ax4.twinx()
    ax4_twin2.spines['right'].set_position(('outward', 60))

    p1 = ax4.bar(x - width, tx_counts, width, label='Transmissions', color='steelblue', alpha=0.8)
    p2 = ax4_twin1.bar(x, avg_rewards, width, label='Avg Reward', color='forestgreen', alpha=0.8)
    p3 = ax4_twin2.bar(x + width, final_errors, width, label='Final Error', color='coral', alpha=0.8)

    ax4.set_xlabel('Policy', fontsize=12)
    ax4.set_ylabel('Transmission Count', fontsize=12, color='steelblue')
    ax4_twin1.set_ylabel('Avg Reward', fontsize=12, color='forestgreen')
    ax4_twin2.set_ylabel('Final State Error', fontsize=12, color='coral')
    ax4.set_title('Policy Statistics', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, rotation=15, ha='right')
    ax4.tick_params(axis='y', labelcolor='steelblue')
    ax4_twin1.tick_params(axis='y', labelcolor='forestgreen')
    ax4_twin2.tick_params(axis='y', labelcolor='coral')

    # Combine legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin1.get_legend_handles_labels()
    lines3, labels3 = ax4_twin2.get_legend_handles_labels()
    ax4.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, fontsize=10, loc='upper left')

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved detailed analysis plot to {save_path}")
    plt.close()


def create_state_evolution_animation(
    trajectories: List[Dict[str, np.ndarray]],
    labels: List[str],
    save_path: str,
    title: str = "State Evolution Animation",
    fps: int = 30,
    speedup: int = 1,
    show_estimates: bool = True,
):
    """
    Create an animated visualization of 2D state evolution.

    Args:
        trajectories: List of trajectory dictories from run_episode()
        labels: List of labels for each trajectory
        save_path: Path to save the animation (MP4)
        title: Animation title
        fps: Frames per second for the video
        speedup: Speed multiplier (e.g., 2 = 2x speed)
        show_estimates: Whether to show controller estimates
    """
    print(f"  Creating animation...")

    # Define colors for different policies
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))

    # Determine the longest trajectory
    max_len = max(len(traj['states']) for traj in trajectories)

    # Set up the figure - single panel for state space
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Compute global axis limits for state space
    all_states = np.vstack([traj['states'] for traj in trajectories])
    if show_estimates:
        all_estimates = np.vstack([traj['estimates'] for traj in trajectories])
        all_points = np.vstack([all_states, all_estimates])
    else:
        all_points = all_states

    # Compute limits with appropriate margin
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()

    # Calculate ranges
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Use larger margin if range is very small
    if x_range < 0.1:
        x_range = 0.1
    if y_range < 0.1:
        y_range = 0.1

    margin = 0.15
    x_min -= margin * x_range
    x_max += margin * x_range
    y_min -= margin * y_range
    y_max += margin * y_range

    # Make axes equal by using the larger range
    max_range = max(x_range, y_range) * (1 + 2 * margin)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    x_min = x_center - max_range / 2
    x_max = x_center + max_range / 2
    y_min = y_center - max_range / 2
    y_max = y_center + max_range / 2

    # Initialize plot elements
    state_lines = []
    estimate_lines = []
    state_points = []
    estimate_points = []

    for i, (traj, label) in enumerate(zip(trajectories, labels)):
        # State space trajectory
        line, = ax.plot([], [], '-', color=colors[i], alpha=0.8, linewidth=2.5, label=f"{label}")
        state_lines.append(line)
        point, = ax.plot([], [], 'o', color=colors[i], markersize=14, zorder=5)
        state_points.append(point)

        if show_estimates:
            est_line, = ax.plot([], [], '--', color=colors[i], alpha=0.5, linewidth=2, label=f"{label} (est.)")
            estimate_lines.append(est_line)
            est_point, = ax.plot([], [], 's', color=colors[i], markersize=10, alpha=0.7, zorder=5)
            estimate_points.append(est_point)

    # Target point
    ax.plot(0, 0, 'r*', markersize=25, label='Target', zorder=10)

    # Configure axes
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('State Dimension 1', fontsize=16, fontweight='bold')
    ax.set_ylabel('State Dimension 2', fontsize=16, fontweight='bold')
    ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linewidth=1.5)
    ax.legend(fontsize=14, loc='best', framealpha=0.9)
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(labelsize=12)

    # Time text
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        fontsize=16, verticalalignment='top', fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    def init():
        """Initialize animation"""
        for line in state_lines + estimate_lines:
            line.set_data([], [])
        for point in state_points + estimate_points:
            point.set_data([], [])
        time_text.set_text('')
        return state_lines + estimate_lines + state_points + estimate_points + [time_text]

    def animate(frame):
        """Update animation for given frame"""
        t = frame * speedup
        time_text.set_text(f'Timestep: {t}')

        for i, traj in enumerate(trajectories):
            # Handle case where trajectory is shorter than max_len
            idx = min(t, len(traj['states']) - 1)

            # Update state trajectory
            state_lines[i].set_data(traj['states'][:idx+1, 0], traj['states'][:idx+1, 1])
            state_points[i].set_data([traj['states'][idx, 0]], [traj['states'][idx, 1]])

            # Update estimates if shown
            if show_estimates and i < len(estimate_lines):
                estimate_lines[i].set_data(traj['estimates'][:idx+1, 0], traj['estimates'][:idx+1, 1])
                estimate_points[i].set_data([traj['estimates'][idx, 0]], [traj['estimates'][idx, 1]])

        artists = state_lines + state_points + [time_text]
        if show_estimates:
            artists += estimate_lines + estimate_points
        return artists

    # Create animation
    num_frames = (max_len - 1) // speedup + 1
    anim = FuncAnimation(fig, animate, init_func=init, frames=num_frames,
                        interval=1000/fps, blit=True, repeat=True)

    # Save animation as MP4
    try:
        writer = FFMpegWriter(fps=fps, bitrate=2400, codec='libx264')
        anim.save(save_path, writer=writer)
        print(f"  ✓ Saved animation to {save_path}")
    except Exception as e:
        print(f"  ✗ Error saving animation: {e}")
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
        help='Path to policy file (SB3/ES) or policy name (heuristic)'
    )
    parser.add_argument(
        '--policy-type',
        type=str,
        choices=['sb3', 'es', 'heuristic', 'marl_torch'],
        help='Type of policy: sb3, es, heuristic, or marl_torch'
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
        choices=['sb3', 'es', 'heuristic', 'marl_torch'],
        help='List of policy types corresponding to --policies'
    )
    parser.add_argument(
        '--marl-agent-index',
        type=int,
        default=0,
        help='Agent index for marl_torch checkpoints (default: 0)'
    )
    parser.add_argument(
        '--multi-agent',
        action='store_true',
        help='For marl_torch: run a true multi-agent rollout (all agents act from the checkpoint)'
    )
    parser.add_argument(
        '--per-agent-videos',
        action='store_true',
        help='In --multi-agent mode with --generate-video, also write one MP4 per agent'
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
        default=1,
        help='Number of agents in the environment (default: 1; in --multi-agent mode must match checkpoint n_agents)'
    )

    # Visualization parameters
    parser.add_argument(
        '--show-estimates',
        action='store_true',
        default=True,
        help='Show controller estimates in plots (default: True)'
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

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output prefix if not provided
    if not args.output_prefix:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_prefix = f"viz_{timestamp}"
    else:
        output_prefix = args.output_prefix

    print(f"\n{'='*60}")
    print(f"NCS Policy Visualization Tool")
    print(f"{'='*60}\n")

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    print(f"✓ Configuration loaded\n")

    # Create environment factory function
    def make_env():
        return NCS_Env(
            n_agents=args.n_agents,
            episode_length=args.episode_length,
            config_path=args.config,
            seed=args.seed
        )

    print(f"Creating environment...")
    if args.multi_agent:
        env: Any = make_env()
        state_dim = int(env.state_dim)
        print(f"✓ Environment created (multi-agent, state_dim={state_dim}, episode_length={args.episode_length})\n")
    else:
        env = SingleAgentWrapper(make_env)
        temp_env = make_env()
        state_dim = int(temp_env.state_dim)
        temp_env.close()
        print(f"✓ Environment created (state_dim={state_dim}, episode_length={args.episode_length})\n")

    # Load policies and run episodes
    trajectories = []
    print(f"Running simulations for {len(policies_to_load)} policy/policies...\n")

    for i, (policy_path, policy_type, label) in enumerate(zip(policies_to_load, policy_types, policy_labels)):
        print(f"[{i+1}/{len(policies_to_load)}] {label}")
        print(f"  Type: {policy_type}")
        print(f"  Path/Name: {policy_path}")

        if args.multi_agent:
            if policy_type.lower() not in {"marl_torch", "marl", "torch_marl"}:
                print("  ✗ Error: --multi-agent only supports --policy-type marl_torch\n")
                continue
            try:
                policy = load_marl_torch_multi_agent_policy(policy_path, env)
            except Exception as e:
                print(f"  ✗ Error loading policy: {e}\n")
                continue
            print(f"  Running multi-agent episode...")
            traj = run_episode_multi_agent(env, policy, args.episode_length, seed=args.seed)
        else:
            obs, info = env.reset(seed=args.seed)
            try:
                policy = load_policy(
                    policy_path,
                    policy_type,
                    env,
                    n_agents=args.n_agents,
                    seed=args.seed,
                    marl_agent_index=int(args.marl_agent_index),
                )
            except Exception as e:
                print(f"  ✗ Error loading policy: {e}\n")
                continue
            print(f"  Running episode...")
            traj = run_episode(env, policy, args.episode_length, deterministic=True)

        # Print summary statistics
        total_reward = float(np.sum(traj["rewards"]))
        tx_count = float(np.sum(traj["actions"]))
        final_error = float(np.mean(traj["state_errors"][-1])) if args.multi_agent else float(traj["state_errors"][-1])
        print(f"  ✓ Episode complete")
        print(f"    - Total reward: {total_reward:.2f}")
        denom = args.episode_length * (args.n_agents if args.multi_agent else 1)
        print(f"    - Transmissions: {int(tx_count)}/{denom}")
        print(f"    - Final state error: {final_error:.4f}\n")

        trajectories.append(traj)

    if len(trajectories) == 0:
        print("No successful policy runs. Exiting.")
        return

    # Generate plots
    print(f"Generating visualizations...\n")

    if args.multi_agent:
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
    else:
        plot1_path = output_dir / f"{output_prefix}_state_evolution.png"
        plot_state_evolution_2d(
            trajectories,
            policy_labels[:len(trajectories)],
            str(plot1_path),
            title="State Evolution in 2D State Space",
            show_estimates=args.show_estimates,
        )

        plot2_path = output_dir / f"{output_prefix}_detailed_analysis.png"
        plot_detailed_analysis(
            trajectories,
            policy_labels[:len(trajectories)],
            str(plot2_path),
            title="Detailed Policy Analysis",
        )

    # Generate video animation if requested
    if args.generate_video:
        print(f"\nGenerating animation...\n")
        if args.multi_agent:
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
        else:
            video_path = output_dir / f"{output_prefix}_animation.mp4"
            create_state_evolution_animation(
                trajectories,
                policy_labels[:len(trajectories)],
                str(video_path),
                title="State Evolution Animation",
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

        if args.multi_agent:
            policy_summary = {
                "label": label,
                "n_agents": int(state_errors.shape[1]),
                "total_reward": float(rewards.sum()),
                "avg_reward_per_agent_step": float(rewards.mean()),
                "transmission_count": int(actions.sum()),
                "transmission_rate_per_agent_step": float(actions.mean()),
                "initial_state_error_mean": float(state_errors[0].mean()),
                "final_state_error_mean": float(state_errors[-1].mean()),
                "avg_state_error_mean": float(state_errors.mean(axis=1).mean()),
                "initial_state_error_per_agent": [float(x) for x in state_errors[0]],
                "final_state_error_per_agent": [float(x) for x in state_errors[-1]],
            }
        else:
            policy_summary = {
                "label": label,
                "total_reward": float(rewards.sum()),
                "avg_reward": float(rewards.mean()),
                "transmission_count": int(actions.sum()),
                "transmission_rate": float(actions.mean()),
                "initial_state_error": float(state_errors[0]),
                "final_state_error": float(state_errors[-1]),
                "avg_state_error": float(state_errors.mean()),
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
