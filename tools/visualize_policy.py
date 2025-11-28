"""
Policy Visualization Tool for Networked Control Systems

This tool visualizes the state evolution of policies trained or defined in the NCS simulator.
It supports both learned policies (Stable-Baselines3) and heuristic policies.

Usage:
    # Visualize a trained PPO policy
    python -m tools.visualize_policy --config configs/perfect_comm.json --policy path/to/model.zip --policy-type sb3

    # Visualize a heuristic policy
    python -m tools.visualize_policy --config configs/perfect_comm.json --policy always_send --policy-type heuristic

    # Compare multiple policies
    python -m tools.visualize_policy --config configs/perfect_comm.json \
        --policies path/to/ppo.zip always_send send_every_5 \
        --policy-types sb3 heuristic heuristic \
        --labels "PPO" "Always Send" "Send Every 5"
"""

import argparse
import os
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
import warnings

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

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
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

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
            model = algorithm.load(model_path, env=env)
            print(f"✓ Loaded {algorithm.__name__} model from {model_path}")
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
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax import flatten_util
        from algorithms.openai_es import PolicyNet
    except ImportError:
        raise ImportError(
            "jax, flax, and evosax are required to load ES policies. "
            "Install with: pip install jax jaxlib flax evosax"
        )

    # Load flattened params
    try:
        data = np.load(model_path)
        flat_params = data['flat_params']
    except Exception as e:
        raise ValueError(f"Could not load numpy data from {model_path}: {e}")

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

    model = PolicyNet(action_dim=action_dim)

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


def load_policy(policy_path: str, policy_type: str, env: Any, n_agents: int = 1, seed: Optional[int] = None):
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
    else:
        raise ValueError(f"Unknown policy type: {policy_type}. Use 'sb3', 'es', or 'heuristic'")


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
        choices=['sb3', 'es', 'heuristic'],
        help='Type of policy: sb3, es, or heuristic'
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
        choices=['sb3', 'es', 'heuristic'],
        help='List of policy types corresponding to --policies'
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
        help='Number of agents (default: 1, only single-agent supported currently)'
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
    os.makedirs(args.output_dir, exist_ok=True)

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
    env = SingleAgentWrapper(make_env)

    # Get state_dim from a temporary instance
    temp_env = make_env()
    state_dim = temp_env.state_dim
    temp_env.close()

    print(f"✓ Environment created (state_dim={state_dim}, episode_length={args.episode_length})\n")

    # Load policies and run episodes
    trajectories = []
    print(f"Running simulations for {len(policies_to_load)} policy/policies...\n")

    for i, (policy_path, policy_type, label) in enumerate(zip(policies_to_load, policy_types, policy_labels)):
        print(f"[{i+1}/{len(policies_to_load)}] {label}")
        print(f"  Type: {policy_type}")
        print(f"  Path/Name: {policy_path}")

        # Reset environment to ensure same initial conditions and process noise for fair comparison
        # This ensures all policies are evaluated on identical random seeds
        obs, info = env.reset(seed=args.seed)

        # Load policy
        try:
            policy = load_policy(policy_path, policy_type, env, n_agents=args.n_agents, seed=args.seed)
        except Exception as e:
            print(f"  ✗ Error loading policy: {e}\n")
            continue

        # Run episode
        print(f"  Running episode...")
        traj = run_episode(env, policy, args.episode_length, deterministic=True)

        # Print summary statistics
        total_reward = np.sum(traj['rewards'])
        tx_count = np.sum(traj['actions'])
        final_error = traj['state_errors'][-1]
        print(f"  ✓ Episode complete")
        print(f"    - Total reward: {total_reward:.2f}")
        print(f"    - Transmissions: {int(tx_count)}/{args.episode_length}")
        print(f"    - Final state error: {final_error:.4f}\n")

        trajectories.append(traj)

    if len(trajectories) == 0:
        print("No successful policy runs. Exiting.")
        return

    # Generate plots
    print(f"Generating visualizations...\n")

    # Plot 1: State evolution 2D
    plot1_path = os.path.join(args.output_dir, f"{output_prefix}_state_evolution.png")
    plot_state_evolution_2d(
        trajectories,
        policy_labels[:len(trajectories)],
        plot1_path,
        title="State Evolution in 2D State Space",
        show_estimates=args.show_estimates
    )

    # Plot 2: Detailed analysis
    plot2_path = os.path.join(args.output_dir, f"{output_prefix}_detailed_analysis.png")
    plot_detailed_analysis(
        trajectories,
        policy_labels[:len(trajectories)],
        plot2_path,
        title="Detailed Policy Analysis"
    )

    # Generate video animation if requested
    if args.generate_video:
        print(f"\nGenerating animation...\n")
        video_path = os.path.join(args.output_dir, f"{output_prefix}_animation.mp4")
        create_state_evolution_animation(
            trajectories,
            policy_labels[:len(trajectories)],
            video_path,
            title="State Evolution Animation",
            fps=args.video_fps,
            speedup=args.video_speedup,
            show_estimates=args.show_estimates
        )

    # Save summary statistics to JSON
    summary_path = os.path.join(args.output_dir, f"{output_prefix}_summary.json")
    summary = {
        'config': args.config,
        'episode_length': args.episode_length,
        'seed': args.seed,
        'policies': []
    }

    for label, traj in zip(policy_labels[:len(trajectories)], trajectories):
        policy_summary = {
            'label': label,
            'total_reward': float(np.sum(traj['rewards'])),
            'avg_reward': float(np.mean(traj['rewards'])),
            'transmission_count': int(np.sum(traj['actions'])),
            'transmission_rate': float(np.mean(traj['actions'])),
            'initial_state_error': float(traj['state_errors'][0]),
            'final_state_error': float(traj['state_errors'][-1]),
            'avg_state_error': float(np.mean(traj['state_errors'])),
        }
        summary['policies'].append(policy_summary)

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary statistics to {summary_path}")

    print(f"\n{'='*60}")
    print(f"Visualization complete!")
    print(f"Output files saved to: {args.output_dir}/")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
