from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ncs_env.config import load_config
from ncs_env.env import NCS_Env
from utils.marl import (
    CentralValueMLP,
    MLPAgent,
    ValueNorm,
    append_agent_id,
    run_evaluation,
    select_device,
    stack_obs,
)
from utils.reward_normalization import reset_shared_running_normalizers
from utils.run_utils import prepare_run_directory, save_config_with_hyperparameters


class MAPPORolloutBuffer:
    def __init__(self, n_steps: int, n_agents: int, obs_dim: int) -> None:
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        self.n_steps = int(n_steps)
        self.n_agents = int(n_agents)
        self.obs_dim = int(obs_dim)
        self.global_obs_dim = self.n_agents * self.obs_dim
        self.reset()

    def reset(self) -> None:
        self.step = 0
        self.obs = np.zeros((self.n_steps, self.n_agents, self.obs_dim), dtype=np.float32)
        self.global_obs = np.zeros((self.n_steps, self.global_obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.n_steps, self.n_agents), dtype=np.int64)
        self.log_probs = np.zeros((self.n_steps, self.n_agents), dtype=np.float32)
        self.rewards = np.zeros((self.n_steps, self.n_agents), dtype=np.float32)
        self.dones = np.zeros((self.n_steps,), dtype=np.float32)
        self.terminated = np.zeros((self.n_steps,), dtype=np.float32)
        self.values = np.zeros((self.n_steps, self.n_agents), dtype=np.float32)
        self.next_values = np.zeros((self.n_steps, self.n_agents), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        global_obs: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        done: bool,
        terminated: bool,
        values: np.ndarray,
        next_values: np.ndarray,
    ) -> None:
        if self.step >= self.n_steps:
            raise RuntimeError("Rollout buffer is full")
        idx = self.step
        self.obs[idx] = obs
        self.global_obs[idx] = global_obs
        self.actions[idx] = actions
        self.log_probs[idx] = log_probs
        self.rewards[idx] = rewards
        self.dones[idx] = float(done)
        self.terminated[idx] = float(terminated)
        self.values[idx] = values
        self.next_values[idx] = next_values
        self.step += 1

    def get(self) -> Tuple[np.ndarray, ...]:
        if self.step == 0:
            raise RuntimeError("Rollout buffer is empty")
        idx = self.step
        return (
            self.obs[:idx],
            self.global_obs[:idx],
            self.actions[:idx],
            self.log_probs[:idx],
            self.rewards[:idx],
            self.dones[:idx],
            self.terminated[:idx],
            self.values[:idx],
            self.next_values[:idx],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MAPPO (shared actor, centralized critic) on multi-agent NCS env.")
    parser.add_argument("--config", type=Path, default=None, help="Config JSON path.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"), help="Output root directory.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--n-agents", type=int, default=3, help="Number of agents (overridden by config if present).")
    parser.add_argument("--episode-length", type=int, default=500, help="Episode length.")
    parser.add_argument("--total-timesteps", type=int, default=200_000, help="Total environment steps.")

    parser.add_argument("--n-steps", type=int, default=256, help="Rollout length per PPO update.")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size for PPO actor updates.")
    parser.add_argument("--n-epochs", type=int, default=4, help="Number of PPO update epochs per rollout.")

    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda.")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clipping range.")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient.")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value loss coefficient.")
    parser.add_argument("--huber-delta", type=float, default=10.0, help="Huber loss delta for value loss.")
    parser.add_argument("--max-grad-norm", type=float, default=10.0, help="Gradient clipping L2 norm.")

    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 128], help="MLP hidden dims.")
    parser.add_argument("--activation", type=str, default="tanh", choices=["relu", "tanh", "elu"], help="Activation.")
    parser.add_argument("--layer-norm", action="store_true", help="Enable LayerNorm in MLP.")
    parser.add_argument("--no-agent-id", action="store_true", help="Disable appending one-hot agent id.")

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Torch device.")
    parser.add_argument("--log-interval", type=int, default=10, help="Print every N episodes.")

    parser.add_argument("--eval-freq", type=int, default=2500, help="Evaluation frequency in env steps.")
    parser.add_argument("--n-eval-episodes", type=int, default=5, help="Number of evaluation episodes.")
    return parser.parse_args()


def _compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    episode_dones: np.ndarray,
    terminated: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_adv = np.zeros((rewards.shape[1],), dtype=np.float32)
    for t in range(rewards.shape[0] - 1, -1, -1):
        mask = 1.0 - float(episode_dones[t])
        bootstrap = 1.0 - float(terminated[t])
        delta = rewards[t] + gamma * next_values[t] * bootstrap - values[t]
        last_adv = delta + gamma * gae_lambda * mask * last_adv
        advantages[t] = last_adv
    returns = advantages + values
    return advantages, returns


def _append_agent_id_flat(obs_flat: torch.Tensor, agent_ids: torch.Tensor, n_agents: int) -> torch.Tensor:
    one_hot = F.one_hot(agent_ids, num_classes=n_agents).to(dtype=obs_flat.dtype)
    return torch.cat([obs_flat, one_hot], dim=-1)


def _huber_loss(error: torch.Tensor, delta: float) -> torch.Tensor:
    abs_error = torch.abs(error)
    quadratic = torch.minimum(abs_error, torch.tensor(delta, device=error.device))
    linear = abs_error - quadratic
    return 0.5 * quadratic ** 2 + float(delta) * linear


def main() -> None:
    reset_shared_running_normalizers()
    args = parse_args()
    device = select_device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed if args.seed is not None else 0)

    config_path_str = str(args.config) if args.config is not None else None
    cfg = load_config(config_path_str)
    system_cfg = cfg.get("system", {})
    n_agents = int(system_cfg.get("n_agents", args.n_agents))
    use_agent_id = not args.no_agent_id

    # Load evaluation reward config if present
    eval_reward_override: Optional[Dict[str, Any]] = None
    reward_cfg = cfg.get("reward", {})
    eval_reward_cfg = reward_cfg.get("evaluation", None)
    if isinstance(eval_reward_cfg, dict):
        eval_reward_override = eval_reward_cfg
    eval_termination_override: Optional[Dict[str, Any]] = None
    termination_cfg = cfg.get("termination", {})
    eval_termination_cfg = termination_cfg.get("evaluation", None)
    if isinstance(eval_termination_cfg, dict):
        eval_termination_override = eval_termination_cfg

    run_dir = prepare_run_directory("mappo", args.config, args.output_root)
    rewards_csv_path = run_dir / "training_rewards.csv"
    eval_csv_path = run_dir / "evaluation_rewards.csv"

    env = NCS_Env(
        n_agents=n_agents,
        episode_length=args.episode_length,
        config_path=config_path_str,
        seed=args.seed,
    )

    eval_env = NCS_Env(
        n_agents=n_agents,
        episode_length=args.episode_length,
        config_path=config_path_str,
        seed=args.seed,
        reward_override=eval_reward_override,
        termination_override=eval_termination_override,
        freeze_running_normalization=True,
    )

    obs_dim = int(env.observation_space.spaces["agent_0"].shape[0])
    n_actions = int(env.action_space.spaces["agent_0"].n)
    actor_input_dim = obs_dim + (n_agents if use_agent_id else 0)
    critic_input_dim = obs_dim * n_agents

    actor = MLPAgent(
        input_dim=actor_input_dim,
        n_actions=n_actions,
        hidden_dims=tuple(args.hidden_dims),
        activation=args.activation,
        layer_norm=args.layer_norm,
    ).to(device)
    critic = CentralValueMLP(
        input_dim=critic_input_dim,
        n_outputs=n_agents,
        hidden_dims=tuple(args.hidden_dims),
        activation=args.activation,
        layer_norm=args.layer_norm,
    ).to(device)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=float(args.learning_rate))
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=float(args.learning_rate))
    value_normalizer = ValueNorm(n_agents, device=device)

    best_eval_reward = -float("inf")
    global_step = 0
    episode = 0
    last_eval_step = 0
    episode_reward_sum = 0.0
    episode_length = 0

    def save_checkpoint(path: Path) -> None:
        ckpt: Dict[str, Any] = {
            "algorithm": "mappo",
            "n_agents": n_agents,
            "obs_dim": obs_dim,
            "n_actions": n_actions,
            "use_agent_id": use_agent_id,
            "parameter_sharing": True,
            "agent_hidden_dims": list(args.hidden_dims),
            "agent_activation": args.activation,
            "agent_layer_norm": args.layer_norm,
            "dueling": False,
            "stream_hidden_dim": None,
            "agent_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "critic_hidden_dims": list(args.hidden_dims),
            "critic_activation": args.activation,
            "critic_layer_norm": args.layer_norm,
        }
        torch.save(ckpt, path)

    with rewards_csv_path.open("w", newline="", encoding="utf-8") as train_f, \
         eval_csv_path.open("w", newline="", encoding="utf-8") as eval_f:
        train_writer = csv.writer(train_f)
        train_writer.writerow(["episode", "reward_sum", "length", "steps"])
        eval_writer = csv.writer(eval_f)
        eval_writer.writerow(["step", "mean_reward", "std_reward"])

        obs_dict, _info = env.reset(seed=args.seed)
        obs = stack_obs(obs_dict, n_agents)

        while global_step < args.total_timesteps:
            buffer = MAPPORolloutBuffer(args.n_steps, n_agents, obs_dim)

            for _ in range(args.n_steps):
                if global_step >= args.total_timesteps:
                    break

                obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
                if use_agent_id:
                    obs_t = append_agent_id(obs_t, n_agents)
                obs_flat = obs_t.view(n_agents, -1)

                with torch.no_grad():
                    logits = actor(obs_flat)
                    dist = Categorical(logits=logits)
                    actions_t = dist.sample()
                    log_probs_t = dist.log_prob(actions_t)

                    global_obs = obs.reshape(-1).astype(np.float32)
                    global_obs_t = torch.as_tensor(global_obs, device=device, dtype=torch.float32).unsqueeze(0)
                    values_t = critic(global_obs_t).squeeze(0)

                actions = actions_t.cpu().numpy().astype(np.int64)
                log_probs = log_probs_t.cpu().numpy().astype(np.float32)
                values = values_t.cpu().numpy().astype(np.float32)

                action_dict = {f"agent_{i}": int(actions[i]) for i in range(n_agents)}
                next_obs_dict, rewards_dict, terminated, truncated, _infos = env.step(action_dict)
                next_obs = stack_obs(next_obs_dict, n_agents)
                rewards = np.asarray([rewards_dict[f"agent_{i}"] for i in range(n_agents)], dtype=np.float32)

                term = any(terminated[f"agent_{i}"] for i in range(n_agents))
                trunc = any(truncated[f"agent_{i}"] for i in range(n_agents))
                done = term or trunc

                with torch.no_grad():
                    next_global_obs = next_obs.reshape(-1).astype(np.float32)
                    next_global_obs_t = torch.as_tensor(
                        next_global_obs, device=device, dtype=torch.float32
                    ).unsqueeze(0)
                    next_values_t = critic(next_global_obs_t).squeeze(0)

                buffer.add(
                    obs=obs,
                    global_obs=global_obs,
                    actions=actions,
                    log_probs=log_probs,
                    rewards=rewards,
                    done=done,
                    terminated=term,
                    values=values,
                    next_values=next_values_t.cpu().numpy().astype(np.float32),
                )

                episode_reward_sum += float(rewards.sum())
                episode_length += 1
                global_step += 1

                if done:
                    train_writer.writerow([episode, episode_reward_sum, episode_length, global_step])
                    train_f.flush()
                    if episode % args.log_interval == 0:
                        print(
                            f"[MAPPO] episode={episode} steps={global_step} "
                            f"reward_sum={episode_reward_sum:.3f}"
                        )
                    episode += 1
                    episode_reward_sum = 0.0
                    episode_length = 0
                    episode_seed = None if args.seed is None else args.seed + episode
                    obs_dict, _info = env.reset(seed=episode_seed)
                    obs = stack_obs(obs_dict, n_agents)
                else:
                    obs = next_obs

                if global_step - last_eval_step >= args.eval_freq:
                    mean_eval_reward, std_eval_reward, _ = run_evaluation(
                        env=eval_env,
                        agent=actor,
                        n_agents=n_agents,
                        n_actions=n_actions,
                        use_agent_id=use_agent_id,
                        device=device,
                        n_episodes=args.n_eval_episodes,
                        seed=args.seed,
                    )
                    eval_writer.writerow([global_step, mean_eval_reward, std_eval_reward])
                    eval_f.flush()

                    if mean_eval_reward > best_eval_reward:
                        best_eval_reward = mean_eval_reward
                        save_checkpoint(run_dir / "best_model.pt")

                    print(
                        f"[MAPPO] Eval at step {global_step}: "
                        f"mean_reward={mean_eval_reward:.3f} std={std_eval_reward:.3f}"
                    )
                    last_eval_step = global_step

            if buffer.step == 0:
                continue

            (
                obs_batch,
                global_obs_batch,
                actions_batch,
                log_probs_batch,
                rewards_batch,
                dones_batch,
                terminated_batch,
                values_batch,
                next_values_batch,
            ) = buffer.get()

            advantages, returns = _compute_gae(
                rewards=rewards_batch,
                values=values_batch,
                next_values=next_values_batch,
                episode_dones=dones_batch,
                terminated=terminated_batch,
                gamma=float(args.gamma),
                gae_lambda=float(args.gae_lambda),
            )

            obs_t = torch.as_tensor(obs_batch, device=device, dtype=torch.float32)
            global_obs_t = torch.as_tensor(global_obs_batch, device=device, dtype=torch.float32)
            actions_t = torch.as_tensor(actions_batch, device=device, dtype=torch.long)
            old_log_probs_t = torch.as_tensor(log_probs_batch, device=device, dtype=torch.float32)
            advantages_t = torch.as_tensor(advantages, device=device, dtype=torch.float32)
            returns_t = torch.as_tensor(returns, device=device, dtype=torch.float32)
            value_normalizer.update(returns_t)

            obs_flat = obs_t.reshape(-1, obs_dim)
            actions_flat = actions_t.reshape(-1)
            old_log_probs_flat = old_log_probs_t.reshape(-1)
            advantages_flat = advantages_t.reshape(-1)
            total_samples = int(advantages_flat.shape[0])

            adv_mean = advantages_flat.mean()
            adv_std = advantages_flat.std()
            if float(adv_std.item()) > 1e-8:
                advantages_flat = (advantages_flat - adv_mean) / (adv_std + 1e-8)
            else:
                advantages_flat = advantages_flat - adv_mean

            agent_ids = torch.arange(total_samples, device=device) % n_agents
            batch_size = min(int(args.batch_size), total_samples)
            if batch_size <= 0:
                batch_size = total_samples

            for _ in range(args.n_epochs):
                indices = rng.permutation(total_samples)
                for start in range(0, total_samples, batch_size):
                    mb_idx = indices[start:start + batch_size]
                    mb_idx_t = torch.as_tensor(mb_idx, device=device, dtype=torch.long)

                    obs_mb = obs_flat[mb_idx_t]
                    actions_mb = actions_flat[mb_idx_t]
                    old_log_probs_mb = old_log_probs_flat[mb_idx_t]
                    advantages_mb = advantages_flat[mb_idx_t]
                    agent_ids_mb = agent_ids[mb_idx_t]

                    if use_agent_id:
                        obs_in = _append_agent_id_flat(obs_mb, agent_ids_mb, n_agents)
                    else:
                        obs_in = obs_mb

                    logits = actor(obs_in)
                    dist = Categorical(logits=logits)
                    new_log_probs = dist.log_prob(actions_mb)
                    ratio = (new_log_probs - old_log_probs_mb).exp()

                    surr1 = ratio * advantages_mb
                    surr2 = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range) * advantages_mb
                    policy_loss = -torch.min(surr1, surr2).mean()
                    entropy = dist.entropy().mean()

                    actor_loss = policy_loss - float(args.ent_coef) * entropy

                    actor_optimizer.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), float(args.max_grad_norm))
                    actor_optimizer.step()

                values_pred = critic(global_obs_t)
                values_old = torch.as_tensor(values_batch, device=device, dtype=torch.float32)

                normalized_returns = value_normalizer.normalize(returns_t)
                normalized_values = value_normalizer.normalize(values_pred)
                normalized_values_old = value_normalizer.normalize(values_old)

                value_pred_clipped = normalized_values_old + (
                    normalized_values - normalized_values_old
                ).clamp(-float(args.clip_range), float(args.clip_range))

                error_original = normalized_returns - normalized_values
                error_clipped = normalized_returns - value_pred_clipped

                value_loss_original = _huber_loss(error_original, args.huber_delta)
                value_loss_clipped = _huber_loss(error_clipped, args.huber_delta)
                value_loss = torch.max(value_loss_original, value_loss_clipped).mean() * float(args.vf_coef)

                critic_optimizer.zero_grad(set_to_none=True)
                value_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), float(args.max_grad_norm))
                critic_optimizer.step()

    latest_path = run_dir / "latest_model.pt"
    save_checkpoint(latest_path)

    hyperparams: Dict[str, Any] = {
        "total_timesteps": args.total_timesteps,
        "episode_length": args.episode_length,
        "n_agents": n_agents,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "huber_delta": args.huber_delta,
        "value_norm": True,
        "max_grad_norm": args.max_grad_norm,
        "hidden_dims": list(args.hidden_dims),
        "activation": args.activation,
        "layer_norm": args.layer_norm,
        "use_agent_id": use_agent_id,
        "eval_freq": args.eval_freq,
        "n_eval_episodes": args.n_eval_episodes,
        "device": str(device),
        "seed": args.seed,
    }
    save_config_with_hyperparameters(run_dir, args.config, "mappo", hyperparams)

    print(f"Run artifacts stored in {run_dir}")
    print(f"  - Latest model: {latest_path}")
    print(f"  - Best model: {run_dir / 'best_model.pt'}")
    print(f"  - Training rewards: {rewards_csv_path}")
    print(f"  - Evaluation rewards: {eval_csv_path}")
    print(f"  - Config with hyperparameters: {run_dir / 'config.json'}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
