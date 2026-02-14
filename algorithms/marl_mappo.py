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

from utils.marl import (
    CentralValueMLP,
    MLPAgent,
    ValueNorm,
    append_agent_id,
    build_mappo_parser,
    save_mappo_checkpoint,
    save_mappo_training_state,
    load_mappo_training_state,
    build_mappo_hyperparams,
    patch_autoreset_final_obs,
)
from utils.marl.vector_env import create_async_vector_env, create_eval_async_vector_env, stack_vector_obs
from utils.marl_training import (
    setup_device_and_rng,
    load_config_with_overrides,
    create_obs_normalizer,
    print_run_summary,
    setup_shared_reward_normalizer,
    evaluate_and_log,
    log_completed_episodes,
)
from utils.reward_normalization import reset_shared_running_normalizers
from utils.run_utils import prepare_run_directory, save_config_with_hyperparameters, BestModelTracker


class MAPPORolloutBuffer:
    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        n_agents: int,
        obs_dim: int,
        value_dim: int,
        global_obs_dim: Optional[int] = None,
    ) -> None:
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if n_envs <= 0:
            raise ValueError("n_envs must be positive")
        self.n_steps = int(n_steps)
        self.n_envs = int(n_envs)
        self.n_agents = int(n_agents)
        self.obs_dim = int(obs_dim)
        self.value_dim = int(value_dim)
        if self.value_dim <= 0:
            raise ValueError("value_dim must be positive")
        self.global_obs_dim = int(global_obs_dim) if global_obs_dim is not None else self.n_agents * self.obs_dim
        self.reset()

    def reset(self) -> None:
        self.step = 0
        self.obs = np.zeros(
            (self.n_steps, self.n_envs, self.n_agents, self.obs_dim), dtype=np.float32
        )
        self.global_obs = np.zeros(
            (self.n_steps, self.n_envs, self.global_obs_dim), dtype=np.float32
        )
        self.next_global_obs = np.zeros(
            (self.n_steps, self.n_envs, self.global_obs_dim), dtype=np.float32
        )
        self.actions = np.zeros(
            (self.n_steps, self.n_envs, self.n_agents), dtype=np.int64
        )
        self.log_probs = np.zeros(
            (self.n_steps, self.n_envs, self.n_agents), dtype=np.float32
        )
        self.rewards = np.zeros(
            (self.n_steps, self.n_envs, self.value_dim), dtype=np.float32
        )
        self.terminated = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.episode_end = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.n_steps, self.n_envs, self.value_dim), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        global_obs: np.ndarray,
        next_global_obs: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        terminated: np.ndarray,
        episode_end: np.ndarray,
        values: np.ndarray,
    ) -> None:
        if self.step >= self.n_steps:
            raise RuntimeError("Rollout buffer is full")
        idx = self.step
        self.obs[idx] = obs
        self.global_obs[idx] = global_obs
        self.next_global_obs[idx] = next_global_obs
        self.actions[idx] = actions
        self.log_probs[idx] = log_probs
        self.rewards[idx] = rewards
        self.terminated[idx] = terminated.astype(np.float32)
        self.episode_end[idx] = episode_end.astype(np.float32)
        self.values[idx] = values
        self.step += 1

    def get(self) -> Tuple[np.ndarray, ...]:
        if self.step == 0:
            raise RuntimeError("Rollout buffer is empty")
        idx = self.step
        return (
            self.obs[:idx],
            self.global_obs[:idx],
            self.next_global_obs[:idx],
            self.actions[:idx],
            self.log_probs[:idx],
            self.rewards[:idx],
            self.terminated[:idx],
            self.episode_end[:idx],
            self.values[:idx],
        )


def parse_args() -> argparse.Namespace:
    parser = build_mappo_parser(
        description="Train MAPPO (shared actor, centralized critic) on multi-agent NCS env."
    )
    return parser.parse_args()


def _compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    episode_end: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_adv = np.zeros(rewards.shape[1:], dtype=np.float32)
    for t in range(rewards.shape[0] - 1, -1, -1):
        bootstrap_mask = 1.0 - terminated[t].reshape(-1, 1)
        delta = rewards[t] + gamma * next_values[t] * bootstrap_mask - values[t]
        cont = 1.0 - episode_end[t].reshape(-1, 1)
        last_adv = delta + gamma * gae_lambda * cont * last_adv
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
    device, rng = setup_device_and_rng(args.device, args.seed)

    cfg, config_path_str, n_agents, use_agent_id, eval_reward_override, eval_termination_override = (
        load_config_with_overrides(args.config, args.n_agents, not args.no_agent_id)
    )

    resuming = args.resume is not None
    if resuming:
        run_dir = Path(args.resume)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Resume directory does not exist: {run_dir}")
    else:
        run_dir = prepare_run_directory("mappo", args.config, args.output_root)
    rewards_csv_path = run_dir / "training_rewards.csv"
    eval_csv_path = run_dir / "evaluation_rewards.csv"

    if args.n_envs <= 0:
        raise ValueError("n_envs must be positive")

    shared_reward_normalizer, _shared_reward_manager = setup_shared_reward_normalizer(
        cfg.get("reward", {}), run_dir
    )

    env, env_seeds = create_async_vector_env(
        n_envs=args.n_envs,
        n_agents=n_agents,
        episode_length=args.episode_length,
        config_path_str=config_path_str,
        seed=args.seed,
        global_state_enabled=True,
        shared_reward_normalizer=shared_reward_normalizer,
        minimal_info=True,
    )
    eval_env = create_eval_async_vector_env(
        n_eval_envs=args.n_eval_envs,
        n_agents=n_agents,
        episode_length=args.episode_length,
        config_path_str=config_path_str,
        seed=args.seed,
        reward_override=eval_reward_override,
        termination_override=eval_termination_override,
        global_state_enabled=True,
    )

    obs_dim = int(env.single_observation_space.spaces["agent_0"].shape[0])
    n_actions = int(env.single_action_space.spaces["agent_0"].n)
    obs_dict, info = env.reset(seed=env_seeds)
    obs_raw = stack_vector_obs(obs_dict, n_agents)
    global_state_raw = np.asarray(info.get("global_state"), dtype=np.float32)
    if global_state_raw.ndim != 2:
        raise ValueError("global_state must have shape (n_envs, state_dim)")
    global_state_dim = int(global_state_raw.shape[-1])
    actor_input_dim = obs_dim + (n_agents if use_agent_id else 0)
    critic_input_dim = global_state_dim
    value_dim = 1 if args.team_reward else n_agents
    obs_normalizer = create_obs_normalizer(
        obs_dim, args.normalize_obs, args.obs_norm_clip, args.obs_norm_eps
    )

    actor = MLPAgent(
        input_dim=actor_input_dim,
        n_actions=n_actions,
        hidden_dims=tuple(args.hidden_dims),
        activation=args.activation,
        layer_norm=args.layer_norm,
    ).to(device)
    critic = CentralValueMLP(
        input_dim=critic_input_dim,
        n_outputs=value_dim,
        hidden_dims=tuple(args.hidden_dims),
        activation=args.activation,
        layer_norm=args.layer_norm,
        use_popart=args.popart,
        popart_beta=float(args.value_norm_beta),
    ).to(device)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=float(args.learning_rate))
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=float(args.learning_rate))
    if args.popart:
        popart_layer = critic.popart_layer()
        value_normalizer = None
    else:
        popart_layer = None
        value_normalizer = ValueNorm((value_dim,), device=device, beta=float(args.value_norm_beta))
    base_lr = float(args.learning_rate)

    best_model_tracker = BestModelTracker()
    global_step = 0
    episode = 0
    last_eval_step = 0

    if resuming:
        training_state_path = run_dir / "training_state.pt"
        if not training_state_path.exists():
            raise FileNotFoundError(f"No training_state.pt found in {run_dir}")
        counters = load_mappo_training_state(
            training_state_path, actor, critic, actor_optimizer, critic_optimizer,
            value_normalizer, obs_normalizer, best_model_tracker,
        )
        global_step = counters["global_step"]
        episode = counters["episode"]
        last_eval_step = counters["last_eval_step"]
        print(f"Resumed from {run_dir} at step {global_step}")

    def save_checkpoint(path: Path) -> None:
        save_mappo_checkpoint(
            path=path,
            n_agents=n_agents,
            obs_dim=obs_dim,
            n_actions=n_actions,
            use_agent_id=use_agent_id,
            team_reward=args.team_reward,
            agent_hidden_dims=list(args.hidden_dims),
            agent_activation=args.activation,
            agent_layer_norm=args.layer_norm,
            critic_hidden_dims=list(args.hidden_dims),
            critic_activation=args.activation,
            critic_layer_norm=args.layer_norm,
            actor=actor,
            critic=critic,
            obs_normalizer=obs_normalizer,
            popart=args.popart,
        )

    def save_training_state() -> None:
        save_mappo_training_state(
            run_dir / "training_state.pt", actor, critic, actor_optimizer,
            critic_optimizer, value_normalizer, obs_normalizer, best_model_tracker,
            global_step, episode, last_eval_step,
        )

    csv_mode = "a" if resuming else "w"
    with rewards_csv_path.open(csv_mode, newline="", encoding="utf-8") as train_f, \
         eval_csv_path.open(csv_mode, newline="", encoding="utf-8") as eval_f:
        train_writer = csv.writer(train_f)
        eval_writer = csv.writer(eval_f)
        if not resuming:
            train_writer.writerow(["episode", "reward_sum", "length", "steps"])
            eval_writer.writerow(["step", "mean_reward", "std_reward"])

        episode_reward_sums = np.zeros((args.n_envs,), dtype=np.float32)
        episode_lengths = np.zeros((args.n_envs,), dtype=np.int64)

        while global_step < args.total_timesteps:
            buffer = MAPPORolloutBuffer(
                args.n_steps,
                args.n_envs,
                n_agents,
                obs_dim,
                value_dim,
                global_obs_dim=global_state_dim,
            )

            for _ in range(args.n_steps):
                if global_step >= args.total_timesteps:
                    break

                if obs_normalizer is not None:
                    obs = obs_normalizer.normalize(obs_raw, update=True)
                else:
                    obs = obs_raw

                obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)
                if use_agent_id:
                    obs_t = append_agent_id(obs_t, n_agents)
                obs_flat = obs_t.reshape(args.n_envs * n_agents, -1)

                with torch.no_grad():
                    logits = actor(obs_flat)
                    dist = Categorical(logits=logits)
                    actions_t = dist.sample()
                    log_probs_t = dist.log_prob(actions_t)

                    global_obs = global_state_raw.astype(np.float32)
                    global_obs_t = torch.as_tensor(global_obs, device=device, dtype=torch.float32)
                    values_t = critic(global_obs_t)

                actions = actions_t.cpu().numpy().astype(np.int64).reshape(args.n_envs, n_agents)
                log_probs = log_probs_t.cpu().numpy().astype(np.float32).reshape(args.n_envs, n_agents)
                values = values_t.cpu().numpy().astype(np.float32)

                action_dict = {f"agent_{i}": actions[:, i] for i in range(n_agents)}
                next_obs_dict, rewards_arr, terminated, truncated, infos = env.step(action_dict)
                next_obs_raw = stack_vector_obs(next_obs_dict, n_agents)
                next_global_state_raw = np.asarray(infos.get("global_state"), dtype=np.float32)

                terminated_any = np.asarray(terminated, dtype=np.bool_)
                truncated_any = np.asarray(truncated, dtype=np.bool_)
                done = np.logical_or(terminated_any, truncated_any)

                next_obs_for_gae_raw, next_global_state_for_gae = patch_autoreset_final_obs(
                    next_obs_raw, infos, done, n_agents,
                    next_global_state_raw=next_global_state_raw,
                )

                if obs_normalizer is not None:
                    next_obs_for_gae = obs_normalizer.normalize(
                        next_obs_for_gae_raw, update=False
                    )
                else:
                    next_obs_for_gae = next_obs_for_gae_raw

                raw_rewards = np.asarray(rewards_arr, dtype=np.float32)
                if args.team_reward:
                    rewards = raw_rewards.sum(axis=1, keepdims=True)
                else:
                    rewards = raw_rewards

                buffer.add(
                    obs=obs,
                    global_obs=global_obs,
                    next_global_obs=next_global_state_for_gae.astype(np.float32),
                    actions=actions,
                    log_probs=log_probs,
                    rewards=rewards,
                    terminated=terminated_any,
                    episode_end=done,
                    values=values,
                )

                episode_reward_sums += raw_rewards.sum(axis=1)
                episode_lengths += 1
                global_step += args.n_envs

                episode = log_completed_episodes(
                    done_reset=done, episode_reward_sums=episode_reward_sums,
                    global_step=global_step, episode=episode,
                    train_writer=train_writer, train_f=train_f,
                    best_model_tracker=best_model_tracker, run_dir=run_dir,
                    save_checkpoint=save_checkpoint, log_interval=args.log_interval,
                    algo_name="MAPPO",
                    episode_lengths=episode_lengths,
                )

                obs_raw = next_obs_raw
                global_state_raw = next_global_state_raw

                if global_step - last_eval_step >= args.eval_freq:
                    evaluate_and_log(
                        eval_env=eval_env, agent=actor,
                        n_eval_envs=args.n_eval_envs, n_agents=n_agents, n_actions=n_actions,
                        use_agent_id=use_agent_id, device=device,
                        n_episodes=args.n_eval_episodes, seed=args.seed,
                        obs_normalizer=obs_normalizer, eval_writer=eval_writer, eval_f=eval_f,
                        best_model_tracker=best_model_tracker, run_dir=run_dir,
                        save_checkpoint=save_checkpoint, global_step=global_step,
                        algo_name="MAPPO",
                    )
                    last_eval_step = global_step
                    save_training_state()

            if buffer.step == 0:
                continue

            (
                obs_batch,
                global_obs_batch,
                next_global_obs_batch,
                actions_batch,
                log_probs_batch,
                rewards_batch,
                terminated_batch,
                episode_end_batch,
                values_batch,
            ) = buffer.get()

            rollout_len = int(obs_batch.shape[0])
            values_t = torch.as_tensor(values_batch, device=device, dtype=torch.float32)
            next_global_obs_t = torch.as_tensor(
                next_global_obs_batch, device=device, dtype=torch.float32
            )
            with torch.no_grad():
                next_values_t = critic(
                    next_global_obs_t.reshape(-1, critic_input_dim)
                ).reshape(rollout_len, args.n_envs, value_dim)
            if args.popart:
                values_raw = popart_layer.denormalize(values_t).cpu().numpy()
                next_values_raw = popart_layer.denormalize(next_values_t).cpu().numpy()
            else:
                values_raw = value_normalizer.denormalize(values_t).cpu().numpy()
                next_values_raw = value_normalizer.denormalize(next_values_t).cpu().numpy()

            advantages, returns = _compute_gae(
                rewards=rewards_batch,
                values=values_raw,
                next_values=next_values_raw,
                terminated=terminated_batch,
                episode_end=episode_end_batch,
                gamma=float(args.gamma),
                gae_lambda=float(args.gae_lambda),
            )

            obs_t = torch.as_tensor(obs_batch, device=device, dtype=torch.float32)
            global_obs_t = torch.as_tensor(global_obs_batch, device=device, dtype=torch.float32)
            actions_t = torch.as_tensor(actions_batch, device=device, dtype=torch.long)
            old_log_probs_t = torch.as_tensor(log_probs_batch, device=device, dtype=torch.float32)
            advantages_t = torch.as_tensor(advantages, device=device, dtype=torch.float32)
            returns_t = torch.as_tensor(returns, device=device, dtype=torch.float32)
            values_old = torch.as_tensor(values_batch, device=device, dtype=torch.float32)

            if args.popart:
                popart_layer.update_and_correct(returns_t)
                normalized_returns_t = popart_layer.normalize_targets(returns_t)
                # Re-normalize old values to current PopArt scale for correct PPO clipping
                values_old = popart_layer.normalize_targets(
                    torch.as_tensor(values_raw, device=device, dtype=torch.float32)
                )
            else:
                value_normalizer.update(returns_t)
                normalized_returns_t = value_normalizer.normalize(returns_t)

            if args.lr_decay:
                progress = min(float(global_step) / float(args.total_timesteps), 1.0)
                lr = max(0.0, base_lr * (1.0 - progress))
                for optimizer in (actor_optimizer, critic_optimizer):
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr

            if args.team_reward:
                advantages_actor_t = advantages_t.repeat(1, 1, n_agents)
            else:
                advantages_actor_t = advantages_t

            obs_flat = obs_t.reshape(-1, obs_dim)
            actions_flat = actions_t.reshape(-1)
            old_log_probs_flat = old_log_probs_t.reshape(-1)
            advantages_flat = advantages_actor_t.reshape(-1)
            total_samples = int(advantages_flat.shape[0])
            if not args.team_reward:
                returns_flat = normalized_returns_t.reshape(-1)
                values_old_flat = values_old.reshape(-1)

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

            if args.team_reward:
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
                        surr2 = torch.clamp(
                            ratio, 1.0 - args.clip_range, 1.0 + args.clip_range
                        ) * advantages_mb
                        policy_loss = -torch.min(surr1, surr2).mean()
                        entropy = dist.entropy().mean()

                        actor_loss = policy_loss - float(args.ent_coef) * entropy

                        actor_optimizer.zero_grad(set_to_none=True)
                        actor_loss.backward()
                        nn.utils.clip_grad_norm_(actor.parameters(), float(args.max_grad_norm))
                        actor_optimizer.step()

                value_targets = normalized_returns_t.squeeze(-1)
                values_old_targets = values_old.squeeze(-1)
                global_obs_flat = global_obs_t.reshape(-1, critic_input_dim)
                value_targets_flat = value_targets.reshape(-1)
                values_old_flat = values_old_targets.reshape(-1)
                value_total_samples = int(value_targets_flat.shape[0])
                value_batch_size = min(int(args.batch_size), value_total_samples)
                if value_batch_size <= 0:
                    value_batch_size = value_total_samples

                for _ in range(args.n_epochs):
                    value_indices = rng.permutation(value_total_samples)
                    for start in range(0, value_total_samples, value_batch_size):
                        value_idx = value_indices[start:start + value_batch_size]
                        value_idx_t = torch.as_tensor(value_idx, device=device, dtype=torch.long)

                        global_obs_mb = global_obs_flat[value_idx_t]
                        values_pred_mb = critic(global_obs_mb).squeeze(-1)
                        returns_mb = value_targets_flat[value_idx_t]
                        values_old_mb = values_old_flat[value_idx_t]

                        value_pred_clipped = values_old_mb + (
                            values_pred_mb - values_old_mb
                        ).clamp(-float(args.clip_range), float(args.clip_range))

                        error_original = returns_mb - values_pred_mb
                        error_clipped = returns_mb - value_pred_clipped

                        value_loss_original = _huber_loss(error_original, args.huber_delta)
                        value_loss_clipped = _huber_loss(error_clipped, args.huber_delta)
                        value_loss = torch.max(value_loss_original, value_loss_clipped).mean() * float(
                            args.vf_coef
                        )

                        critic_optimizer.zero_grad(set_to_none=True)
                        value_loss.backward()
                        nn.utils.clip_grad_norm_(critic.parameters(), float(args.max_grad_norm))
                        critic_optimizer.step()
            else:
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
                        surr2 = torch.clamp(
                            ratio, 1.0 - args.clip_range, 1.0 + args.clip_range
                        ) * advantages_mb
                        policy_loss = -torch.min(surr1, surr2).mean()
                        entropy = dist.entropy().mean()

                        actor_loss = policy_loss - float(args.ent_coef) * entropy

                        actor_optimizer.zero_grad(set_to_none=True)
                        actor_loss.backward()
                        nn.utils.clip_grad_norm_(actor.parameters(), float(args.max_grad_norm))
                        actor_optimizer.step()

                        time_env_ids = mb_idx_t // n_agents
                        returns_mb = returns_flat[mb_idx_t]
                        values_old_mb = values_old_flat[mb_idx_t]

                        global_obs_mb = global_obs_t.reshape(-1, critic_input_dim)[time_env_ids]
                        values_pred_all = critic(global_obs_mb)
                        values_pred_mb = values_pred_all.gather(1, agent_ids_mb.unsqueeze(-1)).squeeze(-1)

                        value_pred_clipped = values_old_mb + (
                            values_pred_mb - values_old_mb
                        ).clamp(-float(args.clip_range), float(args.clip_range))

                        error_original = returns_mb - values_pred_mb
                        error_clipped = returns_mb - value_pred_clipped

                        value_loss_original = _huber_loss(error_original, args.huber_delta)
                        value_loss_clipped = _huber_loss(error_clipped, args.huber_delta)
                        value_loss = torch.max(value_loss_original, value_loss_clipped).mean() * float(
                            args.vf_coef
                        )

                        critic_optimizer.zero_grad(set_to_none=True)
                        value_loss.backward()
                        nn.utils.clip_grad_norm_(critic.parameters(), float(args.max_grad_norm))
                        critic_optimizer.step()

    latest_path = run_dir / "latest_model.pt"
    save_checkpoint(latest_path)
    save_training_state()

    hyperparams = build_mappo_hyperparams(
        args=args,
        n_agents=n_agents,
        use_agent_id=use_agent_id,
        device=device,
    )
    save_config_with_hyperparameters(run_dir, args.config, "mappo", hyperparams)

    print_run_summary(run_dir, latest_path, rewards_csv_path, eval_csv_path)

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
