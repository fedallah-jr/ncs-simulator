from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.marl_mappo import MAPPORolloutBuffer, _compute_gae, _huber_loss
from utils.marl import (
    CentralValueMLP,
    MLPAgent,
    ValueNorm,
    build_happo_parser,
    save_happo_checkpoint,
    save_happo_training_state,
    load_happo_training_state,
    load_happo_arch_args,
    build_happo_hyperparams,
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


def parse_args() -> argparse.Namespace:
    parser = build_happo_parser(
        description="Train HAPPO (independent actors, centralized critic, sequential update) on multi-agent NCS env."
    )
    return parser.parse_args()


def _get_arch_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Extract architecture-defining args that must stay consistent across resume."""
    return {
        "hidden_dims": list(args.hidden_dims),
        "activation": args.activation,
        "layer_norm": args.layer_norm,
        "popart": args.popart,
    }


def _apply_arch_args(args: argparse.Namespace, arch: Dict[str, Any]) -> None:
    """Override current CLI args with saved architecture args for resume consistency."""
    args.hidden_dims = arch["hidden_dims"]
    args.activation = arch["activation"]
    args.layer_norm = arch["layer_norm"]
    args.popart = arch["popart"]


def main() -> None:
    reset_shared_running_normalizers()
    args = parse_args()
    device, rng = setup_device_and_rng(args.device, args.seed)

    cfg, config_path_str, n_agents, _use_agent_id_ignored, eval_reward_override, eval_termination_override = (
        load_config_with_overrides(args.config, args.n_agents, False, args.set_overrides)
    )
    use_agent_id = False  # HAPPO uses independent actors, no agent ID needed

    resuming = args.resume is not None
    if resuming:
        run_dir = Path(args.resume)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Resume directory does not exist: {run_dir}")
        # Restore architecture args from saved training state so model shapes match
        training_state_path = run_dir / "training_state.pt"
        if not training_state_path.exists():
            raise FileNotFoundError(f"No training_state.pt found in {run_dir}")
        saved_arch = load_happo_arch_args(training_state_path)
        if saved_arch is not None:
            _apply_arch_args(args, saved_arch)
        else:
            print("Warning: training_state.pt has no arch_args; using current CLI args for model shapes")
    else:
        run_dir = prepare_run_directory("happo", args.config, args.output_root)
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
    actor_input_dim = obs_dim  # No agent ID for independent actors
    critic_input_dim = global_state_dim
    value_dim = 1  # Paper-faithful: shared team reward, scalar value
    obs_normalizer = create_obs_normalizer(
        obs_dim, args.normalize_obs, args.obs_norm_clip, args.obs_norm_eps
    )

    # Independent actors: one per agent
    actors: List[MLPAgent] = []
    actor_optimizers: List[torch.optim.Adam] = []
    for _ in range(n_agents):
        actor = MLPAgent(
            input_dim=actor_input_dim,
            n_actions=n_actions,
            hidden_dims=tuple(args.hidden_dims),
            activation=args.activation,
            layer_norm=args.layer_norm,
        ).to(device)
        actors.append(actor)
        actor_optimizers.append(torch.optim.Adam(actor.parameters(), lr=float(args.learning_rate)))

    critic = CentralValueMLP(
        input_dim=critic_input_dim,
        n_outputs=value_dim,
        hidden_dims=tuple(args.hidden_dims),
        activation=args.activation,
        layer_norm=args.layer_norm,
        use_popart=args.popart,
        popart_beta=float(args.value_norm_beta),
    ).to(device)
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
        counters = load_happo_training_state(
            training_state_path, actors, critic, actor_optimizers, critic_optimizer,
            value_normalizer, obs_normalizer, best_model_tracker,
        )
        global_step = counters["global_step"]
        episode = counters["episode"]
        last_eval_step = counters["last_eval_step"]
        print(f"Resumed from {run_dir} at step {global_step}")

    arch_args = _get_arch_args(args)

    def save_checkpoint(path: Path) -> None:
        save_happo_checkpoint(
            path=path,
            n_agents=n_agents,
            obs_dim=obs_dim,
            n_actions=n_actions,
            agent_hidden_dims=list(args.hidden_dims),
            agent_activation=args.activation,
            agent_layer_norm=args.layer_norm,
            critic_hidden_dims=list(args.hidden_dims),
            critic_activation=args.activation,
            critic_layer_norm=args.layer_norm,
            actors=actors,
            critic=critic,
            obs_normalizer=obs_normalizer,
            popart=args.popart,
        )

    def save_training_state() -> None:
        save_happo_training_state(
            run_dir / "training_state.pt", actors, critic, actor_optimizers,
            critic_optimizer, value_normalizer, obs_normalizer, best_model_tracker,
            global_step, episode, last_eval_step,
            arch_args=arch_args,
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

            # ---- Rollout phase ----
            for _ in range(args.n_steps):
                if global_step >= args.total_timesteps:
                    break

                if obs_normalizer is not None:
                    obs = obs_normalizer.normalize(obs_raw, update=True)
                else:
                    obs = obs_raw

                obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)

                with torch.no_grad():
                    all_actions = []
                    all_log_probs = []
                    for agent_id in range(n_agents):
                        obs_i = obs_t[:, agent_id, :]  # (n_envs, obs_dim)
                        logits = actors[agent_id](obs_i)
                        dist = Categorical(logits=logits)
                        act = dist.sample()
                        lp = dist.log_prob(act)
                        all_actions.append(act)
                        all_log_probs.append(lp)

                    actions_t = torch.stack(all_actions, dim=1)  # (n_envs, n_agents)
                    log_probs_t = torch.stack(all_log_probs, dim=1)  # (n_envs, n_agents)

                    global_obs = global_state_raw.astype(np.float32)
                    global_obs_t = torch.as_tensor(global_obs, device=device, dtype=torch.float32)
                    values_t = critic(global_obs_t)

                actions = actions_t.cpu().numpy().astype(np.int64)
                log_probs = log_probs_t.cpu().numpy().astype(np.float32)
                values = values_t.cpu().numpy().astype(np.float32)

                action_dict = {f"agent_{i}": actions[:, i] for i in range(n_agents)}
                next_obs_dict, rewards_arr, terminated, truncated, infos = env.step(action_dict)
                next_obs_raw = stack_vector_obs(next_obs_dict, n_agents)
                next_global_state_raw = np.asarray(infos.get("global_state"), dtype=np.float32)

                terminated_any = np.asarray(terminated, dtype=np.bool_)
                truncated_any = np.asarray(truncated, dtype=np.bool_)
                done = np.logical_or(terminated_any, truncated_any)

                _, next_global_state_for_gae = patch_autoreset_final_obs(
                    next_obs_raw, infos, done, n_agents,
                    next_global_state_raw=next_global_state_raw,
                )

                raw_rewards = np.asarray(rewards_arr, dtype=np.float32)
                # Paper-faithful: sum per-agent rewards into shared team reward
                rewards = raw_rewards.sum(axis=1, keepdims=True)  # (n_envs, 1)

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
                    algo_name="HAPPO",
                    episode_lengths=episode_lengths,
                )

                obs_raw = next_obs_raw
                global_state_raw = next_global_state_raw

                if global_step - last_eval_step >= args.eval_freq:
                    evaluate_and_log(
                        eval_env=eval_env, agent=actors,
                        n_eval_envs=args.n_eval_envs, n_agents=n_agents, n_actions=n_actions,
                        use_agent_id=use_agent_id, device=device,
                        n_episodes=args.n_eval_episodes, seed=args.seed,
                        obs_normalizer=obs_normalizer, eval_writer=eval_writer, eval_f=eval_f,
                        best_model_tracker=best_model_tracker, run_dir=run_dir,
                        save_checkpoint=save_checkpoint, global_step=global_step,
                        algo_name="HAPPO",
                    )
                    last_eval_step = global_step
                    save_training_state()

            if buffer.step == 0:
                continue

            # ---- Get rollout data ----
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

            # ---- GAE computation (shared scalar value) ----
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
            # advantages shape: (T, E, 1) — scalar shared advantage

            # Prepare critic training data
            global_obs_t = torch.as_tensor(global_obs_batch, device=device, dtype=torch.float32)
            returns_t = torch.as_tensor(returns, device=device, dtype=torch.float32)
            values_old = torch.as_tensor(values_batch, device=device, dtype=torch.float32)

            if args.popart:
                popart_layer.update_and_correct(returns_t)
                normalized_returns_t = popart_layer.normalize_targets(returns_t)
                values_old = popart_layer.normalize_targets(
                    torch.as_tensor(values_raw, device=device, dtype=torch.float32)
                )
            else:
                value_normalizer.update(returns_t)
                normalized_returns_t = value_normalizer.normalize(returns_t)

            # ---- Learning rate decay ----
            if args.lr_decay:
                progress = min(float(global_step) / float(args.total_timesteps), 1.0)
                lr = max(0.0, base_lr * (1.0 - progress))
                for opt in actor_optimizers:
                    for param_group in opt.param_groups:
                        param_group["lr"] = lr
                for param_group in critic_optimizer.param_groups:
                    param_group["lr"] = lr

            # ---- Sequential actor update (HAPPO core) ----
            # Shared scalar advantage broadcast to all agents
            # advantages shape: (T, E, 1) -> squeeze to (T, E)
            shared_advantages = advantages.squeeze(-1)  # (T, E)

            factor = np.ones((rollout_len, args.n_envs, 1), dtype=np.float32)

            if args.fixed_order:
                agent_order = np.arange(n_agents)
            else:
                agent_order = rng.permutation(n_agents)

            for agent_id in agent_order:
                agent_id = int(agent_id)

                # Extract agent i data
                obs_i = obs_batch[:, :, agent_id, :]       # (T, E, obs_dim)
                actions_i = actions_batch[:, :, agent_id]    # (T, E)
                old_lp_i = log_probs_batch[:, :, agent_id]   # (T, E)

                # Pre-update log probs for factor computation
                obs_i_t = torch.as_tensor(obs_i, device=device, dtype=torch.float32)
                actions_i_t = torch.as_tensor(actions_i, device=device, dtype=torch.long)
                with torch.no_grad():
                    pre_logits = actors[agent_id](obs_i_t.reshape(-1, obs_dim))
                    pre_dist = Categorical(logits=pre_logits)
                    pre_update_lp = pre_dist.log_prob(actions_i_t.reshape(-1))

                # Flatten for mini-batch PPO
                obs_flat = obs_i_t.reshape(-1, obs_dim)
                actions_flat = actions_i_t.reshape(-1)
                old_lp_flat = torch.as_tensor(old_lp_i.reshape(-1), device=device, dtype=torch.float32)
                adv_flat = torch.as_tensor(shared_advantages.reshape(-1), device=device, dtype=torch.float32)
                factor_flat = torch.as_tensor(factor.reshape(-1, 1), device=device, dtype=torch.float32)

                # Normalize advantages
                adv_mean = adv_flat.mean()
                adv_std = adv_flat.std()
                if float(adv_std.item()) > 1e-8:
                    adv_flat = (adv_flat - adv_mean) / (adv_std + 1e-8)
                else:
                    adv_flat = adv_flat - adv_mean

                total_samples = int(adv_flat.shape[0])
                batch_size = min(int(args.batch_size), total_samples)
                if batch_size <= 0:
                    batch_size = total_samples

                # PPO epochs with factor weighting
                for _ in range(args.n_epochs):
                    indices = rng.permutation(total_samples)
                    for start in range(0, total_samples, batch_size):
                        mb_idx = indices[start:start + batch_size]
                        mb_idx_t = torch.as_tensor(mb_idx, device=device, dtype=torch.long)

                        obs_mb = obs_flat[mb_idx_t]
                        actions_mb = actions_flat[mb_idx_t]
                        old_lp_mb = old_lp_flat[mb_idx_t]
                        adv_mb = adv_flat[mb_idx_t]
                        factor_mb = factor_flat[mb_idx_t]  # (mb, 1)

                        logits = actors[agent_id](obs_mb)
                        dist = Categorical(logits=logits)
                        new_lp = dist.log_prob(actions_mb)
                        ratio = (new_lp - old_lp_mb).exp()

                        surr1 = ratio * adv_mb
                        surr2 = torch.clamp(
                            ratio, 1.0 - args.clip_range, 1.0 + args.clip_range
                        ) * adv_mb
                        policy_loss = -(factor_mb.squeeze(-1) * torch.min(surr1, surr2)).mean()
                        entropy = dist.entropy().mean()

                        actor_loss = policy_loss - float(args.ent_coef) * entropy

                        actor_optimizers[agent_id].zero_grad(set_to_none=True)
                        actor_loss.backward()
                        nn.utils.clip_grad_norm_(actors[agent_id].parameters(), float(args.max_grad_norm))
                        actor_optimizers[agent_id].step()

                # Post-update: compute exact factor for next agent (no clamping)
                with torch.no_grad():
                    post_logits = actors[agent_id](obs_i_t.reshape(-1, obs_dim))
                    post_dist = Categorical(logits=post_logits)
                    post_update_lp = post_dist.log_prob(actions_i_t.reshape(-1))
                    ratio_i = (post_update_lp - pre_update_lp).exp()
                    factor = factor * ratio_i.cpu().numpy().reshape(rollout_len, args.n_envs, 1)

            # ---- Critic update (scalar shared value) ----
            # normalized_returns_t shape: (T, E, 1), values_old shape: (T, E, 1)
            value_targets = normalized_returns_t.squeeze(-1)  # (T, E)
            values_old_targets = values_old.squeeze(-1)  # (T, E)
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

    latest_path = run_dir / "latest_model.pt"
    save_checkpoint(latest_path)
    save_training_state()

    hyperparams = build_happo_hyperparams(
        args=args,
        n_agents=n_agents,
        device=device,
    )
    save_config_with_hyperparameters(run_dir, args.config, "happo", hyperparams)

    print_run_summary(run_dir, latest_path, rewards_csv_path, eval_csv_path)

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
