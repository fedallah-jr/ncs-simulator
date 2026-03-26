"""Reimplementation of the reference DIAL codebase:
https://github.com/minqi/learning-to-communicate-pytorch

This version adapts the original shared-parameter recurrent DIAL setup to the
vectorized NCS environment used in this repository.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.marl.networks import DialRNNAgent, DRU, route_messages, append_agent_id, dial_rnn_forward_batched
from utils.marl.learners import MARLDIALLearner
from utils.marl.common import (
    run_evaluation_vectorized_seeded,
    epsilon_by_step,
    patch_autoreset_final_obs,
)
from utils.marl.args_builder import build_base_qlearning_parser
from utils.marl.checkpoint_utils import (
    save_dial_rnn_checkpoint,
    save_dial_rnn_training_state,
    load_dial_rnn_training_state,
)
from utils.marl_training import (
    setup_device_and_rng,
    load_config_with_overrides,
    resolve_training_eval_baseline,
    create_obs_normalizer,
    print_run_summary,
    setup_shared_reward_normalizer,
    log_completed_episodes,
)
from utils.marl.vector_env import (
    create_async_vector_env,
    create_eval_async_vector_env,
    stack_vector_obs,
)
from utils.reward_normalization import reset_shared_running_normalizers
from utils.run_utils import (
    prepare_run_directory,
    save_config_with_hyperparameters,
    BestModelTracker,
)


def parse_args() -> argparse.Namespace:
    parser = build_base_qlearning_parser(
        description="Train MARL-DIAL on multi-agent NCS env.",
        include_replay_buffer_args=False,
        include_mlp_arch_args=False,
    )
    parser.add_argument("--comm-dim", type=int, default=4, help="Message dimension per agent")
    parser.add_argument("--dru-sigma", type=float, default=2.0, help="DRU noise std")
    parser.add_argument("--rnn-hidden-dim", type=int, default=128)
    parser.add_argument("--rnn-layers", type=int, default=2)
    parser.add_argument("--batch-episodes", type=int, default=32, help="Episodes per training batch")
    parser.add_argument("--target-update-steps", type=int, default=100,
                        help="Target net sync every N optimizer steps (reference: step_target)")
    parser.add_argument("--momentum", type=float, default=0.95, help="RMSprop momentum")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Recurrent DIAL evaluation
# ---------------------------------------------------------------------------

def _make_dial_rnn_eval_action_selector(
    agent: DialRNNAgent,
    dru: DRU,
    n_eval_envs: int,
    n_agents: int,
    n_actions: int,
    comm_dim: int,
    device: torch.device,
    rnn_layers: int,
    rnn_hidden_dim: int,
):
    """Create a stateful action selector closure for recurrent DIAL evaluation."""
    hidden = torch.zeros(rnn_layers, n_eval_envs * n_agents, rnn_hidden_dim, device=device)
    prev_msg_logits = np.zeros((n_eval_envs, n_agents, comm_dim), dtype=np.float32)
    prev_actions = np.full((n_eval_envs, n_agents), n_actions, dtype=np.int64)

    def reset_eval_state():
        hidden.zero_()
        prev_msg_logits[:] = 0.0
        prev_actions[:] = n_actions

    @torch.no_grad()
    def action_selector(obs: np.ndarray) -> np.ndarray:
        nonlocal hidden
        obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)
        msg_t = torch.as_tensor(prev_msg_logits, device=device, dtype=torch.float32)
        msg_post_dru = dru(msg_t, train_mode=False)
        recv_msg = route_messages(msg_post_dru, n_agents)

        prev_act_t = torch.as_tensor(prev_actions, device=device, dtype=torch.long)
        q_values, msg_logits, new_hidden = dial_rnn_forward_batched(
            agent,
            obs_t,
            prev_act_t,
            recv_msg,
            hidden,
        )
        hidden = new_hidden

        actions = q_values.argmax(dim=-1).cpu().numpy().astype(np.int64)
        prev_msg_logits[:] = msg_logits.cpu().numpy()
        prev_actions[:] = actions
        return actions

    return action_selector, reset_eval_state


def _format_eta(remaining_seconds: float) -> str:
    """Format remaining seconds as HH:MM:SS."""
    total = int(max(0, remaining_seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _evaluate_and_log_dial_rnn(
    *,
    eval_env,
    agent: DialRNNAgent,
    dru: DRU,
    n_eval_envs: int,
    n_agents: int,
    n_actions: int,
    comm_dim: int,
    device: torch.device,
    rnn_layers: int,
    rnn_hidden_dim: int,
    n_episodes: int,
    seed: Optional[int],
    obs_normalizer,
    eval_writer,
    eval_f,
    best_model_tracker,
    run_dir: Path,
    save_checkpoint,
    global_step: int,
    eval_baseline: Dict[str, Any],
    start_time: Optional[float] = None,
    total_timesteps: Optional[int] = None,
):
    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive")

    episode_seeds: List[Optional[int]]
    if seed is None:
        base_seed = int(np.random.randint(0, 2**31 - 1))
        episode_seeds = [base_seed + ep for ep in range(int(n_episodes))]
    else:
        episode_seeds = [int(seed) + ep for ep in range(int(n_episodes))]

    action_selector, reset_eval = _make_dial_rnn_eval_action_selector(
        agent, dru, n_eval_envs, n_agents, n_actions, comm_dim, device,
        rnn_layers, rnn_hidden_dim,
    )

    all_policy_rewards: List[float] = []
    for start in range(0, len(episode_seeds), n_eval_envs):
        reset_eval()
        batch_seeds = list(episode_seeds[start : start + n_eval_envs])
        active_count = len(batch_seeds)
        if active_count < n_eval_envs:
            batch_seeds.extend([batch_seeds[-1]] * (n_eval_envs - active_count))

        mean_r, std_r, ep_rewards = run_evaluation_vectorized_seeded(
            eval_env=eval_env,
            agent=agent,
            n_eval_envs=n_eval_envs,
            n_agents=n_agents,
            n_actions=n_actions,
            use_agent_id=True,
            device=device,
            episode_seeds=batch_seeds[:active_count],
            obs_normalizer=obs_normalizer,
            action_selector=action_selector,
        )
        all_policy_rewards.extend(ep_rewards)

    mean_eval_reward = float(np.mean(all_policy_rewards))
    std_eval_reward = float(np.std(all_policy_rewards))

    # Baseline
    baseline_label = str(eval_baseline.get("label", "perfect_comm"))
    baseline_policy = str(eval_baseline.get("heuristic_policy", "always_send"))
    baseline_deterministic = bool(eval_baseline.get("deterministic", True))
    baseline_perfect_comm = bool(eval_baseline.get("use_perfect_communication", False))

    current_pc_states = eval_env.call("get_perfect_communication")
    current_pc = bool(current_pc_states[0]) if current_pc_states else False
    if baseline_perfect_comm != current_pc:
        eval_env.call("set_perfect_communication", baseline_perfect_comm)

    try:
        heuristic_name = None if baseline_policy == "always_send" else baseline_policy
        mean_baseline_reward, std_baseline_reward, baseline_rewards = run_evaluation_vectorized_seeded(
            eval_env=eval_env,
            agent=agent,
            n_eval_envs=n_eval_envs,
            n_agents=n_agents,
            n_actions=n_actions,
            use_agent_id=True,
            device=device,
            episode_seeds=episode_seeds,
            obs_normalizer=obs_normalizer,
            heuristic_policy_name=heuristic_name,
            heuristic_deterministic=baseline_deterministic,
            fixed_action=1 if baseline_policy == "always_send" else None,
        )
    finally:
        if baseline_perfect_comm != current_pc:
            eval_env.call("set_perfect_communication", current_pc)

    policy_arr = np.asarray(all_policy_rewards, dtype=np.float64)
    baseline_arr = np.asarray(baseline_rewards, dtype=np.float64)
    n_matched = min(len(policy_arr), len(baseline_arr))
    win_rate = float(np.mean(policy_arr[:n_matched] >= baseline_arr[:n_matched])) if n_matched > 0 else 0.0

    denom = np.maximum(np.abs(baseline_arr[:n_matched]), 1e-8)
    drop_ratios = (baseline_arr[:n_matched] - policy_arr[:n_matched]) / denom
    mean_drop_ratio = float(np.mean(drop_ratios)) if n_matched > 0 else 0.0
    std_drop_ratio = float(np.std(drop_ratios)) if n_matched > 0 else 0.0

    drop_csv_path = run_dir / "evaluation_drop_stats.csv"
    write_drop_header = (not drop_csv_path.exists()) or drop_csv_path.stat().st_size == 0
    with drop_csv_path.open("a", newline="", encoding="utf-8") as drop_f:
        drop_writer = csv.writer(drop_f)
        if write_drop_header:
            drop_writer.writerow([
                "step", "baseline_policy", "baseline_perfect_communication",
                "policy_mean_reward", "policy_std_reward",
                "baseline_mean_reward", "baseline_std_reward",
                "drop_ratio_mean", "drop_ratio_std", "num_episodes",
            ])
        drop_writer.writerow([
            global_step, baseline_label, int(baseline_perfect_comm),
            mean_eval_reward, std_eval_reward,
            mean_baseline_reward, std_baseline_reward,
            mean_drop_ratio, std_drop_ratio, n_matched,
        ])
        drop_f.flush()

    eval_writer.writerow([global_step, mean_eval_reward, std_eval_reward])
    eval_f.flush()

    best_model_tracker.update(
        "eval_drop_ratio", -mean_drop_ratio, run_dir / "best_model.pt", save_checkpoint,
    )

    eta_str = ""
    if start_time is not None and total_timesteps is not None and global_step > 0:
        elapsed = time.time() - start_time
        sps = global_step / elapsed
        remaining = total_timesteps - global_step
        eta_str = f" | ETA={_format_eta(remaining / sps)}"

    print(
        f"[MARL-DIAL] Eval at step {global_step}: "
        f"mean_reward={mean_eval_reward:.3f} std={std_eval_reward:.3f} | "
        f"baseline={baseline_label} "
        f"mean={mean_baseline_reward:.3f} std={std_baseline_reward:.3f} | "
        f"drop_ratio_mean={mean_drop_ratio:.6f} drop_ratio_std={std_drop_ratio:.6f} | "
        f"win={win_rate:.0%}{eta_str}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    reset_shared_running_normalizers()
    args = parse_args()
    device, rng = setup_device_and_rng(args.device, args.seed)

    cfg, config_path_str, n_agents, use_agent_id, eval_reward_override, eval_termination_override, network_override, training_reward_override = (
        load_config_with_overrides(args.config, args.n_agents, True, args.set_overrides)
    )
    eval_baseline = resolve_training_eval_baseline(cfg, n_agents)

    algo_label = "marl_dial"

    resuming = args.resume is not None
    if resuming:
        run_dir = Path(args.resume)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Resume directory does not exist: {run_dir}")
    else:
        run_dir = prepare_run_directory(algo_label, args.config, args.output_root)
    rewards_csv_path = run_dir / "training_rewards.csv"
    eval_csv_path = run_dir / "evaluation_rewards.csv"

    if args.n_envs <= 0:
        raise ValueError("n_envs must be positive")

    shared_reward_normalizer, _shared_reward_manager = setup_shared_reward_normalizer(
        cfg.get("reward", {}), run_dir,
    )

    env, env_seeds = create_async_vector_env(
        n_envs=args.n_envs,
        n_agents=n_agents,
        episode_length=args.episode_length,
        config_path_str=config_path_str,
        seed=args.seed,
        shared_reward_normalizer=shared_reward_normalizer,
        network_override=network_override,
        reward_override=training_reward_override,
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
    )

    obs_dim = int(env.single_observation_space.spaces["agent_0"].shape[0])
    n_actions = int(env.single_action_space.spaces["agent_0"].n)
    comm_dim = args.comm_dim
    obs_normalizer = create_obs_normalizer(
        obs_dim, args.normalize_obs, args.obs_norm_clip, args.obs_norm_eps,
    )

    agent = DialRNNAgent(
        obs_dim=obs_dim,
        n_agents=n_agents,
        n_actions=n_actions,
        comm_dim=comm_dim,
        rnn_hidden_dim=args.rnn_hidden_dim,
        rnn_layers=args.rnn_layers,
    )

    dru = DRU(sigma=args.dru_sigma)

    learner = MARLDIALLearner(
        agent=agent,
        n_agents=n_agents,
        n_actions=n_actions,
        comm_dim=comm_dim,
        dru=dru,
        gamma=args.gamma,
        lr=args.learning_rate,
        target_update_steps=args.target_update_steps,
        grad_clip_norm=args.grad_clip_norm,
        device=device,
        momentum=args.momentum,
        optimizer_type=args.optimizer,
    )

    best_model_tracker = BestModelTracker()
    global_step = 0
    episode = 0
    last_eval_step = 0
    eval_seed = args.seed
    vector_step = 0

    if resuming:
        training_state_path = run_dir / "training_state.pt"
        if not training_state_path.exists():
            raise FileNotFoundError(f"No training_state.pt found in {run_dir}")
        counters = load_dial_rnn_training_state(
            training_state_path, learner, obs_normalizer, best_model_tracker,
        )
        global_step = counters["global_step"]
        episode = counters["episode"]
        last_eval_step = counters["last_eval_step"]
        vector_step = counters["vector_step"]
        print(f"Resumed from {run_dir} at step {global_step}")

    learner.agent.train()

    def save_checkpoint(path: Path) -> None:
        save_dial_rnn_checkpoint(
            path=path,
            n_agents=n_agents,
            obs_dim=obs_dim,
            n_actions=n_actions,
            agent=learner.agent,
            obs_normalizer=obs_normalizer,
            comm_dim=comm_dim,
            dru_sigma=args.dru_sigma,
            rnn_hidden_dim=args.rnn_hidden_dim,
            rnn_layers=args.rnn_layers,
        )

    def save_training_state() -> None:
        save_dial_rnn_training_state(
            run_dir / "training_state.pt", learner,
            obs_normalizer, best_model_tracker, global_step, episode,
            last_eval_step, vector_step,
        )

    csv_mode = "a" if resuming else "w"
    with rewards_csv_path.open(csv_mode, newline="", encoding="utf-8") as train_f, \
         eval_csv_path.open(csv_mode, newline="", encoding="utf-8") as eval_f:
        train_writer = csv.writer(train_f)
        eval_writer = csv.writer(eval_f)
        if not resuming:
            train_writer.writerow(["episode", "reward_sum", "epsilon", "steps"])
            eval_writer.writerow(["step", "mean_reward", "std_reward"])

        obs_dict, _info = env.reset(seed=env_seeds)
        obs_raw = stack_vector_obs(obs_dict, n_agents)
        episode_reward_sums = np.zeros((args.n_envs,), dtype=np.float32)
        start_time = time.time()

        # On-policy DIAL state (tensors for gradient flow)
        hidden = torch.zeros(
            args.rnn_layers, args.n_envs * n_agents, args.rnn_hidden_dim, device=device,
        )
        prev_action = torch.full(
            (args.n_envs, n_agents), n_actions, dtype=torch.long, device=device,
        )
        recv_msg = torch.zeros(
            args.n_envs, n_agents, n_agents * comm_dim, device=device,
        )

        # Per-env episode accumulators
        env_online_q = [[] for _ in range(args.n_envs)]
        env_obs_raw = [[] for _ in range(args.n_envs)]
        env_actions = [[] for _ in range(args.n_envs)]
        env_rewards = [[] for _ in range(args.n_envs)]
        env_terminated = [[] for _ in range(args.n_envs)]
        pending_episodes: list[dict] = []
        seg_init_hidden = [
            hidden[:, env_idx * n_agents : (env_idx + 1) * n_agents, :].detach().clone()
            for env_idx in range(args.n_envs)
        ]
        seg_init_prev_action = [
            prev_action[env_idx].detach().clone() for env_idx in range(args.n_envs)
        ]
        seg_init_recv_msg = [
            recv_msg[env_idx].detach().clone() for env_idx in range(args.n_envs)
        ]

        def flush_env_segment(env_idx: int, next_obs_segment: np.ndarray) -> None:
            """Finalize the current per-env segment into a learner episode."""
            if not env_online_q[env_idx]:
                return
            pending_episodes.append({
                "online_q": env_online_q[env_idx],
                "obs_raw": np.stack(env_obs_raw[env_idx]),
                "actions": np.stack(env_actions[env_idx]),
                "rewards": np.stack(env_rewards[env_idx]),
                "terminated": np.array(env_terminated[env_idx], dtype=np.float32),
                "next_obs_raw": np.asarray(next_obs_segment, dtype=np.float32).copy(),
                "init_hidden": seg_init_hidden[env_idx],
                "init_prev_action": seg_init_prev_action[env_idx],
                "init_recv_msg": seg_init_recv_msg[env_idx],
            })
            env_online_q[env_idx] = []
            env_obs_raw[env_idx] = []
            env_actions[env_idx] = []
            env_rewards[env_idx] = []
            env_terminated[env_idx] = []

        while global_step < args.total_timesteps:
            # Normalize obs
            if obs_normalizer is not None:
                obs = obs_normalizer.normalize(obs_raw, update=True)
            else:
                obs = obs_raw

            eps = epsilon_by_step(
                global_step, args.epsilon_start, args.epsilon_end, args.epsilon_decay_steps,
            )

            # Forward pass WITH gradients (on-policy DIAL)
            obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)
            q_t, m_t, new_hidden = dial_rnn_forward_batched(
                learner.agent,
                obs_t,
                prev_action,
                recv_msg,
                hidden,
            )

            # Epsilon-greedy action selection (detached)
            greedy = q_t.detach().argmax(dim=-1).cpu().numpy().astype(np.int64)
            actions = greedy.copy()
            explore_mask = rng.random((args.n_envs, n_agents)) < eps
            if np.any(explore_mask):
                actions[explore_mask] = rng.integers(
                    0, n_actions, size=int(explore_mask.sum()),
                    endpoint=False, dtype=np.int64,
                )

            # Step environment
            action_dict = {f"agent_{i}": actions[:, i] for i in range(n_agents)}
            next_obs_dict, rewards_arr, terminated, truncated, infos = env.step(action_dict)
            next_obs_raw = stack_vector_obs(next_obs_dict, n_agents)

            terminated_any = np.asarray(terminated, dtype=np.bool_)
            truncated_any = np.asarray(truncated, dtype=np.bool_)
            done_reset = np.logical_or(terminated_any, truncated_any)

            rewards_np = np.asarray(rewards_arr, dtype=np.float32)
            team_rewards = rewards_np.sum(axis=1)
            rewards_team = np.repeat(
                team_rewards[:, None], n_agents, axis=1,
            ).astype(np.float32)
            episode_reward_sums += team_rewards

            next_obs_for_buffer, _ = patch_autoreset_final_obs(
                next_obs_raw, infos, done_reset, n_agents,
            )

            # Store per-env data (Q-values keep grad for end-to-end DIAL)
            for i in range(args.n_envs):
                env_online_q[i].append(q_t[i])
                env_obs_raw[i].append(obs_raw[i].copy())
                env_actions[i].append(actions[i].copy())
                env_rewards[i].append(rewards_team[i].copy())
                env_terminated[i].append(float(terminated_any[i]))

            # DRU + message routing (with grad)
            new_recv = route_messages(dru(m_t, train_mode=True), n_agents)

            # Handle done envs: finalize episodes and reset state
            done_indices = np.where(done_reset)[0]
            for i in done_indices:
                flush_env_segment(i, next_obs_for_buffer[i])

            # Update recurrent state (torch.where preserves grad for ongoing envs)
            if len(done_indices) > 0:
                reset_mask = torch.zeros(args.n_envs, dtype=torch.bool, device=device)
                reset_mask[done_indices] = True
                h_mask = reset_mask.unsqueeze(1).expand(
                    args.n_envs, n_agents,
                ).reshape(1, args.n_envs * n_agents, 1).expand_as(new_hidden)
                hidden = torch.where(h_mask, torch.zeros_like(new_hidden), new_hidden)
                m_mask = reset_mask.view(args.n_envs, 1, 1).expand_as(new_recv)
                recv_msg = torch.where(m_mask, torch.zeros_like(new_recv), new_recv)
                actions_t = torch.as_tensor(actions, device=device, dtype=torch.long)
                a_mask = reset_mask.view(args.n_envs, 1).expand_as(actions_t)
                prev_action = torch.where(
                    a_mask, torch.full_like(actions_t, n_actions), actions_t,
                )
            else:
                hidden = new_hidden
                recv_msg = new_recv
                prev_action = torch.as_tensor(actions, device=device, dtype=torch.long)

            for i in range(args.n_envs):
                if env_online_q[i]:
                    continue
                seg_init_hidden[i] = hidden[
                    :, i * n_agents : (i + 1) * n_agents, :
                ].detach().clone()
                seg_init_prev_action[i] = prev_action[i].detach().clone()
                seg_init_recv_msg[i] = recv_msg[i].detach().clone()

            obs_raw = next_obs_raw
            global_step += args.n_envs
            vector_step += 1

            episode = log_completed_episodes(
                done_reset=done_reset, episode_reward_sums=episode_reward_sums,
                global_step=global_step, episode=episode,
                train_writer=train_writer, train_f=train_f,
                best_model_tracker=best_model_tracker, run_dir=run_dir,
                save_checkpoint=save_checkpoint, log_interval=args.log_interval,
                algo_name="MARL-DIAL",
                extra_csv_values=eps,
                extra_log_str=f" eps={eps:.3f}",
                start_time=start_time, total_timesteps=args.total_timesteps,
            )

            # On-policy training: update as soon as enough episodes are ready
            if len(pending_episodes) >= args.batch_episodes:
                # Truncate any still-running segments at the optimizer boundary.
                # The next segment will continue from the detached recurrent state.
                for i in range(args.n_envs):
                    flush_env_segment(i, obs_raw[i])
                batch_eps = pending_episodes
                pending_episodes = []
                learner.update_online(batch_eps, obs_normalizer=obs_normalizer)

                # Detach recurrent state at the truncation boundary.
                hidden = hidden.detach()
                recv_msg = recv_msg.detach()
                prev_action = prev_action.detach()
                for i in range(args.n_envs):
                    seg_init_hidden[i] = hidden[
                        :, i * n_agents : (i + 1) * n_agents, :
                    ].detach().clone()
                    seg_init_prev_action[i] = prev_action[i].detach().clone()
                    seg_init_recv_msg[i] = recv_msg[i].detach().clone()

            if global_step - last_eval_step >= args.eval_freq:
                _evaluate_and_log_dial_rnn(
                    eval_env=eval_env, agent=learner.agent, dru=dru,
                    n_eval_envs=args.n_eval_envs, n_agents=n_agents,
                    n_actions=n_actions, comm_dim=comm_dim,
                    device=device,
                    rnn_layers=args.rnn_layers,
                    rnn_hidden_dim=args.rnn_hidden_dim,
                    n_episodes=args.n_eval_episodes, seed=eval_seed,
                    obs_normalizer=obs_normalizer,
                    eval_writer=eval_writer, eval_f=eval_f,
                    best_model_tracker=best_model_tracker, run_dir=run_dir,
                    save_checkpoint=save_checkpoint, global_step=global_step,
                    eval_baseline=eval_baseline,
                    start_time=start_time, total_timesteps=args.total_timesteps,
                )
                eval_seed += args.n_eval_episodes
                last_eval_step = global_step
                save_training_state()

        if pending_episodes or any(env_online_q[i] for i in range(args.n_envs)):
            for i in range(args.n_envs):
                flush_env_segment(i, obs_raw[i])
            if pending_episodes:
                learner.update_online(pending_episodes, obs_normalizer=obs_normalizer)

    latest_path = run_dir / "latest_model.pt"
    save_checkpoint(latest_path)
    save_training_state()

    hyperparams = {
        "algorithm": algo_label,
        "total_timesteps": args.total_timesteps,
        "episode_length": args.episode_length,
        "n_agents": n_agents,
        "n_envs": args.n_envs,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "target_update_steps": args.target_update_steps,
        "grad_clip_norm": args.grad_clip_norm,
        "optimizer": args.optimizer,
        "momentum": args.momentum,
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
        "epsilon_decay_steps": args.epsilon_decay_steps,
        "rnn_hidden_dim": args.rnn_hidden_dim,
        "rnn_layers": args.rnn_layers,
        "batch_episodes": args.batch_episodes,
        "comm_dim": comm_dim,
        "dru_sigma": args.dru_sigma,
        "normalize_obs": args.normalize_obs,
        "obs_norm_clip": args.obs_norm_clip,
        "obs_norm_eps": args.obs_norm_eps,
        "eval_freq": args.eval_freq,
        "n_eval_episodes": args.n_eval_episodes,
        "n_eval_envs": args.n_eval_envs,
        "device": str(device),
        "seed": args.seed,
    }
    save_config_with_hyperparameters(
        run_dir, args.config, algo_label, hyperparams,
        resolved_config=cfg, set_overrides=args.set_overrides,
    )

    print_run_summary(run_dir, latest_path, rewards_csv_path, eval_csv_path)

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
