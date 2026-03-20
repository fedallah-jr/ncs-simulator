from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.marl import (
    DialSequenceBuffer,
    OnlineBatchCollector,
    IQLDIALLearner,
    DialMLPAgent,
    DRU,
    route_messages,
    build_base_qlearning_parser,
    save_dial_checkpoint,
    save_dial_training_state,
    load_dial_training_state,
    build_qlearning_hyperparams,
    dial_collect_transition,
    patch_autoreset_final_obs,
)
from utils.marl.buffer import DialChunkAccumulator
from utils.marl.common import run_evaluation_vectorized_seeded, epsilon_by_step
from utils.marl.networks import append_agent_id
from utils.marl_training import (
    setup_device_and_rng,
    load_config_with_overrides,
    resolve_training_eval_baseline,
    create_obs_normalizer,
    print_run_summary,
    setup_shared_reward_normalizer,
    log_completed_episodes,
)
from utils.marl.vector_env import create_async_vector_env, create_eval_async_vector_env, stack_vector_obs
from utils.reward_normalization import reset_shared_running_normalizers
from utils.run_utils import prepare_run_directory, save_config_with_hyperparameters, BestModelTracker


def parse_args() -> argparse.Namespace:
    parser = build_base_qlearning_parser(
        description="Train IQL-DIAL (shared MLP + differentiable communication) on multi-agent NCS env."
    )
    parser.add_argument("--dial-mode", choices=["online", "replay"], default="online",
                        help="'online' = no replay buffer (paper-aligned); 'replay' = experience replay")
    parser.set_defaults(batch_size=2000)
    parser.add_argument("--comm-dim", type=int, default=1, help="Message dimension per agent")
    parser.add_argument("--dru-sigma", type=float, default=2.0, help="DRU noise std")
    parser.add_argument("--seq-len", type=int, default=2, help="Sequence chunk length (minimum 2)")
    return parser.parse_args()


def _make_dial_eval_action_selector(
    agent: DialMLPAgent,
    dru: DRU,
    n_eval_envs: int,
    n_agents: int,
    n_actions: int,
    comm_dim: int,
    use_agent_id: bool,
    device: torch.device,
    obs_normalizer=None,
):
    """Create a stateful action selector closure for DIAL evaluation."""
    prev_msg_logits = np.zeros((n_eval_envs, n_agents, comm_dim), dtype=np.float32)
    done_flags = np.ones(n_eval_envs, dtype=np.bool_)  # start as "done" to init on first call

    def reset_eval_state():
        prev_msg_logits[:] = 0.0
        done_flags[:] = True

    @torch.no_grad()
    def action_selector(obs: np.ndarray) -> np.ndarray:
        """obs: (n_eval_envs, n_agents, obs_dim) — already normalized by caller."""
        obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)
        if use_agent_id:
            obs_t = append_agent_id(obs_t, n_agents)

        msg_logits_t = torch.as_tensor(prev_msg_logits, device=device, dtype=torch.float32)
        msg_post_dru = dru(msg_logits_t, train_mode=False)
        recv_msg = route_messages(msg_post_dru, n_agents)

        inp = torch.cat([obs_t, recv_msg], dim=-1)
        q_values, msg_logits = agent(inp.reshape(n_eval_envs * n_agents, -1))
        q_values = q_values.view(n_eval_envs, n_agents, n_actions)
        msg_logits = msg_logits.view(n_eval_envs, n_agents, comm_dim)

        actions = q_values.argmax(dim=-1).cpu().numpy().astype(np.int64)
        prev_msg_logits[:] = msg_logits.cpu().numpy()
        return actions

    return action_selector, reset_eval_state


def _evaluate_and_log_dial(
    *,
    eval_env,
    agent: DialMLPAgent,
    dru: DRU,
    n_eval_envs: int,
    n_agents: int,
    n_actions: int,
    comm_dim: int,
    use_agent_id: bool,
    device: torch.device,
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
):
    """DIAL-aware evaluation that maintains message state."""
    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive")

    episode_seeds: List[Optional[int]]
    if seed is None:
        base_seed = int(np.random.randint(0, 2**31 - 1))
        episode_seeds = [base_seed + ep for ep in range(int(n_episodes))]
    else:
        episode_seeds = [int(seed) + ep for ep in range(int(n_episodes))]

    agent.eval()
    action_selector, reset_eval = _make_dial_eval_action_selector(
        agent, dru, n_eval_envs, n_agents, n_actions, comm_dim, use_agent_id, device,
        obs_normalizer,
    )

    # We process seeds in batches of n_eval_envs, resetting eval state each batch
    all_policy_rewards: List[float] = []
    for start in range(0, len(episode_seeds), n_eval_envs):
        reset_eval()
        batch_seeds = list(episode_seeds[start:start + n_eval_envs])
        active_count = len(batch_seeds)
        if active_count < n_eval_envs:
            batch_seeds.extend([batch_seeds[-1]] * (n_eval_envs - active_count))

        mean_r, std_r, ep_rewards = run_evaluation_vectorized_seeded(
            eval_env=eval_env,
            agent=agent,
            n_eval_envs=n_eval_envs,
            n_agents=n_agents,
            n_actions=n_actions,
            use_agent_id=use_agent_id,
            device=device,
            episode_seeds=batch_seeds[:active_count],
            obs_normalizer=obs_normalizer,
            action_selector=action_selector,
        )
        all_policy_rewards.extend(ep_rewards)

    mean_eval_reward = float(np.mean(all_policy_rewards))
    std_eval_reward = float(np.std(all_policy_rewards))

    # Baseline evaluation (no DIAL, just fixed actions)
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
            use_agent_id=use_agent_id,
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
    if n_matched > 0:
        win_rate = float(np.mean(policy_arr[:n_matched] >= baseline_arr[:n_matched]))
    else:
        win_rate = 0.0

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

    algo_tag = "IQL-DIAL"
    print(
        f"[{algo_tag}] Eval at step {global_step}: "
        f"mean_reward={mean_eval_reward:.3f} std={std_eval_reward:.3f} | "
        f"baseline={baseline_label} "
        f"mean={mean_baseline_reward:.3f} std={std_baseline_reward:.3f} | "
        f"drop_ratio_mean={mean_drop_ratio:.6f} drop_ratio_std={std_drop_ratio:.6f} | "
        f"win={win_rate:.0%}"
    )
    agent.train()


def main() -> None:
    reset_shared_running_normalizers()
    args = parse_args()
    device, rng = setup_device_and_rng(args.device, args.seed)

    if args.seq_len < 2:
        raise ValueError("--seq-len must be >= 2")

    cfg, config_path_str, n_agents, use_agent_id, eval_reward_override, eval_termination_override, network_override, training_reward_override = (
        load_config_with_overrides(args.config, args.n_agents, not args.no_agent_id, args.set_overrides)
    )
    eval_baseline = resolve_training_eval_baseline(cfg, n_agents)

    online_mode = args.dial_mode == "online"
    algo_label = "iql_dial_online" if online_mode else "iql_dial_replay"

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
        cfg.get("reward", {}), run_dir
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
    # input_dim = obs + agent_id + recv_msg
    input_dim = obs_dim + (n_agents if use_agent_id else 0) + n_agents * comm_dim
    obs_normalizer = create_obs_normalizer(
        obs_dim, args.normalize_obs, args.obs_norm_clip, args.obs_norm_eps
    )

    agent = DialMLPAgent(
        input_dim=input_dim,
        n_actions=n_actions,
        comm_dim=comm_dim,
        hidden_dims=tuple(args.hidden_dims),
        activation=args.activation,
        feature_norm=args.feature_norm,
        layer_norm=args.layer_norm,
        dueling=args.dueling,
        stream_hidden_dim=args.stream_hidden_dim,
    )

    dru = DRU(sigma=args.dru_sigma)

    learner = IQLDIALLearner(
        agent=agent,
        n_agents=n_agents,
        n_actions=n_actions,
        comm_dim=comm_dim,
        dru=dru,
        gamma=args.gamma,
        lr=args.learning_rate,
        target_update_interval=args.target_update_interval,
        grad_clip_norm=args.grad_clip_norm,
        use_agent_id=use_agent_id,
        double_q=args.double_q,
        device=device,
        optimizer_type=args.optimizer,
    )

    if online_mode:
        buffer = OnlineBatchCollector(device=device)
    else:
        buffer = DialSequenceBuffer(
            capacity=args.buffer_size,
            seq_len=args.seq_len,
            n_agents=n_agents,
            obs_dim=obs_dim,
            comm_dim=comm_dim,
            device=device,
            rng=rng,
        )

    accumulator = DialChunkAccumulator(n_envs=args.n_envs, seq_len=args.seq_len)
    prev_msg_logits = np.zeros((args.n_envs, n_agents, comm_dim), dtype=np.float32)

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
        counters = load_dial_training_state(
            training_state_path, learner, buffer, accumulator, obs_normalizer, best_model_tracker,
        )
        global_step = counters["global_step"]
        episode = counters["episode"]
        last_eval_step = counters["last_eval_step"]
        vector_step = counters["vector_step"]
        # Clear accumulator and message state: env resets on resume so any
        # saved partial windows belong to the old episode and must not be
        # paired with fresh post-resume transitions.
        accumulator = DialChunkAccumulator(n_envs=args.n_envs, seq_len=args.seq_len)
        prev_msg_logits[:] = 0.0
        print(f"Resumed from {run_dir} at step {global_step}")

    def save_checkpoint(path: Path) -> None:
        save_dial_checkpoint(
            path=path,
            n_agents=n_agents,
            obs_dim=obs_dim,
            n_actions=n_actions,
            use_agent_id=use_agent_id,
            agent_hidden_dims=list(args.hidden_dims),
            agent_activation=args.activation,
            agent=learner.agent,
            obs_normalizer=obs_normalizer,
            comm_dim=comm_dim,
            dru_sigma=args.dru_sigma,
            feature_norm=args.feature_norm,
            layer_norm=args.layer_norm,
            dueling=args.dueling,
            stream_hidden_dim=args.stream_hidden_dim if args.dueling else None,
        )

    def save_training_state() -> None:
        save_dial_training_state(
            run_dir / "training_state.pt", learner, buffer, accumulator,
            obs_normalizer, best_model_tracker, global_step, episode,
            last_eval_step, vector_step, dial_mode=args.dial_mode,
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

        while global_step < args.total_timesteps:
            step = dial_collect_transition(
                env=env, agent=learner.agent, obs_raw=obs_raw,
                obs_normalizer=obs_normalizer, global_step=global_step,
                epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end,
                epsilon_decay_steps=args.epsilon_decay_steps,
                n_envs=args.n_envs, n_agents=n_agents, n_actions=n_actions,
                use_agent_id=use_agent_id, rng=rng, device=device,
                prev_msg_logits=prev_msg_logits, dru=dru,
                accumulator=accumulator, buffer=buffer, comm_dim=comm_dim,
            )

            raw_rewards = step.rewards_arr
            team_rewards = raw_rewards.sum(axis=1)
            episode_reward_sums += team_rewards

            obs_raw = step.next_obs_raw
            global_step += args.n_envs
            vector_step += 1

            episode = log_completed_episodes(
                done_reset=step.done_reset, episode_reward_sums=episode_reward_sums,
                global_step=global_step, episode=episode,
                train_writer=train_writer, train_f=train_f,
                best_model_tracker=best_model_tracker, run_dir=run_dir,
                save_checkpoint=save_checkpoint, log_interval=args.log_interval,
                algo_name="IQL-DIAL-online" if online_mode else "IQL-DIAL-replay",
                extra_csv_values=step.epsilon,
                extra_log_str=f" eps={step.epsilon:.3f}",
            )

            if online_mode:
                if len(buffer) >= args.batch_size:
                    batch = buffer.pop_batch(obs_normalizer=obs_normalizer)
                    learner.update(batch)
            elif len(buffer) >= args.start_learning and vector_step % args.train_interval == 0:
                batch = buffer.sample(args.batch_size, obs_normalizer=obs_normalizer)
                learner.update(batch)

            if global_step - last_eval_step >= args.eval_freq:
                _evaluate_and_log_dial(
                    eval_env=eval_env, agent=learner.agent, dru=dru,
                    n_eval_envs=args.n_eval_envs, n_agents=n_agents,
                    n_actions=n_actions, comm_dim=comm_dim,
                    use_agent_id=use_agent_id, device=device,
                    n_episodes=args.n_eval_episodes, seed=eval_seed,
                    obs_normalizer=obs_normalizer,
                    eval_writer=eval_writer, eval_f=eval_f,
                    best_model_tracker=best_model_tracker, run_dir=run_dir,
                    save_checkpoint=save_checkpoint, global_step=global_step,
                    eval_baseline=eval_baseline,
                )
                eval_seed += args.n_eval_episodes
                last_eval_step = global_step
                save_training_state()

    latest_path = run_dir / "latest_model.pt"
    save_checkpoint(latest_path)
    save_training_state()

    hyperparams = build_qlearning_hyperparams(
        algorithm=algo_label,
        args=args,
        n_agents=n_agents,
        use_agent_id=use_agent_id,
        device=device,
    )
    hyperparams.update({
        "comm_dim": comm_dim,
        "dru_sigma": args.dru_sigma,
        "seq_len": args.seq_len,
        "dial_mode": args.dial_mode,
    })
    save_config_with_hyperparameters(
        run_dir,
        args.config,
        algo_label,
        hyperparams,
        resolved_config=cfg,
        set_overrides=args.set_overrides,
    )

    print_run_summary(run_dir, latest_path, rewards_csv_path, eval_csv_path)

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
