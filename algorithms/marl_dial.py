"""Reimplementation of the reference DIAL codebase:
https://github.com/minqi/learning-to-communicate-pytorch

This version adapts the original shared-parameter recurrent DIAL setup to the
vectorized NCS environment used in this repository.
"""

from __future__ import annotations

import argparse
import copy
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

from utils.marl.networks import DialRNNAgent, DRU, route_messages, dial_rnn_forward_batched
from utils.marl.learners import MARLDIALLearner
from utils.marl.common import (
    run_evaluation_vectorized_seeded,
    dial_rnn_collect_transition,
)
from utils.marl.buffer import DialRNNEpisodeCollector
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
    parser.set_defaults(
        optimizer="adam",
        grad_clip_norm=10.0,
        epsilon_start=0.05,
        epsilon_end=0.05,
        epsilon_decay_steps=1,
    )
    parser.add_argument("--comm-dim", type=int, default=4, help="Message dimension per agent")
    parser.add_argument("--dru-sigma", type=float, default=2.0, help="DRU noise std")
    parser.add_argument("--rnn-hidden-dim", type=int, default=128)
    parser.add_argument("--rnn-layers", type=int, default=2)
    parser.add_argument("--batch-episodes", type=int, default=32, help="Episodes per training batch")
    parser.add_argument("--target-update-steps", type=int, default=100,
                        help="Target net sync every N optimizer steps (reference: step_target)")
    parser.add_argument("--momentum", type=float, default=0.05, help="RMSprop momentum")
    parser.add_argument("--use-vdn-mixer", action="store_true",
                        help="(legacy) Equivalent to --mixer vdn")
    parser.add_argument("--mixer", type=str, default="none", choices=["none", "vdn", "qmix"],
                        help="Value decomposition mixer for joint TD loss (default: none = per-agent IQL)")
    parser.add_argument("--qmix-mixing-hidden-dim", type=int, default=32,
                        help="QMIX mixing network hidden dim")
    parser.add_argument("--qmix-hypernet-hidden-dim", type=int, default=64,
                        help="QMIX hypernetwork hidden dim")
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


def _recurrent_observation_override() -> Dict[str, Any]:
    """Reduce handcrafted temporal features so the recurrent policy carries memory."""
    return {
        "history_window": 0,
        "state_history_window": 0,
        "include_current_throughput": False,
    }


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
    observation_override = _recurrent_observation_override()
    cfg_effective = copy.deepcopy(cfg)
    cfg_effective.setdefault("observation", {}).update(observation_override)

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
        observation_override=observation_override,
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
        observation_override=observation_override,
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

    # Resolve mixer: --use-vdn-mixer (legacy) or --mixer vdn/qmix
    mixer_type = args.mixer
    if args.use_vdn_mixer and mixer_type == "none":
        mixer_type = "vdn"

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
        mixer_type=mixer_type,
        obs_dim=obs_dim,
        qmix_mixing_hidden_dim=args.qmix_mixing_hidden_dim,
        qmix_hypernet_hidden_dim=args.qmix_hypernet_hidden_dim,
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
            mixer_type=mixer_type,
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

        # Complete-episode collection state (reference-faithful).
        # Collects transitions without gradients, then recomputes the
        # full recurrent trace during learning — preserving within-
        # episode DIAL credit assignment without keeping rollout graphs
        # alive or truncating at arbitrary optimizer boundaries.
        collector = DialRNNEpisodeCollector(args.n_envs, device)
        hidden_states = torch.zeros(
            args.rnn_layers, args.n_envs * n_agents, args.rnn_hidden_dim, device=device,
        )
        prev_msg_logits = np.zeros((args.n_envs, n_agents, comm_dim), dtype=np.float32)
        prev_actions = np.full((args.n_envs, n_agents), n_actions, dtype=np.int64)
        episode_start_mask = np.ones(args.n_envs, dtype=bool)

        while global_step < args.total_timesteps:
            step_result = dial_rnn_collect_transition(
                env=env,
                agent=learner.agent,
                obs_raw=obs_raw,
                obs_normalizer=obs_normalizer,
                global_step=global_step,
                epsilon_start=args.epsilon_start,
                epsilon_end=args.epsilon_end,
                epsilon_decay_steps=args.epsilon_decay_steps,
                n_envs=args.n_envs,
                n_agents=n_agents,
                n_actions=n_actions,
                rng=rng,
                device=device,
                prev_msg_logits=prev_msg_logits,
                prev_actions=prev_actions,
                hidden_states=hidden_states,
                dru=dru,
                collector=collector,
                comm_dim=comm_dim,
                episode_start_mask=episode_start_mask,
            )

            team_rewards = step_result.rewards_arr.sum(axis=1)
            episode_reward_sums += team_rewards
            obs_raw = step_result.next_obs_raw
            global_step += args.n_envs
            vector_step += 1

            episode = log_completed_episodes(
                done_reset=step_result.done_reset,
                episode_reward_sums=episode_reward_sums,
                global_step=global_step, episode=episode,
                train_writer=train_writer, train_f=train_f,
                best_model_tracker=best_model_tracker, run_dir=run_dir,
                save_checkpoint=save_checkpoint, log_interval=args.log_interval,
                algo_name="MARL-DIAL",
                extra_csv_values=step_result.epsilon,
                extra_log_str=f" eps={step_result.epsilon:.3f}",
                start_time=start_time, total_timesteps=args.total_timesteps,
            )

            # Train on complete episodes (full-episode backprop)
            if collector.has_episodes(args.batch_episodes):
                batch = collector.pop_batch(
                    args.batch_episodes, obs_normalizer=obs_normalizer,
                )
                learner.update(batch)

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
        "mixer": mixer_type,
        "qmix_mixing_hidden_dim": args.qmix_mixing_hidden_dim,
        "qmix_hypernet_hidden_dim": args.qmix_hypernet_hidden_dim,
        "force_recurrent_observation": True,
    }
    save_config_with_hyperparameters(
        run_dir, args.config, algo_label, hyperparams,
        resolved_config=cfg_effective, set_overrides=args.set_overrides,
    )

    print_run_summary(run_dir, latest_path, rewards_csv_path, eval_csv_path)

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
