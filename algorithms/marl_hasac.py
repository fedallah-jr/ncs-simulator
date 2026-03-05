from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.distributions import Categorical

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.marl import (
    MARLReplayBuffer,
    HASACLearner,
    MLPAgent,
    TwinQNetwork,
    build_hasac_parser,
    build_hasac_hyperparams,
    save_hasac_checkpoint,
    save_hasac_training_state,
    load_hasac_training_state,
    patch_autoreset_final_obs,
    select_actions_batched,
    append_agent_id,
)
from utils.marl_training import (
    setup_device_and_rng,
    load_config_with_overrides,
    resolve_training_eval_baseline,
    create_obs_normalizer,
    print_run_summary,
    setup_shared_reward_normalizer,
    evaluate_and_log,
    log_completed_episodes,
)
from utils.marl.vector_env import create_async_vector_env, create_eval_async_vector_env, stack_vector_obs
from utils.reward_normalization import reset_shared_running_normalizers
from utils.run_utils import prepare_run_directory, save_config_with_hyperparameters, BestModelTracker


def parse_args() -> argparse.Namespace:
    parser = build_hasac_parser(
        description="Train HASAC (Heterogeneous-Agent SAC) on multi-agent NCS env."
    )
    return parser.parse_args()


def _select_actions_stochastic(
    actors: list,
    obs: np.ndarray,
    n_envs: int,
    n_agents: int,
    n_actions: int,
    device: torch.device,
) -> np.ndarray:
    """Sample actions from actor policies (Categorical)."""
    obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)
    actions = np.empty((n_envs, n_agents), dtype=np.int64)
    with torch.no_grad():
        for i, actor in enumerate(actors):
            logits_i = actor(obs_t[:, i, :])
            dist = Categorical(logits=logits_i)
            actions[:, i] = dist.sample().cpu().numpy()
    return actions


def main() -> None:
    reset_shared_running_normalizers()
    args = parse_args()
    device, rng = setup_device_and_rng(args.device, args.seed)

    # HASAC never uses agent_id for actors (independent per-agent networks)
    use_agent_id = False

    cfg, config_path_str, n_agents, _use_agent_id, eval_reward_override, eval_termination_override, network_override, training_reward_override = (
        load_config_with_overrides(args.config, args.n_agents, use_agent_id, args.set_overrides)
    )
    eval_baseline = resolve_training_eval_baseline(cfg, n_agents)

    resuming = args.resume is not None
    if resuming:
        run_dir = Path(args.resume)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Resume directory does not exist: {run_dir}")
    else:
        run_dir = prepare_run_directory("hasac", args.config, args.output_root)
    rewards_csv_path = run_dir / "training_rewards.csv"
    eval_csv_path = run_dir / "evaluation_rewards.csv"

    if args.n_envs <= 0:
        raise ValueError("n_envs must be positive")
    if args.train_interval <= 0:
        raise ValueError("train_interval must be positive")

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
        global_state_enabled=True,
    )

    obs_dim = int(env.single_observation_space.spaces["agent_0"].shape[0])
    n_actions = int(env.single_action_space.spaces["agent_0"].n)
    obs_normalizer = create_obs_normalizer(
        obs_dim, args.normalize_obs, args.obs_norm_clip, args.obs_norm_eps
    )

    obs_dict, info = env.reset(seed=env_seeds)
    obs_raw = stack_vector_obs(obs_dict, n_agents)
    global_state_raw = np.asarray(info.get("global_state"), dtype=np.float32)
    if global_state_raw.ndim != 2:
        raise ValueError("global_state must have shape (n_envs, state_dim)")
    state_dim = int(global_state_raw.shape[-1])

    # Resolve learning rates
    actor_lr = args.actor_lr if args.actor_lr is not None else args.learning_rate
    critic_lr = args.critic_lr if args.critic_lr is not None else args.learning_rate

    # Build per-agent actors (no agent ID, output_gain=0.01 for small logit init)
    actors = [
        MLPAgent(
            input_dim=obs_dim,
            n_actions=n_actions,
            hidden_dims=tuple(args.hidden_dims),
            activation=args.activation,
            feature_norm=args.feature_norm,
            layer_norm=args.layer_norm,
            output_gain=0.01,
        )
        for _ in range(n_agents)
    ]

    # Build centralized twin Q-critic
    # Input: global_state + all agents' one-hot actions
    critic_input_dim = state_dim + n_agents * n_actions
    critic = TwinQNetwork(
        input_dim=critic_input_dim,
        hidden_dims=tuple(args.critic_hidden_dims),
        activation=args.activation,
        feature_norm=args.feature_norm,
        layer_norm=args.layer_norm,
    )

    # Create a torch Generator for reproducible random agent ordering
    torch_rng = torch.Generator(device="cpu")
    torch_rng.manual_seed(args.seed)

    learner = HASACLearner(
        actors=actors,
        critic=critic,
        n_agents=n_agents,
        n_actions=n_actions,
        global_state_dim=state_dim,
        gamma=args.gamma,
        polyak=args.polyak,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        alpha=args.alpha,
        auto_alpha=args.auto_alpha,
        alpha_lr=args.alpha_lr,
        target_entropy=args.target_entropy,
        grad_clip_norm=args.max_grad_norm,
        fixed_order=args.fixed_order,
        use_huber_loss=args.use_huber_loss,
        huber_delta=args.huber_delta,
        device=device,
        rng=torch_rng,
    )

    buffer = MARLReplayBuffer(
        capacity=args.buffer_size,
        n_agents=n_agents,
        obs_dim=obs_dim,
        state_dim=state_dim,
        device=device,
        rng=rng,
        n_step=args.n_step,
        gamma=args.gamma,
        n_envs=args.n_envs,
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
        counters = load_hasac_training_state(
            training_state_path, learner, buffer, obs_normalizer, best_model_tracker,
        )
        global_step = counters["global_step"]
        episode = counters["episode"]
        last_eval_step = counters["last_eval_step"]
        vector_step = counters["vector_step"]
        print(f"Resumed from {run_dir} at step {global_step}")

    def save_checkpoint(path: Path) -> None:
        save_hasac_checkpoint(
            path=path,
            n_agents=n_agents,
            obs_dim=obs_dim,
            n_actions=n_actions,
            global_state_dim=state_dim,
            agent_hidden_dims=list(args.hidden_dims),
            agent_activation=args.activation,
            critic_hidden_dims=list(args.critic_hidden_dims),
            actors=learner.actors,
            critic=learner.critic,
            obs_normalizer=obs_normalizer,
            feature_norm=args.feature_norm,
            layer_norm=args.layer_norm,
        )

    def save_training_state() -> None:
        save_hasac_training_state(
            run_dir / "training_state.pt", learner, buffer, obs_normalizer,
            best_model_tracker, global_step, episode, last_eval_step, vector_step,
        )

    csv_mode = "a" if resuming else "w"
    with rewards_csv_path.open(csv_mode, newline="", encoding="utf-8") as train_f, \
         eval_csv_path.open(csv_mode, newline="", encoding="utf-8") as eval_f:
        train_writer = csv.writer(train_f)
        eval_writer = csv.writer(eval_f)
        if not resuming:
            train_writer.writerow(["episode", "reward_sum", "steps"])
            eval_writer.writerow(["step", "mean_reward", "std_reward"])

        episode_reward_sums = np.zeros((args.n_envs,), dtype=np.float32)

        while global_step < args.total_timesteps:
            # Normalize obs
            if obs_normalizer is not None:
                obs = obs_normalizer.normalize(obs_raw, update=True)
            else:
                obs = obs_raw

            # Action selection: warmup -> random, after -> stochastic from actors
            if global_step < args.start_learning:
                actions = rng.integers(0, n_actions, size=(args.n_envs, n_agents))
            else:
                actions = _select_actions_stochastic(
                    learner.actors, obs, args.n_envs, n_agents, n_actions, device,
                )

            # Step environment
            action_dict = {f"agent_{i}": actions[:, i] for i in range(n_agents)}
            next_obs_dict, rewards_arr, terminated, truncated, infos = env.step(action_dict)
            next_obs_raw = stack_vector_obs(next_obs_dict, n_agents)
            next_global_state_raw = np.asarray(infos.get("global_state"), dtype=np.float32)

            rewards_arr = np.asarray(rewards_arr, dtype=np.float32)
            terminated_any = np.asarray(terminated, dtype=np.bool_)
            truncated_any = np.asarray(truncated, dtype=np.bool_)
            done_reset = np.logical_or(terminated_any, truncated_any)

            episode_reward_sums += rewards_arr.sum(axis=1)

            next_obs_for_buffer, next_global_state_for_buffer = patch_autoreset_final_obs(
                next_obs_raw, infos, done_reset, n_agents,
                next_global_state_raw=next_global_state_raw,
            )

            buffer.add_batch(
                obs=obs_raw,
                actions=actions,
                rewards=rewards_arr,
                next_obs=next_obs_for_buffer,
                dones=terminated_any,
                states=global_state_raw,
                next_states=next_global_state_for_buffer,
                resets=done_reset.astype(np.float32),
            )

            obs_raw = next_obs_raw
            global_state_raw = next_global_state_raw
            global_step += args.n_envs
            vector_step += 1

            episode = log_completed_episodes(
                done_reset=done_reset, episode_reward_sums=episode_reward_sums,
                global_step=global_step, episode=episode,
                train_writer=train_writer, train_f=train_f,
                best_model_tracker=best_model_tracker, run_dir=run_dir,
                save_checkpoint=save_checkpoint, log_interval=args.log_interval,
                algo_name="HASAC",
            )

            if len(buffer) >= args.start_learning and vector_step % args.train_interval == 0:
                batch = buffer.sample(args.batch_size, obs_normalizer=obs_normalizer)
                learner.update(batch)

            if global_step - last_eval_step >= args.eval_freq:
                # For evaluation, wrap actors in a ModuleList so select_actions_batched works
                eval_agent = torch.nn.ModuleList(learner.actors)
                evaluate_and_log(
                    eval_env=eval_env, agent=eval_agent,
                    n_eval_envs=args.n_eval_envs, n_agents=n_agents, n_actions=n_actions,
                    use_agent_id=use_agent_id, device=device,
                    n_episodes=args.n_eval_episodes, seed=eval_seed,
                    obs_normalizer=obs_normalizer, eval_writer=eval_writer, eval_f=eval_f,
                    best_model_tracker=best_model_tracker, run_dir=run_dir,
                    save_checkpoint=save_checkpoint, global_step=global_step,
                    algo_name="HASAC",
                    eval_baseline=eval_baseline,
                )
                eval_seed += args.n_eval_episodes
                last_eval_step = global_step
                save_training_state()

    latest_path = run_dir / "latest_model.pt"
    save_checkpoint(latest_path)
    save_training_state()

    hyperparams = build_hasac_hyperparams(
        args=args,
        n_agents=n_agents,
        device=device,
    )
    save_config_with_hyperparameters(
        run_dir,
        args.config,
        "hasac",
        hyperparams,
        resolved_config=cfg,
        set_overrides=args.set_overrides,
    )

    print_run_summary(run_dir, latest_path, rewards_csv_path, eval_csv_path)

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
