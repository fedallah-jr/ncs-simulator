from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ncs_env.joint_action_env import CentralizedJointActionEnv
from utils.marl_training import load_config_with_overrides
from utils.run_utils import BestModelTracker, prepare_run_directory, save_config_with_hyperparameters

try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv, VecNormalize, sync_envs_normalization
except ImportError as exc:  # pragma: no cover - exercised only when dependency is missing
    DQN = None  # type: ignore[assignment]
    BaseCallback = object  # type: ignore[assignment]
    SubprocVecEnv = None  # type: ignore[assignment]
    VecEnv = object  # type: ignore[assignment]
    VecNormalize = object  # type: ignore[assignment]
    sync_envs_normalization = None  # type: ignore[assignment]
    SB3_IMPORT_ERROR: Optional[ImportError] = exc
else:
    SB3_IMPORT_ERROR = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a centralized joint-action DQN (Stable-Baselines3) on the multi-agent NCS env. "
            "Observations are concatenated across agents and rewards are summed."
        )
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--episode-length", type=int, default=500)
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=8)

    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--target-update-interval", type=int, default=500)
    parser.add_argument("--exploration-fraction", type=float, default=0.6)
    parser.add_argument("--exploration-initial-eps", type=float, default=1.0)
    parser.add_argument("--exploration-final-eps", type=float, default=0.05)
    parser.add_argument("--net-arch", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    obs_norm_group = parser.add_mutually_exclusive_group()
    obs_norm_group.add_argument("--normalize-obs", action="store_true")
    obs_norm_group.add_argument("--no-normalize-obs", action="store_false", dest="normalize_obs")
    parser.add_argument("--obs-norm-clip", type=float, default=5.0)
    parser.add_argument("--obs-norm-eps", type=float, default=1e-8)

    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-freq", type=int, default=5_000)
    parser.add_argument("--n-eval-episodes", type=int, default=30)
    parser.add_argument("--n-eval-envs", type=int, default=4)
    parser.set_defaults(normalize_obs=True)
    return parser.parse_args()


def make_joint_env_factory(
    *,
    n_agents: int,
    episode_length: int,
    config_path_str: Optional[str],
    seed: Optional[int],
    reward_override: Optional[Dict[str, Any]] = None,
    termination_override: Optional[Dict[str, Any]] = None,
    freeze_running_normalization: bool = False,
    minimal_info: bool = True,
) -> Callable[[], CentralizedJointActionEnv]:
    def _thunk() -> CentralizedJointActionEnv:
        return CentralizedJointActionEnv(
            n_agents=n_agents,
            episode_length=episode_length,
            config_path=config_path_str,
            seed=seed,
            reward_override=reward_override,
            termination_override=termination_override,
            freeze_running_normalization=freeze_running_normalization,
            minimal_info=minimal_info,
        )

    return _thunk


def evaluate_joint_policy_vectorized(
    model: DQN,
    eval_env: VecEnv,
    n_eval_envs: int,
    n_episodes: int,
    seed: Optional[int],
    train_env_for_norm: Optional[VecEnv],
) -> Tuple[float, float]:
    if train_env_for_norm is not None:
        try:
            sync_envs_normalization(train_env_for_norm, eval_env)
        except AttributeError:
            # No VecNormalize wrappers were present; nothing to sync.
            pass

    if seed is not None:
        eval_env.seed(int(seed))
    obs = eval_env.reset()
    env_reward_sums = np.zeros(int(n_eval_envs), dtype=np.float64)
    episode_rewards: list[float] = []

    while len(episode_rewards) < int(n_episodes):
        actions, _state = model.predict(obs, deterministic=True)
        obs, rewards, dones, _infos = eval_env.step(actions)
        rewards_arr = np.asarray(rewards, dtype=np.float64).reshape(int(n_eval_envs))
        env_reward_sums += rewards_arr

        done_arr = np.asarray(dones, dtype=bool).reshape(int(n_eval_envs))
        if np.any(done_arr):
            for env_idx in np.where(done_arr)[0]:
                if len(episode_rewards) < int(n_episodes):
                    episode_rewards.append(float(env_reward_sums[env_idx]))
                env_reward_sums[env_idx] = 0.0

    reward_arr = np.asarray(episode_rewards, dtype=np.float32)
    return float(np.mean(reward_arr)), float(np.std(reward_arr))


class JointTrainEvalCallback(BaseCallback):
    def __init__(
        self,
        *,
        eval_env: VecEnv,
        n_eval_envs: int,
        train_env_for_norm: VecEnv,
        eval_freq: int,
        n_eval_episodes: int,
        seed: Optional[int],
        train_writer: csv.writer,
        train_file: Any,
        eval_writer: csv.writer,
        eval_file: Any,
        best_model_tracker: BestModelTracker,
        run_dir: Path,
        log_interval: int,
    ) -> None:
        super().__init__(verbose=0)
        self.eval_env = eval_env
        self.n_eval_envs = int(n_eval_envs)
        self.train_env_for_norm = train_env_for_norm
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.seed = seed
        self.train_writer = train_writer
        self.train_file = train_file
        self.eval_writer = eval_writer
        self.eval_file = eval_file
        self.best_model_tracker = best_model_tracker
        self.run_dir = run_dir
        self.log_interval = int(log_interval)

        self.n_train_envs = int(train_env_for_norm.num_envs)
        self.train_episode_rewards = np.zeros(self.n_train_envs, dtype=np.float64)
        self.train_episode_lengths = np.zeros(self.n_train_envs, dtype=np.int64)
        self.episode = 0
        self.last_eval_step = 0

    def _save_model(self, path: Path) -> None:
        self.model.save(str(path))
        if isinstance(self.train_env_for_norm, VecNormalize):
            self.train_env_for_norm.save(str(path.with_suffix(".vecnormalize.pkl")))

    def _run_eval(self, step: int) -> None:
        mean_reward, std_reward = evaluate_joint_policy_vectorized(
            model=self.model,
            eval_env=self.eval_env,
            n_eval_envs=self.n_eval_envs,
            n_episodes=self.n_eval_episodes,
            seed=self.seed,
            train_env_for_norm=self.train_env_for_norm,
        )
        self.eval_writer.writerow([int(step), float(mean_reward), float(std_reward)])
        self.eval_file.flush()
        self.best_model_tracker.update(
            "eval",
            float(mean_reward),
            self.run_dir / "best_model.zip",
            self._save_model,
        )
        print(
            f"[JOINT_DQN_SB3] Eval at step {step}: "
            f"mean_reward={mean_reward:.3f} std={std_reward:.3f}"
        )

    def run_final_eval_if_needed(self) -> None:
        current_step = int(self.model.num_timesteps)
        if current_step == self.last_eval_step:
            return
        self._run_eval(current_step)
        self.last_eval_step = current_step

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")

        if rewards is not None and dones is not None:
            rewards_arr = np.asarray(rewards, dtype=np.float64).reshape(self.n_train_envs)
            dones_arr = np.asarray(dones, dtype=bool).reshape(self.n_train_envs)
            self.train_episode_rewards += rewards_arr
            self.train_episode_lengths += 1

            if np.any(dones_arr):
                step = int(self.num_timesteps)
                for env_idx in np.where(dones_arr)[0]:
                    episode_reward = float(self.train_episode_rewards[env_idx])
                    episode_length = int(self.train_episode_lengths[env_idx])
                    self.train_writer.writerow([self.episode, episode_reward, episode_length, step])
                    self.train_file.flush()

                    self.best_model_tracker.update(
                        "train",
                        episode_reward,
                        self.run_dir / "best_train_model.zip",
                        self._save_model,
                    )

                    if self.episode % self.log_interval == 0:
                        print(
                            f"[JOINT_DQN_SB3] episode={self.episode} steps={step} "
                            f"reward_sum={episode_reward:.3f}"
                        )

                    self.episode += 1
                    self.train_episode_rewards[env_idx] = 0.0
                    self.train_episode_lengths[env_idx] = 0

        if self.eval_freq > 0 and (int(self.num_timesteps) - self.last_eval_step) >= self.eval_freq:
            self._run_eval(int(self.num_timesteps))
            self.last_eval_step = int(self.num_timesteps)

        return True


def main() -> None:
    if DQN is None:
        raise ImportError(
            "stable-baselines3 is required for algorithms.joint_dqn_sb3. "
            "Install dependencies from requirements.txt."
        ) from SB3_IMPORT_ERROR

    args = parse_args()
    (
        cfg,
        config_path_str,
        n_agents,
        _use_agent_id,
        eval_reward_override,
        eval_termination_override,
        _network_override,
        _training_reward_override,
    ) = load_config_with_overrides(args.config, args.n_agents, False)

    run_dir = prepare_run_directory("joint_dqn_sb3", args.config, args.output_root)
    rewards_csv_path = run_dir / "training_rewards.csv"
    eval_csv_path = run_dir / "evaluation_rewards.csv"
    # Joint SB3 trainers use VecNormalize for reward scaling; disable env-side reward normalization.
    train_reward_override: Dict[str, Any] = {"normalize": False}
    eval_reward_override_merged: Dict[str, Any] = (
        dict(eval_reward_override) if isinstance(eval_reward_override, dict) else {}
    )
    eval_reward_override_merged["normalize"] = False

    train_base_env = SubprocVecEnv(
        [
            make_joint_env_factory(
                n_agents=n_agents,
                episode_length=args.episode_length,
                config_path_str=config_path_str,
                seed=None if args.seed is None else int(args.seed) + env_idx,
                reward_override=train_reward_override,
                minimal_info=True,
            )
            for env_idx in range(int(args.n_envs))
        ]
    )
    eval_base_env = SubprocVecEnv(
        [
            make_joint_env_factory(
                n_agents=n_agents,
                episode_length=args.episode_length,
                config_path_str=config_path_str,
                seed=None if args.seed is None else int(args.seed) + env_idx,
                reward_override=eval_reward_override_merged,
                termination_override=eval_termination_override,
                freeze_running_normalization=True,
                minimal_info=True,
            )
            for env_idx in range(int(args.n_eval_envs))
        ]
    )

    train_env: VecEnv = VecNormalize(
        train_base_env,
        training=True,
        norm_obs=bool(args.normalize_obs),
        norm_reward=True,
        clip_obs=float(args.obs_norm_clip),
        epsilon=float(args.obs_norm_eps),
    )
    eval_env: VecEnv = VecNormalize(
        eval_base_env,
        training=False,
        norm_obs=bool(args.normalize_obs),
        norm_reward=False,
        clip_obs=float(args.obs_norm_clip),
        epsilon=float(args.obs_norm_eps),
    )

    # Create a temporary env to read action-space metadata (SubprocVecEnv doesn't expose .envs).
    _tmp_env = CentralizedJointActionEnv(
        n_agents=n_agents, episode_length=args.episode_length, config_path=config_path_str,
    )
    per_agent_n_actions = int(_tmp_env.per_agent_n_actions)
    n_joint_actions = int(_tmp_env.n_joint_actions)
    _tmp_env.close()

    policy_kwargs: Dict[str, Any] = {"net_arch": list(args.net_arch)}
    model = DQN(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=float(args.learning_rate),
        buffer_size=int(args.buffer_size),
        learning_starts=int(args.learning_starts),
        batch_size=int(args.batch_size),
        tau=float(args.tau),
        gamma=float(args.gamma),
        train_freq=int(args.train_freq),
        gradient_steps=int(args.gradient_steps),
        target_update_interval=int(args.target_update_interval),
        exploration_fraction=float(args.exploration_fraction),
        exploration_initial_eps=float(args.exploration_initial_eps),
        exploration_final_eps=float(args.exploration_final_eps),
        policy_kwargs=policy_kwargs,
        seed=args.seed,
        device=args.device,
        verbose=0,
    )

    best_model_tracker = BestModelTracker()

    with rewards_csv_path.open("w", newline="", encoding="utf-8") as train_f, \
         eval_csv_path.open("w", newline="", encoding="utf-8") as eval_f:
        train_writer = csv.writer(train_f)
        eval_writer = csv.writer(eval_f)
        train_writer.writerow(["episode", "reward_sum", "episode_length", "steps"])
        eval_writer.writerow(["step", "mean_reward", "std_reward"])

        callback = JointTrainEvalCallback(
            eval_env=eval_env,
            n_eval_envs=args.n_eval_envs,
            train_env_for_norm=train_env,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            seed=args.seed,
            train_writer=train_writer,
            train_file=train_f,
            eval_writer=eval_writer,
            eval_file=eval_f,
            best_model_tracker=best_model_tracker,
            run_dir=run_dir,
            log_interval=args.log_interval,
        )

        model.learn(total_timesteps=int(args.total_timesteps), callback=callback, progress_bar=False)
        callback.run_final_eval_if_needed()

    latest_path = run_dir / "latest_model.zip"
    model.save(str(latest_path))
    if isinstance(train_env, VecNormalize):
        train_env.save(str(latest_path.with_suffix(".vecnormalize.pkl")))

    hyperparams: Dict[str, Any] = {
        "total_timesteps": int(args.total_timesteps),
        "episode_length": int(args.episode_length),
        "n_agents": int(n_agents),
        "n_envs": int(args.n_envs),
        "per_agent_n_actions": per_agent_n_actions,
        "joint_action_dim": n_joint_actions,
        "learning_rate": float(args.learning_rate),
        "buffer_size": int(args.buffer_size),
        "learning_starts": int(args.learning_starts),
        "batch_size": int(args.batch_size),
        "tau": float(args.tau),
        "gamma": float(args.gamma),
        "train_freq": int(args.train_freq),
        "gradient_steps": int(args.gradient_steps),
        "target_update_interval": int(args.target_update_interval),
        "exploration_fraction": float(args.exploration_fraction),
        "exploration_initial_eps": float(args.exploration_initial_eps),
        "exploration_final_eps": float(args.exploration_final_eps),
        "net_arch": list(args.net_arch),
        "device": args.device,
        "normalize_obs": bool(args.normalize_obs),
        "sb3_norm_reward_train": True,
        "sb3_norm_reward_eval": False,
        "env_reward_normalization": False,
        "obs_norm_clip": float(args.obs_norm_clip),
        "obs_norm_eps": float(args.obs_norm_eps),
        "eval_freq": int(args.eval_freq),
        "n_eval_episodes": int(args.n_eval_episodes),
        "n_eval_envs": int(args.n_eval_envs),
        "log_interval": int(args.log_interval),
        "reward_mode": cfg.get("reward", {}).get("state_error_reward", "absolute"),
        "team_reward_aggregation": "sum",
    }
    save_config_with_hyperparameters(run_dir, args.config, "joint_dqn_sb3", hyperparams)

    print(f"Run artifacts stored in {run_dir}")
    print(f"  - Latest model: {latest_path}")
    print(f"  - Best eval model: {run_dir / 'best_model.zip'}")
    print(f"  - Best train model: {run_dir / 'best_train_model.zip'}")
    print(f"  - Training rewards: {rewards_csv_path}")
    print(f"  - Evaluation rewards: {eval_csv_path}")
    print(f"  - Config with hyperparameters: {run_dir / 'config.json'}")
    if isinstance(train_env, VecNormalize):
        print(f"  - Latest VecNormalize stats: {latest_path.with_suffix('.vecnormalize.pkl')}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
