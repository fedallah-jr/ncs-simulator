from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING
import torch
if TYPE_CHECKING:
    from utils.marl import RunningObsNormalizer

def save_qlearning_checkpoint(
    path: Path, algorithm: str, n_agents: int, obs_dim: int, n_actions: int,
    use_agent_id: bool, parameter_sharing: bool, agent_hidden_dims: List[int],
    agent_activation: str, agent_layer_norm: bool, dueling: bool,
    stream_hidden_dim: Optional[int], agent: torch.nn.Module,
    obs_normalizer: Optional["RunningObsNormalizer"],
    mixer: Optional[torch.nn.Module] = None, mixer_params: Optional[Dict[str, Any]] = None,
) -> None:
    ckpt: Dict[str, Any] = {
        "algorithm": algorithm, "n_agents": n_agents, "obs_dim": obs_dim,
        "n_actions": n_actions, "use_agent_id": use_agent_id,
        "parameter_sharing": parameter_sharing, "agent_hidden_dims": agent_hidden_dims,
        "agent_activation": agent_activation, "agent_layer_norm": agent_layer_norm,
        "dueling": dueling, "stream_hidden_dim": stream_hidden_dim if dueling else None,
    }
    if mixer_params is not None:
        ckpt.update(mixer_params)
    ckpt["obs_normalization"] = (
        obs_normalizer.state_dict() if obs_normalizer is not None else {"enabled": False}
    )
    if isinstance(agent, torch.nn.ModuleList):
        ckpt["agent_state_dicts"] = [net.state_dict() for net in agent]
    else:
        ckpt["agent_state_dict"] = agent.state_dict()
    if mixer is not None:
        ckpt["mixer_state_dict"] = mixer.state_dict()
    torch.save(ckpt, path)

def save_mappo_checkpoint(
    path: Path, n_agents: int, obs_dim: int, n_actions: int, use_agent_id: bool,
    team_reward: bool, agent_hidden_dims: List[int], agent_activation: str,
    agent_layer_norm: bool, critic_hidden_dims: List[int], critic_activation: str,
    critic_layer_norm: bool, actor: torch.nn.Module, critic: torch.nn.Module,
    obs_normalizer: Optional["RunningObsNormalizer"],
) -> None:
    ckpt: Dict[str, Any] = {
        "algorithm": "mappo", "n_agents": n_agents, "obs_dim": obs_dim,
        "n_actions": n_actions, "use_agent_id": use_agent_id,
        "parameter_sharing": True, "team_reward": team_reward,
        "agent_hidden_dims": agent_hidden_dims, "agent_activation": agent_activation,
        "agent_layer_norm": agent_layer_norm, "dueling": False, "stream_hidden_dim": None,
        "agent_state_dict": actor.state_dict(), "critic_state_dict": critic.state_dict(),
        "critic_hidden_dims": critic_hidden_dims, "critic_activation": critic_activation,
        "critic_layer_norm": critic_layer_norm,
    }
    ckpt["obs_normalization"] = (
        obs_normalizer.state_dict() if obs_normalizer is not None else {"enabled": False}
    )
    torch.save(ckpt, path)

def save_qlearning_training_state(
    path: Path, learner: Any, buffer: Any, obs_normalizer: Any,
    best_model_tracker: Any, global_step: int, episode: int,
    last_eval_step: int, vector_step: int,
) -> None:
    state: Dict[str, Any] = {
        "learner": learner.state_dict(),
        "buffer": buffer.state_dict(),
        "obs_normalizer": obs_normalizer.state_dict() if obs_normalizer is not None else None,
        "best_model_tracker": dict(best_model_tracker._best),
        "global_step": global_step,
        "episode": episode,
        "last_eval_step": last_eval_step,
        "vector_step": vector_step,
    }
    torch.save(state, path)

def load_qlearning_training_state(
    path: Path, learner: Any, buffer: Any, obs_normalizer: Any,
    best_model_tracker: Any,
) -> Dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    learner.load_state_dict(state["learner"])
    buffer.load_state_dict(state["buffer"])
    if state["obs_normalizer"] is not None and obs_normalizer is not None:
        from utils.marl.obs_normalization import RunningObsNormalizer
        restored = RunningObsNormalizer.from_state_dict(state["obs_normalizer"])
        obs_normalizer.mean = restored.mean
        obs_normalizer.m2 = restored.m2
        obs_normalizer.count = restored.count
    best_model_tracker._best = dict(state["best_model_tracker"])
    return {
        "global_step": int(state["global_step"]),
        "episode": int(state["episode"]),
        "last_eval_step": int(state["last_eval_step"]),
        "vector_step": int(state["vector_step"]),
    }

def save_mappo_training_state(
    path: Path, actor: torch.nn.Module, critic: torch.nn.Module,
    actor_optimizer: torch.optim.Optimizer, critic_optimizer: torch.optim.Optimizer,
    value_normalizer: Any, obs_normalizer: Any, best_model_tracker: Any,
    global_step: int, episode: int, last_eval_step: int,
) -> None:
    state: Dict[str, Any] = {
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "actor_optimizer": actor_optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
        "value_normalizer": value_normalizer.state_dict(),
        "obs_normalizer": obs_normalizer.state_dict() if obs_normalizer is not None else None,
        "best_model_tracker": dict(best_model_tracker._best),
        "global_step": global_step,
        "episode": episode,
        "last_eval_step": last_eval_step,
    }
    torch.save(state, path)

def load_mappo_training_state(
    path: Path, actor: torch.nn.Module, critic: torch.nn.Module,
    actor_optimizer: torch.optim.Optimizer, critic_optimizer: torch.optim.Optimizer,
    value_normalizer: Any, obs_normalizer: Any, best_model_tracker: Any,
) -> Dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    actor.load_state_dict(state["actor"])
    critic.load_state_dict(state["critic"])
    actor_optimizer.load_state_dict(state["actor_optimizer"])
    critic_optimizer.load_state_dict(state["critic_optimizer"])
    value_normalizer.load_state_dict(state["value_normalizer"])
    if state["obs_normalizer"] is not None and obs_normalizer is not None:
        from utils.marl.obs_normalization import RunningObsNormalizer
        restored = RunningObsNormalizer.from_state_dict(state["obs_normalizer"])
        obs_normalizer.mean = restored.mean
        obs_normalizer.m2 = restored.m2
        obs_normalizer.count = restored.count
    best_model_tracker._best = dict(state["best_model_tracker"])
    return {
        "global_step": int(state["global_step"]),
        "episode": int(state["episode"]),
        "last_eval_step": int(state["last_eval_step"]),
    }

def build_qlearning_hyperparams(
    algorithm: str, args: Any, n_agents: int, use_agent_id: bool,
    device: torch.device, mixer_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    hyperparams: Dict[str, Any] = {
        "total_timesteps": args.total_timesteps, "episode_length": args.episode_length,
        "n_agents": n_agents, "n_envs": args.n_envs,
        "buffer_size": args.buffer_size, "batch_size": args.batch_size,
        "start_learning": args.start_learning, "train_interval": args.train_interval,
        "learning_rate": args.learning_rate, "gamma": args.gamma,
        "target_update_interval": args.target_update_interval,
        "grad_clip_norm": args.grad_clip_norm, "double_q": args.double_q,
        "optimizer": args.optimizer, "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end, "epsilon_decay_steps": args.epsilon_decay_steps,
        "hidden_dims": list(args.hidden_dims), "activation": args.activation,
        "layer_norm": args.layer_norm, "dueling": args.dueling,
        "stream_hidden_dim": args.stream_hidden_dim, "use_agent_id": use_agent_id,
        "independent_agents": args.independent_agents, "normalize_obs": args.normalize_obs,
        "obs_norm_clip": args.obs_norm_clip, "obs_norm_eps": args.obs_norm_eps,
        "eval_freq": args.eval_freq, "n_eval_episodes": args.n_eval_episodes,
        "n_eval_envs": args.n_eval_envs, "device": str(device), "seed": args.seed,
    }
    if algorithm == "iql" and hasattr(args, "team_reward"):
        hyperparams["team_reward"] = args.team_reward
    if mixer_params is not None:
        hyperparams.update(mixer_params)
    return hyperparams

def build_mappo_hyperparams(
    args: Any, n_agents: int, use_agent_id: bool, device: torch.device,
) -> Dict[str, Any]:
    return {
        "total_timesteps": args.total_timesteps, "episode_length": args.episode_length,
        "n_agents": n_agents, "n_envs": args.n_envs, "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs, "learning_rate": args.learning_rate,
        "lr_decay": args.lr_decay, "gamma": args.gamma,
        "gae_lambda": args.gae_lambda, "clip_range": args.clip_range,
        "ent_coef": args.ent_coef, "vf_coef": args.vf_coef, "huber_delta": args.huber_delta,
        "value_norm": True, "value_norm_beta": args.value_norm_beta,
        "team_reward": args.team_reward, "normalize_obs": args.normalize_obs,
        "obs_norm_clip": args.obs_norm_clip, "obs_norm_eps": args.obs_norm_eps,
        "max_grad_norm": args.max_grad_norm, "hidden_dims": list(args.hidden_dims),
        "activation": args.activation, "layer_norm": args.layer_norm,
        "use_agent_id": use_agent_id, "eval_freq": args.eval_freq,
        "n_eval_episodes": args.n_eval_episodes, "n_eval_envs": args.n_eval_envs,
        "device": str(device), "seed": args.seed,
    }
