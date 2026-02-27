from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any, Dict


def _add_set_override_argument(parser: argparse.ArgumentParser) -> None:
    """Add the ``--set KEY=VALUE`` argument to an argparse parser."""
    parser.add_argument(
        "--set",
        action="append",
        default=None,
        dest="set_overrides",
        metavar="KEY=VALUE",
        help="Override config values using dot notation (e.g., --set reward.state_error_reward=kf_info_m_noise). Repeatable.",
    )


def build_base_qlearning_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--resume", type=Path, default=None, help="Resume training from a previous run directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--episode-length", type=int, default=250)
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--start-learning", type=int, default=1_000)
    parser.add_argument("--train-interval", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target-update-interval", type=int, default=500)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--double-q", action="store_true")
    parser.add_argument("--optimizer", type=str, default="rmsprop", choices=["adam", "rmsprop"])
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=100_000)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--activation", type=str, default="tanh", choices=["relu", "tanh", "elu"])
    parser.add_argument("--feature-norm", action="store_true",
                        help="Apply LayerNorm to input features (first layer).")
    parser.add_argument("--layer-norm", action="store_true",
                        help="Apply LayerNorm after each hidden layer.")
    parser.add_argument("--dueling", action="store_true")
    parser.add_argument("--stream-hidden-dim", type=int, default=64)
    parser.add_argument("--no-agent-id", action="store_true")
    parser.add_argument("--independent-agents", action="store_true")
    obs_norm_group = parser.add_mutually_exclusive_group()
    obs_norm_group.add_argument("--normalize-obs", action="store_true")
    obs_norm_group.add_argument("--no-normalize-obs", action="store_false", dest="normalize_obs")
    parser.add_argument("--obs-norm-clip", type=float, default=5.0)
    parser.add_argument("--obs-norm-eps", type=float, default=1e-8)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--n-eval-episodes", type=int, default=30)
    parser.add_argument("--n-eval-envs", type=int, default=4)
    _add_set_override_argument(parser)
    parser.set_defaults(normalize_obs=True)
    return parser

def add_qmix_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mixer-hidden-dim", type=int, default=32)
    parser.add_argument("--hypernet-hidden-dim", type=int, default=64)

def add_qplex_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mixing-embed-dim", type=int, default=32)
    parser.add_argument("--hypernet-embed", type=int, default=64)
    parser.add_argument("--adv-hypernet-layers", type=int, default=3)
    parser.add_argument("--adv-hypernet-embed", type=int, default=64)
    parser.add_argument("--num-kernel", type=int, default=10)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--attend-reg-coef", type=float, default=0.001)
    parser.add_argument("--nonlinear", action="store_true")
    parser.add_argument("--no-weighted-head", action="store_false", dest="weighted_head")
    parser.add_argument("--no-state-bias", action="store_false", dest="state_bias")
    parser.add_argument("--no-is-minus-one", action="store_false", dest="is_minus_one")
    parser.set_defaults(weighted_head=True, state_bias=True, is_minus_one=True)

def build_happo_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--resume", type=Path, default=None, help="Resume training from a previous run directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--episode-length", type=int, default=250)
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument(
        "--num-mini-batch",
        type=int,
        default=1,
        help="Number of mini-batches per PPO epoch (1 = full-batch update).",
    )
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--lr-decay", action="store_true", dest="lr_decay")
    parser.add_argument("--no-lr-decay", action="store_false", dest="lr_decay")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=1.0)
    parser.add_argument("--huber-delta", type=float, default=10.0)
    parser.add_argument("--value-norm-beta", type=float, default=0.99999)
    parser.add_argument(
        "--value-norm-per-element-update",
        action="store_true",
        help="Scale ValueNorm decay by the number of elements per update (on-policy style).",
    )
    parser.add_argument("--popart", action="store_true",
                        help="Use PopArt value normalization (output-preserving weight correction)")
    parser.add_argument("--popart-beta", type=float, default=0.999)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "tanh", "elu"])
    parser.add_argument("--feature-norm", action="store_true",
                        help="Apply LayerNorm to input features (first layer).")
    parser.add_argument("--layer-norm", action="store_true",
                        help="Apply LayerNorm after each hidden layer.")
    parser.add_argument("--fixed-order", action="store_true",
                        help="Use fixed agent update order instead of random shuffle each iteration")
    obs_norm_group = parser.add_mutually_exclusive_group()
    obs_norm_group.add_argument("--normalize-obs", action="store_true")
    obs_norm_group.add_argument("--no-normalize-obs", action="store_false", dest="normalize_obs")
    parser.add_argument("--obs-norm-clip", type=float, default=5.0)
    parser.add_argument("--obs-norm-eps", type=float, default=1e-8)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--n-eval-episodes", type=int, default=30)
    parser.add_argument("--n-eval-envs", type=int, default=4)
    _add_set_override_argument(parser)
    parser.set_defaults(normalize_obs=True, lr_decay=False)
    return parser


def build_happo_hyperparams(
    args: Any, n_agents: int, device: Any,
) -> Dict[str, Any]:
    return {
        "total_timesteps": args.total_timesteps, "episode_length": args.episode_length,
        "n_agents": n_agents, "n_envs": args.n_envs, "n_steps": args.n_steps,
        "num_mini_batch": args.num_mini_batch,
        "n_epochs": args.n_epochs, "learning_rate": args.learning_rate,
        "lr_decay": args.lr_decay, "gamma": args.gamma,
        "gae_lambda": args.gae_lambda, "clip_range": args.clip_range,
        "ent_coef": args.ent_coef, "vf_coef": args.vf_coef, "huber_delta": args.huber_delta,
        "value_norm": True,
        "value_norm_beta": args.value_norm_beta,
        "value_norm_per_element_update": args.value_norm_per_element_update,
        "popart": args.popart,
        "popart_beta": args.popart_beta,
        "normalize_obs": args.normalize_obs,
        "obs_norm_clip": args.obs_norm_clip, "obs_norm_eps": args.obs_norm_eps,
        "max_grad_norm": args.max_grad_norm, "hidden_dims": list(args.hidden_dims),
        "activation": args.activation, "feature_norm": args.feature_norm,
        "layer_norm": args.layer_norm,
        "fixed_order": args.fixed_order, "parameter_sharing": False,
        "eval_freq": args.eval_freq,
        "n_eval_episodes": args.n_eval_episodes, "n_eval_envs": args.n_eval_envs,
        "device": str(device), "seed": args.seed,
    }


def build_mappo_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--resume", type=Path, default=None, help="Resume training from a previous run directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--episode-length", type=int, default=250)
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument(
        "--num-mini-batch",
        type=int,
        default=1,
        help="Number of mini-batches per PPO epoch (1 = full-batch update).",
    )
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--lr-decay", action="store_true", dest="lr_decay")
    parser.add_argument("--no-lr-decay", action="store_false", dest="lr_decay")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=1.0)
    parser.add_argument("--huber-delta", type=float, default=10.0)
    parser.add_argument("--value-norm-beta", type=float, default=0.99999)
    parser.add_argument(
        "--value-norm-per-element-update",
        action="store_true",
        help="Scale ValueNorm decay by the number of elements per update (on-policy style).",
    )
    parser.add_argument("--popart", action="store_true",
                        help="Use PopArt value normalization (output-preserving weight correction)")
    parser.add_argument("--popart-beta", type=float, default=0.999)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "tanh", "elu"])
    parser.add_argument("--feature-norm", action="store_true",
                        help="Apply LayerNorm to input features (first layer).")
    parser.add_argument("--layer-norm", action="store_true",
                        help="Apply LayerNorm after each hidden layer.")
    parser.add_argument("--no-agent-id", action="store_true")
    parser.add_argument("--independent-agents", action="store_true")
    obs_norm_group = parser.add_mutually_exclusive_group()
    obs_norm_group.add_argument("--normalize-obs", action="store_true")
    obs_norm_group.add_argument("--no-normalize-obs", action="store_false", dest="normalize_obs")
    parser.add_argument("--obs-norm-clip", type=float, default=5.0)
    parser.add_argument("--obs-norm-eps", type=float, default=1e-8)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--n-eval-episodes", type=int, default=30)
    parser.add_argument("--n-eval-envs", type=int, default=4)
    _add_set_override_argument(parser)
    parser.set_defaults(normalize_obs=True, lr_decay=False)
    return parser
