"""
Multi-agent reinforcement learning (MARL) utilities.

This package provides shared components used by executable training scripts under
`algorithms/` (e.g., IQL/VDN/QMIX).
"""

from __future__ import annotations

from .buffer import MARLReplayBuffer, MARLBatch
from .common import (
    select_device,
    epsilon_by_step,
    stack_obs,
    select_actions,
    select_actions_batched,
    run_evaluation,
    run_evaluation_vectorized,
    patch_autoreset_final_obs,
    QLearnStepResult,
    qlearning_collect_transition,
)
from .learners import IQLLearner, VDNLearner, QMIXLearner, QPLEXLearner
from .networks import (
    CentralValueMLP,
    MLPAgent,
    DuelingMLPAgent,
    QMixer,
    QPLEXMixer,
    VDNMixer,
    append_agent_id,
)
from .obs_normalization import RunningObsNormalizer
from .popart import PopArtLayer
from .value_norm import ValueNorm
from .args_builder import (
    build_base_qlearning_parser,
    add_team_reward_arg,
    add_qmix_args,
    add_qplex_args,
    build_happo_parser,
    build_happo_hyperparams,
    build_mappo_parser,
)
from .checkpoint_utils import (
    save_qlearning_checkpoint,
    save_happo_checkpoint,
    save_happo_training_state,
    load_happo_training_state,
    load_happo_arch_args,
    save_mappo_checkpoint,
    save_qlearning_training_state,
    load_qlearning_training_state,
    save_mappo_training_state,
    load_mappo_training_state,
    build_qlearning_hyperparams,
    build_mappo_hyperparams,
)

__all__ = [
    "MARLBatch",
    "MARLReplayBuffer",
    "QLearnStepResult",
    "add_qmix_args",
    "add_qplex_args",
    "add_team_reward_arg",
    "append_agent_id",
    "build_base_qlearning_parser",
    "build_happo_hyperparams",
    "build_happo_parser",
    "build_mappo_hyperparams",
    "build_mappo_parser",
    "build_qlearning_hyperparams",
    "CentralValueMLP",
    "DuelingMLPAgent",
    "epsilon_by_step",
    "IQLLearner",
    "load_happo_arch_args",
    "load_happo_training_state",
    "load_mappo_training_state",
    "load_qlearning_training_state",
    "MLPAgent",
    "QMIXLearner",
    "QMixer",
    "QPLEXLearner",
    "QPLEXMixer",
    "RunningObsNormalizer",
    "patch_autoreset_final_obs",
    "PopArtLayer",
    "qlearning_collect_transition",
    "run_evaluation",
    "run_evaluation_vectorized",
    "save_happo_checkpoint",
    "save_happo_training_state",
    "save_mappo_checkpoint",
    "save_mappo_training_state",
    "save_qlearning_checkpoint",
    "save_qlearning_training_state",
    "select_actions",
    "select_actions_batched",
    "select_device",
    "stack_obs",
    "VDNLearner",
    "VDNMixer",
    "ValueNorm",
]
