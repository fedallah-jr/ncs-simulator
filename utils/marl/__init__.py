"""
Multi-agent reinforcement learning (MARL) utilities.

This package provides shared components used by executable training scripts under
`algorithms/` (e.g., IQL/VDN/QMIX).
"""

from __future__ import annotations

from .buffer import MARLReplayBuffer, MARLBatch
from .common import select_device, epsilon_by_step, stack_obs, select_actions, run_evaluation
from .learners import IQLLearner, VDNLearner, QMIXLearner
from .networks import CentralValueMLP, MLPAgent, DuelingMLPAgent, QMixer, VDNMixer, append_agent_id
from .obs_normalization import RunningObsNormalizer
from .value_norm import ValueNorm

__all__ = [
    "MARLBatch",
    "MARLReplayBuffer",
    "append_agent_id",
    "CentralValueMLP",
    "DuelingMLPAgent",
    "epsilon_by_step",
    "IQLLearner",
    "MLPAgent",
    "QMIXLearner",
    "QMixer",
    "RunningObsNormalizer",
    "run_evaluation",
    "select_actions",
    "select_device",
    "stack_obs",
    "VDNLearner",
    "VDNMixer",
    "ValueNorm",
]
