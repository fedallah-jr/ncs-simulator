"""
Multi-agent reinforcement learning (MARL) utilities.

This package provides shared components used by executable training scripts under
`algorithms/` (e.g., IQL/VDN/QMIX).
"""

from __future__ import annotations

from .buffer import MARLReplayBuffer, MARLBatch
from .learners import IQLLearner, VDNLearner, QMIXLearner
from .networks import MLPAgent, QMixer, VDNMixer, append_agent_id

__all__ = [
    "MARLBatch",
    "MARLReplayBuffer",
    "append_agent_id",
    "IQLLearner",
    "MLPAgent",
    "QMIXLearner",
    "QMixer",
    "VDNLearner",
    "VDNMixer",
]

