from __future__ import annotations

import pytest

from ncs_env.joint_action_env import decode_joint_action, encode_joint_action


def test_joint_action_roundtrip_binary_three_agents() -> None:
    n_agents = 3
    n_actions = 2
    n_joint_actions = n_actions ** n_agents
    for action_index in range(n_joint_actions):
        decoded = decode_joint_action(action_index, n_agents=n_agents, n_actions=n_actions)
        encoded = encode_joint_action(decoded, n_actions=n_actions)
        assert encoded == action_index


def test_joint_action_roundtrip_nonbinary() -> None:
    n_agents = 2
    n_actions = 3
    n_joint_actions = n_actions ** n_agents
    for action_index in range(n_joint_actions):
        decoded = decode_joint_action(action_index, n_agents=n_agents, n_actions=n_actions)
        encoded = encode_joint_action(decoded, n_actions=n_actions)
        assert encoded == action_index


def test_joint_action_decode_out_of_bounds() -> None:
    with pytest.raises(ValueError):
        decode_joint_action(-1, n_agents=3, n_actions=2)
    with pytest.raises(ValueError):
        decode_joint_action(8, n_agents=3, n_actions=2)


def test_joint_action_encode_out_of_bounds() -> None:
    with pytest.raises(ValueError):
        encode_joint_action([0, 2], n_actions=2)
