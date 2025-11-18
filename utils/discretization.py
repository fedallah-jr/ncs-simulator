"""
Tools for mapping continuous observations to discrete representations.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def _ensure_positive(value: int, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


class ObservationDiscretizer:
    """
    Map continuous observations to a discrete key for tabular Q-learning.

    Observation layout (flat vector):
        [current_state..., current_throughput,
         prev_states..., prev_statuses..., prev_throughputs...]
    Each block is binned independently; status entries are already discrete.
    """

    def __init__(
        self,
        history_window: int,
        state_dim: int,
        *,
        history_levels: int = 3,
        throughput_bins: int = 5,
        throughput_range: Tuple[float, float] = (0.0, 50.0),
        state_bins: int = 7,
        state_limits: Sequence[float] | float = 5.0,
        state_history_window: int | None = None,
        throughput_history_window: int | None = None,
    ):
        self.history_window = _ensure_positive(history_window, "history_window")
        self.state_dim = _ensure_positive(state_dim, "state_dim")
        self.state_history_window = _ensure_positive(
            state_history_window if state_history_window is not None else history_window,
            "state_history_window",
        )
        self.throughput_history_window = _ensure_positive(
            throughput_history_window if throughput_history_window is not None else history_window,
            "throughput_history_window",
        )
        self.history_levels = _ensure_positive(history_levels, "history_levels")
        self.throughput_bins = _ensure_positive(throughput_bins, "throughput_bins")
        self.state_bins = _ensure_positive(state_bins, "state_bins")

        if throughput_range[0] >= throughput_range[1]:
            raise ValueError("throughput_range must be an increasing interval")
        self.throughput_clip = throughput_range
        if self.throughput_bins == 1:
            self.throughput_edges = np.array([], dtype=np.float32)
        else:
            self.throughput_edges = np.linspace(
                throughput_range[0], throughput_range[1], throughput_bins - 1, dtype=np.float32
            )

        if isinstance(state_limits, Sequence):
            if len(state_limits) != state_dim:
                raise ValueError("state_limits length must match state_dim")
            self.state_limits = np.asarray(state_limits, dtype=np.float32)
        else:
            self.state_limits = np.full(state_dim, float(state_limits), dtype=np.float32)
        if np.any(self.state_limits <= 0):
            raise ValueError("state_limits must be positive")

        if self.state_bins == 1:
            self.state_edges = np.zeros((state_dim, 0), dtype=np.float32)
        else:
            self.state_edges = np.stack(
                [
                    np.linspace(-limit, limit, self.state_bins - 1, dtype=np.float32)
                    for limit in self.state_limits
                ]
            )

    def discretize(self, observation: np.ndarray) -> Tuple[int, ...]:
        """Return a tuple key that indexes the Q-table."""
        expected_len = (
            self.state_dim
            + 1
            + self.state_history_window * self.state_dim
            + self.history_window
            + self.throughput_history_window
        )
        if observation.shape[0] != expected_len:
            raise ValueError("Unexpected observation dimension")

        idx = 0
        current_state = observation[idx : idx + self.state_dim]
        idx += self.state_dim

        current_throughput = float(observation[idx])
        idx += 1

        prev_states_flat = observation[idx : idx + self.state_history_window * self.state_dim]
        idx += self.state_history_window * self.state_dim

        prev_statuses = observation[idx : idx + self.history_window]
        idx += self.history_window

        prev_throughputs = observation[idx : idx + self.throughput_history_window]

        status_codes = tuple(
            int(np.clip(round(value), 0, self.history_levels - 1)) for value in prev_statuses
        )

        def bin_throughput(value: float) -> int:
            clipped = float(np.clip(value, *self.throughput_clip))
            if self.throughput_bins == 1:
                return 0
            return int(np.digitize(clipped, self.throughput_edges))

        throughput_code = bin_throughput(current_throughput)
        throughput_history_codes = tuple(bin_throughput(float(v)) for v in prev_throughputs)

        def bin_state_value(dim: int, value: float) -> int:
            clipped = float(np.clip(value, -self.state_limits[dim], self.state_limits[dim]))
            if self.state_bins == 1:
                return 0
            return int(np.digitize(clipped, self.state_edges[dim]))

        current_state_codes = tuple(
            bin_state_value(dim, float(value)) for dim, value in enumerate(current_state)
        )

        prev_state_codes = []
        for j, value in enumerate(prev_states_flat):
            dim = j % self.state_dim
            prev_state_codes.append(bin_state_value(dim, float(value)))

        return (
            current_state_codes
            + (throughput_code,)
            + tuple(prev_state_codes)
            + status_codes
            + throughput_history_codes
        )
