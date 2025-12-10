from __future__ import annotations

import math
from typing import Any, Callable, Mapping


def _clamp01(value: float) -> float:
    """Clamp a float into [0, 1]."""
    return max(0.0, min(1.0, value))


def constant_scheduler(value: float) -> Callable[[int], float]:
    """Always return the same mixing weight."""
    const = _clamp01(float(value))

    def schedule(_: int) -> float:
        return const

    return schedule


def linear_scheduler(start_value: float, end_value: float, total_steps: int) -> Callable[[int], float]:
    """Linear interpolation between start_value and end_value over total_steps."""
    start = float(start_value)
    end = float(end_value)
    steps = max(1, int(total_steps))

    def schedule(step: int) -> float:
        progress = max(0, min(int(step), steps))
        fraction = progress / steps
        value = start + (end - start) * fraction
        return _clamp01(value)

    return schedule


def cosine_scheduler(start_value: float, end_value: float, total_steps: int) -> Callable[[int], float]:
    """
    Cosine ramp from start_value to end_value over total_steps.

    Uses the classic half-cosine warmup: w = 0.5 * (1 - cos(pi * t/T)).
    """
    start = float(start_value)
    end = float(end_value)
    steps = max(1, int(total_steps))

    def schedule(step: int) -> float:
        progress = max(0, min(int(step), steps))
        fraction = progress / steps
        weight = 0.5 * (1 - math.cos(math.pi * fraction))
        value = start + (end - start) * weight
        return _clamp01(value)

    return schedule


def build_scheduler(
    config: Mapping[str, Any] | None,
    *,
    default_start: float = 0.0,
    default_end: float = 1.0,
    default_steps: int = 100_000,
) -> Callable[[int], float]:
    """
    Build a scheduler callable from a config mapping.

    Supported types:
    - linear (default)
    - cosine
    - constant
    """
    cfg = config or {}
    schedule_type = str(cfg.get("type", "linear")).lower()
    start_value = float(cfg.get("start_value", default_start))
    end_value = float(cfg.get("end_value", default_end))
    total_steps = int(cfg.get("total_steps", default_steps))

    if schedule_type == "linear":
        return linear_scheduler(start_value, end_value, total_steps)
    if schedule_type == "cosine":
        return cosine_scheduler(start_value, end_value, total_steps)
    if schedule_type == "constant":
        value = float(cfg.get("value", end_value))
        return constant_scheduler(value)

    raise ValueError(
        f"Unknown scheduler type '{schedule_type}'. Supported types: linear, cosine, constant."
    )
