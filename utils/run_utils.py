"""
Helpers for organizing training run outputs and metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np

from ncs_env.config import DEFAULT_CONFIG_PATH, load_config


@dataclass
class ConfigMetadata:
    algorithm: str
    config_path: Path
    process_noise_first: float
    measurement_noise_first: float
    initial_state_scale_min: Iterable[float]
    initial_state_scale_max: Iterable[float]
    q_first: float
    r_first: float
    reward_type: str


def _format_float(value: float) -> str:
    """Format floats consistently for filesystem-friendly strings."""
    return f"{value:.6g}"


def _ensure_array(value: Any, default: np.ndarray) -> np.ndarray:
    if value is None:
        return default
    arr = np.array(value, dtype=float)
    if arr.size == 0:
        return default
    return arr


def _resolve_initial_scale_range(system_cfg: Mapping[str, Any], state_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resolve initial state scale range from config with backward compatibility.

    Prefers new `initial_state_scale_min`/`initial_state_scale_max` fields; falls back
    to legacy `initial_state_scale` as a symmetric bound. Defaults to [0.9, 1.0].
    """

    def _to_array(cfg_value: Any, default: float) -> np.ndarray:
        arr = np.array(cfg_value if cfg_value is not None else default, dtype=float).flatten()
        if arr.size == 0:
            arr = np.array(default, dtype=float).flatten()
        if arr.size == 1:
            return np.full(state_dim, float(arr.item()))
        if arr.size != state_dim:
            raise ValueError("initial_state_scale entries must be scalar or match the state dimension")
        return arr

    legacy = system_cfg.get("initial_state_scale", None)
    min_cfg = system_cfg.get("initial_state_scale_min", None)
    max_cfg = system_cfg.get("initial_state_scale_max", None)

    if legacy is not None:
        legacy_arr = np.abs(_to_array(legacy, 1.0))
        return legacy_arr, legacy_arr

    default_min, default_max = 0.9, 1.0
    scale_min = np.abs(_to_array(min_cfg, default_min))
    scale_max = np.abs(_to_array(max_cfg, default_max))
    if np.any(scale_max < scale_min):
        raise ValueError("initial_state_scale_max must be >= initial_state_scale_min for all dimensions")
    return scale_min, scale_max


def _build_run_name(metadata: ConfigMetadata) -> str:
    init_min = "-".join(_format_float(float(x)) for x in metadata.initial_state_scale_min)
    init_max = "-".join(_format_float(float(x)) for x in metadata.initial_state_scale_max)
    return (
        f"{metadata.algorithm}"
        f"_process{_format_float(metadata.process_noise_first)}"
        f"_measurement{_format_float(metadata.measurement_noise_first)}"
        f"_initial{init_min}to{init_max}"
        f"_Q{_format_float(metadata.q_first)}"
        f"-R{_format_float(metadata.r_first)}"
        f"-{metadata.reward_type}"
    )


def _create_unique_run_dir(base: Path, base_name: str) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    index = 0
    while True:
        candidate = base / f"{base_name}_run{index}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        index += 1


def gather_config_metadata(algorithm: str, config_path: Optional[Path]) -> ConfigMetadata:
    resolved_config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    config = load_config(str(resolved_config_path))
    system_cfg: Dict[str, Any] = config.get("system", {})
    lqr_cfg: Dict[str, Any] = config.get("lqr", {})
    reward_cfg: Dict[str, Any] = config.get("reward", {})

    A = np.array(system_cfg.get("A"), dtype=float)
    B = np.array(system_cfg.get("B"), dtype=float)
    if A.size == 0 or B.size == 0:
        raise ValueError("Config must define system matrices 'A' and 'B'.")
    state_dim = A.shape[0]
    control_dim = B.shape[1]

    process_cov = _ensure_array(system_cfg.get("process_noise_cov"), np.eye(state_dim))
    measurement_cov = _ensure_array(
        system_cfg.get("measurement_noise_cov"), 0.01 * np.eye(state_dim)
    )
    initial_scale_min, initial_scale_max = _resolve_initial_scale_range(system_cfg, state_dim)
    q_matrix = _ensure_array(lqr_cfg.get("Q"), np.eye(state_dim))
    r_matrix = _ensure_array(lqr_cfg.get("R"), np.eye(control_dim))
    reward_type = str(reward_cfg.get("state_error_reward", "difference"))
    reward_mixing_cfg = reward_cfg.get("reward_mixing", {})
    if isinstance(reward_mixing_cfg, bool):
        reward_mixing_cfg = {"enabled": reward_mixing_cfg}
    if reward_mixing_cfg.get("enabled"):
        rewards_list = reward_mixing_cfg.get("rewards", [])
        if isinstance(rewards_list, list) and len(rewards_list) == 2:
            reward_modes = [
                str(entry.get("state_error_reward", reward_type)) for entry in rewards_list
            ]
            reward_type = f"mix-{reward_modes[0]}-{reward_modes[1]}"
        else:
            reward_type = f"mix-{reward_type}"

    return ConfigMetadata(
        algorithm=algorithm,
        config_path=resolved_config_path,
        process_noise_first=float(process_cov.flat[0]),
        measurement_noise_first=float(measurement_cov.flat[0]),
        initial_state_scale_min=tuple(float(x) for x in initial_scale_min),
        initial_state_scale_max=tuple(float(x) for x in initial_scale_max),
        q_first=float(q_matrix.flat[0]),
        r_first=float(r_matrix.flat[0]),
        reward_type=reward_type,
    )


def prepare_run_directory(
    algorithm: str, config_path: Optional[Path], output_root: Path
) -> Tuple[Path, ConfigMetadata]:
    metadata = gather_config_metadata(algorithm, config_path)
    base_name = _build_run_name(metadata)
    run_dir = _create_unique_run_dir(output_root, base_name)
    return run_dir, metadata


def save_config_with_hyperparameters(
    run_dir: Path,
    config_path: Optional[Path],
    algorithm: str,
    hyperparams: Mapping[str, Any]
) -> None:
    """
    Save the full configuration JSON with an added 'training_run' section.

    This preserves the complete config in structured format and appends
    hyperparameters and metadata for this specific training run.

    Args:
        run_dir: Directory where the config will be saved
        config_path: Path to the original config file (or None for default)
        algorithm: Algorithm name (e.g., 'dqn', 'ppo')
        hyperparams: Dictionary of hyperparameters for this run
    """
    from ncs_env.config import load_config, DEFAULT_CONFIG_PATH
    import json

    # Load the original config
    resolved_config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    config = load_config(str(resolved_config_path))

    # Add training_run section with hyperparameters and metadata
    config["training_run"] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "algorithm": algorithm,
        "source_config_path": str(resolved_config_path),
        "hyperparameters": dict(hyperparams)
    }

    # Save to run directory
    output_path = run_dir / "config.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")  # Add trailing newline
