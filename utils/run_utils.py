"""
Helpers for organizing training run outputs.
"""

from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence


def _create_unique_run_dir(base: Path, algorithm: str) -> Path:
    """Create a unique run directory with sequential numbering: {algorithm}_{number}"""
    base.mkdir(parents=True, exist_ok=True)
    index = 0
    while True:
        candidate = base / f"{algorithm}_{index}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        index += 1


def prepare_run_directory(algorithm: str, config_path: Optional[Path], output_root: Path) -> Path:
    """
    Create a unique run directory with simple sequential naming.

    Returns:
        Path to the created run directory (e.g., outputs/iql_0, outputs/iql_1, ...)
    """
    return _create_unique_run_dir(output_root, algorithm)


def save_config_with_hyperparameters(
    run_dir: Path,
    config_path: Optional[Path],
    algorithm: str,
    hyperparams: Mapping[str, Any],
    *,
    resolved_config: Optional[Mapping[str, Any]] = None,
    set_overrides: Optional[Sequence[str]] = None,
) -> None:
    """
    Save the full configuration JSON with an added 'training_run' section.

    This preserves the complete config in structured format and appends
    hyperparameters and metadata for this specific training run.

    Args:
        run_dir: Directory where the config will be saved
        config_path: Path to the original config file (or None for default)
        algorithm: Algorithm name (e.g., 'iql', 'mappo')
        hyperparams: Dictionary of hyperparameters for this run
        resolved_config: Effective config used by training (for example, after
            applying CLI ``--set`` overrides). If omitted, the config is loaded
            from ``config_path``.
        set_overrides: Raw ``--set key=value`` entries applied for this run.
            Stored in ``training_run`` metadata for traceability.
    """
    from ncs_env.config import load_config, DEFAULT_CONFIG_PATH
    import json

    # Load the effective config.
    resolved_config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if resolved_config is not None:
        config = copy.deepcopy(dict(resolved_config))
    else:
        config = load_config(str(resolved_config_path))

        # Backward-compatible fallback: apply --set overrides if provided and
        # caller did not pass a pre-resolved config.
        if set_overrides:
            from tools._common import deep_merge, parse_set_overrides

            overrides = parse_set_overrides(list(set_overrides))
            if overrides:
                deep_merge(config, overrides)

    output_path = run_dir / "config.json"

    # Preserve prior run metadata when resuming into the same run directory.
    training_run_history: list[Dict[str, Any]] = []
    if output_path.exists():
        try:
            with output_path.open("r", encoding="utf-8") as f:
                existing_config = json.load(f)
            existing_history = existing_config.get("training_run_history", [])
            if isinstance(existing_history, list):
                training_run_history.extend(
                    entry for entry in existing_history if isinstance(entry, dict)
                )
            existing_entry = existing_config.get("training_run")
            if isinstance(existing_entry, dict):
                training_run_history.append(existing_entry)
        except Exception:
            # If prior metadata cannot be read, continue and write fresh metadata.
            training_run_history = []

    # Add training_run section with hyperparameters and metadata.
    config["training_run"] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "algorithm": algorithm,
        "source_config_path": str(resolved_config_path),
        "hyperparameters": dict(hyperparams),
        "set_overrides": list(set_overrides) if set_overrides is not None else [],
    }
    if training_run_history:
        config["training_run_history"] = training_run_history

    # Save to run directory.
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")  # Add trailing newline


class BestModelTracker:
    """Track best rewards and save checkpoints when improved."""

    def __init__(self) -> None:
        self._best: Dict[str, float] = {}

    def update(
        self,
        key: str,
        reward: float,
        save_path: Path,
        save_fn: Callable[[Path], None],
    ) -> bool:
        """
        Check if reward is best for the given key, save checkpoint if so.

        Args:
            key: Identifier for this reward type (e.g., "eval", "train")
            reward: Current reward value
            save_path: Path to save the checkpoint
            save_fn: Function that saves the checkpoint to the given path

        Returns:
            True if new best was found and saved, False otherwise.
        """
        current_best = self._best.get(key, -float("inf"))
        if reward > current_best:
            self._best[key] = reward
            save_fn(save_path)
            return True
        return False

    def get_best(self, key: str) -> float:
        """Get the current best reward for a key, or -inf if not tracked."""
        return self._best.get(key, -float("inf"))
