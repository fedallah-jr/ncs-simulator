from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "default_config.json"


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load simulation configuration from a JSON file.

    Args:
        config_path: Optional path to a JSON config. When omitted, the package
            default located next to this module is used.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_measurement_noise_scale_range(
    system_cfg: Dict[str, Any],
) -> Optional[Tuple[float, float]]:
    """
    Resolve the measurement noise scale range from config.

    Returns:
        (min_scale, max_scale) if provided, otherwise None.
    """
    range_cfg = system_cfg.get("measurement_noise_scale_range", None)
    if range_cfg is None:
        return None
    if isinstance(range_cfg, (int, float)):
        min_scale = 0.0
        max_scale = float(range_cfg)
    elif isinstance(range_cfg, (list, tuple)) and len(range_cfg) == 2:
        min_scale = float(range_cfg[0])
        max_scale = float(range_cfg[1])
    else:
        raise ValueError("measurement_noise_scale_range must be a scalar or [min, max] list")
    if min_scale < 0.0 or max_scale < 0.0:
        raise ValueError("measurement_noise_scale_range values must be >= 0")
    if max_scale < min_scale:
        raise ValueError("measurement_noise_scale_range max must be >= min")
    return min_scale, max_scale
