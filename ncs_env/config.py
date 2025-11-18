import json
from pathlib import Path
from typing import Any, Dict, Optional


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
