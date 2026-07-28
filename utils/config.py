from pathlib import Path
import json
import yaml

from typing import Optional

def dump_config(config_dict, path):
    with open(path, "w") as f:
        json.dump(config_dict, f, indent=2)

def load_config(path: Optional[str], **kwargs) -> dict:
    """
    Load a configuration file.

    Supported formats:
        - .json
        - .yaml
        - .yml

    Args:
        path: Path to the configuration file or None
        kwargs: other key args to overwrite the config

    Returns:
        A dictionary containing the configuration.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is unsupported.
    """
    if path is None:
        return kwargs
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()

    with path.open("r") as f:
        if suffix == ".json":
            config = json.load(f)
        elif suffix in {".yaml", ".yml"}:
            config = yaml.safe_load(f)
        else:
            raise ValueError(
                f"Unsupported config format '{suffix}'. "
                "Supported formats are .json, .yaml, and .yml."
            )
    
    config.update(kwargs)
    return config