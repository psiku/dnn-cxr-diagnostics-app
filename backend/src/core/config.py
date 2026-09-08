from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel


class ModelConfig(BaseModel):
    num_classes: int
    backbone_name: str
    pretrained: bool
    grayscale: bool
    backbone_trainable_layers: list[str]
    in_features: int | None = None
    transition_dim: int
    use_transition: bool
    pooling: str
    lse_r: float
    dropout: float
    use_mask_channel: bool = False


class Config(BaseModel):
    model_path: str
    model_cfg: ModelConfig
    thresholds_path: str | None = None
    segmentation_weights_path: str | None = None
    use_mask: bool = False


def load_config(config_path: str) -> Config:
    """Load configuration from YAML; resolve relative paths from backend root."""
    config_file = Path(config_path).resolve()
    config_dir = config_file.parent
    backend_root = config_dir.parent if config_dir.name == "config" else config_dir

    with open(config_file, encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    for key in ("model_path", "thresholds_path", "segmentation_weights_path"):
        raw = config_dict.get(key)
        if not raw:
            continue
        path = Path(raw)
        if not path.is_absolute():
            config_dict[key] = str((backend_root / path).resolve())

    return Config(**config_dict)


def validate_mask_flags(config: Config) -> None:
    """
    Ensure deploy-time mask flags are consistent with segmentation and model config.

    Raises:
        ValueError: On incompatible flag combinations.
    """
    use_mask_channel = config.model_cfg.use_mask_channel

    if use_mask_channel and not config.use_mask:
        raise ValueError(
            "model_cfg.use_mask_channel=true requires use_mask=true so that "
            "segmentation can produce the extra input channel."
        )

    if (config.use_mask or use_mask_channel) and not config.segmentation_weights_path:
        raise ValueError(
            "use_mask / use_mask_channel requires segmentation_weights_path "
            "in model_config.yml."
        )


def validate_runtime_alignment(
    *,
    num_classes: int,
    pathologies: tuple[str, ...] | list[str],
    thresholds: np.ndarray | None,
    thresholds_path: str | None = None,
) -> None:
    """
    Ensure model class count, pathology labels, and optional thresholds agree.

    Raises:
        ValueError: On mismatch, with a message pointing at the config sources.
    """
    pathology_count = len(pathologies)
    if num_classes != pathology_count:
        raise ValueError(
            f"model_cfg.num_classes is {num_classes}, but pathologies.json "
            f"defines {pathology_count} classes. Update model_config.yml or "
            "config/pathologies.json so they match."
        )

    if thresholds is None:
        return

    threshold_count = int(np.asarray(thresholds).reshape(-1).shape[0])
    if threshold_count != pathology_count:
        source = thresholds_path or "thresholds file"
        raise ValueError(
            f"{source} provides {threshold_count} thresholds, but "
            f"pathologies.json defines {pathology_count} classes."
        )
