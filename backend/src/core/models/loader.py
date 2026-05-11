"""Model and threshold loading utilities."""
import json
import os
from collections.abc import Iterable, Mapping

import numpy as np
import torch
import yaml

from src.core.models.classifier import ChestXRayClassifier
from src.core.config import Config


def load_model(config: Config, device: str = "cpu") -> ChestXRayClassifier:
    """
    Load a ChestXRayClassifier model from a checkpoint.

    Args:
        config: Configuration object with model_path and model_cfg
        device: Device to load model to ("cpu" or "cuda")

    Returns:
        Model in eval mode. Any ``thresholds`` entry bundled in the checkpoint
        is discarded; per-class thresholds are sourced exclusively from
        ``config.thresholds_path`` via :func:`load_thresholds`.
    """
    model = ChestXRayClassifier(**config.model_cfg.model_dump())

    checkpoint = torch.load(
        config.model_path,
        map_location=torch.device(device),
        weights_only=False,
    )

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        # Remove "model." prefix if present (Lightning convention)
        if any(key.startswith("model.") for key in state_dict.keys()):
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        state_dict.pop("thresholds", None)
        model.load_state_dict(state_dict)
    else:
        checkpoint.pop("thresholds", None)
        model.load_state_dict(checkpoint)

    model.eval()
    model.to(device)
    return model


def load_thresholds(
    path: str | None,
    pathologies: list[str] | None = None,
) -> np.ndarray | None:
    """
    Load per-class probability thresholds from a file.

    Supported formats are picked from the file extension:
        * ``.json`` - list of floats or mapping ``{pathology_name: threshold}``
        * ``.yaml`` / ``.yml`` - same shapes as JSON
        * ``.npy`` - numpy array saved with :func:`numpy.save`
        * ``.pt`` / ``.pth`` - torch tensor or dict serialised with ``torch.save``

    Args:
        path: Path to the thresholds file. ``None`` or empty returns ``None``.
        pathologies: Ordered list of class names. When the file contains a
            mapping, values are reordered to match this list. Ignored for
            list/array files.

    Returns:
        1D float32 numpy array, or ``None`` if ``path`` is empty.
    """
    if not path:
        return None

    ext = os.path.splitext(path)[1].lower()

    if ext == ".json":
        with open(path, "r") as f:
            data = json.load(f)
    elif ext in (".yaml", ".yml"):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    elif ext == ".npy":
        return np.asarray(np.load(path), dtype=np.float32).reshape(-1)
    elif ext in (".pt", ".pth"):
        data = torch.load(path, map_location="cpu", weights_only=False)
    else:
        raise ValueError(f"Unsupported thresholds file format: {ext!r}")

    return _coerce_thresholds(data, pathologies=pathologies, source=path)


def _coerce_thresholds(
    data,
    pathologies: list[str] | None,
    source: str,
) -> np.ndarray:
    if isinstance(data, torch.Tensor):
        return _to_numpy(data)

    if isinstance(data, Mapping):
        if pathologies is None:
            return np.asarray(list(data.values()), dtype=np.float32).reshape(-1)
        try:
            return np.asarray([data[name] for name in pathologies], dtype=np.float32)
        except KeyError as exc:
            raise KeyError(
                f"Threshold for pathology {exc.args[0]!r} missing in {source}"
            ) from exc

    if isinstance(data, Iterable) and not isinstance(data, (str, bytes)):
        return np.asarray(list(data), dtype=np.float32).reshape(-1)

    raise TypeError(
        f"Unsupported thresholds payload in {source}: {type(data).__name__}"
    )


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float32).reshape(-1)
    return np.asarray(value, dtype=np.float32).reshape(-1)
