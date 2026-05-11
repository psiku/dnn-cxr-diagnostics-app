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


class Config(BaseModel):
    model_path: str
    model_cfg: ModelConfig
    thresholds_path: str | None = None


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)
