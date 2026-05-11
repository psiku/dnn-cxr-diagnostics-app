import yaml

from src.core.config import load_config


def test_load_config(tmp_path):
    # 1. Setup
    config_file = tmp_path / "test_config.yaml"

    config_data = {
        "model_path": "path/to/model.pth",
        "model_cfg": {
            "num_classes": 5,
            "backbone_name": "resnet50",
            "pretrained": True,
            "grayscale": False,
            "backbone_trainable_layers": [],
            "in_features": 2048,
            "transition_dim": 512,
            "use_transition": True,
            "pooling": "avg",
            "lse_r": 5.0,
            "dropout": 0.5
        },
        "thresholds_path": "path/to/thresholds.json",
    }

    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    # 2. Action
    config = load_config(str(config_file))

    # 3. Assert
    assert config.model_path == "path/to/model.pth"
    assert config.model_cfg.num_classes == 5
    assert config.model_cfg.backbone_name == "resnet50"
    assert config.model_cfg.pretrained is True
    assert config.model_cfg.grayscale is False
    assert config.model_cfg.backbone_trainable_layers == []
    assert config.model_cfg.in_features == 2048
    assert config.model_cfg.transition_dim == 512
    assert config.model_cfg.use_transition is True
    assert config.model_cfg.pooling == "avg"
    assert config.model_cfg.lse_r == 5.0
    assert config.model_cfg.dropout == 0.5
    assert config.thresholds_path == "path/to/thresholds.json"
