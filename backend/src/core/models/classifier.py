import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import(
    resnet18,
    resnet50,
    resnet101,
    resnet152,
    densenet121,
    densenet161,
    densenet169,
    densenet201,
    efficientnet_b0,
    efficientnet_b1,
    efficientnet_b2,
    efficientnet_b3,
    efficientnet_b4,
    efficientnet_b5,
    efficientnet_b6,
    efficientnet_b7,
    ResNet18_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    DenseNet121_Weights,
    DenseNet161_Weights,
    DenseNet169_Weights,
    DenseNet201_Weights,
    EfficientNet_B0_Weights,
    EfficientNet_B1_Weights,
    EfficientNet_B2_Weights,
    EfficientNet_B3_Weights,
    EfficientNet_B4_Weights,
    EfficientNet_B5_Weights,
    EfficientNet_B6_Weights,
    EfficientNet_B7_Weights,
)


def get_nested_attr(obj, path):
    for p in path:
        obj = obj[p] if isinstance(p, int) else getattr(obj, p)
    return obj


def set_nested_attr(obj, path, value):
    parent = get_nested_attr(obj, path[:-1])
    last = path[-1]
    if isinstance(last, int):
        parent[last] = value
    else:
        setattr(parent, last, value)


class TorchvisionBackbone(nn.Module):
    CONFIGS = {
        "resnet18":  (resnet18,  ResNet18_Weights.DEFAULT,  ["conv1"], None),
        "resnet50":  (resnet50,  ResNet50_Weights.DEFAULT,  ["conv1"], None),
        "resnet101": (resnet101, ResNet101_Weights.DEFAULT, ["conv1"], None),
        "resnet152": (resnet152, ResNet152_Weights.DEFAULT, ["conv1"], None),

        "densenet121": (densenet121, DenseNet121_Weights.DEFAULT, ["features", "conv0"], ["features"]),
        "densenet161": (densenet161, DenseNet161_Weights.DEFAULT, ["features", "conv0"], ["features"]),
        "densenet169": (densenet169, DenseNet169_Weights.DEFAULT, ["features", "conv0"], ["features"]),
        "densenet201": (densenet201, DenseNet201_Weights.DEFAULT, ["features", "conv0"], ["features"]),

        "efficientnet_b0": (efficientnet_b0, EfficientNet_B0_Weights.DEFAULT, ["features", 0, 0], ["features"]),
        "efficientnet_b1": (efficientnet_b1, EfficientNet_B1_Weights.DEFAULT, ["features", 0, 0], ["features"]),
        "efficientnet_b2": (efficientnet_b2, EfficientNet_B2_Weights.DEFAULT, ["features", 0, 0], ["features"]),
        "efficientnet_b3": (efficientnet_b3, EfficientNet_B3_Weights.DEFAULT, ["features", 0, 0], ["features"]),
        "efficientnet_b4": (efficientnet_b4, EfficientNet_B4_Weights.DEFAULT, ["features", 0, 0], ["features"]),
        "efficientnet_b5": (efficientnet_b5, EfficientNet_B5_Weights.DEFAULT, ["features", 0, 0], ["features"]),
        "efficientnet_b6": (efficientnet_b6, EfficientNet_B6_Weights.DEFAULT, ["features", 0, 0], ["features"]),
        "efficientnet_b7": (efficientnet_b7, EfficientNet_B7_Weights.DEFAULT, ["features", 0, 0], ["features"]),
    }

    def __init__(self, name: str, pretrained: bool = True, grayscale: bool = True):
        super().__init__()

        if name not in self.CONFIGS:
            raise ValueError(f"Unsupported backbone: {name}. Choose from {list(self.CONFIGS)}")

        builder, default_weights, first_conv_path, features_path = self.CONFIGS[name]

        weights = default_weights if pretrained else None
        model = builder(weights=weights)

        if grayscale:
            self._convert_first_conv_to_grayscale(model, first_conv_path)

        if name.startswith("resnet"):
            self.features = nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4,
            )
        else:
            self.features = get_nested_attr(model, features_path)

    def _convert_first_conv_to_grayscale(self, model, conv_path):
        old_conv = get_nested_attr(model, conv_path)

        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        set_nested_attr(model, conv_path, new_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    def set_trainable_layers(self, trainable_layers: list[str] | None = None):
        """
        trainable_layers: list of substrings of parameter names to unfreeze.
        Everything else is frozen.
        """

        if not trainable_layers:
            # freeze everything
            for p in self.parameters():
                p.requires_grad = False
            self.eval()
            return

        for name, param in self.named_parameters():
            trainable = any(layer in name for layer in trainable_layers)
            param.requires_grad = trainable

        # set eval/train modes correctly
        for name, module in self.named_modules():
            trainable = any(layer in name for layer in trainable_layers)
            if not trainable:
                module.eval()
            else:
                module.train()

class LSEPool2d(nn.Module):
    """
    Log-Sum-Exp pooling.
    Returns tensor [B, C].
    """

    def __init__(self, r: float = 10.0, eps: float = 1e-6):
        super().__init__()
        self.r = r
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x_max = x.amax(dim=(2, 3), keepdim=True)
        pooled = x_max + (1.0 / self.r) * torch.log(
            torch.mean(torch.exp(self.r * (x - x_max)), dim=(2, 3), keepdim=True) + self.eps
        )
        return pooled.flatten(1)  # [B, C]


class ChestXRayClassifier(nn.Module):
    """
    Chest X-Ray multi-label classifier with configurable backbone and pooling.
    """

    def __init__(
        self,
        num_classes: int = 14,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        grayscale: bool = True,
        backbone_trainable_layers: list[str] | None = None,
        in_features: int | None = None,
        transition_dim: int = 2048,
        use_transition: bool = True,
        pooling: str = "lse",
        lse_r: float = 10.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        if backbone_trainable_layers is None:
            backbone_trainable_layers = []

        self.num_classes = num_classes
        self.pooling = pooling
        self.use_transition = use_transition

        self.backbone = TorchvisionBackbone(
            name=backbone_name,
            pretrained=pretrained,
            grayscale=grayscale,
        )

        self.backbone.set_trainable_layers(backbone_trainable_layers)

        if in_features is None:
            in_features = self._infer_backbone_channels(grayscale)

        if use_transition:
            self.transition = nn.Sequential(
                nn.Conv2d(in_features, transition_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(transition_dim),
                nn.ReLU(inplace=True),
            )
            classifier_dim = transition_dim
        else:
            self.transition = nn.Identity()
            classifier_dim = in_features

        if pooling == "lse":
            self.global_pool = LSEPool2d(r=lse_r)
        elif pooling == "avg":
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        elif pooling == "max":
            self.global_pool = nn.AdaptiveMaxPool2d(1)
        else:
            raise ValueError(f"Unsupported pooling: {pooling}")

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.prediction = nn.Linear(classifier_dim, num_classes)

    def _infer_backbone_channels(self, grayscale: bool) -> int:
        device = next(self.backbone.parameters()).device
        channels = 1 if grayscale else 3

        with torch.no_grad():
            dummy = torch.zeros(1, channels, 224, 224, device=device)
            out = self.backbone(dummy)

        return out.shape[1]

    def _pool_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.global_pool(x)

        if x.ndim == 4:
            x = torch.flatten(x, 1)

        return x

    @staticmethod
    def _normalize_map(cam: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + eps)
        return cam

    def forward(self, image: torch.Tensor, retain_transition_grad: bool = False):
        conv_maps = self.backbone(image)
        transition_maps = self.transition(conv_maps)

        if retain_transition_grad:
            transition_maps.retain_grad()

        pooled_features = self._pool_features(transition_maps)
        pooled_features = self.dropout(pooled_features)

        logits = self.prediction(pooled_features)

        return {
            "logits": logits,
            "transition_maps": transition_maps,
            "pooled_features": pooled_features,
        }

    @torch.no_grad()
    def cam(self, image: torch.Tensor, class_idx: int):
        self.eval()

        out = self.forward(image, retain_transition_grad=False)

        transition_maps = out["transition_maps"]
        class_weights = self.prediction.weight[class_idx]

        cam = torch.einsum("d,bdhw->bhw", class_weights, transition_maps)
        cam = F.relu(cam)
        cam = self._normalize_map(cam)

        return cam, out
