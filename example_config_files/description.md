# Model Configurations Directory

Each folder in this directory contains two primary configuration files required for model execution and inference:

- **`model_config.yml`**: Defines model architecture details, preprocessing parameters, and pipeline configurations.
- **`thresholds.json`**: Contains disease-specific classification thresholds used for diagnostic decision-making.

---

## Directory Breakdown

| Directory Path | Model Architecture | Description / Input Type |
| :--- | :--- | :--- |
| `full_images/` | **EfficientNet-B2** | Trained to process complete, full-resolution images. |
| `cropped_images/` | **EfficientNet-B1** | Trained to process focused, cropped image inputs. |
| `cropped_images_with_mask/` | **DenseNet161** | Accepts cropped images alongside segmentation masks as multi-channel input. |
| `resnet_cropped/` | **ResNet101** | Alternative model architecture trained on cropped image inputs. |