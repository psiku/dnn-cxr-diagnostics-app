"""FastAPI application setup."""

from fastapi.middleware.cors import CORSMiddleware
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
import logging

from src.api.exception_handlers import register_exception_handlers
from src.api.routes import annotations, health, predictions, pathologies
from src.core.annotations_config import annotations_data_root
from src.core.logging_config import configure_logging
from src.core.config import load_config, validate_mask_flags, validate_runtime_alignment
from src.core.pathologies import PATHOLOGIES
from src.core.models import (
    load_model,
    load_thresholds,
    ChestXRayPredictor,
    load_segmentation_model,
)
from src.repositories import DescriptionRepository, LabelingRepository
from src.services import PredictionService, SegmentationService, XRayTriageService
from src.services.description_pdf import DescriptionPdfBuilder
from src.services.description_service import DescriptionService
from src.services.labeling_service import LabelingService


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    configure_logging()
    logger.info("Starting Chest X-Ray Diagnostics API")
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "config",
        "model_config.yml",
    )

    config = load_config(config_path)
    validate_mask_flags(config)
    device = "cuda" if os.environ.get("USE_CUDA", "false").lower() == "true" else "cpu"

    model = load_model(config, device=device)
    thresholds = load_thresholds(
        config.thresholds_path,
        pathologies=list(PATHOLOGIES),
    )

    validate_runtime_alignment(
        num_classes=config.model_cfg.num_classes,
        pathologies=PATHOLOGIES,
        thresholds=thresholds,
        thresholds_path=config.thresholds_path,
    )
    if model.num_classes != config.model_cfg.num_classes:
        raise ValueError(
            f"Loaded model has {model.num_classes} output classes, but "
            f"model_cfg.num_classes is {config.model_cfg.num_classes}."
        )

    segmentation_service = None
    if config.segmentation_weights_path:
        seg_artifacts = load_segmentation_model(
            config.segmentation_weights_path, device
        )
        segmentation_service = SegmentationService(seg_artifacts)

    predictor = ChestXRayPredictor(
        model,
        device=device,
        grayscale=config.model_cfg.grayscale,
    )
    prediction_service = PredictionService(
        predictor,
        segmentation_service=segmentation_service,
        use_mask=config.use_mask,
        use_mask_channel=config.model_cfg.use_mask_channel,
    )
    triage_service = XRayTriageService(thresholds=thresholds)

    annotations_root: Path = annotations_data_root()
    annotations_root.mkdir(parents=True, exist_ok=True)
    description_repository = DescriptionRepository(annotations_root)
    labeling_repository = LabelingRepository(annotations_root)
    description_service = DescriptionService(
        description_repository, DescriptionPdfBuilder()
    )
    labeling_service = LabelingService(labeling_repository)

    app.state.model = model
    app.state.predictor = predictor
    app.state.prediction_service = prediction_service
    app.state.triage_service = triage_service
    app.state.segmentation_service = segmentation_service
    app.state.device = device
    app.state.description_service = description_service
    app.state.labeling_service = labeling_service
    app.state.annotations_root = annotations_root

    logger.info(
        "Classifier model loaded on %s (grayscale=%s, use_mask_channel=%s)",
        device,
        config.model_cfg.grayscale,
        config.model_cfg.use_mask_channel,
    )
    if segmentation_service is not None:
        logger.info(
            "Segmentation enabled; weights loaded from %s (use_mask=%s)",
            config.segmentation_weights_path,
            config.use_mask,
        )
    else:
        logger.warning(
            "Segmentation disabled; set use_mask=true and segmentation_weights_path "
            "in config to enable the thoracic crop path"
        )
    if thresholds is not None:
        logger.info("Thresholds loaded from %s", config.thresholds_path)
    else:
        logger.warning(
            "No thresholds_path configured; using default %.2f for all classes",
            XRayTriageService.DEFAULT_THRESHOLD,
        )

    yield

    logger.info("Shutting down Chest X-Ray Diagnostics API")
    # Cleanup
    app.state.model = None
    app.state.predictor = None
    app.state.prediction_service = None
    app.state.triage_service = None
    app.state.segmentation_service = None
    app.state.description_service = None
    app.state.labeling_service = None


app = FastAPI(
    title="Chest X-Ray Diagnostics API",
    description="Medical AI API for X-ray analysis and triage",
    version="1.0.0",
    lifespan=lifespan,
)
register_exception_handlers(app)

_default_cors = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8080,http://127.0.0.1:8080"
_cors_origins = [
    origin.strip()
    for origin in os.environ.get("CORS_ORIGINS", _default_cors).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predictions.router)
app.include_router(annotations.router)
app.include_router(pathologies.router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Chest X-Ray Diagnostics API",
        "docs": "/docs",
        "health": "/health",
    }
