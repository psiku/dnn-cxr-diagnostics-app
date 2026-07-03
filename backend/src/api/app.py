"""FastAPI application setup."""
from fastapi.middleware.cors import CORSMiddleware
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from src.api.routes import annotations, health, predictions, pathologies
from src.core.annotations_config import annotations_data_root
from src.core.config import load_config
from src.core.pathologies import PATHOLOGIES
from src.core.models import load_model, load_thresholds, ChestXRayPredictor
from src.repositories import DescriptionRepository, LabelingRepository
from src.services import XRayTriageService
from src.services.description_pdf import DescriptionPdfBuilder
from src.services.description_service import DescriptionService
from src.services.labeling_service import LabelingService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    # Startup
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "config",
        "model_config.yml"
    )

    config = load_config(config_path)
    device = "cuda" if os.environ.get("USE_CUDA", "false").lower() == "true" else "cpu"

    model = load_model(config, device=device)
    thresholds = load_thresholds(
        config.thresholds_path,
        pathologies=list(PATHOLOGIES),
    )

    predictor = ChestXRayPredictor(model, device=device)
    triage_service = XRayTriageService(predictor, thresholds=thresholds)

    annotations_root: Path = annotations_data_root()
    annotations_root.mkdir(parents=True, exist_ok=True)
    description_repository = DescriptionRepository(annotations_root)
    labeling_repository = LabelingRepository(annotations_root)
    description_service = DescriptionService(description_repository, DescriptionPdfBuilder())
    labeling_service = LabelingService(labeling_repository)

    app.state.model = model
    app.state.predictor = predictor
    app.state.triage_service = triage_service
    app.state.device = device
    app.state.description_service = description_service
    app.state.labeling_service = labeling_service
    app.state.annotations_root = annotations_root

    print(f"Model loaded on {device}")
    if thresholds is not None:
        print(f"Thresholds loaded from {config.thresholds_path}: {thresholds.tolist()}")
    else:
        print(
            "No thresholds_path configured; "
            f"using default {XRayTriageService.DEFAULT_THRESHOLD} for all classes"
        )

    yield

    # Cleanup
    app.state.model = None
    app.state.predictor = None
    app.state.triage_service = None
    app.state.description_service = None
    app.state.labeling_service = None


app = FastAPI(
    title="Chest X-Ray Diagnostics API",
    description="Medical AI API for X-ray analysis and triage",
    version="1.0.0",
    lifespan=lifespan,
)

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
        "health": "/health"
    }
