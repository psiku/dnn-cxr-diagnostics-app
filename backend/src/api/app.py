"""FastAPI application setup."""
from fastapi.middleware.cors import CORSMiddleware
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.api.routes import health, predictions
from src.core.config import load_config
from src.core.models import load_model, load_thresholds, ChestXRayPredictor
from src.services import XRayTriageService


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
        pathologies=XRayTriageService.PATHOLOGIES,
    )

    predictor = ChestXRayPredictor(model, device=device)
    triage_service = XRayTriageService(predictor, thresholds=thresholds)

    app.state.model = model
    app.state.predictor = predictor
    app.state.triage_service = triage_service
    app.state.device = device

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


app = FastAPI(
    title="Chest X-Ray Diagnostics API",
    description="Medical AI API for X-ray analysis and triage",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predictions.router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Chest X-Ray Diagnostics API",
        "docs": "/docs",
        "health": "/health"
    }
