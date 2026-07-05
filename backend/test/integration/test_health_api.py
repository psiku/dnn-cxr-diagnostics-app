from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.routes.health import router as health_router


def test_health_check_reports_readiness_when_services_loaded():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["device"] in {"cpu", "cuda"}
    assert isinstance(body["segmentation_enabled"], bool)


def test_health_check_returns_503_when_services_not_ready():
    bare_app = FastAPI()
    bare_app.include_router(health_router)

    with TestClient(bare_app) as client:
        response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["detail"] == "Inference services are not ready"
