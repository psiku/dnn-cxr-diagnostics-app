from fastapi.testclient import TestClient

from src.api.app import app
from src.core.pathologies import PATHOLOGIES


client = TestClient(app)


def test_list_pathologies():
    response = client.get("/pathologies")
    assert response.status_code == 200
    assert response.json() == {"pathologies": list(PATHOLOGIES)}
