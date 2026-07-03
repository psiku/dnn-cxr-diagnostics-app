# dnn-cxr-diagnostics-app

Aplikacja (frontend + backend) do analizy RTG klatki piersiowej — praca inżynierska.

## Wymagania

- [Docker](https://docs.docker.com/get-docker/) i Docker Compose v2
- Wagi modelu w `backend/models/` (np. `models/app/last.ckpt`)
- Progi w `backend/config/best_thresholds.json`

## Uruchomienie w Dockerze (zalecane)

Z katalogu głównego repozytorium:

```bash
docker compose up --build
```

Pierwszy build może trwać kilkanaście minut (PyTorch CPU).

| Usługa   | Adres |
|----------|--------|
| Aplikacja (UI) | http://localhost:8080 |
| API (przez proxy) | http://localhost:8080/api/… |
| Swagger | http://localhost:8080/docs |

Na innym urządzeniu w tej samej sieci użyj IP hosta, np. `http://192.168.1.10:8080`.

Zatrzymanie:

```bash
docker compose down
```

Dane adnotacji (opisy, labelki) zapisują się w:

`backend\data\annotations\` — normalny folder w projekcie (widoczny w Eksploratorze plików).

## Uruchomienie lokalne (bez Dockera)

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python main.py
```

API: http://127.0.0.1:8000

### Frontend

```bash
cd frontend
npm install
npm run dev
```

UI: http://localhost:5173 (woła API pod `http://127.0.0.1:8000`)

## Struktura

- `backend/` — FastAPI, model ML, adnotacje
- `frontend/` — React (Vite)
- `docker-compose.yml` — backend + frontend (nginx)
