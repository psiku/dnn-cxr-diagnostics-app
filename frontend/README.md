# Frontend — Chest X-ray Diagnostics

Interfejs użytkownika aplikacji do analizy RTG klatki piersiowej (React + Vite).

Pełne uruchomienie (Docker, backend, dane): [`../README.md`](../README.md).

## Stack

- React 19, React Router
- Vite 8
- Axios → FastAPI backend

## Uruchomienie (dev)

Backend musi działać na `http://127.0.0.1:8000` (patrz `../backend/`).

```bash
cd frontend
npm install
npm run dev
```

Aplikacja: http://localhost:5173

## Skrypty

| Polecenie | Opis |
|-----------|------|
| `npm run dev` | Serwer deweloperski z HMR |
| `npm run build` | Build produkcyjny → `dist/` |
| `npm run preview` | Podgląd buildu lokalnie |
| `npm run lint` | ESLint |


## Struktura `src/`

```
src/
├── pages/           # ekrany aplikacji
├── context/         # WorkspaceProvider — wspólne badanie
├── hooks/           # useWorkspace, usePathologies (GET /pathologies)
├── utils/           # np. konwersje obrazów
└── config.js        # adres API
```