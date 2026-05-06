from app.api import health_check
from fastapi import FastAPI

app = FastAPI(
    title="Aplikacja Medyczna API",
    version="1.0.0",
)

app.include_router(health_check.router)


@app.get("/")
def read_root():
    return {"message": "Witaj w głównym API"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)