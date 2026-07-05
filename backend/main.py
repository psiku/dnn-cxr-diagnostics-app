"""Main entry point for the application."""

import uvicorn

from src.core.logging_config import configure_logging


if __name__ == "__main__":
    configure_logging()
    uvicorn.run(
        "src.api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
