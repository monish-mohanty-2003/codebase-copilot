"""FastAPI application factory."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.api.state import AppState
from app.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.app_state = AppState.build(settings)
    logging.getLogger(__name__).info("Codebase Copilot ready on %s:%d", settings.host, settings.port)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Codebase Copilot",
        description="Local AI assistant grounded in your repository.",
        version="0.1.0",
        lifespan=lifespan,
    )
    # VS Code webviews talk over HTTP — allow localhost origins.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router, prefix="/api")
    return app


app = create_app()


def main() -> None:
    """Entrypoint for `python -m app.api.server`."""
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "app.api.server:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()