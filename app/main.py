"""HTTP app: FastAPI over the existing caption pipeline. Dashboard is optional."""

from __future__ import annotations

import argparse

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import captions, jobs, settings, upload
from app.config import ROOT
from app.pipeline.profanity import DISCLAIMER

DASHBOARD = ROOT / "dashboard"


def create_app() -> FastAPI:
    app = FastAPI(
        title="youtube-kids-caption",
        description="Creator review API for the child-focused caption pipeline. "
        + DISCLAIMER,
        version="0.5.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(upload.router, prefix="/api")
    app.include_router(jobs.router, prefix="/api")
    app.include_router(captions.router, prefix="/api")
    app.include_router(settings.router, prefix="/api")

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "disclaimer": DISCLAIMER}

    if DASHBOARD.is_dir():
        assets = DASHBOARD / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

        @app.get("/", response_model=None)
        def dashboard_index() -> FileResponse:
            index = DASHBOARD / "index.html"
            if not index.is_file():
                from fastapi import HTTPException

                raise HTTPException(status_code=404, detail="Dashboard is not installed")
            return FileResponse(index)

        @app.get("/styles.css", response_model=None)
        def dashboard_css() -> FileResponse:
            return FileResponse(DASHBOARD / "styles.css", media_type="text/css")

        @app.get("/app.js", response_model=None)
        def dashboard_js() -> FileResponse:
            return FileResponse(DASHBOARD / "app.js", media_type="application/javascript")

    return app


app = create_app()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m app.main",
        description="Creator dashboard API. The CLI pipeline (python -m app.pipeline) does not need this.",
    )
    parser.add_argument("--serve", action="store_true", help="Start the HTTP server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    if not args.serve:
        parser.print_help()
        print("\nStart the dashboard with: python -m app.main --serve")
        print("CLI captions still work without it: python -m app.pipeline VIDEO.mp4")
        return
    import uvicorn

    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
