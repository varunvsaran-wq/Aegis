"""FastAPI service exposing the Aegis engine.

Endpoints:

- ``GET  /health`` — liveness probe.
- ``POST /index``  — build the in-memory corpus from a list of documents.
- ``POST /answer`` — answer a question, ``harness`` toggling the reliability
  stack (query structuring, injection defense, citation contract, NLI
  verification).
- ``POST /compare`` — the same question with the harness off and on.

Run with ``uvicorn aegis.serve.api:app`` or ``aegis serve --api``. The default
model is the deterministic mock so the service boots with no credentials; set
``AEGIS_SERVE_MODEL`` to any registry alias to use a real model.
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field

_engine = None


def get_engine():
    """Lazily construct the process-wide :class:`AegisEngine`."""
    global _engine
    if _engine is None:
        from aegis.config import load_dotenv
        from aegis.serve.engine import AegisEngine

        load_dotenv()

        _engine = AegisEngine(model=os.environ.get("AEGIS_SERVE_MODEL", "mock"))
    return _engine


class IndexRequest(BaseModel):
    documents: list[str] = Field(..., description="Raw documents to index.")


class IndexResponse(BaseModel):
    n_chunks: int


class AnswerRequest(BaseModel):
    question: str
    harness: bool = True


def create_app():
    """Build and return the FastAPI application."""
    from fastapi import FastAPI, HTTPException

    app = FastAPI(
        title="Aegis",
        description="Model-agnostic RAG reliability harness.",
        version="0.1.0",
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/index", response_model=IndexResponse)
    def index(req: IndexRequest) -> IndexResponse:
        if not req.documents:
            raise HTTPException(status_code=400, detail="documents must be non-empty")
        n = get_engine().index(req.documents)
        return IndexResponse(n_chunks=n)

    @app.post("/answer")
    def answer(req: AnswerRequest) -> dict:
        engine = get_engine()
        if engine._retriever is None:
            raise HTTPException(status_code=409, detail="call /index first")
        from dataclasses import asdict

        return asdict(engine.answer(req.question, harness=req.harness))

    @app.post("/compare")
    def compare(req: AnswerRequest) -> dict:
        """Answer with the harness off and on, side by side."""
        engine = get_engine()
        if engine._retriever is None:
            raise HTTPException(status_code=409, detail="call /index first")
        from dataclasses import asdict

        off, on = engine.compare(req.question)
        return {"harness_off": asdict(off), "harness_on": asdict(on)}

    return app


app = create_app()
