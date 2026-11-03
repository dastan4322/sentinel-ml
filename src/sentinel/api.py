import hashlib
import os
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Annotated

import joblib
import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from .data import FEATURES


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transaction_id: str = Field(min_length=1, max_length=100)
    amount: FiniteFloat = Field(ge=0, le=1e9)
    v: list[Annotated[FiniteFloat, Field(ge=-1e6, le=1e6)]] = Field(min_length=28, max_length=28)


class Batch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transactions: list[Transaction] = Field(min_length=1, max_length=1000)


def create_app(model_path=None):
    registry = CollectorRegistry()
    predictions = Counter("sentinel_predictions", "Scored transactions", registry=registry)
    alerts = Counter("sentinel_alerts", "Transactions flagged for review", registry=registry)
    latency = Histogram("sentinel_batch_seconds", "Prediction batch latency", registry=registry)
    path = Path(model_path or os.environ.get("MODEL_PATH", "artifacts/demo/model.joblib"))

    @asynccontextmanager
    async def lifespan(app):
        # joblib is executable serialization: only load your own trusted local artifact.
        bundle = joblib.load(path)
        if bundle["features"] != FEATURES or bundle["sklearn"] != sklearn.__version__:
            raise RuntimeError(
                "Incompatible model schema or sklearn version; retrain in this environment"
            )
        app.state.bundle = bundle
        app.state.version = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        yield

    app = FastAPI(title="Sentinel ML", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        # Avoid echoing sensitive inputs or non-JSON numeric values in error responses.
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    {"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in exc.errors()
                ]
            },
        )

    @app.get("/health")
    def health():
        return {
            "status": "ready",
            "model_version": app.state.version,
            "source": app.state.bundle["source"],
        }

    @app.post("/predict")
    def predict(batch: Batch):
        ids = [t.transaction_id for t in batch.transactions]
        if len(ids) != len(set(ids)):
            raise HTTPException(422, "Duplicate transaction IDs in batch")
        started = perf_counter()
        frame = pd.DataFrame([dict(zip(FEATURES, [*t.v, t.amount])) for t in batch.transactions])
        bundle = app.state.bundle
        scores = bundle["model"].predict_proba(frame[FEATURES])[:, 1]
        decisions = scores >= bundle["threshold"]
        predictions.inc(len(scores))
        alerts.inc(int(decisions.sum()))
        latency.observe(perf_counter() - started)
        return {
            "model_version": app.state.version,
            "source": bundle["source"],
            "threshold": bundle["threshold"],
            "predictions": [
                {"transaction_id": t.transaction_id, "risk_score": float(s), "review": bool(d)}
                for t, s, d in zip(batch.transactions, scores, decisions)
            ],
        }

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
