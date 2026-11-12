"""FastAPI app. Run: uvicorn src.api.main:app --reload"""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.api.db import make_engine, make_session_factory
from src.api.service import STATES, HazardService, InvalidTransition, NotFound


class ReportIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    latitude: Optional[float] = Field(default=None, ge=-90, le=90)
    longitude: Optional[float] = Field(default=None, ge=-180, le=180)
    location_text: Optional[str] = None
    timestamp: Optional[datetime] = None
    source: Literal["social", "app", "sms"] = "app"
    media_flag: bool = False
    language: Optional[str] = None


class ReviewIn(BaseModel):
    state: Literal["Unverified", "Under Review", "Confirmed", "Rejected", "Escalated", "Needs More Info"]
    reviewer: str = Field(min_length=1, max_length=80)
    notes: str = Field(default="", max_length=2000)


def _naive(dt):
    return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt is not None and dt.tzinfo else dt


def create_app(database_url: str | None = None, classifier=None) -> FastAPI:
    holder = {}

    @asynccontextmanager
    async def lifespan(app):
        holder["svc"] = HazardService(make_session_factory(make_engine(database_url)), classifier)
        yield

    app = FastAPI(title="Coastal Hazard Intelligence API", version="1.0", lifespan=lifespan,
                  description="Public reports are REPORTED, never confirmed, until a human reviewer confirms the cluster.")
    svc = lambda: holder["svc"]

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        return JSONResponse(status_code=500, content={"detail": "internal error; the report/cluster state was not changed", "type": type(exc).__name__})

    @app.get("/health")
    def health():
        stats = svc().live_stats()
        return {"status": "ok", "classifier": stats["classifier"], "embedder": stats["embedder"], "degraded": stats["classifier"] == "fallback_keyword"}

    @app.post("/reports", status_code=201)
    def post_report(r: ReportIn):
        try:
            out, warnings = svc().ingest([r.model_dump()])
        except ValueError as e:
            raise HTTPException(422, str(e))
        rep = out[0]
        note = {"triage": "missing location/time -> manual triage queue (not clustered)",
                "not_relevant": "classified as not a coastal hazard; stored, not clustered",
                "clustered": "grouped into an incident cluster (reported, unverified)"}[rep["queue"]]
        return {"report": rep, "queue_note": note, "warnings": warnings}

    @app.get("/reports")
    def get_reports(triage: Optional[bool] = None, cluster_id: Optional[str] = None, limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)):
        return svc().list_reports(triage, cluster_id, limit, offset)

    @app.get("/clusters")
    def get_clusters(hazard_type: Optional[str] = None, priority: Optional[str] = None, status: Optional[str] = None,
                     since: Optional[datetime] = None, until: Optional[datetime] = None, min_unique: int = Query(1, ge=1),
                     include_superseded: bool = False):
        """hazard_type / priority / status accept comma-separated values."""
        if status and not set(status.split(",")) <= set(STATES):
            raise HTTPException(422, f"status must be among {STATES}")
        return svc().list_clusters(hazard_type, priority, status, _naive(since), _naive(until), min_unique, include_superseded)

    @app.get("/clusters/{cluster_id}")
    def get_cluster(cluster_id: str):
        try:
            return svc().get_cluster(cluster_id)
        except NotFound:
            raise HTTPException(404, "cluster not found")

    @app.post("/clusters/{cluster_id}/review")
    def review(cluster_id: str, body: ReviewIn):
        try:
            return svc().review(cluster_id, body.state, body.reviewer, body.notes)
        except NotFound:
            raise HTTPException(404, "cluster not found")
        except InvalidTransition as e:
            raise HTTPException(409, str(e))

    @app.post("/recluster")
    def recluster():
        return {"n_clusters": svc().recluster()}

    @app.get("/metrics")
    def metrics():
        return {"live": svc().live_stats(), "offline_evaluation": svc().offline_metrics()}

    return app


app = create_app()
