from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import (
    ingest, score, review, decisions, audit_log, metrics, health,
    process, entity_drift_demo, ieee_pipeline,
)

app = FastAPI(
    title="Fraud/Risk Flag Agent",
    description="Three-agent fraud/risk detection: Pattern/Evasion Agent -> Detection & "
                 "Scoring Agent -> Reviewer Agent, with append-only audit logging and "
                 "threshold-based decision routing.",
    version="0.1.0",
)

# CORS: allows the frontend (guard-analytics-ui / Fraudline Console, running on a
# different origin -- Vite dev server or a deployed static host) to call this API.
# Tighten allow_origins to the real deployed frontend URL before going to production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8080", "http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["health"])
app.include_router(ingest.router, tags=["ingest"])
app.include_router(score.router, tags=["score"])
app.include_router(review.router, tags=["review"])
app.include_router(decisions.router, tags=["decisions"])
app.include_router(audit_log.router, tags=["audit"])
app.include_router(metrics.router, tags=["metrics"])
app.include_router(process.router, tags=["pipeline"])
app.include_router(entity_drift_demo.router, tags=["demo"])
app.include_router(ieee_pipeline.router, tags=["ieee-pipeline"])
