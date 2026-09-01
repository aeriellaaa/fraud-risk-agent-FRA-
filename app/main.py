from fastapi import FastAPI
from app.routers import ingest, score, review, decisions, audit_log, metrics, health, process, entity_drift_demo

app = FastAPI(
    title="Fraud/Risk Flag Agent",
    description="Three-agent fraud/risk detection: Pattern/Evasion Agent -> Detection & "
                 "Scoring Agent -> Reviewer Agent, with append-only audit logging and "
                 "threshold-based decision routing.",
    version="0.1.0",
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


from app.routers import ieee_pipeline
app.include_router(ieee_pipeline.router, tags=["ieee-pipeline"])
