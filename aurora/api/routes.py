"""API route definitions"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
import time

router = APIRouter()

class Alert(BaseModel):
    service_name: str = Field(..., description="Name of the affected service")
    alert_type: str = Field(..., description="Type of alert")
    severity: str = Field(..., description="Severity level, e.g., CRITICAL, HIGH, MEDIUM, LOW")
    timestamp: Optional[datetime] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class RootCause(BaseModel):
    service: str
    score: float
    confidence: float

class Analysis(BaseModel):
    incident_id: str
    root_causes: List[RootCause]
    causal_graph: Dict[str, Any]
    recommendations: List[str]
    execution_time: float

class AnalysisResponse(BaseModel):
    incident_id: str
    status: str
    analysis: Optional[Analysis] = None
    error: Optional[str] = None

@router.post("/api/v1/analyze", response_model=AnalysisResponse)
async def analyze_incident(alert: Alert):
    """Analyze an incident (mocked implementation suitable for local testing)."""
    start = time.perf_counter()
    if alert.severity not in {"CRITICAL", "HIGH", "MEDIUM", "LOW"}:
        raise HTTPException(status_code=400, detail="Invalid severity")
    incident_id = f"inc-{int(time.time())}"
    # trivial "analysis"
    root_causes = [
        RootCause(service=alert.service_name, score=0.82, confidence=0.76),
        RootCause(service=f"{alert.service_name}-dependency", score=0.41, confidence=0.38),
    ]
    graph = {
        "nodes": [alert.service_name, f"{alert.service_name}-dependency"],
        "edges": [{"source": f"{alert.service_name}-dependency", "target": alert.service_name, "weight": 0.41}],
    }
    recs = [
        f"Restart deployment {alert.service_name}",
        f"Scale {alert.service_name}-dependency to 3 replicas",
    ]
    elapsed = time.perf_counter() - start
    return AnalysisResponse(
        incident_id=incident_id,
        status="ok",
        analysis=Analysis(
            incident_id=incident_id,
            root_causes=root_causes,
            causal_graph=graph,
            recommendations=recs,
            execution_time=elapsed,
        ),
    )

@router.get("/api/v1/incidents/{incident_id}", response_model=AnalysisResponse)
async def get_incident(incident_id: str):
    """Return a canned incident for demo purposes."""
    return AnalysisResponse(incident_id=incident_id, status="ok")

@router.get("/api/v1/health")
async def health_check():
    return {"status": "healthy"}
