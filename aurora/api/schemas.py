"""
Pydantic schemas for API
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

class Alert(BaseModel):
    service_name: str = Field(..., description="Name of the affected service")
    alert_type: str = Field(..., description="Type of alert")
    severity: str = Field(..., description="Severity level")
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
    causal_graph: Dict
    recommendations: List[str]
    execution_time: float

class AnalysisResponse(BaseModel):
    incident_id: str
    status: str
    analysis: Optional[Analysis] = None
    error: Optional[str] = None
