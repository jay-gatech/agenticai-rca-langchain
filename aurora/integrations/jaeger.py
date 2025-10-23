"""Jaeger integration (optional deps)."""
from typing import Dict, List
from datetime import datetime
try:
    import structlog
    _LOGGER = structlog.get_logger()
except Exception:  # pragma: no cover
    import logging
    logging.basicConfig(level=logging.INFO)
    _LOGGER = logging.getLogger(__name__)
try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None

class JaegerClient:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.client = httpx.AsyncClient() if httpx else None
    
    async def get_traces(self, service_name: str, start_time: datetime, end_time: datetime, limit: int = 100) -> List[Dict]:
        if not self.client:
            _LOGGER.warning("httpx not available; returning empty trace list")
            return []
        params = {
            "service": service_name,
            "start": int(start_time.timestamp() * 1_000_000),
            "end": int(end_time.timestamp() * 1_000_000),
            "limit": limit
        }
        response = await self.client.get(f"{self.endpoint}/api/traces", params=params)
        if response.status_code == 200:
            data = response.json()
            return data.get("data", [])
        return []
