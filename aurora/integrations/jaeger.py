"""
Jaeger integration for distributed tracing
"""
from typing import Dict, List
from datetime import datetime
import httpx
import structlog

logger = structlog.get_logger()

class JaegerClient:
    """Client for querying Jaeger traces"""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.client = httpx.AsyncClient()

    async def get_traces(
        self,
        service_name: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 100
    ) -> List[Dict]:
        """Get traces for a service"""

        params = {
            "service": service_name,
            "start": int(start_time.timestamp() * 1000000),
            "end": int(end_time.timestamp() * 1000000),
            "limit": limit
        }

        response = await self.client.get(
            f"{self.endpoint}/api/traces",
            params=params
        )

        if response.status_code == 200:
            data = response.json()
            return data.get("data", [])

        return []
