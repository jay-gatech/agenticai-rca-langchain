"""Elasticsearch integration (optional deps)."""
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
    from elasticsearch import AsyncElasticsearch  # type: ignore
except Exception:  # pragma: no cover
    class AsyncElasticsearch:
        def __init__(self, *args, **kwargs): pass
        async def search(self, *args, **kwargs): return {"hits": {"hits": []}}

class ElasticsearchClient:
    def __init__(self, hosts: List[str]):
        self.client = AsyncElasticsearch(hosts)
    
    async def search_logs(self, service_name: str, start_time: datetime, end_time: datetime, severity: List[str] = None) -> List[Dict]:
        query = {
            "bool": {
                "must": [
                    {"match": {"service": service_name}},
                    {"range": {"@timestamp": {"gte": start_time.isoformat(), "lte": end_time.isoformat()}}}
                ]
            }
        }
        if severity:
            query["bool"]["must"].append({"terms": {"severity": severity}})
        result = await self.client.search(index="microservices-logs-*", query=query, size=1000)
        return [hit.get("_source", {}) for hit in result.get("hits", {}).get("hits", [])]
