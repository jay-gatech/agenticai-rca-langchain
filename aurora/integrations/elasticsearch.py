"""
Elasticsearch integration for log analysis
"""
from typing import Dict, List
from datetime import datetime
from elasticsearch import AsyncElasticsearch
import structlog

logger = structlog.get_logger()

class ElasticsearchClient:
    """Client for querying Elasticsearch logs"""

    def __init__(self, hosts: List[str]):
        self.client = AsyncElasticsearch(hosts)

    async def search_logs(
        self,
        service_name: str,
        start_time: datetime,
        end_time: datetime,
        severity: List[str] = None
    ) -> List[Dict]:
        """Search logs for a service"""

        query = {
            "bool": {
                "must": [
                    {"match": {"service": service_name}},
                    {"range": {
                        "@timestamp": {
                            "gte": start_time.isoformat(),
                            "lte": end_time.isoformat()
                        }
                    }}
                ]
            }
        }

        if severity:
            query["bool"]["must"].append({
                "terms": {"severity": severity}
            })

        result = await self.client.search(
            index="microservices-logs-*",
            query=query,
            size=1000
        )

        return [hit["_source"] for hit in result["hits"]["hits"]]
