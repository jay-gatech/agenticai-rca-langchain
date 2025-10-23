"""Base Agent class with common functionality"""
from abc import ABC, abstractmethod
from typing import Dict, Any
try:
    import structlog
    _LOGGER = structlog.get_logger()
except Exception:  # pragma: no cover
    import logging
    logging.basicConfig(level=logging.INFO)
    _LOGGER = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all specialist agents"""
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.logger = _LOGGER
        self.logger.info("agent_initialized: %s", agent_name)
    
    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
    
    def log_execution(self, action: str, **kwargs):
        self.logger.info("action=%s %s", action, kwargs)
