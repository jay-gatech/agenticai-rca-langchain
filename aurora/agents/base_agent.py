"""
Base Agent class with common functionality
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
import structlog

logger = structlog.get_logger()

class BaseAgent(ABC):
    """Base class for all specialist agents"""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.logger = logger.bind(agent=agent_name)
        self.logger.info("agent_initialized")

    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent's primary function"""
        raise NotImplementedError

    def log_execution(self, action: str, **kwargs):
        """Log agent execution"""
        self.logger.info(action, **kwargs)
