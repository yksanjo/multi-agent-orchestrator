"""Base agent class and related types."""

from enum import Enum, auto
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import uuid


class AgentRole(Enum):
    """Agent roles in the hierarchy."""
    STRATEGIC = auto()     # 3-5 subordinates, high-level coordination
    TACTICAL = auto()      # 5-15 subordinates, operational management
    OPERATIONAL = auto()   # Direct task execution


class AgentStatus(Enum):
    """Agent execution status."""
    IDLE = auto()
    WORKING = auto()
    WAITING = auto()
    ERROR = auto()
    OFFLINE = auto()


@dataclass
class Task:
    """Task definition for agent execution."""
    task_id: str = ""
    objective: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    priority: int = 5  # 1-10, higher is more urgent
    deadline: Optional[datetime] = None
    parent_task_id: Optional[str] = None
    required_capabilities: List[str] = field(default_factory=list)
    risk_level: str = "low"  # low, medium, high, critical
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not self.task_id:
            self.task_id = str(uuid.uuid4())
        if not self.objective:
            self.objective = "Untitled Task"


@dataclass
class TaskResult:
    """Result of task execution."""
    task_id: str
    agent_id: str
    success: bool
    result: Any
    execution_time: float
    completed_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


class Agent:
    """
    Base agent class with common functionality.
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        role: AgentRole = AgentRole.OPERATIONAL,
        capabilities: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.name = name or f"Agent-{self.agent_id[:8]}"
        self.role = role
        self.capabilities = set(capabilities or [])
        self.config = config or {}
        self.status = AgentStatus.IDLE
        self.created_at = datetime.utcnow()
        
        # Statistics
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.total_execution_time = 0.0
        
        # Callbacks
        self._status_callbacks: List[Callable] = []
        self._result_callbacks: List[Callable] = []
    
    def has_capability(self, capability: str) -> bool:
        """Check if agent has a specific capability."""
        return capability in self.capabilities
    
    def add_capability(self, capability: str) -> None:
        """Add a capability to the agent."""
        self.capabilities.add(capability)
    
    def set_status(self, status: AgentStatus) -> None:
        """Update agent status and notify listeners."""
        old_status = self.status
        self.status = status
        for callback in self._status_callbacks:
            callback(self, old_status, status)
    
    def on_status_change(self, callback: Callable) -> None:
        """Register a status change callback."""
        self._status_callbacks.append(callback)
    
    def on_result(self, callback: Callable) -> None:
        """Register a result callback."""
        self._result_callbacks.append(callback)
    
    def _notify_result(self, result: TaskResult) -> None:
        """Notify result listeners."""
        for callback in self._result_callbacks:
            callback(self, result)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        total_tasks = self.tasks_completed + self.tasks_failed
        success_rate = (
            self.tasks_completed / total_tasks * 100
            if total_tasks > 0 else 0
        )
        avg_time = (
            self.total_execution_time / total_tasks
            if total_tasks > 0 else 0
        )
        
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "role": self.role.name,
            "status": self.status.name,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "success_rate": success_rate,
            "avg_execution_time": avg_time,
            "capabilities": list(self.capabilities),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize agent to dictionary."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "role": self.role.name,
            "capabilities": list(self.capabilities),
            "status": self.status.name,
            "created_at": self.created_at.isoformat(),
            "stats": self.get_stats(),
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name}, {self.role.name})"
