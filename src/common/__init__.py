"""Common utilities and types for the multi-agent orchestrator."""

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import asyncio
import json


class AgentType(Enum):
    """Supported accelerator types for agents."""
    NVIDIA_GPU = "nvidia"
    AWS_TRAINIUM = "trainium"
    GOOGLE_TPU = "tpu"
    CPU = "cpu"


class Protocol(Enum):
    """Supported agent communication protocols."""
    MCP = "mcp"
    A2A = "a2a"
    CUSTOM = "custom"
    HTTP = "http"


class StateStatus(Enum):
    """Status of a workflow state."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class Agent:
    """Represents an agent in the system."""
    agent_id: str
    name: str
    agent_type: AgentType
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    location: Optional[str] = None
    status: str = "idle"
    protocol: Protocol = Protocol.CUSTOM
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "agent_type": self.agent_type.value,
            "capabilities": self.capabilities,
            "metadata": self.metadata,
            "location": self.location,
            "status": self.status,
            "protocol": self.protocol.value
        }


@dataclass
class Task:
    """Represents a task to be executed by agents."""
    task_id: str
    description: str
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    deadline: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "priority": self.priority,
            "metadata": self.metadata,
            "deadline": self.deadline.isoformat() if self.deadline else None
        }


@dataclass
class Message:
    """Represents a message between agents."""
    message_id: str
    sender: str
    receiver: str
    content: Any
    protocol: Protocol = Protocol.CUSTOM
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "protocol": self.protocol.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class Result:
    """Result from an agent or consensus operation."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


class EventEmitter:
    """Simple event emitter for state changes."""
    
    def __init__(self):
        self._listeners: Dict[str, List[Callable]] = {}
    
    def on(self, event: str, callback: Callable) -> None:
        if event not in self._listeners:
            self._listeners[event] = []
        self._listeners[event].append(callback)
    
    def emit(self, event: str, *args, **kwargs) -> None:
        if event in self._listeners:
            for callback in self._listeners[event]:
                callback(*args, **kwargs)
    
    def off(self, event: str, callback: Callable) -> None:
        if event in self._listeners:
            self._listeners[event] = [
                cb for cb in self._listeners[event] if cb != callback
            ]


async def async_retry(
    func: Callable,
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0
) -> Any:
    """Async retry with exponential backoff."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                await asyncio.sleep(delay * (backoff ** attempt))
    raise last_error


def serialize(obj: Any) -> str:
    """Serialize an object to JSON."""
    if hasattr(obj, 'to_dict'):
        return json.dumps(obj.to_dict())
    return json.dumps(obj)


def deserialize(data: str, target_type: type) -> Any:
    """Deserialize JSON to an object."""
    obj = json.loads(data)
    if target_type == Agent:
        return Agent(**obj)
    elif target_type == Task:
        return Task(**obj)
    elif target_type == Message:
        return Message(**obj)
    return obj
