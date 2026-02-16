"""Inter-agent communication and messaging."""

from enum import Enum, auto
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import uuid


class MessageType(Enum):
    """Types of messages between agents."""
    REQUEST = auto()
    RESPONSE = auto()
    NOTIFICATION = auto()
    BROADCAST = auto()
    ESCALATION = auto()


class MessagePriority(Enum):
    """Message priority levels."""
    LOW = auto()
    NORMAL = auto()
    HIGH = auto()
    URGENT = auto()


@dataclass
class Message:
    """A message between agents."""
    message_id: str
    sender_id: str
    receiver_id: str
    message_type: MessageType
    content: Any
    priority: MessagePriority = MessagePriority.NORMAL
    correlation_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    read: bool = False


class MessageBus:
    """Central message bus for agent communication."""
    
    def __init__(self):
        self.messages: Dict[str, Message] = {}
        self.queues: Dict[str, List[Message]] = {}  # agent_id -> messages
        self.subscriptions: Dict[str, List[Callable]] = {}  # agent_id -> callbacks
        self.broadcast_handlers: List[Callable] = []
    
    def send_message(
        self,
        sender_id: str,
        receiver_id: str,
        message_type: MessageType,
        content: Any,
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> Message:
        """Send a message to an agent."""
        message = Message(
            message_id=str(uuid.uuid4()),
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            priority=priority,
            correlation_id=str(uuid.uuid4()),
        )
        
        self.messages[message.message_id] = message
        
        # Add to receiver's queue
        if receiver_id not in self.queues:
            self.queues[receiver_id] = []
        self.queues[receiver_id].append(message)
        
        # Trigger subscriptions
        self._triggerSubscriptions(receiver_id, message)
        
        return message
    
    def broadcast(
        self,
        sender_id: str,
        content: Any,
        message_type: MessageType = MessageType.BROADCAST,
    ) -> List[Message]:
        """Broadcast a message to all agents."""
        messages = []
        
        for receiver_id in self.queues.keys():
            if receiver_id != sender_id:
                msg = self.send_message(
                    sender_id=sender_id,
                    receiver_id=receiver_id,
                    message_type=message_type,
                    content=content,
                )
                messages.append(msg)
        
        # Trigger broadcast handlers
        for handler in self.broadcast_handlers:
            handler(sender_id, content)
        
        return messages
    
    def receive_message(self, agent_id: str) -> Optional[Message]:
        """Receive the next message for an agent."""
        queue = self.queues.get(agent_id, [])
        
        if queue:
            # Return highest priority message
            queue.sort(key=lambda m: m.priority.value, reverse=True)
            message = queue.pop(0)
            message.read = True
            return message
        
        return None
    
    def peek_messages(self, agent_id: str, count: int = 10) -> List[Message]:
        """Peek at messages without removing them."""
        queue = self.queues.get(agent_id, [])
        sorted_queue = sorted(queue, key=lambda m: m.priority.value, reverse=True)
        return sorted_queue[:count]
    
    def subscribe(self, agent_id: str, callback: Callable) -> None:
        """Subscribe to messages for an agent."""
        if agent_id not in self.subscriptions:
            self.subscriptions[agent_id] = []
        self.subscriptions[agent_id].append(callback)
    
    def unsubscribe(self, agent_id: str, callback: Callable) -> None:
        """Unsubscribe from messages."""
        if agent_id in self.subscriptions:
            self.subscriptions[agent_id] = [
                cb for cb in self.subscriptions[agent_id]
                if cb != callback
            ]
    
    def _triggerSubscriptions(self, agent_id: str, message: Message) -> None:
        """Trigger subscription callbacks."""
        if agent_id in self.subscriptions:
            for callback in self.subscriptions[agent_id]:
                try:
                    callback(message)
                except Exception:
                    pass  # Don't let callback errors break messaging
    
    def add_broadcast_handler(self, handler: Callable) -> None:
        """Add a handler for broadcast messages."""
        self.broadcast_handlers.append(handler)
    
    def get_message_count(self, agent_id: str) -> int:
        """Get count of unread messages."""
        return len(self.queues.get(agent_id, []))


class AgentProtocol:
    """Defines communication protocols for agents."""
    
    @staticmethod
    def request_task(
        message_bus: MessageBus,
        sender_id: str,
        receiver_id: str,
        task_description: str,
    ) -> Message:
        """Send a task request to another agent."""
        return message_bus.send_message(
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=MessageType.REQUEST,
            content={"action": "task_request", "description": task_description},
            priority=MessagePriority.HIGH,
        )
    
    @staticmethod
    def respond_to_request(
        message_bus: MessageBus,
        sender_id: str,
        receiver_id: str,
        original_message_id: str,
        response_content: Any,
    ) -> Message:
        """Respond to a request message."""
        return message_bus.send_message(
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=MessageType.RESPONSE,
            content=response_content,
            priority=MessagePriority.NORMAL,
        )
    
    @staticmethod
    def escalate(
        message_bus: MessageBus,
        sender_id: str,
        receiver_id: str,
        issue_description: str,
        context: Dict[str, Any],
    ) -> Message:
        """Escalate an issue to another agent."""
        return message_bus.send_message(
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=MessageType.ESCALATION,
            content={
                "issue": issue_description,
                "context": context,
            },
            priority=MessagePriority.URGENT,
        )
    
    @staticmethod
    def notify(
        message_bus: MessageBus,
        sender_id: str,
        receiver_id: str,
        notification: str,
    ) -> Message:
        """Send a notification to an agent."""
        return message_bus.send_message(
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=MessageType.NOTIFICATION,
            content={"notification": notification},
            priority=MessagePriority.NORMAL,
        )
