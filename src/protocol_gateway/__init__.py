"""Agent-to-Agent Protocol Gateway - Translates between different agent communication standards."""

from typing import Dict, List, Optional, Any, Callable, Protocol
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import asyncio
import uuid
import json

from ..common import Agent, Message, Protocol as CommonProtocol, EventEmitter


class GatewayProtocol(Enum):
    """Supported protocols in the gateway."""
    MCP = "mcp"
    A2A = "a2a"
    CUSTOM = "custom"
    HTTP = "http"
    WEBSOCKET = "websocket"


@dataclass
class ProtocolAdapter:
    """Adapter for a specific protocol."""
    adapter_id: str
    protocol: GatewayProtocol
    name: str
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    async def send_message(self, message: Message, target: Agent) -> bool:
        """Send a message using this protocol."""
        raise NotImplementedError
    
    async def receive_message(self, raw_message: Any) -> Message:
        """Receive and parse a message."""
        raise NotImplementedError
    
    async def connect(self, agent: Agent) -> bool:
        """Connect to an agent."""
        raise NotImplementedError
    
    async def disconnect(self, agent: Agent) -> bool:
        """Disconnect from an agent."""
        raise NotImplementedError
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "protocol": self.protocol.value,
            "name": self.name,
            "version": self.version,
            "metadata": self.metadata
        }


@dataclass
class MCPPayload:
    """Model Context Protocol payload structure."""
    method: str
    params: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "params": self.params,
            "id": self.id
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MCPPayload':
        return cls(
            method=data.get("method", ""),
            params=data.get("params", {}),
            id=data.get("id")
        )


@dataclass
class A2APayload:
    """Agent-to-Agent Protocol payload structure."""
    action: str
    agent_id: str
    task: str
    context: Dict[str, Any] = field(default_factory=dict)
    callback: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "agent_id": self.agent_id,
            "task": self.task,
            "context": self.context,
            "callback": self.callback
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'A2APayload':
        return cls(
            action=data.get("action", ""),
            agent_id=data.get("agent_id", ""),
            task=data.get("task", ""),
            context=data.get("context", {}),
            callback=data.get("callback")
        )


@dataclass
class TranslationResult:
    """Result of a protocol translation."""
    success: bool
    original_message: Optional[Message] = None
    translated_message: Optional[Message] = None
    source_protocol: Optional[GatewayProtocol] = None
    target_protocol: Optional[GatewayProtocol] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "original_message": original_message.to_dict() if self.original_message else None,
            "translated_message": self.translated_message.to_dict() if self.translated_message else None,
            "source_protocol": self.source_protocol.value if self.source_protocol else None,
            "target_protocol": self.target_protocol.value if self.target_protocol else None,
            "error": self.error,
            "metadata": self.metadata
        }


@dataclass
class GatewayConfig:
    """Configuration for the protocol gateway."""
    default_protocol: GatewayProtocol = GatewayProtocol.CUSTOM
    enable_translation: bool = True
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "default_protocol": self.default_protocol.value,
            "enable_translation": self.enable_translation,
            "enable_caching": self.enable_caching,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "timeout_seconds": self.timeout_seconds,
            "retry_attempts": self.retry_attempts
        }


class MCPAdapter(ProtocolAdapter):
    """Adapter for Model Context Protocol."""
    
    def __init__(self):
        super().__init__(
            adapter_id=str(uuid.uuid4()),
            protocol=GatewayProtocol.MCP,
            name="MCP Adapter",
            version="1.0"
        )
        self._connections: Dict[str, Any] = {}
    
    async def send_message(self, message: Message, target: Agent) -> bool:
        """Send a message using MCP."""
        # Convert to MCP payload
        mcp_payload = MCPPayload(
            method="agent.invoke",
            params={
                "target": target.agent_id,
                "content": message.content,
                "metadata": message.metadata
            },
            id=message.message_id
        )
        
        # Simulate sending (placeholder for actual implementation)
        print(f"[MCP] Sending: {json.dumps(mcp_payload.to_dict())}")
        return True
    
    async def receive_message(self, raw_message: Any) -> Message:
        """Receive and parse an MCP message."""
        if isinstance(raw_message, dict):
            payload = MCPPayload.from_dict(raw_message)
            
            return Message(
                message_id=payload.id or str(uuid.uuid4()),
                sender=payload.params.get("source", "unknown"),
                receiver="local",
                content=payload.params.get("content", ""),
                protocol=CommonProtocol.MCP,
                metadata=payload.params.get("metadata", {})
            )
        
        return Message(
            message_id=str(uuid.uuid4()),
            sender="unknown",
            receiver="local",
            content=str(raw_message),
            protocol=CommonProtocol.MCP
        )
    
    async def connect(self, agent: Agent) -> bool:
        """Connect to an agent via MCP."""
        self._connections[agent.agent_id] = {
            "connected": True,
            "timestamp": datetime.now()
        }
        return True
    
    async def disconnect(self, agent: Agent) -> bool:
        """Disconnect from an agent."""
        if agent.agent_id in self._connections:
            del self._connections[agent.agent_id]
        return True


class A2AAdapter(ProtocolAdapter):
    """Adapter for Agent-to-Agent Protocol."""
    
    def __init__(self):
        super().__init__(
            adapter_id=str(uuid.uuid4()),
            protocol=GatewayProtocol.A2A,
            name="A2A Adapter",
            version="1.0"
        )
        self._connections: Dict[str, Any] = {}
    
    async def send_message(self, message: Message, target: Agent) -> bool:
        """Send a message using A2A."""
        # Convert to A2A payload
        a2a_payload = A2APayload(
            action="delegate",
            agent_id=target.agent_id,
            task=str(message.content),
            context=message.metadata
        )
        
        # Simulate sending
        print(f"[A2A] Sending: {json.dumps(a2a_payload.to_dict())}")
        return True
    
    async def receive_message(self, raw_message: Any) -> Message:
        """Receive and parse an A2A message."""
        if isinstance(raw_message, dict):
            payload = A2APayload.from_dict(raw_message)
            
            return Message(
                message_id=str(uuid.uuid4()),
                sender=payload.agent_id,
                receiver="local",
                content=payload.task,
                protocol=CommonProtocol.A2A,
                metadata=payload.context
            )
        
        return Message(
            message_id=str(uuid.uuid4()),
            sender="unknown",
            receiver="local",
            content=str(raw_message),
            protocol=CommonProtocol.A2A
        )
    
    async def connect(self, agent: Agent) -> bool:
        """Connect to an agent via A2A."""
        self._connections[agent.agent_id] = {
            "connected": True,
            "timestamp": datetime.now()
        }
        return True
    
    async def disconnect(self, agent: Agent) -> bool:
        """Disconnect from an agent."""
        if agent.agent_id in self._connections:
            del self._connections[agent.agent_id]
        return True


class CustomAdapter(ProtocolAdapter):
    """Adapter for custom/protocol-agnostic communication."""
    
    def __init__(self):
        super().__init__(
            adapter_id=str(uuid.uuid4()),
            protocol=GatewayProtocol.CUSTOM,
            name="Custom Adapter",
            version="1.0"
        )
        self._connections: Dict[str, Any] = {}
    
    async def send_message(self, message: Message, target: Agent) -> bool:
        """Send a message using custom protocol."""
        # No translation needed - use as-is
        print(f"[Custom] Sending to {target.agent_id}: {message.content}")
        return True
    
    async def receive_message(self, raw_message: Any) -> Message:
        """Receive a custom message."""
        return Message(
            message_id=str(uuid.uuid4()),
            sender="unknown",
            receiver="local",
            content=raw_message,
            protocol=CommonProtocol.CUSTOM
        )
    
    async def connect(self, agent: Agent) -> bool:
        """Connect to an agent."""
        self._connections[agent.agent_id] = {
            "connected": True,
            "timestamp": datetime.now()
        }
        return True
    
    async def disconnect(self, agent: Agent) -> bool:
        """Disconnect from an agent."""
        if agent.agent_id in self._connections:
            del self._connections[agent.agent_id]
        return True


class HTTPAdapter(ProtocolAdapter):
    """Adapter for HTTP-based communication."""
    
    def __init__(self):
        super().__init__(
            adapter_id=str(uuid.uuid4()),
            protocol=GatewayProtocol.HTTP,
            name="HTTP Adapter",
            version="1.0"
        )
        self._connections: Dict[str, Any] = {}
    
    async def send_message(self, message: Message, target: Agent) -> bool:
        """Send a message via HTTP."""
        # Simulate HTTP POST
        print(f"[HTTP] POST to {target.agent_id}: {message.content}")
        return True
    
    async def receive_message(self, raw_message: Any) -> Message:
        """Receive an HTTP message."""
        return Message(
            message_id=str(uuid.uuid4()),
            sender="http-source",
            receiver="local",
            content=raw_message,
            protocol=CommonProtocol.HTTP
        )
    
    async def connect(self, agent: Agent) -> bool:
        """Connect to an agent via HTTP."""
        self._connections[agent.agent_id] = {
            "connected": True,
            "timestamp": datetime.now()
        }
        return True
    
    async def disconnect(self, agent: Agent) -> bool:
        """Disconnect from an agent."""
        if agent.agent_id in self._connections:
            del self._connections[agent.agent_id]
        return True


class ProtocolGateway:
    """Gateway for translating between different agent communication protocols."""
    
    def __init__(self, config: Optional[GatewayConfig] = None):
        self.config = config or GatewayConfig()
        self.adapters: Dict[GatewayProtocol, ProtocolAdapter] = {}
        self.agents: Dict[str, Agent] = {}
        self.connections: Dict[str, Dict[str, Any]] = {}  # agent_id -> connection info
        self.events = EventEmitter()
        self._message_cache: Dict[str, Any] = {}
        
        # Register default adapters
        self.register_adapter(GatewayProtocol.MCP, MCPAdapter())
        self.register_adapter(GatewayProtocol.A2A, A2AAdapter())
        self.register_adapter(GatewayProtocol.CUSTOM, CustomAdapter())
        self.register_adapter(GatewayProtocol.HTTP, HTTPAdapter())
    
    def register_adapter(self, protocol: GatewayProtocol, adapter: ProtocolAdapter) -> None:
        """Register a protocol adapter."""
        self.adapters[protocol] = adapter
        self.events.emit("adapter_registered", protocol, adapter)
    
    def get_adapter(self, protocol: GatewayProtocol) -> Optional[ProtocolAdapter]:
        """Get an adapter for a protocol."""
        return self.adapters.get(protocol)
    
    def register_agent(self, agent: Agent) -> None:
        """Register an agent with the gateway."""
        self.agents[agent.agent_id] = agent
        self.events.emit("agent_registered", agent)
    
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent."""
        if agent_id in self.agents:
            agent = self.agents.pop(agent_id)
            self.connections.pop(agent_id, None)
            self.events.emit("agent_unregistered", agent)
    
    async def connect_agent(
        self,
        agent_id: str,
        protocol: Optional[GatewayProtocol] = None
    ) -> bool:
        """Connect an agent using a specific protocol."""
        agent = self.agents.get(agent_id)
        if not agent:
            return False
        
        protocol = protocol or self.config.default_protocol
        adapter = self.adapters.get(protocol)
        
        if not adapter:
            return False
        
        success = await adapter.connect(agent)
        
        if success:
            self.connections[agent_id] = {
                "protocol": protocol,
                "connected_at": datetime.now(),
                "adapter_id": adapter.adapter_id
            }
            self.events.emit("agent_connected", agent, protocol)
        
        return success
    
    async def disconnect_agent(self, agent_id: str) -> bool:
        """Disconnect an agent."""
        if agent_id not in self.connections:
            return False
        
        connection = self.connections[agent_id]
        protocol = connection.get("protocol")
        
        adapter = self.adapters.get(protocol)
        agent = self.agents.get(agent_id)
        
        if adapter and agent:
            success = await adapter.disconnect(agent)
            if success:
                del self.connections[agent_id]
                self.events.emit("agent_disconnected", agent)
            return success
        
        return False
    
    async def send_message(
        self,
        message: Message,
        target_agent_id: str,
        target_protocol: Optional[GatewayProtocol] = None
    ) -> TranslationResult:
        """Send a message to an agent, translating if necessary."""
        target_agent = self.agents.get(target_agent_id)
        if not target_agent:
            return TranslationResult(
                success=False,
                error=f"Agent {target_agent_id} not found"
            )
        
        # Determine target protocol
        target_protocol = target_protocol or self._get_agent_protocol(target_agent_id)
        
        source_adapter = self.adapters.get(GatewayProtocol(message.protocol.value))
        target_adapter = self.adapters.get(target_protocol)
        
        if not source_adapter or not target_adapter:
            return TranslationResult(
                success=False,
                error="Protocol adapter not found"
            )
        
        try:
            # Send using target adapter
            success = await target_adapter.send_message(message, target_agent)
            
            if success:
                self.events.emit("message_sent", message, target_agent)
                return TranslationResult(
                    success=True,
                    original_message=message,
                    source_protocol=GatewayProtocol(message.protocol.value),
                    target_protocol=target_protocol
                )
            else:
                return TranslationResult(
                    success=False,
                    error="Failed to send message"
                )
        
        except Exception as e:
            return TranslationResult(
                success=False,
                error=str(e)
            )
    
    async def translate_message(
        self,
        message: Message,
        target_protocol: GatewayProtocol
    ) -> TranslationResult:
        """Translate a message from one protocol to another."""
        source_protocol = GatewayProtocol(message.protocol.value)
        
        if source_protocol == target_protocol:
            return TranslationResult(
                success=True,
                original_message=message,
                translated_message=message,
                source_protocol=source_protocol,
                target_protocol=target_protocol
            )
        
        source_adapter = self.adapters.get(source_protocol)
        target_adapter = self.adapters.get(target_protocol)
        
        if not source_adapter or not target_adapter:
            return TranslationResult(
                success=False,
                error="Protocol adapter not found"
            )
        
        try:
            # Create translated message
            translated_message = Message(
                message_id=str(uuid.uuid4()),
                sender=message.sender,
                receiver=message.receiver,
                content=message.content,
                protocol=CommonProtocol(target_protocol.value),
                metadata={
                    **message.metadata,
                    "translated_from": source_protocol.value,
                    "original_message_id": message.message_id
                }
            )
            
            self.events.emit("message_translated", message, translated_message)
            
            return TranslationResult(
                success=True,
                original_message=message,
                translated_message=translated_message,
                source_protocol=source_protocol,
                target_protocol=target_protocol,
                metadata={"translation_time": datetime.now().isoformat()}
            )
        
        except Exception as e:
            return TranslationResult(
                success=False,
                error=str(e)
            )
    
    async def broadcast(
        self,
        message: Message,
        agent_ids: List[str],
        protocol: Optional[GatewayProtocol] = None
    ) -> Dict[str, TranslationResult]:
        """Broadcast a message to multiple agents."""
        results = {}
        
        for agent_id in agent_ids:
            result = await self.send_message(message, agent_id, protocol)
            results[agent_id] = result
        
        return results
    
    def _get_agent_protocol(self, agent_id: str) -> GatewayProtocol:
        """Get the protocol for an agent."""
        if agent_id in self.connections:
            return self.connections[agent_id].get("protocol", self.config.default_protocol)
        
        agent = self.agents.get(agent_id)
        if agent:
            # Use agent's native protocol
            return GatewayProtocol(agent.protocol.value)
        
        return self.config.default_protocol
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get gateway statistics."""
        return {
            "registered_agents": len(self.agents),
            "active_connections": len(self.connections),
            "registered_adapters": len(self.adapters),
            "config": self.config.to_dict(),
            "connections": {
                agent_id: {
                    "protocol": conn.get("protocol", "").value if conn.get("protocol") else "",
                    "connected_at": conn.get("connected_at", "").isoformat() if conn.get("connected_at") else ""
                }
                for agent_id, conn in self.connections.items()
            }
        }
    
    def list_agents_by_protocol(self, protocol: GatewayProtocol) -> List[str]:
        """List agents connected via a specific protocol."""
        return [
            agent_id for agent_id, conn in self.connections.items()
            if conn.get("protocol") == protocol
        ]
