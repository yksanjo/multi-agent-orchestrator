"""Multi-Agent Orchestration Framework - A comprehensive framework for building complex multi-agent systems."""

from .common import (
    Agent,
    AgentType,
    Task,
    Message,
    Result,
    Protocol,
    StateStatus,
    EventEmitter,
    serialize,
    deserialize
)

from .state_machine import (
    StateMachineDesigner,
    WorkflowExecutor,
    State,
    Transition,
    RecoveryPath,
    Workflow,
    TransitionType,
    DecompositionStrategy
)

from .load_balancer import (
    AgentLoadBalancer,
    AgentGroup,
    AgentMetrics,
    Batch,
    RoutingConfig,
    BatchConfig,
    RoutingStrategy,
    BatchStrategy
)

from .consensus import (
    ConsensusEngine,
    Perspective,
    DebateRound,
    ConsensusResult,
    DebateConfig,
    DebatePhase,
    SynthesisMethod
)

from .protocol_gateway import (
    ProtocolGateway,
    ProtocolAdapter,
    MCPAdapter,
    A2AAdapter,
    CustomAdapter,
    HTTPAdapter,
    GatewayProtocol,
    GatewayConfig,
    TranslationResult,
    MCPPayload,
    A2APayload
)

__version__ = "0.1.0"

__all__ = [
    # Common
    "Agent",
    "AgentType",
    "Task",
    "Message",
    "Result",
    "Protocol",
    "StateStatus",
    "EventEmitter",
    "serialize",
    "deserialize",
    
    # State Machine
    "StateMachineDesigner",
    "WorkflowExecutor",
    "State",
    "Transition",
    "RecoveryPath",
    "Workflow",
    "TransitionType",
    "DecompositionStrategy",
    
    # Load Balancer
    "AgentLoadBalancer",
    "AgentGroup",
    "AgentMetrics",
    "Batch",
    "RoutingConfig",
    "BatchConfig",
    "RoutingStrategy",
    "BatchStrategy",
    
    # Consensus
    "ConsensusEngine",
    "Perspective",
    "DebateRound",
    "ConsensusResult",
    "DebateConfig",
    "DebatePhase",
    "SynthesisMethod",
    
    # Protocol Gateway
    "ProtocolGateway",
    "ProtocolAdapter",
    "MCPAdapter",
    "A2AAdapter",
    "CustomAdapter",
    "HTTPAdapter",
    "GatewayProtocol",
    "GatewayConfig",
    "TranslationResult",
    "MCPPayload",
    "A2APayload",
]
