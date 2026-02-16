"""
Multi-Agent Orchestrator

An advanced orchestration system for managing complex multi-agent interactions,
featuring dynamic team formation, role-based collaboration, and inter-agent communication.
"""

from hierarchical_agent_coordinator import (
    HierarchicalCoordinator,
    ManagerAgent,
    WorkerAgent,
    AgentRole,
    Task,
    TaskResult,
    DelegationMode,
    ResultAggregation,
    AuditTrail,
    AuditLevel,
)

from .team import AgentTeam, TeamFormation, TeamRole
from .communication import Message, MessageBus, AgentProtocol
from .collaboration import CollaborationGraph, AgentInteraction

__version__ = "1.0.0"
__all__ = [
    "HierarchicalCoordinator",
    "ManagerAgent",
    "WorkerAgent",
    "AgentRole",
    "Task",
    "TaskResult",
    "DelegationMode",
    "ResultAggregation",
    "AuditTrail",
    "AuditLevel",
    "AgentTeam",
    "TeamFormation",
    "TeamRole",
    "Message",
    "MessageBus",
    "AgentProtocol",
    "CollaborationGraph",
    "AgentInteraction",
]
