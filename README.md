# Multi-Agent Orchestrator

An advanced orchestration system for managing complex multi-agent interactions, featuring dynamic team formation, role-based collaboration, and inter-agent communication.

## Features

- **Dynamic Team Formation**: Create and manage agent teams with defined roles
- **Role-Based Collaboration**: LEADER, SPECIALIST, COORDINATOR, EXECUTOR, REVIEWER roles
- **Inter-Agent Communication**: Message bus with pub/sub patterns
- **Collaboration Tracking**: Graph-based collaboration analysis
- **Protocol Support**: REQUEST, RESPONSE, NOTIFICATION, BROADCAST, ESCALATION

## Installation

```bash
pip install -e .
```

## Usage

```python
from multi_agent_orchestrator import (
    AgentTeam,
    TeamFormation,
    TeamRole,
    MessageBus,
    AgentProtocol,
    CollaborationGraph,
)

# Create team
formation = TeamFormation(
    formation_id="team_001",
    name="Analysis Team",
    required_roles=[TeamRole.LEADER, TeamRole.SPECIALIST],
    required_capabilities={"data_analysis", "reporting"},
)

team = AgentTeam(name="Analysis Team", formation=formation)
team.add_member("agent_1", TeamRole.LEADER, ["data_analysis"])
team.add_member("agent_2", TeamRole.SPECIALIST, ["data_analysis", "reporting"])

# Message bus
bus = MessageBus()
bus.send_message("agent_1", "agent_2", MessageType.REQUEST, "Analyze this data")

# Collaboration tracking
graph = CollaborationGraph()
graph.add_interaction("agent_1", "agent_2", InteractionType.COLLABORATION)
```

## License

MIT
