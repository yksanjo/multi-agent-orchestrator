"""Collaboration tracking and interaction graphs."""

from enum import Enum, auto
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import json


class InteractionType(Enum):
    """Types of agent interactions."""
    TASK_DELEGATION = auto()
    TASK_COMPLETION = auto()
    INFORMATION_SHARING = auto()
    COLLABORATION = auto()
    CONFLICT = auto()
    ESCALATION = auto()


@dataclass
class AgentInteraction:
    """Represents an interaction between two agents."""
    interaction_id: str
    agent_a_id: str
    agent_b_id: str
    interaction_type: InteractionType
    task_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True


class CollaborationGraph:
    """Tracks collaboration patterns between agents."""
    
    def __init__(self):
        self.interactions: List[AgentInteraction] = []
        self.agent_connections: Dict[str, Set[str]] = {}  # agent_id -> connected agent_ids
        self.interaction_counts: Dict[Tuple[str, str], int] = {}
    
    def add_interaction(
        self,
        agent_a_id: str,
        agent_b_id: str,
        interaction_type: InteractionType,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        success: bool = True,
    ) -> AgentInteraction:
        """Record an interaction between two agents."""
        interaction = AgentInteraction(
            interaction_id=str(uuid.uuid4()),
            agent_a_id=agent_a_id,
            agent_b_id=agent_b_id,
            interaction_type=interaction_type,
            task_id=task_id,
            metadata=metadata or {},
            success=success,
        )
        
        self.interactions.append(interaction)
        
        # Update connections
        if agent_a_id not in self.agent_connections:
            self.agent_connections[agent_a_id] = set()
        if agent_b_id not in self.agent_connections:
            self.agent_connections[agent_b_id] = set()
        
        self.agent_connections[agent_a_id].add(agent_b_id)
        self.agent_connections[agent_b_id].add(agent_a_id)
        
        # Update counts
        pair = tuple(sorted([agent_a_id, agent_b_id]))
        self.interaction_counts[pair] = self.interaction_counts.get(pair, 0) + 1
        
        return interaction
    
    def get_collaborators(self, agent_id: str) -> Set[str]:
        """Get all agents that have interacted with this agent."""
        return self.agent_connections.get(agent_id, set())
    
    def get_interaction_count(self, agent_a_id: str, agent_b_id: str) -> int:
        """Get the number of interactions between two agents."""
        pair = tuple(sorted([agent_a_id, agent_b_id]))
        return self.interaction_counts.get(pair, 0)
    
    def get_most_collaborative(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get the most collaborative agent pairs."""
        sorted_pairs = sorted(
            self.interaction_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_pairs[:limit]
    
    def get_interaction_history(
        self,
        agent_id: str,
        limit: int = 100,
    ) -> List[AgentInteraction]:
        """Get interaction history for an agent."""
        history = [
            i for i in self.interactions
            if i.agent_a_id == agent_id or i.agent_b_id == agent_id
        ]
        return sorted(
            history,
            key=lambda x: x.timestamp,
            reverse=True,
        )[:limit]
    
    def get_success_rate(self, agent_id: str) -> float:
        """Get success rate for an agent's interactions."""
        history = self.get_interaction_history(agent_id, limit=1000)
        if not history:
            return 0.0
        
        successful = sum(1 for i in history if i.success)
        return successful / len(history)
    
    def export_graph(self) -> Dict[str, Any]:
        """Export the collaboration graph as JSON-serializable dict."""
        return {
            "agents": list(self.agent_connections.keys()),
            "connections": {
                f"{a}-{b}": count
                for (a, b), count in self.interaction_counts.items()
            },
            "total_interactions": len(self.interactions),
        }
    
    def find_collaboration_patterns(self) -> List[Dict[str, Any]]:
        """Analyze and return collaboration patterns."""
        patterns = []
        
        # Find agents that frequently collaborate
        frequent_pairs = [
            (pair, count)
            for pair, count in self.interaction_counts.items()
            if count >= 3
        ]
        
        for pair, count in frequent_pairs:
            patterns.append({
                "type": "frequent_collaboration",
                "agents": list(pair),
                "interaction_count": count,
            })
        
        # Find agents that always succeed
        for agent_id in self.agent_connections.keys():
            if self.get_success_rate(agent_id) >= 0.9:
                patterns.append({
                    "type": "high_success",
                    "agent": agent_id,
                    "success_rate": self.get_success_rate(agent_id),
                })
        
        return patterns


class CollaborationAnalyzer:
    """Analyzes collaboration patterns and effectiveness."""
    
    def __init__(self, graph: CollaborationGraph):
        self.graph = graph
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive collaboration report."""
        return {
            "total_agents": len(self.graph.agent_connections),
            "total_interactions": len(self.graph.interactions),
            "most_collaborative": self.graph.get_most_collaborative(5),
            "patterns": self.graph.find_collaboration_patterns(),
            "graph": self.graph.export_graph(),
        }
    
    def find_collaboration_gaps(self, all_agents: List[str]) -> List[str]:
        """Find agents that aren't collaborating enough."""
        isolated = []
        for agent_id in all_agents:
            collaborators = self.graph.get_collaborators(agent_id)
            if len(collaborators) <= 1:  # Only collaborated with one other agent
                isolated.append(agent_id)
        return isolated
    
    def recommend_collaborators(
        self,
        agent_id: str,
        candidate_agents: List[str],
    ) -> List[Tuple[str, float]]:
        """Recommend potential collaborators based on patterns."""
        recommendations = []
        
        # Find agents that collaborate with agents this agent works with
        agent_collaborators = self.graph.get_collaborators(agent_id)
        
        for candidate in candidate_agents:
            if candidate == agent_id or candidate in agent_collaborators:
                continue
            
            # Calculate similarity score
            candidate_collaborators = self.graph.get_collaborators(candidate)
            common = len(agent_collaborators & candidate_collaborators)
            score = common / max(len(agent_collaborators | candidate_collaborators), 1)
            
            recommendations.append((candidate, score))
        
        return sorted(recommendations, key=lambda x: x[1], reverse=True)
