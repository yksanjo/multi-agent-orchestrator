"""Dynamic team formation and role-based collaboration."""

from enum import Enum, auto
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import uuid


class TeamRole(Enum):
    """Roles within a team."""
    LEADER = auto()
    SPECIALIST = auto()
    COORDINATOR = auto()
    EXECUTOR = auto()
    REVIEWER = auto()


@dataclass
class TeamFormation:
    """Defines how a team should be formed."""
    formation_id: str
    name: str
    required_roles: List[TeamRole]
    required_capabilities: Set[str] = field(default_factory=set)
    min_size: int = 1
    max_size: int = 10
    collaboration_mode: str = "synchronous"


class AgentTeam:
    """A dynamically formed team of agents."""
    
    def __init__(
        self,
        team_id: str = "",
        name: str = "",
        formation: Optional[TeamFormation] = None,
    ):
        self.team_id = team_id or str(uuid.uuid4())
        self.name = name or f"Team-{self.team_id[:8]}"
        self.formation = formation
        self.members: Dict[str, Dict[str, Any]] = {}
        self.leader: Optional[str] = None
        self.created_at = datetime.utcnow()
        self.active = False
    
    def add_member(self, agent_id: str, role: TeamRole, capabilities: List[str]) -> None:
        """Add a member to the team."""
        self.members[agent_id] = {
            "role": role,
            "capabilities": set(capabilities),
            "joined_at": datetime.utcnow(),
        }
        
        # Auto-assign leader if first member
        if not self.leader:
            self.leader = agent_id
            self.members[agent_id]["role"] = TeamRole.LEADER
    
    def remove_member(self, agent_id: str) -> bool:
        """Remove a member from the team."""
        if agent_id in self.members:
            del self.members[agent_id]
            
            # Reassign leader if needed
            if self.leader == agent_id and self.members:
                self.leader = next(iter(self.members))
                self.members[self.leader]["role"] = TeamRole.LEADER
            
            return True
        return False
    
    def get_role(self, agent_id: str) -> Optional[TeamRole]:
        """Get a member's role."""
        member = self.members.get(agent_id)
        return member["role"] if member else None
    
    def get_members_by_role(self, role: TeamRole) -> List[str]:
        """Get all members with a specific role."""
        return [
            aid for aid, info in self.members.items()
            if info["role"] == role
        ]
    
    def has_capability(self, capability: str) -> bool:
        """Check if team has a capability."""
        return any(
            capability in info["capabilities"]
            for info in self.members.values()
        )
    
    def get_capabilities(self) -> Set[str]:
        """Get all capabilities in the team."""
        caps = set()
        for info in self.members.values():
            caps.update(info["capabilities"])
        return caps
    
    def is_complete(self) -> bool:
        """Check if team meets formation requirements."""
        if not self.formation:
            return len(self.members) > 0
        
        # Check size
        if not (self.formation.min_size <= len(self.members) <= self.formation.max_size):
            return False
        
        # Check required roles
        current_roles = {info["role"] for info in self.members.values()}
        for required in self.formation.required_roles:
            if required not in current_roles:
                return False
        
        # Check required capabilities
        team_caps = self.get_capabilities()
        for required in self.formation.required_capabilities:
            if required not in team_caps:
                return False
        
        return True
    
    def activate(self) -> bool:
        """Activate the team."""
        if self.is_complete():
            self.active = True
            return True
        return False
    
    def deactivate(self) -> None:
        """Deactivate the team."""
        self.active = False


class TeamRegistry:
    """Registry for managing multiple teams."""
    
    def __init__(self):
        self.teams: Dict[str, AgentTeam] = {}
        self.agent_teams: Dict[str, Set[str]] = {}  # agent_id -> team_ids
    
    def create_team(
        self,
        name: str,
        formation: Optional[TeamFormation] = None,
    ) -> AgentTeam:
        """Create a new team."""
        team = AgentTeam(name=name, formation=formation)
        self.teams[team.team_id] = team
        return team
    
    def disband_team(self, team_id: str) -> bool:
        """Disband a team."""
        if team_id in self.teams:
            team = self.teams[team_id]
            
            # Remove team from all members
            for agent_id in team.members:
                if agent_id in self.agent_teams:
                    self.agent_teams[agent_id].discard(team_id)
            
            del self.teams[team_id]
            return True
        return False
    
    def register_agent(self, agent_id: str, team_id: str) -> bool:
        """Register an agent to a team."""
        team = self.teams.get(team_id)
        if not team:
            return False
        
        if agent_id not in team.members:
            return False
        
        if agent_id not in self.agent_teams:
            self.agent_teams[agent_id] = set()
        self.agent_teams[agent_id].add(team_id)
        return True
    
    def get_agent_teams(self, agent_id: str) -> List[AgentTeam]:
        """Get all teams an agent belongs to."""
        team_ids = self.agent_teams.get(agent_id, set())
        return [self.teams[tid] for tid in team_ids if tid in self.teams]
    
    def find_teams_for_capability(self, capability: str) -> List[AgentTeam]:
        """Find teams with a specific capability."""
        return [
            team for team in self.teams.values()
            if team.has_capability(capability) and team.active
        ]
