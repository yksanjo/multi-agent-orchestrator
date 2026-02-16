"""Agent Load Balancer - Geographic routing and dynamic batching system for multi-agent systems."""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import uuid
import heapq

from ..common import Agent, AgentType, Task, Result, EventEmitter


class RoutingStrategy(Enum):
    """Routing strategies for load balancing."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    LATENCY_BASED = "latency_based"
    THROUGHPUT_OPTIMIZED = "throughput_optimized"
    GEOGRAPHIC = "geographic"
    WEIGHTED = "weighted"


class BatchStrategy(Enum):
    """Task batching strategies."""
    FIXED_SIZE = "fixed_size"
    TIME_WINDOW = "time_window"
    DYNAMIC = "dynamic"
    PRIORITY = "priority"


@dataclass
class AgentGroup:
    """Group of agents with similar capabilities."""
    group_id: str
    name: str
    agent_type: AgentType
    agent_ids: List[str] = field(default_factory=list)
    location: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "name": self.name,
            "agent_type": self.agent_type.value,
            "agent_ids": self.agent_ids,
            "location": self.location,
            "metadata": self.metadata,
            "weight": self.weight
        }


@dataclass
class AgentMetrics:
    """Metrics for an agent."""
    agent_id: str
    current_load: int = 0
    max_capacity: int = 10
    avg_latency: float = 0.0
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    last_heartbeat: datetime = field(default_factory=datetime.now)
    success_rate: float = 1.0
    throughput: float = 0.0  # tasks per second
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "current_load": self.current_load,
            "max_capacity": self.max_capacity,
            "avg_latency": self.avg_latency,
            "total_tasks_completed": self.total_tasks_completed,
            "total_tasks_failed": self.total_tasks_failed,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "success_rate": self.success_rate,
            "throughput": self.throughput
        }
    
    @property
    def utilization(self) -> float:
        """Calculate agent utilization."""
        if self.max_capacity == 0:
            return 0.0
        return self.current_load / self.max_capacity
    
    @property
    def is_available(self) -> bool:
        """Check if agent is available."""
        return self.current_load < self.max_capacity


@dataclass
class Batch:
    """A batch of tasks to be processed together."""
    batch_id: str
    tasks: List[Task] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    assigned_agents: List[str] = field(default_factory=list)
    status: str = "pending"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "tasks": [t.to_dict() for t in self.tasks],
            "created_at": self.created_at.isoformat(),
            "assigned_agents": self.assigned_agents,
            "status": self.status,
            "metadata": self.metadata
        }


@dataclass
class RoutingConfig:
    """Configuration for routing decisions."""
    strategy: RoutingStrategy = RoutingStrategy.LEAST_LOADED
    latency_weight: float = 0.5  # Weight for latency vs throughput
    throughput_weight: float = 0.5
    max_latency_threshold: float = 5000.0  # ms
    min_success_rate: float = 0.8
    enable_geo_routing: bool = True
    prefer_local: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "latency_weight": self.latency_weight,
            "throughput_weight": self.throughput_weight,
            "max_latency_threshold": self.max_latency_threshold,
            "min_success_rate": self.min_success_rate,
            "enable_geo_routing": self.enable_geo_routing,
            "prefer_local": self.prefer_local
        }


@dataclass
class BatchConfig:
    """Configuration for task batching."""
    strategy: BatchStrategy = BatchStrategy.DYNAMIC
    fixed_batch_size: int = 10
    time_window_ms: int = 1000
    max_batch_size: int = 50
    min_batch_size: int = 1
    priority_enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "fixed_batch_size": self.fixed_batch_size,
            "time_window_ms": self.time_window_ms,
            "max_batch_size": self.max_batch_size,
            "min_batch_size": self.min_batch_size,
            "priority_enabled": self.priority_enabled
        }


class AgentLoadBalancer:
    """Geographic routing and dynamic batching system for agents."""
    
    def __init__(
        self,
        routing_config: Optional[RoutingConfig] = None,
        batch_config: Optional[BatchConfig] = None
    ):
        self.agent_groups: Dict[str, AgentGroup] = {}
        self.agent_metrics: Dict[str, AgentMetrics] = {}
        self.agents: Dict[str, Agent] = {}
        self.pending_tasks: List[Task] = []
        self.pending_batches: List[Batch] = []
        
        self.routing_config = routing_config or RoutingConfig()
        self.batch_config = batch_config or BatchConfig()
        
        self.events = EventEmitter()
        self._routing_counters: Dict[str, int] = {}  # For round-robin
        
        self._batch_timer: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
    
    def add_agent(self, agent: Agent) -> None:
        """Register an agent with the load balancer."""
        self.agents[agent.agent_id] = agent
        
        # Initialize metrics
        self.agent_metrics[agent.agent_id] = AgentMetrics(
            agent_id=agent.agent_id,
            max_capacity=agent.metadata.get("max_capacity", 10)
        )
        
        self.events.emit("agent_registered", agent)
    
    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent from the load balancer."""
        if agent_id in self.agents:
            agent = self.agents.pop(agent_id)
            self.agent_metrics.pop(agent_id, None)
            
            # Remove from groups
            for group in self.agent_groups.values():
                if agent_id in group.agent_ids:
                    group.agent_ids.remove(agent_id)
            
            self.events.emit("agent_removed", agent)
    
    def add_agent_group(
        self,
        name: str,
        agent_ids: List[str],
        agent_type: AgentType,
        location: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        weight: float = 1.0
    ) -> AgentGroup:
        """Add an agent group."""
        group_id = str(uuid.uuid4())
        group = AgentGroup(
            group_id=group_id,
            name=name,
            agent_type=agent_type,
            agent_ids=agent_ids,
            location=location,
            metadata=metadata or {},
            weight=weight
        )
        
        self.agent_groups[group_id] = group
        
        # Register agents
        for agent_id in agent_ids:
            if agent_id not in self.agents:
                self.add_agent(Agent(
                    agent_id=agent_id,
                    name=agent_id,
                    agent_type=agent_type,
                    location=location
                ))
        
        self.events.emit("group_added", group)
        return group
    
    def get_agent_group(self, group_id: str) -> Optional[AgentGroup]:
        """Get an agent group by ID."""
        return self.agent_groups.get(group_id)
    
    def get_agents_by_type(self, agent_type: AgentType) -> List[Agent]:
        """Get all agents of a specific type."""
        return [a for a in self.agents.values() if a.agent_type == agent_type]
    
    def get_agents_by_location(self, location: str) -> List[Agent]:
        """Get all agents in a specific location."""
        return [a for a in self.agents.values() if a.location == location]
    
    def select_agent(
        self,
        task: Optional[Task] = None,
        preferred_type: Optional[AgentType] = None,
        preferred_location: Optional[str] = None,
        required_capabilities: Optional[List[str]] = None
    ) -> Optional[str]:
        """Select the best agent for a task based on routing strategy."""
        candidates = []
        
        # Filter candidates
        for agent_id, agent in self.agents.items():
            metrics = self.agent_metrics.get(agent_id)
            if not metrics or not metrics.is_available:
                continue
            
            # Filter by type
            if preferred_type and agent.agent_type != preferred_type:
                continue
            
            # Filter by location
            if preferred_location and agent.location != preferred_location:
                continue
            
            # Filter by capabilities
            if required_capabilities:
                if not all(cap in agent.capabilities for cap in required_capabilities):
                    continue
            
            candidates.append((agent_id, agent, metrics))
        
        if not candidates:
            return None
        
        # Apply routing strategy
        if self.routing_config.strategy == RoutingStrategy.ROUND_ROBIN:
            return self._round_robin_select(candidates)
        elif self.routing_config.strategy == RoutingStrategy.LEAST_LOADED:
            return self._least_loaded_select(candidates)
        elif self.routing_config.strategy == RoutingStrategy.LATENCY_BASED:
            return self._latency_based_select(candidates)
        elif self.routing_config.strategy == RoutingStrategy.THROUGHPUT_OPTIMIZED:
            return self._throughput_optimized_select(candidates)
        elif self.routing_config.strategy == RoutingStrategy.WEIGHTED:
            return self._weighted_select(candidates)
        else:
            return candidates[0][0] if candidates else None
    
    def _round_robin_select(
        self,
        candidates: List[tuple]
    ) -> Optional[str]:
        """Round-robin selection."""
        for agent_id, agent, metrics in candidates:
            counter = self._routing_counters.get(agent_id, 0)
            self._routing_counters[agent_id] = counter + 1
            return agent_id
        return None
    
    def _least_loaded_select(
        self,
        candidates: List[tuple]
    ) -> Optional[str]:
        """Select agent with lowest load."""
        return min(candidates, key=lambda x: x[2].utilization)[0]
    
    def _latency_based_select(
        self,
        candidates: List[tuple]
    ) -> Optional[str]:
        """Select agent with lowest latency."""
        return min(candidates, key=lambda x: x[2].avg_latency)[0]
    
    def _throughput_optimized_select(
        self,
        candidates: List[tuple]
    ) -> Optional[str]:
        """Select agent with highest throughput."""
        return max(candidates, key=lambda x: x[2].throughput)[0]
    
    def _weighted_select(
        self,
        candidates: List[tuple]
    ) -> Optional[str]:
        """Weighted selection based on agent group weight."""
        # Get group weights
        weights = []
        for agent_id, agent, metrics in candidates:
            weight = 1.0
            for group in self.agent_groups.values():
                if agent_id in group.agent_ids:
                    weight = group.weight
                    break
            # Adjust by availability
            weight *= (1 - metrics.utilization)
            weights.append(weight)
        
        # Weighted random selection
        total = sum(weights)
        if total == 0:
            return candidates[0][0] if candidates else None
        
        import random
        r = random.uniform(0, total)
        cumsum = 0
        for i, w in enumerate(weights):
            cumsum += w
            if cumsum >= r:
                return candidates[i][0]
        
        return candidates[0][0]
    
    async def submit_task(
        self,
        task: Task,
        callback: Optional[Callable] = None
    ) -> str:
        """Submit a task for processing."""
        async with self._lock:
            self.pending_tasks.append(task)
        
        task_id = task.task_id
        self.events.emit("task_submitted", task)
        
        # Start batch processing if not running
        if self._batch_timer is None:
            self._batch_timer = asyncio.create_task(self._batch_processor())
        
        return task_id
    
    async def submit_tasks(
        self,
        tasks: List[Task],
        callback: Optional[Callable] = None
    ) -> List[str]:
        """Submit multiple tasks."""
        task_ids = []
        for task in tasks:
            task_id = await self.submit_task(task, callback)
            task_ids.append(task_id)
        return task_ids
    
    async def _batch_processor(self) -> None:
        """Process pending tasks into batches."""
        while True:
            await asyncio.sleep(self.batch_config.time_window_ms / 1000.0)
            
            async with self._lock:
                if not self.pending_tasks:
                    continue
                
                # Get tasks to batch
                batch_tasks = []
                if self.batch_config.strategy == BatchStrategy.FIXED_SIZE:
                    batch_tasks = self.pending_tasks[:self.batch_config.fixed_batch_size]
                elif self.batch_config.strategy == BatchStrategy.PRIORITY:
                    batch_tasks = sorted(
                        self.pending_tasks,
                        key=lambda t: t.priority,
                        reverse=True
                    )[:self.batch_config.fixed_batch_size]
                else:
                    # Dynamic - take all pending
                    batch_tasks = self.pending_tasks[:self.batch_config.max_batch_size]
                
                if not batch_tasks:
                    continue
                
                # Remove from pending
                for task in batch_tasks:
                    self.pending_tasks.remove(task)
            
            # Create batch
            batch = Batch(
                batch_id=str(uuid.uuid4()),
                tasks=batch_tasks
            )
            
            # Assign agents to batch
            agents = await self._assign_batch_agents(batch)
            batch.assigned_agents = agents
            
            self.pending_batches.append(batch)
            self.events.emit("batch_created", batch)
            
            # Process batch
            await self._process_batch(batch)
    
    async def _assign_batch_agents(self, batch: Batch) -> List[str]:
        """Assign agents to a batch."""
        assigned = []
        tasks_remaining = len(batch.tasks)
        
        # Sort agents by availability
        sorted_agents = sorted(
            self.agent_metrics.items(),
            key=lambda x: x[1].utilization
        )
        
        for agent_id, metrics in sorted_agents:
            if not metrics.is_available:
                continue
            
            capacity = metrics.max_capacity - metrics.current_load
            agents_needed = min(capacity, tasks_remaining)
            
            for _ in range(agents_needed):
                assigned.append(agent_id)
                metrics.current_load += 1
            
            tasks_remaining -= agents_needed
            
            if tasks_remaining == 0:
                break
        
        return assigned
    
    async def _process_batch(self, batch: Batch) -> None:
        """Process a batch of tasks."""
        batch.status = "processing"
        self.events.emit("batch_started", batch)
        
        # Simulate processing
        await asyncio.sleep(0.1)
        
        # Update metrics
        for agent_id in batch.assigned_agents:
            metrics = self.agent_metrics.get(agent_id)
            if metrics:
                metrics.current_load = max(0, metrics.current_load - len(batch.tasks))
                metrics.total_tasks_completed += len(batch.tasks)
        
        batch.status = "completed"
        self.events.emit("batch_completed", batch)
    
    def update_agent_metrics(
        self,
        agent_id: str,
        latency: Optional[float] = None,
        success: Optional[bool] = None
    ) -> None:
        """Update agent metrics."""
        metrics = self.agent_metrics.get(agent_id)
        if not metrics:
            return
        
        if latency is not None:
            # Update moving average latency
            n = metrics.total_tasks_completed + metrics.total_tasks_failed
            if n > 0:
                metrics.avg_latency = (metrics.avg_latency * n + latency) / (n + 1)
            else:
                metrics.avg_latency = latency
        
        if success is not None:
            if success:
                metrics.total_tasks_completed += 1
            else:
                metrics.total_tasks_failed += 1
            
            # Update success rate
            total = metrics.total_tasks_completed + metrics.total_tasks_failed
            if total > 0:
                metrics.success_rate = metrics.total_tasks_completed / total
        
        metrics.last_heartbeat = datetime.now()
    
    def heartbeat(self, agent_id: str) -> None:
        """Record agent heartbeat."""
        metrics = self.agent_metrics.get(agent_id)
        if metrics:
            metrics.last_heartbeat = datetime.now()
    
    def get_healthy_agents(
        self,
        max_latency: Optional[float] = None,
        min_success_rate: Optional[float] = None
    ) -> List[str]:
        """Get list of healthy agent IDs."""
        max_latency = max_latency or self.routing_config.max_latency_threshold
        min_success_rate = min_success_rate or self.routing_config.min_success_rate
        
        healthy = []
        for agent_id, metrics in self.agent_metrics.items():
            if metrics.avg_latency <= max_latency and metrics.success_rate >= min_success_rate:
                healthy.append(agent_id)
        
        return healthy
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        total_agents = len(self.agents)
        available_agents = len(self.get_healthy_agents())
        
        total_load = sum(m.current_load for m in self.agent_metrics.values())
        total_capacity = sum(m.max_capacity for m in self.agent_metrics.values())
        
        return {
            "total_agents": total_agents,
            "available_agents": available_agents,
            "total_groups": len(self.agent_groups),
            "pending_tasks": len(self.pending_tasks),
            "pending_batches": len(self.pending_batches),
            "total_load": total_load,
            "total_capacity": total_capacity,
            "overall_utilization": total_load / total_capacity if total_capacity > 0 else 0,
            "routing_config": self.routing_config.to_dict(),
            "batch_config": self.batch_config.to_dict(),
            "agent_metrics": {
                agent_id: metrics.to_dict()
                for agent_id, metrics in self.agent_metrics.items()
            }
        }
    
    def set_routing_strategy(self, strategy: RoutingStrategy) -> None:
        """Change routing strategy."""
        self.routing_config.strategy = strategy
    
    def set_batch_strategy(self, strategy: BatchStrategy) -> None:
        """Change batch strategy."""
        self.batch_config.strategy = strategy
