"""Multi-Agent State Machine Designer - Visual workflow builder for defining complex agent collaboration patterns."""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid
import json

from ..common import Agent, Task, StateStatus, EventEmitter


class TransitionType(Enum):
    """Types of state transitions."""
    SUCCESS = "success"
    FAILURE = "failure"
    CONDITIONAL = "conditional"
    MANUAL = "manual"
    TIMEOUT = "timeout"


class DecompositionStrategy(Enum):
    """Task decomposition strategies."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    FANOUT_FANIN = "fanout_fanin"
    PIPELINE = "pipeline"


@dataclass
class State:
    """Represents a state in the workflow."""
    state_id: str
    name: str
    description: str = ""
    agent_assignments: List[str] = field(default_factory=list)  # agent IDs
    task: Optional[Task] = None
    timeout: Optional[int] = None  # seconds
    retry_count: int = 0
    max_retries: int = 3
    on_enter: Optional[str] = None  # callback name
    on_exit: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: StateStatus = StateStatus.PENDING
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_id": self.state_id,
            "name": self.name,
            "description": self.description,
            "agent_assignments": self.agent_assignments,
            "task": self.task.to_dict() if self.task else None,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "on_enter": self.on_enter,
            "on_exit": self.on_exit,
            "metadata": self.metadata,
            "status": self.status.value
        }


@dataclass
class Transition:
    """Represents a transition between states."""
    transition_id: str
    from_state: str
    to_state: str
    transition_type: TransitionType
    condition: Optional[str] = None  # Python expression
    guard: Optional[Callable] = None
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "transition_type": self.transition_type.value,
            "condition": self.condition,
            "description": self.description,
            "metadata": self.metadata
        }


@dataclass
class RecoveryPath:
    """Defines a failure recovery path."""
    path_id: str
    from_state: str
    recovery_state: str
    retry_states: List[str] = field(default_factory=list)
    fallback_state: Optional[str] = None
    max_recovery_attempts: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "path_id": self.path_id,
            "from_state": self.from_state,
            "recovery_state": self.recovery_state,
            "retry_states": self.retry_states,
            "fallback_state": self.fallback_state,
            "max_recovery_attempts": self.max_recovery_attempts
        }


@dataclass
class Workflow:
    """Represents a complete workflow."""
    workflow_id: str
    name: str
    description: str = ""
    states: Dict[str, State] = field(default_factory=dict)
    transitions: List[Transition] = field(default_factory=list)
    recovery_paths: List[RecoveryPath] = field(default_factory=list)
    initial_state: Optional[str] = None
    final_states: List[str] = field(default_factory=list)
    decomposition_strategy: DecompositionStrategy = DecompositionStrategy.SEQUENTIAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "description": self.description,
            "states": {k: v.to_dict() for k, v in self.states.items()},
            "transitions": [t.to_dict() for t in self.transitions],
            "recovery_paths": [r.to_dict() for r in self.recovery_paths],
            "initial_state": self.initial_state,
            "final_states": self.final_states,
            "decomposition_strategy": self.decomposition_strategy.value,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class StateMachineDesigner:
    """Visual workflow builder for defining complex agent collaboration patterns."""
    
    def __init__(self):
        self.workflows: Dict[str, Workflow] = {}
        self.events = EventEmitter()
        self._callbacks: Dict[str, Callable] = {}
    
    def create_workflow(
        self,
        name: str,
        description: str = "",
        decomposition_strategy: DecompositionStrategy = DecompositionStrategy.SEQUENTIAL
    ) -> Workflow:
        """Create a new workflow."""
        workflow_id = str(uuid.uuid4())
        workflow = Workflow(
            workflow_id=workflow_id,
            name=name,
            description=description,
            decomposition_strategy=decomposition_strategy
        )
        self.workflows[workflow_id] = workflow
        self.events.emit("workflow_created", workflow)
        return workflow
    
    def add_state(
        self,
        workflow_id: str,
        name: str,
        description: str = "",
        agent_assignments: List[str] = None,
        task: Optional[Task] = None,
        timeout: Optional[int] = None,
        max_retries: int = 3,
        on_enter: Optional[str] = None,
        on_exit: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ) -> State:
        """Add a state to a workflow."""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        state_id = str(uuid.uuid4())
        state = State(
            state_id=state_id,
            name=name,
            description=description,
            agent_assignments=agent_assignments or [],
            task=task,
            timeout=timeout,
            max_retries=max_retries,
            on_enter=on_enter,
            on_exit=on_exit,
            metadata=metadata or {}
        )
        
        workflow.states[state_id] = state
        workflow.updated_at = datetime.now()
        
        if workflow.initial_state is None:
            workflow.initial_state = state_id
        
        self.events.emit("state_added", workflow, state)
        return state
    
    def add_transition(
        self,
        workflow_id: str,
        from_state: str,
        to_state: str,
        transition_type: TransitionType = TransitionType.SUCCESS,
        condition: Optional[str] = None,
        description: str = ""
    ) -> Transition:
        """Add a transition between states."""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        if from_state not in workflow.states:
            raise ValueError(f"State {from_state} not found")
        if to_state not in workflow.states:
            raise ValueError(f"State {to_state} not found")
        
        transition = Transition(
            transition_id=str(uuid.uuid4()),
            from_state=from_state,
            to_state=to_state,
            transition_type=transition_type,
            condition=condition,
            description=description
        )
        
        workflow.transitions.append(transition)
        workflow.updated_at = datetime.now()
        
        self.events.emit("transition_added", workflow, transition)
        return transition
    
    def add_recovery_path(
        self,
        workflow_id: str,
        from_state: str,
        recovery_state: str,
        retry_states: List[str] = None,
        fallback_state: Optional[str] = None,
        max_recovery_attempts: int = 3
    ) -> RecoveryPath:
        """Add a failure recovery path."""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        recovery_path = RecoveryPath(
            path_id=str(uuid.uuid4()),
            from_state=from_state,
            recovery_state=recovery_state,
            retry_states=retry_states or [],
            fallback_state=fallback_state,
            max_recovery_attempts=max_recovery_attempts
        )
        
        workflow.recovery_paths.append(recovery_path)
        workflow.updated_at = datetime.now()
        
        self.events.emit("recovery_path_added", workflow, recovery_path)
        return recovery_path
    
    def set_final_state(self, workflow_id: str, state_id: str) -> None:
        """Mark a state as final."""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        if state_id not in workflow.states:
            raise ValueError(f"State {state_id} not found")
        
        workflow.final_states.append(state_id)
        workflow.updated_at = datetime.now()
    
    def register_callback(self, name: str, callback: Callable) -> None:
        """Register a callback function."""
        self._callbacks[name] = callback
    
    def execute_callback(self, name: str, *args, **kwargs) -> Any:
        """Execute a registered callback."""
        if name in self._callbacks:
            return self._callbacks[name](*args, **kwargs)
        return None
    
    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get a workflow by ID."""
        return self.workflows.get(workflow_id)
    
    def list_workflows(self) -> List[Workflow]:
        """List all workflows."""
        return list(self.workflows.values())
    
    def export_workflow(self, workflow_id: str, filepath: str) -> None:
        """Export workflow to JSON file."""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        with open(filepath, 'w') as f:
            f.write(workflow.to_json())
    
    def import_workflow(self, filepath: str) -> Workflow:
        """Import workflow from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        workflow = Workflow(
            workflow_id=data["workflow_id"],
            name=data["name"],
            description=data.get("description", ""),
            initial_state=data.get("initial_state"),
            final_states=data.get("final_states", []),
            decomposition_strategy=DecompositionStrategy(
                data.get("decomposition_strategy", "sequential")
            ),
            metadata=data.get("metadata", {})
        )
        
        # Restore states
        for state_id, state_data in data.get("states", {}).items():
            task_data = state_data.get("task")
            task = Task(**task_data) if task_data else None
            workflow.states[state_id] = State(
                state_id=state_id,
                name=state_data["name"],
                description=state_data.get("description", ""),
                agent_assignments=state_data.get("agent_assignments", []),
                task=task,
                timeout=state_data.get("timeout"),
                retry_count=state_data.get("retry_count", 0),
                max_retries=state_data.get("max_retries", 3),
                on_enter=state_data.get("on_enter"),
                on_exit=state_data.get("on_exit"),
                metadata=state_data.get("metadata", {}),
                status=StateStatus(state_data.get("status", "pending"))
            )
        
        # Restore transitions
        for trans_data in data.get("transitions", []):
            workflow.transitions.append(Transition(
                transition_id=trans_data["transition_id"],
                from_state=trans_data["from_state"],
                to_state=trans_data["to_state"],
                transition_type=TransitionType(trans_data["transition_type"]),
                condition=trans_data.get("condition"),
                description=trans_data.get("description", ""),
                metadata=trans_data.get("metadata", {})
            ))
        
        # Restore recovery paths
        for rec_data in data.get("recovery_paths", []):
            workflow.recovery_paths.append(RecoveryPath(
                path_id=rec_data["path_id"],
                from_state=rec_data["from_state"],
                recovery_state=rec_data["recovery_state"],
                retry_states=rec_data.get("retry_states", []),
                fallback_state=rec_data.get("fallback_state"),
                max_recovery_attempts=rec_data.get("max_recovery_attempts", 3)
            ))
        
        self.workflows[workflow.workflow_id] = workflow
        return workflow
    
    def get_state_diagram(self, workflow_id: str) -> Dict[str, Any]:
        """Get state diagram for visualization."""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        nodes = []
        edges = []
        
        for state_id, state in workflow.states.items():
            nodes.append({
                "id": state_id,
                "label": state.name,
                "description": state.description,
                "status": state.status.value,
                "agents": state.agent_assignments
            })
        
        for trans in workflow.transitions:
            edges.append({
                "from": trans.from_state,
                "to": trans.to_state,
                "type": trans.transition_type.value,
                "condition": trans.condition,
                "description": trans.description
            })
        
        return {
            "workflow": workflow.to_dict(),
            "nodes": nodes,
            "edges": edges,
            "initial": workflow.initial_state,
            "final": workflow.final_states
        }


class WorkflowExecutor:
    """Executes workflows created by the StateMachineDesigner."""
    
    def __init__(self, state_machine: StateMachineDesigner):
        self.state_machine = state_machine
        self.events = EventEmitter()
        self._execution_state: Dict[str, str] = {}  # workflow_id -> current_state
    
    async def execute(
        self,
        workflow_id: str,
        agents: Dict[str, Agent],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Execute a workflow."""
        workflow = self.state_machine.get_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        context = context or {}
        execution_log = []
        
        current_state_id = workflow.initial_state
        if not current_state_id:
            raise ValueError("Workflow has no initial state")
        
        self._execution_state[workflow_id] = current_state_id
        
        while current_state_id:
            state = workflow.states.get(current_state_id)
            if not state:
                break
            
            state.status = StateStatus.RUNNING
            execution_log.append({
                "state": state.name,
                "status": "started",
                "timestamp": datetime.now().isoformat()
            })
            
            self.events.emit("state_started", workflow, state, context)
            
            # Execute on_enter callback
            if state.on_enter:
                self.state_machine.execute_callback(state.on_enter, context)
            
            # Execute task with assigned agents
            if state.task and state.agent_assignments:
                for agent_id in state.agent_assignments:
                    agent = agents.get(agent_id)
                    if agent:
                        # Simulate agent execution
                        result = await self._execute_agent_task(agent, state.task)
                        context[f"result_{state.state_id}"] = result
            
            # Check for transitions
            next_state_id = self._find_next_state(workflow, current_state_id, context)
            
            state.status = StateStatus.COMPLETED
            execution_log.append({
                "state": state.name,
                "status": "completed",
                "timestamp": datetime.now().isoformat()
            })
            
            self.events.emit("state_completed", workflow, state, context)
            
            # Execute on_exit callback
            if state.on_exit:
                self.state_machine.execute_callback(state.on_exit, context)
            
            # Check if current state is final
            if current_state_id in workflow.final_states:
                break
            
            current_state_id = next_state_id
            self._execution_state[workflow_id] = current_state_id
        
        return {
            "workflow_id": workflow_id,
            "success": True,
            "execution_log": execution_log,
            "context": context
        }
    
    async def _execute_agent_task(self, agent: Agent, task: Task) -> Any:
        """Execute a task with an agent (placeholder)."""
        # Placeholder for actual agent execution
        return {"agent": agent.agent_id, "task": task.task_id, "status": "executed"}
    
    def _find_next_state(
        self,
        workflow: Workflow,
        current_state_id: str,
        context: Dict[str, Any]
    ) -> Optional[str]:
        """Find the next state based on transitions."""
        for trans in workflow.transitions:
            if trans.from_state == current_state_id:
                # Check condition
                if trans.condition:
                    try:
                        # Evaluate condition
                        result = eval(trans.condition, {"context": context})
                        if not result:
                            continue
                    except Exception:
                        continue
                
                # Check for recovery path
                if trans.transition_type == TransitionType.FAILURE:
                    # Find recovery path
                    for recovery in workflow.recovery_paths:
                        if recovery.from_state == current_state_id:
                            return recovery.recovery_state
                
                return trans.to_state
        
        return None
    
    def get_current_state(self, workflow_id: str) -> Optional[str]:
        """Get the current execution state of a workflow."""
        return self._execution_state.get(workflow_id)
