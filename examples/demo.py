#!/usr/bin/env python3
"""Demo script showcasing all four orchestration and coordination systems."""

import asyncio
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (
    # Common
    Agent, AgentType, Task, Message, Protocol,
    
    # State Machine
    StateMachineDesigner, WorkflowExecutor, TransitionType, DecompositionStrategy,
    
    # Load Balancer
    AgentLoadBalancer, RoutingStrategy, BatchStrategy,
    
    # Consensus
    ConsensusEngine, SynthesisMethod,
    
    # Protocol Gateway
    ProtocolGateway, GatewayProtocol
)


async def demo_state_machine():
    """Demo the Multi-Agent State Machine Designer."""
    print("\n" + "="*60)
    print("1. Multi-Agent State Machine Designer Demo")
    print("="*60)
    
    # Create a state machine designer
    sm = StateMachineDesigner()
    
    # Create a data processing workflow
    workflow = sm.create_workflow(
        name="Data Processing Pipeline",
        description="Multi-stage data processing with error recovery",
        decomposition_strategy=DecompositionStrategy.PIPELINE
    )
    
    # Add states
    ingest = sm.add_state(
        workflow_id=workflow.workflow_id,
        name="Ingest",
        description="Ingest raw data from source",
        agent_assignments=["agent-1"],
        metadata={"timeout": 60}
    )
    
    validate = sm.add_state(
        workflow_id=workflow.workflow_id,
        name="Validate",
        description="Validate data integrity",
        agent_assignments=["agent-2"],
        metadata={"timeout": 30}
    )
    
    process = sm.add_state(
        workflow_id=workflow.workflow_id,
        name="Process",
        description="Process and transform data",
        agent_assignments=["agent-3", "agent-4"],
        metadata={"timeout": 120}
    )
    
    analyze = sm.add_state(
        workflow_id=workflow.workflow_id,
        name="Analyze",
        description="Analyze processed data",
        agent_assignments=["agent-5"],
        metadata={"timeout": 60}
    )
    
    error_recovery = sm.add_state(
        workflow_id=workflow.workflow_id,
        name="Error Recovery",
        description="Handle errors and retry",
        agent_assignments=["agent-1"]
    )
    
    # Add transitions
    sm.add_transition(
        workflow_id=workflow.workflow_id,
        from_state=ingest.state_id,
        to_state=validate.state_id,
        transition_type=TransitionType.SUCCESS,
        description="Data validated successfully"
    )
    
    sm.add_transition(
        workflow_id=workflow.workflow_id,
        from_state=validate.state_id,
        to_state=process.state_id,
        transition_type=TransitionType.SUCCESS,
        description="Validation passed"
    )
    
    sm.add_transition(
        workflow_id=workflow.workflow_id,
        from_state=process.state_id,
        to_state=analyze.state_id,
        transition_type=TransitionType.SUCCESS,
        description="Processing complete"
    )
    
    # Add failure transitions
    sm.add_transition(
        workflow_id=workflow.workflow_id,
        from_state=ingest.state_id,
        to_state=error_recovery.state_id,
        transition_type=TransitionType.FAILURE,
        description="Ingestion failed"
    )
    
    sm.add_transition(
        workflow_id=workflow.workflow_id,
        from_state=validate.state_id,
        to_state=error_recovery.state_id,
        transition_type=TransitionType.FAILURE,
        description="Validation failed"
    )
    
    # Add recovery path
    sm.add_recovery_path(
        workflow_id=workflow.workflow_id,
        from_state=ingest.state_id,
        recovery_state=ingest.state_id,
        retry_states=[ingest.state_id],
        fallback_state=error_recovery.state_id,
        max_recovery_attempts=3
    )
    
    # Mark final state
    sm.set_final_state(workflow_id=workflow.workflow_id, state_id=analyze.state_id)
    
    # Get state diagram for visualization
    diagram = sm.get_state_diagram(workflow.workflow_id)
    
    print(f"\nWorkflow: {workflow.name}")
    print(f"Description: {workflow.description}")
    print(f"States: {len(diagram['nodes'])}")
    print(f"Transitions: {len(diagram['edges'])}")
    print(f"Initial State: {diagram['initial']}")
    print(f"Final States: {diagram['final']}")
    
    print("\nState Diagram Nodes:")
    for node in diagram['nodes']:
        print(f"  - {node['label']}: {node['status']}")
    
    print("\nState Diagram Edges:")
    for edge in diagram['edges']:
        print(f"  - {edge['from']} -> {edge['to']} ({edge['type']})")
    
    # Export workflow
    sm.export_workflow(workflow.workflow_id, "/tmp/workflow.json")
    print("\n✓ Workflow exported to /tmp/workflow.json")
    
    return workflow


async def demo_load_balancer():
    """Demo the Agent Load Balancer."""
    print("\n" + "="*60)
    print("2. Agent Load Balancer Demo")
    print("="*60)
    
    # Create load balancer with custom config
    lb = AgentLoadBalancer(
        routing_config=None,  # Uses default
        batch_config=None
    )
    
    # Create agents with different types
    agents = [
        Agent(agent_id="gpu-agent-1", name="GPU Agent 1", agent_type=AgentType.NVIDIA_GPU, 
              location="us-east", capabilities=["inference", "training"]),
        Agent(agent_id="gpu-agent-2", name="GPU Agent 2", agent_type=AgentType.NVIDIA_GPU,
              location="us-west", capabilities=["inference"]),
        Agent(agent_id="trainium-agent-1", name="Trainium Agent 1", agent_type=AgentType.AWS_TRAINIUM,
              location="eu-west", capabilities=["training"]),
        Agent(agent_id="tpu-agent-1", name="TPU Agent 1", agent_type=AgentType.GOOGLE_TPU,
              location="asia-east", capabilities=["inference", "training"]),
        Agent(agent_id="cpu-agent-1", name="CPU Agent 1", agent_type=AgentType.CPU,
              location="us-east", capabilities=["preprocessing"]),
    ]
    
    # Register agents
    for agent in agents:
        lb.add_agent(agent)
    
    # Create agent groups
    lb.add_agent_group(
        name="GPU Cluster",
        agent_ids=["gpu-agent-1", "gpu-agent-2"],
        agent_type=AgentType.NVIDIA_GPU,
        location="us-west",
        weight=2.0
    )
    
    lb.add_agent_group(
        name="Trainium Cluster",
        agent_ids=["trainium-agent-1"],
        agent_type=AgentType.AWS_TRAINIUM,
        location="eu-west",
        weight=1.5
    )
    
    lb.add_agent_group(
        name="TPU Cluster",
        agent_ids=["tpu-agent-1"],
        agent_type=AgentType.GOOGLE_TPU,
        location="asia-east",
        weight=1.5
    )
    
    print(f"\nRegistered {len(agents)} agents")
    print(f"Created {len(lb.agent_groups)} agent groups")
    
    # Test different routing strategies
    print("\n--- Testing Routing Strategies ---")
    
    # Round Robin
    lb.set_routing_strategy(RoutingStrategy.ROUND_ROBIN)
    selected = lb.select_agent(preferred_type=AgentType.NVIDIA_GPU)
    print(f"Round Robin selected: {selected}")
    
    # Least Loaded
    lb.set_routing_strategy(RoutingStrategy.LEAST_LOADED)
    selected = lb.select_agent(preferred_type=AgentType.NVIDIA_GPU)
    print(f"Least Loaded selected: {selected}")
    
    # Latency Based
    lb.set_routing_strategy(RoutingStrategy.LATENCY_BASED)
    selected = lb.select_agent(preferred_type=AgentType.NVIDIA_GPU)
    print(f"Latency Based selected: {selected}")
    
    # Submit tasks
    print("\n--- Submitting Tasks ---")
    tasks = [
        Task(task_id=f"task-{i}", description=f"Task {i}", priority=i % 3)
        for i in range(10)
    ]
    
    task_ids = await lb.submit_tasks(tasks)
    print(f"Submitted {len(task_ids)} tasks")
    
    # Wait for batch processing
    await asyncio.sleep(2)
    
    # Get statistics
    stats = lb.get_statistics()
    print(f"\nLoad Balancer Statistics:")
    print(f"  Total Agents: {stats['total_agents']}")
    print(f"  Available Agents: {stats['available_agents']}")
    print(f"  Pending Tasks: {stats['pending_tasks']}")
    print(f"  Overall Utilization: {stats['overall_utilization']:.2%}")
    
    return lb


async def demo_consensus_engine():
    """Demo the Consensus Engine."""
    print("\n" + "="*60)
    print("3. Consensus Engine Demo")
    print("="*60)
    
    # Create consensus engine
    ce = ConsensusEngine()
    
    # Create agents for deliberation
    agents = [
        Agent(agent_id="analyst-1", name="Analyst 1", agent_type=AgentType.CPU,
              capabilities=["analysis"]),
        Agent(agent_id="analyst-2", name="Analyst 2", agent_type=AgentType.CPU,
              capabilities=["analysis"]),
        Agent(agent_id="expert-1", name="Domain Expert", agent_type=AgentType.NVIDIA_GPU,
              capabilities=["expert-knowledge"]),
    ]
    
    print(f"\nCreated {len(agents)} agents for deliberation")
    
    # Question for deliberation
    question = "What is the best approach for scaling our agent infrastructure?"
    
    # Delegate and deliberate
    print(f"\nQuestion: {question}")
    print("Running deliberation...\n")
    
    result = await ce.delegate_and_deliberate(question, agents)
    
    print(f"Consensus Result:")
    print(f"  Success: {result.success}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Agreement Level: {result.agreement_level:.2f}")
    print(f"  Synthesis: {result.synthesis[:100]}...")
    
    print(f"\n  Perspectives collected: {len(result.perspectives)}")
    for p in result.perspectives:
        print(f"    - Agent {p.agent_id}: confidence={p.confidence:.2f}")
    
    print(f"\n  Debate rounds: {len(result.debate_rounds)}")
    for r in result.debate_rounds:
        print(f"    - {r.phase.value}: {len(r.statements)} statements")
    
    if result.dissenting_opinions:
        print(f"\n  Dissenting opinions: {len(result.dissenting_opinions)}")
    
    # Quick consensus (without full debate)
    print("\n--- Quick Consensus ---")
    quick_result = await ce.quick_consensus(question, agents)
    print(f"Quick consensus confidence: {quick_result.confidence:.2f}")
    
    return ce


async def demo_protocol_gateway():
    """Demo the Protocol Gateway."""
    print("\n" + "="*60)
    print("4. Agent-to-Agent Protocol Gateway Demo")
    print("="*60)
    
    # Create protocol gateway
    pg = ProtocolGateway()
    
    # Register agents with different native protocols
    agents = [
        Agent(agent_id="mcp-agent", name="MCP Agent", agent_type=AgentType.CPU,
              protocol=Protocol.MCP, location="us-east"),
        Agent(agent_id="a2a-agent", name="A2A Agent", agent_type=AgentType.CPU,
              protocol=Protocol.A2A, location="us-west"),
        Agent(agent_id="custom-agent", name="Custom Agent", agent_type=AgentType.CPU,
              protocol=Protocol.CUSTOM, location="eu-west"),
    ]
    
    for agent in agents:
        pg.register_agent(agent)
    
    print(f"\nRegistered {len(agents)} agents")
    print("Available adapters:")
    for protocol, adapter in pg.adapters.items():
        print(f"  - {protocol.value}: {adapter.name}")
    
    # Connect agents
    print("\n--- Connecting Agents ---")
    await pg.connect_agent("mcp-agent", GatewayProtocol.MCP)
    await pg.connect_agent("a2a-agent", GatewayProtocol.A2A)
    await pg.connect_agent("custom-agent", GatewayProtocol.CUSTOM)
    
    # Send message from MCP agent to A2A agent (protocol translation)
    print("\n--- Sending Messages (with Translation) ---")
    
    message = Message(
        message_id="msg-1",
        sender="mcp-agent",
        receiver="a2a-agent",
        content="Hello from MCP agent!",
        protocol=Protocol.MCP
    )
    
    result = await pg.send_message(message, "a2a-agent", GatewayProtocol.A2A)
    print(f"MCP -> A2A: {result.success}")
    
    # Send message from A2A to Custom
    message2 = Message(
        message_id="msg-2",
        sender="a2a-agent",
        receiver="custom-agent",
        content="Hello from A2A agent!",
        protocol=Protocol.A2A
    )
    
    result2 = await pg.send_message(message2, "custom-agent", GatewayProtocol.CUSTOM)
    print(f"A2A -> Custom: {result2.success}")
    
    # Translate message
    print("\n--- Message Translation ---")
    translated = await pg.translate_message(message, GatewayProtocol.CUSTOM)
    print(f"Translated from {translated.source_protocol.value} to {translated.target_protocol.value}")
    print(f"Original message ID: {translated.original_message.message_id}")
    print(f"Translated message ID: {translated.translated_message.message_id}")
    
    # Broadcast message
    print("\n--- Broadcasting ---")
    broadcast_msg = Message(
        message_id="broadcast-1",
        sender="system",
        receiver="all",
        content="System-wide announcement",
        protocol=Protocol.CUSTOM
    )
    
    broadcast_results = await pg.broadcast(
        broadcast_msg,
        ["mcp-agent", "a2a-agent", "custom-agent"]
    )
    
    print(f"Broadcast results:")
    for agent_id, result in broadcast_results.items():
        print(f"  - {agent_id}: {result.success}")
    
    # Get statistics
    stats = pg.get_statistics()
    print(f"\nGateway Statistics:")
    print(f"  Registered Agents: {stats['registered_agents']}")
    print(f"  Active Connections: {stats['active_connections']}")
    print(f"  Registered Adapters: {stats['registered_adapters']}")
    
    return pg


async def demo_integration():
    """Demo all systems working together."""
    print("\n" + "="*60)
    print("INTEGRATION DEMO: All Systems Working Together")
    print("="*60)
    
    # Initialize all systems
    sm = StateMachineDesigner()
    lb = AgentLoadBalancer()
    ce = ConsensusEngine()
    pg = ProtocolGateway()
    
    # Create shared agents
    agents = [
        Agent(agent_id="agent-1", name="Agent 1", agent_type=AgentType.NVIDIA_GPU,
              location="us-east", protocol=Protocol.MCP),
        Agent(agent_id="agent-2", name="Agent 2", agent_type=AgentType.AWS_TRAINIUM,
              location="us-west", protocol=Protocol.A2A),
    ]
    
    # Register agents with all systems
    for agent in agents:
        lb.add_agent(agent)
        pg.register_agent(agent)
    
    print("\n✓ Systems initialized")
    print("✓ Agents registered with Load Balancer and Protocol Gateway")
    
    # Use consensus to decide on workflow
    question = "Should we use parallel or sequential task decomposition?"
    result = await ce.delegate_and_deliberate(question, agents)
    print(f"\n✓ Consensus reached: {result.synthesis[:80]}...")
    
    # Select agent using load balancer
    selected = lb.select_agent(preferred_type=AgentType.NVIDIA_GPU)
    print(f"✓ Load balancer selected: {selected}")
    
    # Send task to agent via protocol gateway
    task_msg = Message(
        message_id="task-msg-1",
        sender="orchestrator",
        receiver=selected,
        content="Execute workflow task",
        protocol=Protocol.CUSTOM
    )
    
    send_result = await pg.send_message(task_msg, selected)
    print(f"✓ Task sent via protocol gateway: {send_result.success}")
    
    print("\n" + "="*60)
    print("All systems integrated successfully!")
    print("="*60)


async def main():
    """Run all demos."""
    print("\n" + "#"*60)
    print("# Multi-Agent Orchestration Framework Demo")
    print("#"*60)
    
    # Run individual demos
    await demo_state_machine()
    await demo_load_balancer()
    await demo_consensus_engine()
    await demo_protocol_gateway()
    
    # Run integration demo
    await demo_integration()
    
    print("\n" + "#"*60)
    print("# All demos completed successfully!")
    print("#"*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
