"""Consensus Engine - Coordinates multiple agent perspectives on the same problem."""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import asyncio
import uuid
import json

from ..common import Agent, Task, Result, EventEmitter


class DebatePhase(Enum):
    """Phases of structured debate."""
    OPENING = "opening"
    ARGUMENT = "argument"
    REBUTTAL = "rebuttal"
    SYNTHESIS = "synthesis"
    VOTING = "voting"
    CONCLUSION = "conclusion"


class SynthesisMethod(Enum):
    """Methods for synthesizing competing interpretations."""
    WEIGHTED_AVERAGE = "weighted_average"
    MAJORITY_VOTE = "majority_vote"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    HIERARCHICAL = "hierarchical"
    ROUND_ROBIN = "round_robin"


@dataclass
class Perspective:
    """Represents an agent's perspective on a problem."""
    perspective_id: str
    agent_id: str
    interpretation: str
    confidence: float = 0.0
    supporting_evidence: List[str] = field(default_factory=list)
    counter_arguments: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "perspective_id": self.perspective_id,
            "agent_id": self.agent_id,
            "interpretation": self.interpretation,
            "confidence": self.confidence,
            "supporting_evidence": self.supporting_evidence,
            "counter_arguments": self.counter_arguments,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class DebateRound:
    """A single round of debate."""
    round_id: str
    phase: DebatePhase
    participant_ids: List[str] = field(default_factory=list)
    statements: Dict[str, str] = field(default_factory=dict)  # agent_id -> statement
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_id": self.round_id,
            "phase": self.phase.value,
            "participant_ids": self.participant_ids,
            "statements": self.statements,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class ConsensusResult:
    """Result of a consensus operation."""
    success: bool
    synthesis: str = ""
    confidence: float = 0.0
    perspectives: List[Perspective] = field(default_factory=list)
    debate_rounds: List[DebateRound] = field(default_factory=list)
    votes: Dict[str, float] = field(default_factory=dict)  # agent_id -> vote
    agreement_level: float = 0.0  # 0-1 scale
    dissenting_opinions: List[str] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "synthesis": self.synthesis,
            "confidence": self.confidence,
            "perspectives": [p.to_dict() for p in self.perspectives],
            "debate_rounds": [r.to_dict() for r in self.debate_rounds],
            "votes": self.votes,
            "agreement_level": self.agreement_level,
            "dissenting_opinions": self.dissenting_opinions,
            "error": self.error,
            "metadata": self.metadata
        }


@dataclass
class DebateConfig:
    """Configuration for structured debate."""
    max_rounds: int = 5
    min_participants: int = 2
    confidence_threshold: float = 0.7
    agreement_threshold: float = 0.8
    enable_rebuttal: bool = True
    enable_synthesis: bool = True
    synthesis_method: SynthesisMethod = SynthesisMethod.CONFIDENCE_WEIGHTED
    timeout_seconds: float = 60.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_rounds": self.max_rounds,
            "min_participants": self.min_participants,
            "confidence_threshold": self.confidence_threshold,
            "agreement_threshold": self.agreement_threshold,
            "enable_rebuttal": self.enable_rebuttal,
            "enable_synthesis": self.enable_synthesis,
            "synthesis_method": self.synthesis_method.value,
            "timeout_seconds": self.timeout_seconds
        }


class ConsensusEngine:
    """Coordinates multiple agent perspectives with structured debate protocols."""
    
    def __init__(self, config: Optional[DebateConfig] = None):
        self.config = config or DebateConfig()
        self.events = EventEmitter()
        self._active_debates: Dict[str, List[DebateRound]] = {}
        self._perspectives: Dict[str, List[Perspective]] = {}
    
    async def delegate_and_deliberate(
        self,
        question: str,
        agents: List[Agent],
        context: Optional[Dict[str, Any]] = None
    ) -> ConsensusResult:
        """Main entry point: delegate question to agents and deliberate."""
        if len(agents) < self.config.min_participants:
            return ConsensusResult(
                success=False,
                error=f"Need at least {self.config.min_participants} agents"
            )
        
        context = context or {}
        perspectives = []
        
        # Phase 1: Collect initial perspectives
        self.events.emit("deliberation_started", question, agents)
        
        for agent in agents:
            perspective = await self._collect_perspective(agent, question, context)
            perspectives.append(perspective)
        
        # Phase 2: Structured debate (optional)
        debate_rounds = []
        if self.config.enable_rebuttal and len(agents) > 1:
            debate_rounds = await self._run_debate(question, perspectives, agents)
        
        # Phase 3: Synthesis
        result = self._synthesize(perspectives, debate_rounds, agents)
        
        self.events.emit("deliberation_completed", result)
        return result
    
    async def _collect_perspective(
        self,
        agent: Agent,
        question: str,
        context: Dict[str, Any]
    ) -> Perspective:
        """Collect an agent's perspective on the question."""
        perspective_id = str(uuid.uuid4())
        
        # Simulate agent reasoning (placeholder)
        # In production, this would call the actual agent
        interpretation = f"Perspective from {agent.name}: {question}"
        confidence = 0.75  # Placeholder
        
        perspective = Perspective(
            perspective_id=perspective_id,
            agent_id=agent.agent_id,
            interpretation=interpretation,
            confidence=confidence,
            supporting_evidence=["evidence1", "evidence2"],
            counter_arguments=[],
            metadata={"agent_name": agent.name}
        )
        
        self.events.emit("perspective_collected", perspective)
        return perspective
    
    async def _run_debate(
        self,
        question: str,
        perspectives: List[Perspective],
        agents: List[Agent]
    ) -> List[DebateRound]:
        """Run structured debate between agents."""
        debate_rounds = []
        participant_ids = [a.agent_id for a in agents]
        
        # Opening phase
        round_id = str(uuid.uuid4())
        opening_statements = {p.agent_id: p.interpretation for p in perspectives}
        
        debate_round = DebateRound(
            round_id=round_id,
            phase=DebatePhase.OPENING,
            participant_ids=participant_ids,
            statements=opening_statements
        )
        debate_rounds.append(debate_round)
        self.events.emit("debate_round", debate_round)
        
        # Argument and rebuttal phases
        for round_num in range(1, self.config.max_rounds):
            # Argument phase
            round_id = str(uuid.uuid4())
            arguments = {}
            for perspective in perspectives:
                arguments[perspective.agent_id] = (
                    f"Argument from {perspective.agent_id} in round {round_num}"
                )
            
            debate_round = DebateRound(
                round_id=round_id,
                phase=DebatePhase.ARGUMENT,
                participant_ids=participant_ids,
                statements=arguments
            )
            debate_rounds.append(debate_round)
            self.events.emit("debate_round", debate_round)
            
            # Rebuttal phase (if enabled)
            if self.config.enable_rebuttal:
                round_id = str(uuid.uuid4())
                rebuttals = {}
                for perspective in perspectives:
                    rebuttals[perspective.agent_id] = (
                        f"Rebuttal from {perspective.agent_id} in round {round_num}"
                    )
                
                debate_round = DebateRound(
                    round_id=round_id,
                    phase=DebatePhase.REBUTTAL,
                    participant_ids=participant_ids,
                    statements=rebuttals
                )
                debate_rounds.append(debate_round)
                self.events.emit("debate_round", debate_round)
        
        # Synthesis phase
        round_id = str(uuid.uuid4())
        synthesis_statements = {
            p.agent_id: self._generate_synthesis(perspectives)
            for p in perspectives
        }
        
        debate_round = DebateRound(
            round_id=round_id,
            phase=DebatePhase.SYNTHESIS,
            participant_ids=participant_ids,
            statements=synthesis_statements
        )
        debate_rounds.append(debate_round)
        self.events.emit("debate_round", debate_round)
        
        return debate_rounds
    
    def _generate_synthesis(self, perspectives: List[Perspective]) -> str:
        """Generate synthesis statement."""
        if not perspectives:
            return ""
        
        high_confidence = [p for p in perspectives if p.confidence >= self.config.confidence_threshold]
        
        if high_confidence:
            best = max(high_confidence, key=lambda p: p.confidence)
            return f"Synthesis: {best.interpretation}"
        
        return f"Synthesis of {len(perspectives)} perspectives"
    
    def _synthesize(
        self,
        perspectives: List[Perspective],
        debate_rounds: List[DebateRound],
        agents: List[Agent]
    ) -> ConsensusResult:
        """Synthesize perspectives into a final result."""
        if not perspectives:
            return ConsensusResult(success=False, error="No perspectives to synthesize")
        
        # Calculate agreement level
        agreement_level = self._calculate_agreement(perspectives)
        
        # Generate synthesis based on method
        if self.config.synthesis_method == SynthesisMethod.WEIGHTED_AVERAGE:
            synthesis = self._weighted_average_synthesis(perspectives)
            confidence = sum(p.confidence for p in perspectives) / len(perspectives)
        elif self.config.synthesis_method == SynthesisMethod.MAJORITY_VOTE:
            synthesis, votes = self._majority_vote_synthesis(perspectives, agents)
            confidence = max(votes.values()) / sum(votes.values()) if votes else 0.0
        elif self.config.synthesis_method == SynthesisMethod.CONFIDENCE_WEIGHTED:
            synthesis, confidence = self._confidence_weighted_synthesis(perspectives)
        else:
            synthesis = perspectives[0].interpretation
            confidence = perspectives[0].confidence
        
        # Identify dissenting opinions
        dissenting = self._identify_dissenting(perspectives, agreement_level)
        
        return ConsensusResult(
            success=True,
            synthesis=synthesis,
            confidence=confidence,
            perspectives=perspectives,
            debate_rounds=debate_rounds,
            agreement_level=agreement_level,
            dissenting_opinions=dissenting,
            metadata={"method": self.config.synthesis_method.value}
        )
    
    def _calculate_agreement(self, perspectives: List[Perspective]) -> float:
        """Calculate how much agents agree (0-1 scale)."""
        if len(perspectives) < 2:
            return 1.0
        
        # Simple agreement based on confidence variance
        confidences = [p.confidence for p in perspectives]
        avg_conf = sum(confidences) / len(confidences)
        
        if avg_conf == 0:
            return 0.0
        
        # Agreement is inverse of standard deviation normalized
        variance = sum((c - avg_conf) ** 2 for c in confidences) / len(confidences)
        std_dev = variance ** 0.5
        
        agreement = 1.0 - min(std_dev, 1.0)
        return agreement
    
    def _weighted_average_synthesis(self, perspectives: List[Perspective]) -> str:
        """Weighted average synthesis."""
        weights = [p.confidence for p in perspectives]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return perspectives[0].interpretation if perspectives else ""
        
        # Simple weighted combination
        return " / ".join(p.interpretation for p in perspectives)
    
    def _majority_vote_synthesis(
        self,
        perspectives: List[Perspective],
        agents: List[Agent]
    ) -> tuple[str, Dict[str, float]]:
        """Majority vote synthesis."""
        votes = {a.agent_id: 1.0 / len(agents) for a in agents}
        
        # In a real implementation, agents would vote on options
        # Here we just use confidence as a proxy
        best_perspective = max(perspectives, key=lambda p: p.confidence)
        
        return best_perspective.interpretation, votes
    
    def _confidence_weighted_synthesis(
        self,
        perspectives: List[Perspective]
    ) -> tuple[str, float]:
        """Confidence-weighted synthesis."""
        if not perspectives:
            return "", 0.0
        
        # Weight by confidence
        total_confidence = sum(p.confidence for p in perspectives)
        
        if total_confidence == 0:
            return perspectives[0].interpretation, 0.0
        
        # Get highest confidence perspective as primary
        best = max(perspectives, key=lambda p: p.confidence)
        
        # Combine with weights
        synthesis_parts = []
        for p in perspectives:
            weight = p.confidence / total_confidence
            if weight > 0.2:  # Only include significant contributions
                synthesis_parts.append(p.interpretation[:100])
        
        synthesis = f"Synthesized ({len(synthesis_parts)} perspectives): " + " | ".join(synthesis_parts[:3])
        
        return synthesis, best.confidence
    
    def _identify_dissenting(
        self,
        perspectives: List[Perspective],
        agreement_level: float
    ) -> List[str]:
        """Identify dissenting opinions."""
        if agreement_level >= self.config.agreement_threshold:
            return []
        
        # Find perspectives that significantly differ
        avg_confidence = sum(p.confidence for p in perspectives) / len(perspectives)
        dissenting = []
        
        for p in perspectives:
            if p.confidence < avg_confidence * 0.7:
                dissenting.append(p.interpretation)
        
        return dissenting[:3]  # Limit to top 3
    
    async def quick_consensus(
        self,
        question: str,
        agents: List[Agent]
    ) -> ConsensusResult:
        """Quick consensus without full debate."""
        original_rebuttal = self.config.enable_rebuttal
        self.config.enable_rebuttal = False
        
        result = await self.delegate_and_deliberate(question, agents)
        
        self.config.enable_rebuttal = original_rebuttal
        return result
    
    def get_confidence_score(
        self,
        perspectives: List[Perspective]
    ) -> float:
        """Calculate overall confidence score."""
        if not perspectives:
            return 0.0
        
        # Weighted by agreement
        agreement = self._calculate_agreement(perspectives)
        avg_confidence = sum(p.confidence for p in perspectives) / len(perspectives)
        
        return (agreement + avg_confidence) / 2
    
    def compare_perspectives(
        self,
        perspective1_id: str,
        perspective2_id: str
    ) -> Dict[str, Any]:
        """Compare two perspectives."""
        # Find perspectives
        p1 = None
        p2 = None
        
        for perspectives in self._perspectives.values():
            for p in perspectives:
                if p.perspective_id == perspective1_id:
                    p1 = p
                if p.perspective_id == perspective2_id:
                    p2 = p
        
        if not p1 or not p2:
            return {"error": "Perspective not found"}
        
        # Calculate similarity
        similarity = 1.0 - abs(p1.confidence - p2.confidence)
        
        return {
            "perspective1": p1.to_dict(),
            "perspective2": p2.to_dict(),
            "similarity": similarity,
            "confidence_diff": abs(p1.confidence - p2.confidence)
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get consensus engine statistics."""
        total_perspectives = sum(len(p) for p in self._perspectives.values())
        
        return {
            "active_debates": len(self._active_debates),
            "total_perspectives_collected": total_perspectives,
            "config": self.config.to_dict()
        }
    
    def set_config(self, **kwargs) -> None:
        """Update debate configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
