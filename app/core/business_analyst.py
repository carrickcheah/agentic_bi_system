"""
Autonomous Business Analyst - 5-Phase Investigation Orchestrator
Central coordinator that integrates all phases of the autonomous business intelligence workflow.
"""

from typing import Dict, List, Any, Optional, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime, timezone
import asyncio
import uuid
import logging

# Import phase runners
from intelligence.runner import IntelligenceModuleRunner
from fastmcp.mcp_client_manager import MCPClientManager
from investigation.runner import conduct_autonomous_investigation
from insight_synthesis.runner import InsightSynthesizer, OutputFormat

# Set up logger
logger = logging.getLogger("business_analyst")


@dataclass
class InvestigationProgress:
    """Investigation progress tracking."""
    investigation_id: str
    current_phase: str
    phase_number: int
    total_phases: int
    progress_percentage: float
    status: str
    message: str
    timestamp: datetime
    phase_results: Optional[Dict[str, Any]] = None


class AutonomousBusinessAnalyst:
    """
    Central orchestrator for autonomous business intelligence investigations.
    Coordinates the complete 5-phase workflow from query to strategic insights.
    """
    
    def __init__(self):
        self.investigation_cache = {}
        self.active_investigations = {}
        
        # Phase runners
        self.intelligence_runner = None
        self.synthesizer = None
        self.mcp_manager = None
        
        logger.info("AutonomousBusinessAnalyst initialized")
    
    async def initialize(self):
        """Initialize all phase components."""
        try:
            # Initialize MCP client manager
            self.mcp_manager = MCPClientManager()
            await self.mcp_manager.initialize()
            
            # Initialize phase runners
            self.intelligence_runner = IntelligenceModuleRunner()
            self.synthesizer = InsightSynthesizer()
            
            logger.info("✅ All phase components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize business analyst: {e}")
            raise
    
    async def conduct_investigation(
        self,
        business_question: str,
        user_context: Dict[str, Any],
        organization_context: Dict[str, Any],
        stream_progress: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Conduct complete 5-phase autonomous business intelligence investigation.
        
        Args:
            business_question: Natural language business question
            user_context: User information and preferences
            organization_context: Organizational context and constraints
            stream_progress: Whether to stream real-time progress updates
            
        Yields:
            Progress updates and final results
        """
        investigation_id = str(uuid.uuid4())
        
        try:
            # Initialize investigation tracking
            self.active_investigations[investigation_id] = {
                "status": "running",
                "start_time": datetime.now(timezone.utc),
                "business_question": business_question,
                "user_context": user_context,
                "organization_context": organization_context
            }
            
            # Phase results accumulator
            phase_results = {}
            
            # Phase 1 & 2: Intelligence Planning (Query Processing + Strategy Planning)
            if stream_progress:
                yield self._create_progress_update(
                    investigation_id, "intelligence_planning", 1, 5, 10.0,
                    "Analyzing business question and planning investigation strategy..."
                )
            
            intelligence_result = await self._execute_intelligence_planning(
                business_question, user_context, organization_context
            )
            phase_results["intelligence"] = intelligence_result
            
            if stream_progress:
                yield self._create_progress_update(
                    investigation_id, "intelligence_planning", 1, 5, 25.0,
                    f"Strategy planned: {intelligence_result.get('strategy_summary', 'Investigation approach determined')}",
                    intelligence_result
                )
            
            # Phase 3: Service Orchestration (handled internally by MCP manager)
            if stream_progress:
                yield self._create_progress_update(
                    investigation_id, "service_orchestration", 2, 5, 35.0,
                    "MCP database services ready for investigation..."
                )
            
            orchestration_result = {
                "mcp_manager": self.mcp_manager,
                "services_ready": True,
                "active_services": ["database_client"]
            }
            phase_results["orchestration"] = orchestration_result
            
            if stream_progress:
                yield self._create_progress_update(
                    investigation_id, "service_orchestration", 2, 5, 50.0,
                    "Database services coordinated and ready",
                    orchestration_result
                )
            
            # Phase 4: Investigation Execution
            if stream_progress:
                yield self._create_progress_update(
                    investigation_id, "investigation_execution", 3, 5, 60.0,
                    "Executing autonomous investigation with 7-step framework..."
                )
            
            investigation_result = await self._execute_investigation(
                business_question, intelligence_result, user_context
            )
            phase_results["investigation"] = investigation_result
            
            if stream_progress:
                yield self._create_progress_update(
                    investigation_id, "investigation_execution", 3, 5, 80.0,
                    f"Investigation completed: {len(investigation_result.get('step_results', {}))} steps executed",
                    investigation_result
                )
            
            # Phase 5: Insight Synthesis
            if stream_progress:
                yield self._create_progress_update(
                    investigation_id, "insight_synthesis", 4, 5, 90.0,
                    "Synthesizing strategic insights and generating recommendations..."
                )
            
            synthesis_result = await self._execute_insight_synthesis(
                investigation_result, intelligence_result, user_context, organization_context
            )
            phase_results["synthesis"] = synthesis_result
            
            # Complete investigation
            final_result = {
                "investigation_id": investigation_id,
                "type": "investigation_completed",
                "status": "completed",
                "business_question": business_question,
                "insights": {
                    "strategic_insights": [
                        {
                            "id": insight.id,
                            "type": insight.type.value,
                            "title": insight.title,
                            "description": insight.description,
                            "confidence": insight.confidence,
                            "business_impact": insight.business_impact,
                            "strategic_depth": insight.strategic_depth,
                            "actionability": insight.actionability,
                            "stakeholders": insight.stakeholders,
                            "related_domains": insight.related_domains
                        } for insight in synthesis_result.insights
                    ],
                    "recommendations": [
                        {
                            "id": rec.id,
                            "type": rec.type.value,
                            "title": rec.title,
                            "description": rec.description,
                            "priority": rec.priority,
                            "timeline": rec.timeline,
                            "feasibility": rec.feasibility,
                            "resource_requirements": rec.resource_requirements,
                            "expected_outcomes": rec.expected_outcomes,
                            "success_metrics": rec.success_metrics
                        } for rec in synthesis_result.recommendations
                    ],
                    "executive_summary": synthesis_result.executive_summary,
                    "business_impact_assessment": synthesis_result.business_impact_assessment
                },
                "metadata": {
                    "investigation_phases": phase_results,
                    "organizational_learning": {
                        "pattern_id": synthesis_result.organizational_learning.pattern_id,
                        "pattern_description": synthesis_result.organizational_learning.pattern_description,
                        "business_value": synthesis_result.organizational_learning.business_value,
                        "best_practices": synthesis_result.organizational_learning.best_practices,
                        "lessons_learned": synthesis_result.organizational_learning.lessons_learned
                    },
                    "completion_time": datetime.now(timezone.utc).isoformat(),
                    "synthesis_metadata": synthesis_result.synthesis_metadata
                }
            }
            
            # Update investigation cache
            self.investigation_cache[investigation_id] = final_result
            self.active_investigations[investigation_id]["status"] = "completed"
            
            if stream_progress:
                yield self._create_progress_update(
                    investigation_id, "completed", 5, 5, 100.0,
                    f"Investigation completed with {len(synthesis_result.insights)} insights and {len(synthesis_result.recommendations)} recommendations",
                    final_result
                )
            
            # Final result
            yield final_result
            
        except Exception as e:
            logger.error(f"Investigation {investigation_id} failed: {e}")
            
            # Update status
            if investigation_id in self.active_investigations:
                self.active_investigations[investigation_id]["status"] = "failed"
                self.active_investigations[investigation_id]["error"] = str(e)
            
            # Yield error result
            yield {
                "investigation_id": investigation_id,
                "type": "investigation_failed",
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def _execute_intelligence_planning(
        self,
        business_question: str,
        user_context: Dict[str, Any],
        organization_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute Phase 1 & 2: Intelligence Planning."""
        try:
            # Run intelligence analysis
            planning_result = await self.intelligence_runner.plan_investigation_strategy(
                business_question=business_question,
                user_context=user_context,
                organization_context=organization_context
            )
            
            return {
                "business_intent": planning_result.business_intent,
                "contextual_strategy": planning_result.contextual_strategy,
                "hypothesis_set": planning_result.hypothesis_set,
                "strategy_summary": f"{planning_result.contextual_strategy.adapted_methodology.value} methodology selected",
                "complexity_level": planning_result.contextual_strategy.base_methodology,
                "estimated_timeline": planning_result.contextual_strategy.estimated_timeline
            }
            
        except Exception as e:
            logger.error(f"Intelligence planning failed: {e}")
            raise
    
    
    async def _execute_investigation(
        self,
        business_question: str,
        intelligence_result: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute Phase 4: Investigation Execution."""
        try:
            # Run autonomous investigation
            investigation_result = await conduct_autonomous_investigation(
                business_question=business_question,
                business_intent=intelligence_result["business_intent"],
                investigation_strategy=intelligence_result["contextual_strategy"],
                mcp_client_manager=self.mcp_manager,
                user_context=user_context
            )
            
            return investigation_result
            
        except Exception as e:
            logger.error(f"Investigation execution failed: {e}")
            raise
    
    async def _execute_insight_synthesis(
        self,
        investigation_result: Dict[str, Any],
        intelligence_result: Dict[str, Any],
        user_context: Dict[str, Any],
        organization_context: Dict[str, Any]
    ) -> Any:  # Returns SynthesisResult
        """Execute Phase 5: Insight Synthesis."""
        try:
            # Prepare business context for synthesis
            business_context = {
                **organization_context,
                "investigation_strategy": intelligence_result.get("contextual_strategy"),
                "complexity_level": intelligence_result.get("complexity_level"),
                "user_preferences": user_context
            }
            
            # Determine user role for tailored output
            user_role = user_context.get("role", "analyst")
            
            # Run insight synthesis
            synthesis_result = await self.synthesizer.synthesize_insights(
                investigation_results=investigation_result,
                business_context=business_context,
                user_role=user_role,
                output_format=OutputFormat.DETAILED_REPORT
            )
            
            return synthesis_result
            
        except Exception as e:
            logger.error(f"Insight synthesis failed: {e}")
            raise
    
    def _create_progress_update(
        self,
        investigation_id: str,
        phase: str,
        phase_number: int,
        total_phases: int,
        progress: float,
        message: str,
        phase_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a progress update dictionary."""
        return {
            "type": "progress_update",
            "investigation_id": investigation_id,
            "current_phase": phase,
            "phase_number": phase_number,
            "total_phases": total_phases,
            "progress_percentage": progress,
            "status": "running",
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phase_results": phase_results
        }
    
    async def get_investigation_status(self, investigation_id: str) -> Dict[str, Any]:
        """Get current status of an investigation."""
        if investigation_id in self.active_investigations:
            investigation = self.active_investigations[investigation_id]
            return {
                "investigation_id": investigation_id,
                "status": investigation["status"],
                "progress_percentage": 0.0,  # Would be tracked in real implementation
                "current_phase": "unknown",
                "estimated_completion": None,
                "start_time": investigation["start_time"].isoformat()
            }
        elif investigation_id in self.investigation_cache:
            return {
                "investigation_id": investigation_id,
                "status": "completed",
                "progress_percentage": 100.0,
                "current_phase": "completed",
                "estimated_completion": None
            }
        else:
            raise ValueError(f"Investigation {investigation_id} not found")
    
    async def get_organizational_insights(
        self,
        organization_id: str,
        business_domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get organizational insights and investigation history."""
        # This would integrate with organizational memory system
        return {
            "organization_id": organization_id,
            "total_investigations": len(self.investigation_cache),
            "successful_investigations": len([
                inv for inv in self.investigation_cache.values()
                if inv.get("status") == "completed"
            ]),
            "business_domains": [],  # Would be extracted from cache
            "patterns": [],  # Would be from pattern recognition
            "organizational_learning": {}  # Would be from synthesis results
        }
    
    async def collaborate_on_investigation(
        self,
        investigation_id: str,
        user_context: Dict[str, Any],
        expert_feedback: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enable collaboration on an investigation."""
        # This would implement real-time collaboration features
        return {
            "collaboration_status": "feedback_received",
            "feedback_integration": "pending",
            "investigation_id": investigation_id,
            "user_id": user_context.get("user_id"),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.mcp_manager:
            await self.mcp_manager.cleanup()
        
        logger.info("AutonomousBusinessAnalyst cleanup completed")