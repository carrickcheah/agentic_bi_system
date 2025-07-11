"""
Autonomous Business Analyst - 5-Phase Investigation Orchestrator
Central coordinator that integrates all phases of the autonomous business intelligence workflow.
"""

from typing import Dict, List, Any, Optional, AsyncGenerator, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
import asyncio
import uuid
import logging

# Import phase runners
from intelligence.runner import IntelligenceModuleRunner
from fastmcp.client_manager import MCPClientManager
from investigation.runner import conduct_autonomous_investigation
from insight_synthesis.runner import InsightSynthesizer, OutputFormat
from config import settings

# Import components for parallel processing
from intelligence.query_intent_classifier import QueryIntentClassifier
from intelligence.domain_expert import DomainExpert
from intelligence.complexity_analyzer import ComplexityAnalyzer

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
    
    def __init__(self, model_manager=None):
        self.investigation_cache = {}
        self.active_investigations = {}
        
        # Phase runners
        self.intelligence_runner = None
        self.synthesizer = None
        self.mcp_manager = None
        self.vector_service = None
        self.model_manager = model_manager
        
        # Components for parallel processing
        self.intent_classifier = None
        self.domain_expert = None
        self.complexity_analyzer = None
        
        logger.info("AutonomousBusinessAnalyst initialized")
    
    async def initialize(self):
        """Initialize all phase components."""
        try:
            # Initialize MCP client manager but don't connect yet
            self.mcp_manager = MCPClientManager()
            # Don't initialize during chat startup - initialize on-demand
            
            # Get vector service directly if available
            if settings.use_qdrant:
                try:
                    from qdrant import get_qdrant_service
                    
                    # Initialize Qdrant service
                    self.vector_service = await get_qdrant_service()
                    
                    if self.vector_service:
                        logger.info("✅ Qdrant vector search initialized")
                    else:
                        logger.warning("Qdrant service not available")
                except Exception as e:
                    logger.warning(f"Could not initialize Qdrant: {e}")
                    logger.info("System will continue without vector search capabilities")
            
            # Initialize phase runners
            self.intelligence_runner = IntelligenceModuleRunner()
            self.synthesizer = InsightSynthesizer(model_manager=self.model_manager)
            
            logger.info("✅ All phase components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize business analyst: {e}")
            raise
    
    async def _parallel_query_analysis(
        self, 
        business_question: str
    ) -> Tuple[Any, Optional[List[Dict[str, Any]]], Any, Any]:
        """
        Run intent classification, Qdrant search, and complexity analysis in parallel.
        
        Args:
            business_question: Natural language business question
            
        Returns:
            Tuple of (intent_classification, qdrant_results, complexity_score, business_intent)
        """
        # Initialize components if not already done
        if self.intent_classifier is None:
            self.intent_classifier = QueryIntentClassifier()
        
        if self.domain_expert is None:
            self.domain_expert = DomainExpert()
            
        if self.complexity_analyzer is None:
            self.complexity_analyzer = ComplexityAnalyzer()
        
        logger.info("Starting parallel query analysis...")
        start_time = asyncio.get_event_loop().time()
        
        # Create parallel tasks
        tasks = []
        
        # 1. Extract Intent (sync function, needs asyncio.to_thread)
        intent_task = asyncio.create_task(
            asyncio.to_thread(
                self.intent_classifier.classify_intent, 
                business_question
            ),
            name="intent_classification"
        )
        tasks.append(intent_task)
        
        # 2. Search Qdrant (already async)
        qdrant_task = None
        if self.vector_service:
            qdrant_task = asyncio.create_task(
                self.vector_service.search_similar_queries(
                    business_question, 
                    limit=5, 
                    threshold=0.85
                ),
                name="qdrant_search"
            )
            tasks.append(qdrant_task)
        
        # 3. Business Intent Classification (needed for complexity analysis)
        business_intent_task = asyncio.create_task(
            asyncio.to_thread(
                self.domain_expert.classify_business_intent,
                business_question
            ),
            name="business_intent"
        )
        
        # First, get business intent as complexity analysis depends on it
        business_intent = await business_intent_task
        
        # 4. Analyze Complexity (depends on business intent)
        complexity_task = asyncio.create_task(
            asyncio.to_thread(
                self.complexity_analyzer.analyze_complexity,
                business_intent,
                business_question
            ),
            name="complexity_analysis"
        )
        tasks.append(complexity_task)
        
        # Wait for remaining tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle results and errors
        intent_classification = results[0] if not isinstance(results[0], Exception) else None
        qdrant_results = None
        complexity_score = None
        
        # Extract Qdrant results if task was created
        if qdrant_task:
            qdrant_idx = 1  # Qdrant is second in tasks list
            qdrant_results = results[qdrant_idx] if not isinstance(results[qdrant_idx], Exception) else None
            complexity_score = results[-1] if not isinstance(results[-1], Exception) else None
        else:
            complexity_score = results[-1] if not isinstance(results[-1], Exception) else None
        
        # Log performance
        elapsed_time = asyncio.get_event_loop().time() - start_time
        logger.info(f"Parallel query analysis completed in {elapsed_time:.3f}s")
        
        # Log individual results
        if intent_classification:
            logger.debug(f"Intent: {intent_classification.intent.value} (confidence: {intent_classification.confidence:.2f})")
        if qdrant_results:
            logger.debug(f"Qdrant found {len(qdrant_results)} similar queries")
        if complexity_score:
            logger.debug(f"Complexity: {complexity_score.level.value} (score: {complexity_score.score:.2f})")
        
        return intent_classification, qdrant_results, complexity_score, business_intent
    
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
            
            # PARALLEL QUERY ANALYSIS: Run intent, Qdrant, and complexity analysis simultaneously
            # This replaces the sequential processing for better performance
            try:
                intent_classification, qdrant_results, complexity_score, business_intent = \
                    await self._parallel_query_analysis(business_question)
                
                # Handle non-business intents immediately (greetings, help, etc.)
                logger.info(f"Intent classification result: {intent_classification}")
                if intent_classification and hasattr(intent_classification, 'intent'):
                    logger.info(f"Intent type: {intent_classification.intent}, Is greeting: {intent_classification.intent.value}")
                    # Check if it's a non-business query using QueryIntent enum
                    from intelligence.query_intent_classifier import QueryIntent
                    if intent_classification.intent in [QueryIntent.GREETING, QueryIntent.HELP, QueryIntent.CASUAL]:
                        # Generate appropriate response based on intent type
                        response_text = self._generate_non_business_response(
                            intent_classification.intent,
                            business_question
                        )
                        
                        if stream_progress:
                            yield {
                                "type": "non_business_response",
                                "investigation_id": investigation_id,
                                "intent_type": intent_classification.intent.value,
                                "response": response_text,
                                "confidence": intent_classification.confidence
                            }
                        
                        # Complete the investigation with the response
                        yield {
                            "type": "investigation_completed",
                            "investigation_id": investigation_id,
                            "insights": {
                                "executive_summary": response_text,
                                "strategic_insights": [],
                                "recommendations": [],
                                "is_non_business_response": True,
                                "intent_type": intent_classification.intent.value
                            },
                            "metadata": {
                                "processing_time_seconds": 0.1,
                                "intent_classification": intent_classification.intent.value,
                                "bypassed_investigation": True
                            }
                        }
                        return
                
                # Log Qdrant results if found
                if qdrant_results:
                    logger.info(f"Found {len(qdrant_results)} similar queries in Qdrant")
                    # Could potentially use these for fast response in future iterations
                
                # Store parallel analysis results for use in phases
                phase_results["parallel_analysis"] = {
                    "intent_classification": intent_classification,
                    "qdrant_results": qdrant_results,
                    "complexity_score": complexity_score,
                    "business_intent": business_intent
                }
                
            except Exception as e:
                logger.error(f"Parallel query analysis failed: {e}", exc_info=True)
                # Fallback to basic analysis if parallel processing fails
                intent_classification = None
                qdrant_results = None
                complexity_score = None
                business_intent = None
            
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
                    f"Investigation completed: {len(investigation_result.completed_steps)} steps executed",
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
            
            # Store successful investigation in vector database (optional)
            # Skip storing in vector service for now (no sql_queries in InvestigationResults)
            if False:  # self.vector_service and hasattr(investigation_result, 'sql_queries'):
                try:
                    # Store each SQL query with its context
                    for sql_info in []:
                        await self.vector_service.store_query(
                            query_id=f"{investigation_id}_{sql_info.get('step', 'unknown')}",
                            sql_query=sql_info.get("query", ""),
                            business_question=business_question,
                            database=sql_info.get("database", "mariadb"),
                            metadata={
                                "investigation_id": investigation_id,
                                "executive_summary": synthesis_result.executive_summary,
                                "insights_count": len(synthesis_result.insights),
                                "recommendations_count": len(synthesis_result.recommendations)
                            }
                        )
                    logger.info(f"Stored {len(investigation_result.get('sql_queries', []))} queries in vector database")
                except Exception as e:
                    logger.debug(f"Failed to store investigation in vector database (non-critical): {e}")
            
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
            # Initialize MCP manager on-demand if not already initialized
            if self.mcp_manager and not self.mcp_manager._initialized:
                try:
                    await self.mcp_manager.initialize()
                except Exception as e:
                    logger.warning(f"MCP initialization failed: {e}")
                    # Continue without MCP - investigation can still use fallback methods
            
            # Run autonomous investigation
            # Prepare coordinated services
            coordinated_services = {
                "mariadb": {"enabled": True, "priority": 1},
                "postgres": {"enabled": True, "priority": 2},
                "qdrant": {"enabled": True, "priority": 3}
            }
            
            # Prepare execution context
            execution_context = {
                "business_intent": intelligence_result["business_intent"],
                "investigation_strategy": intelligence_result["contextual_strategy"],
                "user_context": user_context
            }
            
            investigation_result = await conduct_autonomous_investigation(
                coordinated_services=coordinated_services,
                investigation_request=business_question,
                execution_context=execution_context,
                mcp_client_manager=self.mcp_manager,
                model_manager=self.model_manager
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
    
    def _generate_non_business_response(self, intent_type: Any, question: str) -> str:
        """Generate appropriate response for non-business queries."""
        from intelligence.query_intent_classifier import QueryIntent
        
        if intent_type == QueryIntent.GREETING:
            return "Hello! I'm your Autonomous Business Analyst. I'm here to help you analyze business data and generate strategic insights. What business question can I help you investigate today?"
        
        elif intent_type == QueryIntent.HELP:
            return """I can help you with business intelligence and data analysis. Here are some examples:
            
• Sales Analysis: "What were last quarter's sales trends?"
• Performance Metrics: "Show me production efficiency by line"
• Root Cause Analysis: "Why did customer satisfaction drop?"
• Forecasting: "Predict next month's demand"
• Strategic Planning: "Compare revenue across product categories"

Just ask me any business question and I'll conduct a comprehensive investigation!"""
        
        elif intent_type == QueryIntent.CASUAL:
            return "I'm specialized in business intelligence and data analysis. While I appreciate the casual chat, I'm most helpful when you have business questions about your data. What business insights can I help you discover?"
        
        else:
            return "I'm ready to help with your business analysis needs. Please provide a business question and I'll investigate it for you."
    
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