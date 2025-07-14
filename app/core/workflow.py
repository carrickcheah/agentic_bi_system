"""
Agentic BI Workflow - Clean, Simple, Direct
This is the ONLY flow controller. No more confusion.
"""

import uuid
import logging
from typing import Dict, Any, AsyncGenerator, List
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class AgenticWorkflow:
    """
    The ONE and ONLY workflow controller.
    Simple, clean, no bullshit.
    """
    
    def __init__(self):
        self.intelligence = None
        self.investigation = None
        self.synthesizer = None
        self.mcp_manager = None
        self.qdrant = None
        self.model_manager = None
        self.question_checker = None
        
    async def initialize(self):
        """Initialize all services."""
        # Import what we need
        from intelligence.runner import IntelligenceModuleRunner
        from investigation.runner import conduct_autonomous_investigation
        from insight_synthesis.runner import InsightSynthesizer, OutputFormat
        from main import model_manager, question_checker, qdrant_service
        from fastmcp.client_manager import MCPClientManager
        
        # Set up services
        self.intelligence = IntelligenceModuleRunner()
        self.synthesizer = InsightSynthesizer(model_manager=model_manager)
        self.mcp_manager = MCPClientManager()
        self.model_manager = model_manager
        self.question_checker = question_checker
        self.qdrant = qdrant_service
        self.conduct_investigation_func = conduct_autonomous_investigation
        self.OutputFormat = OutputFormat  # Store OutputFormat enum
        
        logger.info(f"Workflow initialized - Qdrant service: {qdrant_service is not None}")
        
    async def process_query(self, question: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Main entry point. Process any query.
        
        Args:
            question: User's question
            
        Yields:
            Results and progress updates
        """
        investigation_id = str(uuid.uuid4())
        
        try:
            # 1. CHECK IF BUSINESS QUESTION (Using QuestionChecker)
            should_process, rejection_msg = await self.question_checker.check_and_respond(question)
            if not should_process:
                yield {
                    "type": "non_business_response",
                    "response": rejection_msg
                }
                return
            
            # 2. PARALLEL ANALYSIS (Intent + Qdrant + Complexity)
            intent, qdrant_results, complexity, business_intent = await self.intelligence.analyze_query_parallel(
                question, self.qdrant
            )
            
            # 3. CHECK QDRANT CACHE (Fast path)
            if qdrant_results and len(qdrant_results) > 0:
                best_match = qdrant_results[0]
                score = best_match.get('score', 0)
                logger.info(f"Qdrant best match score: {score:.3f}")
                
                if score >= 0.80:  # Lowered from 0.95 for more cache hits
                    # Execute cached SQL directly
                    logger.info(f"Using cached SQL with confidence {score:.3f}")
                    result = await self._execute_cached_query(best_match, investigation_id)
                    if result:
                        yield result
                        return
            
            # 4. CHECK FOR SIMPLE QUERY FAST PATH
            # Skip full investigation for simple queries
            if complexity and complexity.score < 0.3:
                logger.info(f"Simple query detected (complexity: {complexity.score:.2f}), using fast path")
                
                # Try to generate SQL directly for simple queries
                fast_result = await self._handle_simple_query(question, business_intent, investigation_id)
                if fast_result:
                    yield fast_result
                    return
            
            # 5. PLAN INVESTIGATION (for complex queries)
            logger.info(f"Complex query (complexity: {complexity.score if complexity else 'unknown'}), full investigation needed")
            plan = await self.intelligence.plan_investigation_strategy(
                business_question=question,
                user_context={"role": "analyst"},
                organization_context={}
            )
            
            # 6. EXECUTE INVESTIGATION
            # Initialize MCP if needed
            if not self.mcp_manager._initialized:
                await self.mcp_manager.initialize()
            
            investigation_result = await self.conduct_investigation_func(
                coordinated_services={
                    "mariadb": {"enabled": True, "priority": 1},
                    "postgres": {"enabled": True, "priority": 2},
                    "qdrant": {"enabled": True, "priority": 3}
                },
                investigation_request=question,
                execution_context={
                    "business_intent": plan.business_intent,
                    "investigation_strategy": plan.contextual_strategy,
                    "complexity_score": complexity.score if complexity else 0.5
                },
                mcp_client_manager=self.mcp_manager,
                model_manager=self.model_manager
            )
            
            # 7. SYNTHESIZE INSIGHTS
            synthesis = await self.synthesizer.synthesize_insights(
                investigation_results=investigation_result,
                business_context={
                    "investigation_strategy": plan.contextual_strategy,
                    "complexity_level": complexity.level if complexity else "moderate"
                },
                user_role="analyst",
                output_format=self.OutputFormat.EXECUTIVE_SUMMARY
            )
            
            # 8. RETURN FINAL RESULT
            yield {
                "type": "investigation_completed",
                "investigation_id": investigation_id,
                "question": question,
                "insights": {
                    "executive_summary": synthesis.executive_summary,
                    "strategic_insights": [
                        {
                            "title": i.title,
                            "description": i.description,
                            "confidence": i.confidence
                        } for i in synthesis.insights[:3]  # Top 3
                    ],
                    "recommendations": [
                        {
                            "title": r.title,
                            "description": r.description,
                            "priority": r.priority
                        } for r in synthesis.recommendations[:3]  # Top 3
                    ]
                },
                "metadata": {
                    "complexity": complexity.score if complexity else 0,
                    "processing_time": "calculated",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Workflow error: {e}")
            yield {
                "type": "error",
                "error": str(e),
                "investigation_id": investigation_id
            }
    
    async def _execute_cached_query(self, match: Dict, investigation_id: str) -> Dict[str, Any]:
        """Execute cached SQL from Qdrant match."""
        try:
            cached_sql = match.get('payload', {}).get('sql_query', '')
            if not cached_sql:
                return None
                
            mariadb = self.mcp_manager.get_client("mariadb")
            if not mariadb:
                return None
                
            result = await mariadb.execute_query(cached_sql)
            
            return {
                "type": "cached_response",
                "investigation_id": investigation_id,
                "confidence": match.get('score'),
                "result": result,
                "source": "qdrant_cache"
            }
            
        except Exception as e:
            logger.warning(f"Cache execution failed: {e}")
            return None
    
    async def _handle_simple_query(self, question: str, business_intent: Any, investigation_id: str) -> Dict[str, Any]:
        """Handle simple queries with minimal overhead."""
        try:
            # For simple queries, we can use a lightweight approach
            logger.info("Executing simple query fast path")
            
            # Initialize MCP with only MariaDB for simple queries
            if not self.mcp_manager._initialized:
                await self.mcp_manager.initialize()
            
            # Get table patterns from the question
            table_patterns = self._extract_table_patterns(question)
            
            # Simple investigation with minimal steps
            from investigation.runner import AutonomousInvestigationEngine
            engine = AutonomousInvestigationEngine(self.model_manager)
            
            # Create minimal execution context
            execution_context = {
                "business_intent": business_intent,
                "complexity_score": 0.2,  # Force simple mode
                "table_patterns": table_patterns,
                "skip_exploration": True,  # Skip data exploration step
                "skip_validation": True,   # Skip cross-validation
                "fast_mode": True
            }
            
            # Run streamlined investigation
            result = await engine.conduct_investigation(
                coordinated_services={"mariadb": {"enabled": True, "priority": 1}},
                investigation_request=question,
                execution_context=execution_context,
                mcp_client_manager=self.mcp_manager
            )
            
            # Return simplified result
            return {
                "type": "fast_response",
                "investigation_id": investigation_id,
                "insights": {
                    "executive_summary": f"Query executed successfully. {result.investigation_findings.get('summary', '')}",
                    "data": result.investigation_findings.get('core_analysis', {}).get('results', [])
                },
                "metadata": {
                    "execution_type": "fast",
                    "complexity": 0.2,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            }
            
        except Exception as e:
            logger.warning(f"Simple query fast path failed: {e}")
            return None
    
    def _extract_table_patterns(self, question: str) -> List[str]:
        """Extract potential table names from question."""
        # Common business terms to table mappings
        patterns = []
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['sale', 'revenue', 'order']):
            patterns.extend(['sale', 'order', 'revenue'])
        if any(word in question_lower for word in ['product', 'item', 'sku']):
            patterns.extend(['product', 'item', 'sku'])
        if any(word in question_lower for word in ['customer', 'client', 'user']):
            patterns.extend(['customer', 'client', 'user'])
        if any(word in question_lower for word in ['inventory', 'stock']):
            patterns.extend(['inventory', 'stock'])
            
        return list(set(patterns))  # Remove duplicates


# Simple interface functions
async def process_question(question: str) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Simple function to process a question.
    
    Usage:
        async for result in process_question("What are top products?"):
            print(result)
    """
    workflow = AgenticWorkflow()
    await workflow.initialize()
    
    async for result in workflow.process_query(question):
        yield result


# For direct execution
if __name__ == "__main__":
    import asyncio
    
    async def test():
        async for result in process_question("What are our top products?"):
            print(f"Result: {result}")
    
    asyncio.run(test())