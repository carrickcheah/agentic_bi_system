from model import ModelManager
from utils import QuestionChecker
from qdrant import get_qdrant_service
from fastmcp.client_manager import MCPClientManager
from cache import CacheManager
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

# Initialize core services synchronously
# OpenAI embeddings are now handled within the model package
model_manager = ModelManager()
question_checker = QuestionChecker(model_manager)

# Initialize Qdrant (async initialization handled lazily)
qdrant_service = None

# Initialize FastMCP (async initialization handled lazily)
mcp_client_manager = None
_mcp_initialized = False

# Initialize Cache (async initialization handled lazily)/login
cache_manager = None
_cache_initialized = False

async def initialize_async_services():
    """Initialize services that require async setup."""  # Sets up Qdrant vector database
    global qdrant_service
    
    # Initialize Qdrant with vector search
    qdrant_service = await get_qdrant_service()
    
    return qdrant_service

async def get_mcp_client_manager():
    """Get or initialize MCP client manager on demand."""  # Lazy loads MCP database connections
    global mcp_client_manager, _mcp_initialized
    
    if not _mcp_initialized:
        mcp_client_manager = MCPClientManager()
        # Don't initialize any services by default - let Phase 3 decide
        _mcp_initialized = True
    
    return mcp_client_manager

async def get_cache_manager():
    """Get or initialize cache manager on demand."""  # Initializes Anthropic cache only
    global cache_manager, _cache_initialized
    
    if not _cache_initialized:
        # Initialize cache manager with Anthropic cache only
        # No MCP needed for caching - PostgreSQL is only for chat history
        cache_manager = CacheManager()
        await cache_manager.initialize()
        
        _cache_initialized = True
    
    return cache_manager

# Export all services for other modules
__all__ = [
    "model_manager", 
    "question_checker", 
    "qdrant_service",
    "get_mcp_client_manager",
    "get_cache_manager",
    "initialize_async_services"
]


class AgenticBiFlow:
    """
    Simple interface that makes the entire business intelligence flow visible.
    This is the 'facade' that provides a clean API for the 5-phase investigation workflow.
    
    Example usage:
        flow = AgenticBiFlow()
        await flow.initialize()
        result = await flow.quick_query("What were yesterday's sales?")
    """
    
    def __init__(self):
        self.business_analyst = None
        self.cache = None
        self.is_initialized = False
    
    async def initialize(self):
        """
        Step 1: Initialize all services needed for business intelligence.
        This includes cache, MCP services, and the business analyst.
        """  # Prepares BI flow dependencies
        # Get cache manager
        self.cache = await get_cache_manager()
        
        # Initialize business analyst
        from core.business_analyst import AutonomousBusinessAnalyst
        self.business_analyst = AutonomousBusinessAnalyst()
        await self.business_analyst.initialize()
        
        self.is_initialized = True
        return self
    
    async def investigate_query(
        self, 
        question: str,
        user_context: dict = None,
        organization_context: dict = None,
        stream_progress: bool = True
    ):
        """
        Step 2: Main investigation flow - visible at high level.  # Orchestrates 5-phase investigation
        
        The 5-Phase Flow:
        1. Cache Check (50-100ms if hit)
        2. Parallel Analysis (intent, qdrant, complexity)
        3. Service Orchestration (MCP setup)
        4. Investigation Execution (data retrieval)
        5. Insight Synthesis (strategic recommendations)
        
        Args:
            question: Natural language business question
            user_context: User information (role, department, etc.)
            organization_context: Organization settings and constraints
            stream_progress: Whether to yield progress updates
            
        Yields:
            Progress updates and final investigation results
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Delegate to business analyst but make flow visible here
        async for result in self.business_analyst.conduct_investigation(
            question,
            user_context or {},
            organization_context or {},
            stream_progress=stream_progress
        ):
            yield result
    
    async def quick_query(self, question: str) -> dict:
        """
        Convenience method for simple queries - returns just the final result.
        Perfect for when you don't need streaming updates.  # Non-streaming query execution
        
        Args:
            question: Business question to investigate
            
        Returns:
            Final investigation results with insights and recommendations
        """
        results = []
        async for result in self.investigate_query(question, stream_progress=True):
            if result.get("type") == "investigation_completed":
                return result
            results.append(result)
        return results[-1] if results else None
    
    async def get_investigation_status(self, investigation_id: str) -> dict:
        """Check status of an ongoing investigation."""  # Returns current investigation progress
        return await self.business_analyst.get_investigation_status(investigation_id)
    
    async def get_cache_stats(self) -> dict:
        """Get cache performance statistics."""  # Retrieves cache hit/miss metrics
        if self.cache:
            return self.cache.cache_stats
        return {}


# Add to exports
__all__.append("AgenticBiFlow")


def _combined_intent_analysis(query: str, intent_classifier, domain_expert):
    """Combines general and business intent classification into one operation."""  # Unified intent analysis
    general_intent = intent_classifier.classify_intent(query)
    business_intent = domain_expert.classify_business_intent(query)
    return {
        "general": general_intent,
        "business": business_intent
    }


async def process_query_with_validation_and_5_phases(
    query: str, 
    user_context: dict = None,
    organization_context: dict = None,
    stream_progress: bool = False
):
    """
    Complete 5-phase investigation flow with QuestionChecker validation first.
    All phases implemented here for maximum control and visibility.  # Full 5-phase BI investigation pipeline
    
    Flow:
    0. QuestionChecker - Validate if business question
    1. Cache Check - Multi-tier cache lookup
    2. Intelligence Planning - Parallel analysis (intent, qdrant, complexity)
    3. Service Orchestration - MCP setup
    4. Investigation Execution - Data retrieval
    5. Insight Synthesis - Strategic recommendations
    
    Args:
        query: Natural language query
        user_context: User information (role, department, etc.)
        organization_context: Organization settings
        stream_progress: Whether to yield progress updates
        
    Yields/Returns:
        Progress updates if streaming, otherwise final result
    """
    from datetime import datetime
    import hashlib
    import uuid
    
    investigation_id = str(uuid.uuid4())
    results = {"phases": {}}
    
    # ========== PRE-CHECK: Question Validation ==========
    if stream_progress:
        yield {"phase": "pre-check", "name": "Question Validation", "status": "checking"}
    
    # Check if it's a valid business question
    is_valid, rejection_message = await question_checker.is_database_question(query)
    
    if not is_valid:
        # Not a business question - return immediately
        if stream_progress:
            yield {
                "phase": "pre-check", 
                "name": "Question Validation", 
                "status": "rejected",
                "message": rejection_message
            }
        
        yield {
            "type": "non_business_response",
            "investigation_id": investigation_id,
            "valid": False,
            "response": rejection_message,
            "bypassed_investigation": True
        }
        return
    
    if stream_progress:
        yield {"phase": "pre-check", "name": "Question Validation", "status": "passed"}
    
    # ========== PHASE 1: Multi-Tier Cache Check ==========
    if stream_progress:
        yield {"phase": 1, "name": "Cache Check", "status": "starting"}
    
    cache = await get_cache_manager()
    cache_key = hashlib.md5(query.encode()).hexdigest()
    
    cached_result = await cache.get_cached_insights(
        semantic_hash=cache_key,
        business_domain="general",
        semantic_intent={"query": query},
        user_context=user_context or {},
        organization_context=organization_context or {}
    )
    
    if cached_result:
        if stream_progress:
            yield {"phase": 1, "name": "Cache Check", "status": "hit", "data": cached_result}
        yield {
            "type": "cached_response",
            "investigation_id": investigation_id,
            "from_cache": True,
            "insights": cached_result,
            "processing_time": 0.05  # 50ms cache hit
        }
        return
    
    if stream_progress:
        yield {"phase": 1, "name": "Cache Check", "status": "miss"}
    
    # ========== PHASE 2: Intelligence Planning (Parallel Analysis) ==========
    if stream_progress:
        yield {"phase": 2, "name": "Intelligence Planning", "status": "starting"}
    
    # Import components
    from intelligence.query_intent_classifier import QueryIntentClassifier
    from intelligence.domain_expert import DomainExpert
    from intelligence.complexity_analyzer import ComplexityAnalyzer
    
    # Initialize components
    intent_classifier = QueryIntentClassifier()
    domain_expert = DomainExpert()
    complexity_analyzer = ComplexityAnalyzer()
    
    # Initialize qdrant if needed
    if qdrant_service is None:
        await initialize_async_services()
    
    # Run parallel analysis - EXACTLY 3 TASKS as per flow diagram
    import asyncio
    
    # Task 1: Combined Intent Analysis (Extract Intent)
    intent_task = asyncio.create_task(
        asyncio.to_thread(_combined_intent_analysis, query, intent_classifier, domain_expert)
    )
    
    # Task 2: Search Qdrant
    qdrant_task = asyncio.create_task(
        qdrant_service.search_similar_queries(query, limit=5, threshold=0.85)
        if qdrant_service else asyncio.sleep(0)  # Dummy task if no qdrant
    )
    
    # Task 3: Analyze Complexity (simplified - no business intent dependency)
    complexity_task = asyncio.create_task(
        asyncio.to_thread(
            complexity_analyzer.analyze_query_complexity,  # Use simpler method
            query
        )
    )
    
    # Gather all 3 results simultaneously - TRUE PARALLELISM
    intent_results, qdrant_results, complexity_result = await asyncio.gather(
        intent_task,
        qdrant_task,
        complexity_task
    )
    
    # Extract intent components
    intent_result = intent_results["general"]
    business_intent = intent_results["business"]
    
    # NEW: Enhance complexity with business intent
    if business_intent and complexity_result:
        try:
            # Store original score for comparison
            original_score = complexity_result.score
            
            # Enhance complexity with business context
            complexity_result = complexity_analyzer.enhance_complexity_with_intent(
                complexity_result,
                business_intent
            )
            
            # Log the enhancement
            print(f"   ✓ Complexity enhanced: {original_score:.2f} → {complexity_result.score:.2f}")
            
            if stream_progress:
                yield {"phase": 2, "name": "Intelligence Planning", 
                       "enhancement": "complexity_enhanced",
                       "original_score": original_score,
                       "enhanced_score": complexity_result.score}
        except Exception as e:
            # If enhancement fails, continue with base complexity
            print(f"   ⚠️  Complexity enhancement failed: {e}")
    
    results["phases"]["intelligence"] = {
        "intent": intent_result.intent.value if intent_result else "unknown",
        "qdrant_matches": len(qdrant_results) if qdrant_results else 0,
        "complexity": complexity_result.score if complexity_result else 0,
        "methodology": complexity_result.methodology.value if complexity_result else "unknown"
    }
    
    if stream_progress:
        yield {"phase": 2, "name": "Intelligence Planning", "status": "completed", 
               "data": results["phases"]["intelligence"]}
    
    # Check for Qdrant exact match
    if qdrant_results and len(qdrant_results) > 0:
        best_match = qdrant_results[0]  # Results are sorted by score
        
        if best_match.get("score", 0) > 0.9:  # High confidence exact match
            if stream_progress:
                yield {"phase": "qdrant_match", "status": "exact_match", 
                       "confidence": best_match["score"],
                       "matched_query": best_match.get("business_question", "")}
            
            # Fast execute using stored SQL pattern
            stored_sql = best_match.get("sql_query", "")
            if stored_sql:
                if stream_progress:
                    yield {"phase": "fast_execute", "status": "using_cached_sql"}
                
                # Return fast response with cached SQL
                yield {
                    "type": "qdrant_fast_response",
                    "investigation_id": investigation_id,
                    "from_qdrant": True,
                    "confidence": best_match["score"],
                    "cached_sql": stored_sql,
                    "original_question": best_match.get("business_question", ""),
                    "result": f"Found exact match (confidence: {best_match['score']:.2f})",
                    "note": "Using cached SQL pattern from previous investigation"
                }
                return
            else:
                # High confidence match but no SQL stored - continue to normal flow
                if stream_progress:
                    yield {"phase": "qdrant_match", "status": "no_sql_cached", 
                           "note": "High confidence match but no SQL pattern stored"}
        else:
            # Low confidence - continue to normal routing
            if stream_progress:
                yield {"phase": "qdrant_match", "status": "low_confidence", 
                       "best_score": best_match.get("score", 0)}
    
    # Route based on complexity
    complexity_score = complexity_result.score if complexity_result else 0.5
    
    if complexity_score < 0.3:
        # Fast SQL Path
        if stream_progress:
            yield {"phase": "routing", "route": "fast_sql", "complexity": complexity_score}
        async for result in _fast_sql_execution(query, business_intent, qdrant_results, investigation_id, results, stream_progress):
            yield result
        return
    elif complexity_score < 0.5:
        # Hybrid Path
        if stream_progress:
            yield {"phase": "routing", "route": "hybrid", "complexity": complexity_score}
        async for result in _hybrid_investigation(query, business_intent, qdrant_results, investigation_id, results, stream_progress):
            yield result
        return
    else:
        # Full 5-phase investigation - continue to phases 3-5
        if stream_progress:
            yield {"phase": "routing", "route": "full_investigation", "complexity": complexity_score}
    
    # ========== PHASE 3: Service Orchestration ==========
    if stream_progress:
        yield {"phase": 3, "name": "Service Orchestration", "status": "starting"}
    
    # Determine which services are needed based on complexity
    complexity_score = complexity_result.score if complexity_result else 0.5
    services_needed = []
    
    if complexity_score < 0.8:
        # Simple to Moderate: Only MariaDB needed
        services_needed = ["mariadb"]
    else:
        # Complex: MariaDB + Qdrant for pattern matching
        services_needed = ["mariadb"]
        # Initialize Qdrant if not already initialized
        if qdrant_service is None:
            await initialize_async_services()
    
    # Initialize MCP with only required services
    mcp = MCPClientManager()
    await mcp.initialize(services=services_needed)
    
    # Store globally for other functions to use
    global mcp_client_manager
    mcp_client_manager = mcp
    
    # Verify services are ready
    services_ready = {
        "mariadb": "mariadb" in mcp.clients,
        "postgres": "postgres" in mcp.clients,  # Should be False
        "qdrant": qdrant_service is not None
    }
    
    results["phases"]["orchestration"] = {
        "services_initialized": services_needed,
        "services_ready": services_ready,
        "complexity_routing": "simple" if complexity_score < 0.3 else "moderate" if complexity_score < 0.8 else "complex",
        "complexity_score": complexity_score
    }
    
    if stream_progress:
        yield {"phase": 3, "name": "Service Orchestration", "status": "completed",
               "data": {
                   "services_initialized": services_needed,
                   "complexity_routing": results["phases"]["orchestration"]["complexity_routing"]
               }}
    
    # ========== PHASE 4: Investigation Execution ==========
    if stream_progress:
        yield {"phase": 4, "name": "Investigation Execution", "status": "starting"}
    
    # Import investigation runner
    from investigation.runner import conduct_autonomous_investigation
    
    # Prepare execution context
    execution_context = {
        "business_intent": business_intent,
        "investigation_strategy": complexity_result,
        "user_context": user_context or {},
        "complexity_score": complexity_score  # Pass complexity for adaptive execution
    }
    
    # Run investigation
    investigation_result = await conduct_autonomous_investigation(
        coordinated_services={"mariadb": {"enabled": True}},
        investigation_request=query,
        execution_context=execution_context,
        mcp_client_manager=mcp
    )
    
    results["phases"]["investigation"] = investigation_result
    
    if stream_progress:
        yield {"phase": 4, "name": "Investigation Execution", "status": "completed",
               "data": {"queries_executed": len(investigation_result.get("sql_queries", []))}}
    
    # ========== PHASE 5: Strategic Insight Synthesis ==========
    if stream_progress:
        yield {"phase": 5, "name": "Insight Synthesis", "status": "starting"}
    
    # Import synthesizer
    from insight_synthesis.runner import InsightSynthesizer, OutputFormat
    
    synthesizer = InsightSynthesizer()
    
    # Prepare business context
    business_context = {
        **(organization_context or {}),
        "investigation_strategy": complexity_result,
        "complexity_level": complexity_result.level.value if complexity_result else "unknown",
        "user_preferences": user_context or {}
    }
    
    # Synthesize insights
    synthesis_result = await synthesizer.synthesize_insights(
        investigation_results=investigation_result,
        business_context=business_context,
        user_role=user_context.get("role", "analyst") if user_context else "analyst",
        output_format=OutputFormat.DETAILED_REPORT
    )
    
    results["phases"]["synthesis"] = {
        "insights_count": len(synthesis_result.insights),
        "recommendations_count": len(synthesis_result.recommendations),
        "executive_summary": synthesis_result.executive_summary
    }
    
    # Save to cache for future use
    cache_metadata = {
        "investigation_id": investigation_id,
        "complexity": complexity_score,
        "methodology": complexity_result.methodology.value if complexity_result else "unknown"
    }
    
    # Store in Anthropic cache (if we have conversation context)
    conversation_context = {
        "original_query": query,
        "investigation_id": investigation_id,
        "business_intent": business_intent,
        "user_context": user_context,
        "organization_context": organization_context
    }
    
    await cache.store_insights(
        semantic_hash=cache_key,
        business_domain="general",
        semantic_intent=semantic_intent,
        user_context=user_context or {},
        organization_context=organization_context or {},
        insights=synthesis_result.executive_summary,
        conversation_context=conversation_context
    )
    
    # Save chat history to PostgreSQL
    # Initialize PostgreSQL if not already done
    if "postgres" not in mcp.clients:
        await mcp.initialize_service("postgres")
    
    if mcp.postgres:
        try:
            # Create or update session
            session_id = user_context.get("session_id", investigation_id)
            user_id = user_context.get("user_id", "anonymous")
            
            # Store conversation in chat history
            await mcp.postgres.store_short_term_memory(
                session_id=session_id,
                memory_key=f"chat_{investigation_id}",
                memory_value={
                    "type": "conversation",
                    "timestamp": datetime.utcnow().isoformat(),
                    "user_message": {
                        "type": "user_question",
                        "content": query,
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    "ai_response": {
                        "type": "ai_response",
                        "content": synthesis_result.executive_summary,
                        "insights": [
                            {
                                "id": insight.id,
                                "title": insight.title,
                                "description": insight.description
                            } for insight in synthesis_result.insights[:3]  # Top 3 insights
                        ],
                        "recommendations": [
                            {
                                "id": rec.id,
                                "title": rec.title,
                                "description": rec.description
                            } for rec in synthesis_result.recommendations[:3]  # Top 3 recommendations
                        ],
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    "metadata": {
                        "investigation_id": investigation_id,
                        "complexity_score": complexity_score,
                        "processing_time": results.get("total_time", 0),
                        "services_used": results["phases"]["orchestration"]["services_initialized"]
                    }
                },
                ttl_seconds=2592000  # 30 days
            )
            
            logger.debug(f"Saved conversation to PostgreSQL history for session {session_id}")
            
        except Exception as e:
            logger.warning(f"Failed to save chat history: {e}")
            # Don't fail the whole request if history storage fails
    
    if stream_progress:
        yield {"phase": 5, "name": "Insight Synthesis", "status": "completed",
               "data": results["phases"]["synthesis"]}
    
    # Return final result
    final_result = {
        "type": "investigation_completed",
        "investigation_id": investigation_id,
        "insights": {
            "strategic_insights": [
                {
                    "id": insight.id,
                    "type": insight.type.value,
                    "title": insight.title,
                    "description": insight.description,
                    "confidence": insight.confidence,
                    "business_impact": insight.business_impact
                } for insight in synthesis_result.insights
            ],
            "recommendations": [
                {
                    "id": rec.id,
                    "type": rec.type.value,
                    "title": rec.title,
                    "description": rec.description,
                    "priority": rec.priority,
                    "timeline": rec.timeline
                } for rec in synthesis_result.recommendations
            ],
            "executive_summary": synthesis_result.executive_summary,
            "business_impact_assessment": synthesis_result.business_impact_assessment
        },
        
        "metadata": results
    }
    
    yield final_result


# Fast path implementation
async def _fast_sql_execution(query, business_intent, qdrant_results, investigation_id, results, stream_progress):
    """Direct SQL execution for simple queries."""  # Bypasses investigation for simple SQL
    if stream_progress:
        yield {"phase": "fast_path", "status": "executing"}
    
    # Ensure MCP is available
    global mcp_client_manager
    if not mcp_client_manager or "mariadb" not in mcp_client_manager.clients:
        # Initialize MariaDB only
        mcp = MCPClientManager()
        await mcp.initialize(services=["mariadb"])
        mcp_client_manager = mcp
    
    try:
        # Generate simple SQL using model
        sql_prompt = f"""Generate a simple SQL query for this business question: {query}

Context: This is a simple query with low complexity. Generate only the SQL query without explanation.
Business intent: {business_intent.get('intent', 'unknown') if business_intent else 'unknown'}

Return only the SQL query in a code block."""
        
        sql_response = await model_manager.generate_response(sql_prompt)
        
        # Extract SQL from response
        sql_query = _extract_sql_from_response(sql_response)
        
        if stream_progress:
            yield {"phase": "fast_path", "status": "executing_sql", "sql": sql_query[:100] + "..."}
        
        # Execute directly on MariaDB
        start_time = datetime.utcnow()
        result = await mcp_client_manager.mariadb.execute_query(sql_query)
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Format the result
        formatted_result = {
            "type": "fast_response",
            "investigation_id": investigation_id,
            "query": query,
            "sql": sql_query,
            "data": result.data[:10] if result.data else [],  # Limit to 10 rows
            "total_rows": result.row_count,
            "execution_time": execution_time,
            "complexity": results["phases"]["intelligence"]["complexity"],
            "note": "Direct SQL execution for simple query"
        }
        
        # Simple summarization if we have data
        if result.data:
            summary_prompt = f"""Summarize this data result for the question: {query}

Data (first 10 rows):
{result.data[:10]}

Total rows: {result.row_count}

Provide a brief, clear summary in 1-2 sentences."""
            
            summary = await model_manager.generate_response(summary_prompt)
            formatted_result["summary"] = summary
        
        # Save to chat history if PostgreSQL is available
        if hasattr(mcp_client_manager, 'postgres') and mcp_client_manager.postgres:
            try:
                session_id = results.get("session_id", investigation_id)
                await mcp_client_manager.postgres.store_short_term_memory(
                    session_id=session_id,
                    memory_key=f"fast_query_{investigation_id}",
                    memory_value={
                        "type": "fast_query",
                        "question": query,
                        "sql": sql_query,
                        "summary": formatted_result.get("summary", ""),
                        "row_count": result.row_count,
                        "execution_time": execution_time,
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    ttl_seconds=86400  # 1 day for simple queries
                )
            except Exception as e:
                logger.debug(f"Could not save fast query to history: {e}")
        
        yield formatted_result
        
    except Exception as e:
        yield {
            "type": "fast_response_error",
            "investigation_id": investigation_id,
            "error": str(e),
            "note": "Fast SQL execution failed, consider using full investigation"
        }


# Hybrid path implementation  
async def _hybrid_investigation(query, business_intent, qdrant_results, investigation_id, results, stream_progress):
    """Partial investigation for moderate complexity."""  # Uses Qdrant plus limited investigation
    if stream_progress:
        yield {"phase": "hybrid_path", "status": "executing"}
    
    # Ensure MCP is available
    global mcp_client_manager
    if not mcp_client_manager or "mariadb" not in mcp_client_manager.clients:
        # Initialize MariaDB only
        mcp = MCPClientManager()
        await mcp.initialize(services=["mariadb"])
        mcp_client_manager = mcp
    
    try:
        # Use Qdrant results to provide context
        context_info = []
        if qdrant_results:
            for i, match in enumerate(qdrant_results[:3], 1):
                context_info.append({
                    "similar_query": match.get("business_question", ""),
                    "sql_pattern": match.get("sql_query", ""),
                    "confidence": match.get("score", 0)
                })
        
        # Generate SQL with context from similar queries
        sql_prompt = f"""Generate an SQL query for this business question: {query}

Context: This is a moderate complexity query. We have found similar queries that might help.
Business intent: {business_intent.get('intent', 'unknown') if business_intent else 'unknown'}

Similar successful queries:
{json.dumps(context_info, indent=2) if context_info else "No similar queries found"}

Generate an appropriate SQL query based on the question and similar patterns.
Return only the SQL query in a code block."""
        
        sql_response = await model_manager.generate_response(sql_prompt)
        sql_query = _extract_sql_from_response(sql_response)
        
        if stream_progress:
            yield {"phase": "hybrid_path", "status": "executing_sql", "sql": sql_query[:100] + "..."}
        
        # Execute on MariaDB
        start_time = datetime.utcnow()
        result = await mcp_client_manager.mariadb.execute_query(sql_query)
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        # More detailed analysis than fast path
        analysis_prompt = f"""Analyze this data to answer the business question: {query}

Data Summary:
- Total rows: {result.row_count}
- Columns: {', '.join(result.columns) if result.columns else 'Unknown'}
- Sample data (first 5 rows): {result.data[:5] if result.data else 'No data'}

Provide:
1. A clear answer to the business question
2. Key insights from the data
3. Any notable patterns or trends

Keep the response concise but informative."""
        
        analysis = await model_manager.generate_response(analysis_prompt)
        
        # Generate simple recommendations
        recommendation_prompt = f"""Based on this analysis, provide 1-2 actionable recommendations:

Question: {query}
Analysis: {analysis}

Provide brief, practical recommendations."""
        
        recommendations = await model_manager.generate_response(recommendation_prompt)
        
        # Format the result
        formatted_result = {
            "type": "hybrid_response",
            "investigation_id": investigation_id,
            "query": query,
            "sql": sql_query,
            "data_summary": {
                "total_rows": result.row_count,
                "columns": result.columns,
                "sample_data": result.data[:5] if result.data else []
            },
            "analysis": analysis,
            "recommendations": recommendations,
            "execution_time": execution_time,
            "complexity": results["phases"]["intelligence"]["complexity"],
            "qdrant_matches_used": len(context_info),
            "note": "Hybrid investigation using pattern matching and direct SQL"
        }
        
        # Save to chat history if PostgreSQL is available  
        if hasattr(mcp_client_manager, 'postgres') and mcp_client_manager.postgres:
            try:
                session_id = results.get("session_id", investigation_id)
                await mcp_client_manager.postgres.store_short_term_memory(
                    session_id=session_id,
                    memory_key=f"hybrid_query_{investigation_id}",
                    memory_value={
                        "type": "hybrid_query",
                        "question": query,
                        "sql": sql_query,
                        "analysis": analysis,
                        "recommendations": recommendations,
                        "row_count": result.row_count,
                        "execution_time": execution_time,
                        "qdrant_patterns_used": len(context_info),
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    ttl_seconds=604800  # 7 days for moderate queries
                )
            except Exception as e:
                logger.debug(f"Could not save hybrid query to history: {e}")
        
        yield formatted_result
        
    except Exception as e:
        yield {
            "type": "hybrid_response_error",
            "investigation_id": investigation_id,
            "error": str(e),
            "note": "Hybrid investigation failed, consider using full investigation"
        }

# Helper function to extract SQL from model response
def _extract_sql_from_response(response: str) -> str:
    """Extract SQL query from model response."""
    import re
    
    # Look for SQL in code blocks
    sql_pattern = r'```sql\n(.*?)\n```'
    match = re.search(sql_pattern, response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Also try without sql marker
    code_pattern = r'```\n(.*?)\n```'
    match = re.search(code_pattern, response, re.DOTALL)
    if match:
        potential_sql = match.group(1).strip()
        # Check if it looks like SQL
        if any(keyword in potential_sql.upper() for keyword in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']):
            return potential_sql
    
    # Fallback: look for SELECT/INSERT/UPDATE statements
    lines = response.split('\n')
    for line in lines:
        stripped = line.strip()
        if any(stripped.upper().startswith(cmd) for cmd in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']):
            # Collect the full query
            sql_lines = [stripped]
            idx = lines.index(line) + 1
            while idx < len(lines) and not lines[idx].strip().endswith(';'):
                sql_lines.append(lines[idx])
                idx += 1
            if idx < len(lines):
                sql_lines.append(lines[idx])
            return '\n'.join(sql_lines).strip()
    
    raise ValueError("No SQL query found in response")

# Add new functions to exports
__all__.extend([
    "process_query_with_validation_and_5_phases",
    "_fast_sql_execution",
    "_hybrid_investigation",
    "_extract_sql_from_response"
])




async def simple_chat():
    """Simple chat interface using the 5-phase flow directly in main.py."""  # Interactive BI chat loop
    import uuid
    
    print("=" * 60)
    print("Agentic BI Chat - Direct Flow Control")
    print("=" * 60)
    print("\n💡 Type 'exit' to quit, 'help' for examples\n")
    
    session_id = str(uuid.uuid4())
    
    while True:
        try:
            # Get user input
            query = input("\n🤖 You: ").strip()
            
            if not query:
                continue
                
            if query.lower() in ['exit', 'quit', 'bye']:
                print("\n👋 Goodbye!")
                break
                
            if query.lower() == 'help':
                print("\n📚 Example questions:")
                print("  • What were yesterday's sales?")
                print("  • Why did customer satisfaction drop?")
                print("  • Show me top products by revenue")
                continue
            
            # Process query using our 5-phase flow
            print("\n" + "-" * 40)
            
            async for update in process_query_with_validation_and_5_phases(
                query=query,
                user_context={"session_id": session_id, "role": "analyst"},
                stream_progress=True
            ):
                # Handle different update types
                if update.get("type") == "non_business_response":
                    print(f"\n💬 {update.get('response')}")
                    break
                    
                elif update.get("type") == "cached_response":
                    print(f"\n⚡ Cache Hit! (saved ~15s)")
                    insights = update.get("insights", {})
                    print(f"\n📋 {insights.get('executive_summary', 'Cached result')}")
                    break
                    
                elif update.get("type") == "qdrant_fast_response":
                    print(f"\n🎯 Qdrant Exact Match! (confidence: {update.get('confidence', 0):.2f})")
                    print(f"   Original: {update.get('original_question', 'N/A')}")
                    print(f"   SQL: {update.get('cached_sql', 'N/A')[:100]}...")
                    print(f"\n📋 {update.get('result')}")
                    break
                    
                elif "phase" in update:
                    phase = update.get("phase")
                    status = update.get("status", "")
                    name = update.get("name", "")
                    
                    if phase == "pre-check" and status == "rejected":
                        # Already handled by non_business_response
                        continue
                    elif phase == "qdrant_match":
                        status = update.get("status", "")
                        if status == "exact_match":
                            print(f"\n🎯 Qdrant Match Found! Confidence: {update.get('confidence', 0):.2f}")
                        elif status == "low_confidence":
                            print(f"   ✓ Qdrant best match: {update.get('best_score', 0):.2f} (below threshold)")
                    elif phase == "routing":
                        route = update.get("route", "")
                        complexity = update.get("complexity", 0)
                        print(f"\n🔀 Routing: {route} (complexity: {complexity:.2f})")
                    elif status == "starting":
                        print(f"\n🔄 Phase {phase}: {name}...")
                    elif status == "completed":
                        if phase == 2:  # Intelligence planning
                            data = update.get("data", {})
                            print(f"   ✓ Complexity: {data.get('complexity', 0):.2f}")
                            print(f"   ✓ Method: {data.get('methodology', 'unknown')}")
                
                elif update.get("type") in ["fast_response", "hybrid_response"]:
                    print(f"\n🚀 {update.get('type')}: {update.get('result')}")
                    print(f"   Note: {update.get('note')}")
                    
                elif update.get("type") == "investigation_completed":
                    insights = update.get("insights", {})
                    print(f"\n✅ Investigation Complete!")
                    print(f"\n📋 Summary: {insights.get('executive_summary', 'No summary')}")
                    
                    # Show insights
                    strategic_insights = insights.get("strategic_insights", [])
                    if strategic_insights:
                        print(f"\n💡 Insights ({len(strategic_insights)}):")
                        for i, insight in enumerate(strategic_insights[:3], 1):
                            print(f"   {i}. {insight.get('title', 'Untitled')}")
                    
                    # Show recommendations
                    recommendations = insights.get("recommendations", [])
                    if recommendations:
                        print(f"\n🎯 Recommendations ({len(recommendations)}):")
                        for i, rec in enumerate(recommendations[:3], 1):
                            print(f"   {i}. {rec.get('title', 'Untitled')} [{rec.get('priority', 'Medium')}]")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted. Type 'exit' to quit.")
            continue
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue


if __name__ == "__main__":
    # Validate and initialize all services when running directly
    import asyncio
    import sys
    
    async def startup(skip_validation=False, chat_mode=False):
        """Initialize and validate all services."""  # Bootstraps app with optional chat
        if not skip_validation:
            # Validate models
            print("🔍 Validating model API keys...")
            await model_manager.validate_models()
            print("✅ Models validated")
        else:
            print("⏭️  Skipping model validation")
        
        # Initialize only essential services
        print("🔌 Initializing services...")
        await initialize_async_services()
        print("✅ Qdrant initialized")
        
        if not chat_mode:
            # Show available services
            print("\n📦 Available Services:")
            print(f"  - model_manager: {model_manager}")
            print(f"  - question_checker: {question_checker}")
            print(f"  - qdrant_service: {qdrant_service}")
            print(f"  - get_mcp_client_manager: <lazy initialization>")
            print(f"  - get_cache_manager: <lazy initialization>")
            print(f"  - AgenticBiFlow: <high-level interface>")
            print(f"  - process_query_with_validation_and_5_phases: <full control flow>")
            
            print("\n✅ All services ready!")
            print("💡 MCP and Cache will be initialized on first use")
            print("\n💬 Run with --chat for interactive mode")
        else:
            # Start chat mode
            await simple_chat()
    
    # Check for flags
    skip_validation = "--skip-validation" in sys.argv
    chat_mode = "--chat" in sys.argv
    
    try:
        asyncio.run(startup(skip_validation, chat_mode))
    except KeyboardInterrupt:
        print("\n\nShutting down gracefully...")