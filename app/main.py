from model import ModelManager
from utils import QuestionChecker
from qdrant import get_qdrant_service
from fastmcp.client_manager import MCPClientManager
from cache import CacheManager

# Initialize core services synchronously
# OpenAI embeddings are now handled within the model package
model_manager = ModelManager()
question_checker = QuestionChecker(model_manager)

# Initialize Qdrant (async initialization handled lazily)
qdrant_service = None

# Initialize FastMCP (async initialization handled lazily)
mcp_client_manager = None
_mcp_initialized = False

# Initialize Cache (async initialization handled lazily)
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
        await mcp_client_manager.initialize()
        _mcp_initialized = True
    
    return mcp_client_manager

async def get_cache_manager():
    """Get or initialize cache manager on demand."""  # Initializes multi-tier caching system
    global cache_manager, _cache_initialized
    
    if not _cache_initialized:
        # Cache depends on MCP for PostgreSQL
        mcp = await get_mcp_client_manager()
        
        cache_manager = CacheManager()
        await cache_manager.initialize()
        
        # Inject PostgreSQL client into cache tiers that need it
        if mcp.postgres:
            cache_manager.postgresql_cache.postgres_client = mcp.postgres
            cache_manager.semantic_cache.postgres_client = mcp.postgres
        
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
    
    # Get MCP services
    mcp = await get_mcp_client_manager()
    
    services_ready = {
        "mariadb": mcp.clients.get("mariadb") is not None if hasattr(mcp, 'clients') else False,
        "postgres": mcp.postgres is not None if hasattr(mcp, 'postgres') else False,
        "qdrant": qdrant_service is not None
    }
    
    results["phases"]["orchestration"] = {
        "services_ready": services_ready,
        "mcp_initialized": True
    }
    
    if stream_progress:
        yield {"phase": 3, "name": "Service Orchestration", "status": "completed",
               "data": results["phases"]["orchestration"]}
    
    # ========== PHASE 4: Investigation Execution ==========
    if stream_progress:
        yield {"phase": 4, "name": "Investigation Execution", "status": "starting"}
    
    # Import investigation runner
    from investigation.runner import conduct_autonomous_investigation
    
    # Prepare execution context
    execution_context = {
        "business_intent": business_intent,
        "investigation_strategy": complexity_result,
        "user_context": user_context or {}
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
    
    # Store in cache (simplified for now - would need proper formatting)
    # await cache.store_insights(cache_key, synthesis_result, cache_metadata)
    
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
    
    # TODO: Implement direct SQL generation and execution
    # For now, return placeholder
    yield {
        "type": "fast_response",
        "investigation_id": investigation_id,
        "query": query,
        "complexity": results["phases"]["intelligence"]["complexity"],
        "result": "Fast SQL execution not yet implemented",
        "note": "Would execute simple SQL directly without full investigation"
    }


# Hybrid path implementation  
async def _hybrid_investigation(query, business_intent, qdrant_results, investigation_id, results, stream_progress):
    """Partial investigation for moderate complexity."""  # Uses Qdrant plus limited investigation
    if stream_progress:
        yield {"phase": "hybrid_path", "status": "executing"}
    
    # TODO: Implement hybrid investigation
    # Uses some investigation steps but not all
    yield {
        "type": "hybrid_response",
        "investigation_id": investigation_id,
        "query": query,
        "complexity": results["phases"]["intelligence"]["complexity"],
        "qdrant_matches": len(qdrant_results) if qdrant_results else 0,
        "result": "Hybrid investigation not yet implemented",
        "note": "Would use Qdrant results and partial investigation"
    }

# Add new functions to exports
__all__.extend([
    "process_query_with_validation_and_5_phases",
    "_fast_sql_execution",
    "_hybrid_investigation"
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
                    
                elif "phase" in update:
                    phase = update.get("phase")
                    status = update.get("status", "")
                    name = update.get("name", "")
                    
                    if phase == "pre-check" and status == "rejected":
                        # Already handled by non_business_response
                        continue
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