"""
Autonomous SQL Agent - The Main Brain

This is the core autonomous agent that investigates SQL/data questions
using a Claude Code-style iterative approach.
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from ..utils.logging import logger
from ..utils.exceptions import AgentError, InvestigationError
from .investigation import InvestigationEngine
from .planner import TaskPlanner
from .memory import MemoryManager
from ..model.sonnet import SonnetClient
from ..mcp.tools import MCPClient
from ..rag.faq_matcher import FAQMatcher


class AutonomousSQLAgent:
    """
    Main autonomous SQL investigation agent.
    
    Works like Claude Code but for data analysis:
    1. Takes high-level business questions
    2. Plans investigation strategy
    3. Executes iteratively with self-correction
    4. Continues until complete insights are discovered
    """
    
    def __init__(self):
        self.agent_id = str(uuid.uuid4())
        self.planner = TaskPlanner()
        self.investigation_engine = InvestigationEngine()
        self.memory = MemoryManager()
        self.sonnet = SonnetClient()
        self.mcp_client = MCPClient()
        self.faq_matcher = FAQMatcher()
        
        # Agent state
        self.active_investigations: Dict[str, Dict] = {}
        self.is_initialized = False
        
        logger.info(f"Autonomous SQL Agent initialized with ID: {self.agent_id}")
    
    async def initialize(self):
        """Initialize all components."""
        try:
            await self.memory.initialize()
            await self.sonnet.initialize()
            await self.mcp_client.initialize()
            await self.faq_matcher.initialize()
            
            self.is_initialized = True
            logger.info("Agent fully initialized and ready for investigations")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise AgentError(f"Agent initialization failed: {e}")
    
    async def investigate(
        self, 
        user_query: str, 
        user_id: str,
        session_id: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Main investigation method - Claude Code style autonomous loop.
        
        Args:
            user_query: Natural language business question
            user_id: User identifier for personalization
            session_id: Session identifier for context
            context: Additional context information
            
        Returns:
            Complete investigation results with insights and recommendations
        """
        if not self.is_initialized:
            await self.initialize()
        
        investigation_id = str(uuid.uuid4())
        session_id = session_id or str(uuid.uuid4())
        
        logger.info(f"Starting investigation {investigation_id} for query: {user_query}")
        
        try:
            # Phase 1: Check FAQ patterns first (fastest path)
            faq_result = await self._check_faq_patterns(user_query, user_id)
            if faq_result and faq_result.get("confidence", 0) > 0.9:
                logger.info(f"FAQ match found with high confidence: {faq_result['confidence']}")
                return await self._execute_faq_response(faq_result, investigation_id, session_id)
            
            # Phase 2: Full autonomous investigation
            return await self._run_autonomous_investigation(
                user_query=user_query,
                user_id=user_id,
                session_id=session_id,
                investigation_id=investigation_id,
                context=context,
                faq_hint=faq_result
            )
            
        except Exception as e:
            logger.error(f"Investigation {investigation_id} failed: {e}")
            await self._handle_investigation_error(investigation_id, e)
            raise InvestigationError(f"Investigation failed: {e}")
    
    async def _check_faq_patterns(self, user_query: str, user_id: str) -> Optional[Dict]:
        """Check if query matches existing FAQ patterns."""
        try:
            # Get user's historical patterns for personalization
            user_patterns = await self.memory.get_user_patterns(user_id)
            
            # Match against FAQ database
            faq_result = await self.faq_matcher.match_query(
                query=user_query,
                user_patterns=user_patterns
            )
            
            return faq_result
            
        except Exception as e:
            logger.warning(f"FAQ matching failed: {e}")
            return None
    
    async def _execute_faq_response(
        self, 
        faq_result: Dict, 
        investigation_id: str, 
        session_id: str
    ) -> Dict[str, Any]:
        """Execute pre-optimized FAQ query."""
        logger.info(f"Executing FAQ response: {faq_result['pattern_name']}")
        
        try:
            # Execute the pre-optimized SQL
            result = await self.mcp_client.call_tool(
                "execute_sql_safely",
                {
                    "sql": faq_result["sql_template"],
                    "database": faq_result["database"],
                    "parameters": faq_result.get("parameters", {})
                }
            )
            
            # Store successful FAQ usage
            await self.memory.record_faq_usage(
                faq_id=faq_result["faq_id"],
                investigation_id=investigation_id,
                success=True
            )
            
            return {
                "investigation_id": investigation_id,
                "session_id": session_id,
                "type": "faq_response",
                "pattern_name": faq_result["pattern_name"],
                "sql_executed": faq_result["sql_template"],
                "results": result,
                "execution_time": result.get("execution_time", 0),
                "confidence": faq_result["confidence"],
                "completed_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"FAQ execution failed: {e}")
            # Fall back to full investigation
            return await self._run_autonomous_investigation(
                user_query=faq_result["original_query"],
                user_id=faq_result["user_id"],
                session_id=session_id,
                investigation_id=investigation_id,
                context={"faq_fallback": True, "failed_faq": faq_result}
            )
    
    async def _run_autonomous_investigation(
        self,
        user_query: str,
        user_id: str,
        session_id: str,
        investigation_id: str,
        context: Optional[Dict] = None,
        faq_hint: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Run full autonomous investigation loop - like Claude Code iteration.
        """
        logger.info(f"Starting autonomous investigation loop for {investigation_id}")
        
        # Initialize investigation state
        investigation_state = {
            "id": investigation_id,
            "session_id": session_id,
            "user_id": user_id,
            "query": user_query,
            "status": "planning",
            "steps": [],
            "findings": [],
            "current_step": 0,
            "max_steps": 20,
            "started_at": datetime.utcnow(),
            "context": context or {},
            "faq_hint": faq_hint
        }
        
        self.active_investigations[investigation_id] = investigation_state
        
        try:
            # Phase 1: Planning - like Claude Code analyzing requirements
            logger.info("🔍 Phase 1: Planning investigation strategy...")
            plan = await self.planner.create_investigation_plan(
                query=user_query,
                user_context=await self.memory.get_user_context(user_id),
                session_context=await self.memory.get_session_context(session_id),
                faq_hint=faq_hint
            )
            
            investigation_state["plan"] = plan
            investigation_state["status"] = "investigating"
            
            # Phase 2: Autonomous execution loop - like Claude Code's iterative development
            logger.info("🚀 Phase 2: Starting autonomous execution loop...")
            
            while not self._is_investigation_complete(investigation_state):
                step_number = investigation_state["current_step"] + 1
                
                if step_number > investigation_state["max_steps"]:
                    logger.warning(f"Investigation {investigation_id} hit max steps limit")
                    break
                
                logger.info(f"🔄 Executing step {step_number}...")
                
                # Execute next investigation step
                step_result = await self.investigation_engine.execute_step(
                    investigation_state=investigation_state,
                    mcp_client=self.mcp_client,
                    sonnet_client=self.sonnet
                )
                
                # Update investigation state
                investigation_state["steps"].append(step_result)
                investigation_state["current_step"] = step_number
                
                # Analyze results and adapt strategy (like Claude Code debugging)
                if step_result.get("needs_adaptation"):
                    logger.info("🔧 Adapting investigation strategy based on findings...")
                    await self._adapt_investigation_strategy(investigation_state, step_result)
                
                # Check if we discovered something significant
                if step_result.get("significant_finding"):
                    investigation_state["findings"].append(step_result["finding"])
                
                # Brief pause to prevent overwhelming
                await asyncio.sleep(0.1)
            
            # Phase 3: Synthesis - like Claude Code final documentation
            logger.info("🔬 Phase 3: Synthesizing findings...")
            final_results = await self._synthesize_investigation_results(investigation_state)
            
            # Phase 4: Learning - store patterns for future use
            logger.info("🧠 Phase 4: Learning from investigation...")
            await self._learn_from_investigation(investigation_state, final_results)
            
            investigation_state["status"] = "completed"
            investigation_state["completed_at"] = datetime.utcnow()
            investigation_state["final_results"] = final_results
            
            logger.info(f"✅ Investigation {investigation_id} completed successfully")
            
            return final_results
            
        except Exception as e:
            investigation_state["status"] = "failed"
            investigation_state["error"] = str(e)
            logger.error(f"Investigation {investigation_id} failed: {e}")
            raise
        finally:
            # Clean up active investigation
            self.active_investigations.pop(investigation_id, None)
    
    def _is_investigation_complete(self, investigation_state: Dict) -> bool:
        """Determine if investigation has sufficient findings to complete."""
        findings = investigation_state.get("findings", [])
        steps = investigation_state.get("steps", [])
        
        # Complete if we have actionable insights
        if len(findings) >= 3 and any(f.get("actionable") for f in findings):
            return True
        
        # Complete if last few steps haven't yielded new insights
        if len(steps) >= 5:
            recent_steps = steps[-3:]
            if all(not step.get("significant_finding") for step in recent_steps):
                return True
        
        return False
    
    async def _adapt_investigation_strategy(
        self, 
        investigation_state: Dict, 
        step_result: Dict
    ):
        """Adapt investigation strategy based on current findings."""
        # Use Sonnet to analyze current state and suggest adaptations
        adaptation = await self.sonnet.adapt_investigation_strategy(
            current_plan=investigation_state["plan"],
            executed_steps=investigation_state["steps"],
            latest_result=step_result
        )
        
        if adaptation.get("new_approach"):
            investigation_state["plan"].update(adaptation["new_approach"])
            logger.info(f"Strategy adapted: {adaptation['reasoning']}")
    
    async def _synthesize_investigation_results(self, investigation_state: Dict) -> Dict[str, Any]:
        """Synthesize all findings into comprehensive results."""
        synthesis = await self.sonnet.synthesize_investigation(
            query=investigation_state["query"],
            steps=investigation_state["steps"],
            findings=investigation_state["findings"],
            context=investigation_state["context"]
        )
        
        return {
            "investigation_id": investigation_state["id"],
            "session_id": investigation_state["session_id"],
            "type": "autonomous_investigation",
            "query": investigation_state["query"],
            "summary": synthesis["summary"],
            "key_insights": synthesis["insights"],
            "recommendations": synthesis["recommendations"],
            "sql_queries": [step.get("sql") for step in investigation_state["steps"] if step.get("sql")],
            "data_sources": synthesis["data_sources"],
            "confidence_score": synthesis["confidence"],
            "steps_executed": len(investigation_state["steps"]),
            "investigation_time": (datetime.utcnow() - investigation_state["started_at"]).total_seconds(),
            "completed_at": datetime.utcnow().isoformat()
        }
    
    async def _learn_from_investigation(
        self, 
        investigation_state: Dict, 
        final_results: Dict
    ):
        """Learn from successful investigation to improve future performance."""
        try:
            # Store successful patterns
            if final_results.get("confidence_score", 0) > 0.7:
                await self.memory.store_successful_pattern(
                    query_pattern=investigation_state["query"],
                    investigation_steps=investigation_state["steps"],
                    results=final_results,
                    user_id=investigation_state["user_id"]
                )
            
            # Update user preferences
            await self.memory.update_user_patterns(
                user_id=investigation_state["user_id"],
                investigation=investigation_state
            )
            
        except Exception as e:
            logger.warning(f"Failed to learn from investigation: {e}")
    
    async def _handle_investigation_error(self, investigation_id: str, error: Exception):
        """Handle investigation errors gracefully."""
        if investigation_id in self.active_investigations:
            self.active_investigations[investigation_id]["status"] = "failed"
            self.active_investigations[investigation_id]["error"] = str(error)
    
    async def get_investigation_status(self, investigation_id: str) -> Optional[Dict]:
        """Get current status of an active investigation."""
        return self.active_investigations.get(investigation_id)
    
    async def stop_investigation(self, investigation_id: str) -> bool:
        """Stop an active investigation."""
        if investigation_id in self.active_investigations:
            self.active_investigations[investigation_id]["status"] = "stopped"
            return True
        return False