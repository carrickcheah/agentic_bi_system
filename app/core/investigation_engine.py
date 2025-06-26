"""
Investigation Engine - Multi-Phase Autonomous Execution

Implements Phase 4: Autonomous Investigation Execution from the five-phase workflow.

Conducts sophisticated multi-phase investigations:
- Dynamic investigation flow adapting to findings
- Hypothesis generation and testing
- Cross-domain validation
- Real-time collaboration
- Iterative deep diving based on evidence
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

from ..utils.logging import logger
from ..fastapi.websocket import send_investigation_progress, send_investigation_result, send_investigation_error


class StepType(Enum):
    """Types of investigation steps."""
    SCHEMA_ANALYSIS = "schema_analysis"
    DATA_EXPLORATION = "data_exploration"
    HYPOTHESIS_TESTING = "hypothesis_testing"
    PATTERN_DISCOVERY = "pattern_discovery"
    VALIDATION = "validation"
    OPTIMIZATION = "optimization"
    SYNTHESIS = "synthesis"


class StepStatus(Enum):
    """Investigation step status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class InvestigationEngine:
    """
    Core engine for executing autonomous investigation steps.
    
    Like Claude Code's execution engine, this iteratively works through
    investigation tasks, adapting based on results and errors.
    """
    
    def __init__(self):
        self.step_executors = {
            StepType.SCHEMA_ANALYSIS: self._execute_schema_analysis,
            StepType.DATA_EXPLORATION: self._execute_data_exploration,
            StepType.HYPOTHESIS_TESTING: self._execute_hypothesis_testing,
            StepType.PATTERN_DISCOVERY: self._execute_pattern_discovery,
            StepType.VALIDATION: self._execute_validation,
            StepType.OPTIMIZATION: self._execute_optimization,
            StepType.SYNTHESIS: self._execute_synthesis,
        }
    
    async def execute_step(
        self,
        investigation_state: Dict,
        mcp_client,
        sonnet_client,
        step_override: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Execute the next investigation step.
        
        Args:
            investigation_state: Current investigation state
            mcp_client: MCP client for tool access
            sonnet_client: Sonnet client for AI reasoning
            step_override: Optional step configuration override
            
        Returns:
            Step execution result with findings and next actions
        """
        step_number = investigation_state["current_step"] + 1
        
        logger.info(f"Executing investigation step {step_number}")
        
        try:
            # Determine next step from plan or adapt dynamically
            next_step = step_override or await self._determine_next_step(
                investigation_state, sonnet_client
            )
            
            step_type = StepType(next_step["type"])
            
            # Create step context
            step_context = {
                "step_number": step_number,
                "step_type": step_type,
                "investigation_id": investigation_state["id"],
                "started_at": datetime.utcnow(),
                "status": StepStatus.RUNNING,
                "config": next_step.get("config", {}),
                "previous_steps": investigation_state.get("steps", []),
                "findings_so_far": investigation_state.get("findings", [])
            }
            
            # Execute the step
            executor = self.step_executors[step_type]
            result = await executor(
                step_context=step_context,
                investigation_state=investigation_state,
                mcp_client=mcp_client,
                sonnet_client=sonnet_client
            )
            
            # Process step result
            step_context.update({
                "status": StepStatus.COMPLETED,
                "completed_at": datetime.utcnow(),
                "execution_time": (datetime.utcnow() - step_context["started_at"]).total_seconds(),
                "result": result
            })
            
            # Analyze if result is significant
            step_context["significant_finding"] = await self._analyze_significance(
                result, investigation_state, sonnet_client
            )
            
            # Determine if adaptation is needed
            step_context["needs_adaptation"] = await self._check_adaptation_needed(
                result, investigation_state, sonnet_client
            )
            
            logger.info(f"Step {step_number} completed: {step_type.value}")
            
            return step_context
            
        except Exception as e:
            logger.error(f"Step {step_number} failed: {e}")
            
            # Return failed step context
            return {
                "step_number": step_number,
                "status": StepStatus.FAILED,
                "error": str(e),
                "completed_at": datetime.utcnow(),
                "needs_adaptation": True,
                "significant_finding": False
            }
    
    async def _determine_next_step(
        self,
        investigation_state: Dict,
        sonnet_client
    ) -> Dict[str, Any]:
        """Determine the next investigation step using AI reasoning."""
        
        # Ask Sonnet to analyze current state and suggest next step
        next_step_analysis = await sonnet_client.determine_next_investigation_step(
            query=investigation_state["query"],
            plan=investigation_state.get("plan", {}),
            completed_steps=investigation_state.get("steps", []),
            findings=investigation_state.get("findings", []),
            context=investigation_state.get("context", {})
        )
        
        return next_step_analysis
    
    # Step Executors
    
    async def _execute_schema_analysis(
        self,
        step_context: Dict,
        investigation_state: Dict,
        mcp_client,
        sonnet_client
    ) -> Dict[str, Any]:
        """Analyze database schema to understand available data."""
        
        logger.info("Executing schema analysis...")
        
        try:
            # Get schema information for relevant databases
            schema_info = await mcp_client.call_tool(
                "analyze_database_schema",
                {"database_name": "mariadb_company"}
            )
            
            # Ask Sonnet to analyze schema relevance to query
            schema_analysis = await sonnet_client.analyze_schema_relevance(
                query=investigation_state["query"],
                schema_info=schema_info
            )
            
            return {
                "step_type": "schema_analysis",
                "raw_schema": schema_info,
                "relevant_tables": schema_analysis["relevant_tables"],
                "potential_joins": schema_analysis["potential_joins"],
                "data_quality_notes": schema_analysis.get("data_quality_notes", []),
                "confidence": schema_analysis.get("confidence", 0.8)
            }
            
        except Exception as e:
            raise InvestigationStepError(f"Schema analysis failed: {e}")
    
    async def _execute_data_exploration(
        self,
        step_context: Dict,
        investigation_state: Dict,
        mcp_client,
        sonnet_client
    ) -> Dict[str, Any]:
        """Explore data to understand patterns and distributions."""
        
        logger.info("Executing data exploration...")
        
        try:
            # Get sample data from relevant tables
            tables_to_explore = step_context["config"].get("tables", [])
            
            exploration_results = {}
            for table in tables_to_explore:
                sample_data = await mcp_client.call_tool(
                    "get_table_sample_data",
                    {"table_name": table, "database": "mariadb_company"}
                )
                
                exploration_results[table] = sample_data
            
            # Analyze patterns with Sonnet
            pattern_analysis = await sonnet_client.analyze_data_patterns(
                query=investigation_state["query"],
                sample_data=exploration_results
            )
            
            return {
                "step_type": "data_exploration",
                "explored_tables": list(exploration_results.keys()),
                "sample_data": exploration_results,
                "patterns_discovered": pattern_analysis["patterns"],
                "anomalies": pattern_analysis.get("anomalies", []),
                "next_hypotheses": pattern_analysis.get("hypotheses", []),
                "confidence": pattern_analysis.get("confidence", 0.7)
            }
            
        except Exception as e:
            raise InvestigationStepError(f"Data exploration failed: {e}")
    
    async def _execute_hypothesis_testing(
        self,
        step_context: Dict,
        investigation_state: Dict,
        mcp_client,
        sonnet_client
    ) -> Dict[str, Any]:
        """Test specific hypotheses about the data."""
        
        logger.info("Executing hypothesis testing...")
        
        try:
            hypothesis = step_context["config"].get("hypothesis")
            if not hypothesis:
                raise InvestigationStepError("No hypothesis provided for testing")
            
            # Generate SQL to test hypothesis
            test_sql = await sonnet_client.generate_hypothesis_test_sql(
                hypothesis=hypothesis,
                available_schema=investigation_state.get("schema_info"),
                previous_findings=investigation_state.get("findings", [])
            )
            
            # Execute the test
            test_result = await mcp_client.call_tool(
                "execute_sql_safely",
                {
                    "sql": test_sql["query"],
                    "database": "mariadb_company"
                }
            )
            
            # Analyze test results
            hypothesis_analysis = await sonnet_client.analyze_hypothesis_results(
                hypothesis=hypothesis,
                test_sql=test_sql["query"],
                results=test_result,
                expected_outcome=test_sql.get("expected_outcome")
            )
            
            return {
                "step_type": "hypothesis_testing",
                "hypothesis": hypothesis,
                "test_sql": test_sql["query"],
                "test_results": test_result,
                "hypothesis_supported": hypothesis_analysis["supported"],
                "confidence": hypothesis_analysis["confidence"],
                "insights": hypothesis_analysis.get("insights", []),
                "follow_up_questions": hypothesis_analysis.get("follow_up", [])
            }
            
        except Exception as e:
            raise InvestigationStepError(f"Hypothesis testing failed: {e}")
    
    async def _execute_pattern_discovery(
        self,
        step_context: Dict,
        investigation_state: Dict,
        mcp_client,
        sonnet_client
    ) -> Dict[str, Any]:
        """Discover patterns in the data using advanced analytics."""
        
        logger.info("Executing pattern discovery...")
        
        try:
            # Generate analytical SQL for pattern discovery
            pattern_sql = await sonnet_client.generate_pattern_discovery_sql(
                query=investigation_state["query"],
                available_data=investigation_state.get("explored_data", {}),
                focus_area=step_context["config"].get("focus_area")
            )
            
            # Execute pattern analysis
            pattern_results = await mcp_client.call_tool(
                "execute_sql_safely",
                {
                    "sql": pattern_sql["query"],
                    "database": "mariadb_company"
                }
            )
            
            # Analyze discovered patterns
            pattern_analysis = await sonnet_client.analyze_discovered_patterns(
                original_query=investigation_state["query"],
                pattern_sql=pattern_sql["query"],
                results=pattern_results
            )
            
            return {
                "step_type": "pattern_discovery",
                "analysis_sql": pattern_sql["query"],
                "raw_results": pattern_results,
                "patterns_found": pattern_analysis["patterns"],
                "statistical_significance": pattern_analysis.get("significance", {}),
                "business_implications": pattern_analysis.get("implications", []),
                "confidence": pattern_analysis.get("confidence", 0.6)
            }
            
        except Exception as e:
            raise InvestigationStepError(f"Pattern discovery failed: {e}")
    
    async def _execute_validation(
        self,
        step_context: Dict,
        investigation_state: Dict,
        mcp_client,
        sonnet_client
    ) -> Dict[str, Any]:
        """Validate findings using different approaches or data sources."""
        
        logger.info("Executing validation...")
        
        try:
            findings_to_validate = step_context["config"].get("findings", [])
            
            validation_results = []
            for finding in findings_to_validate:
                # Generate validation SQL
                validation_sql = await sonnet_client.generate_validation_sql(
                    finding=finding,
                    original_query=investigation_state["query"],
                    available_schema=investigation_state.get("schema_info")
                )
                
                # Execute validation
                validation_result = await mcp_client.call_tool(
                    "execute_sql_safely",
                    {
                        "sql": validation_sql["query"],
                        "database": "mariadb_company"
                    }
                )
                
                # Analyze validation outcome
                validation_analysis = await sonnet_client.analyze_validation_results(
                    original_finding=finding,
                    validation_sql=validation_sql["query"],
                    validation_results=validation_result
                )
                
                validation_results.append({
                    "finding": finding,
                    "validation_sql": validation_sql["query"],
                    "validation_data": validation_result,
                    "is_validated": validation_analysis["validated"],
                    "confidence": validation_analysis["confidence"],
                    "notes": validation_analysis.get("notes", [])
                })
            
            return {
                "step_type": "validation",
                "validated_findings": validation_results,
                "overall_confidence": sum(r["confidence"] for r in validation_results) / len(validation_results) if validation_results else 0,
                "validation_summary": await sonnet_client.summarize_validation_results(validation_results)
            }
            
        except Exception as e:
            raise InvestigationStepError(f"Validation failed: {e}")
    
    async def _execute_optimization(
        self,
        step_context: Dict,
        investigation_state: Dict,
        mcp_client,
        sonnet_client
    ) -> Dict[str, Any]:
        """Optimize queries for better performance."""
        
        logger.info("Executing optimization...")
        
        try:
            queries_to_optimize = [
                step.get("result", {}).get("sql") 
                for step in investigation_state.get("steps", [])
                if step.get("result", {}).get("sql")
            ]
            
            optimization_results = []
            for sql in queries_to_optimize:
                if sql:
                    # Analyze query performance
                    performance_analysis = await mcp_client.call_tool(
                        "analyze_query_performance",
                        {"sql": sql, "database": "mariadb_company"}
                    )
                    
                    # Get optimization suggestions
                    optimization = await sonnet_client.optimize_sql_query(
                        sql=sql,
                        performance_data=performance_analysis
                    )
                    
                    optimization_results.append({
                        "original_sql": sql,
                        "performance_analysis": performance_analysis,
                        "optimized_sql": optimization.get("optimized_sql"),
                        "expected_improvement": optimization.get("improvement_estimate"),
                        "optimization_notes": optimization.get("notes", [])
                    })
            
            return {
                "step_type": "optimization",
                "optimizations": optimization_results,
                "performance_summary": await sonnet_client.summarize_optimizations(optimization_results)
            }
            
        except Exception as e:
            raise InvestigationStepError(f"Optimization failed: {e}")
    
    async def _execute_synthesis(
        self,
        step_context: Dict,
        investigation_state: Dict,
        mcp_client,
        sonnet_client
    ) -> Dict[str, Any]:
        """Synthesize all findings into final insights."""
        
        logger.info("Executing synthesis...")
        
        try:
            # Gather all findings from previous steps
            all_findings = []
            for step in investigation_state.get("steps", []):
                if step.get("result") and step.get("significant_finding"):
                    all_findings.append(step["result"])
            
            # Generate comprehensive synthesis
            synthesis = await sonnet_client.synthesize_investigation_findings(
                original_query=investigation_state["query"],
                all_findings=all_findings,
                context=investigation_state.get("context", {})
            )
            
            return {
                "step_type": "synthesis",
                "findings_synthesized": len(all_findings),
                "key_insights": synthesis["insights"],
                "recommendations": synthesis["recommendations"],
                "confidence_score": synthesis["confidence"],
                "business_impact": synthesis.get("business_impact", {}),
                "next_steps": synthesis.get("next_steps", [])
            }
            
        except Exception as e:
            raise InvestigationStepError(f"Synthesis failed: {e}")
    
    # Analysis Methods
    
    async def _analyze_significance(
        self,
        step_result: Dict,
        investigation_state: Dict,
        sonnet_client
    ) -> bool:
        """Determine if step result contains significant findings."""
        
        try:
            significance_analysis = await sonnet_client.analyze_finding_significance(
                step_result=step_result,
                investigation_context=investigation_state,
                previous_findings=investigation_state.get("findings", [])
            )
            
            return significance_analysis.get("significant", False)
            
        except Exception as e:
            logger.warning(f"Significance analysis failed: {e}")
            # Default heuristics
            confidence = step_result.get("confidence", 0)
            has_insights = bool(step_result.get("insights") or step_result.get("patterns_found"))
            return confidence > 0.7 and has_insights
    
    async def _check_adaptation_needed(
        self,
        step_result: Dict,
        investigation_state: Dict,
        sonnet_client
    ) -> bool:
        """Check if investigation strategy needs adaptation."""
        
        try:
            adaptation_analysis = await sonnet_client.check_adaptation_needed(
                step_result=step_result,
                investigation_state=investigation_state
            )
            
            return adaptation_analysis.get("needs_adaptation", False)
            
        except Exception as e:
            logger.warning(f"Adaptation check failed: {e}")
            # Default heuristics
            has_errors = bool(step_result.get("error"))
            low_confidence = step_result.get("confidence", 1.0) < 0.5
            return has_errors or low_confidence


class SimpleInvestigationEngine:
    """
    Simplified investigation engine for initial implementation.
    
    This engine provides a basic autonomous investigation flow
    while we develop the full MCP integration.
    """
    
    def __init__(self, investigation_id: str):
        self.investigation_id = investigation_id
        self.steps_completed = 0
        self.max_steps = 20
        self.results = {}
        self.context = {}
    
    async def investigate(self, query: str, context: Dict[str, Any] = None):
        """
        Run autonomous investigation for the given query.
        
        This method provides a simplified investigation flow that will
        be enhanced with real MCP tool calls.
        """
        self.context = context or {}
        
        try:
            await self._send_progress("Starting investigation", 0.0)
            
            # Step 1: Analyze the query and plan investigation
            await self._send_progress("Analyzing query", 0.1)
            investigation_plan = await self._plan_investigation(query)
            
            # Step 2: Execute investigation steps
            await self._send_progress("Executing investigation plan", 0.2)
            results = await self._execute_investigation_plan(investigation_plan)
            
            # Step 3: Synthesize final results
            await self._send_progress("Synthesizing results", 0.9)
            final_results = await self._synthesize_results(results)
            
            # Send final results
            await self._send_progress("Investigation completed", 1.0)
            await send_investigation_result(self.investigation_id, final_results)
            
            logger.info(f"Investigation {self.investigation_id} completed successfully")
            
        except Exception as e:
            error_msg = f"Investigation failed: {str(e)}"
            logger.error(f"Investigation {self.investigation_id} failed: {e}")
            await send_investigation_error(self.investigation_id, error_msg)
            raise
    
    async def _plan_investigation(self, query: str) -> Dict[str, Any]:
        """Plan investigation strategy (simplified version)."""
        # Simple planning logic - will be enhanced with LLM integration
        plan = {
            "query": query,
            "steps": [
                {"action": "get_schema", "description": "Discover available tables"},
                {"action": "analyze_query", "description": "Understand what data is needed"},
                {"action": "execute_queries", "description": "Execute SQL to get data"},
                {"action": "analyze_results", "description": "Analyze and interpret results"}
            ]
        }
        return plan
    
    async def _execute_investigation_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute investigation plan (simplified version)."""
        results = {"steps": [], "data": {}}
        total_steps = len(plan["steps"])
        
        for i, step in enumerate(plan["steps"]):
            progress = 0.2 + (0.7 * i / total_steps)
            await self._send_progress(step["description"], progress)
            
            try:
                step_result = await self._execute_investigation_step(step)
                results["steps"].append({
                    "step": step,
                    "result": step_result,
                    "success": True
                })
                results["data"][step["action"]] = step_result
                
            except Exception as e:
                results["steps"].append({
                    "step": step,
                    "error": str(e),
                    "success": False
                })
                continue
        
        return results
    
    async def _execute_investigation_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single investigation step (mock implementation)."""
        action = step["action"]
        
        # Simulate work
        await asyncio.sleep(0.5)
        
        if action == "get_schema":
            return {
                "tables": [
                    {"name": "customers", "columns": ["id", "name", "email", "created_at"]},
                    {"name": "orders", "columns": ["id", "customer_id", "amount", "order_date"]},
                    {"name": "products", "columns": ["id", "name", "price", "category"]}
                ]
            }
        elif action == "analyze_query":
            return {
                "intent": "customer_analysis",
                "entities": ["customers", "orders"],
                "metrics": ["count", "revenue"]
            }
        elif action == "execute_queries":
            return {
                "queries_executed": [
                    {"sql": "SELECT COUNT(*) FROM customers", "result": [{"count": 1500}]},
                    {"sql": "SELECT SUM(amount) FROM orders", "result": [{"total": 25340.50}]}
                ]
            }
        elif action == "analyze_results":
            return {
                "insights": [
                    "Total customers: 1,500",
                    "Total revenue: $25,340.50",
                    "Average revenue per customer: $16.89"
                ]
            }
        else:
            raise ValueError(f"Unknown investigation step: {action}")
    
    async def _synthesize_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize final investigation results."""
        successful_steps = [step for step in results["steps"] if step["success"]]
        
        # Extract insights
        insights = []
        if "analyze_results" in results["data"]:
            insights = results["data"]["analyze_results"].get("insights", [])
        
        return {
            "investigation_id": self.investigation_id,
            "summary": {
                "total_steps": len(results["steps"]),
                "successful_steps": len(successful_steps)
            },
            "insights": insights,
            "data": results["data"],
            "completed_at": datetime.utcnow().isoformat()
        }
    
    async def _send_progress(self, message: str, progress: float):
        """Send progress update via WebSocket."""
        await send_investigation_progress(
            self.investigation_id,
            message,
            progress,
            {"steps_completed": self.steps_completed}
        )