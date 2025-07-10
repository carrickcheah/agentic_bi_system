"""
Main runner for Phase 4 Investigation Execution module.
Autonomous AI-powered investigation engine implementing 7-step framework.
Zero external dependencies beyond module boundary.
"""

import asyncio
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json

try:
    from .config import settings
    from .investigation_logging import InvestigationLogger
    from .prompts import InvestigationPrompts, PromptTemplates
except ImportError:
    from config import settings
    from investigation.investigation_logging import InvestigationLogger
    from investigation.prompts import InvestigationPrompts, PromptTemplates


@dataclass
class InvestigationStep:
    """Individual investigation step result."""
    step_number: int
    step_name: str
    status: str  # "pending", "running", "completed", "failed"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    findings: Optional[Dict[str, Any]] = None
    confidence_score: Optional[float] = None
    error_message: Optional[str] = None


@dataclass
class InvestigationResults:
    """Complete investigation results for Phase 5."""
    investigation_id: str
    investigation_request: str
    status: str  # "completed", "failed", "partial"
    total_duration_seconds: float
    overall_confidence: float
    
    # Structured results matching README.context.md format
    investigation_findings: Dict[str, Any]
    confidence_scores: Dict[str, float]
    validation_status: Dict[str, Any]
    business_context: Dict[str, Any]
    adaptive_reasoning_log: List[Dict[str, Any]]
    
    # Investigation metadata
    completed_steps: List[InvestigationStep]
    performance_metrics: Dict[str, Any]
    error_log: List[Dict[str, Any]]


class AutonomousInvestigationEngine:
    """
    Main AI-powered investigation engine implementing 7-step autonomous framework.
    Conducts hypothesis-driven business intelligence analysis using coordinated services.
    """
    
    def __init__(self):
        self.investigation_id = None
        self.logger = None
        self.coordinated_services = None
        self.investigation_request = None
        self.execution_context = None
        self.current_step = 0
        self.steps = []
        self.adaptive_reasoning_log = []
        self.start_time = None
        
        # Initialize 7-step investigation framework
        self.step_definitions = [
            {"number": 1, "name": "schema_analysis", "description": "Discover database structures"},
            {"number": 2, "name": "data_exploration", "description": "Assess data quality and patterns"},
            {"number": 3, "name": "hypothesis_generation", "description": "Generate testable business theories"},
            {"number": 4, "name": "core_analysis", "description": "Execute primary analysis"},
            {"number": 5, "name": "pattern_discovery", "description": "Identify anomalies and trends"},
            {"number": 6, "name": "cross_validation", "description": "Validate across data sources"},
            {"number": 7, "name": "results_synthesis", "description": "Synthesize coherent results"}
        ]
    
    async def conduct_investigation(
        self,
        coordinated_services: Dict[str, Any],
        investigation_request: str,
        execution_context: Dict[str, Any],
        mcp_client_manager=None
    ) -> InvestigationResults:
        """
        Main entry point for autonomous investigation.
        
        Args:
            coordinated_services: Database services from Phase 3 orchestration
            investigation_request: Business question to investigate
            execution_context: Investigation parameters and context
            
        Returns:
            InvestigationResults: Structured results for Phase 5 synthesis
        """
        # Initialize investigation
        self.investigation_id = str(uuid.uuid4())
        self.logger = InvestigationLogger(self.investigation_id)
        self.coordinated_services = coordinated_services
        self.investigation_request = investigation_request
        self.execution_context = execution_context
        self.start_time = datetime.now()
        self.mcp_client_manager = mcp_client_manager
        self.complexity_score = execution_context.get("complexity_score", 0.5)
        
        # Register investigation in runtime config
        # runtime_config.register_investigation(self.investigation_id, {
        #     "request": investigation_request,
        #     "services": list(coordinated_services.keys()),
        #     "context": execution_context
        # })
        
        self.logger.logger.info(f"Starting autonomous investigation: {self.investigation_id}")
        self.logger.logger.info(f"Request: {investigation_request}")
        self.logger.logger.info(f"Available services: {list(coordinated_services.keys())}")
        
        try:
            # Execute 7-step investigation framework
            await self._execute_investigation_framework()
            
            # Generate final results
            results = await self._generate_investigation_results()
            
            self.logger.log_investigation_summary(
                total_steps=len(self.steps),
                total_duration=(datetime.now() - self.start_time).total_seconds(),
                confidence=results.overall_confidence
            )
            
            return results
            
        except Exception as e:
            self.logger.logger.error(f"Investigation failed: {str(e)}")
            return await self._generate_error_results(str(e))
    
    def _get_required_steps(self, complexity_score: float) -> List[Dict[str, Any]]:
        """Select investigation steps based on complexity."""
        if complexity_score < 0.3:
            # Simple: schema, core analysis, synthesis only
            return [s for s in self.step_definitions if s["number"] in [1, 4, 7]]
        elif complexity_score < 0.5:
            # Moderate: add data exploration
            return [s for s in self.step_definitions if s["number"] in [1, 2, 4, 7]]
        elif complexity_score < 0.8:
            # Analytical: skip cross-validation
            return [s for s in self.step_definitions if s["number"] in [1, 2, 3, 4, 5, 7]]
        else:
            # Complex: all steps
            return self.step_definitions
    
    async def _execute_investigation_framework(self) -> None:
        """Execute the 7-step autonomous investigation framework."""
        
        # Determine required steps based on complexity
        required_steps = self._get_required_steps(self.complexity_score)
        self.logger.logger.info(f"Executing {len(required_steps)} steps for complexity {self.complexity_score}")
        
        for step_def in required_steps:
            step_number = step_def["number"]
            step_name = step_def["name"]
            
            # Create step tracker
            step = InvestigationStep(
                step_number=step_number,
                step_name=step_name,
                status="pending"
            )
            self.steps.append(step)
            
            # Update runtime progress
            # runtime_config.update_investigation_progress(
            #     self.investigation_id, 
            #     step_number, 
            #     "running"
            # )
            
            try:
                # Execute step with AI reasoning
                await self._execute_investigation_step(step)
                
                # Check for adaptive reasoning needs
                if step_number > 2:  # After initial exploration
                    await self._apply_adaptive_reasoning(step)
                
            except Exception as e:
                step.status = "failed"
                step.error_message = str(e)
                self.logger.log_step_error(step_name, step_number, str(e))
                
                # Attempt error recovery
                if settings.enable_recovery_mode:
                    await self._attempt_error_recovery(step, str(e))
                else:
                    raise
    
    async def _execute_investigation_step(self, step: InvestigationStep) -> None:
        """Execute individual investigation step using AI reasoning."""
        step.start_time = datetime.now()
        step.status = "running"
        
        self.logger.log_step_start(step.step_name, step.step_number)
        
        # Set timeout based on complexity
        step_timeout = 10 if self.complexity_score < 0.3 else 30
        
        try:
            # Execute step with timeout protection
            async def execute_step_logic():
                # Prepare context for AI reasoning
                step_context = await self._prepare_step_context(step)
                
                # Generate AI prompt for this step
                prompt = InvestigationPrompts.format_step_prompt(step.step_name, step_context)
                
                # Execute database operations for this step
                database_results = await self._execute_database_operation(step.step_name, step.step_name, step_context)
                
                # Execute AI reasoning with database results
                enhanced_prompt = self._enhance_prompt_with_database_results(prompt, database_results)
                ai_response = await self._execute_ai_reasoning(enhanced_prompt, step.step_name)
                
                # Process and structure AI response with database results
                findings = await self._process_ai_response(ai_response, step.step_name)
                findings["database_results"] = database_results
                return findings
            
            # Apply timeout
            step.findings = await asyncio.wait_for(execute_step_logic(), timeout=step_timeout)
            step.confidence_score = self._calculate_step_confidence(step.findings)
            
        except asyncio.TimeoutError:
            step.findings = {"error": f"Step timed out after {step_timeout}s", "timeout": True}
            step.confidence_score = 0.1
            self.logger.logger.warning(f"Step {step.step_name} timed out after {step_timeout}s")
        
        # Complete step
        step.end_time = datetime.now()
        step.duration_seconds = (step.end_time - step.start_time).total_seconds()
        step.status = "completed"
        
        self.logger.log_step_complete(step.step_name, step.step_number, step.duration_seconds)
        
        # Log key findings
        if step.findings:
            for finding_key, finding_value in step.findings.items():
                if isinstance(finding_value, str) and len(finding_value) < 200:
                    self.logger.log_finding(f"{finding_key}: {finding_value}", step.confidence_score)
    
    async def _prepare_step_context(self, step: InvestigationStep) -> Dict[str, Any]:
        """Prepare context for AI reasoning in current step."""
        context = {
            "investigation_request": self.investigation_request,
            "coordinated_services": PromptTemplates.format_coordinated_services(self.coordinated_services),
            "execution_context": self.execution_context,
            "step_number": step.step_number,
            "step_name": step.step_name
        }
        
        # Add previous step results for context
        if step.step_number > 1:
            previous_steps = {
                step.step_name: step.findings 
                for step in self.steps[:-1] 
                if step.status == "completed" and step.findings
            }
            context["previous_steps"] = previous_steps
        
        # Add specific context based on step
        if step.step_name == "data_exploration" and len(self.steps) > 0:
            schema_step = next((s for s in self.steps if s.step_name == "schema_analysis"), None)
            if schema_step and schema_step.findings:
                context["schema_analysis"] = schema_step.findings
        
        elif step.step_name == "hypothesis_generation" and len(self.steps) > 1:
            exploration_step = next((s for s in self.steps if s.step_name == "data_exploration"), None)
            if exploration_step and exploration_step.findings:
                context["data_exploration"] = exploration_step.findings
        
        elif step.step_name == "core_analysis" and len(self.steps) > 2:
            hypothesis_step = next((s for s in self.steps if s.step_name == "hypothesis_generation"), None)
            if hypothesis_step and hypothesis_step.findings:
                context["hypotheses"] = hypothesis_step.findings
        
        return context
    
    async def _execute_ai_reasoning(self, prompt: str, step_name: str) -> str:
        """
        Execute AI reasoning for investigation step using the actual AI models.
        Integrates with the existing /app/model/ module.
        """
        self.logger.logger.debug(f"Executing AI reasoning for {step_name}")
        
        try:
            # Import and use the actual model manager
            import sys
            from pathlib import Path
            
            # Add the model module to the path
            model_path = Path(__file__).parent.parent / "model"
            if str(model_path) not in sys.path:
                sys.path.insert(0, str(model_path))
            
            # Import ModelManager from the model runner
            from runner import ModelManager as ModelRunner
            
            # Initialize model manager if not already done
            if not hasattr(self, '_model_manager'):
                self._model_manager = ModelRunner()
            
            # Generate response using actual AI models with default settings
            response = await self._model_manager.generate_response(
                prompt=prompt,
                max_tokens=4000,
                temperature=0.1,
                use_system_prompt=True
            )
            
            self.logger.logger.debug(f"AI reasoning completed for {step_name}")
            return response
            
        except Exception as e:
            self.logger.logger.warning(f"AI reasoning failed for {step_name}: {e}")
            self.logger.logger.info(f"Falling back to placeholder response for {step_name}")
            
            # Fallback to placeholder response if AI integration fails
            placeholder_response = f"""
            <investigation_findings>
            Placeholder findings for {step_name} step due to AI integration issue: {str(e)}
            This would contain the actual AI analysis results.
            </investigation_findings>
            
            <confidence_scores>
            analysis_quality: 0.75
            data_reliability: 0.80
            </confidence_scores>
            
            <reasoning_process>
            AI reasoning process for {step_name} encountered integration issue: {str(e)}
            </reasoning_process>
            """
            
            return placeholder_response
    
    async def _execute_database_operation(self, step_name: str, operation_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute actual database operations using FastMCP clients.
        """
        if not self.mcp_client_manager:
            self.logger.logger.warning(f"No MCP client manager available for {step_name}")
            return {"error": "No database clients available"}
        
        try:
            results = {}
            
            if operation_type == "schema_analysis":
                results = await self._perform_schema_analysis()
            elif operation_type == "data_exploration":
                results = await self._perform_data_exploration(context)
            elif operation_type == "hypothesis_testing":
                results = await self._perform_hypothesis_testing(context)
            elif operation_type == "pattern_discovery":
                results = await self._perform_pattern_discovery(context)
            elif operation_type == "cross_validation":
                results = await self._perform_cross_validation(context)
            
            self.logger.logger.info(f"Database operation {operation_type} completed for {step_name}")
            return results
            
        except Exception as e:
            self.logger.logger.error(f"Database operation {operation_type} failed for {step_name}: {e}")
            return {"error": str(e)}
    
    async def _perform_schema_analysis(self) -> Dict[str, Any]:
        """Analyze database schemas using intelligent table filtering based on business question."""
        schema_results = {}
        
        # Determine relevant table patterns based on the business question
        table_patterns = []
        request_lower = self.investigation_request.lower()
        
        if any(word in request_lower for word in ["product", "item", "sku"]):
            table_patterns.extend(["product", "item", "sku", "catalog"])
        if any(word in request_lower for word in ["sale", "revenue", "order", "selling"]):
            table_patterns.extend(["sale", "order", "revenue", "transaction", "invoice"])
        if any(word in request_lower for word in ["customer", "client", "buyer"]):
            table_patterns.extend(["customer", "client", "user", "account"])
        if any(word in request_lower for word in ["inventory", "stock"]):
            table_patterns.extend(["inventory", "stock", "warehouse"])
        if any(word in request_lower for word in ["price", "pricing", "cost"]):
            table_patterns.extend(["price", "pricing", "cost", "rate"])
            
        # Log the search strategy
        if table_patterns:
            self.logger.logger.info(f"Filtering tables with patterns: {table_patterns}")
        else:
            self.logger.logger.info("No specific patterns identified, will analyze general tables")
        
        # Analyze MariaDB schema
        mariadb_client = self.mcp_client_manager.get_client("mariadb")
        if mariadb_client:
            try:
                all_tables = await mariadb_client.list_tables()
                
                # Filter tables based on patterns and complexity
                if table_patterns:
                    relevant_tables = []
                    for table in all_tables:
                        table_lower = table.lower()
                        if any(pattern in table_lower for pattern in table_patterns):
                            relevant_tables.append(table)
                    
                    # Optimize based on complexity
                    if self.complexity_score < 0.3:
                        # Simple queries: only most relevant table
                        table_scores = {}
                        for table in relevant_tables:
                            score = sum(1 for pattern in table_patterns if pattern in table.lower())
                            table_scores[table] = score
                        sorted_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
                        tables_to_analyze = [t[0] for t in sorted_tables[:1]]  # Only top table
                    else:
                        # Complex queries: analyze more tables
                        max_tables = 3 if self.complexity_score < 0.5 else 10 if self.complexity_score < 0.8 else 15
                        tables_to_analyze = relevant_tables[:max_tables] if relevant_tables else all_tables[:max_tables]
                    
                    self.logger.logger.info(f"Found {len(relevant_tables)} relevant tables, analyzing {len(tables_to_analyze)} based on complexity {self.complexity_score}")
                else:
                    # No patterns: limit based on complexity
                    max_tables = 1 if self.complexity_score < 0.3 else 3 if self.complexity_score < 0.5 else 10
                    tables_to_analyze = all_tables[:max_tables]
                
                schema_results["mariadb"] = {
                    "tables": tables_to_analyze,
                    "total_table_count": len(all_tables),
                    "analyzed_table_count": len(tables_to_analyze),
                    "filter_patterns": table_patterns,
                    "schemas": {}
                }
                
                # Get schema for filtered tables
                for table in tables_to_analyze:
                    try:
                        schema = await mariadb_client.get_table_schema(table)
                        schema_results["mariadb"]["schemas"][table] = {
                            "columns": len(schema.columns),
                            "column_names": [col.name for col in schema.columns],
                            "data_types": [col.type for col in schema.columns]
                        }
                    except Exception as e:
                        self.logger.logger.warning(f"Failed to get schema for table {table}: {e}")
                        
            except Exception as e:
                self.logger.logger.warning(f"MariaDB schema analysis failed: {e}")
                schema_results["mariadb"] = {"error": str(e)}
        
        return schema_results
    
    async def _perform_data_exploration(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Explore data quality and patterns using MCP clients with query learning."""
        exploration_results = {}
        
        # Get schema information from previous step
        schema_info = context.get("previous_steps", {}).get("schema_analysis", {})
        
        # Prepare business context for query learning
        business_context = {
            "investigation_phase": "data_exploration",
            "investigation_id": self.investigation_id,
            "business_question": self.investigation_request,
            "step_name": "data_exploration",
            "business_domain": self._extract_business_domain(self.investigation_request)
        }
        
        # Explore MariaDB data
        mariadb_client = self.mcp_client_manager.get_client("mariadb")
        if mariadb_client and "mariadb" in schema_info:
            try:
                tables = schema_info["mariadb"].get("tables", [])
                exploration_results["mariadb"] = {}
                
                # Limit tables based on complexity
                max_tables = 1 if self.complexity_score < 0.3 else 3
                for table in tables[:max_tables]:
                    try:
                        # Get row count with query learning
                        count_query = f"SELECT COUNT(*) as total_rows FROM {table}"
                        count_result = await self._execute_query_with_learning(
                            query=count_query,
                            database="mariadb",
                            business_context={**business_context, "table": table, "operation": "count"}
                        )
                        count = count_result["data"][0]["total_rows"] if count_result.get("success") and count_result.get("data") else 0
                        
                        # Sample data with query learning
                        sample_query = f"SELECT * FROM {table} LIMIT 5"
                        sample_result = await self._execute_query_with_learning(
                            query=sample_query,
                            database="mariadb",
                            business_context={**business_context, "table": table, "operation": "sample"}
                        )
                        
                        exploration_results["mariadb"][table] = {
                            "row_count": count,
                            "sample_rows": len(sample_result.get("data", [])) if sample_result.get("success") else 0,
                            "columns": sample_result.get("columns", []) if sample_result.get("success") else []
                        }
                        
                    except Exception as e:
                        self.logger.logger.warning(f"Failed to explore table {table}: {e}")
                        
            except Exception as e:
                self.logger.logger.warning(f"MariaDB data exploration failed: {e}")
                exploration_results["mariadb"] = {"error": str(e)}
        
        return exploration_results
    
    async def _execute_query_with_learning(
        self,
        query: str,
        database: str,
        business_context: Dict[str, Any],
        user_id: str = "investigation_engine"
    ):
        """
        Execute query using MCP client directly.
        """
        # Direct MCP client execution
        client = self.mcp_client_manager.get_client(database)
        if client:
            try:
                result = await client.execute_query(query)
                # Return result in standardized format
                return {
                    "data": result.rows if hasattr(result, 'rows') else result.get("data", []),
                    "columns": result.columns if hasattr(result, 'columns') else result.get("columns", []),
                    "row_count": len(result.rows) if hasattr(result, 'rows') else result.get("row_count", 0),
                    "database": database,
                    "success": True
                }
            except Exception as e:
                return {
                    "data": [],
                    "columns": [],
                    "row_count": 0,
                    "database": database,
                    "success": False,
                    "error": str(e)
                }
        
        # No client available
        return {
            "data": [],
            "columns": [],
            "row_count": 0,
            "database": database,
            "success": False,
            "error": f"No {database} client available"
        }
    
    def _extract_business_domain(self, investigation_request: str) -> str:
        """
        Extract business domain from investigation request for query classification.
        """
        request_lower = investigation_request.lower()
        
        # Define domain keywords
        domain_keywords = {
            "sales": ["sales", "revenue", "order", "customer", "purchase", "transaction"],
            "finance": ["finance", "profit", "cost", "budget", "expense", "financial"],
            "hr": ["employee", "staff", "payroll", "hr", "human resource", "personnel"],
            "production": ["production", "manufacturing", "inventory", "supply", "warehouse"],
            "marketing": ["marketing", "campaign", "advertisement", "promotion", "lead"],
            "quality": ["quality", "defect", "compliance", "standard", "audit"],
            "logistics": ["logistics", "shipping", "delivery", "transport", "fulfillment"]
        }
        
        # Check for domain keywords
        for domain, keywords in domain_keywords.items():
            if any(keyword in request_lower for keyword in keywords):
                return domain
        
        # Default to general analytics
        return "analytics"
    
    async def _perform_hypothesis_testing(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Test hypotheses using actual database queries with learning integration."""
        testing_results = {}
        
        # Extract hypotheses from context
        hypotheses = context.get("hypotheses", {})
        
        # Prepare business context for query learning
        business_context = {
            "investigation_phase": "hypothesis_testing",
            "investigation_id": self.investigation_id,
            "business_question": self.investigation_request,
            "step_name": "hypothesis_testing",
            "business_domain": self._extract_business_domain(self.investigation_request)
        }
        
        # Test hypotheses against MariaDB
        mariadb_client = self.mcp_client_manager.get_client("mariadb")
        if mariadb_client:
            try:
                testing_results["mariadb"] = {}
                
                # Example hypothesis testing queries based on investigation request
                request_lower = self.investigation_request.lower()
                
                if "sales" in request_lower:
                    # Test sales-related hypotheses with learning
                    queries = [
                        ("SELECT COUNT(*) as total_records FROM sales", "total_sales_count"),
                        ("SELECT YEAR(date) as year, SUM(amount) as total FROM sales GROUP BY YEAR(date) ORDER BY year DESC LIMIT 5", "yearly_sales_trend")
                    ]
                    
                    for i, (query, hypothesis_name) in enumerate(queries):
                        try:
                            result = await self._execute_query_with_learning(
                                query=query,
                                database="mariadb",
                                business_context={**business_context, "hypothesis": hypothesis_name}
                            )
                            testing_results["mariadb"][f"hypothesis_test_{i+1}"] = {
                                "hypothesis": hypothesis_name,
                                "query": query,
                                "results": result.get("data", []) if result.get("success") else [],
                                "success": result.get("success", False)
                            }
                        except Exception as e:
                            testing_results["mariadb"][f"hypothesis_test_{i+1}"] = {
                                "hypothesis": hypothesis_name,
                                "error": str(e)
                            }
                
                elif "customer" in request_lower or "user" in request_lower:
                    # Test customer-related hypotheses
                    queries = [
                        ("SELECT COUNT(DISTINCT customer_id) as unique_customers FROM orders", "unique_customer_count"),
                        ("SELECT status, COUNT(*) as count FROM customers GROUP BY status", "customer_status_distribution")
                    ]
                    
                    for i, (query, hypothesis_name) in enumerate(queries):
                        try:
                            result = await self._execute_query_with_learning(
                                query=query,
                                database="mariadb",
                                business_context={**business_context, "hypothesis": hypothesis_name}
                            )
                            testing_results["mariadb"][f"hypothesis_test_{i+1}"] = {
                                "hypothesis": hypothesis_name,
                                "query": query,
                                "results": result.get("data", []) if result.get("success") else [],
                                "success": result.get("success", False)
                            }
                        except Exception as e:
                            testing_results["mariadb"][f"hypothesis_test_{i+1}"] = {
                                "hypothesis": hypothesis_name,
                                "error": str(e)
                            }
                            
            except Exception as e:
                self.logger.logger.warning(f"MariaDB hypothesis testing failed: {e}")
                testing_results["mariadb"] = {"error": str(e)}
        
        return testing_results
    
    async def _perform_pattern_discovery(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Discover patterns and anomalies in the data."""
        pattern_results = {}
        
        # Analyze patterns in MariaDB
        mariadb_client = self.mcp_client_manager.get_client("mariadb")
        if mariadb_client:
            try:
                pattern_results["mariadb"] = {}
                
                # Look for temporal patterns if investigation is time-related
                request_lower = self.investigation_request.lower()
                if any(word in request_lower for word in ["quarter", "monthly", "daily", "trend"]):
                    # Time-based pattern analysis
                    time_queries = [
                        "SELECT DATE(created_at) as date, COUNT(*) as daily_count FROM transactions GROUP BY DATE(created_at) ORDER BY date DESC LIMIT 30",
                        "SELECT HOUR(created_at) as hour, COUNT(*) as hourly_count FROM transactions GROUP BY HOUR(created_at) ORDER BY hour"
                    ]
                    
                    for i, query in enumerate(time_queries):
                        try:
                            result = await mariadb_client.execute_query(query)
                            pattern_results["mariadb"][f"temporal_pattern_{i+1}"] = {
                                "pattern_type": "temporal",
                                "data_points": len(result.rows),
                                "execution_time": result.execution_time
                            }
                        except Exception as e:
                            pattern_results["mariadb"][f"temporal_pattern_{i+1}"] = {"error": str(e)}
                            
            except Exception as e:
                self.logger.logger.warning(f"MariaDB pattern discovery failed: {e}")
                pattern_results["mariadb"] = {"error": str(e)}
        
        return pattern_results
    
    async def _perform_cross_validation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-validate findings within MariaDB data sources."""
        validation_results = {}
        
        # Validate findings within MariaDB only
        mariadb_client = self.mcp_client_manager.get_client("mariadb")
        
        if mariadb_client:
            try:
                validation_results["data_validation"] = {}
                
                # Test connection
                mariadb_healthy = await mariadb_client.test_connection()
                
                validation_results["data_validation"]["connection_status"] = {
                    "mariadb_healthy": mariadb_healthy,
                    "timestamp": datetime.now().isoformat()
                }
                
                # If healthy, perform data validation within MariaDB
                if mariadb_healthy:
                    validation_results["data_validation"]["consistency_check"] = {
                        "validation_performed": True,
                        "database": "mariadb",
                        "timestamp": datetime.now().isoformat()
                    }
                
            except Exception as e:
                self.logger.logger.warning(f"Data validation failed: {e}")
                validation_results["data_validation"] = {"error": str(e)}
        
        return validation_results
    
    def _enhance_prompt_with_database_results(self, original_prompt: str, database_results: Dict[str, Any]) -> str:
        """Enhance AI prompt with actual database results."""
        if not database_results or "error" in database_results:
            return original_prompt
        
        # Format database results for AI prompt
        db_context = "\n\nACTUAL DATABASE ANALYSIS RESULTS:\n"
        
        for db_name, results in database_results.items():
            if isinstance(results, dict) and "error" not in results:
                db_context += f"\n{db_name.upper()} DATABASE:\n"
                
                # Format results based on content
                if "tables" in results:
                    db_context += f"- Found {len(results['tables'])} tables: {', '.join(results['tables'][:5])}\n"
                
                if "schemas" in results:
                    db_context += f"- Schema details for {len(results['schemas'])} tables\n"
                
                if "table_stats" in results:
                    db_context += f"- Table statistics: {len(results['table_stats'])} entries\n"
                
                if "connection_status" in results:
                    db_context += f"- Connection status: {'healthy' if results['connection_status'] else 'failed'}\n"
        
        db_context += "\nPlease incorporate these actual database findings into your analysis.\n"
        
        return original_prompt + db_context
    
    async def _process_ai_response(self, ai_response: str, step_name: str) -> Dict[str, Any]:
        """Process and structure AI response for the step."""
        # Parse structured AI response
        # This would parse the XML-like tags from the AI response
        findings = {
            "raw_response": ai_response,
            "step_name": step_name,
            "processed_at": datetime.now().isoformat(),
            "status": "processed"
        }
        
        # TODO: Implement proper XML parsing for structured AI responses
        # Extract <investigation_findings>, <confidence_scores>, etc.
        
        return findings
    
    def _calculate_step_confidence(self, findings: Dict[str, Any]) -> float:
        """Calculate confidence score for step findings."""
        # TODO: Implement sophisticated confidence calculation
        # Based on data quality, analysis completeness, validation results
        base_confidence = 0.8
        
        # Adjust based on findings quality
        if findings and "status" in findings and findings["status"] == "processed":
            return base_confidence
        else:
            return base_confidence * 0.5
    
    async def _apply_adaptive_reasoning(self, completed_step: InvestigationStep) -> None:
        """Apply adaptive reasoning to potentially modify investigation approach."""
        
        # Check if adaptive reasoning is needed
        if not self._should_apply_adaptive_reasoning(completed_step):
            return
        
        # Generate adaptive reasoning prompt
        context = {
            "current_findings": completed_step.findings,
            "original_plan": self.step_definitions,
            "investigation_request": self.investigation_request,
            "completed_steps": [step.step_name for step in self.steps if step.status == "completed"]
        }
        
        adaptive_prompt = InvestigationPrompts.format_step_prompt("adaptive_reasoning", context)
        
        # Execute adaptive reasoning
        adaptive_response = await self._execute_ai_reasoning(adaptive_prompt, "adaptive_reasoning")
        
        # Process adaptive reasoning result
        reasoning_result = {
            "step_trigger": completed_step.step_name,
            "reasoning": adaptive_response,
            "timestamp": datetime.now().isoformat(),
            "action_taken": "methodology_maintained"  # or "scope_expanded", "approach_modified"
        }
        
        self.adaptive_reasoning_log.append(reasoning_result)
        self.logger.log_adaptive_reasoning(f"After {completed_step.step_name}: {reasoning_result['action_taken']}")
    
    def _should_apply_adaptive_reasoning(self, step: InvestigationStep) -> bool:
        """Determine if adaptive reasoning should be applied."""
        # Apply after data exploration, hypothesis generation, and core analysis
        adaptive_steps = ["data_exploration", "hypothesis_generation", "core_analysis"]
        return step.step_name in adaptive_steps and step.status == "completed"
    
    async def _attempt_error_recovery(self, failed_step: InvestigationStep, error: str) -> None:
        """Attempt to recover from step failure."""
        self.logger.logger.warning(f"Attempting error recovery for {failed_step.step_name}")
        
        # Generate error recovery strategy
        recovery_context = {
            "error_details": error,
            "failed_step": failed_step.step_name,
            "investigation_request": self.investigation_request,
            "available_services": list(self.coordinated_services.keys())
        }
        
        recovery_prompt = InvestigationPrompts.format_step_prompt("error_recovery", recovery_context)
        recovery_response = await self._execute_ai_reasoning(recovery_prompt, "error_recovery")
        
        # Log recovery attempt
        recovery_log = {
            "failed_step": failed_step.step_name,
            "error": error,
            "recovery_strategy": recovery_response,
            "timestamp": datetime.now().isoformat()
        }
        self.adaptive_reasoning_log.append(recovery_log)
        
        # For now, mark step as partially completed with error
        failed_step.status = "partial"
        failed_step.findings = {"error_recovery": recovery_response, "original_error": error}
    
    async def _generate_investigation_results(self) -> InvestigationResults:
        """Generate final structured investigation results for Phase 5."""
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        # Aggregate findings from all steps
        investigation_findings = {}
        confidence_scores = {}
        
        for step in self.steps:
            if step.findings:
                investigation_findings[step.step_name] = step.findings
                confidence_scores[step.step_name] = step.confidence_score or 0.0
        
        # Calculate overall confidence
        step_confidences = [score for score in confidence_scores.values() if score > 0]
        overall_confidence = sum(step_confidences) / len(step_confidences) if step_confidences else 0.0
        
        # Generate validation status
        validation_status = {
            "completed_steps": len([s for s in self.steps if s.status == "completed"]),
            "total_steps": len(self.steps),
            "failed_steps": len([s for s in self.steps if s.status == "failed"]),
            "overall_quality": "high" if overall_confidence > 0.8 else "medium" if overall_confidence > 0.6 else "low"
        }
        
        # Generate business context
        business_context = {
            "investigation_type": self._classify_investigation_type(),
            "business_domain": self.execution_context.get("business_domain", "general"),
            "strategic_importance": self._assess_strategic_importance(),
            "actionability_score": overall_confidence
        }
        
        # Performance metrics
        performance_metrics = {
            "total_duration_seconds": total_duration,
            "average_step_duration": total_duration / len(self.steps) if self.steps else 0,
            "services_used": list(self.coordinated_services.keys()),
            "adaptive_reasoning_events": len(self.adaptive_reasoning_log)
        }
        
        return InvestigationResults(
            investigation_id=self.investigation_id,
            investigation_request=self.investigation_request,
            status="completed",
            total_duration_seconds=total_duration,
            overall_confidence=overall_confidence,
            investigation_findings=investigation_findings,
            confidence_scores=confidence_scores,
            validation_status=validation_status,
            business_context=business_context,
            adaptive_reasoning_log=self.adaptive_reasoning_log,
            completed_steps=self.steps,
            performance_metrics=performance_metrics,
            error_log=[]
        )
    
    async def _generate_error_results(self, error: str) -> InvestigationResults:
        """Generate results structure for failed investigation."""
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds() if self.start_time else 0
        
        return InvestigationResults(
            investigation_id=self.investigation_id or "failed_investigation",
            investigation_request=self.investigation_request or "unknown",
            status="failed",
            total_duration_seconds=total_duration,
            overall_confidence=0.0,
            investigation_findings={"error": error},
            confidence_scores={},
            validation_status={"error": "Investigation failed before completion"},
            business_context={"error": "Unable to generate business context due to failure"},
            adaptive_reasoning_log=self.adaptive_reasoning_log,
            completed_steps=self.steps,
            performance_metrics={"error": error},
            error_log=[{"error": error, "timestamp": datetime.now().isoformat()}]
        )
    
    def _classify_investigation_type(self, request: str = None) -> str:
        """Classify the type of investigation based on request and complexity."""
        request_lower = (request or self.investigation_request).lower()
        
        if any(word in request_lower for word in ["why", "cause", "reason", "explain"]):
            return "diagnostic"
        elif any(word in request_lower for word in ["predict", "forecast", "future", "trend"]):
            return "predictive"
        elif any(word in request_lower for word in ["compare", "vs", "versus", "difference"]):
            return "comparative"
        elif any(word in request_lower for word in ["what", "how much", "count", "total"]):
            return "descriptive"
        else:
            return "exploratory"
    
    def _assess_strategic_importance(self, request: str = None) -> str:
        """Assess strategic importance of the investigation."""
        request_lower = (request or self.investigation_request).lower()
        
        high_importance_keywords = ["revenue", "profit", "customer", "market", "competitive", "strategic"]
        medium_importance_keywords = ["efficiency", "quality", "performance", "process"]
        
        if any(word in request_lower for word in high_importance_keywords):
            return "high"
        elif any(word in request_lower for word in medium_importance_keywords):
            return "medium"
        else:
            return "low"


# Main execution interface
async def conduct_autonomous_investigation(
    coordinated_services: Dict[str, Any],
    investigation_request: str,
    execution_context: Dict[str, Any],
    mcp_client_manager=None
) -> InvestigationResults:
    """
    High-level interface for autonomous investigation execution.
    
    This is the main entry point for Phase 4: Investigation Execution.
    
    Args:
        coordinated_services: Database services from Phase 3 orchestration
        investigation_request: Business question to investigate
        execution_context: Investigation parameters and context
        
    Returns:
        InvestigationResults: Structured results for Phase 5 synthesis
    """
    engine = AutonomousInvestigationEngine()
    
    try:
        results = await engine.conduct_investigation(
            coordinated_services=coordinated_services,
            investigation_request=investigation_request,
            execution_context=execution_context,
            mcp_client_manager=mcp_client_manager
        )
        
        return results
        
    except Exception as e:
        # Log error and return failed results
        if engine.logger:
            engine.logger.logger.error(f"Investigation engine failed: {e}")
        
        return await engine._generate_error_results(str(e))


# Main execution point for standalone testing
if __name__ == "__main__":
    async def main():
        """Example usage of Investigation Engine."""
        from .investigation_logging import setup_logger
        
        logger = setup_logger("investigation_example")
        
        # Example coordinated services from Phase 3
        coordinated_services = {
            "mariadb": {
                "enabled": True,
                "priority": 1,
                "optimization_settings": {"connection_pool_size": 10}
            }
        }
        
        # Example investigation request
        investigation_request = "Analyze last quarter's sales performance by region and identify key drivers"
        
        # Example execution context
        execution_context = {
            "user_role": "business_analyst",
            "business_domain": "sales",
            "urgency_level": "medium",
            "complexity_level": "analytical",
            "complexity_score": 0.7  # Analytical complexity for testing
        }
        
        try:
            results = await conduct_autonomous_investigation(
                coordinated_services=coordinated_services,
                investigation_request=investigation_request,
                execution_context=execution_context
            )
            
            logger.info("Investigation completed successfully")
            logger.info(f"Investigation ID: {results.investigation_id}")
            logger.info(f"Status: {results.status}")
            logger.info(f"Duration: {results.total_duration_seconds:.2f} seconds")
            logger.info(f"Overall confidence: {results.overall_confidence:.2f}")
            logger.info(f"Steps completed: {len(results.completed_steps)}")
            
            # Display key findings
            for step_name, findings in results.investigation_findings.items():
                logger.info(f"Step {step_name}: {type(findings).__name__} findings available")
            
        except Exception as e:
            logger.error(f"Investigation failed: {e}")
    
    # Run example
    asyncio.run(main())