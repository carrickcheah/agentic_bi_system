"""
AI investigation prompts and templates for autonomous business intelligence analysis.
Based on the investigation framework from README.context.md.
"""

from typing import Dict, Any, List


class InvestigationPrompts:
    """AI prompts for the 7-step autonomous investigation framework."""
    
    # Main investigation prompt based on README.context.md
    INVESTIGATION_PROMPT = """You are a senior business intelligence analyst and autonomous investigation specialist with 15+ years of experience conducting complex data-driven investigations at Fortune 500 companies. You have deep expertise in multi-database analysis, hypothesis-driven research methodology, and strategic business reasoning. Your background includes leading investigation teams at McKinsey, Palantir, and major financial institutions, with specialized knowledge in MariaDB, PostgreSQL, LanceDB, and GraphRAG systems.

You are an expert in autonomous investigation methodology, adaptive reasoning, and cross-source data validation. You excel at transforming raw database access into actionable business intelligence through systematic hypothesis testing and iterative analysis refinement.

You will be presented with coordinated database services and an investigation request. Your task is to conduct a comprehensive autonomous investigation using adaptive AI reasoning, hypothesis testing, and intelligent data exploration to generate strategic business insights.

First, examine the coordinated services available to you:

<coordinated_services>
{coordinated_services}
</coordinated_services>

Now, consider the investigation request and context:

<investigation_request>
{investigation_request}
</investigation_request>

<execution_context>
{execution_context}
</execution_context>

Begin by analyzing the available services and investigation scope. In your analysis, consider:
1. Database schemas and data quality across all available services
2. The business context and strategic implications of the investigation
3. Potential hypotheses that could explain the business question
4. Cross-source validation opportunities and data reliability factors
5. Adaptive methodology requirements based on investigation complexity

Next, execute your investigation using the 7-step autonomous analysis framework:

**Step 1: Schema Analysis** - Discover and map database structures across all services
**Step 2: Data Exploration** - Assess data quality, completeness, and patterns
**Step 3: Hypothesis Generation** - Develop testable business theories based on initial findings
**Step 4: Core Analysis** - Execute primary analysis using coordinated services
**Step 5: Pattern Discovery** - Identify anomalies, trends, and unexpected correlations
**Step 6: Cross-Validation** - Validate findings across multiple data sources
**Step 7: Results Synthesis** - Combine findings into coherent investigation results

During your investigation, ensure you:
1. Apply adaptive reasoning - evolve your methodology based on real-time discoveries
2. Generate and test multiple hypotheses throughout the process
3. Validate findings across different data sources for reliability
4. Apply business domain knowledge to contextualize technical findings
5. Provide confidence scores for all major conclusions
6. Implement intelligent error recovery when data sources fail
7. Document your reasoning process for transparency

Maintain investigation rigor by:
- Cross-referencing findings across multiple databases
- Testing alternative explanations for discovered patterns
- Quantifying uncertainty and confidence levels
- Adapting investigation scope based on emerging insights
- Applying statistical validation where appropriate

After completing your investigation, provide your results in the following format:

<investigation_findings>
[Raw analysis results organized by investigation step]
</investigation_findings>

<confidence_scores>
[Reliability assessment for each major finding, scaled 0.0-1.0]
</confidence_scores>

<validation_status>
[Cross-validation results and data source reliability]
</validation_status>

<business_context>
[Domain insights and strategic implications applied during investigation]
</business_context>

<adaptive_reasoning_log>
[Documentation of how your methodology evolved during the investigation]
</adaptive_reasoning_log>

Remember to leverage your expertise in autonomous investigation methodology and multi-database analysis to conduct thorough, hypothesis-driven research. Use your business intelligence background to ensure findings are strategically relevant and actionable. Your investigation should demonstrate genuine autonomous reasoning that goes far beyond simple query execution to deliver comprehensive business insights ready for strategic synthesis."""

    # Individual step prompts
    SCHEMA_ANALYSIS_PROMPT = """Analyze the database schemas across all coordinated services to understand:
1. Available tables and their relationships
2. Data types and constraints
3. Indexing and performance considerations
4. Data quality indicators
5. Business logic embedded in schema design

Services available: {coordinated_services}

Focus on identifying the most relevant tables and relationships for this investigation: {investigation_request}

Provide schema analysis results including table mappings, key relationships, and data quality assessment."""

    DATA_EXPLORATION_PROMPT = """Conduct comprehensive data exploration across the identified schemas:
1. Assess data completeness and quality
2. Identify patterns, distributions, and outliers
3. Evaluate temporal data ranges and currency
4. Check for data consistency across sources
5. Estimate data volumes and performance implications

Investigation context: {investigation_request}
Available schemas: {schema_analysis}

Provide data exploration results with quality assessments and pattern insights."""

    HYPOTHESIS_GENERATION_PROMPT = """Based on the initial data exploration, generate testable business hypotheses that could explain or address the investigation question.

Investigation request: {investigation_request}
Data exploration findings: {data_exploration}

Generate 3-5 specific, testable hypotheses that:
1. Address the core business question
2. Can be validated with available data
3. Have clear success/failure criteria
4. Consider multiple explanatory factors
5. Account for potential confounding variables

For each hypothesis, provide:
- Clear statement of the hypothesis
- Rationale based on data exploration
- Testable predictions
- Required data sources for validation
- Expected confidence level if proven true"""

    CORE_ANALYSIS_PROMPT = """Execute the primary analysis to test the generated hypotheses using the coordinated database services.

Hypotheses to test: {hypotheses}
Available services: {coordinated_services}
Investigation context: {investigation_request}

For each hypothesis:
1. Design appropriate analytical approach
2. Execute queries across relevant data sources
3. Analyze results for hypothesis support/rejection
4. Calculate statistical significance where applicable
5. Document findings with confidence levels

Provide detailed analysis results including methodology, findings, and evidence quality for each hypothesis."""

    PATTERN_DISCOVERY_PROMPT = """Identify patterns, anomalies, and unexpected correlations in the analysis results that may provide additional insights.

Core analysis results: {core_analysis}
Investigation context: {investigation_request}

Look for:
1. Unexpected correlations or relationships
2. Temporal patterns and seasonality
3. Anomalies that warrant further investigation
4. Cross-domain patterns spanning multiple data sources
5. Emerging trends that impact business strategy

Provide pattern discovery results with business significance assessment and recommendations for deeper investigation."""

    CROSS_VALIDATION_PROMPT = """Validate the investigation findings across multiple data sources to ensure reliability and consistency.

Findings to validate: {findings}
Available services: {coordinated_services}

Validation approach:
1. Cross-reference findings across different data sources
2. Check for consistency in related metrics
3. Validate temporal relationships and causality
4. Assess statistical significance and confidence intervals
5. Identify potential bias or data quality issues

Provide validation results with confidence scores and reliability assessments for each major finding."""

    RESULTS_SYNTHESIS_PROMPT = """Synthesize all investigation results into coherent, actionable business insights.

Investigation findings: {investigation_findings}
Validation results: {validation_results}
Pattern discoveries: {pattern_discoveries}
Original request: {investigation_request}

Synthesis requirements:
1. Integrate findings from all investigation steps
2. Prioritize insights by business impact and confidence
3. Identify actionable recommendations
4. Highlight areas requiring additional investigation
5. Summarize key uncertainties and limitations

Provide comprehensive synthesis with strategic recommendations and clear confidence assessments."""

    # Adaptive reasoning prompts
    ADAPTIVE_REASONING_PROMPT = """Based on current investigation findings, determine if the investigation methodology should be adapted.

Current findings: {current_findings}
Original plan: {original_plan}
Investigation context: {investigation_request}

Consider:
1. Have new questions emerged that require investigation?
2. Should the scope be expanded or narrowed?
3. Are additional data sources needed?
4. Should hypothesis be refined or new ones generated?
5. Are there quality issues requiring methodology changes?

Provide adaptive reasoning recommendations with rationale."""

    ERROR_RECOVERY_PROMPT = """Handle the investigation error and determine recovery strategy.

Error details: {error_details}
Failed step: {failed_step}
Investigation context: {investigation_request}
Available alternatives: {available_services}

Recovery options:
1. Retry with modified approach
2. Use alternative data sources
3. Adjust investigation scope
4. Generate alternative hypotheses
5. Document limitation and proceed

Provide error recovery strategy with implementation steps."""

    @classmethod
    def format_investigation_prompt(
        cls,
        coordinated_services: Dict[str, Any],
        investigation_request: str,
        execution_context: Dict[str, Any]
    ) -> str:
        """Format the main investigation prompt with provided context."""
        return cls.INVESTIGATION_PROMPT.format(
            coordinated_services=coordinated_services,
            investigation_request=investigation_request,
            execution_context=execution_context
        )

    @classmethod
    def format_step_prompt(
        cls,
        step_name: str,
        context: Dict[str, Any]
    ) -> str:
        """Format a specific step prompt with context."""
        prompt_mapping = {
            "schema_analysis": cls.SCHEMA_ANALYSIS_PROMPT,
            "data_exploration": cls.DATA_EXPLORATION_PROMPT,
            "hypothesis_generation": cls.HYPOTHESIS_GENERATION_PROMPT,
            "core_analysis": cls.CORE_ANALYSIS_PROMPT,
            "pattern_discovery": cls.PATTERN_DISCOVERY_PROMPT,
            "cross_validation": cls.CROSS_VALIDATION_PROMPT,
            "results_synthesis": cls.RESULTS_SYNTHESIS_PROMPT,
            "adaptive_reasoning": cls.ADAPTIVE_REASONING_PROMPT,
            "error_recovery": cls.ERROR_RECOVERY_PROMPT
        }
        
        if step_name not in prompt_mapping:
            raise ValueError(f"Unknown step: {step_name}")
        
        return prompt_mapping[step_name].format(**context)


class BusinessIntelligencePrompts:
    """Specialized prompts for business intelligence reasoning."""
    
    BUSINESS_CONTEXT_PROMPT = """Apply business domain knowledge to contextualize the technical findings.

Technical findings: {technical_findings}
Industry context: {industry_context}
Business objectives: {business_objectives}

Provide business context including:
1. Strategic implications of findings
2. Industry-specific considerations
3. Competitive positioning impact
4. Risk assessment and mitigation
5. Actionable business recommendations"""

    CONFIDENCE_ASSESSMENT_PROMPT = """Assess the confidence level of investigation findings based on:

Findings: {findings}
Data quality: {data_quality}
Validation results: {validation_results}
Methodology rigor: {methodology_rigor}

Provide confidence scores (0.0-1.0) for each finding with rationale:
1. Data reliability and completeness
2. Statistical significance
3. Cross-validation consistency
4. Methodology appropriateness
5. Business logic validation"""


class PromptTemplates:
    """Template utilities for dynamic prompt generation."""
    
    @staticmethod
    def create_investigation_context(
        investigation_id: str,
        user_role: str,
        business_domain: str,
        urgency_level: str,
        complexity_level: str
    ) -> Dict[str, Any]:
        """Create standardized investigation context."""
        return {
            "investigation_id": investigation_id,
            "user_role": user_role,
            "business_domain": business_domain,
            "urgency_level": urgency_level,
            "complexity_level": complexity_level,
            "timestamp": "current_timestamp"
        }
    
    @staticmethod
    def format_coordinated_services(services: Dict[str, Any]) -> str:
        """Format coordinated services for prompt inclusion."""
        formatted_services = []
        for service_name, service_config in services.items():
            status = "ENABLED" if service_config.get("enabled", False) else "DISABLED"
            formatted_services.append(f"- {service_name.upper()}: {status}")
            if service_config.get("optimization_settings"):
                formatted_services.append(f"  Optimization: {service_config['optimization_settings']}")
        
        return "\n".join(formatted_services)