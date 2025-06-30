"""
Business Context - User and Organizational Context Analysis Component
Self-contained context-aware strategy adaptation for manufacturing environments.
Zero external dependencies beyond module boundary.
"""

from enum import Enum
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import json
from pathlib import Path
from datetime import datetime, timezone

try:
    from .config import settings
    from .intelligence_logging import setup_logger, performance_monitor
    from .domain_expert import BusinessDomain, AnalysisType, BusinessIntent
    from .complexity_analyzer import ComplexityLevel, InvestigationMethodology
except ImportError:
    from config import settings
    from intelligence_logging import setup_logger, performance_monitor
    from domain_expert import BusinessDomain, AnalysisType, BusinessIntent
    from complexity_analyzer import ComplexityLevel, InvestigationMethodology


class UserRole(Enum):
    """User roles with different investigation preferences."""
    MANAGER = "manager"
    ANALYST = "analyst"
    ENGINEER = "engineer"
    EXECUTIVE = "executive"
    OPERATOR = "operator"
    SPECIALIST = "specialist"


class OrganizationalContext(Enum):
    """Organizational context types affecting investigation approach."""
    STARTUP = "startup"          # Fast, iterative, pragmatic
    ENTERPRISE = "enterprise"    # Thorough, validated, documented
    LEAN_MANUFACTURING = "lean_manufacturing"  # Efficient, waste-focused
    SIX_SIGMA = "six_sigma"     # Statistical, precise, methodical
    AGILE = "agile"             # Adaptive, collaborative, rapid
    TRADITIONAL = "traditional"  # Established, systematic, hierarchical


@dataclass
class UserProfile:
    """User profile with preferences and capabilities."""
    user_id: str
    role: UserRole
    experience_level: str  # novice, intermediate, expert
    preferred_detail_level: str  # summary, detailed, comprehensive
    preferred_speed: str  # fast, balanced, thorough
    domain_expertise: List[BusinessDomain]
    investigation_history: Dict[str, int]  # methodology -> count
    success_rate: float  # historical success rate
    last_activity: datetime


@dataclass
class OrganizationalProfile:
    """Organizational profile affecting investigation strategies."""
    organization_id: str
    context_type: OrganizationalContext
    primary_domains: List[BusinessDomain]
    investigation_patterns: Dict[str, float]  # pattern -> success rate
    resource_constraints: Dict[str, str]  # constraint type -> level
    methodology_preferences: Dict[InvestigationMethodology, float]
    time_zone: str
    business_hours: Dict[str, str]  # start_hour, end_hour
    quality_standards: Dict[str, float]  # standard -> threshold


@dataclass
class ContextualStrategy:
    """Context-adapted investigation strategy."""
    base_methodology: InvestigationMethodology
    adapted_methodology: InvestigationMethodology
    context_adjustments: Dict[str, Union[str, float, int]]
    user_preferences: Dict[str, Union[str, float]]
    organizational_constraints: Dict[str, str]
    estimated_timeline: Dict[str, int]  # phase -> minutes
    communication_style: str  # concise, detailed, technical, executive
    deliverable_format: str  # dashboard, report, analysis, recommendations


class BusinessContextAnalyzer:
    """
    Business context analyzer for user and organizational adaptation.
    Adapts investigation strategies based on user roles and organizational context.
    """
    
    def __init__(self):
        self.logger = setup_logger("business_context")
        self._user_profiles = {}
        self._organizational_profiles = {}
        self._role_preferences = self._load_role_preferences()
        self._context_adaptations = self._load_context_adaptations()
        self._default_profiles = self._load_default_profiles()
        
        self.logger.info("Business Context Analyzer initialized with adaptation models")
    
    def _load_role_preferences(self) -> Dict[UserRole, Dict[str, Union[str, float]]]:
        """Load role-based investigation preferences."""
        return {
            UserRole.MANAGER: {
                "speed_preference": 0.8,      # Prefer faster results
                "detail_preference": 0.4,     # Less detailed analysis
                "automation_preference": 0.9, # High automation
                "communication_style": "executive",
                "preferred_format": "dashboard",
                "max_duration_minutes": 30,
                "focus_areas": ["business_impact", "recommendations", "timeline"]
            },
            
            UserRole.ANALYST: {
                "speed_preference": 0.4,      # Prefer thorough analysis
                "detail_preference": 0.9,     # Highly detailed analysis
                "automation_preference": 0.6, # Moderate automation
                "communication_style": "detailed",
                "preferred_format": "analysis",
                "max_duration_minutes": 90,
                "focus_areas": ["methodology", "data_quality", "statistical_significance"]
            },
            
            UserRole.ENGINEER: {
                "speed_preference": 0.5,      # Balanced approach
                "detail_preference": 0.8,     # Good level of detail
                "automation_preference": 0.7, # Good automation
                "communication_style": "technical",
                "preferred_format": "report",
                "max_duration_minutes": 60,
                "focus_areas": ["technical_details", "implementation", "validation"]
            },
            
            UserRole.EXECUTIVE: {
                "speed_preference": 0.9,      # Very fast results
                "detail_preference": 0.3,     # High-level summary
                "automation_preference": 0.8, # High automation
                "communication_style": "executive",
                "preferred_format": "dashboard",
                "max_duration_minutes": 15,
                "focus_areas": ["strategic_impact", "recommendations", "roi"]
            },
            
            UserRole.OPERATOR: {
                "speed_preference": 0.7,      # Fairly quick results
                "detail_preference": 0.5,     # Moderate detail
                "automation_preference": 0.8, # High automation
                "communication_style": "concise",
                "preferred_format": "dashboard",
                "max_duration_minutes": 20,
                "focus_areas": ["operational_impact", "immediate_actions", "alerts"]
            },
            
            UserRole.SPECIALIST: {
                "speed_preference": 0.3,      # Very thorough analysis
                "detail_preference": 1.0,     # Maximum detail
                "automation_preference": 0.5, # Lower automation for control
                "communication_style": "technical",
                "preferred_format": "comprehensive_report",
                "max_duration_minutes": 120,
                "focus_areas": ["deep_analysis", "root_causes", "expert_insights"]
            }
        }
    
    def _load_context_adaptations(self) -> Dict[OrganizationalContext, Dict[str, Union[str, float]]]:
        """Load organizational context adaptations."""
        return {
            OrganizationalContext.STARTUP: {
                "speed_multiplier": 1.3,      # 30% faster
                "resource_efficiency": 0.8,   # Use fewer resources
                "risk_tolerance": 0.7,        # Higher risk tolerance
                "methodology_bias": "rapid_response",
                "communication_style": "concise",
                "validation_level": 0.6       # Less validation needed
            },
            
            OrganizationalContext.ENTERPRISE: {
                "speed_multiplier": 0.8,      # 20% slower for thoroughness
                "resource_efficiency": 1.2,   # Use more resources
                "risk_tolerance": 0.3,        # Lower risk tolerance
                "methodology_bias": "systematic_analysis",
                "communication_style": "detailed",
                "validation_level": 0.9       # High validation requirements
            },
            
            OrganizationalContext.LEAN_MANUFACTURING: {
                "speed_multiplier": 1.1,      # Slightly faster
                "resource_efficiency": 0.7,   # Minimize waste
                "risk_tolerance": 0.5,        # Moderate risk tolerance
                "methodology_bias": "rapid_response",
                "communication_style": "focused",
                "validation_level": 0.7,      # Sufficient validation
                "waste_focus": True           # Focus on waste elimination
            },
            
            OrganizationalContext.SIX_SIGMA: {
                "speed_multiplier": 0.7,      # 30% slower for precision
                "resource_efficiency": 1.3,   # Use more resources for accuracy
                "risk_tolerance": 0.2,        # Very low risk tolerance
                "methodology_bias": "multi_phase_root_cause",
                "communication_style": "statistical",
                "validation_level": 0.95,     # Very high validation
                "statistical_rigor": True     # Emphasize statistical analysis
            },
            
            OrganizationalContext.AGILE: {
                "speed_multiplier": 1.2,      # 20% faster
                "resource_efficiency": 0.9,   # Efficient resource use
                "risk_tolerance": 0.6,        # Moderate-high risk tolerance
                "methodology_bias": "systematic_analysis",
                "communication_style": "collaborative",
                "validation_level": 0.7,      # Iterative validation
                "iteration_focus": True       # Emphasize iterative improvement
            },
            
            OrganizationalContext.TRADITIONAL: {
                "speed_multiplier": 0.9,      # Slightly slower
                "resource_efficiency": 1.0,   # Standard resource use
                "risk_tolerance": 0.4,        # Lower risk tolerance
                "methodology_bias": "systematic_analysis",
                "communication_style": "formal",
                "validation_level": 0.8,      # High validation standards
                "hierarchy_respect": True     # Respect organizational hierarchy
            }
        }
    
    def _load_default_profiles(self) -> Dict[str, Union[UserProfile, OrganizationalProfile]]:
        """Load default profiles for unknown users/organizations."""
        default_user = UserProfile(
            user_id="default",
            role=UserRole.ANALYST,
            experience_level="intermediate",
            preferred_detail_level="detailed",
            preferred_speed="balanced",
            domain_expertise=[BusinessDomain.OPERATIONS],
            investigation_history={},
            success_rate=0.7,
            last_activity=datetime.now(timezone.utc)
        )
        
        default_org = OrganizationalProfile(
            organization_id="default",
            context_type=OrganizationalContext.TRADITIONAL,
            primary_domains=[BusinessDomain.PRODUCTION, BusinessDomain.QUALITY],
            investigation_patterns={},
            resource_constraints={"time": "moderate", "compute": "standard"},
            methodology_preferences={},
            time_zone="UTC",
            business_hours={"start_hour": "08:00", "end_hour": "17:00"},
            quality_standards={"confidence_threshold": 0.8, "validation_level": 0.8}
        )
        
        return {"user": default_user, "organization": default_org}
    
    @performance_monitor("context_analysis")
    def analyze_context(
        self,
        business_intent: BusinessIntent,
        complexity_level: ComplexityLevel,
        base_methodology: InvestigationMethodology,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None
    ) -> ContextualStrategy:
        """
        Analyze context and adapt investigation strategy.
        
        Args:
            business_intent: Classified business intent
            complexity_level: Determined complexity level
            base_methodology: Base investigation methodology
            user_id: User identifier for personalization
            organization_id: Organization identifier for context
            
        Returns:
            ContextualStrategy with adapted approach
        """
        # Get user and organizational profiles
        user_profile = self.get_user_profile(user_id)
        org_profile = self.get_organizational_profile(organization_id)
        
        # Apply role-based adaptations
        role_adaptations = self._apply_role_adaptations(
            user_profile, business_intent, complexity_level
        )
        
        # Apply organizational context adaptations
        context_adaptations = self._apply_context_adaptations(
            org_profile, business_intent, base_methodology
        )
        
        # Determine adapted methodology
        adapted_methodology = self._adapt_methodology(
            base_methodology, role_adaptations, context_adaptations
        )
        
        # Calculate timeline adjustments
        timeline_adjustments = self._calculate_timeline_adjustments(
            complexity_level, role_adaptations, context_adaptations
        )
        
        # Determine communication style and format
        communication_style, deliverable_format = self._determine_communication_preferences(
            user_profile, org_profile, business_intent
        )
        
        self.logger.info(
            f"Context analyzed: {user_profile.role.value} user, "
            f"{org_profile.context_type.value} org, "
            f"methodology: {base_methodology.value} -> {adapted_methodology.value}"
        )
        
        return ContextualStrategy(
            base_methodology=base_methodology,
            adapted_methodology=adapted_methodology,
            context_adjustments=context_adaptations,
            user_preferences=role_adaptations,
            organizational_constraints=self._extract_constraints(org_profile),
            estimated_timeline=timeline_adjustments,
            communication_style=communication_style,
            deliverable_format=deliverable_format
        )
    
    def _apply_role_adaptations(
        self,
        user_profile: UserProfile,
        business_intent: BusinessIntent,
        complexity_level: ComplexityLevel
    ) -> Dict[str, Union[str, float]]:
        """Apply role-based adaptations to investigation strategy."""
        role_prefs = self._role_preferences[user_profile.role]
        adaptations = role_prefs.copy()
        
        # Adjust based on user experience
        experience_multipliers = {
            "novice": {"detail_preference": 1.2, "speed_preference": 0.8},
            "intermediate": {"detail_preference": 1.0, "speed_preference": 1.0},
            "expert": {"detail_preference": 0.9, "speed_preference": 1.1}
        }
        
        if user_profile.experience_level in experience_multipliers:
            multipliers = experience_multipliers[user_profile.experience_level]
            for key, multiplier in multipliers.items():
                if key in adaptations:
                    adaptations[key] = min(adaptations[key] * multiplier, 1.0)
        
        # Adjust based on domain expertise
        if business_intent.primary_domain in user_profile.domain_expertise:
            adaptations["domain_familiarity"] = 1.0
            adaptations["confidence_boost"] = 0.1
        else:
            adaptations["domain_familiarity"] = 0.6
            adaptations["confidence_boost"] = -0.1
        
        # Adjust based on historical success rate
        if user_profile.success_rate > 0.8:
            adaptations["success_confidence"] = 1.0
        elif user_profile.success_rate < 0.6:
            adaptations["success_confidence"] = 0.7
            adaptations["detail_preference"] = min(adaptations["detail_preference"] * 1.1, 1.0)
        
        return adaptations
    
    def _apply_context_adaptations(
        self,
        org_profile: OrganizationalProfile,
        business_intent: BusinessIntent,
        base_methodology: InvestigationMethodology
    ) -> Dict[str, Union[str, float]]:
        """Apply organizational context adaptations."""
        context_adaptations = self._context_adaptations[org_profile.context_type].copy()
        
        # Apply resource constraints
        for constraint, level in org_profile.resource_constraints.items():
            if constraint == "time" and level == "limited":
                context_adaptations["speed_multiplier"] = min(
                    context_adaptations["speed_multiplier"] * 1.2, 2.0
                )
            elif constraint == "compute" and level == "limited":
                context_adaptations["resource_efficiency"] = max(
                    context_adaptations["resource_efficiency"] * 0.8, 0.5
                )
        
        # Apply methodology preferences
        if base_methodology in org_profile.methodology_preferences:
            pref_score = org_profile.methodology_preferences[base_methodology]
            if pref_score > 0.8:
                context_adaptations["methodology_confidence"] = 1.0
            elif pref_score < 0.5:
                context_adaptations["methodology_confidence"] = 0.7
        
        # Apply business hours constraints
        current_hour = datetime.now().hour
        start_hour = int(org_profile.business_hours["start_hour"].split(":")[0])
        end_hour = int(org_profile.business_hours["end_hour"].split(":")[0])
        
        if not (start_hour <= current_hour <= end_hour):
            context_adaptations["off_hours_mode"] = True
            context_adaptations["resource_efficiency"] = min(
                context_adaptations["resource_efficiency"] * 0.8, 1.0
            )
        
        return context_adaptations
    
    def _adapt_methodology(
        self,
        base_methodology: InvestigationMethodology,
        role_adaptations: Dict[str, Union[str, float]],
        context_adaptations: Dict[str, Union[str, float]]
    ) -> InvestigationMethodology:
        """Adapt methodology based on role and context preferences."""
        
        # Start with base methodology
        adapted = base_methodology
        
        # Check for role-based methodology preferences
        role_pref = role_adaptations.get("methodology_bias")
        context_pref = context_adaptations.get("methodology_bias")
        
        # Priority order: context > role > base
        if context_pref:
            try:
                adapted = InvestigationMethodology(context_pref)
            except ValueError:
                pass  # Keep base if invalid
        elif role_pref:
            try:
                adapted = InvestigationMethodology(role_pref)
            except ValueError:
                pass  # Keep base if invalid
        
        # Additional logic for specific adaptations
        speed_pref = role_adaptations.get("speed_preference", 0.5)
        speed_mult = context_adaptations.get("speed_multiplier", 1.0)
        
        # If high speed preference and multiplier, prefer faster methodologies
        if speed_pref > 0.8 and speed_mult > 1.1:
            if adapted == InvestigationMethodology.MULTI_PHASE_ROOT_CAUSE:
                adapted = InvestigationMethodology.SYSTEMATIC_ANALYSIS
            elif adapted == InvestigationMethodology.SYSTEMATIC_ANALYSIS:
                adapted = InvestigationMethodology.RAPID_RESPONSE
        
        # If low risk tolerance, prefer more thorough methodologies
        risk_tolerance = context_adaptations.get("risk_tolerance", 0.5)
        if risk_tolerance < 0.4:
            if adapted == InvestigationMethodology.RAPID_RESPONSE:
                adapted = InvestigationMethodology.SYSTEMATIC_ANALYSIS
            elif adapted == InvestigationMethodology.SYSTEMATIC_ANALYSIS:
                adapted = InvestigationMethodology.MULTI_PHASE_ROOT_CAUSE
        
        return adapted
    
    def _calculate_timeline_adjustments(
        self,
        complexity_level: ComplexityLevel,
        role_adaptations: Dict[str, Union[str, float]],
        context_adaptations: Dict[str, Union[str, float]]
    ) -> Dict[str, int]:
        """Calculate timeline adjustments based on context."""
        
        # Base timeline estimates (minutes)
        base_timelines = {
            ComplexityLevel.SIMPLE: {"analysis": 3, "validation": 1, "reporting": 1},
            ComplexityLevel.ANALYTICAL: {"analysis": 8, "validation": 3, "reporting": 4},
            ComplexityLevel.COMPUTATIONAL: {"analysis": 25, "validation": 10, "reporting": 10},
            ComplexityLevel.INVESTIGATIVE: {"analysis": 60, "validation": 20, "reporting": 15}
        }
        
        base_timeline = base_timelines[complexity_level]
        adjusted_timeline = {}
        
        # Apply speed adjustments
        speed_multiplier = context_adaptations.get("speed_multiplier", 1.0)
        role_speed = role_adaptations.get("speed_preference", 0.5)
        
        # Combined speed factor
        speed_factor = speed_multiplier * (0.5 + role_speed * 0.5)
        
        for phase, base_minutes in base_timeline.items():
            adjusted_minutes = int(base_minutes / speed_factor)
            
            # Apply phase-specific adjustments
            if phase == "validation":
                validation_level = context_adaptations.get("validation_level", 0.8)
                adjusted_minutes = int(adjusted_minutes * validation_level)
            
            adjusted_timeline[phase] = max(adjusted_minutes, 1)  # Minimum 1 minute
        
        return adjusted_timeline
    
    def _determine_communication_preferences(
        self,
        user_profile: UserProfile,
        org_profile: OrganizationalProfile,
        business_intent: BusinessIntent
    ) -> Tuple[str, str]:
        """Determine communication style and deliverable format."""
        
        # Get role preferences
        role_prefs = self._role_preferences[user_profile.role]
        role_style = role_prefs.get("communication_style", "detailed")
        role_format = role_prefs.get("preferred_format", "report")
        
        # Get context preferences
        context_adaptations = self._context_adaptations[org_profile.context_type]
        context_style = context_adaptations.get("communication_style", "detailed")
        
        # Combine preferences (context takes precedence)
        communication_style = context_style if context_style != "detailed" else role_style
        
        # Adapt format based on urgency and complexity
        deliverable_format = role_format
        if business_intent.urgency_level == "critical":
            deliverable_format = "dashboard"  # Quick visual format for urgent needs
        elif business_intent.analysis_type == AnalysisType.PRESCRIPTIVE:
            deliverable_format = "executive_summary"
        
        return communication_style, deliverable_format
    
    def _extract_constraints(self, org_profile: OrganizationalProfile) -> Dict[str, str]:
        """Extract organizational constraints for strategy planning."""
        constraints = {}
        
        # Resource constraints
        constraints.update(org_profile.resource_constraints)
        
        # Quality standards
        for standard, threshold in org_profile.quality_standards.items():
            constraints[f"quality_{standard}"] = str(threshold)
        
        # Context-specific constraints
        context_type = org_profile.context_type
        if context_type == OrganizationalContext.LEAN_MANUFACTURING:
            constraints["waste_minimization"] = "required"
        elif context_type == OrganizationalContext.SIX_SIGMA:
            constraints["statistical_rigor"] = "required"
        elif context_type == OrganizationalContext.STARTUP:
            constraints["rapid_iteration"] = "preferred"
        
        return constraints
    
    def get_user_profile(self, user_id: Optional[str]) -> UserProfile:
        """Get user profile or default if not found."""
        if user_id and user_id in self._user_profiles:
            return self._user_profiles[user_id]
        return self._default_profiles["user"]
    
    def get_organizational_profile(self, organization_id: Optional[str]) -> OrganizationalProfile:
        """Get organizational profile or default if not found."""
        if organization_id and organization_id in self._organizational_profiles:
            return self._organizational_profiles[organization_id]
        return self._default_profiles["organization"]
    
    def update_user_profile(self, user_profile: UserProfile) -> None:
        """Update user profile with new information."""
        self._user_profiles[user_profile.user_id] = user_profile
        self.logger.info(f"Updated user profile: {user_profile.user_id}")
    
    def update_organizational_profile(self, org_profile: OrganizationalProfile) -> None:
        """Update organizational profile with new information."""
        self._organizational_profiles[org_profile.organization_id] = org_profile
        self.logger.info(f"Updated organizational profile: {org_profile.organization_id}")
    
    def learn_from_investigation(
        self,
        user_id: str,
        organization_id: str,
        methodology: InvestigationMethodology,
        success: bool,
        duration_minutes: int
    ) -> None:
        """Learn from investigation results to improve future recommendations."""
        
        # Update user profile
        user_profile = self.get_user_profile(user_id)
        if methodology.value not in user_profile.investigation_history:
            user_profile.investigation_history[methodology.value] = 0
        user_profile.investigation_history[methodology.value] += 1
        
        # Update success rate (exponential moving average)
        alpha = 0.2  # Learning rate
        new_success = 1.0 if success else 0.0
        user_profile.success_rate = (
            alpha * new_success + (1 - alpha) * user_profile.success_rate
        )
        
        user_profile.last_activity = datetime.now(timezone.utc)
        self.update_user_profile(user_profile)
        
        # Update organizational profile
        org_profile = self.get_organizational_profile(organization_id)
        if methodology not in org_profile.methodology_preferences:
            org_profile.methodology_preferences[methodology] = 0.7
        
        # Update methodology preference
        current_pref = org_profile.methodology_preferences[methodology]
        org_profile.methodology_preferences[methodology] = (
            alpha * new_success + (1 - alpha) * current_pref
        )
        
        self.update_organizational_profile(org_profile)
        
        self.logger.info(
            f"Learned from investigation: user={user_id}, "
            f"methodology={methodology.value}, success={success}"
        )


# Standalone execution for testing
if __name__ == "__main__":
    from domain_expert import DomainExpert, BusinessIntent, BusinessDomain, AnalysisType
    from complexity_analyzer import ComplexityAnalyzer, ComplexityLevel, InvestigationMethodology
    
    context_analyzer = BusinessContextAnalyzer()
    
    # Create test user profile
    test_user = UserProfile(
        user_id="test_manager",
        role=UserRole.MANAGER,
        experience_level="intermediate",
        preferred_detail_level="summary",
        preferred_speed="fast",
        domain_expertise=[BusinessDomain.PRODUCTION, BusinessDomain.QUALITY],
        investigation_history={"rapid_response": 5, "systematic_analysis": 3},
        success_rate=0.85,
        last_activity=datetime.now(timezone.utc)
    )
    
    # Create test org profile
    test_org = OrganizationalProfile(
        organization_id="test_manufacturing",
        context_type=OrganizationalContext.LEAN_MANUFACTURING,
        primary_domains=[BusinessDomain.PRODUCTION, BusinessDomain.QUALITY],
        investigation_patterns={},
        resource_constraints={"time": "limited", "compute": "standard"},
        methodology_preferences={
            InvestigationMethodology.RAPID_RESPONSE: 0.9,
            InvestigationMethodology.SYSTEMATIC_ANALYSIS: 0.7
        },
        time_zone="UTC",
        business_hours={"start_hour": "06:00", "end_hour": "18:00"},
        quality_standards={"confidence_threshold": 0.8, "validation_level": 0.7}
    )
    
    # Register profiles
    context_analyzer.update_user_profile(test_user)
    context_analyzer.update_organizational_profile(test_org)
    
    # Test context analysis
    business_intent = BusinessIntent(
        primary_domain=BusinessDomain.PRODUCTION,
        secondary_domains=[],
        analysis_type=AnalysisType.DIAGNOSTIC,
        confidence=0.85,
        key_indicators=["efficiency", "line", "drop"],
        business_metrics=["efficiency %"],
        time_context="last_week",
        urgency_level="high"
    )
    
    strategy = context_analyzer.analyze_context(
        business_intent=business_intent,
        complexity_level=ComplexityLevel.ANALYTICAL,
        base_methodology=InvestigationMethodology.SYSTEMATIC_ANALYSIS,
        user_id="test_manager",
        organization_id="test_manufacturing"
    )
    
    print("Business Context Analysis Test")
    print("=" * 50)
    print(f"Base Methodology: {strategy.base_methodology.value}")
    print(f"Adapted Methodology: {strategy.adapted_methodology.value}")
    print(f"Communication Style: {strategy.communication_style}")
    print(f"Deliverable Format: {strategy.deliverable_format}")
    print(f"Timeline: {strategy.estimated_timeline}")
    print(f"Context Adjustments: {list(strategy.context_adjustments.keys())[:3]}")
    print(f"Organizational Constraints: {list(strategy.organizational_constraints.keys())[:3]}")