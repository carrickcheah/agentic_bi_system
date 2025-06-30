"""
Business Intelligence Layer - Phase 2: Strategy Planning Components
Self-contained intelligence module with business domain expertise and strategic planning.
"""

from .domain_expert import (
    DomainExpert, BusinessDomain, AnalysisType, BusinessIntent
)
from .complexity_analyzer import (
    ComplexityAnalyzer, ComplexityLevel, InvestigationMethodology, ComplexityScore
)
from .business_context import (
    BusinessContextAnalyzer, UserRole, OrganizationalContext, 
    UserProfile, OrganizationalProfile, ContextualStrategy
)
from .hypothesis_generator import (
    HypothesisGenerator, HypothesisType, Hypothesis, HypothesisSet
)
from .pattern_recognizer import (
    PatternRecognizer, PatternType, DiscoveredPattern, PatternLibraryUpdate
)
from .config import settings
from .intelligence_logging import setup_logger, performance_monitor

__all__ = [
    # Core Components
    "DomainExpert",
    "ComplexityAnalyzer", 
    "BusinessContextAnalyzer",
    "HypothesisGenerator",
    "PatternRecognizer",
    
    # Domain Expert Types
    "BusinessDomain",
    "AnalysisType", 
    "BusinessIntent",
    
    # Complexity Analyzer Types
    "ComplexityLevel",
    "InvestigationMethodology",
    "ComplexityScore",
    
    # Business Context Types
    "UserRole",
    "OrganizationalContext",
    "UserProfile",
    "OrganizationalProfile", 
    "ContextualStrategy",
    
    # Hypothesis Generator Types
    "HypothesisType",
    "Hypothesis",
    "HypothesisSet",
    
    # Pattern Recognizer Types
    "PatternType",
    "DiscoveredPattern", 
    "PatternLibraryUpdate",
    
    # Configuration and Utilities
    "settings",
    "setup_logger",
    "performance_monitor",
]

__version__ = "1.0.0"
__description__ = "Business Intelligence Layer for Autonomous SQL Investigation System"