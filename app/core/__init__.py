"""
AUTONOMOUS INTELLIGENCE CORE

This module contains the core business intelligence engine that implements
the three revolutionary principles:

1. Business Intelligence First, Technology Second
2. Autonomous Investigation, Not Query Translation  
3. Organizational Learning, Not Individual Tools

The core engine orchestrates five-phase investigations that transform
business questions into strategic insights.
"""

from .workflow import AgenticWorkflow, process_question

# Clean, simple exports
__all__ = [
    "AgenticWorkflow",
    "process_question"
]

# Note: Other components (InvestigationEngine, QueryProcessor, etc.) exist in legacy
# but are integrated through the AutonomousBusinessAnalyst orchestrator