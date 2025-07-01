"""
Insight Synthesis Module - Phase 5: Strategic Business Intelligence Generation
Transforms raw investigation findings into strategic business intelligence.
"""

from .runner import InsightSynthesizer
from .config import InsightSynthesisSettings, settings

__all__ = [
    "InsightSynthesizer",
    "InsightSynthesisSettings", 
    "settings"
]

__version__ = "1.0.0"
__description__ = "Strategic insight synthesis from investigation findings"