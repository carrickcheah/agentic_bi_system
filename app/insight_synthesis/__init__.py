"""
Insight Synthesis module - Strategic insight generation from investigation results.
"""

from .runner import (
    InsightSynthesizer,
    OutputFormat,
    InsightType,
    RecommendationType,
    Insight,
    Recommendation,
    SynthesisResult
)

__all__ = [
    "InsightSynthesizer",
    "OutputFormat",
    "InsightType", 
    "RecommendationType",
    "Insight",
    "Recommendation",
    "SynthesisResult"
]