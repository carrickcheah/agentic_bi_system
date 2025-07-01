"""
Local logging for Investigation module - avoids naming conflicts with stdlib.
Self-contained logging without external dependencies.
"""

import logging
import sys
from typing import Optional


def setup_logger(name: str = "investigation", level: str = "INFO") -> logging.Logger:
    """Set up isolated logger for the investigation module."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:  # Avoid duplicate handlers
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(numeric_level)
        
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(numeric_level)
        
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    
    return logger


class InvestigationLogger:
    """Specialized logger for investigation workflow tracking."""
    
    def __init__(self, investigation_id: Optional[str] = None):
        self.investigation_id = investigation_id or "unknown"
        self.logger = setup_logger(f"investigation_{self.investigation_id}")
    
    def log_step_start(self, step_name: str, step_number: int) -> None:
        """Log the start of an investigation step."""
        self.logger.info(f"Step {step_number}: {step_name} - Starting")
    
    def log_step_complete(self, step_name: str, step_number: int, duration_seconds: float) -> None:
        """Log the completion of an investigation step."""
        self.logger.info(f"Step {step_number}: {step_name} - Completed in {duration_seconds:.2f}s")
    
    def log_step_error(self, step_name: str, step_number: int, error: str) -> None:
        """Log an error during an investigation step."""
        self.logger.error(f"Step {step_number}: {step_name} - Error: {error}")
    
    def log_hypothesis_generated(self, hypothesis: str, confidence: float) -> None:
        """Log generation of a new hypothesis."""
        self.logger.info(f"Hypothesis generated (confidence: {confidence:.2f}): {hypothesis}")
    
    def log_finding(self, finding: str, confidence: float) -> None:
        """Log a new investigation finding."""
        self.logger.info(f"Finding (confidence: {confidence:.2f}): {finding}")
    
    def log_validation_result(self, validation_type: str, result: str, confidence: float) -> None:
        """Log validation results."""
        self.logger.info(f"Validation ({validation_type}): {result} (confidence: {confidence:.2f})")
    
    def log_service_interaction(self, service_name: str, operation: str, status: str) -> None:
        """Log interactions with coordinated services."""
        self.logger.debug(f"Service {service_name}: {operation} - {status}")
    
    def log_adaptive_reasoning(self, reasoning: str) -> None:
        """Log adaptive reasoning decisions."""
        self.logger.info(f"Adaptive Reasoning: {reasoning}")
    
    def log_investigation_summary(self, total_steps: int, total_duration: float, confidence: float) -> None:
        """Log investigation completion summary."""
        self.logger.info(
            f"Investigation completed: {total_steps} steps, "
            f"{total_duration:.2f}s total, "
            f"overall confidence: {confidence:.2f}"
        )


# Default logger instance
logger = setup_logger("investigation", "INFO")