"""
Business Pattern Ingestion Component
Handles ingestion of business intelligence patterns.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger("lance_db.pattern_ingestion")


class BusinessPatternIngestion:
    """Handles ingestion of business intelligence patterns."""
    
    def __init__(self, patterns_dir: Optional[Path] = None):
        self.patterns_dir = patterns_dir or Path(__file__).parent.parent / "patterns"
        self.logger = logger
        
    async def load_patterns(self) -> List[Dict[str, Any]]:
        """Load business patterns from JSON files."""
        patterns = []
        
        if not self.patterns_dir.exists():
            logger.warning(f"Patterns directory not found: {self.patterns_dir}")
            return patterns
            
        # Load all JSON files in patterns directory
        for pattern_file in self.patterns_dir.glob("*.json"):
            try:
                with open(pattern_file, 'r') as f:
                    file_patterns = json.load(f)
                    
                # Handle both single pattern and list of patterns
                if isinstance(file_patterns, list):
                    patterns.extend(file_patterns)
                elif isinstance(file_patterns, dict):
                    patterns.append(file_patterns)
                    
                logger.info(f"Loaded patterns from {pattern_file.name}")
                
            except Exception as e:
                logger.error(f"Failed to load patterns from {pattern_file}: {e}")
                
        return patterns
    
    def validate_pattern(self, pattern: Dict[str, Any]) -> bool:
        """Validate a business pattern has required fields."""
        required_fields = ["pattern_id", "pattern_name", "sql_template"]
        
        for field in required_fields:
            if field not in pattern:
                logger.warning(f"Pattern missing required field: {field}")
                return False
                
        return True
    
    async def process_patterns(self, patterns: List[Dict[str, Any]]) -> Dict[str, int]:
        """Process and prepare patterns for storage."""
        stats = {
            "total": len(patterns),
            "valid": 0,
            "invalid": 0,
            "skipped": 0
        }
        
        processed_patterns = []
        
        for pattern in patterns:
            if self.validate_pattern(pattern):
                stats["valid"] += 1
                processed_patterns.append(pattern)
            else:
                stats["invalid"] += 1
                
        return stats