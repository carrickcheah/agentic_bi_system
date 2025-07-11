"""
Query Intent Classifier - Pre-classification for Greetings vs Business Questions
Self-contained intent classification to filter non-business queries before domain analysis.
Zero external dependencies beyond module boundary.
"""

import re
from enum import Enum
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

try:
    from .intelligence_logging import setup_logger
except ImportError:
    from intelligence_logging import setup_logger


class QueryIntent(Enum):
    """Query intent types for pre-classification."""
    GREETING = "greeting"        # Hello, hi, goodbye, thanks
    HELP = "help"               # Help requests, capability questions
    CASUAL = "casual"           # General conversation, non-business chat
    BUSINESS = "business"       # Actual business intelligence questions
    UNKNOWN = "unknown"         # Unclear intent, default to business


@dataclass
class IntentClassification:
    """Intent classification result."""
    intent: QueryIntent
    confidence: float
    matched_patterns: List[str]
    reasoning: str


class QueryIntentClassifier:
    """Classify query intent before business domain analysis."""
    
    def __init__(self):
        self.logger = setup_logger("query_intent_classifier")
        self._intent_patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> Dict[QueryIntent, Dict[str, List[str]]]:
        """Initialize intent classification patterns."""
        return {
            QueryIntent.GREETING: {
                "exact_matches": [
                    "hi", "hello", "hey", "hiya", "howdy",
                    "good morning", "good afternoon", "good evening",
                    "goodbye", "bye", "see you", "farewell",
                    "thanks", "thank you", "thanks a lot", "much appreciated",
                    "how are you", "how's it going", "what's up"
                ],
                "patterns": [
                    r"^(hi|hello|hey)\s*(there|everyone)?[.!]*$",
                    r"^(good\s+(morning|afternoon|evening|day))[.!]*$",
                    r"^(thanks?|thank\s+you)[.!]*$",
                    r"^(bye|goodbye)[.!]*$",
                    r"^(how\s+(are\s+you|is\s+it\s+going))[?!]*$"
                ],
                "keywords": ["greetings", "salutation", "politeness"]
            },
            
            QueryIntent.HELP: {
                "exact_matches": [
                    "help", "help me", "what can you do", "what are your capabilities",
                    "how do i", "how to", "show me how", "can you help",
                    "what commands", "what functions", "tutorial"
                ],
                "patterns": [
                    r"^(help|assistance)[.!]*$",
                    r"^(what\s+(can\s+you\s+do|are\s+your\s+capabilities))[?!]*$",
                    r"^(how\s+(do\s+i|to))\s+.+[?!]*$",
                    r"^(show\s+me|tell\s+me)\s+(how|what|your).+[?!]*$",
                    r"^(can\s+you\s+(help|show|tell))[?!]*$"
                ],
                "keywords": ["capabilities", "commands", "functions", "tutorial"]
            },
            
            QueryIntent.CASUAL: {
                "exact_matches": [
                    "how's the weather", "nice day", "how was your weekend",
                    "what's new", "anything interesting", "just checking in",
                    "cool", "awesome", "great", "nice", "ok", "okay"
                ],
                "patterns": [
                    r"^(nice|cool|awesome|great|good|ok|okay)[.!]*$",
                    r"^(how's\s+the\s+(weather|day))[?!]*$",
                    r"^(what's\s+(new|up))[?!]*$",
                    r"^(just\s+(checking|saying))[.!]*$"
                ],
                "keywords": ["weather", "personal", "casual"]
            },
            
            QueryIntent.BUSINESS: {
                "exact_matches": [],  # Business queries are detected by content, not exact matches
                "patterns": [
                    r"\b(show\s+me|analyze|report|data|metrics|kpi)\b",
                    r"\b(sales|revenue|profit|cost|production|quality)\b",
                    r"\b(how\s+many|how\s+much|what\s+is\s+the|total)\b",
                    r"\b(performance|efficiency|utilization|trends)\b",
                    r"\b(customers?|products?|orders?|inventory)\b",
                    r"\b(last\s+(month|quarter|year)|this\s+(month|quarter|year))\b"
                ],
                "keywords": [
                    "data", "analysis", "report", "metrics", "kpi", "dashboard",
                    "sales", "revenue", "profit", "cost", "production", "quality",
                    "customers", "products", "orders", "inventory", "finance",
                    "performance", "efficiency", "trends", "forecast"
                ]
            }
        }
    
    def classify_intent(self, query: str) -> IntentClassification:
        """Classify query intent with confidence scoring."""
        if not query or not query.strip():
            return IntentClassification(
                intent=QueryIntent.UNKNOWN,
                confidence=0.0,
                matched_patterns=[],
                reasoning="Empty query"
            )
        
        query_lower = query.lower().strip()
        
        # Remove punctuation for matching
        query_clean = re.sub(r'[^\w\s]', '', query_lower)
        
        # Calculate scores for each intent
        intent_scores = {}
        all_matches = {}
        
        for intent, patterns in self._intent_patterns.items():
            score, matches = self._calculate_intent_score(query_lower, query_clean, patterns)
            if score > 0:
                intent_scores[intent] = score
                all_matches[intent] = matches
        
        # Determine primary intent
        if not intent_scores:
            # No matches - default to business for safety
            return IntentClassification(
                intent=QueryIntent.BUSINESS,
                confidence=0.2,
                matched_patterns=[],
                reasoning="No clear patterns matched, defaulting to business intent"
            )
        
        # Sort by score
        sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
        primary_intent = sorted_intents[0][0]
        confidence = min(sorted_intents[0][1], 1.0)
        matched_patterns = all_matches.get(primary_intent, [])
        
        # Generate reasoning
        reasoning = f"Matched {primary_intent.value} patterns with {confidence:.2f} confidence"
        
        # Apply confidence thresholds
        if confidence < 0.3:
            # Low confidence - default to business
            return IntentClassification(
                intent=QueryIntent.BUSINESS,
                confidence=0.3,
                matched_patterns=[],
                reasoning=f"Low confidence ({confidence:.2f}) for {primary_intent.value}, defaulting to business"
            )
        
        self.logger.info(f"Intent classified: {primary_intent.value} (confidence: {confidence:.3f})")
        
        return IntentClassification(
            intent=primary_intent,
            confidence=confidence,
            matched_patterns=matched_patterns,
            reasoning=reasoning
        )
    
    def _calculate_intent_score(self, query_lower: str, query_clean: str, patterns: Dict[str, List[str]]) -> Tuple[float, List[str]]:
        """Calculate intent score based on patterns."""
        total_score = 0.0
        matches = []
        
        # Exact matches (highest weight: 1.0)
        for exact_match in patterns["exact_matches"]:
            if exact_match == query_lower or exact_match == query_clean:
                total_score += 1.0
                matches.append(f"exact:{exact_match}")
                break  # One exact match is enough
            # Check for repeated greetings like "hihihi" or "hello hello"
            elif query_clean.startswith(exact_match) and len(query_clean) <= len(exact_match) * 3:
                # Check if it's just the greeting repeated (e.g., "hihihi", "hellohello")
                if all(query_clean[i:i+len(exact_match)] == exact_match or 
                       query_clean[i:i+len(exact_match)].startswith(exact_match[:len(query_clean)-i])
                       for i in range(0, len(query_clean), len(exact_match))):
                    total_score += 0.9  # Slightly lower than exact match
                    matches.append(f"repeated:{exact_match}")
                    break
        
        # Pattern matches (medium weight: 0.7)
        for pattern in patterns["patterns"]:
            if re.search(pattern, query_lower, re.IGNORECASE):
                total_score += 0.7
                matches.append(f"pattern:{pattern}")
                break  # One pattern match is enough
        
        # Keyword matches (lower weight: 0.1 per keyword, max 0.5)
        keyword_score = 0.0
        for keyword in patterns["keywords"]:
            if keyword in query_lower:
                keyword_score += 0.1
                matches.append(f"keyword:{keyword}")
        
        total_score += min(keyword_score, 0.5)
        
        # Special boost for very short queries that are likely greetings
        if len(query_clean.split()) <= 2 and total_score > 0:
            if any("greeting" in match for match in matches):
                total_score *= 1.5  # Boost greeting confidence for short queries
        
        return min(total_score, 1.0), matches
    
    def is_business_query(self, query: str) -> bool:
        """Quick check if query is business-related."""
        classification = self.classify_intent(query)
        return classification.intent == QueryIntent.BUSINESS
    
    def is_greeting(self, query: str) -> bool:
        """Quick check if query is a greeting."""
        classification = self.classify_intent(query)
        return classification.intent == QueryIntent.GREETING


# Singleton instance
_classifier_instance = None

def get_intent_classifier() -> QueryIntentClassifier:
    """Get singleton intent classifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = QueryIntentClassifier()
    return _classifier_instance