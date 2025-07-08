"""
Response Templates - Pre-built responses for non-business intents
Self-contained response generation for greetings, help, and casual conversation.
Zero external dependencies beyond module boundary.
"""

import random
from typing import Dict, List, Optional
from enum import Enum

try:
    from .query_intent_classifier import QueryIntent
    from .intelligence_logging import setup_logger
except ImportError:
    from query_intent_classifier import QueryIntent
    from intelligence_logging import setup_logger


class ResponseTemplate:
    """Generate contextual responses for non-business intents."""
    
    def __init__(self):
        self.logger = setup_logger("response_templates")
        self._templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[QueryIntent, Dict[str, List[str]]]:
        """Initialize response templates by intent type."""
        return {
            QueryIntent.GREETING: {
                "standard": [
                    "Hi! I'm your Business Intelligence Assistant. What business question can I help you analyze today?",
                    "Hello! I'm here to help with your business intelligence needs. What would you like to investigate?",
                    "Hi there! I can help you analyze your business data. What specific question do you have?",
                    "Hello! Ready to dive into your business intelligence? What would you like to explore?",
                    "Hi! I'm your AI business analyst. What data insights can I help you discover today?"
                ],
                "morning": [
                    "Good morning! Ready to start the day with some powerful business insights? What shall we analyze?",
                    "Good morning! I'm here to help with your business intelligence needs. What's on your agenda today?",
                    "Good morning! Let's unlock some valuable insights from your data. What would you like to investigate?"
                ],
                "afternoon": [
                    "Good afternoon! How can I assist with your business analysis today?",
                    "Good afternoon! Ready to dive into your business data? What questions do you have?",
                    "Good afternoon! I'm here to help with your business intelligence needs. What shall we explore?"
                ],
                "evening": [
                    "Good evening! Still working hard? Let me help with your business analysis. What do you need?",
                    "Good evening! I'm here to assist with your business intelligence questions. How can I help?",
                    "Good evening! Ready for some data insights? What would you like to analyze?"
                ],
                "goodbye": [
                    "Goodbye! Feel free to return anytime for business intelligence assistance.",
                    "See you later! I'll be here when you need business analysis support.",
                    "Farewell! Come back anytime for data insights and business intelligence.",
                    "Goodbye! I'm always ready to help with your business questions."
                ],
                "thanks": [
                    "You're welcome! Is there anything else you'd like to analyze or investigate?",
                    "Happy to help! Do you have any other business questions I can assist with?",
                    "My pleasure! Feel free to ask about any business intelligence topics.",
                    "You're welcome! I'm here whenever you need business analysis support."
                ]
            },
            
            QueryIntent.HELP: {
                "capabilities": [
                    """I'm your Business Intelligence Assistant with these capabilities:

📊 **Data Analysis & Reporting**
   • Sales performance analysis
   • Cost analysis and optimization
   • Production efficiency metrics
   • Quality control insights

🔍 **Investigation Types**
   • Customer behavior analysis
   • Supply chain optimization  
   • Financial performance review
   • Operational efficiency assessment

💡 **Example Questions You Can Ask:**
   • "What are our top-selling products this month?"
   • "Show me cost trends for the last quarter"
   • "Analyze customer satisfaction metrics"
   • "What's our production efficiency rate?"

🚀 **5-Phase Analysis Process:**
   1. Query Understanding
   2. Strategic Planning  
   3. Database Coordination
   4. Investigation Execution
   5. Insight Synthesis

Try asking me a specific business question to get started!""",

                    """Here's what I can help you with:

🎯 **Business Intelligence Services:**
   • Manufacturing performance analysis
   • Sales and revenue insights
   • Cost optimization recommendations
   • Quality and safety metrics
   • Customer analytics
   • Supply chain analysis

🔧 **Technical Capabilities:**
   • Connected to MariaDB and PostgreSQL databases
   • Vector search for similar query patterns
   • Real-time data analysis
   • Strategic recommendations

💬 **How to Interact:**
   • Ask specific business questions
   • Request data analysis or reports
   • Seek strategic recommendations
   • Explore trends and patterns

**Sample Questions:**
   • "What's our customer retention rate?"
   • "Show me production costs by department"  
   • "Analyze quality issues from last month"

What business question can I help you with?"""
                ],
                "how_to": [
                    """Here's how to get the most from our Business Intelligence system:

**1. Ask Specific Questions**
   ✅ "What are sales trends for Q4?"
   ❌ "Tell me about sales"

**2. Specify Time Periods**
   ✅ "Show production efficiency for last 3 months"
   ❌ "Show production efficiency"

**3. Include Context**
   ✅ "Customer complaints by product category this year"
   ❌ "Customer complaints"

**4. Request Actionable Insights**
   ✅ "Cost reduction opportunities in manufacturing"
   ❌ "Show me costs"

**Examples of Great Questions:**
   • "Which customers have the highest order values?"
   • "What's causing quality issues in our top products?"
   • "How can we improve on-time delivery rates?"
   • "Show me profit margins by product line"

Ready to analyze your business data? Ask me a specific question!""",

                    """**Quick Start Guide:**

**Step 1:** Think about what you want to know
**Step 2:** Ask a specific business question  
**Step 3:** I'll analyze your data through 5 phases
**Step 4:** Review insights and recommendations

**Popular Analysis Types:**
   📈 Sales Performance: "What are our best-selling products?"
   💰 Cost Analysis: "Where can we reduce manufacturing costs?"
   👥 Customer Insights: "Which customers are at risk of churning?"
   🏭 Operations: "How efficient is our production line?"
   📊 Quality Metrics: "What's our defect rate trend?"

**Pro Tips:**
   • Be specific about time periods
   • Ask for actionable recommendations
   • Follow up with deeper questions

What business challenge shall we tackle first?"""
                ]
            },
            
            QueryIntent.CASUAL: {
                "general": [
                    "I appreciate the friendly chat! I'm most helpful when we're discussing business intelligence topics. What data insights can I help you discover?",
                    "Thanks for the conversation! I'm designed to help with business analysis. Is there a specific business question you'd like to explore?",
                    "I enjoy chatting, but I'm at my best analyzing business data! What business challenge can I help you tackle?",
                    "That's nice! I'm here primarily for business intelligence support. What business metrics would you like to investigate?",
                    "I appreciate that! I'm most effective when helping with business analysis. What data questions do you have?"
                ],
                "redirect": [
                    "While I enjoy conversation, my expertise is in business intelligence. What business data would you like to analyze today?",
                    "I'm designed to help with business questions and data analysis. What business insights are you looking for?",
                    "Thanks for that! I'm here to help with business intelligence. What specific business challenge can I assist with?",
                    "I appreciate the chat! Let's focus on business intelligence - what data analysis can I help you with?",
                    "That's thoughtful! I'm most helpful with business questions. What business metrics should we explore?"
                ]
            }
        }
    
    def generate_response(self, intent: QueryIntent, original_query: str = "") -> str:
        """Generate appropriate response based on intent and query context."""
        try:
            if intent not in self._templates:
                return self._get_default_business_redirect()
            
            # Special handling for greetings based on query content
            if intent == QueryIntent.GREETING:
                return self._generate_greeting_response(original_query.lower())
            
            # Special handling for help requests
            elif intent == QueryIntent.HELP:
                return self._generate_help_response(original_query.lower())
            
            # General casual conversation
            elif intent == QueryIntent.CASUAL:
                return self._generate_casual_response()
            
            else:
                return self._get_default_business_redirect()
                
        except Exception as e:
            self.logger.error(f"Error generating response for {intent}: {e}")
            return self._get_default_business_redirect()
    
    def _generate_greeting_response(self, query: str) -> str:
        """Generate contextual greeting response."""
        templates = self._templates[QueryIntent.GREETING]
        
        # Time-based greetings
        if "good morning" in query:
            return random.choice(templates["morning"])
        elif "good afternoon" in query:
            return random.choice(templates["afternoon"])
        elif "good evening" in query:
            return random.choice(templates["evening"])
        elif any(word in query for word in ["bye", "goodbye", "see you", "farewell"]):
            return random.choice(templates["goodbye"])
        elif any(word in query for word in ["thanks", "thank you"]):
            return random.choice(templates["thanks"])
        else:
            return random.choice(templates["standard"])
    
    def _generate_help_response(self, query: str) -> str:
        """Generate contextual help response."""
        templates = self._templates[QueryIntent.HELP]
        
        # Detailed help for specific requests
        if any(phrase in query for phrase in ["how do i", "how to", "show me how"]):
            return random.choice(templates["how_to"])
        else:
            return random.choice(templates["capabilities"])
    
    def _generate_casual_response(self) -> str:
        """Generate casual conversation response."""
        templates = self._templates[QueryIntent.CASUAL]
        
        # Randomize between general and redirect responses
        if random.random() < 0.7:  # 70% chance for redirect
            return random.choice(templates["redirect"])
        else:
            return random.choice(templates["general"])
    
    def _get_default_business_redirect(self) -> str:
        """Default response for unknown intents."""
        return ("I'm your Business Intelligence Assistant, ready to help analyze your business data. "
                "What specific business question would you like me to investigate?")
    
    def get_business_examples(self) -> List[str]:
        """Get example business questions for help responses."""
        return [
            "What are our top-selling products this quarter?",
            "Show me cost trends for the manufacturing department",
            "Analyze customer satisfaction scores by region",
            "What's our inventory turnover rate?",
            "Which customers have the highest lifetime value?",
            "Show me quality defect rates by product line",
            "What are our most profitable product categories?",
            "Analyze production efficiency trends",
            "Show me customer acquisition costs by channel",
            "What's our on-time delivery performance?"
        ]
    
    def format_business_encouragement(self) -> str:
        """Generate encouragement to ask business questions."""
        examples = self.get_business_examples()
        selected_examples = random.sample(examples, 3)
        
        return f"""Here are some example business questions to get you started:

• {selected_examples[0]}
• {selected_examples[1]}
• {selected_examples[2]}

What business challenge would you like to explore?"""


# Singleton instance
_template_instance = None

def get_response_template() -> ResponseTemplate:
    """Get singleton response template instance."""
    global _template_instance
    if _template_instance is None:
        _template_instance = ResponseTemplate()
    return _template_instance