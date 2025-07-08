"""
Question Classification Utility
Checks if user questions are database/business analytics related.
"""

from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class QuestionChecker:
    """Check if questions are database/business analytics related."""
    
    def __init__(self, model_manager):
        """
        Initialize with model manager.
        
        Args:
            model_manager: ModelManager instance for LLM calls
        """
        self.model_manager = model_manager
        logger.info("QuestionChecker initialized")
    
    async def is_database_question(self, question: str) -> Tuple[bool, str]:
        """
        Check if question is related to database/business analytics.
        
        Args:
            question: User's question
            
        Returns:
            Tuple of (is_database_question, response_message)
        """
        classification_prompt = f"""
You are a question classifier for a business analytics system.

Determine if this question is related to:
- Database queries (SQL, data analysis)
- Business analytics (sales, revenue, metrics, KPIs)
- Data reporting (dashboards, insights)
- Business intelligence (trends, forecasts, performance)

Question: "{question}"

Respond with ONLY:
- "YES" if it's a database/business analytics question
- "NO" if it's unrelated (like personal, general knowledge, coding help, etc.)

Classification:"""

        try:
            # Get classification from LLM
            classification = await self.model_manager.generate_response(
                prompt=classification_prompt,
                max_tokens=10,
                temperature=0,
                use_system_prompt=False
            )
            
            classification = classification.strip().upper()
            is_db_question = "YES" in classification
            
            logger.info(f"Question classification: {classification} for '{question[:50]}...'")
            
            if is_db_question:
                return True, ""
            else:
                # Generate polite rejection message
                rejection_prompt = f"""
The user asked: "{question}"

This is not a database or business analytics question. Write a brief, polite response explaining that this system is designed specifically for business analytics and data queries. Suggest they ask questions about sales data, business metrics, reports, or data analysis instead.

Keep response under 2 sentences.

Response:"""

                rejection_message = await self.model_manager.generate_response(
                    prompt=rejection_prompt,
                    max_tokens=100,
                    temperature=0.7,
                    use_system_prompt=False
                )
                
                return False, rejection_message.strip()
                
        except Exception as e:
            logger.error(f"Error classifying question: {e}")
            # Default to allowing the question if classification fails
            return True, ""
    
    async def check_and_respond(self, question: str) -> Tuple[bool, str]:
        """
        Check question and return appropriate response.
        
        Args:
            question: User's question
            
        Returns:
            Tuple of (should_process, message)
            - If should_process is True, message is empty
            - If should_process is False, message contains polite rejection
        """
        is_valid, message = await self.is_database_question(question)
        
        if is_valid:
            logger.info("Question approved for processing")
            return True, ""
        else:
            logger.info("Question rejected - not database related")
            return False, message


# Standalone testing
if __name__ == "__main__":
    import asyncio
    from model import ModelManager
    
    async def test():
        # Initialize model
        model_manager = ModelManager()
        await model_manager.validate_models()
        
        # Initialize checker
        checker = QuestionChecker(model_manager)
        
        # Test questions
        test_questions = [
            "What were last month's sales by region?",
            "How do I cook pasta?",
            "Show me customer retention metrics",
            "What's the weather today?",
            "hi!",
            "I love you",
            "Lets fuck one",
            "Can u help me",
            "Analyze Q4 revenue trends"
        ]
        
        print("Testing Question Checker\n" + "="*50)
        
        for question in test_questions:
            print(f"\nQuestion: {question}")
            should_process, message = await checker.check_and_respond(question)
            
            if should_process:
                print(" Database question - will process")
            else:
                print(f"L Not database question")
                print(f"Response: {message}")
    
    asyncio.run(test())