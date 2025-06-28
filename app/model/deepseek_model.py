"""
DeepSeek Model Integration

Provides async DeepSeek API access using Pydantic settings.
Fallback option #2 for the SQL investigation agent.
Uses OpenAI SDK with DeepSeek's API endpoint.
"""

from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI

from .config import settings
from ..utils.logging import logger
from ..prompt_engineering import SQL_AGENT_SYSTEM_PROMPT


class DeepSeekModel:
    """
    DeepSeek API client with async support.
    
    Uses OpenAI SDK with DeepSeek endpoint - fallback option #2.
    """
    
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url  # https://api.deepseek.com
        )
        self.model = settings.deepseek_model
        logger.info(f"DeepSeek model initialized: {self.model}")
    
    async def generate_response(
        self, 
        prompt: str, 
        max_tokens: int = 2048,
        temperature: float = 0.7,
        use_system_prompt: bool = True
    ) -> str:
        """
        Generate a response from DeepSeek with SQL agent system prompt.
        
        Args:
            prompt: The user prompt
            max_tokens: Maximum tokens in response
            temperature: Response creativity (0-1)
            use_system_prompt: Whether to include SQL agent system prompt
            
        Returns:
            DeepSeek's response text
        """
        try:
            messages = [{"role": "user", "content": prompt}]
            
            # Prepare system message if requested
            if use_system_prompt:
                messages.insert(0, {"role": "system", "content": SQL_AGENT_SYSTEM_PROMPT})
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            raise
    
    async def analyze_sql_query(self, query: str, schema_info: Dict) -> Dict[str, Any]:
        """
        Analyze a SQL query for investigation planning.
        
        Args:
            query: User's SQL query or question
            schema_info: Database schema information
            
        Returns:
            Analysis with suggestions, intent, etc.
        """
        prompt = f"""
        Analyze this SQL query/question for autonomous investigation:
        
        Query: {query}
        
        Available schema: {schema_info}
        
        Provide analysis in JSON format with:
        - intent: What the user wants to find
        - entities: Which tables/columns are relevant  
        - approach: How to investigate this
        - sql_suggestions: Suggested SQL queries
        - confidence: How confident you are (0-1)
        """
        
        try:
            response = await self.generate_response(prompt, max_tokens=1500)
            # In a real implementation, you'd parse this as JSON
            return {
                "raw_response": response,
                "analysis_type": "sql_query_analysis"
            }
            
        except Exception as e:
            logger.error(f"DeepSeek SQL analysis error: {e}")
            return {
                "error": str(e),
                "analysis_type": "sql_query_analysis"
            }
    
    async def synthesize_investigation_results(
        self, 
        original_query: str,
        findings: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Synthesize investigation findings into final insights.
        
        Args:
            original_query: Original user question
            findings: List of investigation step results
            
        Returns:
            Final synthesized insights
        """
        prompt = f"""
        Synthesize these investigation findings into clear insights:
        
        Original question: {original_query}
        
        Investigation findings: {findings}
        
        Provide synthesis in JSON format with:
        - key_insights: Main discoveries
        - answer: Direct answer to original question
        - confidence: How confident in the answer (0-1)
        - recommendations: What user should do next
        - data_quality_notes: Any data quality concerns
        """
        
        try:
            response = await self.generate_response(prompt, max_tokens=2000)
            return {
                "raw_response": response,
                "synthesis_type": "investigation_results"
            }
            
        except Exception as e:
            logger.error(f"DeepSeek synthesis error: {e}")
            return {
                "error": str(e),
                "synthesis_type": "investigation_results"
            }
    
    async def health_check(self) -> bool:
        """
        Check if DeepSeek API is accessible.
        
        Returns:
            True if API is working, False otherwise
        """
        try:
            response = await self.generate_response(
                "Hello! Respond with 'OK' if you're working.",
                max_tokens=10,
                use_system_prompt=False
            )
            return "OK" in response or "ok" in response.lower()
            
        except Exception as e:
            logger.error(f"DeepSeek health check failed: {e}")
            return False


if __name__ == "__main__":
    async def main():
        model = DeepSeekModel()
        print(await model.health_check())

    import asyncio
    asyncio.run(main())