"""
OpenAI Model Integration

Provides async OpenAI API access using Pydantic settings.
Fallback option #3 for the SQL investigation agent.
"""

from typing import List, Dict, Any
import json
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from .config import settings
from .model_logging import logger
from .prompts import SQL_AGENT_SYSTEM_PROMPT


class OpenAIModel:
    """
    OpenAI API client with async support.
    
    Uses Pydantic settings for configuration - fallback option #3.
    """
    
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url  # https://api.openai.com/v1
        )
        self.model = settings.openai_model
        logger.info(f"OpenAI model initialized: {self.model}")
    
    async def generate_response(
        self, 
        prompt: str, 
        max_tokens: int = 2048,
        temperature: float = 0.7,
        use_system_prompt: bool = True
    ) -> str:
        """
        Generate a response from OpenAI with SQL agent system prompt.
        
        Args:
            prompt: The user prompt
            max_tokens: Maximum tokens in response
            temperature: Response creativity (0-1)
            use_system_prompt: Whether to include SQL agent system prompt
            
        Returns:
            OpenAI's response text
        """
        try:
            messages = [{"role": "user", "content": prompt}]
            
            # Prepare system message if requested
            if use_system_prompt:
                messages.insert(0, {"role": "system", "content": SQL_AGENT_SYSTEM_PROMPT})
            
            response: ChatCompletion = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "text"},  # Explicit text format
                stream=False  # Disable streaming for simpler handling
            )
            
            # Handle response with proper null checking
            if response.choices and len(response.choices) > 0:
                message_content = response.choices[0].message.content
                if message_content is not None:
                    return message_content
                else:
                    logger.warning("OpenAI returned null content")
                    return ""
            else:
                logger.error("OpenAI response has no choices")
                return ""
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
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
            response = await self.generate_response(prompt, max_tokens=1500, temperature=0.3)
            
            # Try to parse as JSON, fallback to raw response
            try:
                parsed_analysis = json.loads(response)
                return {
                    "parsed_analysis": parsed_analysis,
                    "raw_response": response,
                    "analysis_type": "sql_query_analysis",
                    "model_used": "openai"
                }
            except json.JSONDecodeError:
                logger.warning("OpenAI response not valid JSON, returning raw response")
                return {
                    "raw_response": response,
                    "analysis_type": "sql_query_analysis",
                    "model_used": "openai"
                }
            
        except Exception as e:
            logger.error(f"OpenAI SQL analysis error: {e}")
            return {
                "error": str(e),
                "analysis_type": "sql_query_analysis",
                "model_used": "openai"
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
            response = await self.generate_response(prompt, max_tokens=2000, temperature=0.2)
            
            # Try to parse as JSON, fallback to raw response
            try:
                parsed_synthesis = json.loads(response)
                return {
                    "parsed_synthesis": parsed_synthesis,
                    "raw_response": response,
                    "synthesis_type": "investigation_results",
                    "model_used": "openai"
                }
            except json.JSONDecodeError:
                logger.warning("OpenAI synthesis response not valid JSON, returning raw response")
                return {
                    "raw_response": response,
                    "synthesis_type": "investigation_results",
                    "model_used": "openai"
                }
            
        except Exception as e:
            logger.error(f"OpenAI synthesis error: {e}")
            return {
                "error": str(e),
                "synthesis_type": "investigation_results",
                "model_used": "openai"
            }
    
    async def health_check(self) -> bool:
        """
        Check if OpenAI API is accessible.
        
        Returns:
            True if API is working, False otherwise
        """
        try:
            response = await self.generate_response(
                "Health check: respond with exactly 'OK' if you're working correctly.",
                max_tokens=10,
                temperature=0.0,
                use_system_prompt=False
            )
            
            # More robust health check
            is_healthy = bool(response and (
                "OK" in response.upper() or 
                "ok" in response.lower() or
                "working" in response.lower()
            ))
            
            logger.info(f"OpenAI health check result: {response} -> {'✅' if is_healthy else '❌'}")
            return is_healthy
            
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            return False


if __name__ == "__main__":
    async def main():
        model = OpenAIModel()
        print(await model.health_check())

    import asyncio
    asyncio.run(main())
