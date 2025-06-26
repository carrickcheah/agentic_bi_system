"""
Security Configuration Settings

Handles safety, rate limiting, SQL validation, and session management.
"""

from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SecuritySettings(BaseSettings):
    """Security and safety configuration settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # SQL query security
    max_query_timeout: int = 30  # seconds
    max_rows_returned: int = 10000
    allowed_sql_keywords: List[str] = [
        "SELECT", "FROM", "WHERE", "JOIN", "GROUP BY", 
        "ORDER BY", "HAVING", "LIMIT", "UNION", "WITH"
    ]
    dangerous_sql_keywords: List[str] = [
        "DROP", "DELETE", "TRUNCATE", "UPDATE", "INSERT", 
        "ALTER", "CREATE", "GRANT", "REVOKE"
    ]
    
    # Rate limiting
    rate_limit_requests: int = 100  # requests per window
    rate_limit_window: int = 60     # seconds
    
    # Session management
    session_timeout: int = 3600     # 1 hour in seconds
    cache_ttl: int = 1800          # 30 minutes in seconds