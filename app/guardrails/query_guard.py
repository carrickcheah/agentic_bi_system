"""
SQL Query Security and Safety Validation

Provides comprehensive SQL query validation and security checks.
"""

import re
from typing import List, Set
from ..config import settings
from ..database.models import ValidationResult


class QueryGuard:
    """SQL query security and safety validator."""
    
    def __init__(self):
        self.dangerous_keywords = set(
            keyword.upper() for keyword in settings.dangerous_sql_keywords
        )
        self.allowed_keywords = set(
            keyword.upper() for keyword in settings.allowed_sql_keywords
        )
    
    def validate_query(self, query: str) -> ValidationResult:
        """
        Comprehensive query validation.
        
        Args:
            query: SQL query to validate
            
        Returns:
            ValidationResult with safety assessment
        """
        query = query.strip()
        
        if not query:
            return ValidationResult(
                is_safe=False,
                syntax_valid=False,
                reason="Empty query"
            )
        
        # Check for dangerous operations
        security_issues = self._check_security_issues(query)
        
        # Check syntax basics
        syntax_valid = self._check_basic_syntax(query)
        
        # Check for allowed operations only
        allowed_operations = self._check_allowed_operations(query)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(query, security_issues)
        
        is_safe = (
            len(security_issues) == 0 and
            syntax_valid and
            allowed_operations
        )
        
        reason = None
        if not is_safe:
            if security_issues:
                reason = f"Security issues: {', '.join(security_issues)}"
            elif not allowed_operations:
                reason = "Query contains disallowed operations"
            elif not syntax_valid:
                reason = "Invalid syntax"
        
        return ValidationResult(
            is_safe=is_safe,
            syntax_valid=syntax_valid,
            security_issues=security_issues,
            suggestions=suggestions,
            reason=reason
        )
    
    def _check_security_issues(self, query: str) -> List[str]:
        """Check for security vulnerabilities."""
        issues = []
        query_upper = query.upper()
        
        # Check for dangerous keywords
        for keyword in self.dangerous_keywords:
            if re.search(rf'\b{keyword}\b', query_upper):
                issues.append(f"Dangerous operation: {keyword}")
        
        # Check for SQL injection patterns
        injection_patterns = [
            r"';\s*--",  # SQL comment injection
            r"';\s*\/\*",  # Comment block injection
            r"\bunion\s+select\b",  # UNION SELECT injection
            r"'.*or.*'.*=.*'",  # OR injection
            r"--.*$",  # Comment at end
            r"\/\*.*\*\/",  # Comment blocks
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                issues.append(f"Potential SQL injection pattern detected")
                break
        
        # Check for suspicious functions
        suspicious_functions = [
            'LOAD_FILE', 'INTO OUTFILE', 'INTO DUMPFILE',
            'BENCHMARK', 'SLEEP', 'WAITFOR',
            'xp_cmdshell', 'sp_execute_external_script'
        ]
        
        for func in suspicious_functions:
            if re.search(rf'\b{func}\b', query_upper):
                issues.append(f"Suspicious function: {func}")
        
        return issues
    
    def _check_basic_syntax(self, query: str) -> bool:
        """Basic syntax validation."""
        query = query.strip()
        
        # Check for balanced parentheses
        if query.count('(') != query.count(')'):
            return False
        
        # Check for balanced quotes
        single_quotes = query.count("'")
        if single_quotes % 2 != 0:
            return False
        
        # Check if query starts with allowed statement
        query_upper = query.upper().strip()
        allowed_starts = ['SELECT', 'WITH', 'SHOW', 'DESCRIBE', 'DESC', 'EXPLAIN']
        
        starts_valid = any(
            query_upper.startswith(start) for start in allowed_starts
        )
        
        return starts_valid
    
    def _check_allowed_operations(self, query: str) -> bool:
        """Check if query only uses allowed operations."""
        query_upper = query.upper()
        
        # Extract all SQL keywords from query
        # This is a simplified approach - could be improved with proper parsing
        words = re.findall(r'\b[A-Z_]+\b', query_upper)
        
        # Check if any disallowed keywords are present
        for word in words:
            if word in self.dangerous_keywords:
                return False
        
        return True
    
    def _generate_suggestions(self, query: str, security_issues: List[str]) -> List[str]:
        """Generate helpful suggestions for query improvement."""
        suggestions = []
        query_upper = query.upper()
        
        # Suggest adding LIMIT if missing
        if 'LIMIT' not in query_upper and 'SELECT' in query_upper:
            suggestions.append("Consider adding LIMIT clause to restrict result size")
        
        # Suggest using parameters for dynamic values
        if "'" in query:
            suggestions.append("Consider using parameterized queries for dynamic values")
        
        # Suggest specific improvements based on issues
        if security_issues:
            suggestions.append("Remove dangerous operations and use read-only queries")
        
        # Suggest proper formatting
        if not query.strip().endswith(';'):
            suggestions.append("Consider ending query with semicolon")
        
        return suggestions
    
    def is_read_only_query(self, query: str) -> bool:
        """Check if query is read-only (safe for execution)."""
        query_upper = query.upper().strip()
        
        # Must start with SELECT, WITH, SHOW, DESCRIBE, or EXPLAIN
        read_only_starts = ['SELECT', 'WITH', 'SHOW', 'DESCRIBE', 'DESC', 'EXPLAIN']
        
        if not any(query_upper.startswith(start) for start in read_only_starts):
            return False
        
        # Must not contain any dangerous keywords
        for keyword in self.dangerous_keywords:
            if re.search(rf'\b{keyword}\b', query_upper):
                return False
        
        return True
    
    def sanitize_query(self, query: str) -> str:
        """Basic query sanitization."""
        # Remove dangerous patterns
        query = re.sub(r'--.*$', '', query, flags=re.MULTILINE)  # Remove comments
        query = re.sub(r'/\*.*?\*/', '', query, flags=re.DOTALL)  # Remove block comments
        
        # Normalize whitespace
        query = ' '.join(query.split())
        
        return query.strip()