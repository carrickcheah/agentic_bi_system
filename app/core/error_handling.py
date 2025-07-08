"""
Comprehensive Error Handling System
Production-grade error handling with correlation tracking, structured logging,
and proper exception hierarchy to prevent silent failures.
"""
import uuid
import traceback
import time
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
import logging
from contextlib import asynccontextmanager
import asyncio

# Error Categories for better classification
class ErrorCategory(Enum):
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization" 
    VALIDATION = "validation"
    BUSINESS_LOGIC = "business_logic"
    DATABASE = "database"
    EXTERNAL_SERVICE = "external_service"
    SYSTEM = "system"
    PERFORMANCE = "performance"
    RESOURCE = "resource"

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ErrorContext:
    """Rich error context for debugging and monitoring"""
    correlation_id: str
    operation: str
    component: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_data: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    execution_time_ms: Optional[float] = None
    stack_trace: Optional[str] = None
    additional_context: Dict[str, Any] = field(default_factory=dict)

class AgenticSQLError(Exception):
    """Base exception for all Agentic SQL errors"""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        correlation_id: Optional[str] = None,
        operation: Optional[str] = None,
        component: Optional[str] = None,
        recoverable: bool = True,
        retry_after: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.category = category
        self.severity = severity
        self.operation = operation or "unknown"
        self.component = component or "unknown"
        self.recoverable = recoverable
        self.retry_after = retry_after
        self.context = context or {}
        self.timestamp = datetime.utcnow()
        
        # Enhanced message with correlation ID
        enhanced_message = f"[{self.correlation_id}] {message}"
        super().__init__(enhanced_message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to structured dictionary for logging"""
        return {
            "correlation_id": self.correlation_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "operation": self.operation,
            "component": self.component,
            "recoverable": self.recoverable,
            "retry_after": self.retry_after,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "message": str(self)
        }

# Specific Exception Classes
class ValidationError(AgenticSQLError):
    """Input validation errors"""
    def __init__(self, message: str, field: str = None, **kwargs):
        super().__init__(
            message, 
            ErrorCategory.VALIDATION, 
            ErrorSeverity.MEDIUM,
            recoverable=True,
            **kwargs
        )
        self.field = field

class AuthenticationError(AgenticSQLError):
    """Authentication failures"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            ErrorCategory.AUTHENTICATION,
            ErrorSeverity.HIGH,
            recoverable=False,
            **kwargs
        )

class AuthorizationError(AgenticSQLError):
    """Authorization failures"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            ErrorCategory.AUTHORIZATION,
            ErrorSeverity.HIGH,
            recoverable=False,
            **kwargs
        )

class BusinessLogicError(AgenticSQLError):
    """Business rule violations"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            ErrorCategory.BUSINESS_LOGIC,
            ErrorSeverity.MEDIUM,
            **kwargs
        )

class DatabaseError(AgenticSQLError):
    """Database operation failures"""
    def __init__(self, message: str, operation: str = None, **kwargs):
        super().__init__(
            message,
            ErrorCategory.DATABASE,
            ErrorSeverity.HIGH,
            operation=operation,
            **kwargs
        )

class ExternalServiceError(AgenticSQLError):
    """External service failures (AI models, MCP servers)"""
    def __init__(self, message: str, service: str = None, **kwargs):
        super().__init__(
            message,
            ErrorCategory.EXTERNAL_SERVICE,
            ErrorSeverity.HIGH,
            **kwargs
        )
        self.service = service

class ResourceExhaustedError(AgenticSQLError):
    """Resource exhaustion (memory, connections, etc.)"""
    def __init__(self, message: str, resource_type: str = None, **kwargs):
        super().__init__(
            message,
            ErrorCategory.RESOURCE,
            ErrorSeverity.CRITICAL,
            recoverable=False,
            retry_after=60,
            **kwargs
        )
        self.resource_type = resource_type

class PerformanceError(AgenticSQLError):
    """Performance degradation"""
    def __init__(self, message: str, operation_time_ms: float = None, **kwargs):
        super().__init__(
            message,
            ErrorCategory.PERFORMANCE,
            ErrorSeverity.MEDIUM,
            **kwargs
        )
        self.operation_time_ms = operation_time_ms

class ErrorTracker:
    """Tracks and aggregates errors for monitoring"""
    
    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.recent_errors: List[ErrorContext] = []
        self.max_recent_errors = 1000
    
    def record_error(self, error: AgenticSQLError, context: ErrorContext):
        """Record error for monitoring and analysis"""
        error_key = f"{error.category.value}:{error.component}:{error.operation}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Store recent errors with full context
        context.stack_trace = traceback.format_exc()
        self.recent_errors.append(context)
        
        # Trim old errors
        if len(self.recent_errors) > self.max_recent_errors:
            self.recent_errors = self.recent_errors[-self.max_recent_errors:]
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics for monitoring"""
        return {
            "total_errors": sum(self.error_counts.values()),
            "error_breakdown": self.error_counts.copy(),
            "recent_error_count": len(self.recent_errors)
        }

# Global error tracker instance
error_tracker = ErrorTracker()

class ErrorHandler:
    """Centralized error handling with structured logging"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def handle_error(
        self,
        error: Exception,
        context: ErrorContext,
        raise_error: bool = True
    ) -> Optional[AgenticSQLError]:
        """Handle any error with proper logging and tracking"""
        
        # Convert to AgenticSQLError if not already
        if isinstance(error, AgenticSQLError):
            agentic_error = error
        else:
            # Classify unknown errors
            agentic_error = self._classify_error(error, context)
        
        # Record for monitoring
        error_tracker.record_error(agentic_error, context)
        
        # Structured logging
        self._log_error(agentic_error, context)
        
        # Optional re-raise
        if raise_error:
            raise agentic_error
        
        return agentic_error
    
    def _classify_error(self, error: Exception, context: ErrorContext) -> AgenticSQLError:
        """Classify unknown errors into appropriate categories"""
        error_msg = str(error)
        error_type = type(error).__name__
        
        # Database-related errors
        if any(keyword in error_msg.lower() for keyword in [
            'connection', 'timeout', 'database', 'sql', 'query'
        ]):
            return DatabaseError(
                f"Database error: {error_msg}",
                correlation_id=context.correlation_id,
                operation=context.operation,
                component=context.component
            )
        
        # Memory/Resource errors
        if any(keyword in error_type.lower() for keyword in [
            'memory', 'resource', 'limit'
        ]):
            return ResourceExhaustedError(
                f"Resource exhausted: {error_msg}",
                correlation_id=context.correlation_id,
                operation=context.operation,
                component=context.component
            )
        
        # Authentication errors
        if any(keyword in error_msg.lower() for keyword in [
            'authentication', 'unauthorized', 'invalid key', 'api key'
        ]):
            return ExternalServiceError(
                f"Authentication failed: {error_msg}",
                correlation_id=context.correlation_id,
                operation=context.operation,
                component=context.component
            )
        
        # Default to system error
        return AgenticSQLError(
            f"System error ({error_type}): {error_msg}",
            ErrorCategory.SYSTEM,
            ErrorSeverity.HIGH,
            correlation_id=context.correlation_id,
            operation=context.operation,
            component=context.component
        )
    
    def _log_error(self, error: AgenticSQLError, context: ErrorContext):
        """Log error with structured data"""
        log_data = {
            **error.to_dict(),
            "context": {
                "operation": context.operation,
                "component": context.component,
                "user_id": context.user_id,
                "session_id": context.session_id,
                "execution_time_ms": context.execution_time_ms,
                "additional_context": context.additional_context
            }
        }
        
        # Log at appropriate level
        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical("Critical error occurred", extra=log_data)
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error("High severity error", extra=log_data)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning("Medium severity error", extra=log_data)
        else:
            self.logger.info("Low severity error", extra=log_data)

@asynccontextmanager
async def error_boundary(
    operation: str,
    component: str,
    correlation_id: Optional[str] = None,
    user_id: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    timeout_seconds: Optional[float] = None
):
    """Async context manager for error boundary with timeout"""
    
    correlation_id = correlation_id or str(uuid.uuid4())
    logger = logger or logging.getLogger(component)
    handler = ErrorHandler(logger)
    
    context = ErrorContext(
        correlation_id=correlation_id,
        operation=operation,
        component=component,
        user_id=user_id
    )
    
    start_time = time.time()
    
    try:
        # Optional timeout
        if timeout_seconds:
            yield context
        else:
            async with asyncio.timeout(timeout_seconds) if timeout_seconds else asyncio.nullcontext():
                yield context
                
    except asyncio.TimeoutError:
        context.execution_time_ms = (time.time() - start_time) * 1000
        timeout_error = PerformanceError(
            f"Operation '{operation}' timed out after {timeout_seconds}s",
            operation_time_ms=context.execution_time_ms,
            correlation_id=correlation_id,
            operation=operation,
            component=component
        )
        handler.handle_error(timeout_error, context)
        
    except Exception as e:
        context.execution_time_ms = (time.time() - start_time) * 1000
        handler.handle_error(e, context)
        
    else:
        # Log successful operation
        context.execution_time_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Operation '{operation}' completed successfully",
            extra={
                "correlation_id": correlation_id,
                "operation": operation,
                "component": component,
                "execution_time_ms": context.execution_time_ms,
                "user_id": user_id
            }
        )

def validate_input(
    data: Any,
    validator_func: callable,
    operation: str,
    component: str,
    correlation_id: Optional[str] = None
) -> Any:
    """Validate input with proper error handling"""
    
    try:
        return validator_func(data)
    except Exception as e:
        raise ValidationError(
            f"Input validation failed: {str(e)}",
            correlation_id=correlation_id,
            operation=operation,
            component=component,
            context={"input_data": str(data)[:200]}  # Truncate for security
        )

def safe_execute(
    func: callable,
    operation: str,
    component: str,
    correlation_id: Optional[str] = None,
    default_return=None,
    raise_on_failure: bool = True
):
    """Safely execute function with error handling"""
    
    correlation_id = correlation_id or str(uuid.uuid4())
    logger = logging.getLogger(component)
    handler = ErrorHandler(logger)
    
    context = ErrorContext(
        correlation_id=correlation_id,
        operation=operation,
        component=component
    )
    
    try:
        return func()
    except Exception as e:
        handler.handle_error(e, context, raise_error=raise_on_failure)
        return default_return

# Decorators for easy error handling
def with_error_handling(
    operation: str,
    component: str,
    timeout_seconds: Optional[float] = None
):
    """Decorator for automatic error handling"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            correlation_id = kwargs.pop('correlation_id', None) or str(uuid.uuid4())
            
            async with error_boundary(
                operation=operation,
                component=component,
                correlation_id=correlation_id,
                timeout_seconds=timeout_seconds
            ) as context:
                # Add correlation_id to function if it accepts it
                import inspect
                sig = inspect.signature(func)
                if 'correlation_id' in sig.parameters:
                    kwargs['correlation_id'] = correlation_id
                
                return await func(*args, **kwargs)
        
        return wrapper
    return decorator