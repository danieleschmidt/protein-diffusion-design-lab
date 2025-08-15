"""
Advanced Error Recovery and Resilience Framework for Protein Diffusion Design Lab.

This module provides comprehensive error handling, automatic recovery,
circuit breakers, and resilience patterns for production environments.
"""

import logging
import time
import json
import traceback
import threading
from typing import Dict, List, Optional, Any, Callable, Type, Union
from dataclasses import dataclass, asdict
from enum import Enum
from functools import wraps
from collections import defaultdict
import queue
import asyncio

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Recovery strategy types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    ESCALATION = "escalation"


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class ErrorMetrics:
    """Error metrics and statistics."""
    total_errors: int = 0
    errors_by_type: Dict[str, int] = None
    errors_by_severity: Dict[str, int] = None
    recovery_success_rate: float = 0.0
    avg_recovery_time: float = 0.0
    circuit_breaker_trips: int = 0
    last_error_timestamp: Optional[float] = None
    
    def __post_init__(self):
        if self.errors_by_type is None:
            self.errors_by_type = defaultdict(int)
        if self.errors_by_severity is None:
            self.errors_by_severity = defaultdict(int)


@dataclass
class ErrorEvent:
    """Error event representation."""
    timestamp: float
    error_type: str
    error_message: str
    severity: ErrorSeverity
    stack_trace: str
    context: Dict[str, Any]
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None
    recovery_time: Optional[float] = None


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3
    monitoring_window: float = 300.0  # 5 minutes


@dataclass
class RetryConfig:
    """Retry configuration."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


class CircuitBreaker:
    """Circuit breaker implementation for service protection."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_attempt_time = None
        self._lock = threading.Lock()
        
        logger.info(f"Circuit breaker initialized: {name}")
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    logger.info(f"Circuit breaker {self.name} moved to HALF_OPEN")
                else:
                    raise CircuitBreakerOpenException(f"Circuit breaker {self.name} is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time is None:
            return True
        
        return time.time() - self.last_failure_time >= self.config.recovery_timeout
    
    def _on_success(self):
        """Handle successful execution."""
        with self._lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    logger.info(f"Circuit breaker {self.name} CLOSED after recovery")
            elif self.state == CircuitBreakerState.CLOSED:
                self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed execution."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
                self.success_count = 0
                logger.warning(f"Circuit breaker {self.name} OPENED from HALF_OPEN")
            elif self.state == CircuitBreakerState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    self.state = CircuitBreakerState.OPEN
                    logger.warning(f"Circuit breaker {self.name} OPENED after {self.failure_count} failures")
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "config": asdict(self.config)
        }


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class RetryManager:
    """Advanced retry manager with exponential backoff and jitter."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
    
    def retry(self, func: Callable, *args, **kwargs):
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                return func(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                
                if attempt == self.config.max_attempts - 1:
                    # Last attempt, don't sleep
                    break
                
                # Calculate delay
                delay = self._calculate_delay(attempt)
                
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s: {e}")
                time.sleep(delay)
        
        # All attempts failed
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for exponential backoff with jitter."""
        delay = min(
            self.config.base_delay * (self.config.exponential_base ** attempt),
            self.config.max_delay
        )
        
        if self.config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # Add 0-50% jitter
        
        return delay


class ErrorRecoveryManager:
    """Main error recovery and resilience manager."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_managers: Dict[str, RetryManager] = {}
        self.error_metrics = ErrorMetrics()
        self.error_events: List[ErrorEvent] = []
        self.fallback_handlers: Dict[str, Callable] = {}
        self.error_handlers: Dict[Type[Exception], Callable] = {}
        self._lock = threading.Lock()
        
        # Error event queue for async processing
        self.error_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self._process_error_events, daemon=True)
        self.processing_thread.start()
        
        logger.info("Error Recovery Manager initialized")
    
    def register_circuit_breaker(self, name: str, config: CircuitBreakerConfig) -> CircuitBreaker:
        """Register a new circuit breaker."""
        circuit_breaker = CircuitBreaker(name, config)
        self.circuit_breakers[name] = circuit_breaker
        return circuit_breaker
    
    def register_retry_manager(self, name: str, config: RetryConfig) -> RetryManager:
        """Register a new retry manager."""
        retry_manager = RetryManager(config)
        self.retry_managers[name] = retry_manager
        return retry_manager
    
    def register_fallback_handler(self, operation_name: str, handler: Callable):
        """Register a fallback handler for an operation."""
        self.fallback_handlers[operation_name] = handler
        logger.info(f"Registered fallback handler for: {operation_name}")
    
    def register_error_handler(self, exception_type: Type[Exception], handler: Callable):
        """Register a custom error handler for specific exception types."""
        self.error_handlers[exception_type] = handler
        logger.info(f"Registered error handler for: {exception_type.__name__}")
    
    def handle_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        operation_name: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM
    ) -> Dict[str, Any]:
        """Handle an error with appropriate recovery strategy."""
        start_time = time.time()
        
        # Create error event
        error_event = ErrorEvent(
            timestamp=start_time,
            error_type=type(error).__name__,
            error_message=str(error),
            severity=severity,
            stack_trace=traceback.format_exc(),
            context=context
        )
        
        # Update metrics
        self._update_error_metrics(error_event)
        
        # Determine recovery strategy
        recovery_strategy = self._determine_recovery_strategy(error, context, operation_name)
        error_event.recovery_strategy = recovery_strategy
        
        # Attempt recovery
        recovery_result = self._attempt_recovery(
            error, context, operation_name, recovery_strategy
        )
        
        # Update event with recovery results
        error_event.recovery_attempted = True
        error_event.recovery_successful = recovery_result.get("success", False)
        error_event.recovery_time = time.time() - start_time
        
        # Queue event for processing
        self.error_queue.put(error_event)
        
        return recovery_result
    
    def _determine_recovery_strategy(
        self,
        error: Exception,
        context: Dict[str, Any],
        operation_name: str
    ) -> RecoveryStrategy:
        """Determine the best recovery strategy for an error."""
        
        # Check for custom error handlers
        for exception_type, handler in self.error_handlers.items():
            if isinstance(error, exception_type):
                return RecoveryStrategy.ESCALATION
        
        # Check error type and context
        if isinstance(error, (ConnectionError, TimeoutError)):
            return RecoveryStrategy.RETRY
        
        elif isinstance(error, CircuitBreakerOpenException):
            return RecoveryStrategy.FALLBACK
        
        elif isinstance(error, (MemoryError, ResourceWarning)):
            return RecoveryStrategy.GRACEFUL_DEGRADATION
        
        elif isinstance(error, (ValueError, TypeError)) and "generation" in operation_name.lower():
            return RecoveryStrategy.FALLBACK
        
        else:
            # Default to retry for most errors
            return RecoveryStrategy.RETRY
    
    def _attempt_recovery(
        self,
        error: Exception,
        context: Dict[str, Any],
        operation_name: str,
        strategy: RecoveryStrategy
    ) -> Dict[str, Any]:
        """Attempt recovery using the specified strategy."""
        
        try:
            if strategy == RecoveryStrategy.RETRY:
                return self._attempt_retry_recovery(error, context, operation_name)
            
            elif strategy == RecoveryStrategy.FALLBACK:
                return self._attempt_fallback_recovery(error, context, operation_name)
            
            elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                return self._attempt_circuit_breaker_recovery(error, context, operation_name)
            
            elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                return self._attempt_graceful_degradation(error, context, operation_name)
            
            elif strategy == RecoveryStrategy.ESCALATION:
                return self._attempt_escalation(error, context, operation_name)
            
            else:
                return {"success": False, "strategy": strategy, "message": "Unknown recovery strategy"}
        
        except Exception as recovery_error:
            logger.error(f"Recovery attempt failed: {recovery_error}")
            return {
                "success": False,
                "strategy": strategy,
                "error": str(recovery_error),
                "message": "Recovery attempt failed"
            }
    
    def _attempt_retry_recovery(
        self, error: Exception, context: Dict[str, Any], operation_name: str
    ) -> Dict[str, Any]:
        """Attempt recovery using retry strategy."""
        retry_manager = self.retry_managers.get(operation_name)
        
        if not retry_manager:
            # Create default retry manager
            retry_config = RetryConfig(max_attempts=3, base_delay=1.0)
            retry_manager = RetryManager(retry_config)
        
        try:
            # Get the original function from context
            original_func = context.get("original_function")
            args = context.get("args", ())
            kwargs = context.get("kwargs", {})
            
            if original_func:
                result = retry_manager.retry(original_func, *args, **kwargs)
                return {
                    "success": True,
                    "strategy": RecoveryStrategy.RETRY,
                    "result": result,
                    "message": "Retry recovery successful"
                }
            else:
                return {
                    "success": False,
                    "strategy": RecoveryStrategy.RETRY,
                    "message": "No original function to retry"
                }
        
        except Exception as e:
            return {
                "success": False,
                "strategy": RecoveryStrategy.RETRY,
                "error": str(e),
                "message": "Retry recovery failed"
            }
    
    def _attempt_fallback_recovery(
        self, error: Exception, context: Dict[str, Any], operation_name: str
    ) -> Dict[str, Any]:
        """Attempt recovery using fallback strategy."""
        fallback_handler = self.fallback_handlers.get(operation_name)
        
        if fallback_handler:
            try:
                args = context.get("args", ())
                kwargs = context.get("kwargs", {})
                result = fallback_handler(*args, **kwargs)
                
                return {
                    "success": True,
                    "strategy": RecoveryStrategy.FALLBACK,
                    "result": result,
                    "message": "Fallback recovery successful"
                }
            
            except Exception as e:
                return {
                    "success": False,
                    "strategy": RecoveryStrategy.FALLBACK,
                    "error": str(e),
                    "message": "Fallback handler failed"
                }
        else:
            # Provide default fallback based on operation type
            return self._provide_default_fallback(error, context, operation_name)
    
    def _provide_default_fallback(
        self, error: Exception, context: Dict[str, Any], operation_name: str
    ) -> Dict[str, Any]:
        """Provide default fallback results."""
        
        if "generation" in operation_name.lower():
            # Return empty generation result
            return {
                "success": True,
                "strategy": RecoveryStrategy.FALLBACK,
                "result": [],
                "message": "Default fallback: empty generation result"
            }
        
        elif "ranking" in operation_name.lower():
            # Return empty ranking result
            return {
                "success": True,
                "strategy": RecoveryStrategy.FALLBACK,
                "result": [],
                "message": "Default fallback: empty ranking result"
            }
        
        else:
            return {
                "success": False,
                "strategy": RecoveryStrategy.FALLBACK,
                "message": "No fallback available"
            }
    
    def _attempt_circuit_breaker_recovery(
        self, error: Exception, context: Dict[str, Any], operation_name: str
    ) -> Dict[str, Any]:
        """Attempt recovery using circuit breaker."""
        # Circuit breaker is more of a prevention mechanism
        # If we're here, the circuit breaker should be managed externally
        return {
            "success": False,
            "strategy": RecoveryStrategy.CIRCUIT_BREAKER,
            "message": "Circuit breaker recovery not implemented"
        }
    
    def _attempt_graceful_degradation(
        self, error: Exception, context: Dict[str, Any], operation_name: str
    ) -> Dict[str, Any]:
        """Attempt graceful degradation."""
        
        # Reduce resource usage or functionality
        degraded_context = context.copy()
        
        # Reduce batch sizes
        if "num_samples" in degraded_context:
            degraded_context["num_samples"] = min(degraded_context["num_samples"], 10)
        
        if "max_length" in degraded_context:
            degraded_context["max_length"] = min(degraded_context["max_length"], 128)
        
        # Try with reduced parameters
        original_func = context.get("original_function")
        if original_func:
            try:
                args = degraded_context.get("args", ())
                kwargs = {k: v for k, v in degraded_context.get("kwargs", {}).items() 
                         if k not in ["num_samples", "max_length"]}
                kwargs.update({
                    "num_samples": degraded_context.get("num_samples", 10),
                    "max_length": degraded_context.get("max_length", 128)
                })
                
                result = original_func(*args, **kwargs)
                
                return {
                    "success": True,
                    "strategy": RecoveryStrategy.GRACEFUL_DEGRADATION,
                    "result": result,
                    "message": "Graceful degradation successful with reduced parameters"
                }
            
            except Exception as e:
                return {
                    "success": False,
                    "strategy": RecoveryStrategy.GRACEFUL_DEGRADATION,
                    "error": str(e),
                    "message": "Graceful degradation failed"
                }
        else:
            return {
                "success": False,
                "strategy": RecoveryStrategy.GRACEFUL_DEGRADATION,
                "message": "No function to degrade"
            }
    
    def _attempt_escalation(
        self, error: Exception, context: Dict[str, Any], operation_name: str
    ) -> Dict[str, Any]:
        """Attempt error escalation to custom handlers."""
        
        for exception_type, handler in self.error_handlers.items():
            if isinstance(error, exception_type):
                try:
                    result = handler(error, context)
                    return {
                        "success": True,
                        "strategy": RecoveryStrategy.ESCALATION,
                        "result": result,
                        "message": f"Custom handler for {exception_type.__name__} successful"
                    }
                
                except Exception as e:
                    return {
                        "success": False,
                        "strategy": RecoveryStrategy.ESCALATION,
                        "error": str(e),
                        "message": f"Custom handler for {exception_type.__name__} failed"
                    }
        
        return {
            "success": False,
            "strategy": RecoveryStrategy.ESCALATION,
            "message": "No custom handler found"
        }
    
    def _update_error_metrics(self, error_event: ErrorEvent):
        """Update error metrics."""
        with self._lock:
            self.error_metrics.total_errors += 1
            self.error_metrics.errors_by_type[error_event.error_type] += 1
            self.error_metrics.errors_by_severity[error_event.severity.value] += 1
            self.error_metrics.last_error_timestamp = error_event.timestamp
    
    def _process_error_events(self):
        """Process error events in background thread."""
        while True:
            try:
                error_event = self.error_queue.get(timeout=1.0)
                
                # Store error event
                with self._lock:
                    self.error_events.append(error_event)
                    
                    # Keep only last 1000 events
                    if len(self.error_events) > 1000:
                        self.error_events = self.error_events[-1000:]
                
                # Update recovery metrics
                if error_event.recovery_attempted:
                    self._update_recovery_metrics(error_event)
                
                self.error_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing error event: {e}")
    
    def _update_recovery_metrics(self, error_event: ErrorEvent):
        """Update recovery success metrics."""
        with self._lock:
            # Calculate recovery success rate
            recent_events = [e for e in self.error_events[-100:] if e.recovery_attempted]
            if recent_events:
                successful_recoveries = sum(1 for e in recent_events if e.recovery_successful)
                self.error_metrics.recovery_success_rate = successful_recoveries / len(recent_events)
            
            # Calculate average recovery time
            recovery_times = [e.recovery_time for e in recent_events if e.recovery_time is not None]
            if recovery_times:
                self.error_metrics.avg_recovery_time = sum(recovery_times) / len(recovery_times)
    
    def get_error_metrics(self) -> ErrorMetrics:
        """Get current error metrics."""
        with self._lock:
            return ErrorMetrics(
                total_errors=self.error_metrics.total_errors,
                errors_by_type=dict(self.error_metrics.errors_by_type),
                errors_by_severity=dict(self.error_metrics.errors_by_severity),
                recovery_success_rate=self.error_metrics.recovery_success_rate,
                avg_recovery_time=self.error_metrics.avg_recovery_time,
                circuit_breaker_trips=sum(1 for cb in self.circuit_breakers.values() 
                                        if cb.state == CircuitBreakerState.OPEN),
                last_error_timestamp=self.error_metrics.last_error_timestamp
            )
    
    def get_recent_errors(self, limit: int = 50) -> List[ErrorEvent]:
        """Get recent error events."""
        with self._lock:
            return self.error_events[-limit:] if self.error_events else []
    
    def get_circuit_breaker_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers."""
        return {name: cb.get_status() for name, cb in self.circuit_breakers.items()}


# Decorators for easy error handling integration

def with_error_recovery(
    operation_name: str,
    recovery_manager: Optional[ErrorRecoveryManager] = None,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
):
    """Decorator to add error recovery to functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal recovery_manager
            
            if recovery_manager is None:
                recovery_manager = ErrorRecoveryManager()
            
            try:
                return func(*args, **kwargs)
            
            except Exception as e:
                context = {
                    "original_function": func,
                    "args": args,
                    "kwargs": kwargs,
                    "function_name": func.__name__
                }
                
                recovery_result = recovery_manager.handle_error(
                    e, context, operation_name, severity
                )
                
                if recovery_result.get("success"):
                    return recovery_result.get("result")
                else:
                    # Re-raise if recovery failed
                    raise
        
        return wrapper
    return decorator


def with_circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None,
    recovery_manager: Optional[ErrorRecoveryManager] = None
):
    """Decorator to add circuit breaker protection to functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal recovery_manager, config
            
            if recovery_manager is None:
                recovery_manager = ErrorRecoveryManager()
            
            if config is None:
                config = CircuitBreakerConfig()
            
            # Get or create circuit breaker
            if name not in recovery_manager.circuit_breakers:
                recovery_manager.register_circuit_breaker(name, config)
            
            circuit_breaker = recovery_manager.circuit_breakers[name]
            
            return circuit_breaker.call(func, *args, **kwargs)
        
        return wrapper
    return decorator


def with_retry(
    config: Optional[RetryConfig] = None,
    recovery_manager: Optional[ErrorRecoveryManager] = None
):
    """Decorator to add retry logic to functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal recovery_manager, config
            
            if recovery_manager is None:
                recovery_manager = ErrorRecoveryManager()
            
            if config is None:
                config = RetryConfig()
            
            # Get or create retry manager
            func_name = f"{func.__module__}.{func.__name__}"
            if func_name not in recovery_manager.retry_managers:
                recovery_manager.register_retry_manager(func_name, config)
            
            retry_manager = recovery_manager.retry_managers[func_name]
            
            return retry_manager.retry(func, *args, **kwargs)
        
        return wrapper
    return decorator


# Global error recovery manager instance
_global_recovery_manager = None

def get_global_recovery_manager() -> ErrorRecoveryManager:
    """Get global error recovery manager instance."""
    global _global_recovery_manager
    if _global_recovery_manager is None:
        _global_recovery_manager = ErrorRecoveryManager()
    return _global_recovery_manager