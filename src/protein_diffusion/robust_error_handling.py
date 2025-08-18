"""
Robust Error Handling System for Protein Diffusion Design Lab

This module provides comprehensive error handling, recovery strategies,
circuit breakers, and fault tolerance mechanisms for the protein
generation and analysis pipeline.
"""

import traceback
import functools
import threading
import time
import logging
import json
import uuid
from typing import Dict, Any, Optional, List, Callable, Union, Tuple, Type
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
from collections import defaultdict, deque
import queue
import warnings

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Error recovery strategies."""
    NONE = "none"  # No recovery, fail immediately
    RETRY = "retry"  # Retry with exponential backoff
    FALLBACK = "fallback"  # Use fallback implementation
    GRACEFUL_DEGRADATION = "graceful_degradation"  # Reduce functionality
    CIRCUIT_BREAKER = "circuit_breaker"  # Stop trying after failures


@dataclass
class ErrorEvent:
    """Represents an error event for tracking and analysis."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    error_type: str = ""
    error_message: str = ""
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    component: str = ""
    function: str = ""
    recovery_attempted: bool = False
    recovery_successful: bool = False
    context: Dict[str, Any] = field(default_factory=dict)
    stack_trace: str = ""


@dataclass 
class CircuitBreakerState:
    """Circuit breaker state tracking."""
    is_open: bool = False
    failure_count: int = 0
    last_failure_time: float = 0
    success_count: int = 0
    total_requests: int = 0


class ProteinDiffusionError(Exception):
    """Base exception for protein diffusion errors."""
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 context: Dict[str, Any] = None):
        super().__init__(message)
        self.severity = severity
        self.context = context or {}
        self.error_id = str(uuid.uuid4())


class ModelLoadError(ProteinDiffusionError):
    """Error during model loading or initialization."""
    pass


class GenerationError(ProteinDiffusionError):
    """Error during protein generation."""
    pass


class ValidationError(ProteinDiffusionError):
    """Error during input validation."""
    pass


class RankingError(ProteinDiffusionError):
    """Error during protein ranking."""
    pass


class SecurityError(ProteinDiffusionError):
    """Security-related error."""
    def __init__(self, message: str, context: Dict[str, Any] = None):
        super().__init__(message, ErrorSeverity.CRITICAL, context)


class ResourceError(ProteinDiffusionError):
    """Resource-related error (memory, GPU, etc.)."""
    pass


class ErrorRecoveryManager:
    """Manages error recovery strategies and circuit breakers."""
    
    def __init__(self):
        self.error_history: deque = deque(maxlen=1000)
        self.circuit_breakers: Dict[str, CircuitBreakerState] = defaultdict(CircuitBreakerState)
        self.error_callbacks: List[Callable[[ErrorEvent], None]] = []
        self.recovery_strategies: Dict[str, RecoveryStrategy] = {}
        self.fallback_functions: Dict[str, Callable] = {}
        self.lock = threading.RLock()
        
        # Default recovery strategies
        self.set_default_strategies()
    
    def set_default_strategies(self):
        """Set default recovery strategies for common components."""
        self.recovery_strategies.update({
            'generation': RecoveryStrategy.RETRY,
            'ranking': RecoveryStrategy.FALLBACK,
            'validation': RecoveryStrategy.GRACEFUL_DEGRADATION,
            'model_loading': RecoveryStrategy.CIRCUIT_BREAKER,
            'structure_prediction': RecoveryStrategy.FALLBACK
        })
    
    def register_error_callback(self, callback: Callable[[ErrorEvent], None]):
        """Register a callback to be called when errors occur."""
        self.error_callbacks.append(callback)
    
    def register_fallback(self, function_name: str, fallback_func: Callable):
        """Register a fallback function for a specific function."""
        self.fallback_functions[function_name] = fallback_func
    
    def set_recovery_strategy(self, component: str, strategy: RecoveryStrategy):
        """Set recovery strategy for a component."""
        self.recovery_strategies[component] = strategy
    
    def record_error(self, error: Exception, component: str, function: str, 
                    context: Dict[str, Any] = None) -> ErrorEvent:
        """Record an error event."""
        error_event = ErrorEvent(
            error_type=type(error).__name__,
            error_message=str(error),
            severity=getattr(error, 'severity', ErrorSeverity.MEDIUM),
            component=component,
            function=function,
            context=context or {},
            stack_trace=traceback.format_exc()
        )
        
        with self.lock:
            self.error_history.append(error_event)
            
            # Update circuit breaker state
            cb_key = f"{component}.{function}"
            cb_state = self.circuit_breakers[cb_key]
            cb_state.failure_count += 1
            cb_state.last_failure_time = time.time()
            cb_state.total_requests += 1
            
            # Check if circuit should open
            if cb_state.failure_count >= 5 and not cb_state.is_open:
                cb_state.is_open = True
                logger.critical(f"Circuit breaker opened for {cb_key}")
        
        # Call error callbacks
        for callback in self.error_callbacks:
            try:
                callback(error_event)
            except Exception as cb_error:
                logger.error(f"Error callback failed: {cb_error}")
        
        return error_event
    
    def record_success(self, component: str, function: str):
        """Record a successful operation."""
        with self.lock:
            cb_key = f"{component}.{function}"
            cb_state = self.circuit_breakers[cb_key]
            cb_state.success_count += 1
            cb_state.total_requests += 1
            
            # Reset circuit breaker if enough successes
            if cb_state.is_open and cb_state.success_count >= 3:
                cb_state.is_open = False
                cb_state.failure_count = 0
                logger.info(f"Circuit breaker reset for {cb_key}")
    
    def is_circuit_open(self, component: str, function: str) -> bool:
        """Check if circuit breaker is open."""
        cb_key = f"{component}.{function}"
        cb_state = self.circuit_breakers[cb_key]
        
        # Check if enough time has passed to try half-open state
        if cb_state.is_open:
            time_since_failure = time.time() - cb_state.last_failure_time
            if time_since_failure > 60:  # 1 minute timeout
                cb_state.is_open = False  # Try half-open
                logger.info(f"Circuit breaker half-open for {cb_key}")
        
        return cb_state.is_open
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        with self.lock:
            if not self.error_history:
                return {'total_errors': 0}
            
            # Analyze error history
            total_errors = len(self.error_history)
            errors_by_type = defaultdict(int)
            errors_by_component = defaultdict(int)
            errors_by_severity = defaultdict(int)
            
            recent_errors = [e for e in self.error_history 
                           if time.time() - e.timestamp < 3600]  # Last hour
            
            for error in self.error_history:
                errors_by_type[error.error_type] += 1
                errors_by_component[error.component] += 1
                errors_by_severity[error.severity.value] += 1
            
            circuit_breaker_states = {
                key: {
                    'is_open': state.is_open,
                    'failure_count': state.failure_count,
                    'success_count': state.success_count,
                    'total_requests': state.total_requests,
                    'failure_rate': state.failure_count / max(state.total_requests, 1)
                }
                for key, state in self.circuit_breakers.items()
            }
            
            return {
                'total_errors': total_errors,
                'recent_errors': len(recent_errors),
                'errors_by_type': dict(errors_by_type),
                'errors_by_component': dict(errors_by_component),
                'errors_by_severity': dict(errors_by_severity),
                'circuit_breakers': circuit_breaker_states
            }


# Global error recovery manager instance
error_recovery = ErrorRecoveryManager()


def robust_execution(component: str, recovery_strategy: RecoveryStrategy = None,
                    max_retries: int = 3, retry_delay: float = 1.0,
                    fallback_result: Any = None):
    """
    Decorator for robust function execution with error handling and recovery.
    
    Args:
        component: Component name for error tracking
        recovery_strategy: Recovery strategy to use
        max_retries: Maximum number of retries
        retry_delay: Initial delay between retries (exponential backoff)
        fallback_result: Result to return if all else fails
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            function_name = func.__name__
            strategy = recovery_strategy or error_recovery.recovery_strategies.get(
                component, RecoveryStrategy.NONE
            )
            
            # Check circuit breaker
            if strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                if error_recovery.is_circuit_open(component, function_name):
                    raise ProteinDiffusionError(
                        f"Circuit breaker open for {component}.{function_name}",
                        ErrorSeverity.HIGH,
                        {'circuit_breaker': True}
                    )
            
            last_exception = None
            retries = 0
            
            while retries <= max_retries:
                try:
                    result = func(*args, **kwargs)
                    
                    # Record success
                    if retries > 0:
                        logger.info(f"Function {function_name} succeeded after {retries} retries")
                    error_recovery.record_success(component, function_name)
                    
                    return result
                    
                except Exception as e:
                    last_exception = e
                    error_event = error_recovery.record_error(
                        e, component, function_name, 
                        {'args_count': len(args), 'kwargs_keys': list(kwargs.keys())}
                    )
                    
                    # Apply recovery strategy
                    if strategy == RecoveryStrategy.NONE or retries >= max_retries:
                        break
                    elif strategy == RecoveryStrategy.RETRY:
                        retries += 1
                        delay = retry_delay * (2 ** (retries - 1))  # Exponential backoff
                        logger.warning(f"Retrying {function_name} in {delay}s (attempt {retries}/{max_retries})")
                        time.sleep(delay)
                        continue
                    elif strategy == RecoveryStrategy.FALLBACK:
                        fallback_func = error_recovery.fallback_functions.get(function_name)
                        if fallback_func:
                            try:
                                logger.info(f"Using fallback for {function_name}")
                                return fallback_func(*args, **kwargs)
                            except Exception as fb_error:
                                logger.error(f"Fallback also failed: {fb_error}")
                        break
                    elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                        logger.warning(f"Graceful degradation for {function_name}")
                        return fallback_result
                    else:
                        break
            
            # All recovery attempts failed
            if isinstance(last_exception, ProteinDiffusionError):
                raise last_exception
            else:
                raise GenerationError(
                    f"Function {function_name} failed after all recovery attempts: {last_exception}",
                    ErrorSeverity.HIGH,
                    {'original_error': str(last_exception), 'retries': retries}
                )
        
        return wrapper
    return decorator


@contextmanager
def error_context(component: str, operation: str, **context_data):
    """Context manager for error handling with additional context."""
    start_time = time.time()
    try:
        yield
    except Exception as e:
        context_data.update({
            'operation': operation,
            'duration': time.time() - start_time
        })
        error_recovery.record_error(e, component, operation, context_data)
        raise
    else:
        error_recovery.record_success(component, operation)


class SafetyValidator:
    """Validates inputs and operations for safety."""
    
    @staticmethod
    def validate_sequence_input(sequence: str) -> Tuple[bool, str]:
        """Validate protein sequence input."""
        if not isinstance(sequence, str):
            return False, f"Sequence must be string, got {type(sequence)}"
        
        if not sequence:
            return False, "Empty sequence not allowed"
        
        if len(sequence) > 10000:
            return False, f"Sequence too long: {len(sequence)} > 10000"
        
        # Check for valid amino acids
        valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
        invalid_chars = [c for c in sequence.upper() if c not in valid_aas and c.isalpha()]
        if invalid_chars:
            return False, f"Invalid amino acids: {set(invalid_chars)}"
        
        # Check for suspicious patterns
        if len(set(sequence)) == 1 and len(sequence) > 50:
            return False, "Suspicious: sequence contains only one amino acid"
        
        return True, "Valid"
    
    @staticmethod
    def validate_generation_params(params: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate generation parameters."""
        if not isinstance(params, dict):
            return False, "Parameters must be dictionary"
        
        # Check required parameters
        num_samples = params.get('num_samples', 1)
        if not isinstance(num_samples, int) or num_samples < 1 or num_samples > 1000:
            return False, f"num_samples must be integer 1-1000, got {num_samples}"
        
        max_length = params.get('max_length', 256)
        if not isinstance(max_length, int) or max_length < 1 or max_length > 2048:
            return False, f"max_length must be integer 1-2048, got {max_length}"
        
        temperature = params.get('temperature', 1.0)
        if not isinstance(temperature, (int, float)) or temperature <= 0 or temperature > 5.0:
            return False, f"temperature must be float 0-5, got {temperature}"
        
        return True, "Valid"
    
    @staticmethod
    def sanitize_motif_input(motif: str) -> str:
        """Sanitize motif input string."""
        if not motif:
            return ""
        
        # Remove non-alphanumeric characters except underscore
        sanitized = re.sub(r'[^A-Za-z0-9_]', '', motif)
        
        # Limit length
        if len(sanitized) > 100:
            sanitized = sanitized[:100]
        
        return sanitized.upper()


class ResourceMonitor:
    """Monitor system resources and prevent resource exhaustion."""
    
    def __init__(self):
        self.memory_threshold = 0.9  # 90% memory usage threshold
        self.gpu_memory_threshold = 0.9  # 90% GPU memory threshold
    
    def check_memory_usage(self) -> Tuple[bool, Dict[str, Any]]:
        """Check system memory usage."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            memory_info = {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_percent': memory.percent / 100,
                'available_percent': (100 - memory.percent) / 100
            }
            
            is_safe = memory_info['used_percent'] < self.memory_threshold
            
            return is_safe, memory_info
            
        except ImportError:
            logger.warning("psutil not available for memory monitoring")
            return True, {'error': 'monitoring_unavailable'}
        except Exception as e:
            logger.error(f"Memory check failed: {e}")
            return True, {'error': str(e)}
    
    def check_gpu_memory(self) -> Tuple[bool, Dict[str, Any]]:
        """Check GPU memory usage if available."""
        if not TORCH_AVAILABLE:
            return True, {'gpu_available': False}
        
        try:
            import torch
            if not torch.cuda.is_available():
                return True, {'cuda_available': False}
            
            gpu_info = {}
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                memory_cached = torch.cuda.memory_reserved(i) / (1024**3)
                memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                
                used_percent = (memory_allocated + memory_cached) / memory_total
                
                gpu_info[f'gpu_{i}'] = {
                    'allocated_gb': memory_allocated,
                    'cached_gb': memory_cached,
                    'total_gb': memory_total,
                    'used_percent': used_percent,
                    'is_safe': used_percent < self.gpu_memory_threshold
                }
            
            overall_safe = all(info['is_safe'] for info in gpu_info.values())
            gpu_info['overall_safe'] = overall_safe
            
            return overall_safe, gpu_info
            
        except Exception as e:
            logger.error(f"GPU memory check failed: {e}")
            return True, {'error': str(e)}
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get overall resource status."""
        memory_safe, memory_info = self.check_memory_usage()
        gpu_safe, gpu_info = self.check_gpu_memory()
        
        return {
            'timestamp': time.time(),
            'memory_safe': memory_safe,
            'memory_info': memory_info,
            'gpu_safe': gpu_safe,
            'gpu_info': gpu_info,
            'overall_safe': memory_safe and gpu_safe
        }


class ErrorReportingSystem:
    """System for collecting and reporting errors."""
    
    def __init__(self):
        self.reports = deque(maxlen=10000)
        self.alert_thresholds = {
            ErrorSeverity.LOW: 50,      # Alert after 50 low severity errors
            ErrorSeverity.MEDIUM: 20,   # Alert after 20 medium severity errors
            ErrorSeverity.HIGH: 5,      # Alert after 5 high severity errors
            ErrorSeverity.CRITICAL: 1   # Alert immediately for critical errors
        }
    
    def generate_error_report(self, time_window: int = 3600) -> Dict[str, Any]:
        """Generate comprehensive error report for given time window."""
        current_time = time.time()
        recent_errors = [
            error for error in error_recovery.error_history
            if current_time - error.timestamp <= time_window
        ]
        
        if not recent_errors:
            return {
                'time_window_hours': time_window / 3600,
                'total_errors': 0,
                'status': 'healthy'
            }
        
        # Analyze errors
        error_analysis = {
            'time_window_hours': time_window / 3600,
            'total_errors': len(recent_errors),
            'errors_by_severity': defaultdict(int),
            'errors_by_component': defaultdict(int),
            'errors_by_type': defaultdict(int),
            'most_frequent_errors': [],
            'critical_errors': [],
            'recommendations': []
        }
        
        for error in recent_errors:
            error_analysis['errors_by_severity'][error.severity.value] += 1
            error_analysis['errors_by_component'][error.component] += 1
            error_analysis['errors_by_type'][error.error_type] += 1
            
            if error.severity == ErrorSeverity.CRITICAL:
                error_analysis['critical_errors'].append({
                    'id': error.id,
                    'timestamp': error.timestamp,
                    'message': error.error_message,
                    'component': error.component
                })
        
        # Find most frequent errors
        error_frequency = defaultdict(int)
        for error in recent_errors:
            key = f"{error.component}.{error.error_type}"
            error_frequency[key] += 1
        
        error_analysis['most_frequent_errors'] = [
            {'error': error, 'count': count}
            for error, count in sorted(error_frequency.items(), 
                                     key=lambda x: x[1], reverse=True)[:5]
        ]
        
        # Generate recommendations
        recommendations = []
        if error_analysis['errors_by_severity']['critical'] > 0:
            recommendations.append("URGENT: Critical errors detected - immediate attention required")
        
        if error_analysis['errors_by_component'].get('generation', 0) > 10:
            recommendations.append("High generation error rate - check model and parameters")
        
        if error_analysis['errors_by_component'].get('ranking', 0) > 10:
            recommendations.append("High ranking error rate - check input validation")
        
        error_analysis['recommendations'] = recommendations
        
        # Determine overall status
        if error_analysis['errors_by_severity']['critical'] > 0:
            error_analysis['status'] = 'critical'
        elif error_analysis['errors_by_severity']['high'] > 5:
            error_analysis['status'] = 'degraded'
        elif len(recent_errors) > 100:
            error_analysis['status'] = 'unstable'
        else:
            error_analysis['status'] = 'stable'
        
        # Convert defaultdicts to regular dicts for JSON serialization
        for key in ['errors_by_severity', 'errors_by_component', 'errors_by_type']:
            error_analysis[key] = dict(error_analysis[key])
        
        return error_analysis


# Global instances
safety_validator = SafetyValidator()
resource_monitor = ResourceMonitor()
error_reporter = ErrorReportingSystem()


# Register default error callback for logging
def default_error_callback(error_event: ErrorEvent):
    """Default error callback that logs errors."""
    if error_event.severity == ErrorSeverity.CRITICAL:
        logger.critical(f"CRITICAL ERROR in {error_event.component}.{error_event.function}: "
                       f"{error_event.error_message}")
    elif error_event.severity == ErrorSeverity.HIGH:
        logger.error(f"HIGH SEVERITY ERROR in {error_event.component}.{error_event.function}: "
                    f"{error_event.error_message}")
    elif error_event.severity == ErrorSeverity.MEDIUM:
        logger.warning(f"ERROR in {error_event.component}.{error_event.function}: "
                      f"{error_event.error_message}")
    else:
        logger.info(f"Low severity error in {error_event.component}.{error_event.function}: "
                   f"{error_event.error_message}")


error_recovery.register_error_callback(default_error_callback)


def health_check() -> Dict[str, Any]:
    """Comprehensive system health check."""
    try:
        # Check resource status
        resource_status = resource_monitor.get_resource_status()
        
        # Get error statistics
        error_stats = error_recovery.get_error_statistics()
        
        # Generate recent error report
        error_report = error_reporter.generate_error_report(time_window=1800)  # 30 minutes
        
        # Determine overall health
        overall_health = "healthy"
        
        if not resource_status['overall_safe']:
            overall_health = "resource_constrained"
        elif error_report['status'] == 'critical':
            overall_health = "critical"
        elif error_report['status'] in ['degraded', 'unstable']:
            overall_health = "degraded"
        elif error_stats['recent_errors'] > 20:
            overall_health = "unstable"
        
        return {
            'timestamp': time.time(),
            'overall_health': overall_health,
            'resource_status': resource_status,
            'error_statistics': error_stats,
            'recent_error_report': error_report,
            'circuit_breakers_open': len([
                cb for cb, state in error_stats.get('circuit_breakers', {}).items()
                if state.get('is_open', False)
            ])
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            'timestamp': time.time(),
            'overall_health': 'health_check_failed',
            'error': str(e),
            'stack_trace': traceback.format_exc()
        }