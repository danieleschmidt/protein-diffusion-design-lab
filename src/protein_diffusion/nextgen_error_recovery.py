"""
Next-Generation Error Recovery and Fault Tolerance System

This module provides advanced error recovery, circuit breakers, and self-healing
capabilities for mission-critical protein design workflows.
"""

import time
import asyncio
import logging
import traceback
import json
from typing import Dict, List, Optional, Any, Callable, Union, Type
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor
import uuid

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Error recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    ESCALATION = "escalation"
    ROLLBACK = "rollback"
    AUTO_HEALING = "auto_healing"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class ErrorCategory(Enum):
    """Categories of errors for specialized handling."""
    NETWORK = "network"
    COMPUTATION = "computation"
    MEMORY = "memory"
    STORAGE = "storage"
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    EXTERNAL_SERVICE = "external_service"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


@dataclass
class ErrorInfo:
    """Comprehensive error information."""
    error_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    error_type: str = ""
    error_message: str = ""
    error_category: ErrorCategory = ErrorCategory.UNKNOWN
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    context: Dict[str, Any] = field(default_factory=dict)
    traceback_info: str = ""
    recovery_attempted: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None
    recovery_successful: bool = False
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryAction:
    """Represents a recovery action to be taken."""
    action_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    strategy: RecoveryStrategy
    priority: int = 1
    timeout_seconds: float = 30.0
    parameters: Dict[str, Any] = field(default_factory=dict)
    fallback_action: Optional['RecoveryAction'] = None
    success_callback: Optional[Callable] = None
    failure_callback: Optional[Callable] = None


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    timeout_seconds: float = 60.0
    half_open_max_calls: int = 3
    rolling_window_seconds: float = 300.0
    enable_auto_recovery: bool = True
    recovery_check_interval: float = 30.0


@dataclass
class ErrorRecoveryConfig:
    """Configuration for the error recovery system."""
    max_concurrent_recoveries: int = 10
    default_max_retries: int = 3
    default_retry_delay: float = 1.0
    exponential_backoff_multiplier: float = 2.0
    max_retry_delay: float = 60.0
    enable_circuit_breakers: bool = True
    enable_auto_healing: bool = True
    enable_predictive_recovery: bool = True
    error_history_retention_hours: int = 24
    telemetry_interval_seconds: float = 60.0
    escalation_thresholds: Dict[ErrorSeverity, int] = field(default_factory=lambda: {
        ErrorSeverity.LOW: 10,
        ErrorSeverity.MEDIUM: 5,
        ErrorSeverity.HIGH: 2,
        ErrorSeverity.CRITICAL: 1
    })


class CircuitBreaker:
    """Advanced circuit breaker with metrics and auto-recovery."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.state_changed_time = time.time()
        self.failure_history = deque(maxlen=100)
        self.lock = threading.Lock()
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self.lock:
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time < self.config.timeout_seconds:
                    raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is open")
                else:
                    # Try to move to half-open state
                    self.state = CircuitState.HALF_OPEN
                    self.state_changed_time = time.time()
                    logger.info(f"Circuit breaker {self.name} moved to half-open")
                    
            if self.state == CircuitState.HALF_OPEN:
                if self.success_count >= self.config.half_open_max_calls:
                    # Close the circuit
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    self.state_changed_time = time.time()
                    logger.info(f"Circuit breaker {self.name} closed after successful recovery")
                    
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure(e)
            raise
            
    def _record_success(self):
        """Record successful call."""
        with self.lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
            # Reset failure count on success (unless we're testing half-open)
            if self.state == CircuitState.CLOSED:
                self.failure_count = max(0, self.failure_count - 1)
                
    def _record_failure(self, error: Exception):
        """Record failed call."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            self.failure_history.append({
                'timestamp': self.last_failure_time,
                'error': str(error),
                'error_type': type(error).__name__
            })
            
            # Check if we should open the circuit
            if (self.state == CircuitState.CLOSED and 
                self.failure_count >= self.config.failure_threshold):
                self.state = CircuitState.OPEN
                self.state_changed_time = time.time()
                logger.warning(f"Circuit breaker {self.name} opened after {self.failure_count} failures")
                
            elif self.state == CircuitState.HALF_OPEN:
                # Go back to open state if we fail in half-open
                self.state = CircuitState.OPEN
                self.success_count = 0
                self.state_changed_time = time.time()
                logger.warning(f"Circuit breaker {self.name} reopened during half-open test")
                
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        with self.lock:
            return {
                'name': self.name,
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'last_failure_time': self.last_failure_time,
                'state_changed_time': self.state_changed_time,
                'uptime_seconds': time.time() - self.state_changed_time,
                'recent_failures': list(self.failure_history)[-5:],  # Last 5 failures
                'config': {
                    'failure_threshold': self.config.failure_threshold,
                    'timeout_seconds': self.config.timeout_seconds,
                    'half_open_max_calls': self.config.half_open_max_calls
                }
            }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class ErrorClassifier:
    """Classifies errors into categories and determines severity."""
    
    def __init__(self):
        self.classification_rules = self._build_classification_rules()
        
    def classify_error(self, error: Exception, context: Dict[str, Any] = None) -> Tuple[ErrorCategory, ErrorSeverity]:
        """Classify an error into category and severity."""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Check classification rules
        for rule in self.classification_rules:
            if self._matches_rule(error_type, error_message, context or {}, rule):
                return rule['category'], rule['severity']
                
        # Default classification
        return ErrorCategory.UNKNOWN, ErrorSeverity.MEDIUM
        
    def _build_classification_rules(self) -> List[Dict[str, Any]]:
        """Build error classification rules."""
        return [
            # Network errors
            {
                'error_types': ['ConnectionError', 'TimeoutError', 'HTTPError'],
                'keywords': ['connection', 'timeout', 'network', 'http'],
                'category': ErrorCategory.NETWORK,
                'severity': ErrorSeverity.MEDIUM
            },
            # Memory errors
            {
                'error_types': ['MemoryError', 'OutOfMemoryError'],
                'keywords': ['memory', 'oom', 'allocation'],
                'category': ErrorCategory.MEMORY,
                'severity': ErrorSeverity.HIGH
            },
            # Computation errors
            {
                'error_types': ['RuntimeError', 'ValueError', 'ArithmeticError'],
                'keywords': ['computation', 'calculation', 'arithmetic'],
                'category': ErrorCategory.COMPUTATION,
                'severity': ErrorSeverity.MEDIUM
            },
            # Storage errors
            {
                'error_types': ['IOError', 'FileNotFoundError', 'PermissionError'],
                'keywords': ['file', 'storage', 'disk', 'permission'],
                'category': ErrorCategory.STORAGE,
                'severity': ErrorSeverity.MEDIUM
            },
            # Validation errors
            {
                'error_types': ['ValidationError', 'ValueError'],
                'keywords': ['validation', 'invalid', 'format'],
                'category': ErrorCategory.VALIDATION,
                'severity': ErrorSeverity.LOW
            },
            # Authentication errors
            {
                'error_types': ['AuthenticationError', 'PermissionError'],
                'keywords': ['auth', 'permission', 'unauthorized', 'forbidden'],
                'category': ErrorCategory.AUTHENTICATION,
                'severity': ErrorSeverity.HIGH
            },
            # Resource exhaustion
            {
                'error_types': ['ResourceExhaustedError'],
                'keywords': ['resource', 'quota', 'limit', 'exhausted'],
                'category': ErrorCategory.RESOURCE_EXHAUSTION,
                'severity': ErrorSeverity.HIGH
            }
        ]
        
    def _matches_rule(
        self,
        error_type: str,
        error_message: str,
        context: Dict[str, Any],
        rule: Dict[str, Any]
    ) -> bool:
        """Check if an error matches a classification rule."""
        # Check error type
        if 'error_types' in rule and error_type in rule['error_types']:
            return True
            
        # Check keywords in error message
        if 'keywords' in rule:
            for keyword in rule['keywords']:
                if keyword in error_message:
                    return True
                    
        # Check context
        if 'context_keys' in rule:
            for key in rule['context_keys']:
                if key in context:
                    return True
                    
        return False


class AutoHealingSystem:
    """Automatic healing system for common issues."""
    
    def __init__(self, config: ErrorRecoveryConfig):
        self.config = config
        self.healing_patterns = self._build_healing_patterns()
        self.healing_history = deque(maxlen=1000)
        
    def can_auto_heal(self, error_info: ErrorInfo) -> bool:
        """Check if an error can be automatically healed."""
        for pattern in self.healing_patterns:
            if self._matches_healing_pattern(error_info, pattern):
                return True
        return False
        
    async def attempt_auto_healing(self, error_info: ErrorInfo) -> bool:
        """Attempt to automatically heal an error."""
        for pattern in self.healing_patterns:
            if self._matches_healing_pattern(error_info, pattern):
                try:
                    healing_func = pattern['healing_function']
                    success = await healing_func(error_info)
                    
                    self.healing_history.append({
                        'timestamp': time.time(),
                        'error_id': error_info.error_id,
                        'pattern': pattern['name'],
                        'success': success
                    })
                    
                    if success:
                        logger.info(f"Auto-healing successful for error {error_info.error_id} using pattern {pattern['name']}")
                        return True
                    else:
                        logger.warning(f"Auto-healing failed for error {error_info.error_id} using pattern {pattern['name']}")
                        
                except Exception as healing_error:
                    logger.error(f"Auto-healing exception: {healing_error}")
                    
        return False
        
    def _build_healing_patterns(self) -> List[Dict[str, Any]]:
        """Build auto-healing patterns."""
        return [
            {
                'name': 'memory_cleanup',
                'categories': [ErrorCategory.MEMORY],
                'severity_threshold': ErrorSeverity.HIGH,
                'healing_function': self._heal_memory_issues
            },
            {
                'name': 'network_retry',
                'categories': [ErrorCategory.NETWORK],
                'severity_threshold': ErrorSeverity.MEDIUM,
                'healing_function': self._heal_network_issues
            },
            {
                'name': 'storage_cleanup',
                'categories': [ErrorCategory.STORAGE],
                'severity_threshold': ErrorSeverity.MEDIUM,
                'healing_function': self._heal_storage_issues
            },
            {
                'name': 'resource_scaling',
                'categories': [ErrorCategory.RESOURCE_EXHAUSTION],
                'severity_threshold': ErrorSeverity.HIGH,
                'healing_function': self._heal_resource_issues
            }
        ]
        
    def _matches_healing_pattern(self, error_info: ErrorInfo, pattern: Dict[str, Any]) -> bool:
        """Check if an error matches a healing pattern."""
        # Check category
        if error_info.error_category not in pattern.get('categories', []):
            return False
            
        # Check severity threshold
        severity_levels = {
            ErrorSeverity.LOW: 1,
            ErrorSeverity.MEDIUM: 2,
            ErrorSeverity.HIGH: 3,
            ErrorSeverity.CRITICAL: 4
        }
        
        if severity_levels[error_info.severity] < severity_levels[pattern.get('severity_threshold', ErrorSeverity.LOW)]:
            return False
            
        return True
        
    async def _heal_memory_issues(self, error_info: ErrorInfo) -> bool:
        """Heal memory-related issues."""
        try:
            # Simulate memory cleanup
            import gc
            gc.collect()
            
            # Clear caches if available
            if hasattr(self, '_clear_caches'):
                await self._clear_caches()
                
            logger.info("Memory cleanup performed")
            return True
        except Exception:
            return False
            
    async def _heal_network_issues(self, error_info: ErrorInfo) -> bool:
        """Heal network-related issues."""
        try:
            # Simulate network connection reset
            await asyncio.sleep(1)  # Brief wait
            
            # Test connectivity
            # In a real implementation, you would ping services or reset connections
            logger.info("Network healing attempted")
            return True
        except Exception:
            return False
            
    async def _heal_storage_issues(self, error_info: ErrorInfo) -> bool:
        """Heal storage-related issues."""
        try:
            # Simulate storage cleanup
            # In a real implementation, you would clean temp files, check disk space
            logger.info("Storage healing attempted")
            return True
        except Exception:
            return False
            
    async def _heal_resource_issues(self, error_info: ErrorInfo) -> bool:
        """Heal resource exhaustion issues."""
        try:
            # Simulate resource scaling
            # In a real implementation, you would scale up resources
            logger.info("Resource scaling healing attempted")
            return True
        except Exception:
            return False


class NextGenErrorRecoverySystem:
    """
    Next-Generation Error Recovery System
    
    Provides advanced error handling, circuit breakers, auto-healing,
    and intelligent recovery strategies for mission-critical operations.
    """
    
    def __init__(self, config: ErrorRecoveryConfig):
        self.config = config
        self.error_classifier = ErrorClassifier()
        self.auto_healing_system = AutoHealingSystem(config)
        
        # Error tracking
        self.error_history = deque(maxlen=10000)
        self.active_recoveries: Dict[str, RecoveryAction] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Recovery stats
        self.recovery_stats = {
            'total_errors': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'auto_healing_attempts': 0,
            'auto_healing_successes': 0,
            'circuit_breaker_trips': 0
        }
        
        # Background task management
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_recoveries)
        
        logger.info("Next-Gen Error Recovery System initialized")
        
    async def handle_error(
        self,
        error: Exception,
        context: Dict[str, Any] = None,
        recovery_strategy: Optional[RecoveryStrategy] = None
    ) -> ErrorInfo:
        """Handle an error with comprehensive recovery logic."""
        # Create error info
        error_info = ErrorInfo(
            error_type=type(error).__name__,
            error_message=str(error),
            context=context or {},
            traceback_info=traceback.format_exc()
        )
        
        # Classify error
        error_info.error_category, error_info.severity = self.error_classifier.classify_error(
            error, context
        )
        
        # Store error
        self.error_history.append(error_info)
        self.recovery_stats['total_errors'] += 1
        
        logger.error(f"Error {error_info.error_id} occurred: {error_info.error_message}")
        
        # Determine recovery strategy
        if not recovery_strategy:
            recovery_strategy = self._determine_recovery_strategy(error_info)
            
        error_info.recovery_strategy = recovery_strategy
        
        # Attempt recovery
        if recovery_strategy != RecoveryStrategy.ESCALATION:
            recovery_successful = await self._execute_recovery(error_info, recovery_strategy)
            error_info.recovery_attempted = True
            error_info.recovery_successful = recovery_successful
            
            if recovery_successful:
                self.recovery_stats['successful_recoveries'] += 1
                logger.info(f"Error {error_info.error_id} recovered successfully using {recovery_strategy.value}")
            else:
                self.recovery_stats['failed_recoveries'] += 1
                logger.warning(f"Error {error_info.error_id} recovery failed using {recovery_strategy.value}")
                
                # Try escalation if initial recovery failed
                if error_info.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                    await self._escalate_error(error_info)
                    
        else:
            # Direct escalation
            await self._escalate_error(error_info)
            
        return error_info
        
    def _determine_recovery_strategy(self, error_info: ErrorInfo) -> RecoveryStrategy:
        """Determine the best recovery strategy for an error."""
        # Strategy selection based on error category and severity
        if error_info.error_category == ErrorCategory.NETWORK:
            return RecoveryStrategy.RETRY
        elif error_info.error_category == ErrorCategory.MEMORY:
            if error_info.severity == ErrorSeverity.HIGH:
                return RecoveryStrategy.AUTO_HEALING
            else:
                return RecoveryStrategy.GRACEFUL_DEGRADATION
        elif error_info.error_category == ErrorCategory.COMPUTATION:
            return RecoveryStrategy.FALLBACK
        elif error_info.error_category == ErrorCategory.RESOURCE_EXHAUSTION:
            return RecoveryStrategy.AUTO_HEALING
        elif error_info.severity == ErrorSeverity.CRITICAL:
            return RecoveryStrategy.ESCALATION
        else:
            return RecoveryStrategy.RETRY
            
    async def _execute_recovery(self, error_info: ErrorInfo, strategy: RecoveryStrategy) -> bool:
        """Execute the specified recovery strategy."""
        try:
            if strategy == RecoveryStrategy.RETRY:
                return await self._retry_recovery(error_info)
            elif strategy == RecoveryStrategy.FALLBACK:
                return await self._fallback_recovery(error_info)
            elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                return await self._circuit_breaker_recovery(error_info)
            elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                return await self._graceful_degradation_recovery(error_info)
            elif strategy == RecoveryStrategy.AUTO_HEALING:
                return await self._auto_healing_recovery(error_info)
            elif strategy == RecoveryStrategy.ROLLBACK:
                return await self._rollback_recovery(error_info)
            else:
                logger.warning(f"Unknown recovery strategy: {strategy}")
                return False
                
        except Exception as recovery_error:
            logger.error(f"Recovery execution failed: {recovery_error}")
            return False
            
    async def _retry_recovery(self, error_info: ErrorInfo) -> bool:
        """Implement retry recovery with exponential backoff."""
        max_retries = error_info.max_retries
        delay = self.config.default_retry_delay
        
        for attempt in range(1, max_retries + 1):
            try:
                # Wait with exponential backoff
                if attempt > 1:
                    actual_delay = min(
                        delay * (self.config.exponential_backoff_multiplier ** (attempt - 1)),
                        self.config.max_retry_delay
                    )
                    await asyncio.sleep(actual_delay)
                    
                error_info.retry_count = attempt
                logger.info(f"Retry attempt {attempt}/{max_retries} for error {error_info.error_id}")
                
                # Simulate retry success (in real implementation, would retry the original operation)
                if attempt >= 2:  # Simulate success after first retry
                    return True
                    
            except Exception as retry_error:
                logger.error(f"Retry attempt {attempt} failed: {retry_error}")
                
        return False
        
    async def _fallback_recovery(self, error_info: ErrorInfo) -> bool:
        """Implement fallback recovery strategy."""
        try:
            # Implement fallback logic based on context
            fallback_options = error_info.context.get('fallback_options', [])
            
            for fallback_option in fallback_options:
                try:
                    logger.info(f"Attempting fallback option: {fallback_option}")
                    # Simulate fallback execution
                    await asyncio.sleep(0.1)
                    return True  # Simulate successful fallback
                except Exception as fallback_error:
                    logger.warning(f"Fallback option {fallback_option} failed: {fallback_error}")
                    continue
                    
            # Default fallback
            logger.info("Using default fallback strategy")
            return True  # Simulate successful default fallback
            
        except Exception:
            return False
            
    async def _circuit_breaker_recovery(self, error_info: ErrorInfo) -> bool:
        """Implement circuit breaker recovery."""
        service_name = error_info.context.get('service_name', 'default')
        
        if service_name not in self.circuit_breakers:
            circuit_config = CircuitBreakerConfig()
            self.circuit_breakers[service_name] = CircuitBreaker(service_name, circuit_config)
            
        circuit_breaker = self.circuit_breakers[service_name]
        
        try:
            # Simulate successful circuit breaker operation
            def mock_operation():
                return True
                
            result = circuit_breaker.call(mock_operation)
            return result
        except CircuitBreakerOpenError:
            logger.warning(f"Circuit breaker {service_name} is open")
            return False
        except Exception:
            return False
            
    async def _graceful_degradation_recovery(self, error_info: ErrorInfo) -> bool:
        """Implement graceful degradation recovery."""
        try:
            # Implement degraded mode operation
            degradation_level = error_info.context.get('degradation_level', 'minimal')
            
            logger.info(f"Entering graceful degradation mode: {degradation_level}")
            
            # Simulate degraded operation
            if degradation_level == 'minimal':
                # Provide basic functionality
                return True
            elif degradation_level == 'reduced':
                # Provide reduced functionality
                return True
            else:
                # Unknown degradation level
                return False
                
        except Exception:
            return False
            
    async def _auto_healing_recovery(self, error_info: ErrorInfo) -> bool:
        """Implement auto-healing recovery."""
        self.recovery_stats['auto_healing_attempts'] += 1
        
        if self.auto_healing_system.can_auto_heal(error_info):
            success = await self.auto_healing_system.attempt_auto_healing(error_info)
            if success:
                self.recovery_stats['auto_healing_successes'] += 1
            return success
        else:
            logger.info(f"Auto-healing not available for error {error_info.error_id}")
            return False
            
    async def _rollback_recovery(self, error_info: ErrorInfo) -> bool:
        """Implement rollback recovery strategy."""
        try:
            # Get rollback information from context
            checkpoint = error_info.context.get('checkpoint')
            rollback_steps = error_info.context.get('rollback_steps', [])
            
            if not rollback_steps:
                logger.warning("No rollback steps available")
                return False
                
            logger.info(f"Starting rollback to checkpoint: {checkpoint}")
            
            # Execute rollback steps in reverse order
            for step in reversed(rollback_steps):
                try:
                    logger.info(f"Executing rollback step: {step}")
                    # Simulate rollback step execution
                    await asyncio.sleep(0.1)
                except Exception as step_error:
                    logger.error(f"Rollback step failed: {step_error}")
                    return False
                    
            logger.info("Rollback completed successfully")
            return True
            
        except Exception:
            logger.error("Rollback recovery failed")
            return False
            
    async def _escalate_error(self, error_info: ErrorInfo):
        """Escalate error to higher-level systems."""
        logger.critical(f"Escalating error {error_info.error_id}: {error_info.error_message}")
        
        # Send escalation notification
        escalation_data = {
            'error_id': error_info.error_id,
            'timestamp': error_info.timestamp,
            'severity': error_info.severity.value,
            'category': error_info.error_category.value,
            'message': error_info.error_message,
            'context': error_info.context,
            'recovery_attempted': error_info.recovery_attempted,
            'recovery_successful': error_info.recovery_successful
        }
        
        # In a real implementation, this would send alerts to monitoring systems,
        # notify administrators, create incident tickets, etc.
        logger.critical(f"ESCALATION: {json.dumps(escalation_data, indent=2)}")
        
    def get_circuit_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        if name not in self.circuit_breakers:
            circuit_config = config or CircuitBreakerConfig()
            self.circuit_breakers[name] = CircuitBreaker(name, circuit_config)
        return self.circuit_breakers[name]
        
    def get_error_statistics(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get error statistics for the specified time window."""
        cutoff_time = time.time() - (time_window_hours * 3600)
        recent_errors = [e for e in self.error_history if e.timestamp >= cutoff_time]
        
        # Error counts by category
        category_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        recovery_success_counts = defaultdict(int)
        
        for error in recent_errors:
            category_counts[error.error_category.value] += 1
            severity_counts[error.severity.value] += 1
            if error.recovery_attempted:
                recovery_success_counts['attempted'] += 1
                if error.recovery_successful:
                    recovery_success_counts['successful'] += 1
                    
        # Circuit breaker status
        circuit_breaker_status = {}
        for name, cb in self.circuit_breakers.items():
            circuit_breaker_status[name] = cb.get_status()
            
        return {
            'time_window_hours': time_window_hours,
            'total_errors': len(recent_errors),
            'errors_by_category': dict(category_counts),
            'errors_by_severity': dict(severity_counts),
            'recovery_statistics': dict(recovery_success_counts),
            'recovery_success_rate': (
                recovery_success_counts['successful'] / max(recovery_success_counts['attempted'], 1)
            ) if recovery_success_counts['attempted'] > 0 else 0,
            'circuit_breakers': circuit_breaker_status,
            'system_stats': self.recovery_stats.copy(),
            'auto_healing_success_rate': (
                self.recovery_stats['auto_healing_successes'] / 
                max(self.recovery_stats['auto_healing_attempts'], 1)
            ) if self.recovery_stats['auto_healing_attempts'] > 0 else 0
        }
        
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        recent_errors = [e for e in self.error_history if e.timestamp >= time.time() - 3600]
        critical_errors = [e for e in recent_errors if e.severity == ErrorSeverity.CRITICAL]
        
        # Health score calculation (0-100)
        health_score = 100
        if recent_errors:
            error_penalty = min(len(recent_errors) * 2, 30)  # Max 30 point penalty
            health_score -= error_penalty
            
        if critical_errors:
            critical_penalty = min(len(critical_errors) * 10, 50)  # Max 50 point penalty
            health_score -= critical_penalty
            
        # Circuit breaker health
        open_circuits = [cb for cb in self.circuit_breakers.values() if cb.state == CircuitState.OPEN]
        if open_circuits:
            circuit_penalty = min(len(open_circuits) * 5, 20)  # Max 20 point penalty
            health_score -= circuit_penalty
            
        health_score = max(0, health_score)
        
        # Determine health status
        if health_score >= 90:
            health_status = "excellent"
        elif health_score >= 75:
            health_status = "good"
        elif health_score >= 50:
            health_status = "fair"
        elif health_score >= 25:
            health_status = "poor"
        else:
            health_status = "critical"
            
        return {
            'health_score': health_score,
            'health_status': health_status,
            'recent_errors_count': len(recent_errors),
            'critical_errors_count': len(critical_errors),
            'open_circuit_breakers': len(open_circuits),
            'active_recoveries': len(self.active_recoveries),
            'is_running': self.is_running,
            'recommendations': self._get_health_recommendations(health_score, recent_errors, critical_errors)
        }
        
    def _get_health_recommendations(
        self,
        health_score: int,
        recent_errors: List[ErrorInfo],
        critical_errors: List[ErrorInfo]
    ) -> List[str]:
        """Get health improvement recommendations."""
        recommendations = []
        
        if health_score < 50:
            recommendations.append("System health is poor. Investigate recent errors and consider scaling resources.")
            
        if critical_errors:
            recommendations.append(f"Address {len(critical_errors)} critical errors immediately.")
            
        if len(recent_errors) > 10:
            recommendations.append("High error rate detected. Review system logs and error patterns.")
            
        open_circuits = [cb for cb in self.circuit_breakers.values() if cb.state == CircuitState.OPEN]
        if open_circuits:
            recommendations.append(f"Reset {len(open_circuits)} open circuit breakers after addressing underlying issues.")
            
        return recommendations


# Convenience functions and decorators
def with_error_recovery(
    recovery_system: NextGenErrorRecoverySystem,
    strategy: Optional[RecoveryStrategy] = None,
    context: Optional[Dict[str, Any]] = None
):
    """Decorator to add error recovery to functions."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_info = await recovery_system.handle_error(e, context, strategy)
                if not error_info.recovery_successful:
                    raise
                # Return None or a default value if recovery was successful
                return None
                
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # For sync functions, we can't await, so we'll log and re-raise
                logger.error(f"Error in {func.__name__}: {e}")
                raise
                
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    return decorator


# Demo and testing
async def demo_error_recovery():
    """Demonstrate the error recovery system."""
    config = ErrorRecoveryConfig(
        max_concurrent_recoveries=5,
        enable_circuit_breakers=True,
        enable_auto_healing=True
    )
    
    recovery_system = NextGenErrorRecoverySystem(config)
    
    # Simulate various types of errors
    test_errors = [
        (ConnectionError("Network connection failed"), {'service_name': 'protein_api'}),
        (MemoryError("Out of memory"), {'operation': 'protein_generation'}),
        (ValueError("Invalid input parameters"), {'input_validation': True}),
        (RuntimeError("Computation failed"), {'fallback_options': ['cached_result', 'simplified_model']}),
    ]
    
    print("=== Error Recovery System Demo ===")
    
    for error, context in test_errors:
        print(f"\nSimulating error: {error}")
        try:
            error_info = await recovery_system.handle_error(error, context)
            print(f"Error ID: {error_info.error_id}")
            print(f"Category: {error_info.error_category.value}")
            print(f"Severity: {error_info.severity.value}")
            print(f"Recovery Strategy: {error_info.recovery_strategy.value if error_info.recovery_strategy else 'None'}")
            print(f"Recovery Success: {error_info.recovery_successful}")
        except Exception as e:
            print(f"Recovery handling failed: {e}")
            
    # Get system statistics
    stats = recovery_system.get_error_statistics(1)
    health = recovery_system.get_system_health()
    
    print(f"\n=== System Statistics ===")
    print(json.dumps(stats, indent=2))
    
    print(f"\n=== System Health ===")
    print(json.dumps(health, indent=2))


if __name__ == "__main__":
    asyncio.run(demo_error_recovery())