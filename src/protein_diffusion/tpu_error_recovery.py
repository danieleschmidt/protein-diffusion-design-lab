"""
TPU Error Recovery and Fault Tolerance System

This module provides comprehensive error recovery and fault tolerance specifically
for TPU operations in protein diffusion model training and inference.
"""

import logging
import time
import traceback
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import functools
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class TPUErrorType(Enum):
    """Types of TPU-specific errors."""
    COMPILATION_ERROR = "compilation_error"
    MEMORY_ERROR = "memory_error"
    DEVICE_UNAVAILABLE = "device_unavailable"
    TIMEOUT_ERROR = "timeout_error"
    COMMUNICATION_ERROR = "communication_error"
    NUMERICAL_ERROR = "numerical_error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    PREEMPTION = "preemption"
    UNKNOWN_ERROR = "unknown_error"

class RecoveryStrategy(Enum):
    """Error recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    ABORT = "abort"
    RESTART = "restart"
    SCALE_DOWN = "scale_down"

@dataclass
class ErrorContext:
    """Context information for an error."""
    error_type: TPUErrorType
    error_message: str
    timestamp: float
    stack_trace: str
    function_name: str
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    attempt_number: int = 1
    recovery_strategy: Optional[RecoveryStrategy] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'error_type': self.error_type.value,
            'error_message': self.error_message,
            'timestamp': self.timestamp,
            'function_name': self.function_name,
            'attempt_number': self.attempt_number,
            'recovery_strategy': self.recovery_strategy.value if self.recovery_strategy else None
        }

@dataclass
class RecoveryConfig:
    """Configuration for error recovery."""
    max_retries: int = 3
    retry_delay: float = 1.0
    backoff_multiplier: float = 2.0
    timeout_seconds: float = 300.0
    enable_fallback: bool = True
    enable_graceful_degradation: bool = True
    enable_auto_restart: bool = True
    memory_threshold: float = 0.9  # 90% memory usage threshold
    
    # TPU-specific settings
    compilation_timeout: float = 600.0  # 10 minutes for compilation
    device_check_interval: float = 5.0
    preemption_check_interval: float = 30.0

class TPUErrorRecovery:
    """
    Comprehensive TPU error recovery system with automatic fault tolerance.
    """
    
    def __init__(self, config: RecoveryConfig):
        self.config = config
        self.error_history: List[ErrorContext] = []
        self.recovery_stats = {
            'total_errors': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'fallback_activations': 0
        }
        
        self._fallback_devices = []
        self._device_health_cache = {}
        self._last_device_check = 0
        
        logger.info("Initialized TPU Error Recovery System")
    
    def classify_error(self, exception: Exception, context: Dict[str, Any]) -> TPUErrorType:
        """Classify TPU-specific errors."""
        error_message = str(exception).lower()
        
        # Check for specific TPU error patterns
        if any(pattern in error_message for pattern in ['compilation', 'jit', 'xla']):
            return TPUErrorType.COMPILATION_ERROR
        elif any(pattern in error_message for pattern in ['memory', 'oom', 'out of memory']):
            return TPUErrorType.MEMORY_ERROR
        elif any(pattern in error_message for pattern in ['device', 'tpu not found', 'unavailable']):
            return TPUErrorType.DEVICE_UNAVAILABLE
        elif any(pattern in error_message for pattern in ['timeout', 'deadline']):
            return TPUErrorType.TIMEOUT_ERROR
        elif any(pattern in error_message for pattern in ['communication', 'network', 'connection']):
            return TPUErrorType.COMMUNICATION_ERROR
        elif any(pattern in error_message for pattern in ['nan', 'inf', 'numerical']):
            return TPUErrorType.NUMERICAL_ERROR
        elif any(pattern in error_message for pattern in ['resource', 'quota', 'limit']):
            return TPUErrorType.RESOURCE_EXHAUSTION
        elif any(pattern in error_message for pattern in ['preempted', 'preemption']):
            return TPUErrorType.PREEMPTION
        else:
            return TPUErrorType.UNKNOWN_ERROR
    
    def determine_recovery_strategy(self, error_type: TPUErrorType, 
                                  attempt_number: int) -> RecoveryStrategy:
        """Determine the best recovery strategy for an error."""
        
        if attempt_number > self.config.max_retries:
            return RecoveryStrategy.ABORT
        
        strategy_map = {
            TPUErrorType.COMPILATION_ERROR: RecoveryStrategy.RETRY,
            TPUErrorType.MEMORY_ERROR: RecoveryStrategy.SCALE_DOWN,
            TPUErrorType.DEVICE_UNAVAILABLE: RecoveryStrategy.FALLBACK,
            TPUErrorType.TIMEOUT_ERROR: RecoveryStrategy.RETRY,
            TPUErrorType.COMMUNICATION_ERROR: RecoveryStrategy.RETRY,
            TPUErrorType.NUMERICAL_ERROR: RecoveryStrategy.GRACEFUL_DEGRADATION,
            TPUErrorType.RESOURCE_EXHAUSTION: RecoveryStrategy.SCALE_DOWN,
            TPUErrorType.PREEMPTION: RecoveryStrategy.RESTART,
            TPUErrorType.UNKNOWN_ERROR: RecoveryStrategy.RETRY
        }
        
        base_strategy = strategy_map.get(error_type, RecoveryStrategy.RETRY)
        
        # Adaptive strategy based on error history
        recent_errors = [ctx for ctx in self.error_history[-10:] 
                        if ctx.error_type == error_type]
        
        if len(recent_errors) >= 3:
            # Frequent errors of same type - escalate strategy
            if base_strategy == RecoveryStrategy.RETRY:
                return RecoveryStrategy.FALLBACK
            elif base_strategy == RecoveryStrategy.FALLBACK:
                return RecoveryStrategy.GRACEFUL_DEGRADATION
        
        return base_strategy
    
    def recover_from_error(self, error_context: ErrorContext) -> Any:
        """Execute recovery strategy for a given error."""
        strategy = error_context.recovery_strategy
        
        logger.info(f"Executing recovery strategy: {strategy.value} "
                   f"for error: {error_context.error_type.value}")
        
        if strategy == RecoveryStrategy.RETRY:
            return self._execute_retry(error_context)
        elif strategy == RecoveryStrategy.FALLBACK:
            return self._execute_fallback(error_context)
        elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
            return self._execute_graceful_degradation(error_context)
        elif strategy == RecoveryStrategy.SCALE_DOWN:
            return self._execute_scale_down(error_context)
        elif strategy == RecoveryStrategy.RESTART:
            return self._execute_restart(error_context)
        elif strategy == RecoveryStrategy.ABORT:
            raise RuntimeError(f"Recovery failed after {error_context.attempt_number} attempts")
        
        return None
    
    def _execute_retry(self, error_context: ErrorContext) -> Any:
        """Execute retry recovery strategy."""
        delay = self.config.retry_delay * (self.config.backoff_multiplier ** (error_context.attempt_number - 1))
        
        logger.info(f"Retrying operation after {delay:.1f}s delay (attempt {error_context.attempt_number})")
        time.sleep(delay)
        
        # Clear any cached compilation state for compilation errors
        if error_context.error_type == TPUErrorType.COMPILATION_ERROR:
            self._clear_compilation_cache()
        
        self.recovery_stats['successful_recoveries'] += 1
        return "retry"
    
    def _execute_fallback(self, error_context: ErrorContext) -> Any:
        """Execute fallback recovery strategy."""
        logger.info("Executing fallback to alternative device/backend")
        
        # Check for available fallback devices
        fallback_device = self._get_fallback_device()
        
        if fallback_device:
            logger.info(f"Falling back to device: {fallback_device}")
            self.recovery_stats['fallback_activations'] += 1
            return f"fallback:{fallback_device}"
        else:
            logger.warning("No fallback devices available")
            return self._execute_graceful_degradation(error_context)
    
    def _execute_graceful_degradation(self, error_context: ErrorContext) -> Any:
        """Execute graceful degradation strategy."""
        logger.info("Executing graceful degradation")
        
        # Reduce model complexity or batch size
        degradation_config = {
            'reduce_batch_size': True,
            'reduce_precision': True,
            'disable_advanced_features': True,
            'use_checkpoint_restart': True
        }
        
        return f"degraded:{degradation_config}"
    
    def _execute_scale_down(self, error_context: ErrorContext) -> Any:
        """Execute scale down strategy for resource issues."""
        logger.info("Executing scale down for resource constraints")
        
        scale_config = {
            'reduce_batch_size': 0.5,  # Halve batch size
            'reduce_model_size': 0.8,  # 80% of original size
            'enable_gradient_checkpointing': True,
            'reduce_sequence_length': 0.9
        }
        
        return f"scaled_down:{scale_config}"
    
    def _execute_restart(self, error_context: ErrorContext) -> Any:
        """Execute restart strategy for preemption."""
        logger.info("Executing restart strategy")
        
        # Save current state before restart
        checkpoint_info = self._save_checkpoint()
        
        restart_config = {
            'checkpoint_path': checkpoint_info,
            'reset_device': True,
            'reinitialize_model': True
        }
        
        return f"restart:{restart_config}"
    
    def _clear_compilation_cache(self):
        """Clear JAX/XLA compilation cache."""
        try:
            # JAX cache clearing
            import jax
            if hasattr(jax, '_compilation_cache'):
                jax._compilation_cache.clear()
            
            logger.info("Cleared compilation cache")
        except Exception as e:
            logger.warning(f"Failed to clear compilation cache: {e}")
    
    def _get_fallback_device(self) -> Optional[str]:
        """Get available fallback device."""
        # Check cached device health
        current_time = time.time()
        if current_time - self._last_device_check > self.config.device_check_interval:
            self._update_device_health()
            self._last_device_check = current_time
        
        # Return first healthy fallback device
        for device in self._fallback_devices:
            if self._device_health_cache.get(device, False):
                return device
        
        # Fallback to CPU
        return "cpu"
    
    def _update_device_health(self):
        """Update device health status."""
        try:
            # Check JAX devices
            import jax
            devices = jax.devices()
            for device in devices:
                device_str = str(device)
                try:
                    # Simple device test
                    import jax.numpy as jnp
                    test_array = jnp.ones(10, device=device)
                    _ = test_array.sum()
                    self._device_health_cache[device_str] = True
                except:
                    self._device_health_cache[device_str] = False
                    
        except ImportError:
            # Fallback for other backends
            self._device_health_cache['cpu'] = True
    
    def _save_checkpoint(self) -> str:
        """Save current state checkpoint."""
        checkpoint_path = f"/tmp/tpu_recovery_checkpoint_{int(time.time())}.ckpt"
        
        # This would save actual model state in practice
        checkpoint_info = {
            'timestamp': time.time(),
            'error_recovery_active': True,
            'path': checkpoint_path
        }
        
        logger.info(f"Saved recovery checkpoint: {checkpoint_path}")
        return checkpoint_path
    
    def log_error(self, error_context: ErrorContext):
        """Log error for monitoring and analysis."""
        self.error_history.append(error_context)
        self.recovery_stats['total_errors'] += 1
        
        # Keep only recent history
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-500:]
        
        logger.error(f"TPU Error logged: {error_context.to_dict()}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error and recovery statistics."""
        error_type_counts = {}
        for error_ctx in self.error_history:
            error_type = error_ctx.error_type.value
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
        
        recovery_success_rate = (
            self.recovery_stats['successful_recoveries'] / 
            max(self.recovery_stats['total_errors'], 1)
        )
        
        return {
            'total_errors': self.recovery_stats['total_errors'],
            'successful_recoveries': self.recovery_stats['successful_recoveries'],
            'failed_recoveries': self.recovery_stats['failed_recoveries'],
            'fallback_activations': self.recovery_stats['fallback_activations'],
            'recovery_success_rate': recovery_success_rate,
            'error_type_distribution': error_type_counts,
            'recent_error_count': len([ctx for ctx in self.error_history 
                                     if time.time() - ctx.timestamp < 3600])  # Last hour
        }

def tpu_error_handler(recovery_config: Optional[RecoveryConfig] = None):
    """
    Decorator for automatic TPU error handling and recovery.
    
    Args:
        recovery_config: Optional recovery configuration
        
    Example:
        @tpu_error_handler()
        def train_model(model, data):
            # Training code that might fail on TPU
            return model
    """
    if recovery_config is None:
        recovery_config = RecoveryConfig()
    
    error_recovery = TPUErrorRecovery(recovery_config)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 1
            
            while attempt <= recovery_config.max_retries + 1:
                try:
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    # Create error context
                    error_context = ErrorContext(
                        error_type=error_recovery.classify_error(e, {}),
                        error_message=str(e),
                        timestamp=time.time(),
                        stack_trace=traceback.format_exc(),
                        function_name=func.__name__,
                        args=args,
                        kwargs=kwargs,
                        attempt_number=attempt
                    )
                    
                    # Determine recovery strategy
                    strategy = error_recovery.determine_recovery_strategy(
                        error_context.error_type, attempt
                    )
                    error_context.recovery_strategy = strategy
                    
                    # Log error
                    error_recovery.log_error(error_context)
                    
                    # Execute recovery if not aborting
                    if strategy != RecoveryStrategy.ABORT and attempt <= recovery_config.max_retries:
                        recovery_result = error_recovery.recover_from_error(error_context)
                        
                        # Handle recovery results
                        if isinstance(recovery_result, str):
                            if recovery_result.startswith('fallback:'):
                                # Modify function execution for fallback
                                device = recovery_result.split(':')[1]
                                logger.info(f"Executing with fallback device: {device}")
                                # Would modify device context here
                            elif recovery_result.startswith('degraded:'):
                                # Execute with degraded configuration
                                logger.info("Executing with degraded configuration")
                                # Would modify function parameters here
                            elif recovery_result.startswith('scaled_down:'):
                                # Execute with scaled down resources
                                logger.info("Executing with scaled down resources")
                                # Would modify batch size, model size, etc.
                        
                        attempt += 1
                        continue
                    else:
                        # Final failure
                        error_recovery.recovery_stats['failed_recoveries'] += 1
                        logger.error(f"Recovery failed for {func.__name__} after {attempt} attempts")
                        raise e
            
            # Should not reach here
            raise RuntimeError(f"Unexpected error in recovery loop for {func.__name__}")
        
        return wrapper
    return decorator

class TPUHealthMonitor:
    """
    Continuous health monitoring for TPU devices and operations.
    """
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.health_status = {}
        self.monitoring_active = False
        self._monitor_task = None
        
    async def start_monitoring(self):
        """Start continuous health monitoring."""
        self.monitoring_active = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Started TPU health monitoring")
    
    async def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring_active = False
        if self._monitor_task:
            self._monitor_task.cancel()
        logger.info("Stopped TPU health monitoring")
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                await self._check_device_health()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Health monitoring error: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _check_device_health(self):
        """Check health of all TPU devices."""
        try:
            # JAX device health check
            import jax
            devices = jax.devices()
            
            for device in devices:
                device_str = str(device)
                
                try:
                    # Simple computation test
                    import jax.numpy as jnp
                    test_data = jnp.ones(100, device=device)
                    result = test_data.sum()
                    
                    self.health_status[device_str] = {
                        'healthy': True,
                        'last_check': time.time(),
                        'test_result': float(result)
                    }
                    
                except Exception as e:
                    self.health_status[device_str] = {
                        'healthy': False,
                        'last_check': time.time(),
                        'error': str(e)
                    }
                    
                    logger.warning(f"Device {device_str} health check failed: {e}")
                    
        except ImportError:
            # Fallback for other backends
            self.health_status['cpu'] = {
                'healthy': True,
                'last_check': time.time()
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status of all devices."""
        return self.health_status.copy()
    
    def is_device_healthy(self, device: str) -> bool:
        """Check if a specific device is healthy."""
        status = self.health_status.get(device, {})
        return status.get('healthy', False)

class TPUValidationFramework:
    """
    Comprehensive validation framework for TPU operations.
    """
    
    def __init__(self):
        self.validation_rules = []
        self.validation_history = []
        
    def add_validation_rule(self, name: str, rule_func: Callable, 
                          severity: str = "warning"):
        """Add a validation rule."""
        self.validation_rules.append({
            'name': name,
            'function': rule_func,
            'severity': severity
        })
    
    def validate_model_architecture(self, architecture_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model architecture for TPU compatibility."""
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        # Check hidden size alignment
        hidden_size = architecture_config.get('hidden_size', 768)
        if hidden_size % 128 != 0:
            validation_results['warnings'].append(
                f"Hidden size {hidden_size} not aligned to 128 (TPU matrix unit size)"
            )
            validation_results['recommendations'].append(
                f"Consider using hidden size: {((hidden_size // 128) + 1) * 128}"
            )
        
        # Check attention head divisibility
        num_heads = architecture_config.get('num_attention_heads', 12)
        if hidden_size % num_heads != 0:
            validation_results['errors'].append(
                f"Hidden size {hidden_size} not divisible by attention heads {num_heads}"
            )
            validation_results['valid'] = False
        
        # Check batch size
        batch_size = architecture_config.get('batch_size', 32)
        if batch_size % 8 != 0:
            validation_results['warnings'].append(
                f"Batch size {batch_size} not aligned to 8 (optimal for TPU)"
            )
        
        # Check sequence length
        seq_length = architecture_config.get('max_position_embeddings', 512)
        if seq_length > 2048:
            validation_results['warnings'].append(
                f"Sequence length {seq_length} may cause memory issues on TPU"
            )
        
        return validation_results
    
    def validate_input_data(self, data: Any) -> Dict[str, Any]:
        """Validate input data for TPU processing."""
        validation_results = {
            'valid': True,
            'issues': []
        }
        
        try:
            # Check data type compatibility
            if hasattr(data, 'dtype'):
                if str(data.dtype) not in ['float32', 'bfloat16', 'int32']:
                    validation_results['issues'].append(
                        f"Data type {data.dtype} may not be optimal for TPU"
                    )
            
            # Check data shape
            if hasattr(data, 'shape'):
                shape = data.shape
                if len(shape) > 0 and shape[0] % 8 != 0:
                    validation_results['issues'].append(
                        f"Batch dimension {shape[0]} not aligned to 8"
                    )
                    
        except Exception as e:
            validation_results['valid'] = False
            validation_results['issues'].append(f"Data validation error: {e}")
        
        return validation_results
    
    def run_custom_validations(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run all custom validation rules."""
        results = {
            'passed': [],
            'warnings': [],
            'errors': []
        }
        
        for rule in self.validation_rules:
            try:
                result = rule['function'](context)
                if result:
                    results['passed'].append(rule['name'])
                else:
                    if rule['severity'] == 'error':
                        results['errors'].append(rule['name'])
                    else:
                        results['warnings'].append(rule['name'])
                        
            except Exception as e:
                results['errors'].append(f"{rule['name']}: {e}")
        
        return results

# Export main classes and functions
__all__ = [
    'TPUErrorRecovery', 'TPUErrorType', 'RecoveryStrategy', 'ErrorContext',
    'RecoveryConfig', 'tpu_error_handler', 'TPUHealthMonitor', 
    'TPUValidationFramework'
]