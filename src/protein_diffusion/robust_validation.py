"""
Robust Validation System for Protein Diffusion

This module implements comprehensive validation including:
- Multi-layer input validation with sanitization
- Real-time health monitoring and circuit breakers
- Comprehensive error handling with recovery strategies
- Data integrity checks and corruption detection
- Performance monitoring and alerting
"""

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    from . import mock_torch as torch
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    class MockNumpy:
        @staticmethod
        def isnan(x): return False
        @staticmethod
        def isinf(x): return False
        @staticmethod
        def mean(x): return 0.5
        @staticmethod
        def std(x): return 1.0
    np = MockNumpy()
    NUMPY_AVAILABLE = False

from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import re
import hashlib
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels."""
    STRICT = "strict"
    NORMAL = "normal"
    PERMISSIVE = "permissive"
    EMERGENCY = "emergency"


class ErrorSeverity(Enum):
    """Error severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class ValidationResult:
    """Result of validation check."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    severity: ErrorSeverity = ErrorSeverity.INFO
    validation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthMetrics:
    """System health metrics."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    gpu_memory_usage: Optional[float]
    active_requests: int
    error_rate: float
    avg_response_time: float
    model_accuracy: float
    is_healthy: bool
    alerts: List[str] = field(default_factory=list)


class BaseValidator(ABC):
    """Abstract base validator."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.NORMAL):
        self.validation_level = validation_level
        self.validation_count = 0
        self.error_count = 0
        
    @abstractmethod
    def validate(self, data: Any) -> ValidationResult:
        """Validate input data."""
        pass
    
    def _create_result(
        self,
        is_valid: bool,
        errors: List[str] = None,
        warnings: List[str] = None,
        severity: ErrorSeverity = ErrorSeverity.INFO,
        metadata: Dict[str, Any] = None
    ) -> ValidationResult:
        """Create validation result."""
        return ValidationResult(
            is_valid=is_valid,
            errors=errors or [],
            warnings=warnings or [],
            severity=severity,
            validation_time=time.time(),
            metadata=metadata or {}
        )


class SequenceValidator(BaseValidator):
    """Validates protein sequences."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.NORMAL):
        super().__init__(validation_level)
        self.valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
        self.suspicious_patterns = [
            r"(.)\1{5,}",  # 6+ repeated characters
            r"[^ACDEFGHIKLMNPQRSTVWY]",  # Invalid characters
            r"^.{0,4}$",  # Too short
            r"^.{1000,}$"  # Too long
        ]
    
    def validate(self, sequence: str) -> ValidationResult:
        """Validate protein sequence."""
        start_time = time.time()
        errors = []
        warnings = []
        
        if not isinstance(sequence, str):
            errors.append(f"Sequence must be string, got {type(sequence)}")
            return self._create_result(False, errors, severity=ErrorSeverity.CRITICAL)
        
        # Basic length checks
        if len(sequence) < 5:
            errors.append(f"Sequence too short: {len(sequence)} < 5")
        elif len(sequence) > 2000:
            if self.validation_level == ValidationLevel.STRICT:
                errors.append(f"Sequence too long: {len(sequence)} > 2000")
            else:
                warnings.append(f"Very long sequence: {len(sequence)}")
        
        # Character validation
        invalid_chars = set(sequence.upper()) - self.valid_amino_acids
        if invalid_chars:
            errors.append(f"Invalid amino acids: {invalid_chars}")
        
        # Pattern detection
        for pattern in self.suspicious_patterns:
            if re.search(pattern, sequence.upper()):
                if "repeated" in pattern:
                    warnings.append("Detected repeated amino acid pattern")
                elif "short" in pattern:
                    errors.append("Sequence too short")
                elif "long" in pattern:
                    errors.append("Sequence too long")
        
        # Composition analysis
        composition = self._analyze_composition(sequence)
        composition_warnings = self._check_composition(composition)
        warnings.extend(composition_warnings)
        
        # Determine severity
        severity = ErrorSeverity.INFO
        if errors:
            severity = ErrorSeverity.HIGH if len(errors) > 2 else ErrorSeverity.MEDIUM
        
        validation_time = time.time() - start_time
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            severity=severity,
            validation_time=validation_time,
            metadata={"composition": composition}
        )
    
    def _analyze_composition(self, sequence: str) -> Dict[str, float]:
        """Analyze amino acid composition."""
        composition = {}
        total = len(sequence)
        
        for aa in self.valid_amino_acids:
            count = sequence.upper().count(aa)
            composition[aa] = count / total if total > 0 else 0.0
        
        return composition
    
    def _check_composition(self, composition: Dict[str, float]) -> List[str]:
        """Check for unusual composition patterns."""
        warnings = []
        
        # Check for amino acids that are too frequent
        for aa, freq in composition.items():
            if freq > 0.3:
                warnings.append(f"High {aa} content: {freq:.1%}")
            elif freq > 0.2 and aa in "GP":  # Glycine and Proline
                warnings.append(f"High {aa} content may affect structure: {freq:.1%}")
        
        # Check hydrophobic/hydrophilic balance
        hydrophobic = sum(composition[aa] for aa in "AILMFPWV")
        hydrophilic = sum(composition[aa] for aa in "DEKRNQSTHY")
        
        if hydrophobic > 0.7:
            warnings.append(f"Very hydrophobic sequence: {hydrophobic:.1%}")
        elif hydrophilic > 0.7:
            warnings.append(f"Very hydrophilic sequence: {hydrophilic:.1%}")
        
        return warnings


class ModelInputValidator(BaseValidator):
    """Validates model inputs and parameters."""
    
    def validate(self, params: Dict[str, Any]) -> ValidationResult:
        """Validate model input parameters."""
        start_time = time.time()
        errors = []
        warnings = []
        
        # Validate individual parameters
        if "num_samples" in params:
            num_samples = params["num_samples"]
            if not isinstance(num_samples, int) or num_samples <= 0:
                errors.append(f"num_samples must be positive integer, got {num_samples}")
            elif num_samples > 1000:
                if self.validation_level == ValidationLevel.STRICT:
                    errors.append(f"num_samples too large: {num_samples} > 1000")
                else:
                    warnings.append(f"Large number of samples: {num_samples}")
        
        if "temperature" in params:
            temp = params["temperature"]
            if not isinstance(temp, (int, float)) or temp <= 0:
                errors.append(f"temperature must be positive number, got {temp}")
            elif temp > 5.0:
                warnings.append(f"Very high temperature: {temp}")
            elif temp < 0.1:
                warnings.append(f"Very low temperature: {temp}")
        
        if "guidance_scale" in params:
            guidance = params["guidance_scale"]
            if not isinstance(guidance, (int, float)) or guidance < 0:
                errors.append(f"guidance_scale must be non-negative, got {guidance}")
            elif guidance > 10.0:
                warnings.append(f"Very high guidance scale: {guidance}")
        
        if "max_length" in params:
            max_len = params["max_length"]
            if not isinstance(max_len, int) or max_len <= 0:
                errors.append(f"max_length must be positive integer, got {max_len}")
            elif max_len > 5000:
                warnings.append(f"Very long sequence requested: {max_len}")
        
        # Cross-parameter validation
        if "num_samples" in params and "max_length" in params:
            total_tokens = params["num_samples"] * params["max_length"]
            if total_tokens > 1000000:  # 1M tokens
                warnings.append(f"Large computation requested: {total_tokens} tokens")
        
        validation_time = time.time() - start_time
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            severity=ErrorSeverity.HIGH if errors else ErrorSeverity.INFO,
            validation_time=validation_time
        )


class DataIntegrityValidator(BaseValidator):
    """Validates data integrity and detects corruption."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.NORMAL):
        super().__init__(validation_level)
        self.checksums = {}
    
    def validate(self, data: Union[torch.Tensor, np.ndarray, Dict]) -> ValidationResult:
        """Validate data integrity."""
        start_time = time.time()
        errors = []
        warnings = []
        metadata = {}
        
        # Type-specific validation
        if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            tensor_result = self._validate_tensor(data)
            errors.extend(tensor_result.errors)
            warnings.extend(tensor_result.warnings)
            metadata.update(tensor_result.metadata)
        
        elif NUMPY_AVAILABLE and isinstance(data, np.ndarray):
            array_result = self._validate_array(data)
            errors.extend(array_result.errors)
            warnings.extend(array_result.warnings)
            metadata.update(array_result.metadata)
        
        elif isinstance(data, dict):
            dict_result = self._validate_dict(data)
            errors.extend(dict_result.errors)
            warnings.extend(dict_result.warnings)
            metadata.update(dict_result.metadata)
        
        # Checksum validation
        data_hash = self._compute_hash(data)
        metadata["hash"] = data_hash
        
        validation_time = time.time() - start_time
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            severity=ErrorSeverity.CRITICAL if "corruption" in str(errors) else ErrorSeverity.INFO,
            validation_time=validation_time,
            metadata=metadata
        )
    
    def _validate_tensor(self, tensor: torch.Tensor) -> ValidationResult:
        """Validate PyTorch tensor."""
        errors = []
        warnings = []
        metadata = {}
        
        # Check for NaN/Inf
        if TORCH_AVAILABLE:
            if torch.isnan(tensor).any():
                errors.append("Tensor contains NaN values")
            if torch.isinf(tensor).any():
                errors.append("Tensor contains infinite values")
            
            # Shape validation
            if tensor.numel() == 0:
                errors.append("Empty tensor")
            elif tensor.numel() > 100000000:  # 100M elements
                warnings.append(f"Very large tensor: {tensor.numel()} elements")
            
            # Value range checks
            if tensor.dtype.is_floating_point:
                tensor_min, tensor_max = tensor.min().item(), tensor.max().item()
                if abs(tensor_min) > 1e6 or abs(tensor_max) > 1e6:
                    warnings.append(f"Large values detected: [{tensor_min:.2e}, {tensor_max:.2e}]")
                
                metadata.update({
                    "dtype": str(tensor.dtype),
                    "shape": list(tensor.shape),
                    "min": tensor_min,
                    "max": tensor_max,
                    "mean": tensor.mean().item(),
                    "std": tensor.std().item()
                })
        
        return self._create_result(len(errors) == 0, errors, warnings, metadata=metadata)
    
    def _validate_array(self, array: np.ndarray) -> ValidationResult:
        """Validate numpy array."""
        errors = []
        warnings = []
        metadata = {}
        
        if NUMPY_AVAILABLE:
            # Check for NaN/Inf
            if np.isnan(array).any():
                errors.append("Array contains NaN values")
            if np.isinf(array).any():
                errors.append("Array contains infinite values")
            
            # Size validation
            if array.size == 0:
                errors.append("Empty array")
            elif array.size > 100000000:
                warnings.append(f"Very large array: {array.size} elements")
            
            metadata.update({
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "size": array.size
            })
        
        return self._create_result(len(errors) == 0, errors, warnings, metadata=metadata)
    
    def _validate_dict(self, data: Dict) -> ValidationResult:
        """Validate dictionary data."""
        errors = []
        warnings = []
        metadata = {}
        
        # Check for required keys in common data structures
        if "sequence" in data:
            if not isinstance(data["sequence"], str):
                errors.append("'sequence' field must be string")
        
        if "confidence" in data:
            conf = data["confidence"]
            if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
                errors.append("'confidence' must be number between 0 and 1")
        
        # Check for circular references
        try:
            import json
            json.dumps(data, default=str)
        except (ValueError, TypeError) as e:
            if "circular reference" in str(e).lower():
                errors.append("Dictionary contains circular references")
        
        metadata["keys"] = list(data.keys())
        metadata["size"] = len(data)
        
        return self._create_result(len(errors) == 0, errors, warnings, metadata=metadata)
    
    def _compute_hash(self, data: Any) -> str:
        """Compute hash for data integrity checking."""
        try:
            if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
                data_bytes = data.detach().cpu().numpy().tobytes()
            elif NUMPY_AVAILABLE and isinstance(data, np.ndarray):
                data_bytes = data.tobytes()
            else:
                data_bytes = str(data).encode('utf-8')
            
            return hashlib.md5(data_bytes).hexdigest()
        except Exception:
            return "hash_error"


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        expected_exception: Exception = Exception
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def call(self, func: Callable, *args, **kwargs):
        """Call function through circuit breaker."""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit."""
        return (
            self.last_failure_time is not None and
            time.time() - self.last_failure_time >= self.timeout
        )
    
    def _on_success(self):
        """Handle successful call."""
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
        self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class HealthMonitor:
    """System health monitoring."""
    
    def __init__(self):
        self.metrics_history = []
        self.alert_handlers = []
        self.thresholds = {
            "cpu_usage": 90.0,
            "memory_usage": 85.0,
            "gpu_memory_usage": 90.0,
            "error_rate": 0.1,
            "avg_response_time": 10.0
        }
    
    def add_alert_handler(self, handler: Callable[[HealthMetrics], None]):
        """Add alert handler."""
        self.alert_handlers.append(handler)
    
    def collect_metrics(self) -> HealthMetrics:
        """Collect current health metrics."""
        import psutil
        
        # System metrics
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        
        # GPU metrics (if available)
        gpu_memory_usage = None
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_stats()
            allocated = gpu_memory.get("allocated_bytes.all.current", 0)
            reserved = gpu_memory.get("reserved_bytes.all.current", 0)
            if reserved > 0:
                gpu_memory_usage = (allocated / reserved) * 100
        
        # Application metrics (simplified)
        active_requests = 0  # Would track actual requests
        error_rate = 0.05  # Would calculate from error log
        avg_response_time = 2.5  # Would calculate from response times
        model_accuracy = 0.92  # Would calculate from model performance
        
        # Check health status
        alerts = []
        is_healthy = True
        
        if cpu_usage > self.thresholds["cpu_usage"]:
            alerts.append(f"High CPU usage: {cpu_usage}%")
            is_healthy = False
        
        if memory_usage > self.thresholds["memory_usage"]:
            alerts.append(f"High memory usage: {memory_usage}%")
            is_healthy = False
        
        if gpu_memory_usage and gpu_memory_usage > self.thresholds["gpu_memory_usage"]:
            alerts.append(f"High GPU memory usage: {gpu_memory_usage}%")
            is_healthy = False
        
        if error_rate > self.thresholds["error_rate"]:
            alerts.append(f"High error rate: {error_rate}")
            is_healthy = False
        
        if avg_response_time > self.thresholds["avg_response_time"]:
            alerts.append(f"High response time: {avg_response_time}s")
            is_healthy = False
        
        metrics = HealthMetrics(
            timestamp=time.time(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_memory_usage=gpu_memory_usage,
            active_requests=active_requests,
            error_rate=error_rate,
            avg_response_time=avg_response_time,
            model_accuracy=model_accuracy,
            is_healthy=is_healthy,
            alerts=alerts
        )
        
        # Store metrics
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > 1000:  # Keep last 1000 metrics
            self.metrics_history.pop(0)
        
        # Trigger alerts
        if alerts:
            for handler in self.alert_handlers:
                try:
                    handler(metrics)
                except Exception as e:
                    logger.error(f"Alert handler failed: {e}")
        
        return metrics
    
    def get_health_trend(self, hours: int = 1) -> Dict[str, float]:
        """Get health trend over specified hours."""
        cutoff_time = time.time() - (hours * 3600)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {}
        
        return {
            "avg_cpu_usage": sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics),
            "avg_memory_usage": sum(m.memory_usage for m in recent_metrics) / len(recent_metrics),
            "avg_error_rate": sum(m.error_rate for m in recent_metrics) / len(recent_metrics),
            "avg_response_time": sum(m.avg_response_time for m in recent_metrics) / len(recent_metrics),
            "uptime_percentage": sum(1 for m in recent_metrics if m.is_healthy) / len(recent_metrics) * 100
        }


class RobustValidationManager:
    """Central validation manager with comprehensive error handling."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.NORMAL):
        self.validation_level = validation_level
        self.validators = {
            "sequence": SequenceValidator(validation_level),
            "model_input": ModelInputValidator(validation_level),
            "data_integrity": DataIntegrityValidator(validation_level)
        }
        
        self.circuit_breaker = CircuitBreaker()
        self.health_monitor = HealthMonitor()
        
        # Statistics
        self.validation_stats = {
            "total_validations": 0,
            "failed_validations": 0,
            "avg_validation_time": 0.0
        }
    
    def comprehensive_validation(
        self,
        sequence: Optional[str] = None,
        generation_params: Optional[Dict[str, Any]] = None,
        model_data: Optional[Any] = None
    ) -> ValidationResult:
        """Perform comprehensive validation."""
        start_time = time.time()
        all_errors = []
        all_warnings = []
        all_metadata = {}
        
        try:
            # Sequence validation
            if sequence:
                seq_result = self.circuit_breaker.call(
                    self.validators["sequence"].validate, sequence
                )
                all_errors.extend(seq_result.errors)
                all_warnings.extend(seq_result.warnings)
                all_metadata["sequence_validation"] = seq_result.metadata
            
            # Parameter validation
            if generation_params:
                param_result = self.circuit_breaker.call(
                    self.validators["model_input"].validate, generation_params
                )
                all_errors.extend(param_result.errors)
                all_warnings.extend(param_result.warnings)
                all_metadata["parameter_validation"] = param_result.metadata
            
            # Data integrity validation
            if model_data:
                integrity_result = self.circuit_breaker.call(
                    self.validators["data_integrity"].validate, model_data
                )
                all_errors.extend(integrity_result.errors)
                all_warnings.extend(integrity_result.warnings)
                all_metadata["integrity_validation"] = integrity_result.metadata
            
            # Update statistics
            validation_time = time.time() - start_time
            self.validation_stats["total_validations"] += 1
            if all_errors:
                self.validation_stats["failed_validations"] += 1
            
            # Update average validation time
            total_time = (self.validation_stats["avg_validation_time"] * 
                         (self.validation_stats["total_validations"] - 1) + validation_time)
            self.validation_stats["avg_validation_time"] = total_time / self.validation_stats["total_validations"]
            
            return ValidationResult(
                is_valid=len(all_errors) == 0,
                errors=all_errors,
                warnings=all_warnings,
                severity=ErrorSeverity.HIGH if all_errors else ErrorSeverity.INFO,
                validation_time=validation_time,
                metadata=all_metadata
            )
            
        except Exception as e:
            logger.error(f"Validation system error: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation system failure: {e}"],
                severity=ErrorSeverity.CRITICAL,
                validation_time=time.time() - start_time
            )
    
    def get_health_status(self) -> HealthMetrics:
        """Get current health status."""
        return self.health_monitor.collect_metrics()
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        stats = self.validation_stats.copy()
        if stats["total_validations"] > 0:
            stats["failure_rate"] = stats["failed_validations"] / stats["total_validations"]
        else:
            stats["failure_rate"] = 0.0
        
        return stats
    
    def reset_circuit_breaker(self):
        """Reset circuit breaker manually."""
        self.circuit_breaker.state = "CLOSED"
        self.circuit_breaker.failure_count = 0
        logger.info("Circuit breaker reset")
    
    def update_validation_level(self, level: ValidationLevel):
        """Update validation level for all validators."""
        self.validation_level = level
        for validator in self.validators.values():
            validator.validation_level = level
        logger.info(f"Validation level updated to {level.value}")


# Alert handlers
def log_alert_handler(metrics: HealthMetrics):
    """Log alerts to standard logger."""
    for alert in metrics.alerts:
        logger.warning(f"Health Alert: {alert}")


def email_alert_handler(metrics: HealthMetrics):
    """Send email alerts (placeholder)."""
    if metrics.alerts:
        # In practice, would send actual email
        logger.info(f"Email alert would be sent: {len(metrics.alerts)} issues")


# Convenience functions
def validate_sequence(sequence: str, level: ValidationLevel = ValidationLevel.NORMAL) -> ValidationResult:
    """Quick sequence validation."""
    validator = SequenceValidator(level)
    return validator.validate(sequence)


def validate_generation_params(params: Dict[str, Any]) -> ValidationResult:
    """Quick parameter validation."""
    validator = ModelInputValidator()
    return validator.validate(params)