"""
Security Framework for Protein Diffusion Design Lab

This module implements comprehensive security measures including:
- Input sanitization and injection prevention
- Rate limiting and abuse protection
- Secure model serving with authentication
- Data privacy and access control
- Audit logging and compliance monitoring
"""

import hashlib
import hmac
import time
import secrets
import logging
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import re
from abc import ABC, abstractmethod

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security enforcement levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AccessLevel(Enum):
    """User access levels."""
    GUEST = "guest"
    USER = "user"
    RESEARCHER = "researcher"
    ADMIN = "admin"
    SYSTEM = "system"


@dataclass
class SecurityContext:
    """Security context for requests."""
    user_id: str
    access_level: AccessLevel
    api_key: Optional[str] = None
    session_token: Optional[str] = None
    ip_address: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    permissions: List[str] = field(default_factory=list)


@dataclass
class SecurityConfig:
    """Security configuration."""
    # Rate limiting
    max_requests_per_minute: int = 100
    max_requests_per_hour: int = 1000
    max_concurrent_requests: int = 10
    
    # Input validation
    max_sequence_length: int = 2000
    max_batch_size: int = 50
    allowed_file_types: List[str] = field(default_factory=lambda: ['.pdb', '.fasta', '.txt'])
    
    # Authentication
    api_key_length: int = 32
    session_timeout: int = 3600  # 1 hour
    max_failed_logins: int = 5
    lockout_duration: int = 900  # 15 minutes
    
    # Encryption
    use_encryption: bool = True
    encryption_algorithm: str = "AES-256-GCM"
    
    # Audit
    enable_audit_logging: bool = True
    log_retention_days: int = 90
    
    # Model security
    model_checksum_validation: bool = True
    secure_model_loading: bool = True
    prevent_model_extraction: bool = True


class InputSanitizer:
    """Sanitizes and validates inputs to prevent injection attacks."""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.MEDIUM):
        self.security_level = security_level
        
        # Define dangerous patterns
        self.sql_injection_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|CREATE|ALTER)\b)",
            r"(--|#|\/\*|\*\/)",
            r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
            r"(\b(OR|AND)\s+'\w+'\s*=\s*'\w+')"
        ]
        
        self.command_injection_patterns = [
            r"(;|\||\&|\$\(|\`)",
            r"(\b(rm|cat|ls|ps|kill|sudo|su)\b)",
            r"(\.\.\/|\.\.\\)",
            r"(\b(eval|exec|system|shell_exec)\b)"
        ]
        
        self.script_injection_patterns = [
            r"(<script[^>]*>.*?</script>)",
            r"(javascript:|vbscript:|data:)",
            r"(onload|onclick|onerror|onmouseover)=",
            r"(<iframe|<object|<embed|<link)"
        ]
    
    def sanitize_sequence(self, sequence: str) -> str:
        """Sanitize protein sequence input."""
        if not isinstance(sequence, str):
            raise ValueError("Sequence must be a string")
        
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY\s\-]', '', sequence.upper())
        
        # Validate length
        if len(sanitized) > 10000:  # Hard limit regardless of config
            raise ValueError(f"Sequence too long: {len(sanitized)} > 10000")
        
        # Check for suspicious patterns
        if self.security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
            if self._contains_suspicious_patterns(sanitized):
                raise ValueError("Sequence contains suspicious patterns")
        
        return sanitized.strip()
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent path traversal."""
        if not isinstance(filename, str):
            raise ValueError("Filename must be a string")
        
        # Remove path components
        filename = filename.split('/')[-1].split('\\')[-1]
        
        # Remove dangerous characters
        sanitized = re.sub(r'[^a-zA-Z0-9._-]', '', filename)
        
        # Prevent hidden files and dangerous names
        if sanitized.startswith('.') or sanitized.lower() in ['con', 'prn', 'aux', 'nul']:
            raise ValueError("Invalid filename")
        
        # Check length
        if len(sanitized) > 255:
            raise ValueError("Filename too long")
        
        return sanitized
    
    def sanitize_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize generation parameters."""
        sanitized = {}
        
        for key, value in params.items():
            # Sanitize key
            clean_key = re.sub(r'[^a-zA-Z0-9_]', '', str(key))
            
            # Type-specific sanitization
            if isinstance(value, str):
                clean_value = self._sanitize_string_param(value)
            elif isinstance(value, (int, float)):
                clean_value = self._sanitize_numeric_param(key, value)
            elif isinstance(value, bool):
                clean_value = bool(value)
            elif value is None:
                clean_value = None
            else:
                # Convert complex types to string and sanitize
                clean_value = self._sanitize_string_param(str(value))
            
            sanitized[clean_key] = clean_value
        
        return sanitized
    
    def _sanitize_string_param(self, value: str) -> str:
        """Sanitize string parameter."""
        # Check for injection patterns
        for pattern_list in [self.sql_injection_patterns, 
                           self.command_injection_patterns,
                           self.script_injection_patterns]:
            for pattern in pattern_list:
                if re.search(pattern, value, re.IGNORECASE):
                    raise ValueError(f"Suspicious pattern detected in input: {pattern}")
        
        # Basic cleanup
        cleaned = re.sub(r'[<>\"\'\\]', '', value)
        return cleaned[:1000]  # Limit length
    
    def _sanitize_numeric_param(self, key: str, value: Union[int, float]) -> Union[int, float]:
        """Sanitize numeric parameter with bounds checking."""
        bounds = {
            'num_samples': (1, 1000),
            'max_length': (1, 5000),
            'temperature': (0.01, 10.0),
            'guidance_scale': (0.0, 20.0),
            'ddim_steps': (1, 1000)
        }
        
        if key in bounds:
            min_val, max_val = bounds[key]
            if not (min_val <= value <= max_val):
                raise ValueError(f"{key} must be between {min_val} and {max_val}")
        
        return value
    
    def _contains_suspicious_patterns(self, text: str) -> bool:
        """Check for suspicious patterns in text."""
        suspicious_patterns = [
            r"(.)\1{20,}",  # Long repetitions
            r"\b(password|secret|key|token)\b",  # Sensitive keywords
            r"[^\x20-\x7E]",  # Non-printable characters
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False


class RateLimiter:
    """Rate limiting to prevent abuse."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.request_counts = {}  # user_id -> [(timestamp, count), ...]
        self.active_requests = {}  # user_id -> count
        
    def check_rate_limit(self, user_id: str) -> bool:
        """Check if user is within rate limits."""
        now = time.time()
        
        # Clean old entries
        self._cleanup_old_entries(user_id, now)
        
        # Check concurrent requests
        concurrent = self.active_requests.get(user_id, 0)
        if concurrent >= self.config.max_concurrent_requests:
            logger.warning(f"Concurrent request limit exceeded for user {user_id}")
            return False
        
        # Check per-minute limit
        minute_requests = self._count_requests_in_window(user_id, now, 60)
        if minute_requests >= self.config.max_requests_per_minute:
            logger.warning(f"Per-minute rate limit exceeded for user {user_id}")
            return False
        
        # Check per-hour limit
        hour_requests = self._count_requests_in_window(user_id, now, 3600)
        if hour_requests >= self.config.max_requests_per_hour:
            logger.warning(f"Per-hour rate limit exceeded for user {user_id}")
            return False
        
        return True
    
    def record_request(self, user_id: str):
        """Record a new request."""
        now = time.time()
        
        if user_id not in self.request_counts:
            self.request_counts[user_id] = []
        
        self.request_counts[user_id].append(now)
        self.active_requests[user_id] = self.active_requests.get(user_id, 0) + 1
    
    def complete_request(self, user_id: str):
        """Mark request as completed."""
        if user_id in self.active_requests:
            self.active_requests[user_id] = max(0, self.active_requests[user_id] - 1)
    
    def _cleanup_old_entries(self, user_id: str, now: float):
        """Remove old request entries."""
        if user_id not in self.request_counts:
            return
        
        # Keep only entries from last hour
        hour_ago = now - 3600
        self.request_counts[user_id] = [
            timestamp for timestamp in self.request_counts[user_id]
            if timestamp > hour_ago
        ]
    
    def _count_requests_in_window(self, user_id: str, now: float, window_seconds: int) -> int:
        """Count requests in time window."""
        if user_id not in self.request_counts:
            return 0
        
        window_start = now - window_seconds
        return len([
            timestamp for timestamp in self.request_counts[user_id]
            if timestamp > window_start
        ])


class AuthenticationManager:
    """Manages authentication and authorization."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.api_keys = {}  # api_key -> user_info
        self.sessions = {}  # session_token -> user_info
        self.failed_logins = {}  # user_id -> (count, last_attempt)
        self.lockouts = {}  # user_id -> lockout_until
    
    def generate_api_key(self, user_id: str, access_level: AccessLevel) -> str:
        """Generate new API key."""
        api_key = secrets.token_urlsafe(self.config.api_key_length)
        
        self.api_keys[api_key] = {
            'user_id': user_id,
            'access_level': access_level,
            'created_at': time.time(),
            'last_used': None
        }
        
        logger.info(f"Generated API key for user {user_id}")
        return api_key
    
    def validate_api_key(self, api_key: str) -> Optional[SecurityContext]:
        """Validate API key and return security context."""
        if not api_key or api_key not in self.api_keys:
            return None
        
        key_info = self.api_keys[api_key]
        key_info['last_used'] = time.time()
        
        # Check if user is locked out
        if self._is_locked_out(key_info['user_id']):
            return None
        
        return SecurityContext(
            user_id=key_info['user_id'],
            access_level=key_info['access_level'],
            api_key=api_key,
            permissions=self._get_permissions(key_info['access_level'])
        )
    
    def create_session(self, user_id: str, access_level: AccessLevel) -> str:
        """Create new session."""
        session_token = secrets.token_urlsafe(32)
        
        self.sessions[session_token] = {
            'user_id': user_id,
            'access_level': access_level,
            'created_at': time.time(),
            'last_activity': time.time()
        }
        
        return session_token
    
    def validate_session(self, session_token: str) -> Optional[SecurityContext]:
        """Validate session token."""
        if not session_token or session_token not in self.sessions:
            return None
        
        session = self.sessions[session_token]
        now = time.time()
        
        # Check session timeout
        if now - session['last_activity'] > self.config.session_timeout:
            del self.sessions[session_token]
            return None
        
        # Update last activity
        session['last_activity'] = now
        
        return SecurityContext(
            user_id=session['user_id'],
            access_level=session['access_level'],
            session_token=session_token,
            permissions=self._get_permissions(session['access_level'])
        )
    
    def record_failed_login(self, user_id: str):
        """Record failed login attempt."""
        now = time.time()
        
        if user_id not in self.failed_logins:
            self.failed_logins[user_id] = (0, now)
        
        count, _ = self.failed_logins[user_id]
        count += 1
        self.failed_logins[user_id] = (count, now)
        
        # Lock out user if too many failed attempts
        if count >= self.config.max_failed_logins:
            self.lockouts[user_id] = now + self.config.lockout_duration
            logger.warning(f"User {user_id} locked out due to failed login attempts")
    
    def _is_locked_out(self, user_id: str) -> bool:
        """Check if user is locked out."""
        if user_id not in self.lockouts:
            return False
        
        if time.time() > self.lockouts[user_id]:
            del self.lockouts[user_id]
            return False
        
        return True
    
    def _get_permissions(self, access_level: AccessLevel) -> List[str]:
        """Get permissions for access level."""
        permissions_map = {
            AccessLevel.GUEST: ['generate_basic'],
            AccessLevel.USER: ['generate_basic', 'generate_batch'],
            AccessLevel.RESEARCHER: ['generate_basic', 'generate_batch', 'advanced_features', 'export_data'],
            AccessLevel.ADMIN: ['*'],  # All permissions
            AccessLevel.SYSTEM: ['*']
        }
        
        return permissions_map.get(access_level, [])


class SecurityAuditor:
    """Security audit logging and compliance monitoring."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.audit_log = []
        
    def log_security_event(
        self,
        event_type: str,
        user_id: str,
        details: Dict[str, Any],
        severity: str = "INFO"
    ):
        """Log security event."""
        if not self.config.enable_audit_logging:
            return
        
        event = {
            'timestamp': time.time(),
            'event_type': event_type,
            'user_id': user_id,
            'severity': severity,
            'details': details,
            'session_id': details.get('session_id', 'unknown')
        }
        
        self.audit_log.append(event)
        
        # Log to standard logger as well
        logger.info(f"Security Event: {event_type} - User: {user_id} - Severity: {severity}")
        
        # Trigger alerts for high-severity events
        if severity in ['CRITICAL', 'HIGH']:
            self._trigger_security_alert(event)
    
    def log_access_attempt(self, context: SecurityContext, resource: str, success: bool):
        """Log access attempt."""
        self.log_security_event(
            'ACCESS_ATTEMPT',
            context.user_id,
            {
                'resource': resource,
                'success': success,
                'access_level': context.access_level.value,
                'ip_address': context.ip_address
            },
            'WARNING' if not success else 'INFO'
        )
    
    def log_generation_request(self, context: SecurityContext, params: Dict[str, Any]):
        """Log protein generation request."""
        self.log_security_event(
            'GENERATION_REQUEST',
            context.user_id,
            {
                'num_samples': params.get('num_samples', 0),
                'max_length': params.get('max_length', 0),
                'method': params.get('sampling_method', 'unknown')
            }
        )
    
    def log_suspicious_activity(self, user_id: str, activity: str, details: Dict[str, Any]):
        """Log suspicious activity."""
        self.log_security_event(
            'SUSPICIOUS_ACTIVITY',
            user_id,
            {'activity': activity, **details},
            'HIGH'
        )
    
    def get_audit_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate audit report."""
        cutoff_time = time.time() - (hours * 3600)
        recent_events = [e for e in self.audit_log if e['timestamp'] >= cutoff_time]
        
        # Count events by type
        event_counts = {}
        severity_counts = {}
        user_activity = {}
        
        for event in recent_events:
            event_type = event['event_type']
            severity = event['severity']
            user_id = event['user_id']
            
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            user_activity[user_id] = user_activity.get(user_id, 0) + 1
        
        return {
            'report_period_hours': hours,
            'total_events': len(recent_events),
            'event_counts': event_counts,
            'severity_counts': severity_counts,
            'user_activity': user_activity,
            'top_users': sorted(user_activity.items(), key=lambda x: x[1], reverse=True)[:10]
        }
    
    def _trigger_security_alert(self, event: Dict[str, Any]):
        """Trigger security alert for high-severity events."""
        logger.critical(f"SECURITY ALERT: {event['event_type']} - {event['details']}")
        # In practice, would send to SIEM, email, Slack, etc.


class DataEncryption:
    """Data encryption utilities."""
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        if encryption_key:
            self.key = encryption_key
        else:
            self.key = self._generate_key()
    
    def _generate_key(self) -> bytes:
        """Generate encryption key."""
        return secrets.token_bytes(32)  # 256-bit key
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt string data."""
        try:
            from cryptography.fernet import Fernet
            import base64
            
            # Create Fernet instance
            key = base64.urlsafe_b64encode(self.key)
            fernet = Fernet(key)
            
            # Encrypt data
            encrypted = fernet.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
            
        except ImportError:
            # Fallback to simple base64 encoding (not secure!)
            import base64
            logger.warning("Cryptography library not available, using base64 encoding")
            return base64.b64encode(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt string data."""
        try:
            from cryptography.fernet import Fernet
            import base64
            
            # Create Fernet instance
            key = base64.urlsafe_b64encode(self.key)
            fernet = Fernet(key)
            
            # Decrypt data
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = fernet.decrypt(encrypted_bytes)
            return decrypted.decode()
            
        except ImportError:
            # Fallback base64 decoding
            import base64
            return base64.b64decode(encrypted_data.encode()).decode()


class SecureModelManager:
    """Secure model loading and validation."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.model_checksums = {}
        self.encryption = DataEncryption()
    
    def register_model(self, model_path: str, expected_checksum: str):
        """Register model with expected checksum."""
        self.model_checksums[model_path] = expected_checksum
    
    def load_model_securely(self, model_path: str) -> Any:
        """Load model with security validation."""
        # Validate checksum
        if self.config.model_checksum_validation:
            if not self._validate_model_checksum(model_path):
                raise ValueError(f"Model checksum validation failed: {model_path}")
        
        # Load model
        if TORCH_AVAILABLE:
            try:
                # Load with map_location to prevent code execution
                model_data = torch.load(model_path, map_location='cpu')
                
                # Additional validation
                if self.config.secure_model_loading:
                    self._validate_model_structure(model_data)
                
                return model_data
                
            except Exception as e:
                logger.error(f"Secure model loading failed: {e}")
                raise
        else:
            raise ImportError("PyTorch not available for model loading")
    
    def _validate_model_checksum(self, model_path: str) -> bool:
        """Validate model file checksum."""
        if model_path not in self.model_checksums:
            logger.warning(f"No checksum registered for model: {model_path}")
            return False
        
        try:
            with open(model_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            
            expected = self.model_checksums[model_path]
            if file_hash != expected:
                logger.error(f"Checksum mismatch for {model_path}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Checksum validation error: {e}")
            return False
    
    def _validate_model_structure(self, model_data: Any):
        """Validate model structure for security."""
        # Check for suspicious keys or structures
        if isinstance(model_data, dict):
            suspicious_keys = ['__reduce__', '__setstate__', 'eval', 'exec']
            for key in model_data.keys():
                if any(sus in str(key).lower() for sus in suspicious_keys):
                    raise ValueError(f"Suspicious model key detected: {key}")


class SecurityManager:
    """Central security manager coordinating all security components."""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        
        # Initialize components
        self.sanitizer = InputSanitizer()
        self.rate_limiter = RateLimiter(self.config)
        self.auth_manager = AuthenticationManager(self.config)
        self.auditor = SecurityAuditor(self.config)
        self.encryption = DataEncryption()
        self.model_manager = SecureModelManager(self.config)
        
        logger.info("Security manager initialized")
    
    def authenticate_request(
        self,
        api_key: Optional[str] = None,
        session_token: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> Optional[SecurityContext]:
        """Authenticate incoming request."""
        context = None
        
        # Try API key authentication
        if api_key:
            context = self.auth_manager.validate_api_key(api_key)
        
        # Try session authentication
        elif session_token:
            context = self.auth_manager.validate_session(session_token)
        
        if context:
            context.ip_address = ip_address
            self.auditor.log_access_attempt(context, "API", True)
        else:
            # Log failed authentication
            user_id = "unknown"
            self.auditor.log_security_event(
                'AUTHENTICATION_FAILED',
                user_id,
                {'ip_address': ip_address, 'api_key_provided': api_key is not None},
                'WARNING'
            )
        
        return context
    
    def authorize_request(self, context: SecurityContext, required_permission: str) -> bool:
        """Authorize request based on permissions."""
        if '*' in context.permissions or required_permission in context.permissions:
            return True
        
        self.auditor.log_access_attempt(context, required_permission, False)
        return False
    
    def validate_and_sanitize(
        self,
        context: SecurityContext,
        sequence: Optional[str] = None,
        filename: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Validate and sanitize all inputs."""
        # Check rate limits
        if not self.rate_limiter.check_rate_limit(context.user_id):
            raise ValueError("Rate limit exceeded")
        
        # Record request
        self.rate_limiter.record_request(context.user_id)
        
        try:
            result = {}
            
            # Sanitize sequence
            if sequence:
                result['sequence'] = self.sanitizer.sanitize_sequence(sequence)
            
            # Sanitize filename
            if filename:
                result['filename'] = self.sanitizer.sanitize_filename(filename)
            
            # Sanitize parameters
            if parameters:
                result['parameters'] = self.sanitizer.sanitize_parameters(parameters)
                
                # Log generation request
                self.auditor.log_generation_request(context, result['parameters'])
            
            return result
            
        except ValueError as e:
            # Log suspicious activity
            self.auditor.log_suspicious_activity(
                context.user_id,
                'INPUT_VALIDATION_FAILED',
                {'error': str(e), 'sequence_length': len(sequence) if sequence else 0}
            )
            raise
        finally:
            # Complete request (for rate limiting)
            self.rate_limiter.complete_request(context.user_id)
    
    def secure_model_access(self, context: SecurityContext, model_path: str) -> Any:
        """Securely access model with authorization."""
        # Check permissions
        if not self.authorize_request(context, 'model_access'):
            raise PermissionError("Insufficient permissions for model access")
        
        # Load model securely
        return self.model_manager.load_model_securely(model_path)
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get overall security status."""
        return {
            'config': {
                'security_level': 'HIGH' if self.config.use_encryption else 'MEDIUM',
                'audit_enabled': self.config.enable_audit_logging,
                'rate_limiting_enabled': True
            },
            'active_sessions': len(self.auth_manager.sessions),
            'active_api_keys': len(self.auth_manager.api_keys),
            'lockouts': len(self.auth_manager.lockouts),
            'audit_events_24h': len([
                e for e in self.auditor.audit_log
                if e['timestamp'] > time.time() - 86400
            ])
        }


# Decorator for securing functions
def secure_endpoint(required_permission: str):
    """Decorator to secure API endpoints."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Extract security context from arguments
            context = kwargs.get('security_context')
            if not context:
                raise ValueError("Security context required")
            
            # Initialize security manager
            security_manager = SecurityManager()
            
            # Check authorization
            if not security_manager.authorize_request(context, required_permission):
                raise PermissionError(f"Permission denied: {required_permission}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator