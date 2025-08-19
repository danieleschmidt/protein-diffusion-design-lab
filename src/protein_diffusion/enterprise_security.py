"""
Enterprise Security Framework - Comprehensive security and compliance system.

This module provides enterprise-grade security features including authentication,
authorization, audit logging, data encryption, and compliance management.
"""

import hashlib
import hmac
import secrets
import time
import logging
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Callable, Union
from pathlib import Path
from enum import Enum
import re
from datetime import datetime, timedelta

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.backends import default_backend
    import base64
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    Fernet = None

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for operations."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class AuthenticationMethod(Enum):
    """Authentication methods supported."""
    API_KEY = "api_key"
    JWT_TOKEN = "jwt_token"
    OAUTH2 = "oauth2"
    CERTIFICATE = "certificate"
    MULTI_FACTOR = "multi_factor"


class AuditEventType(Enum):
    """Types of audit events."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    SYSTEM_OPERATION = "system_operation"
    SECURITY_VIOLATION = "security_violation"
    COMPLIANCE_EVENT = "compliance_event"


@dataclass
class SecurityPolicy:
    """Security policy definition."""
    name: str
    description: str
    security_level: SecurityLevel
    required_permissions: List[str] = field(default_factory=list)
    allowed_operations: List[str] = field(default_factory=list)
    rate_limits: Dict[str, int] = field(default_factory=dict)  # operation -> requests per minute
    ip_whitelist: List[str] = field(default_factory=list)
    ip_blacklist: List[str] = field(default_factory=list)
    require_encryption: bool = True
    require_audit: bool = True
    session_timeout: int = 3600  # seconds
    max_failed_attempts: int = 5
    lockout_duration: int = 900  # 15 minutes


@dataclass
class User:
    """User account information."""
    user_id: str
    username: str
    email: str
    roles: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    is_active: bool = True
    created_at: float = field(default_factory=time.time)
    last_login: Optional[float] = None
    failed_attempts: int = 0
    locked_until: Optional[float] = None
    api_key: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """User session information."""
    session_id: str
    user_id: str
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    expires_at: float = 0
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    permissions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.expires_at == 0:
            self.expires_at = self.created_at + 3600  # 1 hour default
    
    def is_valid(self) -> bool:
        """Check if session is still valid."""
        current_time = time.time()
        return (
            current_time < self.expires_at and
            current_time - self.last_activity < 1800  # 30 min activity timeout
        )
    
    def refresh(self):
        """Refresh session activity."""
        self.last_activity = time.time()


@dataclass
class AuditEvent:
    """Security audit event."""
    event_id: str
    event_type: AuditEventType
    timestamp: float = field(default_factory=time.time)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    resource: Optional[str] = None
    operation: Optional[str] = None
    success: bool = True
    details: Dict[str, Any] = field(default_factory=dict)
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    compliance_flags: List[str] = field(default_factory=list)


@dataclass
class SecurityConfig:
    """Security framework configuration."""
    # Encryption settings
    enable_encryption: bool = True
    encryption_key_rotation_days: int = 90
    
    # Authentication settings
    default_auth_method: AuthenticationMethod = AuthenticationMethod.API_KEY
    jwt_secret: Optional[str] = None
    jwt_expiry_hours: int = 24
    
    # Session management
    session_timeout_minutes: int = 60
    max_concurrent_sessions: int = 5
    
    # Rate limiting
    global_rate_limit: int = 1000  # requests per minute
    user_rate_limit: int = 100
    
    # Audit settings
    enable_audit_logging: bool = True
    audit_log_directory: str = "./security_logs"
    audit_retention_days: int = 365
    
    # Security policies
    default_security_level: SecurityLevel = SecurityLevel.INTERNAL
    require_https: bool = True
    
    # Compliance settings
    gdpr_enabled: bool = True
    hipaa_enabled: bool = False
    sox_enabled: bool = False
    
    # IP filtering
    enable_ip_filtering: bool = False
    allowed_ip_ranges: List[str] = field(default_factory=list)
    blocked_ip_ranges: List[str] = field(default_factory=list)


class EncryptionManager:
    """Handles data encryption and decryption."""
    
    def __init__(self, master_key: Optional[str] = None):
        if not CRYPTO_AVAILABLE:
            logger.warning("Cryptography library not available - encryption disabled")
            self.encryption_enabled = False
            return
        
        self.encryption_enabled = True
        
        if master_key:
            self.master_key = master_key.encode()
        else:
            self.master_key = secrets.token_bytes(32)
        
        # Initialize Fernet cipher
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'protein_diffusion_salt',  # In production, use random salt per key
            iterations=100000,
            backend=default_backend()
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key))
        self.cipher_suite = Fernet(key)
        
        logger.info("Encryption manager initialized")
    
    def encrypt(self, data: Union[str, bytes]) -> bytes:
        """Encrypt data."""
        if not self.encryption_enabled:
            return data.encode() if isinstance(data, str) else data
        
        if isinstance(data, str):
            data = data.encode()
        
        return self.cipher_suite.encrypt(data)
    
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """Decrypt data."""
        if not self.encryption_enabled:
            return encrypted_data
        
        return self.cipher_suite.decrypt(encrypted_data)
    
    def encrypt_dict(self, data: Dict[str, Any]) -> bytes:
        """Encrypt dictionary data."""
        json_data = json.dumps(data, default=str)
        return self.encrypt(json_data)
    
    def decrypt_dict(self, encrypted_data: bytes) -> Dict[str, Any]:
        """Decrypt dictionary data."""
        decrypted_json = self.decrypt(encrypted_data).decode()
        return json.loads(decrypted_json)
    
    def generate_hash(self, data: str, salt: Optional[str] = None) -> str:
        """Generate secure hash of data."""
        if salt is None:
            salt = secrets.token_hex(16)
        
        combined = f"{data}{salt}"
        hash_obj = hashlib.pbkdf2_hmac('sha256', combined.encode(), b'protein_salt', 100000)
        return f"{salt}${hash_obj.hex()}"
    
    def verify_hash(self, data: str, hashed_data: str) -> bool:
        """Verify data against hash."""
        try:
            salt, hash_value = hashed_data.split('$', 1)
            return self.generate_hash(data, salt) == hashed_data
        except ValueError:
            return False


class SecurityManager:
    """
    Enterprise security manager providing comprehensive security controls.
    
    This class handles:
    - User authentication and authorization
    - Session management
    - Security policy enforcement
    - Audit logging
    - Data encryption
    - Compliance monitoring
    
    Example:
        >>> security = SecurityManager()
        >>> user = security.authenticate_user("api_key", "secret-key")
        >>> if security.authorize_operation(user.user_id, "generate_proteins"):
        ...     # Perform operation
        ...     security.audit_operation("generate_proteins", user.user_id)
    """
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        
        # Initialize encryption
        self.encryption = EncryptionManager()
        
        # User and session storage
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Session] = {}
        self.api_keys: Dict[str, str] = {}  # api_key -> user_id
        
        # Security policies
        self.policies: Dict[str, SecurityPolicy] = {}
        self._setup_default_policies()
        
        # Audit logging
        self.audit_events: List[AuditEvent] = []
        
        # Rate limiting
        self.rate_limits: Dict[str, Dict[str, List[float]]] = {}  # user_id -> operation -> timestamps
        
        # Setup audit logging
        if self.config.enable_audit_logging:
            self._setup_audit_logging()
        
        logger.info("Security manager initialized")
    
    def _setup_default_policies(self):
        """Setup default security policies."""
        # Public operations
        self.policies["public"] = SecurityPolicy(
            name="Public Access",
            description="Public read-only operations",
            security_level=SecurityLevel.PUBLIC,
            allowed_operations=["health_check", "get_model_info"],
            rate_limits={"health_check": 60, "get_model_info": 10},
            require_encryption=False,
            require_audit=False
        )
        
        # Internal operations
        self.policies["internal"] = SecurityPolicy(
            name="Internal Access",
            description="Internal operations for authenticated users",
            security_level=SecurityLevel.INTERNAL,
            required_permissions=["authenticated"],
            allowed_operations=["generate_proteins", "evaluate_sequences", "get_rankings"],
            rate_limits={"generate_proteins": 50, "evaluate_sequences": 30, "get_rankings": 20}
        )
        
        # Confidential operations
        self.policies["confidential"] = SecurityPolicy(
            name="Confidential Access",
            description="Confidential operations requiring elevated permissions",
            security_level=SecurityLevel.CONFIDENTIAL,
            required_permissions=["authenticated", "confidential_access"],
            allowed_operations=["batch_generation", "custom_training", "export_data"],
            rate_limits={"batch_generation": 10, "custom_training": 5, "export_data": 3}
        )
        
        # Admin operations
        self.policies["admin"] = SecurityPolicy(
            name="Admin Access",
            description="Administrative operations",
            security_level=SecurityLevel.RESTRICTED,
            required_permissions=["authenticated", "admin"],
            allowed_operations=["system_config", "user_management", "audit_access"],
            rate_limits={"system_config": 5, "user_management": 10, "audit_access": 20}
        )
    
    def _setup_audit_logging(self):
        """Setup audit logging to file."""
        log_dir = Path(self.config.audit_log_directory)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create audit logger
        audit_logger = logging.getLogger('security_audit')
        audit_logger.setLevel(logging.INFO)
        
        # File handler for audit logs
        log_file = log_dir / f"security_audit_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - AUDIT - %(message)s'
        )
        file_handler.setFormatter(formatter)
        audit_logger.addHandler(file_handler)
        
        self.audit_logger = audit_logger
    
    def create_user(
        self,
        username: str,
        email: str,
        roles: List[str] = None,
        security_level: SecurityLevel = None
    ) -> User:
        """Create a new user account."""
        user_id = secrets.token_hex(16)
        api_key = secrets.token_urlsafe(32)
        
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            roles=roles or [],
            security_level=security_level or self.config.default_security_level,
            api_key=api_key
        )
        
        # Generate permissions from roles
        user.permissions = self._generate_permissions_from_roles(user.roles)
        
        self.users[user_id] = user
        self.api_keys[api_key] = user_id
        
        # Audit user creation
        self._audit_event(
            AuditEventType.SYSTEM_OPERATION,
            user_id=user_id,
            operation="create_user",
            details={"username": username, "email": email, "roles": roles}
        )
        
        logger.info(f"Created user {username} with ID {user_id}")
        return user
    
    def _generate_permissions_from_roles(self, roles: List[str]) -> List[str]:
        """Generate permissions based on user roles."""
        permissions = ["authenticated"]  # Base permission
        
        role_permissions = {
            "user": ["generate_proteins", "evaluate_sequences"],
            "researcher": ["generate_proteins", "evaluate_sequences", "get_rankings", "batch_generation"],
            "admin": ["generate_proteins", "evaluate_sequences", "get_rankings", "batch_generation", 
                     "confidential_access", "admin", "system_config", "user_management", "audit_access"],
            "api_client": ["generate_proteins", "evaluate_sequences", "get_rankings"]
        }
        
        for role in roles:
            permissions.extend(role_permissions.get(role, []))
        
        return list(set(permissions))  # Remove duplicates
    
    def authenticate_user(self, auth_method: str, credentials: str, context: Dict[str, Any] = None) -> Optional[User]:
        """Authenticate user with provided credentials."""
        context = context or {}
        
        user = None
        auth_success = False
        
        try:
            if auth_method == "api_key":
                user_id = self.api_keys.get(credentials)
                if user_id and user_id in self.users:
                    user = self.users[user_id]
                    auth_success = user.is_active and self._check_user_lockout(user)
            
            elif auth_method == "jwt_token":
                # JWT validation would go here
                # For demo purposes, treating as API key
                user_id = self.api_keys.get(credentials)
                if user_id and user_id in self.users:
                    user = self.users[user_id]
                    auth_success = user.is_active and self._check_user_lockout(user)
            
            # Update authentication attempt
            if user:
                if auth_success:
                    user.last_login = time.time()
                    user.failed_attempts = 0
                else:
                    user.failed_attempts += 1
                    if user.failed_attempts >= 5:  # Max attempts
                        user.locked_until = time.time() + 900  # 15 min lockout
            
            # Audit authentication attempt
            self._audit_event(
                AuditEventType.AUTHENTICATION,
                user_id=user.user_id if user else None,
                operation="authenticate",
                success=auth_success,
                details={
                    "auth_method": auth_method,
                    "ip_address": context.get("ip_address"),
                    "user_agent": context.get("user_agent"),
                    "failed_attempts": user.failed_attempts if user else 0
                }
            )
            
            return user if auth_success else None
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            self._audit_event(
                AuditEventType.SECURITY_VIOLATION,
                operation="authenticate",
                success=False,
                details={"error": str(e), "auth_method": auth_method}
            )
            return None
    
    def _check_user_lockout(self, user: User) -> bool:
        """Check if user is locked out."""
        if user.locked_until and time.time() < user.locked_until:
            return False
        elif user.locked_until and time.time() >= user.locked_until:
            # Unlock user
            user.locked_until = None
            user.failed_attempts = 0
        return True
    
    def create_session(self, user: User, context: Dict[str, Any] = None) -> Session:
        """Create a new user session."""
        context = context or {}
        
        session_id = secrets.token_urlsafe(32)
        session = Session(
            session_id=session_id,
            user_id=user.user_id,
            ip_address=context.get("ip_address"),
            user_agent=context.get("user_agent"),
            permissions=user.permissions.copy(),
            expires_at=time.time() + (self.config.session_timeout_minutes * 60)
        )
        
        self.sessions[session_id] = session
        
        # Clean up old sessions for user
        self._cleanup_user_sessions(user.user_id)
        
        # Audit session creation
        self._audit_event(
            AuditEventType.SYSTEM_OPERATION,
            user_id=user.user_id,
            session_id=session_id,
            operation="create_session",
            details={"ip_address": session.ip_address, "expires_at": session.expires_at}
        )
        
        return session
    
    def _cleanup_user_sessions(self, user_id: str):
        """Clean up old or excess sessions for a user."""
        user_sessions = [
            (sid, session) for sid, session in self.sessions.items() 
            if session.user_id == user_id and session.is_valid()
        ]
        
        # Remove excess sessions (keep most recent)
        if len(user_sessions) > self.config.max_concurrent_sessions:
            user_sessions.sort(key=lambda x: x[1].last_activity, reverse=True)
            for sid, _ in user_sessions[self.config.max_concurrent_sessions:]:
                del self.sessions[sid]
    
    def validate_session(self, session_id: str) -> Optional[Session]:
        """Validate and refresh session."""
        session = self.sessions.get(session_id)
        
        if session and session.is_valid():
            session.refresh()
            return session
        elif session:
            # Remove invalid session
            del self.sessions[session_id]
        
        return None
    
    def authorize_operation(self, user_id: str, operation: str, resource: Optional[str] = None) -> bool:
        """Authorize user for specific operation."""
        user = self.users.get(user_id)
        if not user or not user.is_active:
            self._audit_event(
                AuditEventType.AUTHORIZATION,
                user_id=user_id,
                operation=operation,
                resource=resource,
                success=False,
                details={"reason": "user_not_found_or_inactive"}
            )
            return False
        
        # Check rate limits
        if not self._check_rate_limit(user_id, operation):
            self._audit_event(
                AuditEventType.SECURITY_VIOLATION,
                user_id=user_id,
                operation=operation,
                success=False,
                details={"reason": "rate_limit_exceeded"}
            )
            return False
        
        # Find applicable policy
        applicable_policy = self._find_applicable_policy(operation, user.security_level)
        
        if not applicable_policy:
            self._audit_event(
                AuditEventType.AUTHORIZATION,
                user_id=user_id,
                operation=operation,
                resource=resource,
                success=False,
                details={"reason": "no_applicable_policy"}
            )
            return False
        
        # Check permissions
        if not all(perm in user.permissions for perm in applicable_policy.required_permissions):
            self._audit_event(
                AuditEventType.AUTHORIZATION,
                user_id=user_id,
                operation=operation,
                resource=resource,
                success=False,
                details={"reason": "insufficient_permissions", "required": applicable_policy.required_permissions}
            )
            return False
        
        # Check if operation is allowed
        if operation not in applicable_policy.allowed_operations:
            self._audit_event(
                AuditEventType.AUTHORIZATION,
                user_id=user_id,
                operation=operation,
                resource=resource,
                success=False,
                details={"reason": "operation_not_allowed"}
            )
            return False
        
        # Authorization successful
        self._audit_event(
            AuditEventType.AUTHORIZATION,
            user_id=user_id,
            operation=operation,
            resource=resource,
            success=True
        )
        
        return True
    
    def _find_applicable_policy(self, operation: str, security_level: SecurityLevel) -> Optional[SecurityPolicy]:
        """Find applicable security policy for operation and security level."""
        for policy in self.policies.values():
            if (
                operation in policy.allowed_operations and
                policy.security_level.value in [security_level.value, SecurityLevel.PUBLIC.value]
            ):
                return policy
        return None
    
    def _check_rate_limit(self, user_id: str, operation: str) -> bool:
        """Check if user has exceeded rate limit for operation."""
        current_time = time.time()
        
        if user_id not in self.rate_limits:
            self.rate_limits[user_id] = {}
        
        if operation not in self.rate_limits[user_id]:
            self.rate_limits[user_id][operation] = []
        
        # Clean old timestamps (older than 1 minute)
        self.rate_limits[user_id][operation] = [
            ts for ts in self.rate_limits[user_id][operation]
            if current_time - ts < 60
        ]
        
        # Check limit
        user = self.users.get(user_id)
        if not user:
            return False
        
        # Get applicable rate limit
        applicable_policy = self._find_applicable_policy(operation, user.security_level)
        if applicable_policy and operation in applicable_policy.rate_limits:
            limit = applicable_policy.rate_limits[operation]
        else:
            limit = self.config.user_rate_limit
        
        if len(self.rate_limits[user_id][operation]) >= limit:
            return False
        
        # Record this request
        self.rate_limits[user_id][operation].append(current_time)
        return True
    
    def _audit_event(
        self,
        event_type: AuditEventType,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        operation: Optional[str] = None,
        resource: Optional[str] = None,
        success: bool = True,
        details: Dict[str, Any] = None
    ):
        """Record audit event."""
        event = AuditEvent(
            event_id=secrets.token_hex(16),
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            operation=operation,
            resource=resource,
            success=success,
            details=details or {}
        )
        
        self.audit_events.append(event)
        
        # Log to audit file
        if hasattr(self, 'audit_logger'):
            self.audit_logger.info(json.dumps({
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "timestamp": event.timestamp,
                "user_id": event.user_id,
                "operation": event.operation,
                "success": event.success,
                "details": event.details
            }, default=str))
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.users.get(user_id)
    
    def validate_client(self, client_id: str, context: Dict[str, Any] = None) -> bool:
        """Validate client access (simplified)."""
        # For demo, treating client_id as user_id
        user = self.users.get(client_id)
        return user is not None and user.is_active
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        current_time = time.time()
        
        # Active sessions
        active_sessions = sum(1 for s in self.sessions.values() if s.is_valid())
        
        # Recent audit events
        recent_events = [
            e for e in self.audit_events
            if current_time - e.timestamp < 3600  # Last hour
        ]
        
        # Failed authentications
        failed_auths = [
            e for e in recent_events
            if e.event_type == AuditEventType.AUTHENTICATION and not e.success
        ]
        
        # Security violations
        security_violations = [
            e for e in recent_events
            if e.event_type == AuditEventType.SECURITY_VIOLATION
        ]
        
        # Locked users
        locked_users = sum(
            1 for u in self.users.values()
            if u.locked_until and current_time < u.locked_until
        )
        
        return {
            "timestamp": current_time,
            "total_users": len(self.users),
            "active_sessions": active_sessions,
            "recent_audit_events": len(recent_events),
            "failed_authentications": len(failed_auths),
            "security_violations": len(security_violations),
            "locked_users": locked_users,
            "encryption_enabled": self.encryption.encryption_enabled,
            "audit_logging_enabled": self.config.enable_audit_logging,
            "policies_active": len(self.policies)
        }
    
    def export_audit_log(self, start_time: Optional[float] = None, end_time: Optional[float] = None) -> List[Dict[str, Any]]:
        """Export audit log for compliance reporting."""
        start_time = start_time or (time.time() - 86400)  # Last 24 hours default
        end_time = end_time or time.time()
        
        filtered_events = [
            e for e in self.audit_events
            if start_time <= e.timestamp <= end_time
        ]
        
        return [
            {
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "timestamp": event.timestamp,
                "user_id": event.user_id,
                "operation": event.operation,
                "resource": event.resource,
                "success": event.success,
                "details": event.details,
                "ip_address": event.ip_address
            }
            for event in filtered_events
        ]
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        expired_sessions = [
            sid for sid, session in self.sessions.items()
            if not session.is_valid()
        ]
        
        for sid in expired_sessions:
            del self.sessions[sid]
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")


# Security decorators
def require_authentication(security_manager: SecurityManager):
    """Decorator to require authentication for function."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Extract authentication from kwargs or context
            auth_token = kwargs.pop('auth_token', None)
            if not auth_token:
                raise PermissionError("Authentication required")
            
            user = security_manager.authenticate_user("api_key", auth_token)
            if not user:
                raise PermissionError("Invalid authentication")
            
            kwargs['authenticated_user'] = user
            return func(*args, **kwargs)
        return wrapper
    return decorator


def require_authorization(security_manager: SecurityManager, operation: str):
    """Decorator to require authorization for specific operation."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            user = kwargs.get('authenticated_user')
            if not user:
                raise PermissionError("User authentication required")
            
            if not security_manager.authorize_operation(user.user_id, operation):
                raise PermissionError(f"Insufficient permissions for {operation}")
            
            # Audit the operation
            security_manager._audit_event(
                AuditEventType.DATA_ACCESS,
                user_id=user.user_id,
                operation=operation,
                success=True
            )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator
