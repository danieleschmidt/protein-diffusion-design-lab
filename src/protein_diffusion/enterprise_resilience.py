"""
Enterprise Resilience System for Protein Diffusion Design Lab.

This module implements enterprise-grade resilience including:
- Advanced circuit breakers and fault tolerance
- Distributed system health monitoring  
- Self-healing infrastructure
- Advanced error recovery strategies
- Multi-layer security controls
- Compliance and audit systems
"""

import asyncio
import time
import logging
import hashlib
import json
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, Future
import random
import statistics
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class SecurityLevel(Enum):
    """Security access levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    success_threshold: int = 3
    timeout: float = 30.0
    max_failures: int = 10

@dataclass
class HealthCheckConfig:
    """Configuration for health checks."""
    check_interval: int = 30
    timeout: float = 10.0
    critical_threshold: float = 0.9
    warning_threshold: float = 0.7
    history_size: int = 100

@dataclass
class SecurityConfig:
    """Configuration for security controls."""
    encryption_enabled: bool = True
    audit_enabled: bool = True
    rate_limiting_enabled: bool = True
    max_requests_per_minute: int = 100
    session_timeout: int = 3600
    password_complexity: bool = True

class CircuitBreaker:
    """Advanced circuit breaker implementation."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.call_history = deque(maxlen=100)
        self.metrics = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'circuit_opens': 0,
            'average_response_time': 0.0
        }
        self._lock = threading.RLock()
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap functions with circuit breaker."""
        def wrapper(*args, **kwargs):
            return self.call(lambda: func(*args, **kwargs))
        return wrapper
    
    def call(self, func: Callable) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            self.metrics['total_calls'] += 1
            
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")
            
            start_time = time.time()
            
            try:
                result = func()
                execution_time = time.time() - start_time
                self._on_success(execution_time)
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                self._on_failure(e, execution_time)
                raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        return time.time() - self.last_failure_time >= self.config.recovery_timeout
    
    def _transition_to_half_open(self):
        """Transition circuit breaker to half-open state."""
        self.state = CircuitBreakerState.HALF_OPEN
        self.success_count = 0
        logger.info(f"Circuit breaker {self.name} transitioned to HALF_OPEN")
    
    def _on_success(self, execution_time: float):
        """Handle successful call."""
        self.call_history.append({
            'timestamp': time.time(),
            'success': True,
            'execution_time': execution_time
        })
        
        self.metrics['successful_calls'] += 1
        self.metrics['average_response_time'] = self._calculate_average_response_time()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()
        
        self.failure_count = 0
    
    def _on_failure(self, error: Exception, execution_time: float):
        """Handle failed call."""
        self.call_history.append({
            'timestamp': time.time(),
            'success': False,
            'execution_time': execution_time,
            'error': str(error)
        })
        
        self.metrics['failed_calls'] += 1
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.config.failure_threshold:
            self._transition_to_open()
    
    def _transition_to_closed(self):
        """Transition circuit breaker to closed state."""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        logger.info(f"Circuit breaker {self.name} transitioned to CLOSED")
    
    def _transition_to_open(self):
        """Transition circuit breaker to open state."""
        self.state = CircuitBreakerState.OPEN
        self.metrics['circuit_opens'] += 1
        logger.warning(f"Circuit breaker {self.name} transitioned to OPEN")
    
    def _calculate_average_response_time(self) -> float:
        """Calculate average response time from call history."""
        if not self.call_history:
            return 0.0
        
        times = [call['execution_time'] for call in self.call_history if call['success']]
        return statistics.mean(times) if times else 0.0
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'metrics': self.metrics.copy(),
            'health_ratio': self._calculate_health_ratio()
        }
    
    def _calculate_health_ratio(self) -> float:
        """Calculate health ratio based on recent calls."""
        recent_calls = [call for call in self.call_history 
                       if time.time() - call['timestamp'] <= 300]  # Last 5 minutes
        
        if not recent_calls:
            return 1.0
        
        successful = len([call for call in recent_calls if call['success']])
        return successful / len(recent_calls)

class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass

class HealthMonitor:
    """Comprehensive system health monitoring."""
    
    def __init__(self, config: HealthCheckConfig = None):
        self.config = config or HealthCheckConfig()
        self.health_checks = {}
        self.health_history = defaultdict(lambda: deque(maxlen=self.config.history_size))
        self.alerts = []
        self.monitoring_active = False
        self._monitor_thread = None
        self._lock = threading.RLock()
    
    def register_health_check(self, name: str, check_func: Callable[[], bool], 
                            critical: bool = False):
        """Register a health check function."""
        self.health_checks[name] = {
            'function': check_func,
            'critical': critical,
            'last_check': 0,
            'last_result': None,
            'consecutive_failures': 0
        }
        logger.info(f"Registered health check: {name}")
    
    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._execute_health_checks()
                time.sleep(self.config.check_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.config.check_interval)
    
    def _execute_health_checks(self):
        """Execute all registered health checks."""
        current_time = time.time()
        
        with self._lock:
            for name, check_info in self.health_checks.items():
                try:
                    start_time = time.time()
                    result = check_info['function']()
                    execution_time = time.time() - start_time
                    
                    health_record = {
                        'timestamp': current_time,
                        'healthy': result,
                        'execution_time': execution_time,
                        'check_name': name
                    }
                    
                    self.health_history[name].append(health_record)
                    check_info['last_check'] = current_time
                    check_info['last_result'] = result
                    
                    if result:
                        check_info['consecutive_failures'] = 0
                    else:
                        check_info['consecutive_failures'] += 1
                        self._handle_health_check_failure(name, check_info)
                    
                except Exception as e:
                    logger.error(f"Health check {name} failed with exception: {e}")
                    check_info['consecutive_failures'] += 1
                    self._handle_health_check_failure(name, check_info)
    
    def _handle_health_check_failure(self, name: str, check_info: Dict[str, Any]):
        """Handle health check failure."""
        failures = check_info['consecutive_failures']
        is_critical = check_info['critical']
        
        if failures >= 3 and is_critical:
            self._create_alert(
                AlertSeverity.CRITICAL,
                f"Critical health check {name} failed {failures} times consecutively"
            )
        elif failures >= 5:
            self._create_alert(
                AlertSeverity.HIGH,
                f"Health check {name} failed {failures} times consecutively"
            )
    
    def _create_alert(self, severity: AlertSeverity, message: str):
        """Create and store alert."""
        alert = {
            'id': str(uuid.uuid4()),
            'timestamp': time.time(),
            'severity': severity.value,
            'message': message,
            'acknowledged': False
        }
        
        self.alerts.append(alert)
        logger.log(
            logging.CRITICAL if severity == AlertSeverity.CRITICAL else logging.WARNING,
            f"ALERT [{severity.value.upper()}]: {message}"
        )
        
        # Keep only recent alerts
        self.alerts = self.alerts[-1000:]
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        with self._lock:
            health_summary = {
                'overall_health': self._calculate_overall_health(),
                'timestamp': time.time(),
                'check_results': {},
                'active_alerts': len([a for a in self.alerts if not a['acknowledged']]),
                'monitoring_active': self.monitoring_active
            }
            
            for name, check_info in self.health_checks.items():
                recent_history = list(self.health_history[name])[-10:]  # Last 10 checks
                
                health_summary['check_results'][name] = {
                    'current_status': check_info['last_result'],
                    'consecutive_failures': check_info['consecutive_failures'],
                    'last_check': check_info['last_check'],
                    'success_rate': self._calculate_success_rate(recent_history),
                    'average_response_time': self._calculate_avg_response_time(recent_history),
                    'critical': check_info['critical']
                }
            
            return health_summary
    
    def _calculate_overall_health(self) -> float:
        """Calculate overall system health score."""
        if not self.health_checks:
            return 1.0
        
        total_weight = 0
        weighted_health = 0
        
        for name, check_info in self.health_checks.items():
            weight = 2.0 if check_info['critical'] else 1.0
            total_weight += weight
            
            if check_info['last_result']:
                weighted_health += weight
            else:
                # Partial credit based on consecutive failures
                failures = check_info['consecutive_failures']
                health_factor = max(0, 1.0 - (failures * 0.2))
                weighted_health += weight * health_factor
        
        return weighted_health / total_weight if total_weight > 0 else 1.0
    
    def _calculate_success_rate(self, history: List[Dict[str, Any]]) -> float:
        """Calculate success rate from history."""
        if not history:
            return 1.0
        
        successful = len([h for h in history if h['healthy']])
        return successful / len(history)
    
    def _calculate_avg_response_time(self, history: List[Dict[str, Any]]) -> float:
        """Calculate average response time."""
        if not history:
            return 0.0
        
        times = [h['execution_time'] for h in history]
        return statistics.mean(times)
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert['id'] == alert_id:
                alert['acknowledged'] = True
                logger.info(f"Alert {alert_id} acknowledged")
                break

class SelfHealingSystem:
    """Self-healing system for automatic recovery."""
    
    def __init__(self):
        self.healing_strategies = {}
        self.healing_history = []
        self.recovery_actions = {}
        self._lock = threading.RLock()
    
    def register_healing_strategy(self, 
                                condition: str, 
                                action: Callable[[], bool],
                                description: str = None,
                                max_attempts: int = 3):
        """Register a self-healing strategy."""
        strategy_id = str(uuid.uuid4())
        self.healing_strategies[condition] = {
            'id': strategy_id,
            'action': action,
            'description': description or f"Healing action for {condition}",
            'max_attempts': max_attempts,
            'attempts_made': 0,
            'last_attempt': 0,
            'success_count': 0,
            'failure_count': 0
        }
        logger.info(f"Registered healing strategy for condition: {condition}")
    
    def trigger_healing(self, condition: str, context: Dict[str, Any] = None) -> bool:
        """Trigger healing for a specific condition."""
        with self._lock:
            if condition not in self.healing_strategies:
                logger.warning(f"No healing strategy found for condition: {condition}")
                return False
            
            strategy = self.healing_strategies[condition]
            
            # Check if we've exceeded max attempts recently
            if (strategy['attempts_made'] >= strategy['max_attempts'] and
                time.time() - strategy['last_attempt'] < 300):  # 5 minute cooldown
                logger.warning(f"Max healing attempts reached for {condition}")
                return False
            
            try:
                logger.info(f"Attempting healing for condition: {condition}")
                strategy['attempts_made'] += 1
                strategy['last_attempt'] = time.time()
                
                # Execute healing action
                success = strategy['action']()
                
                healing_record = {
                    'timestamp': time.time(),
                    'condition': condition,
                    'success': success,
                    'context': context or {},
                    'strategy_id': strategy['id'],
                    'attempt_number': strategy['attempts_made']
                }
                
                self.healing_history.append(healing_record)
                
                if success:
                    strategy['success_count'] += 1
                    strategy['attempts_made'] = 0  # Reset attempts on success
                    logger.info(f"Healing successful for condition: {condition}")
                else:
                    strategy['failure_count'] += 1
                    logger.warning(f"Healing failed for condition: {condition}")
                
                return success
                
            except Exception as e:
                logger.error(f"Healing action failed with exception: {e}")
                strategy['failure_count'] += 1
                return False
    
    def get_healing_status(self) -> Dict[str, Any]:
        """Get self-healing system status."""
        with self._lock:
            return {
                'total_strategies': len(self.healing_strategies),
                'strategies': {
                    condition: {
                        'description': strategy['description'],
                        'success_rate': (strategy['success_count'] / 
                                       max(1, strategy['success_count'] + strategy['failure_count'])),
                        'last_attempt': strategy['last_attempt'],
                        'attempts_made': strategy['attempts_made']
                    }
                    for condition, strategy in self.healing_strategies.items()
                },
                'recent_healing_attempts': len([
                    h for h in self.healing_history 
                    if time.time() - h['timestamp'] <= 3600  # Last hour
                ]),
                'healing_success_rate': self._calculate_healing_success_rate()
            }
    
    def _calculate_healing_success_rate(self) -> float:
        """Calculate overall healing success rate."""
        if not self.healing_history:
            return 1.0
        
        recent_attempts = [h for h in self.healing_history 
                         if time.time() - h['timestamp'] <= 86400]  # Last 24 hours
        
        if not recent_attempts:
            return 1.0
        
        successful = len([h for h in recent_attempts if h['success']])
        return successful / len(recent_attempts)

class SecurityManager:
    """Advanced security management system."""
    
    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()
        self.active_sessions = {}
        self.access_logs = []
        self.security_events = []
        self.rate_limits = defaultdict(lambda: {'count': 0, 'reset_time': time.time()})
        self._lock = threading.RLock()
        
        # Initialize security components
        if self.config.encryption_enabled:
            self._initialize_encryption()
        
        if self.config.audit_enabled:
            self._initialize_audit_system()
    
    def _initialize_encryption(self):
        """Initialize encryption components."""
        # In a real implementation, this would set up proper encryption
        self.encryption_key = hashlib.sha256(b"default_key").digest()
        logger.info("Encryption system initialized")
    
    def _initialize_audit_system(self):
        """Initialize audit logging system."""
        self.audit_log = []
        logger.info("Audit system initialized")
    
    def authenticate_user(self, username: str, password: str, 
                         client_ip: str = None) -> Optional[Dict[str, Any]]:
        """Authenticate user and create session."""
        # Rate limiting check
        if not self._check_rate_limit(client_ip or 'unknown'):
            self._log_security_event(
                'RATE_LIMIT_EXCEEDED',
                {'username': username, 'client_ip': client_ip}
            )
            raise SecurityException("Rate limit exceeded")
        
        # Simulate authentication (in reality, check against secure database)
        if self._verify_credentials(username, password):
            session_id = str(uuid.uuid4())
            session = {
                'session_id': session_id,
                'username': username,
                'client_ip': client_ip,
                'created_at': time.time(),
                'last_activity': time.time(),
                'permissions': self._get_user_permissions(username)
            }
            
            with self._lock:
                self.active_sessions[session_id] = session
            
            self._log_audit_event('USER_LOGIN', {
                'username': username,
                'session_id': session_id,
                'client_ip': client_ip
            })
            
            return session
        else:
            self._log_security_event(
                'AUTHENTICATION_FAILED',
                {'username': username, 'client_ip': client_ip}
            )
            return None
    
    def _verify_credentials(self, username: str, password: str) -> bool:
        """Verify user credentials."""
        # Simulate credential verification
        # In reality, this would check against a secure database with hashed passwords
        return len(username) > 3 and len(password) > 6
    
    def _get_user_permissions(self, username: str) -> List[str]:
        """Get user permissions."""
        # Simulate permission retrieval
        if username.startswith('admin'):
            return ['read', 'write', 'admin', 'delete']
        elif username.startswith('user'):
            return ['read', 'write']
        else:
            return ['read']
    
    def authorize_action(self, session_id: str, action: str, 
                        resource: str = None) -> bool:
        """Authorize user action."""
        with self._lock:
            if session_id not in self.active_sessions:
                return False
            
            session = self.active_sessions[session_id]
            
            # Check session timeout
            if time.time() - session['last_activity'] > self.config.session_timeout:
                self.logout_user(session_id)
                return False
            
            # Update activity
            session['last_activity'] = time.time()
            
            # Check permissions
            required_permission = self._get_required_permission(action)
            if required_permission not in session['permissions']:
                self._log_security_event(
                    'AUTHORIZATION_FAILED',
                    {
                        'username': session['username'],
                        'action': action,
                        'resource': resource,
                        'required_permission': required_permission
                    }
                )
                return False
            
            self._log_audit_event('ACTION_AUTHORIZED', {
                'username': session['username'],
                'action': action,
                'resource': resource
            })
            
            return True
    
    def _get_required_permission(self, action: str) -> str:
        """Get required permission for action."""
        permission_map = {
            'read': 'read',
            'generate': 'write',
            'rank': 'write',
            'delete': 'delete',
            'admin': 'admin'
        }
        return permission_map.get(action, 'read')
    
    def logout_user(self, session_id: str):
        """Log out user and invalidate session."""
        with self._lock:
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                self._log_audit_event('USER_LOGOUT', {
                    'username': session['username'],
                    'session_id': session_id
                })
                del self.active_sessions[session_id]
    
    def _check_rate_limit(self, client_id: str) -> bool:
        """Check rate limiting for client."""
        if not self.config.rate_limiting_enabled:
            return True
        
        current_time = time.time()
        rate_limit = self.rate_limits[client_id]
        
        # Reset counter if minute has passed
        if current_time - rate_limit['reset_time'] >= 60:
            rate_limit['count'] = 0
            rate_limit['reset_time'] = current_time
        
        rate_limit['count'] += 1
        return rate_limit['count'] <= self.config.max_requests_per_minute
    
    def _log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security event."""
        event = {
            'timestamp': time.time(),
            'event_type': event_type,
            'details': details
        }
        self.security_events.append(event)
        logger.warning(f"Security event: {event_type} - {details}")
        
        # Keep only recent events
        self.security_events = self.security_events[-10000:]
    
    def _log_audit_event(self, action: str, details: Dict[str, Any]):
        """Log audit event."""
        if not self.config.audit_enabled:
            return
        
        audit_entry = {
            'timestamp': time.time(),
            'action': action,
            'details': details
        }
        self.audit_log.append(audit_entry)
        
        # Keep only recent audit entries
        self.audit_log = self.audit_log[-50000:]
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get security system status."""
        with self._lock:
            recent_events = [
                e for e in self.security_events 
                if time.time() - e['timestamp'] <= 3600  # Last hour
            ]
            
            return {
                'active_sessions': len(self.active_sessions),
                'recent_security_events': len(recent_events),
                'audit_entries': len(self.audit_log),
                'rate_limits_active': len(self.rate_limits),
                'security_config': {
                    'encryption_enabled': self.config.encryption_enabled,
                    'audit_enabled': self.config.audit_enabled,
                    'rate_limiting_enabled': self.config.rate_limiting_enabled
                }
            }
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        if not self.config.encryption_enabled:
            return data
        
        # Simple encryption simulation (use proper encryption in production)
        encoded = data.encode('utf-8')
        encrypted = hashlib.pbkdf2_hmac('sha256', encoded, b'salt', 100000)
        return encrypted.hex()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        if not self.config.encryption_enabled:
            return encrypted_data
        
        # This is just a simulation - proper decryption would be needed
        return "decrypted_data"

class SecurityException(Exception):
    """Security-related exception."""
    pass

class EnterpriseResilienceSystem:
    """Main enterprise resilience system."""
    
    def __init__(self, 
                 circuit_breaker_config: CircuitBreakerConfig = None,
                 health_check_config: HealthCheckConfig = None,
                 security_config: SecurityConfig = None):
        self.circuit_breakers = {}
        self.health_monitor = HealthMonitor(health_check_config)
        self.self_healing = SelfHealingSystem()
        self.security_manager = SecurityManager(security_config)
        self.system_metrics = {
            'uptime': time.time(),
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0
        }
        
        # Initialize default components
        self._initialize_default_health_checks()
        self._initialize_default_healing_strategies()
        
        logger.info("Enterprise resilience system initialized")
    
    def _initialize_default_health_checks(self):
        """Initialize default health checks."""
        def memory_check():
            # Simulate memory check
            return True  # In reality, check actual memory usage
        
        def cpu_check():
            # Simulate CPU check
            return True  # In reality, check actual CPU usage
        
        def database_check():
            # Simulate database connectivity check
            return True  # In reality, check database connection
        
        self.health_monitor.register_health_check('memory', memory_check, critical=True)
        self.health_monitor.register_health_check('cpu', cpu_check, critical=True)
        self.health_monitor.register_health_check('database', database_check, critical=True)
    
    def _initialize_default_healing_strategies(self):
        """Initialize default healing strategies."""
        def restart_service():
            # Simulate service restart
            logger.info("Simulating service restart")
            return True
        
        def clear_cache():
            # Simulate cache clearing
            logger.info("Simulating cache clear")
            return True
        
        def reconnect_database():
            # Simulate database reconnection
            logger.info("Simulating database reconnection")
            return True
        
        self.self_healing.register_healing_strategy(
            'service_failure', restart_service, "Restart failed service"
        )
        self.self_healing.register_healing_strategy(
            'memory_high', clear_cache, "Clear cache to free memory"
        )
        self.self_healing.register_healing_strategy(
            'database_connection_lost', reconnect_database, "Reconnect to database"
        )
    
    def create_circuit_breaker(self, name: str, 
                             config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Create and register a circuit breaker."""
        circuit_breaker = CircuitBreaker(name, config)
        self.circuit_breakers[name] = circuit_breaker
        return circuit_breaker
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self.circuit_breakers.get(name)
    
    def start_monitoring(self):
        """Start all monitoring systems."""
        self.health_monitor.start_monitoring()
        logger.info("Enterprise resilience monitoring started")
    
    def stop_monitoring(self):
        """Stop all monitoring systems."""
        self.health_monitor.stop_monitoring()
        logger.info("Enterprise resilience monitoring stopped")
    
    def handle_system_failure(self, failure_type: str, context: Dict[str, Any] = None):
        """Handle system failure with automatic recovery."""
        logger.error(f"System failure detected: {failure_type}")
        
        # Attempt self-healing
        healing_success = self.self_healing.trigger_healing(failure_type, context)
        
        if healing_success:
            logger.info(f"System failure {failure_type} resolved through self-healing")
        else:
            logger.error(f"Self-healing failed for {failure_type}, escalating...")
            # In a real system, this would escalate to administrators
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        health_status = self.health_monitor.get_system_health()
        security_status = self.security_manager.get_security_status()
        healing_status = self.self_healing.get_healing_status()
        
        # Calculate uptime
        uptime = time.time() - self.system_metrics['uptime']
        
        # Calculate request success rate
        total_requests = self.system_metrics['total_requests']
        success_rate = (self.system_metrics['successful_requests'] / 
                       max(1, total_requests))
        
        circuit_breaker_status = {
            name: cb.get_status() 
            for name, cb in self.circuit_breakers.items()
        }
        
        return {
            'timestamp': time.time(),
            'uptime_seconds': uptime,
            'overall_health': health_status['overall_health'],
            'success_rate': success_rate,
            'total_requests': total_requests,
            'health_monitoring': health_status,
            'security': security_status,
            'self_healing': healing_status,
            'circuit_breakers': circuit_breaker_status,
            'resilience_grade': self._calculate_resilience_grade(
                health_status['overall_health'], success_rate, healing_status['healing_success_rate']
            )
        }
    
    def _calculate_resilience_grade(self, health: float, success_rate: float, 
                                  healing_rate: float) -> str:
        """Calculate overall resilience grade."""
        overall_score = (health * 0.4 + success_rate * 0.4 + healing_rate * 0.2)
        
        if overall_score >= 0.95:
            return "A+ (Excellent)"
        elif overall_score >= 0.9:
            return "A (Very Good)"
        elif overall_score >= 0.8:
            return "B (Good)"
        elif overall_score >= 0.7:
            return "C (Fair)"
        else:
            return "D (Poor)"
    
    def secure_execute(self, session_id: str, action: str, 
                      func: Callable, *args, **kwargs) -> Any:
        """Execute function with security and resilience checks."""
        # Security check
        if not self.security_manager.authorize_action(session_id, action):
            raise SecurityException(f"Unauthorized action: {action}")
        
        # Get or create circuit breaker for this action
        circuit_breaker = self.get_circuit_breaker(action)
        if circuit_breaker is None:
            circuit_breaker = self.create_circuit_breaker(action)
        
        # Execute with circuit breaker protection
        self.system_metrics['total_requests'] += 1
        
        try:
            result = circuit_breaker.call(lambda: func(*args, **kwargs))
            self.system_metrics['successful_requests'] += 1
            return result
        except Exception as e:
            self.system_metrics['failed_requests'] += 1
            
            # Attempt healing if this looks like a system failure
            if isinstance(e, (ConnectionError, TimeoutError)):
                self.handle_system_failure('connection_failure', {
                    'action': action,
                    'error': str(e)
                })
            
            raise

# Convenience functions for easy integration
def create_resilience_system(config: Dict[str, Any] = None) -> EnterpriseResilienceSystem:
    """Create and configure enterprise resilience system."""
    if config is None:
        config = {}
    
    circuit_config = CircuitBreakerConfig(**config.get('circuit_breaker', {}))
    health_config = HealthCheckConfig(**config.get('health_check', {}))
    security_config = SecurityConfig(**config.get('security', {}))
    
    system = EnterpriseResilienceSystem(circuit_config, health_config, security_config)
    system.start_monitoring()
    
    return system

# Export all classes and functions
__all__ = [
    'EnterpriseResilienceSystem',
    'CircuitBreaker',
    'HealthMonitor',
    'SelfHealingSystem',
    'SecurityManager',
    'CircuitBreakerConfig',
    'HealthCheckConfig',
    'SecurityConfig',
    'CircuitBreakerState',
    'AlertSeverity',
    'SecurityLevel',
    'SecurityException',
    'create_resilience_system'
]