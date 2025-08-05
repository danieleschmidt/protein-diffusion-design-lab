"""
Security utilities for the protein diffusion system.

This module provides input sanitization, rate limiting, and security
measures to ensure safe operation in production environments.
"""

import re
import hashlib
import hmac
import time
import threading
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, deque
import logging
import functools

logger = logging.getLogger(__name__)

@dataclass
class SecurityConfig:
    """Security configuration settings."""
    # Rate limiting
    max_requests_per_minute: int = 60
    max_requests_per_hour: int = 1000
    max_concurrent_requests: int = 10
    
    # Input validation
    max_sequence_length: int = 2000
    max_batch_size: int = 100
    allowed_amino_acids: Set[str] = field(default_factory=lambda: set("ACDEFGHIKLMNPQRSTVWYUOX"))
    
    # File security
    allowed_file_extensions: Set[str] = field(default_factory=lambda: {".pdb", ".fasta", ".txt"})
    max_file_size_mb: int = 50
    
    # Model security
    max_generation_time_seconds: int = 300
    require_authentication: bool = False
    
    # Logging
    log_security_events: bool = True
    alert_on_suspicious_activity: bool = True

class InputSanitizer:
    """Sanitize and validate user inputs."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        
        # Compile regex patterns for efficiency
        self.amino_acid_pattern = re.compile(f"^[{''.join(config.allowed_amino_acids)}]*$", re.IGNORECASE)
        self.malicious_patterns = [
            re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
            re.compile(r'javascript:', re.IGNORECASE),
            re.compile(r'on\w+\s*=', re.IGNORECASE),
            re.compile(r'eval\s*\(', re.IGNORECASE),
            re.compile(r'exec\s*\(', re.IGNORECASE),
            re.compile(r'import\s+os', re.IGNORECASE),
            re.compile(r'__.*__', re.IGNORECASE),
        ]
        
        # SQL injection patterns (even though we might not use SQL)
        self.sql_injection_patterns = [
            re.compile(r"'.*?;.*?--", re.IGNORECASE),
            re.compile(r"union\s+select", re.IGNORECASE),
            re.compile(r"drop\s+table", re.IGNORECASE),
            re.compile(r"delete\s+from", re.IGNORECASE),
        ]
    
    def sanitize_sequence(self, sequence: str) -> str:
        """
        Sanitize a protein sequence.
        
        Args:
            sequence: Raw input sequence
            
        Returns:
            Sanitized sequence
            
        Raises:
            ValueError: If sequence is invalid or potentially malicious
        """
        if not isinstance(sequence, str):
            raise ValueError(f"Sequence must be string, got {type(sequence)}")
        
        # Basic sanitization
        sequence = sequence.strip().upper()
        
        # Remove whitespace and common formatting
        sequence = re.sub(r'\s+', '', sequence)
        sequence = re.sub(r'[^A-Z]', '', sequence)
        
        # Length check
        if len(sequence) > self.config.max_sequence_length:
            raise ValueError(f"Sequence too long: {len(sequence)} > {self.config.max_sequence_length}")
        
        if len(sequence) == 0:
            raise ValueError("Empty sequence after sanitization")
        
        # Check for valid amino acids only
        if not self.amino_acid_pattern.match(sequence):
            invalid_chars = set(sequence) - self.config.allowed_amino_acids
            raise ValueError(f"Invalid amino acid codes: {invalid_chars}")
        
        # Check for malicious patterns
        self._check_malicious_patterns(sequence)
        
        return sequence
    
    def sanitize_sequences(self, sequences: List[str]) -> List[str]:
        """Sanitize a batch of sequences."""
        if len(sequences) > self.config.max_batch_size:
            raise ValueError(f"Batch too large: {len(sequences)} > {self.config.max_batch_size}")
        
        return [self.sanitize_sequence(seq) for seq in sequences]
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize a filename."""
        if not isinstance(filename, str):
            raise ValueError(f"Filename must be string, got {type(filename)}")
        
        # Remove path traversal attempts
        filename = filename.replace('..', '').replace('/', '').replace('\\', '')
        
        # Allow only alphanumeric, dash, underscore, and dot
        filename = re.sub(r'[^a-zA-Z0-9._-]', '', filename)
        
        if not filename:
            raise ValueError("Invalid filename after sanitization")
        
        # Check file extension
        file_path = Path(filename)
        if file_path.suffix.lower() not in self.config.allowed_file_extensions:
            raise ValueError(f"File extension not allowed: {file_path.suffix}")
        
        return filename
    
    def sanitize_text_input(self, text: str, max_length: int = 1000) -> str:
        """Sanitize general text input."""
        if not isinstance(text, str):
            raise ValueError(f"Text must be string, got {type(text)}")
        
        # Basic sanitization
        text = text.strip()
        
        if len(text) > max_length:
            raise ValueError(f"Text too long: {len(text)} > {max_length}")
        
        # Check for malicious patterns
        self._check_malicious_patterns(text)
        
        # Remove potentially dangerous characters
        text = re.sub(r'[<>&"\']', '', text)
        
        return text
    
    def _check_malicious_patterns(self, input_str: str):
        """Check for malicious patterns in input."""
        for pattern in self.malicious_patterns + self.sql_injection_patterns:
            if pattern.search(input_str):
                logger.warning(f"Potentially malicious pattern detected: {pattern.pattern}")
                if self.config.log_security_events:
                    self._log_security_event("malicious_pattern", {"pattern": pattern.pattern, "input": input_str[:100]})
                raise ValueError("Input contains potentially malicious content")
    
    def _log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security events."""
        logger.warning(f"SECURITY EVENT: {event_type} - {details}")

class RateLimiter:
    """Rate limiting implementation."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.request_history = defaultdict(deque)  # client_id -> deque of timestamps
        self.concurrent_requests = defaultdict(int)  # client_id -> count
        self._lock = threading.RLock()
    
    def is_allowed(self, client_id: str) -> Tuple[bool, str]:
        """
        Check if request is allowed for client.
        
        Args:
            client_id: Identifier for the client
            
        Returns:
            Tuple of (is_allowed, reason)
        """
        with self._lock:
            current_time = time.time()
            
            # Check concurrent requests
            if self.concurrent_requests[client_id] >= self.config.max_concurrent_requests:
                return False, f"Too many concurrent requests: {self.concurrent_requests[client_id]}"
            
            # Clean old requests
            self._clean_old_requests(client_id, current_time)
            
            # Check per-minute limit
            minute_requests = sum(1 for t in self.request_history[client_id] if current_time - t <= 60)
            if minute_requests >= self.config.max_requests_per_minute:
                return False, f"Rate limit exceeded: {minute_requests} requests per minute"
            
            # Check per-hour limit
            hour_requests = sum(1 for t in self.request_history[client_id] if current_time - t <= 3600)
            if hour_requests >= self.config.max_requests_per_hour:
                return False, f"Rate limit exceeded: {hour_requests} requests per hour"
            
            return True, "OK"
    
    def record_request(self, client_id: str):
        """Record a request from client."""
        with self._lock:
            current_time = time.time()
            self.request_history[client_id].append(current_time)
            self.concurrent_requests[client_id] += 1
    
    def release_request(self, client_id: str):
        """Release a concurrent request slot."""
        with self._lock:
            if self.concurrent_requests[client_id] > 0:
                self.concurrent_requests[client_id] -= 1
    
    def _clean_old_requests(self, client_id: str, current_time: float):
        """Remove old request timestamps."""
        # Keep only requests from the last hour
        cutoff_time = current_time - 3600
        history = self.request_history[client_id]
        
        while history and history[0] < cutoff_time:
            history.popleft()
    
    def get_stats(self, client_id: str) -> Dict[str, Any]:
        """Get rate limiting stats for client."""
        with self._lock:
            current_time = time.time()
            self._clean_old_requests(client_id, current_time)
            
            minute_requests = sum(1 for t in self.request_history[client_id] if current_time - t <= 60)
            hour_requests = len(self.request_history[client_id])
            
            return {
                "requests_last_minute": minute_requests,
                "requests_last_hour": hour_requests,
                "concurrent_requests": self.concurrent_requests[client_id],
                "limits": {
                    "max_per_minute": self.config.max_requests_per_minute,
                    "max_per_hour": self.config.max_requests_per_hour,
                    "max_concurrent": self.config.max_concurrent_requests,
                }
            }

def rate_limited(rate_limiter: RateLimiter, client_id_func: Optional[callable] = None):
    """Decorator for rate limiting function calls."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Determine client ID
            if client_id_func:
                client_id = client_id_func(*args, **kwargs)
            else:
                client_id = "default"
            
            # Check rate limit
            allowed, reason = rate_limiter.is_allowed(client_id)
            if not allowed:
                raise ValueError(f"Rate limit exceeded: {reason}")
            
            # Record request and execute
            rate_limiter.record_request(client_id)
            try:
                return func(*args, **kwargs)
            finally:
                rate_limiter.release_request(client_id)
        
        return wrapper
    return decorator

class FileValidator:
    """Validate uploaded files for security."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        
        # Magic bytes for common file types
        self.magic_bytes = {
            b'HEADER': 'pdb',  # PDB files often start with HEADER
            b'>': 'fasta',     # FASTA files start with >
            b'ATOM': 'pdb',    # PDB files contain ATOM records
        }
    
    def validate_file(self, file_path: Path, expected_type: Optional[str] = None) -> bool:
        """
        Validate a file for security and type.
        
        Args:
            file_path: Path to the file
            expected_type: Expected file type
            
        Returns:
            True if file is valid
            
        Raises:
            ValueError: If file is invalid or dangerous
        """
        if not file_path.exists():
            raise ValueError(f"File does not exist: {file_path}")
        
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.config.max_file_size_mb:
            raise ValueError(f"File too large: {file_size_mb:.1f}MB > {self.config.max_file_size_mb}MB")
        
        # Check file extension
        if file_path.suffix.lower() not in self.config.allowed_file_extensions:
            raise ValueError(f"File extension not allowed: {file_path.suffix}")
        
        # Check file content
        try:
            with open(file_path, 'rb') as f:
                header = f.read(1024)  # Read first 1KB
                
            # Basic content validation
            self._validate_file_content(header, file_path.suffix.lower())
            
        except Exception as e:
            raise ValueError(f"File content validation failed: {e}")
        
        return True
    
    def _validate_file_content(self, header: bytes, extension: str):
        """Validate file content based on extension."""
        try:
            # Convert to text for text files
            if extension in ['.fasta', '.txt']:
                header_text = header.decode('utf-8', errors='ignore')
                
                # Check for malicious content
                if any(pattern in header_text.lower() for pattern in ['<script', 'javascript:', 'eval(']):
                    raise ValueError("Potentially malicious content detected")
                
                # FASTA-specific validation
                if extension == '.fasta' and not header_text.lstrip().startswith('>'):
                    logger.warning("FASTA file doesn't start with '>'")
            
            elif extension == '.pdb':
                header_text = header.decode('utf-8', errors='ignore')
                
                # PDB files should contain specific keywords
                pdb_keywords = ['HEADER', 'ATOM', 'HETATM', 'REMARK']
                if not any(keyword in header_text for keyword in pdb_keywords):
                    logger.warning("PDB file doesn't contain expected keywords")
                
        except UnicodeDecodeError:
            # Binary files are suspicious for our use case
            raise ValueError("File contains binary data")

class SecurityManager:
    """Central security management."""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.sanitizer = InputSanitizer(self.config)
        self.rate_limiter = RateLimiter(self.config)
        self.file_validator = FileValidator(self.config)
        
        # Security event tracking
        self.security_events = deque(maxlen=1000)
        self._lock = threading.RLock()
    
    def validate_request(
        self,
        client_id: str,
        sequences: Optional[List[str]] = None,
        files: Optional[List[Path]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Comprehensive request validation.
        
        Args:
            client_id: Client identifier
            sequences: Protein sequences to validate
            files: Files to validate
            **kwargs: Additional parameters to validate
            
        Returns:
            Validation results
            
        Raises:
            ValueError: If request is invalid or not allowed
        """
        results = {
            "client_id": client_id,
            "timestamp": datetime.now().isoformat(),
            "validation_passed": True,
            "warnings": []
        }
        
        try:
            # Rate limiting check
            allowed, reason = self.rate_limiter.is_allowed(client_id)
            if not allowed:
                self._record_security_event("rate_limit_exceeded", {"client_id": client_id, "reason": reason})
                raise ValueError(f"Request not allowed: {reason}")
            
            # Validate sequences
            if sequences:
                sanitized_sequences = self.sanitizer.sanitize_sequences(sequences)
                results["sanitized_sequences"] = sanitized_sequences
                results["sequence_count"] = len(sanitized_sequences)
            
            # Validate files
            if files:
                for file_path in files:
                    self.file_validator.validate_file(file_path)
                results["validated_files"] = [str(f) for f in files]
            
            # Validate other parameters
            for key, value in kwargs.items():
                if isinstance(value, str) and len(value) > 0:
                    sanitized_value = self.sanitizer.sanitize_text_input(value)
                    results[f"sanitized_{key}"] = sanitized_value
            
            # Record successful request
            self.rate_limiter.record_request(client_id)
            
        except Exception as e:
            results["validation_passed"] = False
            results["error"] = str(e)
            self._record_security_event("validation_failed", {"client_id": client_id, "error": str(e)})
            raise
        
        return results
    
    def _record_security_event(self, event_type: str, details: Dict[str, Any]):
        """Record a security event."""
        with self._lock:
            event = {
                "timestamp": datetime.now().isoformat(),
                "type": event_type,
                "details": details
            }
            self.security_events.append(event)
            
            if self.config.log_security_events:
                logger.warning(f"SECURITY EVENT: {event_type} - {details}")
            
            if self.config.alert_on_suspicious_activity:
                self._check_suspicious_activity(event_type, details)
    
    def _check_suspicious_activity(self, event_type: str, details: Dict[str, Any]):
        """Check for patterns of suspicious activity."""
        client_id = details.get("client_id")
        if not client_id:
            return
        
        # Count recent events from this client
        current_time = datetime.now()
        recent_events = [
            event for event in self.security_events
            if event["details"].get("client_id") == client_id
            and datetime.fromisoformat(event["timestamp"]) > current_time - timedelta(minutes=10)
        ]
        
        # Alert if too many security events from same client
        if len(recent_events) >= 5:
            logger.critical(f"SUSPICIOUS ACTIVITY: Client {client_id} has {len(recent_events)} security events in 10 minutes")
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics."""
        with self._lock:
            event_types = defaultdict(int)
            for event in self.security_events:
                event_types[event["type"]] += 1
            
            return {
                "total_events": len(self.security_events),
                "event_types": dict(event_types),
                "rate_limiter_stats": {
                    "active_clients": len(self.rate_limiter.concurrent_requests),
                    "total_concurrent": sum(self.rate_limiter.concurrent_requests.values()),
                },
                "config": {
                    "max_requests_per_minute": self.config.max_requests_per_minute,
                    "max_requests_per_hour": self.config.max_requests_per_hour,
                    "max_sequence_length": self.config.max_sequence_length,
                    "max_batch_size": self.config.max_batch_size,
                }
            }
    
    def release_client_request(self, client_id: str):
        """Release a client's concurrent request slot."""
        self.rate_limiter.release_request(client_id)

def secure_operation(security_manager: SecurityManager, client_id_func: Optional[callable] = None):
    """Decorator for securing operations."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Determine client ID
            if client_id_func:
                client_id = client_id_func(*args, **kwargs)
            else:
                client_id = kwargs.get("client_id", "default")
            
            try:
                # Validate request
                validation_results = security_manager.validate_request(client_id, **kwargs)
                
                # Execute function
                result = func(*args, **kwargs)
                
                return result
            
            finally:
                # Always release the request slot
                security_manager.release_client_request(client_id)
        
        return wrapper
    return decorator

# Global security manager instance
_security_manager = None

def get_security_manager(config: Optional[SecurityConfig] = None) -> SecurityManager:
    """Get the global security manager instance."""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager(config)
    return _security_manager