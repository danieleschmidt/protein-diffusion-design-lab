"""
Unit tests for security module.
"""

import pytest
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.protein_diffusion.security import (
    SecurityConfig,
    InputSanitizer,
    RateLimiter,
    FileValidator,
    SecurityManager,
    rate_limited,
    secure_operation,
    get_security_manager
)


class TestSecurityConfig:
    """Test SecurityConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SecurityConfig()
        assert config.max_requests_per_minute == 60
        assert config.max_requests_per_hour == 1000
        assert config.max_sequence_length == 2000
        assert config.require_authentication == False
        assert config.log_security_events == True


class TestInputSanitizer:
    """Test InputSanitizer."""
    
    @pytest.fixture
    def sanitizer(self):
        """Create an input sanitizer."""
        config = SecurityConfig()
        return InputSanitizer(config)
    
    def test_sanitize_valid_sequence(self, sanitizer):
        """Test sanitizing valid protein sequence."""
        sequence = "MKLLLAVAAAA"
        result = sanitizer.sanitize_sequence(sequence)
        assert result == sequence
    
    def test_sanitize_sequence_with_spaces(self, sanitizer):
        """Test sanitizing sequence with whitespace."""
        sequence = "MKL LLAVAA AA"
        result = sanitizer.sanitize_sequence(sequence)
        assert result == "MKLLLAVAAAA"
    
    def test_sanitize_lowercase_sequence(self, sanitizer):
        """Test sanitizing lowercase sequence."""
        sequence = "mklllavaaaa"
        result = sanitizer.sanitize_sequence(sequence)
        assert result == "MKLLLAVAAAA"
    
    def test_sanitize_sequence_non_string(self, sanitizer):
        """Test sanitizing non-string input."""
        with pytest.raises(ValueError, match="must be string"):
            sanitizer.sanitize_sequence(123)
    
    def test_sanitize_sequence_too_long(self, sanitizer):
        """Test sanitizing sequence that's too long."""
        sequence = "M" * (sanitizer.config.max_sequence_length + 1)
        with pytest.raises(ValueError, match="too long"):
            sanitizer.sanitize_sequence(sequence)
    
    def test_sanitize_empty_sequence(self, sanitizer):
        """Test sanitizing empty sequence."""
        with pytest.raises(ValueError, match="Empty sequence"):
            sanitizer.sanitize_sequence("")
    
    def test_sanitize_sequence_invalid_chars(self, sanitizer):
        """Test sanitizing sequence with invalid characters."""
        sequence = "MKLBZJ"  # B, Z, J are not standard amino acids
        with pytest.raises(ValueError, match="Invalid amino acid codes"):
            sanitizer.sanitize_sequence(sequence)
    
    def test_sanitize_sequences_batch(self, sanitizer):
        """Test batch sequence sanitization."""
        sequences = ["MKLL", "AVAA", "PPKK"]
        results = sanitizer.sanitize_sequences(sequences)
        assert results == sequences
    
    def test_sanitize_sequences_batch_too_large(self, sanitizer):
        """Test batch that's too large."""
        sequences = ["M"] * (sanitizer.config.max_batch_size + 1)
        with pytest.raises(ValueError, match="Batch too large"):
            sanitizer.sanitize_sequences(sequences)
    
    def test_sanitize_filename_valid(self, sanitizer):
        """Test sanitizing valid filename."""
        filename = "protein_structure.pdb"
        result = sanitizer.sanitize_filename(filename)
        assert result == filename
    
    def test_sanitize_filename_path_traversal(self, sanitizer):
        """Test sanitizing filename with path traversal."""
        filename = "../../../etc/passwd"
        result = sanitizer.sanitize_filename(filename)
        assert ".." not in result
        assert "/" not in result
    
    def test_sanitize_filename_invalid_extension(self, sanitizer):
        """Test sanitizing filename with invalid extension."""
        filename = "malicious_script.py"
        with pytest.raises(ValueError, match="extension not allowed"):
            sanitizer.sanitize_filename(filename)
    
    def test_sanitize_text_input_valid(self, sanitizer):
        """Test sanitizing valid text input."""
        text = "This is a valid protein description"
        result = sanitizer.sanitize_text_input(text)
        assert result == text
    
    def test_sanitize_text_input_too_long(self, sanitizer):
        """Test sanitizing text that's too long."""
        text = "A" * 1001  # Default max is 1000
        with pytest.raises(ValueError, match="too long"):
            sanitizer.sanitize_text_input(text)
    
    def test_sanitize_text_input_malicious_content(self, sanitizer):
        """Test sanitizing text with potentially malicious content."""
        malicious_texts = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "eval(malicious_code)",
            "os.system('rm -rf /')"
        ]
        
        for text in malicious_texts:
            with pytest.raises(ValueError, match="malicious content"):
                sanitizer.sanitize_text_input(text)


class TestRateLimiter:
    """Test RateLimiter."""
    
    @pytest.fixture
    def rate_limiter(self):
        """Create a rate limiter."""
        config = SecurityConfig()
        config.max_requests_per_minute = 5
        config.max_requests_per_hour = 50
        config.max_concurrent_requests = 2
        return RateLimiter(config)
    
    def test_is_allowed_first_request(self, rate_limiter):
        """Test first request is allowed."""
        allowed, reason = rate_limiter.is_allowed("client1")
        assert allowed == True
        assert reason == "OK"
    
    def test_record_and_release_request(self, rate_limiter):
        """Test recording and releasing requests."""
        client_id = "client1"
        
        # Record request
        rate_limiter.record_request(client_id)
        assert rate_limiter.concurrent_requests[client_id] == 1
        
        # Release request
        rate_limiter.release_request(client_id)
        assert rate_limiter.concurrent_requests[client_id] == 0
    
    def test_concurrent_limit(self, rate_limiter):
        """Test concurrent request limit."""
        client_id = "client1"
        
        # Fill up concurrent slots
        for _ in range(rate_limiter.config.max_concurrent_requests):
            rate_limiter.record_request(client_id)
        
        # Next request should be denied
        allowed, reason = rate_limiter.is_allowed(client_id)
        assert allowed == False
        assert "concurrent" in reason.lower()
    
    def test_minute_rate_limit(self, rate_limiter):
        """Test per-minute rate limit."""
        client_id = "client1"
        
        # Use up all requests in the minute
        for _ in range(rate_limiter.config.max_requests_per_minute):
            allowed, _ = rate_limiter.is_allowed(client_id)
            if allowed:
                rate_limiter.record_request(client_id)
                rate_limiter.release_request(client_id)
        
        # Next request should be denied
        allowed, reason = rate_limiter.is_allowed(client_id)
        assert allowed == False
        assert "per minute" in reason.lower()
    
    def test_get_stats(self, rate_limiter):
        """Test getting rate limiter statistics."""
        client_id = "client1"
        
        # Make some requests
        for _ in range(3):
            rate_limiter.record_request(client_id)
            rate_limiter.release_request(client_id)
        
        stats = rate_limiter.get_stats(client_id)
        assert "requests_last_minute" in stats
        assert "requests_last_hour" in stats
        assert "concurrent_requests" in stats
        assert "limits" in stats


class TestFileValidator:
    """Test FileValidator."""
    
    @pytest.fixture
    def file_validator(self):
        """Create a file validator."""
        config = SecurityConfig()
        return FileValidator(config)
    
    def test_validate_nonexistent_file(self, file_validator):
        """Test validating non-existent file."""
        fake_path = Path("/nonexistent/file.pdb")
        with pytest.raises(ValueError, match="does not exist"):
            file_validator.validate_file(fake_path)
    
    def test_validate_valid_pdb_file(self, file_validator):
        """Test validating a valid PDB file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write("HEADER    TEST PROTEIN\nATOM      1  N   ALA A   1      20.154  16.967  23.000  1.00 20.00           N\n")
            f.flush()
            
            try:
                result = file_validator.validate_file(Path(f.name))
                assert result == True
            finally:
                Path(f.name).unlink()
    
    def test_validate_valid_fasta_file(self, file_validator):
        """Test validating a valid FASTA file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(">test_protein\nMKLLLAVAAAAPPPKKK\n")
            f.flush()
            
            try:
                result = file_validator.validate_file(Path(f.name))
                assert result == True
            finally:
                Path(f.name).unlink()
    
    def test_validate_file_wrong_extension(self, file_validator):
        """Test validating file with wrong extension."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.exe', delete=False) as f:
            f.write("some content")
            f.flush()
            
            try:
                with pytest.raises(ValueError, match="extension not allowed"):
                    file_validator.validate_file(Path(f.name))
            finally:
                Path(f.name).unlink()
    
    def test_validate_file_too_large(self, file_validator):
        """Test validating file that's too large."""
        # Temporarily reduce max file size for testing
        original_max_size = file_validator.config.max_file_size_mb
        file_validator.config.max_file_size_mb = 0.001  # 1KB
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
                f.write("X" * 2000)  # 2KB file
                f.flush()
                
                try:
                    with pytest.raises(ValueError, match="too large"):
                        file_validator.validate_file(Path(f.name))
                finally:
                    Path(f.name).unlink()
        finally:
            file_validator.config.max_file_size_mb = original_max_size


class TestSecurityManager:
    """Test SecurityManager."""
    
    @pytest.fixture
    def security_manager(self):
        """Create a security manager."""
        config = SecurityConfig()
        config.max_requests_per_minute = 10
        return SecurityManager(config)
    
    def test_initialization(self, security_manager):
        """Test SecurityManager initialization."""
        assert hasattr(security_manager, 'sanitizer')
        assert hasattr(security_manager, 'rate_limiter')
        assert hasattr(security_manager, 'file_validator')
    
    def test_validate_request_valid(self, security_manager):
        """Test validating a valid request."""
        result = security_manager.validate_request(
            client_id="client1",
            sequences=["MKLL", "AVAA"]
        )
        
        assert result["validation_passed"] == True
        assert "sanitized_sequences" in result
        assert result["sequence_count"] == 2
    
    def test_validate_request_invalid_sequence(self, security_manager):
        """Test validating request with invalid sequence."""
        with pytest.raises(ValueError):
            security_manager.validate_request(
                client_id="client1",
                sequences=[""]  # Empty sequence
            )
    
    def test_validate_request_rate_limited(self, security_manager):
        """Test rate limiting."""
        client_id = "client1"
        
        # Use up rate limit
        for _ in range(security_manager.config.max_requests_per_minute + 1):
            try:
                security_manager.validate_request(
                    client_id=client_id,
                    sequences=["MKLL"]
                )
            except ValueError as e:
                if "rate limit" in str(e).lower():
                    break
        else:
            pytest.fail("Rate limiting should have been triggered")
    
    def test_get_security_stats(self, security_manager):
        """Test getting security statistics."""
        # Make a request to generate some data
        try:
            security_manager.validate_request(
                client_id="client1",
                sequences=["MKLL"]
            )
        except:
            pass  # Ignore any errors for stats test
        
        stats = security_manager.get_security_stats()
        assert "total_events" in stats
        assert "rate_limiter_stats" in stats
        assert "config" in stats


class TestDecorators:
    """Test security decorators."""
    
    def test_rate_limited_decorator(self):
        """Test rate_limited decorator."""
        config = SecurityConfig()
        config.max_requests_per_minute = 2
        config.max_concurrent_requests = 1
        rate_limiter = RateLimiter(config)
        
        @rate_limited(rate_limiter)
        def test_function():
            return "success"
        
        # First few calls should succeed
        for _ in range(2):
            result = test_function()
            assert result == "success"
        
        # Next call should fail due to rate limiting
        with pytest.raises(ValueError, match="rate limit"):
            test_function()
    
    def test_secure_operation_decorator(self):
        """Test secure_operation decorator."""
        security_manager = SecurityManager()
        
        @secure_operation(security_manager)
        def test_function(sequences=None):
            return "success"
        
        # Valid call should succeed
        result = test_function(sequences=["MKLL"])
        assert result == "success"
        
        # Invalid call should fail
        with pytest.raises(ValueError):
            test_function(sequences=[""])


class TestThreadSafety:
    """Test thread safety of security components."""
    
    def test_rate_limiter_thread_safety(self):
        """Test rate limiter thread safety."""
        config = SecurityConfig()
        config.max_requests_per_minute = 100
        rate_limiter = RateLimiter(config)
        
        results = []
        errors = []
        
        def worker():
            try:
                for _ in range(10):
                    allowed, _ = rate_limiter.is_allowed("client1")
                    if allowed:
                        rate_limiter.record_request("client1")
                        time.sleep(0.001)  # Simulate work
                        rate_limiter.release_request("client1")
                    results.append(allowed)
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for _ in range(5):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) > 0
        assert any(results)  # At least some requests should be allowed
    
    def test_security_manager_thread_safety(self):
        """Test security manager thread safety."""
        security_manager = SecurityManager()
        
        results = []
        errors = []
        
        def worker(client_id):
            try:
                for i in range(5):
                    result = security_manager.validate_request(
                        client_id=f"{client_id}_{i}",
                        sequences=["MKLL"]
                    )
                    results.append(result["validation_passed"])
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for i in range(3):
            t = threading.Thread(target=worker, args=(f"client{i}",))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Check results - some requests should succeed
        assert len(results) > 0
        assert any(results)


class TestGlobalInstances:
    """Test global instance management."""
    
    def test_get_security_manager_singleton(self):
        """Test that get_security_manager returns singleton."""
        manager1 = get_security_manager()
        manager2 = get_security_manager()
        
        # Should be the same instance
        assert manager1 is manager2
    
    def test_get_security_manager_with_config(self):
        """Test get_security_manager with custom config."""
        # Reset global instance
        import src.protein_diffusion.security as security_module
        security_module._security_manager = None
        
        config = SecurityConfig()
        config.max_requests_per_minute = 123
        
        manager = get_security_manager(config)
        assert manager.config.max_requests_per_minute == 123