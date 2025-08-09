#!/usr/bin/env python3
"""
Test just the config classes without full module imports.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_config_classes():
    """Test that config classes can be imported and created."""
    
    try:
        # Test diffusion transformer config
        from protein_diffusion.models import DiffusionTransformerConfig
        config = DiffusionTransformerConfig()
        print(f"✅ DiffusionTransformerConfig: {config.d_model}D, {config.num_layers} layers")
        
        # Test DDPM config
        from protein_diffusion.models import DDPMConfig
        ddpm_config = DDPMConfig()
        print(f"✅ DDPMConfig: {ddpm_config.timesteps} timesteps")
        
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tokenizer_standalone():
    """Test tokenizer in isolation."""
    try:
        from protein_diffusion.tokenization.selfies_tokenizer import TokenizerConfig
        config = TokenizerConfig()
        print(f"✅ TokenizerConfig: {config.vocab_size} vocab, {config.max_length} max length")
        
        # Try to create tokenizer
        from protein_diffusion.tokenization.selfies_tokenizer import SELFIESTokenizer
        tokenizer = SELFIESTokenizer(config)
        
        # Test basic functionality
        test_seq = "MAKLL"
        tokens = tokenizer.tokenize(test_seq)
        print(f"✅ Tokenization: '{test_seq}' -> {len(tokens)} tokens")
        
        return True
    except Exception as e:
        print(f"❌ Tokenizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_validation_standalone():
    """Test validation in isolation."""
    try:
        from protein_diffusion.validation import ValidationLevel, SequenceValidator
        
        validator = SequenceValidator(ValidationLevel.MODERATE)
        result = validator.validate_sequence("MAKLLILTCLVAVAL")
        
        print(f"✅ Validation: Valid sequence -> {result.is_valid}")
        
        # Test invalid sequence
        result = validator.validate_sequence("XYZ123")
        print(f"   - Invalid sequence -> Valid: {result.is_valid}, Errors: {len(result.errors)}")
        
        return True
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_security_standalone():
    """Test security in isolation."""
    try:
        from protein_diffusion.security import SecurityConfig, InputSanitizer
        
        config = SecurityConfig()
        sanitizer = InputSanitizer(config)
        
        # Test basic sanitization
        clean = sanitizer.sanitize_sequence("  MAKLL  ")
        print(f"✅ Security: Sanitized '  MAKLL  ' -> '{clean}'")
        
        # Test malicious input rejection
        try:
            sanitizer.sanitize_sequence("<script>alert('hack')</script>")
            print("❌ Should have rejected malicious input")
            return False
        except ValueError:
            print("   - Malicious input rejected ✓")
        
        return True
    except Exception as e:
        print(f"❌ Security test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run standalone tests."""
    print("🧬 Testing Protein Diffusion - Config Classes Only\n")
    
    tests = [
        ("Config Classes", test_config_classes),
        ("Tokenizer Standalone", test_tokenizer_standalone),
        ("Validation Standalone", test_validation_standalone),
        ("Security Standalone", test_security_standalone),
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"--- Testing {test_name} ---")
        if test_func():
            passed += 1
            print(f"✅ {test_name} passed\n")
        else:
            print(f"❌ {test_name} failed\n")
    
    print(f"🏆 Results: {passed}/{len(tests)} tests passed")
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)