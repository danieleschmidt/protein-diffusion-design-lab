#!/usr/bin/env python3
"""
Basic test to verify the core structure works.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that basic imports work."""
    try:
        # Test basic config classes
        from protein_diffusion.models import DiffusionTransformerConfig, DDPMConfig
        from protein_diffusion.tokenization.selfies_tokenizer import TokenizerConfig
        from protein_diffusion.folding.structure_predictor import StructurePredictorConfig
        from protein_diffusion.diffuser import ProteinDiffuserConfig
        from protein_diffusion.ranker import AffinityRankerConfig
        
        print("✅ All config classes import successfully")
        
        # Test that configs can be created
        model_config = DiffusionTransformerConfig()
        ddpm_config = DDPMConfig()
        tokenizer_config = TokenizerConfig()
        structure_config = StructurePredictorConfig()
        diffuser_config = ProteinDiffuserConfig()
        ranker_config = AffinityRankerConfig()
        
        print("✅ All config objects created successfully")
        print(f"   - Model config: {model_config.d_model} dimensions, {model_config.num_layers} layers")
        print(f"   - DDPM config: {ddpm_config.timesteps} timesteps")
        print(f"   - Tokenizer config: {tokenizer_config.vocab_size} vocab size")
        print(f"   - Structure config: {structure_config.method} method")
        print(f"   - Diffuser config: {diffuser_config.num_samples} samples")
        print(f"   - Ranker config: {ranker_config.max_results} max results")
        
        return True
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_tokenizer():
    """Test basic tokenizer functionality."""
    try:
        from protein_diffusion.tokenization.selfies_tokenizer import SELFIESTokenizer, TokenizerConfig
        
        # Create tokenizer
        config = TokenizerConfig()
        tokenizer = SELFIESTokenizer(config)
        
        # Test basic tokenization
        test_sequence = "MAKLLILTCLVAVAL"
        tokens = tokenizer.tokenize(test_sequence)
        
        print(f"✅ Tokenizer works: '{test_sequence}' -> {len(tokens)} tokens")
        print(f"   - Tokens: {tokens[:5]}{'...' if len(tokens) > 5 else ''}")
        
        # Test encoding/decoding
        encoding = tokenizer.encode(test_sequence, max_length=50)
        print(f"   - Encoded to {len(encoding['input_ids'])} token IDs")
        
        # Test decoding
        decoded = tokenizer.decode(encoding['input_ids'])
        print(f"   - Decoded back to: '{decoded}'")
        
        return True
    except Exception as e:
        print(f"❌ Tokenizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_validation():
    """Test validation functionality."""
    try:
        from protein_diffusion.validation import SequenceValidator, ValidationLevel
        
        validator = SequenceValidator(ValidationLevel.MODERATE)
        
        # Test valid sequence
        result = validator.validate_sequence("MAKLLILTCLVAVAL")
        print(f"✅ Validation works: Valid sequence -> {result.is_valid}")
        
        # Test invalid sequence
        result = validator.validate_sequence("XYZ123")
        print(f"   - Invalid sequence -> {result.is_valid}, errors: {len(result.errors)}")
        
        return True
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_security():
    """Test security functionality."""
    try:
        from protein_diffusion.security import InputSanitizer, SecurityConfig
        
        config = SecurityConfig()
        sanitizer = InputSanitizer(config)
        
        # Test sequence sanitization
        clean_seq = sanitizer.sanitize_sequence("  MAKLLILTCLVAVAL  ")
        print(f"✅ Security works: Sanitized sequence -> '{clean_seq}'")
        
        # Test malicious input detection
        try:
            sanitizer.sanitize_sequence("<script>alert('hack')</script>")
            print("❌ Security failed: Should have rejected malicious input")
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
    """Run all basic tests."""
    print("🧬 Testing Protein Diffusion Design Lab - Basic Functionality\n")
    
    tests = [
        ("Config Imports", test_imports),
        ("Tokenizer", test_tokenizer),
        ("Validation", test_validation),
        ("Security", test_security),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- Testing {test_name} ---")
        if test_func():
            passed += 1
            print(f"✅ {test_name} test passed")
        else:
            print(f"❌ {test_name} test failed")
    
    print(f"\n🏆 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic functionality tests passed!")
        print("✨ Ready for Generation 1 completion!")
        return True
    else:
        print("⚠️ Some tests failed - needs fixes before proceeding")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)