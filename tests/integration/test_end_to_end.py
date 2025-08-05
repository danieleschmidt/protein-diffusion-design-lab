"""
End-to-end integration tests for the protein diffusion system.

This module tests the complete pipeline from sequence generation
through structure prediction and ranking.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, Mock
import torch

from src.protein_diffusion.diffuser import ProteinDiffuser, ProteinDiffuserConfig  
from src.protein_diffusion.ranker import AffinityRanker, AffinityRankerConfig
from src.protein_diffusion.folding import StructurePredictor, StructurePredictorConfig
from src.protein_diffusion.tokenization import SELFIESTokenizer, TokenizerConfig
from src.protein_diffusion.validation import ValidationManager, ValidationLevel


class TestEndToEndPipeline:
    """Test the complete protein generation and analysis pipeline."""
    
    @pytest.fixture
    def sample_sequences(self):
        """Sample protein sequences for testing."""
        return [
            "MKLLILTCLVAVALARPKHPIPWDQAITVAYASRALGRGLVVMAQDGNRGGKFHPWTVNQGPLKDYICQAYDMGTTTEVPGTMGMLRRRSNVWSCLPRLLCERVAAPNLDPEGFVVAVPIPVYEAWDFGDPKLNLRQNTVAVTCTGVQTLAVRGRVGNLLSNGVPIGRGLPHIPSKGSGATFEFIGSDLKAELATDQAGVLQVDVQQVEACWFASQGGGVDTDYTGQPWDGGKPTVTGAMCGAFSCRHDGKRDVRVGTAAGVGGGYCSDGDGPVKPVVSNPNQALAFGLSEAGSRRLHPFTTARQGAGSM",
            "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWUACFAVEFVDQYQDWLNR"[:200],  # Truncated for testing
            "MAQEHFYLVDQSTTGHLAVVPWLAFLHYRQVTAASLALALEHAATLHGYDTSVALTDAVATAQHPAGEALVFLRSLLLHLFERALGRPVDLPPRAAHDWLLTRLRLAELLWQARRAGRRLSEWLLTRALLRLRYLRLASLLLQL"[:150],  # Truncated
        ]
    
    @pytest.fixture
    def mock_esm_available(self):
        """Mock ESM availability for testing."""
        with patch('src.protein_diffusion.tokenization.protein_embeddings.ESM_AVAILABLE', False):
            with patch('src.protein_diffusion.folding.structure_predictor.ESM_FOLD_AVAILABLE', False):
                yield
    
    def test_sequence_generation_pipeline(self, mock_esm_available):
        """Test the complete sequence generation pipeline."""
        # Configure for testing
        config = ProteinDiffuserConfig()
        config.device = "cpu"
        config.dtype = torch.float32
        config.num_samples = 3
        config.max_length = 100
        
        with patch.multiple(
            'src.protein_diffusion.diffuser',
            DiffusionTransformer=Mock(return_value=self._create_mock_model()),
            DDPM=Mock(return_value=self._create_mock_ddpm()),
            SELFIESTokenizer=Mock(return_value=self._create_mock_tokenizer()),
            ProteinEmbeddings=Mock(return_value=self._create_mock_embeddings()),
            StructurePredictor=Mock(return_value=Mock())
        ):
            # Create diffuser
            diffuser = ProteinDiffuser(config)
            
            # Generate sequences
            results = diffuser.generate(
                motif="HELIX_SHEET_HELIX",
                num_samples=3,
                temperature=1.0
            )
            
            # Verify results
            assert len(results) == 3
            for i, result in enumerate(results):
                assert "sequence" in result
                assert "confidence" in result
                assert "length" in result
                assert "motif" in result
                assert "sample_id" in result
                
                assert result["motif"] == "HELIX_SHEET_HELIX"
                assert result["sample_id"] == i
                assert isinstance(result["sequence"], str)
                assert isinstance(result["confidence"], float)
                assert 0 <= result["confidence"] <= 1
    
    def test_ranking_pipeline(self, sample_sequences, mock_esm_available):
        """Test the complete ranking pipeline."""
        # Configure ranking
        config = AffinityRankerConfig()
        config.max_results = len(sample_sequences)
        
        with patch('src.protein_diffusion.folding.structure_predictor.BIOPYTHON_AVAILABLE', False):
            # Create ranker with mocked structure predictor
            ranker = AffinityRanker(config)
            ranker.structure_predictor = self._create_mock_structure_predictor()
            
            # Rank sequences
            results = ranker.rank(sample_sequences, return_detailed=True)
            
            # Verify results
            assert len(results) == len(sample_sequences)
            
            # Check that results are sorted by composite score
            scores = [r["composite_score"] for r in results]
            assert scores == sorted(scores, reverse=True)
            
            # Verify required fields
            for result in results:
                assert "sequence" in result
                assert "composite_score" in result
                assert "structure_quality" in result
                assert "confidence" in result
                assert "binding_affinity" in result
                assert "novelty_score" in result
                assert "diversity_score" in result
    
    def test_structure_prediction_pipeline(self, sample_sequences, mock_esm_available):
        """Test structure prediction pipeline."""
        config = StructurePredictorConfig()
        config.method = "esmfold"  # Will be mocked
        
        with patch('src.protein_diffusion.folding.structure_predictor.ESMFoldPredictor') as mock_predictor_class:
            # Create mock predictor
            mock_predictor = Mock()
            mock_predictor.predict.return_value = {
                "pdb_string": "HEADER    TEST\nATOM      1  N   ALA A   1      20.154  21.875  22.101  1.00 10.00           N",
                "confidence": 0.85,
                "plddt_scores": [0.8, 0.9, 0.7],
            }
            mock_predictor_class.return_value = mock_predictor
            
            # Create structure predictor
            predictor = StructurePredictor(config)
            
            # Test single prediction
            result = predictor.predict_structure(sample_sequences[0])
            
            # Verify result
            assert "pdb_string" in result
            assert "confidence" in result
            assert "structure_quality" in result
            assert "prediction_method" in result
            
            assert result["prediction_method"] == "esmfold"
            assert isinstance(result["confidence"], float)
            assert 0 <= result["confidence"] <= 1
    
    def test_tokenization_pipeline(self):
        """Test tokenization pipeline."""
        config = TokenizerConfig()
        tokenizer = SELFIESTokenizer(config)
        
        # Test sequence tokenization
        test_sequence = "MKLLILTCLVAVALARPKHPIP"
        
        # Test encoding
        encoded = tokenizer.encode(test_sequence, max_length=50)
        
        assert "input_ids" in encoded
        assert "attention_mask" in encoded
        assert len(encoded["input_ids"]) == 50  # Padded to max_length
        assert len(encoded["attention_mask"]) == 50
        
        # Test decoding
        decoded = tokenizer.decode(encoded["input_ids"])
        assert isinstance(decoded, str)
        assert len(decoded) > 0
        
        # Test batch encoding
        sequences = ["MKLLILTCLVAVALARPKHPIP", "AEAAAKEAAAKA", "IVIVIV"]
        batch_encoded = tokenizer.batch_encode(sequences)
        
        assert "input_ids" in batch_encoded
        assert "attention_mask" in batch_encoded
        assert len(batch_encoded["input_ids"]) == len(sequences)
    
    def test_validation_pipeline(self, sample_sequences):
        """Test validation pipeline."""
        validator = ValidationManager(ValidationLevel.MODERATE)
        
        # Test sequence validation
        valid_sequences, invalid_sequences = validator.sequence_validator.filter_valid_sequences(
            sample_sequences + ["INVALID_SEQUENCE_WITH_123", ""]
        )
        
        # Should have filtered out invalid sequences
        assert len(valid_sequences) == len(sample_sequences)
        assert len(invalid_sequences) == 2
        
        # Test comprehensive validation
        result = validator.comprehensive_validation(
            sequences=sample_sequences[:2],
            generation_params={
                "num_samples": 10,
                "max_length": 256,
                "temperature": 1.0,
                "guidance_scale": 1.0
            }
        )
        
        assert result.is_valid
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)
    
    def test_complete_workflow(self, mock_esm_available):
        """Test complete workflow from generation to ranking."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Step 1: Generate sequences
            diffuser_config = ProteinDiffuserConfig()
            diffuser_config.device = "cpu"
            diffuser_config.dtype = torch.float32
            diffuser_config.num_samples = 5
            
            with patch.multiple(
                'src.protein_diffusion.diffuser',
                DiffusionTransformer=Mock(return_value=self._create_mock_model()),
                DDPM=Mock(return_value=self._create_mock_ddpm()),
                SELFIESTokenizer=Mock(return_value=self._create_mock_tokenizer()),
                ProteinEmbeddings=Mock(return_value=self._create_mock_embeddings()),
                StructurePredictor=Mock(return_value=Mock())
            ):
                diffuser = ProteinDiffuser(diffuser_config)
                generated_results = diffuser.generate(motif="HELIX")
            
            # Extract sequences
            sequences = [r["sequence"] for r in generated_results]
            
            # Step 2: Rank sequences
            ranker_config = AffinityRankerConfig()
            ranker_config.max_results = len(sequences)
            
            with patch('src.protein_diffusion.folding.structure_predictor.BIOPYTHON_AVAILABLE', False):
                ranker = AffinityRanker(ranker_config)
                ranker.structure_predictor = self._create_mock_structure_predictor()
                ranked_results = ranker.rank(sequences)
            
            # Step 3: Verify complete workflow
            assert len(ranked_results) == len(sequences)
            
            # Verify ranking order
            scores = [r["composite_score"] for r in ranked_results]
            assert scores == sorted(scores, reverse=True)
            
            # Step 4: Save and verify results
            results_file = Path(temp_dir) / "workflow_results.json"
            with open(results_file, 'w') as f:
                json.dump({
                    "generated": generated_results,
                    "ranked": ranked_results
                }, f, indent=2, default=str)
            
            assert results_file.exists()
            
            # Verify saved data
            with open(results_file, 'r') as f:
                saved_data = json.load(f)
            
            assert "generated" in saved_data
            assert "ranked" in saved_data
            assert len(saved_data["generated"]) == 5
            assert len(saved_data["ranked"]) == 5
    
    def test_error_handling_pipeline(self, mock_esm_available):
        """Test error handling throughout the pipeline."""
        # Test with invalid configuration
        invalid_config = ProteinDiffuserConfig()
        invalid_config.num_samples = -1  # Invalid
        
        with patch.multiple(
            'src.protein_diffusion.diffuser',
            DiffusionTransformer=Mock(return_value=self._create_mock_model()),
            DDPM=Mock(return_value=self._create_mock_ddpm()),
            SELFIESTokenizer=Mock(return_value=self._create_mock_tokenizer()),
            ProteinEmbeddings=Mock(return_value=self._create_mock_embeddings()),
            StructurePredictor=Mock(return_value=Mock())
        ):
            diffuser = ProteinDiffuser(invalid_config)
            
            # Should handle invalid parameters gracefully
            results = diffuser.generate(num_samples=1)  # Override invalid config
            assert len(results) == 1
    
    def test_performance_characteristics(self, sample_sequences, mock_esm_available):
        """Test performance characteristics of the pipeline."""
        import time
        
        # Test ranking performance with different batch sizes
        config = AffinityRankerConfig()
        
        with patch('src.protein_diffusion.folding.structure_predictor.BIOPYTHON_AVAILABLE', False):
            ranker = AffinityRanker(config)
            ranker.structure_predictor = self._create_mock_structure_predictor()
            
            # Small batch
            start_time = time.time()
            results_small = ranker.rank(sample_sequences[:1])
            small_time = time.time() - start_time
            
            # Larger batch
            start_time = time.time()
            results_large = ranker.rank(sample_sequences)
            large_time = time.time() - start_time
            
            # Verify results
            assert len(results_small) == 1
            assert len(results_large) == len(sample_sequences)
            
            # Performance should scale reasonably
            assert large_time < small_time * len(sample_sequences) * 2  # Allow some overhead
    
    # Helper methods for creating mocks
    
    def _create_mock_model(self):
        """Create a mock transformer model."""
        model = Mock()
        model.config = Mock()
        model.config.vocab_size = 32
        model.eval.return_value = model
        model.to.return_value = model
        model.get_num_parameters.return_value = 1000000
        
        def mock_forward(*args, **kwargs):
            batch_size, seq_len = args[0].shape
            vocab_size = 32
            return torch.randn(batch_size, seq_len, vocab_size)
        
        model.forward = mock_forward
        model.__call__ = mock_forward
        return model
    
    def _create_mock_ddpm(self):
        """Create a mock DDPM."""
        ddpm = Mock()
        ddpm.to.return_value = ddpm
        
        def mock_sample(shape, device, **kwargs):
            batch_size, seq_len = shape
            vocab_size = 32
            return torch.randn(batch_size, seq_len, vocab_size)
        
        ddpm.sample = mock_sample
        ddpm.ddim_sample = mock_sample
        return ddpm
    
    def _create_mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.__len__.return_value = 32
        
        # Mock different sequences for variety
        sequences = [
            "MKLLILTCLVAVALARPKHPIP",
            "AEAAAKEAAAAKALIVFGDS",
            "IVIVIPGGSGGSDWERFVTQ"
        ]
        sequence_iter = iter(sequences * 10)  # Cycle through sequences
        
        def mock_decode(*args, **kwargs):
            try:
                return next(sequence_iter)
            except StopIteration:
                return "MKLLILTCLVAVALARPKHPIP"  # Fallback
        
        tokenizer.decode = mock_decode
        return tokenizer
    
    def _create_mock_embeddings(self):
        """Create mock embeddings."""
        embeddings = Mock()
        embeddings.to.return_value = embeddings
        embeddings.eval.return_value = embeddings
        
        def mock_forward(sequences):
            batch_size = len(sequences) if isinstance(sequences, list) else 1
            seq_len = 256
            d_model = 1024
            return torch.randn(batch_size, seq_len, d_model)
        
        embeddings.__call__ = mock_forward
        return embeddings
    
    def _create_mock_structure_predictor(self):
        """Create mock structure predictor."""
        predictor = Mock()
        
        def mock_predict_structure(sequence):
            import random
            return {
                "structure_quality": random.uniform(0.6, 0.95),
                "confidence": random.uniform(0.7, 0.9),
                "ramachandran_score": random.uniform(0.8, 0.95),
                "clash_score": random.uniform(0.0, 0.1),
                "compactness_score": random.uniform(0.7, 0.9),
                "pdb_string": "HEADER    MOCK\nATOM      1  N   ALA A   1      20.154  21.875  22.101  1.00 10.00           N"
            }
        
        def mock_evaluate_binding(sequence, target_pdb):
            import random
            return {
                "binding_affinity": random.uniform(-15.0, -5.0),
                "binding_method": "mock_autodock"
            }
        
        predictor.predict_structure = mock_predict_structure
        predictor.evaluate_binding = mock_evaluate_binding
        return predictor


class TestCLIIntegration:
    """Test integration with CLI interface."""
    
    def test_cli_generate_command(self):
        """Test CLI generate command integration."""
        from src.protein_diffusion.cli import generate_command
        from argparse import Namespace
        
        # Mock arguments
        args = Namespace(
            config=None,
            checkpoint=None,
            motif="HELIX",
            num_samples=2,
            max_length=100,
            temperature=1.0,
            sampling_method="ddpm",
            ddim_steps=50,
            device="cpu",
            output="./test_output",
            quiet=True,
            verbose=False
        )
        
        with patch.multiple(
            'src.protein_diffusion.cli',
            ProteinDiffuser=Mock(return_value=self._create_mock_diffuser()),
            Path=Mock()
        ):
            # Mock Path operations
            mock_path = Mock()
            mock_path.mkdir = Mock()
            mock_path.__truediv__ = Mock(return_value=mock_path)
            
            with patch('src.protein_diffusion.cli.Path', return_value=mock_path):
                with patch('builtins.open', create=True) as mock_open:
                    mock_file = Mock()
                    mock_open.return_value.__enter__.return_value = mock_file
                    
                    # This should not raise an exception
                    result = generate_command(args)
                    
                    # CLI should return 0 for success
                    assert result == 0
    
    def _create_mock_diffuser(self):
        """Create mock diffuser for CLI testing."""
        diffuser = Mock()
        diffuser.generate.return_value = [
            {
                "sequence": "MKLLILTCLVAVALARPKHPIP",
                "confidence": 0.85,
                "length": 21,
                "sample_id": 0
            },
            {
                "sequence": "AEAAAKEAAAAKALIVFGDS",
                "confidence": 0.78,
                "length": 20,
                "sample_id": 1
            }
        ]
        return diffuser


if __name__ == "__main__":
    pytest.main([__file__])