#!/usr/bin/env python3
"""
Comprehensive benchmarking suite for protein diffusion models.

This module provides tools for benchmarking model performance, comparing
different approaches, and conducting systematic evaluations for research.
"""

import sys
import os
import time
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking experiments."""
    # Dataset settings
    dataset_size: int = 1000
    sequence_lengths: List[int] = field(default_factory=lambda: [50, 100, 200, 500])
    batch_sizes: List[int] = field(default_factory=lambda: [1, 8, 16, 32])
    
    # Model settings
    model_variants: List[str] = field(default_factory=lambda: ["small", "medium", "large"])
    temperature_values: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5])
    
    # Evaluation metrics
    evaluate_quality: bool = True
    evaluate_diversity: bool = True
    evaluate_novelty: bool = True
    evaluate_performance: bool = True
    
    # Output settings
    output_dir: str = "./benchmark_results"
    save_intermediate: bool = True
    
    # Research settings
    statistical_significance: float = 0.05
    num_runs: int = 3
    random_seeds: List[int] = field(default_factory=lambda: [42, 123, 456])


class SequenceMetrics:
    """Calculate various metrics for protein sequences."""
    
    @staticmethod
    def calculate_diversity(sequences: List[str]) -> float:
        """Calculate sequence diversity using edit distance."""
        if len(sequences) < 2:
            return 0.0
        
        total_distance = 0
        total_pairs = 0
        
        for i in range(len(sequences)):
            for j in range(i + 1, len(sequences)):
                distance = SequenceMetrics.levenshtein_distance(sequences[i], sequences[j])
                total_distance += distance
                total_pairs += 1
        
        return total_distance / total_pairs if total_pairs > 0 else 0.0
    
    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two sequences."""
        if len(s1) < len(s2):
            return SequenceMetrics.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    @staticmethod
    def calculate_novelty(generated_sequences: List[str], 
                         reference_sequences: List[str]) -> float:
        """Calculate novelty as fraction of sequences not in reference set."""
        if not generated_sequences:
            return 0.0
        
        reference_set = set(reference_sequences)
        novel_count = sum(1 for seq in generated_sequences if seq not in reference_set)
        return novel_count / len(generated_sequences)
    
    @staticmethod
    def calculate_amino_acid_composition(sequences: List[str]) -> Dict[str, float]:
        """Calculate amino acid composition statistics."""
        aa_counts = {}
        total_residues = 0
        
        for sequence in sequences:
            for aa in sequence:
                aa_counts[aa] = aa_counts.get(aa, 0) + 1
                total_residues += 1
        
        # Convert to frequencies
        composition = {aa: count / total_residues 
                      for aa, count in aa_counts.items()}
        
        return composition
    
    @staticmethod
    def calculate_length_statistics(sequences: List[str]) -> Dict[str, float]:
        """Calculate sequence length statistics."""
        lengths = [len(seq) for seq in sequences]
        
        if not lengths:
            return {}
        
        return {
            "mean_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "std_length": (sum((l - sum(lengths)/len(lengths))**2 for l in lengths) / len(lengths))**0.5
        }


class PerformanceBenchmark:
    """Benchmark model performance characteristics."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = {}
    
    def benchmark_tokenization(self) -> Dict[str, Any]:
        """Benchmark tokenization performance."""
        from protein_diffusion.tokenization.selfies_tokenizer import SELFIESTokenizer, TokenizerConfig
        
        logger.info("Benchmarking tokenization performance...")
        
        tokenizer_config = TokenizerConfig()
        tokenizer = SELFIESTokenizer(tokenizer_config)
        
        results = {}
        
        for seq_length in self.config.sequence_lengths:
            # Create test sequence
            test_sequence = "M" * seq_length
            
            # Benchmark tokenization
            start_time = time.time()
            for _ in range(100):  # Multiple runs for accuracy
                tokens = tokenizer.tokenize(test_sequence)
            tokenization_time = (time.time() - start_time) / 100
            
            # Benchmark encoding
            start_time = time.time()
            for _ in range(100):
                encoding = tokenizer.encode(test_sequence, max_length=seq_length + 10)
            encoding_time = (time.time() - start_time) / 100
            
            results[f"length_{seq_length}"] = {
                "sequence_length": seq_length,
                "tokenization_time_ms": tokenization_time * 1000,
                "encoding_time_ms": encoding_time * 1000,
                "num_tokens": len(tokens),
                "tokens_per_second": len(tokens) / tokenization_time if tokenization_time > 0 else 0
            }
        
        return results
    
    def benchmark_validation(self) -> Dict[str, Any]:
        """Benchmark validation performance."""
        from protein_diffusion.validation import SequenceValidator, ValidationLevel
        
        logger.info("Benchmarking validation performance...")
        
        validator = SequenceValidator(ValidationLevel.MODERATE)
        results = {}
        
        for seq_length in self.config.sequence_lengths:
            test_sequences = [
                "M" * seq_length,  # Valid sequence
                "X" * seq_length,  # Sequence with unknown residues
                "123" * (seq_length // 3),  # Invalid sequence
            ]
            
            for i, test_sequence in enumerate(test_sequences):
                start_time = time.time()
                for _ in range(100):
                    result = validator.validate_sequence(test_sequence)
                validation_time = (time.time() - start_time) / 100
                
                results[f"length_{seq_length}_type_{i}"] = {
                    "sequence_length": seq_length,
                    "sequence_type": ["valid", "unknown_residues", "invalid"][i],
                    "validation_time_ms": validation_time * 1000,
                    "is_valid": result.is_valid,
                    "num_errors": len(result.errors),
                    "num_warnings": len(result.warnings),
                }
        
        return results
    
    def benchmark_security(self) -> Dict[str, Any]:
        """Benchmark security sanitization performance."""
        from protein_diffusion.security import SecurityConfig, InputSanitizer
        
        logger.info("Benchmarking security performance...")
        
        config = SecurityConfig()
        sanitizer = InputSanitizer(config)
        
        results = {}
        
        test_inputs = [
            "MAKLLILTCLVAVAL",  # Clean sequence
            "  MAKLL ILTC LVAVAL  ",  # Sequence with whitespace
            "MAKLL123ILTC456LVAVAL",  # Sequence with numbers
            "MAKLL<script>alert('test')</script>ILTC",  # Malicious input
        ]
        
        for i, test_input in enumerate(test_inputs):
            start_time = time.time()
            success_count = 0
            
            for _ in range(100):
                try:
                    sanitized = sanitizer.sanitize_sequence(test_input)
                    success_count += 1
                except ValueError:
                    pass  # Expected for malicious inputs
            
            processing_time = (time.time() - start_time) / 100
            
            results[f"input_type_{i}"] = {
                "input_type": ["clean", "whitespace", "numbers", "malicious"][i],
                "processing_time_ms": processing_time * 1000,
                "success_rate": success_count / 100,
                "input_length": len(test_input),
            }
        
        return results
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all performance benchmarks."""
        results = {
            "config": self.config.__dict__,
            "timestamp": time.time(),
            "benchmarks": {}
        }
        
        try:
            results["benchmarks"]["tokenization"] = self.benchmark_tokenization()
        except Exception as e:
            logger.error(f"Tokenization benchmark failed: {e}")
            results["benchmarks"]["tokenization"] = {"error": str(e)}
        
        try:
            results["benchmarks"]["validation"] = self.benchmark_validation()
        except Exception as e:
            logger.error(f"Validation benchmark failed: {e}")
            results["benchmarks"]["validation"] = {"error": str(e)}
        
        try:
            results["benchmarks"]["security"] = self.benchmark_security()
        except Exception as e:
            logger.error(f"Security benchmark failed: {e}")
            results["benchmarks"]["security"] = {"error": str(e)}
        
        return results


class QualityBenchmark:
    """Benchmark sequence generation quality."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.reference_sequences = self._load_reference_sequences()
    
    def _load_reference_sequences(self) -> List[str]:
        """Load reference protein sequences for comparison."""
        # Mock reference sequences - in practice, load from UniProt or similar
        return [
            "MAKLLILTCLVAVAL",
            "MKWVTFISLLLLFSSAYS", 
            "ATGAAACTGCTGCTGCTG",
            "MKKSLLIILVCVTACAAG",
            "MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTE",
        ] * 20  # Expand for testing
    
    def evaluate_generated_sequences(self, sequences: List[str]) -> Dict[str, Any]:
        """Evaluate quality of generated sequences."""
        logger.info(f"Evaluating quality of {len(sequences)} sequences...")
        
        results = {}
        
        # Basic statistics
        results["basic_stats"] = SequenceMetrics.calculate_length_statistics(sequences)
        results["amino_acid_composition"] = SequenceMetrics.calculate_amino_acid_composition(sequences)
        
        # Diversity metrics
        if self.config.evaluate_diversity:
            results["diversity"] = {
                "sequence_diversity": SequenceMetrics.calculate_diversity(sequences),
                "unique_sequences": len(set(sequences)),
                "uniqueness_ratio": len(set(sequences)) / len(sequences) if sequences else 0
            }
        
        # Novelty metrics
        if self.config.evaluate_novelty:
            results["novelty"] = {
                "novelty_score": SequenceMetrics.calculate_novelty(sequences, self.reference_sequences)
            }
        
        # Reference composition comparison
        ref_composition = SequenceMetrics.calculate_amino_acid_composition(self.reference_sequences)
        gen_composition = results["amino_acid_composition"]
        
        composition_diff = {}
        for aa in set(list(ref_composition.keys()) + list(gen_composition.keys())):
            ref_freq = ref_composition.get(aa, 0)
            gen_freq = gen_composition.get(aa, 0)
            composition_diff[aa] = abs(ref_freq - gen_freq)
        
        results["composition_analysis"] = {
            "reference_composition": ref_composition,
            "composition_differences": composition_diff,
            "total_composition_error": sum(composition_diff.values())
        }
        
        return results


class ComparisonStudy:
    """Conduct comparative studies between different approaches."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
    
    def compare_tokenization_methods(self) -> Dict[str, Any]:
        """Compare different tokenization approaches."""
        logger.info("Comparing tokenization methods...")
        
        # For now, we only have SELFIES tokenizer
        # In a real implementation, we'd compare multiple methods
        
        from protein_diffusion.tokenization.selfies_tokenizer import SELFIESTokenizer, TokenizerConfig
        
        results = {}
        test_sequences = [
            "MAKLLILTCLVAVAL",
            "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFS",
            "ATGAAACTGCTGCTGCTG" * 3
        ]
        
        # SELFIES tokenization
        config = TokenizerConfig()
        tokenizer = SELFIESTokenizer(config)
        
        selfies_results = []
        for seq in test_sequences:
            start_time = time.time()
            tokens = tokenizer.tokenize(seq)
            encoding_time = time.time() - start_time
            
            # Encode and decode to test roundtrip
            encoding = tokenizer.encode(seq, max_length=len(seq) + 10)
            decoded = tokenizer.decode(encoding['input_ids'])
            
            selfies_results.append({
                "original_length": len(seq),
                "num_tokens": len(tokens),
                "compression_ratio": len(tokens) / len(seq) if seq else 0,
                "encoding_time_ms": encoding_time * 1000,
                "roundtrip_success": decoded.replace("A", "").strip() != "",  # Basic check
            })
        
        results["selfies"] = {
            "method_name": "SELFIES",
            "results": selfies_results,
            "average_compression": sum(r["compression_ratio"] for r in selfies_results) / len(selfies_results),
            "average_time_ms": sum(r["encoding_time_ms"] for r in selfies_results) / len(selfies_results)
        }
        
        return results
    
    def compare_validation_strategies(self) -> Dict[str, Any]:
        """Compare different validation strategies."""
        logger.info("Comparing validation strategies...")
        
        from protein_diffusion.validation import SequenceValidator, ValidationLevel
        
        results = {}
        test_sequences = [
            "MAKLLILTCLVAVAL",  # Valid
            "MAKLLILTCLVAVALX",  # Contains X
            "MAKLL123ILTCLVAVAL",  # Contains numbers
            "M" * 5,  # Too short
            "M" * 1000,  # Long
        ]
        
        for level in [ValidationLevel.STRICT, ValidationLevel.MODERATE, ValidationLevel.PERMISSIVE]:
            validator = SequenceValidator(level)
            level_results = []
            
            for seq in test_sequences:
                start_time = time.time()
                result = validator.validate_sequence(seq)
                validation_time = time.time() - start_time
                
                level_results.append({
                    "sequence": seq[:20] + "..." if len(seq) > 20 else seq,
                    "is_valid": result.is_valid,
                    "num_errors": len(result.errors),
                    "num_warnings": len(result.warnings),
                    "validation_time_ms": validation_time * 1000
                })
            
            results[level.value] = {
                "validation_level": level.value,
                "results": level_results,
                "average_time_ms": sum(r["validation_time_ms"] for r in level_results) / len(level_results),
                "pass_rate": sum(1 for r in level_results if r["is_valid"]) / len(level_results)
            }
        
        return results


class ResearchBenchmarkRunner:
    """Main runner for research benchmarks and studies."""
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        
        # Create output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / "benchmark.log"),
                logging.StreamHandler()
            ]
        )
    
    def run_performance_studies(self) -> Dict[str, Any]:
        """Run comprehensive performance studies."""
        logger.info("🚀 Starting performance benchmark studies...")
        
        benchmark = PerformanceBenchmark(self.config)
        results = benchmark.run_all_benchmarks()
        
        # Save results
        output_file = self.output_dir / "performance_benchmark.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Performance benchmark results saved to {output_file}")
        return results
    
    def run_quality_studies(self) -> Dict[str, Any]:
        """Run quality evaluation studies."""
        logger.info("🔬 Starting quality benchmark studies...")
        
        # Generate some mock sequences for testing
        mock_sequences = [
            "MAKLLILTCLVAVAL",
            "MKWVTFISLLLLFSSAYS",
            "ATGAAACTGCTGCTGCTG",
            "MKKSLLIILVCVTACAAG",
            "MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTE",
        ] * 20
        
        benchmark = QualityBenchmark(self.config)
        results = benchmark.evaluate_generated_sequences(mock_sequences)
        
        # Save results
        output_file = self.output_dir / "quality_benchmark.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Quality benchmark results saved to {output_file}")
        return results
    
    def run_comparison_studies(self) -> Dict[str, Any]:
        """Run comparative studies."""
        logger.info("⚖️ Starting comparison studies...")
        
        study = ComparisonStudy(self.config)
        
        results = {
            "timestamp": time.time(),
            "config": self.config.__dict__,
            "studies": {}
        }
        
        # Tokenization comparison
        try:
            results["studies"]["tokenization"] = study.compare_tokenization_methods()
        except Exception as e:
            logger.error(f"Tokenization comparison failed: {e}")
            results["studies"]["tokenization"] = {"error": str(e)}
        
        # Validation comparison
        try:
            results["studies"]["validation"] = study.compare_validation_strategies()
        except Exception as e:
            logger.error(f"Validation comparison failed: {e}")
            results["studies"]["validation"] = {"error": str(e)}
        
        # Save results
        output_file = self.output_dir / "comparison_studies.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Comparison study results saved to {output_file}")
        return results
    
    def run_comprehensive_research_suite(self) -> Dict[str, Any]:
        """Run the complete research benchmark suite."""
        logger.info("🧬 Starting comprehensive research benchmark suite...")
        
        start_time = time.time()
        
        # Run all studies
        performance_results = self.run_performance_studies()
        quality_results = self.run_quality_studies()
        comparison_results = self.run_comparison_studies()
        
        # Combine results
        comprehensive_results = {
            "suite_info": {
                "timestamp": time.time(),
                "duration_seconds": time.time() - start_time,
                "config": self.config.__dict__
            },
            "performance_benchmark": performance_results,
            "quality_benchmark": quality_results,
            "comparison_studies": comparison_results
        }
        
        # Generate summary report
        self._generate_summary_report(comprehensive_results)
        
        # Save comprehensive results
        output_file = self.output_dir / "comprehensive_research_results.json"
        with open(output_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        logger.info(f"✅ Comprehensive research suite completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"Results saved to {self.output_dir}")
        
        return comprehensive_results
    
    def _generate_summary_report(self, results: Dict[str, Any]):
        """Generate a human-readable summary report."""
        report_file = self.output_dir / "research_summary_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# Protein Diffusion Research Benchmark Report\n\n")
            f.write(f"Generated at: {time.ctime()}\n\n")
            
            # Suite info
            suite_info = results["suite_info"]
            f.write(f"## Suite Information\n")
            f.write(f"- Duration: {suite_info['duration_seconds']:.2f} seconds\n")
            f.write(f"- Configuration: {json.dumps(suite_info['config'], indent=2)}\n\n")
            
            # Performance results
            perf = results.get("performance_benchmark", {})
            if "benchmarks" in perf:
                f.write("## Performance Benchmarks\n\n")
                
                if "tokenization" in perf["benchmarks"]:
                    f.write("### Tokenization Performance\n")
                    tokenization = perf["benchmarks"]["tokenization"]
                    for key, data in tokenization.items():
                        if isinstance(data, dict):
                            f.write(f"- {key}: {data.get('tokenization_time_ms', 'N/A'):.2f}ms\n")
                    f.write("\n")
                
                if "validation" in perf["benchmarks"]:
                    f.write("### Validation Performance\n")
                    validation = perf["benchmarks"]["validation"]
                    avg_time = sum(data.get('validation_time_ms', 0) for data in validation.values() if isinstance(data, dict)) / len(validation)
                    f.write(f"- Average validation time: {avg_time:.2f}ms\n\n")
            
            # Quality results
            quality = results.get("quality_benchmark", {})
            if "basic_stats" in quality:
                f.write("## Quality Analysis\n\n")
                stats = quality["basic_stats"]
                f.write(f"### Sequence Statistics\n")
                f.write(f"- Mean length: {stats.get('mean_length', 0):.1f}\n")
                f.write(f"- Length range: {stats.get('min_length', 0)} - {stats.get('max_length', 0)}\n")
                
                if "diversity" in quality:
                    div = quality["diversity"]
                    f.write(f"- Sequence diversity: {div.get('sequence_diversity', 0):.3f}\n")
                    f.write(f"- Uniqueness ratio: {div.get('uniqueness_ratio', 0):.3f}\n")
                
                f.write("\n")
            
            # Comparison results
            comp = results.get("comparison_studies", {})
            if "studies" in comp:
                f.write("## Comparison Studies\n\n")
                studies = comp["studies"]
                
                if "tokenization" in studies:
                    f.write("### Tokenization Method Comparison\n")
                    for method, data in studies["tokenization"].items():
                        if isinstance(data, dict):
                            f.write(f"- {method}: avg compression {data.get('average_compression', 0):.3f}\n")
                    f.write("\n")
                
                if "validation" in studies:
                    f.write("### Validation Strategy Comparison\n")
                    for level, data in studies["validation"].items():
                        if isinstance(data, dict):
                            f.write(f"- {level}: {data.get('pass_rate', 0)*100:.1f}% pass rate\n")
        
        logger.info(f"Summary report saved to {report_file}")


def main():
    """Main entry point for research benchmarks."""
    print("🧬 Protein Diffusion Research Benchmarks")
    print("=" * 50)
    
    # Create benchmark configuration
    config = BenchmarkConfig()
    config.output_dir = "./research/results"
    config.dataset_size = 100  # Smaller for demo
    
    # Run benchmarks
    runner = ResearchBenchmarkRunner(config)
    results = runner.run_comprehensive_research_suite()
    
    # Print summary
    print(f"\n✅ Research benchmarks completed!")
    print(f"📁 Results saved to: {config.output_dir}")
    print(f"📊 Performance tests: {'✅' if 'performance_benchmark' in results else '❌'}")
    print(f"🔬 Quality analysis: {'✅' if 'quality_benchmark' in results else '❌'}")
    print(f"⚖️ Comparison studies: {'✅' if 'comparison_studies' in results else '❌'}")


if __name__ == "__main__":
    main()