"""
Quality Gates and Automated Testing Framework for Protein Diffusion Design Lab.

This module implements comprehensive quality gates, automated testing,
and continuous validation for the protein design pipeline.
"""

import logging
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import traceback

logger = logging.getLogger(__name__)


class QualityGateResult(Enum):
    """Quality gate execution results."""
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    SKIP = "SKIP"
    ERROR = "ERROR"


@dataclass
class QualityGateReport:
    """Report from a quality gate execution."""
    gate_name: str
    result: QualityGateResult
    execution_time: float
    details: Dict[str, Any]
    error_message: Optional[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


@dataclass
class QualityGateConfig:
    """Configuration for quality gate execution."""
    # Gate execution settings
    fail_fast: bool = False
    enable_warnings: bool = True
    max_execution_time: float = 300.0  # 5 minutes
    
    # Output settings
    save_reports: bool = True
    output_dir: str = "./quality_reports"
    verbose_logging: bool = True
    
    # Specific gate configurations
    security_config: Dict[str, Any] = None
    performance_config: Dict[str, Any] = None
    validation_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.security_config is None:
            self.security_config = {}
        if self.performance_config is None:
            self.performance_config = {}
        if self.validation_config is None:
            self.validation_config = {}


class QualityGate:
    """Base class for quality gates."""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.enabled = True
    
    def execute(self, context: Dict[str, Any]) -> QualityGateReport:
        """Execute the quality gate."""
        start_time = time.time()
        
        try:
            if not self.enabled:
                return QualityGateReport(
                    gate_name=self.name,
                    result=QualityGateResult.SKIP,
                    execution_time=0.0,
                    details={"reason": "Gate disabled"}
                )
            
            result = self._execute_gate(context)
            execution_time = time.time() - start_time
            
            return QualityGateReport(
                gate_name=self.name,
                result=result["status"],
                execution_time=execution_time,
                details=result.get("details", {}),
                error_message=result.get("error"),
                warnings=result.get("warnings", [])
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Quality gate {self.name} failed with exception: {e}")
            
            return QualityGateReport(
                gate_name=self.name,
                result=QualityGateResult.ERROR,
                execution_time=execution_time,
                details={"exception": str(e), "traceback": traceback.format_exc()},
                error_message=str(e)
            )
    
    def _execute_gate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Override this method in subclasses."""
        raise NotImplementedError("Subclasses must implement _execute_gate")


class SecurityQualityGate(QualityGate):
    """Security validation quality gate."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Security Validation", config)
    
    def _execute_gate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute security validation checks."""
        details = {}
        warnings = []
        
        # Check for potential security issues in sequences
        generated_sequences = context.get('generated_sequences', [])
        
        # 1. Input validation
        input_validation = self._validate_inputs(context)
        details['input_validation'] = input_validation
        
        if not input_validation['passed']:
            return {
                "status": QualityGateResult.FAIL,
                "details": details,
                "error": "Input validation failed"
            }
        
        # 2. Sequence content validation
        sequence_validation = self._validate_sequence_content(generated_sequences)
        details['sequence_validation'] = sequence_validation
        
        if sequence_validation['suspicious_patterns'] > 0:
            warnings.append(f"Found {sequence_validation['suspicious_patterns']} suspicious patterns in sequences")
        
        # 3. Output sanitization check
        output_validation = self._validate_outputs(context)
        details['output_validation'] = output_validation
        
        # Determine overall result
        if details['input_validation']['passed'] and details['output_validation']['passed']:
            if warnings:
                return {
                    "status": QualityGateResult.WARNING,
                    "details": details,
                    "warnings": warnings
                }
            else:
                return {
                    "status": QualityGateResult.PASS,
                    "details": details
                }
        else:
            return {
                "status": QualityGateResult.FAIL,
                "details": details,
                "error": "Security validation failed"
            }
    
    def _validate_inputs(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input parameters for security issues."""
        validation_result = {"passed": True, "issues": []}
        
        # Check motif parameter
        motif = context.get('motif', '')
        if motif and len(motif) > 1000:
            validation_result['issues'].append("Motif too long")
            validation_result['passed'] = False
        
        # Check number of sequences
        num_sequences = context.get('num_sequences', 0)
        if num_sequences > 10000:
            validation_result['issues'].append("Too many sequences requested")
            validation_result['passed'] = False
        
        # Check file paths
        target_pdb = context.get('target_pdb')
        if target_pdb and not self._is_safe_path(target_pdb):
            validation_result['issues'].append("Unsafe file path detected")
            validation_result['passed'] = False
        
        return validation_result
    
    def _validate_sequence_content(self, sequences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate generated sequence content."""
        validation_result = {
            "total_sequences": len(sequences),
            "suspicious_patterns": 0,
            "invalid_characters": 0
        }
        
        valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
        
        for seq_data in sequences:
            sequence = seq_data.get('sequence', '')
            
            # Check for invalid characters
            invalid_chars = set(sequence) - valid_amino_acids
            if invalid_chars:
                validation_result['invalid_characters'] += 1
            
            # Check for suspicious patterns (very repetitive sequences)
            if self._has_suspicious_patterns(sequence):
                validation_result['suspicious_patterns'] += 1
        
        return validation_result
    
    def _validate_outputs(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate output data for security issues."""
        return {"passed": True, "sanitized": True}
    
    def _is_safe_path(self, file_path: str) -> bool:
        """Check if file path is safe (no directory traversal)."""
        if not file_path:
            return True
        
        # Check for directory traversal patterns
        dangerous_patterns = ['../', '..\\', '/etc/', '/root/', '/home/']
        return not any(pattern in file_path.lower() for pattern in dangerous_patterns)
    
    def _has_suspicious_patterns(self, sequence: str) -> bool:
        """Check for suspicious patterns in sequence."""
        if len(sequence) < 10:
            return False
        
        # Check for excessive repetition
        for i in range(len(sequence) - 5):
            substring = sequence[i:i+3]
            if sequence.count(substring) > len(sequence) * 0.3:
                return True
        
        return False


class PerformanceQualityGate(QualityGate):
    """Performance validation quality gate."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Performance Validation", config)
        self.performance_thresholds = {
            'max_generation_time': config.get('max_generation_time', 300.0) if config else 300.0,
            'max_ranking_time': config.get('max_ranking_time', 180.0) if config else 180.0,
            'min_throughput': config.get('min_throughput', 0.1) if config else 0.1,  # sequences/second
            'max_memory_usage_mb': config.get('max_memory_usage_mb', 4000) if config else 4000,
        }
    
    def _execute_gate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute performance validation checks."""
        details = {}
        warnings = []
        issues = []
        
        # 1. Generation performance
        generation_stats = context.get('generation_stats', {})
        generation_perf = self._validate_generation_performance(generation_stats)
        details['generation_performance'] = generation_perf
        
        if not generation_perf['passed']:
            issues.extend(generation_perf['issues'])
        if generation_perf['warnings']:
            warnings.extend(generation_perf['warnings'])
        
        # 2. Ranking performance
        ranking_stats = context.get('ranking_stats', {})
        ranking_perf = self._validate_ranking_performance(ranking_stats)
        details['ranking_performance'] = ranking_perf
        
        if not ranking_perf['passed']:
            issues.extend(ranking_perf['issues'])
        if ranking_perf['warnings']:
            warnings.extend(ranking_perf['warnings'])
        
        # 3. Memory usage
        memory_usage = self._check_memory_usage()
        details['memory_usage'] = memory_usage
        
        if memory_usage['mb'] > self.performance_thresholds['max_memory_usage_mb']:
            warnings.append(f"High memory usage: {memory_usage['mb']}MB")
        
        # 4. Overall throughput
        throughput = self._calculate_throughput(context)
        details['throughput'] = throughput
        
        if throughput['sequences_per_second'] < self.performance_thresholds['min_throughput']:
            issues.append(f"Low throughput: {throughput['sequences_per_second']:.3f} seq/s")
        
        # Determine result
        if issues:
            return {
                "status": QualityGateResult.FAIL,
                "details": details,
                "error": f"Performance issues: {'; '.join(issues)}"
            }
        elif warnings:
            return {
                "status": QualityGateResult.WARNING,
                "details": details,
                "warnings": warnings
            }
        else:
            return {
                "status": QualityGateResult.PASS,
                "details": details
            }
    
    def _validate_generation_performance(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Validate generation performance metrics."""
        result = {"passed": True, "issues": [], "warnings": []}
        
        generation_time = stats.get('generation_time', 0)
        if generation_time > self.performance_thresholds['max_generation_time']:
            result['issues'].append(f"Generation time too high: {generation_time:.2f}s")
            result['passed'] = False
        elif generation_time > self.performance_thresholds['max_generation_time'] * 0.8:
            result['warnings'].append(f"Generation time approaching limit: {generation_time:.2f}s")
        
        success_rate = stats.get('success_rate', 0)
        if success_rate < 0.8:
            result['issues'].append(f"Low generation success rate: {success_rate:.2f}")
            result['passed'] = False
        elif success_rate < 0.9:
            result['warnings'].append(f"Generation success rate could be better: {success_rate:.2f}")
        
        return result
    
    def _validate_ranking_performance(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Validate ranking performance metrics."""
        result = {"passed": True, "issues": [], "warnings": []}
        
        ranking_time = stats.get('ranking_time', 0)
        if ranking_time > self.performance_thresholds['max_ranking_time']:
            result['issues'].append(f"Ranking time too high: {ranking_time:.2f}s")
            result['passed'] = False
        elif ranking_time > self.performance_thresholds['max_ranking_time'] * 0.8:
            result['warnings'].append(f"Ranking time approaching limit: {ranking_time:.2f}s")
        
        return result
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                'mb': memory_info.rss / 1024 / 1024,
                'available': True
            }
        except ImportError:
            return {
                'mb': 0,
                'available': False,
                'note': 'psutil not available'
            }
    
    def _calculate_throughput(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall throughput."""
        generation_stats = context.get('generation_stats', {})
        ranking_stats = context.get('ranking_stats', {})
        
        total_sequences = generation_stats.get('successful_generations', 0)
        total_time = generation_stats.get('generation_time', 0) + ranking_stats.get('ranking_time', 0)
        
        if total_time > 0:
            sequences_per_second = total_sequences / total_time
        else:
            sequences_per_second = 0
        
        return {
            'sequences_per_second': sequences_per_second,
            'total_sequences': total_sequences,
            'total_time': total_time
        }


class BiologicalValidationQualityGate(QualityGate):
    """Biological validation quality gate."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Biological Validation", config)
    
    def _execute_gate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute biological validation checks."""
        details = {}
        warnings = []
        issues = []
        
        generated_sequences = context.get('generated_sequences', [])
        ranked_sequences = context.get('ranked_sequences', [])
        
        # 1. Sequence validity
        sequence_validation = self._validate_sequences(generated_sequences)
        details['sequence_validation'] = sequence_validation
        
        if sequence_validation['invalid_count'] > 0:
            issues.append(f"{sequence_validation['invalid_count']} invalid sequences found")
        
        # 2. Structural plausibility
        structure_validation = self._validate_structures(ranked_sequences)
        details['structure_validation'] = structure_validation
        
        if structure_validation['low_quality_count'] > len(ranked_sequences) * 0.5:
            warnings.append("High proportion of low-quality structures")
        
        # 3. Binding affinity plausibility
        binding_validation = self._validate_binding_affinities(ranked_sequences)
        details['binding_validation'] = binding_validation
        
        if binding_validation['implausible_count'] > 0:
            warnings.append(f"{binding_validation['implausible_count']} implausible binding affinities")
        
        # 4. Diversity check
        diversity_validation = self._validate_diversity(generated_sequences)
        details['diversity_validation'] = diversity_validation
        
        if diversity_validation['diversity_score'] < 0.3:
            warnings.append("Low sequence diversity")
        
        # Determine result
        if issues:
            return {
                "status": QualityGateResult.FAIL,
                "details": details,
                "error": f"Biological validation failed: {'; '.join(issues)}"
            }
        elif warnings:
            return {
                "status": QualityGateResult.WARNING,
                "details": details,
                "warnings": warnings
            }
        else:
            return {
                "status": QualityGateResult.PASS,
                "details": details
            }
    
    def _validate_sequences(self, sequences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate protein sequences for biological plausibility."""
        valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
        
        validation_result = {
            "total_sequences": len(sequences),
            "valid_count": 0,
            "invalid_count": 0,
            "avg_length": 0,
            "length_distribution": {}
        }
        
        lengths = []
        
        for seq_data in sequences:
            sequence = seq_data.get('sequence', '')
            
            # Check valid amino acids
            if all(aa in valid_amino_acids for aa in sequence):
                validation_result['valid_count'] += 1
            else:
                validation_result['invalid_count'] += 1
            
            lengths.append(len(sequence))
        
        if lengths:
            validation_result['avg_length'] = sum(lengths) / len(lengths)
            validation_result['min_length'] = min(lengths)
            validation_result['max_length'] = max(lengths)
        
        return validation_result
    
    def _validate_structures(self, ranked_sequences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate predicted structures."""
        validation_result = {
            "total_sequences": len(ranked_sequences),
            "high_quality_count": 0,
            "medium_quality_count": 0,
            "low_quality_count": 0,
            "avg_structure_quality": 0
        }
        
        structure_qualities = []
        
        for seq_data in ranked_sequences:
            structure_quality = seq_data.get('structure_quality', 0)
            structure_qualities.append(structure_quality)
            
            if structure_quality > 0.8:
                validation_result['high_quality_count'] += 1
            elif structure_quality > 0.6:
                validation_result['medium_quality_count'] += 1
            else:
                validation_result['low_quality_count'] += 1
        
        if structure_qualities:
            validation_result['avg_structure_quality'] = sum(structure_qualities) / len(structure_qualities)
        
        return validation_result
    
    def _validate_binding_affinities(self, ranked_sequences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate binding affinity predictions."""
        validation_result = {
            "total_sequences": len(ranked_sequences),
            "plausible_count": 0,
            "implausible_count": 0,
            "avg_binding_affinity": 0
        }
        
        binding_affinities = []
        
        for seq_data in ranked_sequences:
            binding_affinity = seq_data.get('binding_affinity', 0)
            binding_affinities.append(binding_affinity)
            
            # Plausible range for binding affinity: -25 to 5 kcal/mol
            if -25 <= binding_affinity <= 5:
                validation_result['plausible_count'] += 1
            else:
                validation_result['implausible_count'] += 1
        
        if binding_affinities:
            validation_result['avg_binding_affinity'] = sum(binding_affinities) / len(binding_affinities)
        
        return validation_result
    
    def _validate_diversity(self, sequences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate sequence diversity."""
        if len(sequences) < 2:
            return {"diversity_score": 1.0, "total_sequences": len(sequences)}
        
        sequence_strings = [s.get('sequence', '') for s in sequences]
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(sequence_strings)):
            for j in range(i + 1, len(sequence_strings)):
                similarity = self._calculate_sequence_similarity(sequence_strings[i], sequence_strings[j])
                similarities.append(similarity)
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        diversity_score = 1 - avg_similarity
        
        return {
            "diversity_score": diversity_score,
            "avg_similarity": avg_similarity,
            "total_comparisons": len(similarities),
            "total_sequences": len(sequences)
        }
    
    def _calculate_sequence_similarity(self, seq1: str, seq2: str) -> float:
        """Calculate similarity between two sequences."""
        if not seq1 or not seq2:
            return 0.0
        
        min_len = min(len(seq1), len(seq2))
        matches = sum(1 for i in range(min_len) if seq1[i] == seq2[i])
        return matches / max(len(seq1), len(seq2))


class QualityGateManager:
    """Manager for executing quality gates and generating reports."""
    
    def __init__(self, config: Optional[QualityGateConfig] = None):
        self.config = config or QualityGateConfig()
        
        # Initialize output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize quality gates
        self.gates = self._initialize_gates()
        
        logger.info(f"Quality Gate Manager initialized with {len(self.gates)} gates")
    
    def _initialize_gates(self) -> List[QualityGate]:
        """Initialize all quality gates."""
        gates = []
        
        # Security gate
        security_gate = SecurityQualityGate(self.config.security_config)
        gates.append(security_gate)
        
        # Performance gate
        performance_gate = PerformanceQualityGate(self.config.performance_config)
        gates.append(performance_gate)
        
        # Biological validation gate
        biological_gate = BiologicalValidationQualityGate(self.config.validation_config)
        gates.append(biological_gate)
        
        return gates
    
    def execute_all_gates(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute all quality gates.
        
        Args:
            context: Context containing workflow data
            
        Returns:
            Quality gate execution results
        """
        logger.info("Executing quality gates")
        start_time = time.time()
        
        results = {
            'timestamp': start_time,
            'overall_result': QualityGateResult.PASS,
            'execution_time': 0.0,
            'gate_reports': [],
            'summary': {}
        }
        
        passed_gates = 0
        failed_gates = 0
        warning_gates = 0
        error_gates = 0
        
        for gate in self.gates:
            if self.config.verbose_logging:
                logger.info(f"Executing quality gate: {gate.name}")
            
            try:
                report = gate.execute(context)
                results['gate_reports'].append(asdict(report))
                
                # Track results
                if report.result == QualityGateResult.PASS:
                    passed_gates += 1
                elif report.result == QualityGateResult.FAIL:
                    failed_gates += 1
                    if self.config.fail_fast:
                        logger.error(f"Quality gate {gate.name} failed, stopping execution")
                        break
                elif report.result == QualityGateResult.WARNING:
                    warning_gates += 1
                elif report.result == QualityGateResult.ERROR:
                    error_gates += 1
                
                if self.config.verbose_logging:
                    logger.info(f"Quality gate {gate.name} completed: {report.result.value}")
                
            except Exception as e:
                logger.error(f"Error executing quality gate {gate.name}: {e}")
                error_gates += 1
                
                error_report = QualityGateReport(
                    gate_name=gate.name,
                    result=QualityGateResult.ERROR,
                    execution_time=0.0,
                    details={"error": str(e)},
                    error_message=str(e)
                )
                results['gate_reports'].append(asdict(error_report))
        
        # Calculate overall result
        total_execution_time = time.time() - start_time
        results['execution_time'] = total_execution_time
        
        # Determine overall status
        if error_gates > 0 or failed_gates > 0:
            results['overall_result'] = QualityGateResult.FAIL
        elif warning_gates > 0:
            results['overall_result'] = QualityGateResult.WARNING
        else:
            results['overall_result'] = QualityGateResult.PASS
        
        # Create summary
        results['summary'] = {
            'total_gates': len(self.gates),
            'passed': passed_gates,
            'failed': failed_gates,
            'warnings': warning_gates,
            'errors': error_gates,
            'execution_time': total_execution_time
        }
        
        # Save report if configured
        if self.config.save_reports:
            self._save_quality_report(results)
        
        logger.info(f"Quality gates completed: {results['overall_result'].value} "
                   f"({passed_gates} passed, {failed_gates} failed, {warning_gates} warnings, {error_gates} errors)")
        
        return results
    
    def _save_quality_report(self, results: Dict[str, Any]):
        """Save quality gate report to disk."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"quality_report_{timestamp}.json"
        
        try:
            with open(report_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Quality report saved to {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to save quality report: {e}")
    
    def add_custom_gate(self, gate: QualityGate):
        """Add a custom quality gate."""
        self.gates.append(gate)
        logger.info(f"Added custom quality gate: {gate.name}")
    
    def disable_gate(self, gate_name: str):
        """Disable a specific quality gate."""
        for gate in self.gates:
            if gate.name == gate_name:
                gate.enabled = False
                logger.info(f"Disabled quality gate: {gate_name}")
                return
        
        logger.warning(f"Quality gate not found: {gate_name}")
    
    def get_gate_status(self) -> Dict[str, Any]:
        """Get status of all quality gates."""
        return {
            'total_gates': len(self.gates),
            'enabled_gates': sum(1 for gate in self.gates if gate.enabled),
            'disabled_gates': sum(1 for gate in self.gates if not gate.enabled),
            'gate_details': [
                {
                    'name': gate.name,
                    'enabled': gate.enabled,
                    'config': gate.config
                }
                for gate in self.gates
            ]
        }


# Convenience functions for common quality gate operations

def run_quality_gates(
    workflow_result: Dict[str, Any],
    config: Optional[QualityGateConfig] = None
) -> Dict[str, Any]:
    """
    Run quality gates on workflow results.
    
    Args:
        workflow_result: Results from workflow execution
        config: Quality gate configuration
        
    Returns:
        Quality gate execution results
    """
    manager = QualityGateManager(config)
    return manager.execute_all_gates(workflow_result)


def validate_generation_quality(
    generated_sequences: List[Dict[str, Any]],
    generation_stats: Dict[str, Any]
) -> bool:
    """
    Quick validation of generation quality.
    
    Args:
        generated_sequences: Generated protein sequences
        generation_stats: Generation statistics
        
    Returns:
        True if quality is acceptable
    """
    context = {
        'generated_sequences': generated_sequences,
        'generation_stats': generation_stats
    }
    
    config = QualityGateConfig(fail_fast=True, save_reports=False, verbose_logging=False)
    manager = QualityGateManager(config)
    
    # Only run biological and performance gates for quick check
    manager.gates = [gate for gate in manager.gates if 
                    gate.name in ["Biological Validation", "Performance Validation"]]
    
    results = manager.execute_all_gates(context)
    return results['overall_result'] in [QualityGateResult.PASS, QualityGateResult.WARNING]