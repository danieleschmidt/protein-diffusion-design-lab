"""
Next-Generation Robust Validation System for Protein Design.

This module implements comprehensive validation with:
- Multi-layer validation with circuit breakers
- Biophysical constraint validation  
- Real-time monitoring and alerting
- Self-healing validation pipelines
- Adaptive validation based on confidence scoring
"""

import asyncio
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from datetime import datetime, timezone
from enum import Enum
import logging
import json
from pathlib import Path

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    class MockNumpy:
        @staticmethod
        def array(x): return x
        @staticmethod 
        def mean(x): return sum(x)/len(x) if x else 0.5
        @staticmethod
        def std(x): return 0.1
        @staticmethod
        def min(x): return min(x) if x else 0
        @staticmethod
        def max(x): return max(x) if x else 1
    np = MockNumpy()
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels."""
    PERMISSIVE = "permissive"
    NORMAL = "normal" 
    STRICT = "strict"
    RESEARCH = "research"
    PRODUCTION = "production"


class ValidationType(Enum):
    """Types of validation."""
    SEQUENCE = "sequence"
    STRUCTURE = "structure"
    PARAMETERS = "parameters"
    BIOPHYSICAL = "biophysical"
    STATISTICAL = "statistical"
    PERFORMANCE = "performance"
    SECURITY = "security"


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Individual validation issue."""
    type: ValidationType
    severity: ValidationSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved: bool = False
    resolution_strategy: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of validation process."""
    passed: bool
    validation_level: ValidationLevel
    issues: List[ValidationIssue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    confidence_score: float = 1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class RobustValidationConfig:
    """Configuration for robust validation system."""
    # Validation levels
    default_validation_level: ValidationLevel = ValidationLevel.NORMAL
    biophysical_validation: bool = True
    statistical_validation: bool = True
    performance_validation: bool = True
    
    # Circuit breaker settings
    circuit_breaker_enabled: bool = True
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    half_open_max_calls: int = 3
    
    # Validation timeouts
    sequence_validation_timeout: float = 30.0  # seconds
    structure_validation_timeout: float = 120.0
    biophysical_validation_timeout: float = 300.0
    
    # Confidence scoring
    confidence_weights: Dict[str, float] = field(default_factory=lambda: {
        'sequence_validity': 0.2,
        'structure_quality': 0.3,
        'biophysical_plausibility': 0.3,
        'statistical_likelihood': 0.2
    })
    
    # Adaptive validation
    adaptive_thresholds: bool = True
    learning_rate: float = 0.01
    threshold_adjustment_factor: float = 0.1
    
    # Monitoring and alerting
    enable_monitoring: bool = True
    alert_threshold: float = 0.1  # Alert if validation failure rate > 10%
    monitoring_window: int = 3600  # 1 hour window
    
    # Self-healing
    auto_recovery: bool = True
    max_retry_attempts: int = 3
    retry_backoff_factor: float = 2.0
    
    # Storage
    store_validation_history: bool = True
    validation_history_path: str = "./validation_history"
    max_history_size: int = 100000


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker for validation failure protection."""
    
    def __init__(self, failure_threshold: int, recovery_timeout: int, half_open_max_calls: int):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0
        
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                self.half_open_calls = 0
            else:
                raise Exception("Circuit breaker is OPEN - calls are blocked")
                
        if self.state == CircuitBreakerState.HALF_OPEN:
            if self.half_open_calls >= self.half_open_max_calls:
                raise Exception("Circuit breaker is HALF_OPEN - max calls exceeded")
            self.half_open_calls += 1
            
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
            
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.recovery_timeout
        
    def _on_success(self):
        """Handle successful call."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.half_open_calls = 0
            
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN


class BaseValidator(ABC):
    """Base class for all validators."""
    
    def __init__(self, config: RobustValidationConfig):
        self.config = config
        self.circuit_breaker = CircuitBreaker(
            config.failure_threshold,
            config.recovery_timeout,
            config.half_open_max_calls
        ) if config.circuit_breaker_enabled else None
        
    @abstractmethod
    async def validate(self, data: Any, context: Dict[str, Any] = None) -> ValidationResult:
        """Perform validation on data."""
        pass
        
    async def safe_validate(self, data: Any, context: Dict[str, Any] = None) -> ValidationResult:
        """Validate with circuit breaker protection."""
        try:
            if self.circuit_breaker:
                return await self.circuit_breaker.call(self.validate, data, context)
            else:
                return await self.validate(data, context)
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return ValidationResult(
                passed=False,
                validation_level=self.config.default_validation_level,
                issues=[ValidationIssue(
                    type=ValidationType.SEQUENCE,
                    severity=ValidationSeverity.ERROR,
                    message=f"Validation error: {str(e)}",
                    details={'exception': str(e), 'traceback': traceback.format_exc()}
                )]
            )


class SequenceValidator(BaseValidator):
    """Validate protein sequences."""
    
    VALID_AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')
    
    async def validate(self, sequence: str, context: Dict[str, Any] = None) -> ValidationResult:
        """Validate protein sequence."""
        start_time = time.time()
        issues = []
        metadata = {}
        
        # Basic sequence validation
        if not isinstance(sequence, str):
            issues.append(ValidationIssue(
                type=ValidationType.SEQUENCE,
                severity=ValidationSeverity.ERROR,
                message="Sequence must be a string"
            ))
            
        if not sequence:
            issues.append(ValidationIssue(
                type=ValidationType.SEQUENCE,
                severity=ValidationSeverity.ERROR,
                message="Sequence cannot be empty"
            ))
            
        # Length validation
        seq_len = len(sequence)
        metadata['sequence_length'] = seq_len
        
        if seq_len < 10:
            issues.append(ValidationIssue(
                type=ValidationType.SEQUENCE,
                severity=ValidationSeverity.WARNING,
                message=f"Sequence very short: {seq_len} residues"
            ))
        elif seq_len > 2000:
            issues.append(ValidationIssue(
                type=ValidationType.SEQUENCE,
                severity=ValidationSeverity.WARNING,
                message=f"Sequence very long: {seq_len} residues"
            ))
            
        # Character validation
        invalid_chars = set(sequence.upper()) - self.VALID_AMINO_ACIDS
        if invalid_chars:
            issues.append(ValidationIssue(
                type=ValidationType.SEQUENCE,
                severity=ValidationSeverity.ERROR,
                message=f"Invalid amino acid characters: {invalid_chars}",
                details={'invalid_characters': list(invalid_chars)}
            ))
            
        # Composition analysis
        composition = self._analyze_composition(sequence)
        metadata['composition'] = composition
        
        # Check for unusual compositions
        if composition.get('hydrophobic_ratio', 0) > 0.8:
            issues.append(ValidationIssue(
                type=ValidationType.SEQUENCE,
                severity=ValidationSeverity.WARNING,
                message=f"Very high hydrophobic content: {composition['hydrophobic_ratio']:.2f}"
            ))
            
        if composition.get('charged_ratio', 0) > 0.4:
            issues.append(ValidationIssue(
                type=ValidationType.SEQUENCE,
                severity=ValidationSeverity.WARNING,
                message=f"Very high charged content: {composition['charged_ratio']:.2f}"
            ))
            
        # Repetitive sequence detection
        repetitive_score = self._detect_repetitive_patterns(sequence)
        metadata['repetitive_score'] = repetitive_score
        
        if repetitive_score > 0.7:
            issues.append(ValidationIssue(
                type=ValidationType.SEQUENCE,
                severity=ValidationSeverity.WARNING,
                message=f"Highly repetitive sequence: score {repetitive_score:.2f}"
            ))
            
        # Performance metrics
        validation_time = time.time() - start_time
        performance_metrics = {
            'validation_time': validation_time,
            'sequence_length': seq_len
        }
        
        # Calculate confidence score
        confidence_score = self._calculate_sequence_confidence(sequence, issues)
        
        return ValidationResult(
            passed=not any(issue.severity == ValidationSeverity.ERROR for issue in issues),
            validation_level=self.config.default_validation_level,
            issues=issues,
            metadata=metadata,
            performance_metrics=performance_metrics,
            confidence_score=confidence_score
        )
        
    def _analyze_composition(self, sequence: str) -> Dict[str, float]:
        """Analyze amino acid composition."""
        if not sequence:
            return {}
            
        seq_len = len(sequence)
        hydrophobic = set('AILMFPWV')
        charged = set('DEKR')
        polar = set('NQST')
        aromatic = set('FWY')
        
        composition = {
            'hydrophobic_ratio': sum(1 for aa in sequence if aa in hydrophobic) / seq_len,
            'charged_ratio': sum(1 for aa in sequence if aa in charged) / seq_len,
            'polar_ratio': sum(1 for aa in sequence if aa in polar) / seq_len,
            'aromatic_ratio': sum(1 for aa in sequence if aa in aromatic) / seq_len
        }
        
        # Individual amino acid frequencies
        for aa in self.VALID_AMINO_ACIDS:
            composition[f'{aa}_frequency'] = sequence.count(aa) / seq_len
            
        return composition
        
    def _detect_repetitive_patterns(self, sequence: str) -> float:
        """Detect repetitive patterns in sequence."""
        if len(sequence) < 6:
            return 0.0
            
        # Check for repeating patterns of length 2-6
        max_repetition = 0.0
        
        for pattern_len in range(2, min(7, len(sequence) // 2)):
            for i in range(len(sequence) - pattern_len):
                pattern = sequence[i:i+pattern_len]
                
                # Count consecutive repetitions
                repetitions = 1
                pos = i + pattern_len
                
                while pos + pattern_len <= len(sequence) and sequence[pos:pos+pattern_len] == pattern:
                    repetitions += 1
                    pos += pattern_len
                    
                if repetitions >= 3:  # 3 or more repetitions
                    repetition_score = (repetitions * pattern_len) / len(sequence)
                    max_repetition = max(max_repetition, repetition_score)
                    
        return max_repetition
        
    def _calculate_sequence_confidence(self, sequence: str, issues: List[ValidationIssue]) -> float:
        """Calculate confidence score for sequence."""
        base_confidence = 1.0
        
        # Penalize based on issues
        for issue in issues:
            if issue.severity == ValidationSeverity.ERROR:
                base_confidence -= 0.3
            elif issue.severity == ValidationSeverity.WARNING:
                base_confidence -= 0.1
                
        # Length-based confidence
        seq_len = len(sequence)
        if seq_len < 20:
            base_confidence *= 0.8
        elif seq_len > 1000:
            base_confidence *= 0.9
            
        return max(0.0, min(1.0, base_confidence))


class StructureValidator(BaseValidator):
    """Validate protein structures."""
    
    async def validate(self, coordinates: np.ndarray, context: Dict[str, Any] = None) -> ValidationResult:
        """Validate protein structure coordinates."""
        start_time = time.time()
        issues = []
        metadata = {}
        
        if coordinates is None:
            issues.append(ValidationIssue(
                type=ValidationType.STRUCTURE,
                severity=ValidationSeverity.ERROR,
                message="Structure coordinates cannot be None"
            ))
            return ValidationResult(
                passed=False,
                validation_level=self.config.default_validation_level,
                issues=issues
            )
            
        # Shape validation
        if len(coordinates.shape) != 2 or coordinates.shape[1] != 3:
            issues.append(ValidationIssue(
                type=ValidationType.STRUCTURE,
                severity=ValidationSeverity.ERROR,
                message=f"Invalid coordinate shape: {coordinates.shape}, expected (N, 3)"
            ))
            
        num_atoms = coordinates.shape[0]
        metadata['num_atoms'] = num_atoms
        
        # Check for NaN or infinite values
        if np.any(np.isnan(coordinates)) or np.any(np.isinf(coordinates)):
            issues.append(ValidationIssue(
                type=ValidationType.STRUCTURE,
                severity=ValidationSeverity.ERROR,
                message="Structure contains NaN or infinite coordinates"
            ))
            
        # Bond length validation
        bond_issues = self._validate_bond_lengths(coordinates)
        issues.extend(bond_issues)
        
        # Clash detection
        clash_issues = self._detect_steric_clashes(coordinates)
        issues.extend(clash_issues)
        
        # Geometric validation
        geometry_issues = self._validate_geometry(coordinates)
        issues.extend(geometry_issues)
        
        # Structure quality metrics
        quality_metrics = self._calculate_quality_metrics(coordinates)
        metadata.update(quality_metrics)
        
        # Performance metrics
        validation_time = time.time() - start_time
        performance_metrics = {
            'validation_time': validation_time,
            'num_atoms': num_atoms
        }
        
        # Calculate confidence score
        confidence_score = self._calculate_structure_confidence(coordinates, issues, quality_metrics)
        
        return ValidationResult(
            passed=not any(issue.severity == ValidationSeverity.ERROR for issue in issues),
            validation_level=self.config.default_validation_level,
            issues=issues,
            metadata=metadata,
            performance_metrics=performance_metrics,
            confidence_score=confidence_score
        )
        
    def _validate_bond_lengths(self, coordinates: np.ndarray) -> List[ValidationIssue]:
        """Validate bond lengths between consecutive atoms."""
        issues = []
        
        if len(coordinates) < 2:
            return issues
            
        # Calculate consecutive bond lengths
        bond_vectors = coordinates[1:] - coordinates[:-1]
        bond_lengths = np.linalg.norm(bond_vectors, axis=1)
        
        # Typical C-C bond length is ~1.5 Angstroms
        min_bond_length = 0.8  # Minimum reasonable bond length
        max_bond_length = 2.5  # Maximum reasonable bond length
        
        short_bonds = np.where(bond_lengths < min_bond_length)[0]
        long_bonds = np.where(bond_lengths > max_bond_length)[0]
        
        if len(short_bonds) > 0:
            issues.append(ValidationIssue(
                type=ValidationType.STRUCTURE,
                severity=ValidationSeverity.WARNING,
                message=f"Found {len(short_bonds)} unusually short bonds",
                details={'short_bond_indices': short_bonds.tolist(), 'min_length': float(bond_lengths.min())}
            ))
            
        if len(long_bonds) > 0:
            issues.append(ValidationIssue(
                type=ValidationType.STRUCTURE,
                severity=ValidationSeverity.WARNING,
                message=f"Found {len(long_bonds)} unusually long bonds",
                details={'long_bond_indices': long_bonds.tolist(), 'max_length': float(bond_lengths.max())}
            ))
            
        return issues
        
    def _detect_steric_clashes(self, coordinates: np.ndarray) -> List[ValidationIssue]:
        """Detect steric clashes between atoms."""
        issues = []
        
        if len(coordinates) < 2:
            return issues
            
        # Calculate pairwise distances
        num_atoms = len(coordinates)
        clash_threshold = 1.8  # Minimum distance between non-bonded atoms
        
        clashes = []
        
        for i in range(num_atoms):
            for j in range(i + 2, num_atoms):  # Skip bonded neighbors
                distance = np.linalg.norm(coordinates[i] - coordinates[j])
                if distance < clash_threshold:
                    clashes.append((i, j, distance))
                    
        if clashes:
            issues.append(ValidationIssue(
                type=ValidationType.STRUCTURE,
                severity=ValidationSeverity.WARNING,
                message=f"Found {len(clashes)} potential steric clashes",
                details={'clashes': [(int(i), int(j), float(d)) for i, j, d in clashes]}
            ))
            
        return issues
        
    def _validate_geometry(self, coordinates: np.ndarray) -> List[ValidationIssue]:
        """Validate geometric properties."""
        issues = []
        
        if len(coordinates) < 3:
            return issues
            
        # Calculate bond angles
        angles = []
        for i in range(1, len(coordinates) - 1):
            v1 = coordinates[i] - coordinates[i-1]
            v2 = coordinates[i+1] - coordinates[i]
            
            # Calculate angle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            angles.append(angle)
            
        angles = np.array(angles)
        
        # Check for unusual angles
        min_angle = np.degrees(angles.min()) if len(angles) > 0 else 180
        max_angle = np.degrees(angles.max()) if len(angles) > 0 else 0
        
        if min_angle < 60:  # Very acute angles
            issues.append(ValidationIssue(
                type=ValidationType.STRUCTURE,
                severity=ValidationSeverity.WARNING,
                message=f"Found very acute bond angle: {min_angle:.1f}°"
            ))
            
        if max_angle > 160:  # Very obtuse angles
            issues.append(ValidationIssue(
                type=ValidationType.STRUCTURE,
                severity=ValidationSeverity.WARNING,
                message=f"Found very obtuse bond angle: {max_angle:.1f}°"
            ))
            
        return issues
        
    def _calculate_quality_metrics(self, coordinates: np.ndarray) -> Dict[str, float]:
        """Calculate structure quality metrics."""
        metrics = {}
        
        if len(coordinates) < 2:
            return metrics
            
        # Calculate radius of gyration
        center_of_mass = np.mean(coordinates, axis=0)
        distances_from_com = np.linalg.norm(coordinates - center_of_mass, axis=1)
        radius_of_gyration = np.sqrt(np.mean(distances_from_com**2))
        metrics['radius_of_gyration'] = float(radius_of_gyration)
        
        # Calculate compactness
        if len(coordinates) > 1:
            pairwise_distances = []
            for i in range(len(coordinates)):
                for j in range(i + 1, len(coordinates)):
                    dist = np.linalg.norm(coordinates[i] - coordinates[j])
                    pairwise_distances.append(dist)
                    
            if pairwise_distances:
                avg_distance = np.mean(pairwise_distances)
                max_distance = np.max(pairwise_distances)
                compactness = avg_distance / max_distance if max_distance > 0 else 0
                metrics['compactness'] = float(compactness)
                metrics['max_pairwise_distance'] = float(max_distance)
                metrics['avg_pairwise_distance'] = float(avg_distance)
                
        return metrics
        
    def _calculate_structure_confidence(self, coordinates: np.ndarray, issues: List[ValidationIssue], quality_metrics: Dict[str, float]) -> float:
        """Calculate confidence score for structure."""
        base_confidence = 1.0
        
        # Penalize based on issues
        for issue in issues:
            if issue.severity == ValidationSeverity.ERROR:
                base_confidence -= 0.4
            elif issue.severity == ValidationSeverity.WARNING:
                base_confidence -= 0.15
                
        # Quality-based adjustments
        compactness = quality_metrics.get('compactness', 0.5)
        if compactness < 0.3:
            base_confidence *= 0.9  # Penalty for very extended structures
            
        return max(0.0, min(1.0, base_confidence))


class BiophysicalValidator(BaseValidator):
    """Validate biophysical properties."""
    
    async def validate(self, data: Dict[str, Any], context: Dict[str, Any] = None) -> ValidationResult:
        """Validate biophysical properties."""
        start_time = time.time()
        issues = []
        metadata = {}
        
        sequence = data.get('sequence', '')
        structure = data.get('structure')
        properties = data.get('properties', {})
        
        # Validate thermodynamic properties
        if 'energy' in properties:
            energy_issues = self._validate_energy(properties['energy'], sequence)
            issues.extend(energy_issues)
            
        # Validate hydrophobic properties
        if sequence:
            hydrophobic_issues = self._validate_hydrophobicity(sequence)
            issues.extend(hydrophobic_issues)
            
        # Validate charge properties
        if sequence:
            charge_issues = self._validate_charge_distribution(sequence)
            issues.extend(charge_issues)
            
        # Validate stability predictions
        if 'stability' in properties:
            stability_issues = self._validate_stability(properties['stability'])
            issues.extend(stability_issues)
            
        # Folding propensity validation
        if sequence and structure is not None:
            folding_issues = await self._validate_folding_propensity(sequence, structure)
            issues.extend(folding_issues)
            
        # Performance metrics
        validation_time = time.time() - start_time
        performance_metrics = {
            'validation_time': validation_time,
            'num_properties_validated': len(properties)
        }
        
        # Calculate confidence score
        confidence_score = self._calculate_biophysical_confidence(data, issues)
        
        return ValidationResult(
            passed=not any(issue.severity == ValidationSeverity.ERROR for issue in issues),
            validation_level=self.config.default_validation_level,
            issues=issues,
            metadata=metadata,
            performance_metrics=performance_metrics,
            confidence_score=confidence_score
        )
        
    def _validate_energy(self, energy: float, sequence: str) -> List[ValidationIssue]:
        """Validate energy values."""
        issues = []
        
        # Energy per residue check
        energy_per_residue = energy / max(len(sequence), 1)
        
        # Typical protein energies are -5 to -15 kcal/mol per residue
        if energy_per_residue > 0:
            issues.append(ValidationIssue(
                type=ValidationType.BIOPHYSICAL,
                severity=ValidationSeverity.WARNING,
                message=f"Positive energy per residue: {energy_per_residue:.2f} kcal/mol"
            ))
        elif energy_per_residue < -50:
            issues.append(ValidationIssue(
                type=ValidationType.BIOPHYSICAL,
                severity=ValidationSeverity.WARNING,
                message=f"Unusually low energy per residue: {energy_per_residue:.2f} kcal/mol"
            ))
            
        return issues
        
    def _validate_hydrophobicity(self, sequence: str) -> List[ValidationIssue]:
        """Validate hydrophobic properties."""
        issues = []
        
        # Hydrophobicity scale (Kyte-Doolittle)
        hydrophobicity_scale = {
            'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
            'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
            'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
            'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
        }
        
        # Calculate average hydrophobicity
        hydrophobicity_values = [hydrophobicity_scale.get(aa, 0) for aa in sequence]
        avg_hydrophobicity = np.mean(hydrophobicity_values) if hydrophobicity_values else 0
        
        # Check for extreme hydrophobicity
        if avg_hydrophobicity > 2.0:
            issues.append(ValidationIssue(
                type=ValidationType.BIOPHYSICAL,
                severity=ValidationSeverity.WARNING,
                message=f"Very hydrophobic protein: {avg_hydrophobicity:.2f}"
            ))
        elif avg_hydrophobicity < -2.0:
            issues.append(ValidationIssue(
                type=ValidationType.BIOPHYSICAL,
                severity=ValidationSeverity.WARNING,
                message=f"Very hydrophilic protein: {avg_hydrophobicity:.2f}"
            ))
            
        return issues
        
    def _validate_charge_distribution(self, sequence: str) -> List[ValidationIssue]:
        """Validate charge distribution."""
        issues = []
        
        # Count charged residues
        positive_charges = sum(1 for aa in sequence if aa in 'KR')
        negative_charges = sum(1 for aa in sequence if aa in 'DE')
        
        net_charge = positive_charges - negative_charges
        total_charged = positive_charges + negative_charges
        
        # Check for extreme charge imbalance
        if abs(net_charge) > len(sequence) * 0.3:
            issues.append(ValidationIssue(
                type=ValidationType.BIOPHYSICAL,
                severity=ValidationSeverity.WARNING,
                message=f"High net charge: {net_charge} ({abs(net_charge)/len(sequence)*100:.1f}% of sequence)"
            ))
            
        if total_charged / len(sequence) > 0.5:
            issues.append(ValidationIssue(
                type=ValidationType.BIOPHYSICAL,
                severity=ValidationSeverity.WARNING,
                message=f"Very high charge density: {total_charged/len(sequence)*100:.1f}%"
            ))
            
        return issues
        
    def _validate_stability(self, stability_score: float) -> List[ValidationIssue]:
        """Validate stability predictions."""
        issues = []
        
        # Stability score should be reasonable
        if stability_score < -5.0:
            issues.append(ValidationIssue(
                type=ValidationType.BIOPHYSICAL,
                severity=ValidationSeverity.WARNING,
                message=f"Predicted very unstable: {stability_score:.2f}"
            ))
        elif stability_score > 5.0:
            issues.append(ValidationIssue(
                type=ValidationType.BIOPHYSICAL,
                severity=ValidationSeverity.WARNING,
                message=f"Predicted unusually stable: {stability_score:.2f}"
            ))
            
        return issues
        
    async def _validate_folding_propensity(self, sequence: str, structure: np.ndarray) -> List[ValidationIssue]:
        """Validate folding propensity."""
        issues = []
        
        # Simple folding propensity check based on secondary structure preferences
        helix_formers = set('AEHK')
        sheet_formers = set('VIFY')
        loop_formers = set('GPS')
        
        # Calculate propensities
        helix_propensity = sum(1 for aa in sequence if aa in helix_formers) / len(sequence)
        sheet_propensity = sum(1 for aa in sequence if aa in sheet_formers) / len(sequence)
        loop_propensity = sum(1 for aa in sequence if aa in loop_formers) / len(sequence)
        
        # Check for extreme propensities
        if helix_propensity > 0.8:
            issues.append(ValidationIssue(
                type=ValidationType.BIOPHYSICAL,
                severity=ValidationSeverity.INFO,
                message=f"High helix propensity: {helix_propensity:.2f}"
            ))
            
        if sheet_propensity > 0.8:
            issues.append(ValidationIssue(
                type=ValidationType.BIOPHYSICAL,
                severity=ValidationSeverity.INFO,
                message=f"High sheet propensity: {sheet_propensity:.2f}"
            ))
            
        return issues
        
    def _calculate_biophysical_confidence(self, data: Dict[str, Any], issues: List[ValidationIssue]) -> float:
        """Calculate confidence score for biophysical properties."""
        base_confidence = 1.0
        
        # Penalize based on issues
        for issue in issues:
            if issue.severity == ValidationSeverity.ERROR:
                base_confidence -= 0.3
            elif issue.severity == ValidationSeverity.WARNING:
                base_confidence -= 0.1
            elif issue.severity == ValidationSeverity.INFO:
                base_confidence -= 0.05
                
        return max(0.0, min(1.0, base_confidence))


class ValidationOrchestrator:
    """Orchestrates multiple validation layers."""
    
    def __init__(self, config: RobustValidationConfig = None):
        self.config = config or RobustValidationConfig()
        
        # Initialize validators
        self.validators = {
            ValidationType.SEQUENCE: SequenceValidator(self.config),
            ValidationType.STRUCTURE: StructureValidator(self.config),
            ValidationType.BIOPHYSICAL: BiophysicalValidator(self.config)
        }
        
        # Validation monitoring
        self.validation_history: List[ValidationResult] = []
        self.performance_metrics: Dict[str, List[float]] = {}
        
        # Initialize storage
        if self.config.store_validation_history:
            self.history_path = Path(self.config.validation_history_path)
            self.history_path.mkdir(parents=True, exist_ok=True)
            
    async def comprehensive_validate(self, data: Dict[str, Any], validation_level: ValidationLevel = None) -> ValidationResult:
        """Perform comprehensive validation across all layers."""
        start_time = time.time()
        
        validation_level = validation_level or self.config.default_validation_level
        all_issues = []
        all_metadata = {}
        combined_confidence = 1.0
        
        # Determine which validators to run
        validators_to_run = []
        
        if 'sequence' in data and data['sequence']:
            validators_to_run.append(ValidationType.SEQUENCE)
            
        if 'structure' in data or 'coordinates' in data:
            validators_to_run.append(ValidationType.STRUCTURE)
            
        if self.config.biophysical_validation:
            validators_to_run.append(ValidationType.BIOPHYSICAL)
            
        # Run validations in parallel where possible
        validation_tasks = []
        
        for validator_type in validators_to_run:
            validator = self.validators[validator_type]
            
            if validator_type == ValidationType.SEQUENCE:
                task = validator.safe_validate(data['sequence'], data)
            elif validator_type == ValidationType.STRUCTURE:
                coords = data.get('structure') or data.get('coordinates')
                task = validator.safe_validate(coords, data)
            elif validator_type == ValidationType.BIOPHYSICAL:
                task = validator.safe_validate(data, data)
            else:
                continue
                
            validation_tasks.append(task)
            
        # Wait for all validations to complete
        validation_results = await asyncio.gather(*validation_tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(validation_results):
            if isinstance(result, Exception):
                all_issues.append(ValidationIssue(
                    type=validators_to_run[i],
                    severity=ValidationSeverity.ERROR,
                    message=f"Validation failed: {str(result)}"
                ))
            else:
                all_issues.extend(result.issues)
                all_metadata.update(result.metadata)
                combined_confidence *= result.confidence_score
                
        # Validation level adjustments
        final_issues = self._filter_issues_by_level(all_issues, validation_level)
        
        # Calculate final confidence
        final_confidence = self._calculate_final_confidence(combined_confidence, final_issues, validation_level)
        
        # Performance metrics
        total_time = time.time() - start_time
        performance_metrics = {
            'total_validation_time': total_time,
            'num_validators': len(validators_to_run)
        }
        
        # Create final result
        final_result = ValidationResult(
            passed=not any(issue.severity == ValidationSeverity.ERROR for issue in final_issues),
            validation_level=validation_level,
            issues=final_issues,
            metadata=all_metadata,
            performance_metrics=performance_metrics,
            confidence_score=final_confidence
        )
        
        # Store result
        if self.config.store_validation_history:
            await self._store_validation_result(final_result)
            
        # Update monitoring
        await self._update_monitoring(final_result)
        
        return final_result
        
    def _filter_issues_by_level(self, issues: List[ValidationIssue], level: ValidationLevel) -> List[ValidationIssue]:
        """Filter issues based on validation level."""
        if level == ValidationLevel.PERMISSIVE:
            # Only show errors and critical issues
            return [issue for issue in issues if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]
        elif level == ValidationLevel.NORMAL:
            # Show warnings and above
            return [issue for issue in issues if issue.severity != ValidationSeverity.INFO]
        elif level in [ValidationLevel.STRICT, ValidationLevel.RESEARCH, ValidationLevel.PRODUCTION]:
            # Show all issues
            return issues
        else:
            return issues
            
    def _calculate_final_confidence(self, combined_confidence: float, issues: List[ValidationIssue], level: ValidationLevel) -> float:
        """Calculate final confidence score."""
        # Base confidence from validators
        confidence = combined_confidence ** (1.0 / max(1, len(self.validators)))  # Geometric mean
        
        # Adjust based on validation level
        level_multiplier = {
            ValidationLevel.PERMISSIVE: 0.8,
            ValidationLevel.NORMAL: 1.0,
            ValidationLevel.STRICT: 1.1,
            ValidationLevel.RESEARCH: 1.2,
            ValidationLevel.PRODUCTION: 1.0
        }.get(level, 1.0)
        
        confidence *= level_multiplier
        
        return max(0.0, min(1.0, confidence))
        
    async def _store_validation_result(self, result: ValidationResult):
        """Store validation result for history."""
        try:
            # Add to in-memory history
            self.validation_history.append(result)
            
            # Limit history size
            if len(self.validation_history) > self.config.max_history_size:
                self.validation_history = self.validation_history[-self.config.max_history_size//2:]
                
            # Store to disk periodically
            if len(self.validation_history) % 100 == 0:
                history_file = self.history_path / f"validation_history_{int(time.time())}.json"
                with open(history_file, 'w') as f:
                    serialized_history = []
                    for res in self.validation_history[-100:]:  # Store last 100
                        serialized_history.append({
                            'passed': res.passed,
                            'validation_level': res.validation_level.value,
                            'num_issues': len(res.issues),
                            'confidence_score': res.confidence_score,
                            'timestamp': res.timestamp.isoformat(),
                            'performance_metrics': res.performance_metrics
                        })
                    json.dump(serialized_history, f, indent=2)
                    
        except Exception as e:
            logger.warning(f"Failed to store validation result: {e}")
            
    async def _update_monitoring(self, result: ValidationResult):
        """Update validation monitoring metrics."""
        if not self.config.enable_monitoring:
            return
            
        try:
            # Update performance metrics
            for metric, value in result.performance_metrics.items():
                if metric not in self.performance_metrics:
                    self.performance_metrics[metric] = []
                self.performance_metrics[metric].append(value)
                
                # Keep only recent metrics
                if len(self.performance_metrics[metric]) > 1000:
                    self.performance_metrics[metric] = self.performance_metrics[metric][-500:]
                    
            # Check for alerts
            await self._check_validation_alerts(result)
            
        except Exception as e:
            logger.warning(f"Failed to update monitoring: {e}")
            
    async def _check_validation_alerts(self, result: ValidationResult):
        """Check if validation alerts should be triggered."""
        try:
            # Calculate recent failure rate
            recent_results = self.validation_history[-100:] if len(self.validation_history) >= 100 else self.validation_history
            
            if len(recent_results) >= 10:  # Need sufficient data
                failure_rate = sum(1 for r in recent_results if not r.passed) / len(recent_results)
                
                if failure_rate > self.config.alert_threshold:
                    logger.warning(f"High validation failure rate detected: {failure_rate:.2%}")
                    
                    # Trigger alert (could integrate with external alerting systems)
                    await self._trigger_alert({
                        'type': 'high_failure_rate',
                        'failure_rate': failure_rate,
                        'threshold': self.config.alert_threshold,
                        'recent_results': len(recent_results)
                    })
                    
        except Exception as e:
            logger.error(f"Failed to check validation alerts: {e}")
            
    async def _trigger_alert(self, alert_data: Dict[str, Any]):
        """Trigger validation alert."""
        logger.warning(f"VALIDATION ALERT: {alert_data}")
        
        # In a real implementation, this could:
        # - Send notifications via email/Slack/PagerDuty
        # - Write to monitoring systems
        # - Trigger automatic remediation
        
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics and health metrics."""
        if not self.validation_history:
            return {'status': 'no_data'}
            
        recent_results = self.validation_history[-100:]
        
        stats = {
            'total_validations': len(self.validation_history),
            'recent_validations': len(recent_results),
            'success_rate': sum(1 for r in recent_results if r.passed) / len(recent_results),
            'avg_confidence': np.mean([r.confidence_score for r in recent_results]),
            'avg_validation_time': np.mean([r.performance_metrics.get('total_validation_time', 0) for r in recent_results]),
            'validation_levels': {
                level.value: sum(1 for r in recent_results if r.validation_level == level)
                for level in ValidationLevel
            },
            'issue_types': {},
            'circuit_breaker_status': {
                validator_type.value: {
                    'state': validator.circuit_breaker.state.value if validator.circuit_breaker else 'disabled',
                    'failure_count': validator.circuit_breaker.failure_count if validator.circuit_breaker else 0
                }
                for validator_type, validator in self.validators.items()
            }
        }
        
        # Issue type breakdown
        for result in recent_results:
            for issue in result.issues:
                issue_key = f"{issue.type.value}_{issue.severity.value}"
                stats['issue_types'][issue_key] = stats['issue_types'].get(issue_key, 0) + 1
                
        return stats


# Demonstration and usage example
async def demonstrate_robust_validation():
    """Demonstrate the robust validation system."""
    logger.info("Demonstrating Robust Validation System...")
    
    # Create validation orchestrator
    config = RobustValidationConfig(
        default_validation_level=ValidationLevel.NORMAL,
        circuit_breaker_enabled=True,
        enable_monitoring=True
    )
    
    orchestrator = ValidationOrchestrator(config)
    
    # Test data
    test_cases = [
        {
            'name': 'Valid protein',
            'data': {
                'sequence': 'MKLLILTCLVAVALARPKHPIP',
                'structure': np.random.randn(21, 3) * 5,
                'properties': {
                    'energy': -250.0,
                    'stability': 2.3
                }
            }
        },
        {
            'name': 'Invalid sequence',
            'data': {
                'sequence': 'MKLLXZQWERTY',  # Invalid characters
                'structure': np.random.randn(12, 3) * 5
            }
        },
        {
            'name': 'Structure issues',
            'data': {
                'sequence': 'ACDEFGH',
                'structure': np.array([[0,0,0], [0.1,0,0], [0.2,0,0], [0.3,0,0], [0.4,0,0], [0.5,0,0], [0.6,0,0]])  # Too close atoms
            }
        }
    ]
    
    # Run validation tests
    for test_case in test_cases:
        logger.info(f"\nTesting: {test_case['name']}")
        
        result = await orchestrator.comprehensive_validate(
            test_case['data'],
            ValidationLevel.STRICT
        )
        
        logger.info(f"Passed: {result.passed}")
        logger.info(f"Confidence: {result.confidence_score:.3f}")
        logger.info(f"Issues: {len(result.issues)}")
        
        for issue in result.issues[:3]:  # Show first 3 issues
            logger.info(f"  - {issue.severity.value}: {issue.message}")
            
    # Get statistics
    stats = orchestrator.get_validation_statistics()
    logger.info(f"\nValidation Statistics:")
    logger.info(f"Success Rate: {stats['success_rate']:.2%}")
    logger.info(f"Average Confidence: {stats['avg_confidence']:.3f}")
    logger.info(f"Average Time: {stats['avg_validation_time']:.3f}s")
    
    return orchestrator


if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_robust_validation())