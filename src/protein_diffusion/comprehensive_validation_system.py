"""
Comprehensive Validation System for Protein Diffusion Design Lab

This module provides multi-layer validation, data integrity checks, and
quality assurance for all protein design workflows and data pipelines.
"""

import re
import json
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Union, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging

# Mock imports for environments without full dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

try:
    from Bio.Seq import Seq
    from Bio.SeqUtils import molecular_weight, seq1, seq3
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    # Mock Bio classes for basic functionality
    class Seq:
        def __init__(self, sequence):
            self.sequence = sequence
        def __str__(self):
            return self.sequence

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels."""
    STRICT = "strict"
    NORMAL = "normal"
    PERMISSIVE = "permissive"
    EMERGENCY = "emergency"


class ValidationResult(Enum):
    """Validation result types."""
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    ERROR = "error"


class DataType(Enum):
    """Types of data that can be validated."""
    PROTEIN_SEQUENCE = "protein_sequence"
    DNA_SEQUENCE = "dna_sequence"
    RNA_SEQUENCE = "rna_sequence"
    STRUCTURE_PDB = "structure_pdb"
    BINDING_AFFINITY = "binding_affinity"
    GENERATION_PARAMETERS = "generation_parameters"
    API_REQUEST = "api_request"
    USER_INPUT = "user_input"
    SYSTEM_CONFIG = "system_config"
    EXPERIMENTAL_DATA = "experimental_data"


@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    issue_id: str
    timestamp: float = field(default_factory=time.time)
    data_type: DataType = DataType.USER_INPUT
    validation_level: ValidationLevel = ValidationLevel.NORMAL
    result: ValidationResult = ValidationResult.FAIL
    severity: str = "medium"
    message: str = ""
    field_name: Optional[str] = None
    invalid_value: Any = None
    expected_format: Optional[str] = None
    suggestion: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationSummary:
    """Summary of validation results."""
    total_checks: int = 0
    passed_checks: int = 0
    warning_checks: int = 0
    failed_checks: int = 0
    error_checks: int = 0
    issues: List[ValidationIssue] = field(default_factory=list)
    validation_time_ms: float = 0.0
    data_integrity_score: float = 0.0
    overall_result: ValidationResult = ValidationResult.PASS


@dataclass
class ValidationConfig:
    """Configuration for the validation system."""
    default_level: ValidationLevel = ValidationLevel.NORMAL
    enable_data_integrity_checks: bool = True
    enable_security_validation: bool = True
    enable_performance_validation: bool = True
    enable_biological_validation: bool = True
    max_sequence_length: int = 10000
    min_sequence_length: int = 1
    allowed_amino_acids: str = "ACDEFGHIKLMNPQRSTVWY"
    allowed_nucleotides: str = "ATCG"
    enable_checksum_validation: bool = True
    enable_format_validation: bool = True
    validation_cache_size: int = 1000
    enable_async_validation: bool = True


class BaseValidator(ABC):
    """Abstract base class for validators."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        
    @abstractmethod
    def validate(self, data: Any, level: ValidationLevel = None) -> List[ValidationIssue]:
        """Validate data and return list of issues."""
        pass
        
    @abstractmethod
    def get_data_type(self) -> DataType:
        """Get the data type this validator handles."""
        pass
        
    def create_issue(
        self,
        message: str,
        result: ValidationResult = ValidationResult.FAIL,
        field_name: Optional[str] = None,
        invalid_value: Any = None,
        expected_format: Optional[str] = None,
        suggestion: Optional[str] = None,
        severity: str = "medium"
    ) -> ValidationIssue:
        """Create a validation issue."""
        return ValidationIssue(
            issue_id=f"{self.get_data_type().value}_{int(time.time() * 1000)}",
            data_type=self.get_data_type(),
            result=result,
            message=message,
            field_name=field_name,
            invalid_value=invalid_value,
            expected_format=expected_format,
            suggestion=suggestion,
            severity=severity
        )


class ProteinSequenceValidator(BaseValidator):
    """Validator for protein sequences."""
    
    def get_data_type(self) -> DataType:
        return DataType.PROTEIN_SEQUENCE
        
    def validate(self, sequence: str, level: ValidationLevel = None) -> List[ValidationIssue]:
        """Validate protein sequence."""
        issues = []
        level = level or self.config.default_level
        
        if not sequence:
            issues.append(self.create_issue(
                "Protein sequence is empty",
                ValidationResult.FAIL,
                suggestion="Provide a valid protein sequence"
            ))
            return issues
            
        # Basic format validation
        if not isinstance(sequence, str):
            issues.append(self.create_issue(
                "Protein sequence must be a string",
                ValidationResult.FAIL,
                invalid_value=type(sequence).__name__,
                expected_format="string",
                suggestion="Convert sequence to string format"
            ))
            return issues
            
        # Clean sequence (remove whitespace and convert to uppercase)
        clean_sequence = sequence.strip().upper().replace(' ', '').replace('\n', '').replace('\t', '')
        
        # Length validation
        if len(clean_sequence) < self.config.min_sequence_length:
            issues.append(self.create_issue(
                f"Protein sequence too short: {len(clean_sequence)} amino acids",
                ValidationResult.FAIL,
                invalid_value=len(clean_sequence),
                expected_format=f"minimum {self.config.min_sequence_length} amino acids",
                suggestion="Provide a longer protein sequence"
            ))
            
        if len(clean_sequence) > self.config.max_sequence_length:
            if level == ValidationLevel.STRICT:
                issues.append(self.create_issue(
                    f"Protein sequence too long: {len(clean_sequence)} amino acids",
                    ValidationResult.FAIL,
                    invalid_value=len(clean_sequence),
                    expected_format=f"maximum {self.config.max_sequence_length} amino acids",
                    suggestion="Truncate sequence or increase max_sequence_length"
                ))
            else:
                issues.append(self.create_issue(
                    f"Protein sequence is very long: {len(clean_sequence)} amino acids",
                    ValidationResult.WARNING,
                    invalid_value=len(clean_sequence),
                    suggestion="Consider if this length is intentional"
                ))
                
        # Amino acid composition validation
        invalid_chars = []
        for char in clean_sequence:
            if char not in self.config.allowed_amino_acids:
                if char not in invalid_chars:  # Avoid duplicates
                    invalid_chars.append(char)
                    
        if invalid_chars:
            if level in [ValidationLevel.STRICT, ValidationLevel.NORMAL]:
                issues.append(self.create_issue(
                    f"Invalid amino acids found: {', '.join(invalid_chars)}",
                    ValidationResult.FAIL,
                    invalid_value=invalid_chars,
                    expected_format=f"only {self.config.allowed_amino_acids}",
                    suggestion="Remove or replace invalid characters"
                ))
            else:
                issues.append(self.create_issue(
                    f"Non-standard amino acids found: {', '.join(invalid_chars)}",
                    ValidationResult.WARNING,
                    invalid_value=invalid_chars,
                    suggestion="Verify non-standard amino acids are intentional"
                ))
                
        # Biological validation
        if self.config.enable_biological_validation and not issues:
            bio_issues = self._validate_biological_properties(clean_sequence, level)
            issues.extend(bio_issues)
            
        return issues
        
    def _validate_biological_properties(self, sequence: str, level: ValidationLevel) -> List[ValidationIssue]:
        """Validate biological properties of protein sequence."""
        issues = []
        
        # Check for extremely unusual amino acid compositions
        seq_length = len(sequence)
        
        # Proline content check (high proline can affect structure)
        proline_count = sequence.count('P')
        proline_percentage = (proline_count / seq_length) * 100 if seq_length > 0 else 0
        
        if proline_percentage > 20:  # >20% proline is unusual
            if level == ValidationLevel.STRICT:
                issues.append(self.create_issue(
                    f"Unusually high proline content: {proline_percentage:.1f}%",
                    ValidationResult.WARNING,
                    invalid_value=proline_percentage,
                    suggestion="High proline content may affect protein folding"
                ))
                
        # Check for long homopolymeric stretches
        for aa in self.config.allowed_amino_acids:
            max_stretch = self._find_longest_stretch(sequence, aa)
            if max_stretch > 10:  # Stretches >10 are unusual
                issues.append(self.create_issue(
                    f"Long stretch of {aa}: {max_stretch} consecutive residues",
                    ValidationResult.WARNING,
                    invalid_value=max_stretch,
                    suggestion="Long homopolymeric stretches may cause aggregation"
                ))
                
        # Check for start/stop codons if translating from DNA
        if sequence.startswith('M') and level == ValidationLevel.PERMISSIVE:
            # This is expected for many proteins
            pass
        elif not sequence.startswith('M') and level == ValidationLevel.STRICT:
            issues.append(self.create_issue(
                "Protein sequence does not start with methionine (M)",
                ValidationResult.WARNING,
                suggestion="Consider if N-terminal methionine is needed"
            ))
            
        return issues
        
    def _find_longest_stretch(self, sequence: str, amino_acid: str) -> int:
        """Find the longest consecutive stretch of an amino acid."""
        max_stretch = 0
        current_stretch = 0
        
        for aa in sequence:
            if aa == amino_acid:
                current_stretch += 1
                max_stretch = max(max_stretch, current_stretch)
            else:
                current_stretch = 0
                
        return max_stretch


class StructurePDBValidator(BaseValidator):
    """Validator for PDB structure data."""
    
    def get_data_type(self) -> DataType:
        return DataType.STRUCTURE_PDB
        
    def validate(self, pdb_data: str, level: ValidationLevel = None) -> List[ValidationIssue]:
        """Validate PDB structure data."""
        issues = []
        level = level or self.config.default_level
        
        if not pdb_data:
            issues.append(self.create_issue(
                "PDB data is empty",
                ValidationResult.FAIL,
                suggestion="Provide valid PDB structure data"
            ))
            return issues
            
        # Basic format validation
        if not isinstance(pdb_data, str):
            issues.append(self.create_issue(
                "PDB data must be a string",
                ValidationResult.FAIL,
                invalid_value=type(pdb_data).__name__,
                expected_format="string",
                suggestion="Convert PDB data to string format"
            ))
            return issues
            
        lines = pdb_data.strip().split('\n')
        
        # Check for required PDB records
        has_header = any(line.startswith('HEADER') for line in lines)
        has_atom_records = any(line.startswith('ATOM') for line in lines)
        has_end = any(line.startswith('END') for line in lines)
        
        if not has_header and level in [ValidationLevel.STRICT, ValidationLevel.NORMAL]:
            issues.append(self.create_issue(
                "Missing HEADER record in PDB data",
                ValidationResult.WARNING,
                suggestion="Add HEADER record with structure information"
            ))
            
        if not has_atom_records:
            issues.append(self.create_issue(
                "No ATOM records found in PDB data",
                ValidationResult.FAIL,
                suggestion="PDB data must contain ATOM records"
            ))
            
        if not has_end and level == ValidationLevel.STRICT:
            issues.append(self.create_issue(
                "Missing END record in PDB data",
                ValidationResult.WARNING,
                suggestion="Add END record to properly terminate PDB file"
            ))
            
        # Validate ATOM record format
        if has_atom_records:
            atom_issues = self._validate_atom_records(lines, level)
            issues.extend(atom_issues)
            
        return issues
        
    def _validate_atom_records(self, lines: List[str], level: ValidationLevel) -> List[ValidationIssue]:
        """Validate ATOM record formatting."""
        issues = []
        atom_count = 0
        
        for line_num, line in enumerate(lines, 1):
            if not line.startswith('ATOM'):
                continue
                
            atom_count += 1
            
            # Check line length (PDB records should be exactly 80 characters)
            if len(line) < 54:  # Minimum required fields
                issues.append(self.create_issue(
                    f"ATOM record too short at line {line_num}",
                    ValidationResult.FAIL,
                    invalid_value=len(line),
                    expected_format="at least 54 characters",
                    suggestion="Ensure all required ATOM fields are present"
                ))
                continue
                
            try:
                # Extract key fields (using PDB format specification)
                atom_num = line[6:11].strip()
                atom_name = line[12:16].strip()
                residue_name = line[17:20].strip()
                chain_id = line[21:22].strip()
                residue_num = line[22:26].strip()
                x_coord = line[30:38].strip()
                y_coord = line[38:46].strip()
                z_coord = line[46:54].strip()
                
                # Validate numeric fields
                try:
                    int(atom_num)
                except ValueError:
                    issues.append(self.create_issue(
                        f"Invalid atom number at line {line_num}: {atom_num}",
                        ValidationResult.FAIL,
                        field_name="atom_number",
                        invalid_value=atom_num,
                        suggestion="Atom number must be an integer"
                    ))
                    
                try:
                    int(residue_num)
                except ValueError:
                    issues.append(self.create_issue(
                        f"Invalid residue number at line {line_num}: {residue_num}",
                        ValidationResult.FAIL,
                        field_name="residue_number", 
                        invalid_value=residue_num,
                        suggestion="Residue number must be an integer"
                    ))
                    
                # Validate coordinates
                for coord_name, coord_value in [('x', x_coord), ('y', y_coord), ('z', z_coord)]:
                    try:
                        coord_float = float(coord_value)
                        # Check for reasonable coordinate ranges
                        if abs(coord_float) > 9999.999:
                            issues.append(self.create_issue(
                                f"Unreasonable {coord_name} coordinate at line {line_num}: {coord_float}",
                                ValidationResult.WARNING,
                                field_name=f"{coord_name}_coordinate",
                                invalid_value=coord_float,
                                suggestion="Coordinate values seem unusually large"
                            ))
                    except ValueError:
                        issues.append(self.create_issue(
                            f"Invalid {coord_name} coordinate at line {line_num}: {coord_value}",
                            ValidationResult.FAIL,
                            field_name=f"{coord_name}_coordinate",
                            invalid_value=coord_value,
                            suggestion="Coordinates must be numeric values"
                        ))
                        
                # Validate residue name (should be standard amino acid)
                if self.config.enable_biological_validation:
                    standard_residues = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 
                                        'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 
                                        'THR', 'TRP', 'TYR', 'VAL']
                    if residue_name not in standard_residues and level in [ValidationLevel.STRICT, ValidationLevel.NORMAL]:
                        issues.append(self.create_issue(
                            f"Non-standard residue at line {line_num}: {residue_name}",
                            ValidationResult.WARNING,
                            field_name="residue_name",
                            invalid_value=residue_name,
                            suggestion="Verify non-standard residue is intentional"
                        ))
                        
            except Exception as e:
                issues.append(self.create_issue(
                    f"Error parsing ATOM record at line {line_num}: {str(e)}",
                    ValidationResult.ERROR,
                    suggestion="Check ATOM record format"
                ))
                
        if atom_count == 0:
            issues.append(self.create_issue(
                "No valid ATOM records found",
                ValidationResult.FAIL,
                suggestion="Ensure PDB data contains properly formatted ATOM records"
            ))
            
        return issues


class BindingAffinityValidator(BaseValidator):
    """Validator for binding affinity data."""
    
    def get_data_type(self) -> DataType:
        return DataType.BINDING_AFFINITY
        
    def validate(self, affinity_data: Dict[str, Any], level: ValidationLevel = None) -> List[ValidationIssue]:
        """Validate binding affinity data."""
        issues = []
        level = level or self.config.default_level
        
        if not affinity_data:
            issues.append(self.create_issue(
                "Binding affinity data is empty",
                ValidationResult.FAIL,
                suggestion="Provide binding affinity measurements"
            ))
            return issues
            
        # Required fields
        required_fields = ['binding_affinity_kcal_mol', 'confidence']
        for field in required_fields:
            if field not in affinity_data:
                issues.append(self.create_issue(
                    f"Missing required field: {field}",
                    ValidationResult.FAIL,
                    field_name=field,
                    suggestion=f"Include {field} in affinity data"
                ))
                
        # Validate binding affinity value
        if 'binding_affinity_kcal_mol' in affinity_data:
            affinity = affinity_data['binding_affinity_kcal_mol']
            try:
                affinity_float = float(affinity)
                
                # Reasonable range check (typical protein-protein interactions: -15 to +5 kcal/mol)
                if affinity_float < -20 or affinity_float > 10:
                    severity = ValidationResult.WARNING if level == ValidationLevel.PERMISSIVE else ValidationResult.FAIL
                    issues.append(self.create_issue(
                        f"Unusual binding affinity value: {affinity_float} kcal/mol",
                        severity,
                        field_name="binding_affinity_kcal_mol",
                        invalid_value=affinity_float,
                        expected_format="-20 to +10 kcal/mol (typical range)",
                        suggestion="Verify binding affinity measurement"
                    ))
                    
            except (ValueError, TypeError):
                issues.append(self.create_issue(
                    f"Invalid binding affinity value: {affinity}",
                    ValidationResult.FAIL,
                    field_name="binding_affinity_kcal_mol",
                    invalid_value=affinity,
                    expected_format="numeric value in kcal/mol",
                    suggestion="Provide numeric binding affinity value"
                ))
                
        # Validate confidence value
        if 'confidence' in affinity_data:
            confidence = affinity_data['confidence']
            try:
                confidence_float = float(confidence)
                
                if confidence_float < 0 or confidence_float > 1:
                    issues.append(self.create_issue(
                        f"Confidence value out of range: {confidence_float}",
                        ValidationResult.FAIL,
                        field_name="confidence",
                        invalid_value=confidence_float,
                        expected_format="0.0 to 1.0",
                        suggestion="Confidence should be between 0 and 1"
                    ))
                elif confidence_float < 0.5 and level == ValidationLevel.STRICT:
                    issues.append(self.create_issue(
                        f"Low confidence value: {confidence_float}",
                        ValidationResult.WARNING,
                        field_name="confidence",
                        invalid_value=confidence_float,
                        suggestion="Consider if low confidence results should be used"
                    ))
                    
            except (ValueError, TypeError):
                issues.append(self.create_issue(
                    f"Invalid confidence value: {confidence}",
                    ValidationResult.FAIL,
                    field_name="confidence",
                    invalid_value=confidence,
                    expected_format="numeric value between 0 and 1",
                    suggestion="Provide numeric confidence value"
                ))
                
        return issues


class GenerationParametersValidator(BaseValidator):
    """Validator for protein generation parameters."""
    
    def get_data_type(self) -> DataType:
        return DataType.GENERATION_PARAMETERS
        
    def validate(self, params: Dict[str, Any], level: ValidationLevel = None) -> List[ValidationIssue]:
        """Validate protein generation parameters."""
        issues = []
        level = level or self.config.default_level
        
        if not params:
            issues.append(self.create_issue(
                "Generation parameters are empty",
                ValidationResult.FAIL,
                suggestion="Provide generation parameters"
            ))
            return issues
            
        # Validate temperature parameter
        if 'temperature' in params:
            temp = params['temperature']
            try:
                temp_float = float(temp)
                
                if temp_float <= 0:
                    issues.append(self.create_issue(
                        f"Temperature must be positive: {temp_float}",
                        ValidationResult.FAIL,
                        field_name="temperature",
                        invalid_value=temp_float,
                        expected_format="positive number",
                        suggestion="Use temperature > 0"
                    ))
                elif temp_float > 2.0:
                    severity = ValidationResult.WARNING if level == ValidationLevel.PERMISSIVE else ValidationResult.FAIL
                    issues.append(self.create_issue(
                        f"Very high temperature: {temp_float}",
                        severity,
                        field_name="temperature",
                        invalid_value=temp_float,
                        suggestion="High temperature may produce low-quality results"
                    ))
                    
            except (ValueError, TypeError):
                issues.append(self.create_issue(
                    f"Invalid temperature value: {temp}",
                    ValidationResult.FAIL,
                    field_name="temperature",
                    invalid_value=temp,
                    expected_format="numeric value",
                    suggestion="Provide numeric temperature value"
                ))
                
        # Validate num_proteins parameter
        if 'num_proteins' in params:
            num_proteins = params['num_proteins']
            try:
                num_int = int(num_proteins)
                
                if num_int <= 0:
                    issues.append(self.create_issue(
                        f"Number of proteins must be positive: {num_int}",
                        ValidationResult.FAIL,
                        field_name="num_proteins",
                        invalid_value=num_int,
                        expected_format="positive integer",
                        suggestion="Use num_proteins > 0"
                    ))
                elif num_int > 10000:
                    severity = ValidationResult.WARNING if level == ValidationLevel.PERMISSIVE else ValidationResult.FAIL
                    issues.append(self.create_issue(
                        f"Very large number of proteins: {num_int}",
                        severity,
                        field_name="num_proteins",
                        invalid_value=num_int,
                        suggestion="Large numbers may take long time and resources"
                    ))
                    
            except (ValueError, TypeError):
                issues.append(self.create_issue(
                    f"Invalid num_proteins value: {num_proteins}",
                    ValidationResult.FAIL,
                    field_name="num_proteins",
                    invalid_value=num_proteins,
                    expected_format="integer",
                    suggestion="Provide integer number of proteins"
                ))
                
        # Validate motif parameter
        if 'motif' in params:
            motif = params['motif']
            if not isinstance(motif, str):
                issues.append(self.create_issue(
                    f"Motif must be a string: {type(motif).__name__}",
                    ValidationResult.FAIL,
                    field_name="motif",
                    invalid_value=type(motif).__name__,
                    expected_format="string",
                    suggestion="Provide motif as string"
                ))
            elif not motif.strip():
                issues.append(self.create_issue(
                    "Motif cannot be empty",
                    ValidationResult.FAIL,
                    field_name="motif",
                    suggestion="Provide a valid motif description"
                ))
                
        return issues


class SecurityValidator(BaseValidator):
    """Validator for security-related checks."""
    
    def get_data_type(self) -> DataType:
        return DataType.USER_INPUT
        
    def validate(self, data: Any, level: ValidationLevel = None) -> List[ValidationIssue]:
        """Validate data for security issues."""
        issues = []
        level = level or self.config.default_level
        
        if not self.config.enable_security_validation:
            return issues
            
        if isinstance(data, str):
            issues.extend(self._validate_string_security(data, level))
        elif isinstance(data, dict):
            issues.extend(self._validate_dict_security(data, level))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                item_issues = self.validate(item, level)
                for issue in item_issues:
                    issue.field_name = f"[{i}].{issue.field_name}" if issue.field_name else f"[{i}]"
                issues.extend(item_issues)
                
        return issues
        
    def _validate_string_security(self, text: str, level: ValidationLevel) -> List[ValidationIssue]:
        """Validate string for security issues."""
        issues = []
        
        # Check for potential injection attacks
        sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'UNION', 'EXEC']
        for keyword in sql_keywords:
            if keyword.upper() in text.upper():
                issues.append(self.create_issue(
                    f"Potential SQL injection detected: {keyword}",
                    ValidationResult.FAIL,
                    invalid_value=keyword,
                    suggestion="Remove SQL keywords from input",
                    severity="high"
                ))
                
        # Check for script injection
        script_patterns = ['<script', 'javascript:', 'onload=', 'onerror=']
        for pattern in script_patterns:
            if pattern.lower() in text.lower():
                issues.append(self.create_issue(
                    f"Potential script injection detected: {pattern}",
                    ValidationResult.FAIL,
                    invalid_value=pattern,
                    suggestion="Remove script elements from input",
                    severity="high"
                ))
                
        # Check for path traversal
        if '../' in text or '..\\' in text:
            issues.append(self.create_issue(
                "Potential path traversal detected",
                ValidationResult.FAIL,
                invalid_value="../ or ..\\",
                suggestion="Remove path traversal sequences",
                severity="high"
            ))
            
        # Check for very long strings (potential DoS)
        if len(text) > 100000:
            issues.append(self.create_issue(
                f"Extremely long input: {len(text)} characters",
                ValidationResult.WARNING,
                invalid_value=len(text),
                suggestion="Consider if such long input is necessary",
                severity="medium"
            ))
            
        return issues
        
    def _validate_dict_security(self, data_dict: Dict[str, Any], level: ValidationLevel) -> List[ValidationIssue]:
        """Validate dictionary for security issues."""
        issues = []
        
        # Check for sensitive key names
        sensitive_keys = ['password', 'secret', 'token', 'key', 'auth', 'credential']
        for key in data_dict.keys():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                issues.append(self.create_issue(
                    f"Sensitive key detected: {key}",
                    ValidationResult.WARNING,
                    field_name=key,
                    suggestion="Ensure sensitive data is properly handled",
                    severity="medium"
                ))
                
        # Recursively validate values
        for key, value in data_dict.items():
            value_issues = self.validate(value, level)
            for issue in value_issues:
                issue.field_name = f"{key}.{issue.field_name}" if issue.field_name else key
            issues.extend(value_issues)
            
        return issues


class DataIntegrityValidator:
    """Validates data integrity using checksums and consistency checks."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        
    def validate_checksum(self, data: str, expected_checksum: str, algorithm: str = "sha256") -> bool:
        """Validate data checksum."""
        if not self.config.enable_checksum_validation:
            return True
            
        if algorithm == "sha256":
            actual_checksum = hashlib.sha256(data.encode()).hexdigest()
        elif algorithm == "md5":
            actual_checksum = hashlib.md5(data.encode()).hexdigest()
        else:
            logger.warning(f"Unsupported checksum algorithm: {algorithm}")
            return False
            
        return actual_checksum == expected_checksum
        
    def generate_checksum(self, data: str, algorithm: str = "sha256") -> str:
        """Generate checksum for data."""
        if algorithm == "sha256":
            return hashlib.sha256(data.encode()).hexdigest()
        elif algorithm == "md5":
            return hashlib.md5(data.encode()).hexdigest()
        else:
            raise ValueError(f"Unsupported checksum algorithm: {algorithm}")


class ComprehensiveValidationSystem:
    """
    Comprehensive Validation System
    
    Provides multi-layer validation for all protein design data types
    with configurable strictness levels and comprehensive reporting.
    """
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        
        # Initialize validators
        self.validators: Dict[DataType, BaseValidator] = {
            DataType.PROTEIN_SEQUENCE: ProteinSequenceValidator(config),
            DataType.STRUCTURE_PDB: StructurePDBValidator(config),
            DataType.BINDING_AFFINITY: BindingAffinityValidator(config),
            DataType.GENERATION_PARAMETERS: GenerationParametersValidator(config),
            DataType.USER_INPUT: SecurityValidator(config),
        }
        
        # Data integrity validator
        self.integrity_validator = DataIntegrityValidator(config)
        
        # Validation cache
        self.validation_cache: Dict[str, ValidationSummary] = {}
        
        # Statistics
        self.validation_stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'cached_validations': 0
        }
        
        logger.info("Comprehensive Validation System initialized")
        
    def validate_data(
        self,
        data: Any,
        data_type: DataType,
        level: ValidationLevel = None,
        enable_cache: bool = True
    ) -> ValidationSummary:
        """Validate data comprehensively."""
        start_time = time.time()
        level = level or self.config.default_level
        
        # Generate cache key
        cache_key = None
        if enable_cache:
            data_str = json.dumps(data, sort_keys=True, default=str)
            cache_key = hashlib.md5(f"{data_type.value}_{level.value}_{data_str}".encode()).hexdigest()
            
            if cache_key in self.validation_cache:
                self.validation_stats['cached_validations'] += 1
                return self.validation_cache[cache_key]
                
        # Perform validation
        all_issues = []
        
        # Data type specific validation
        if data_type in self.validators:
            validator = self.validators[data_type]
            issues = validator.validate(data, level)
            all_issues.extend(issues)
        else:
            logger.warning(f"No validator available for data type: {data_type}")
            
        # Security validation (for all data types if enabled)
        if self.config.enable_security_validation and data_type != DataType.USER_INPUT:
            security_validator = self.validators[DataType.USER_INPUT]
            security_issues = security_validator.validate(data, level)
            all_issues.extend(security_issues)
            
        # Create validation summary
        summary = self._create_validation_summary(all_issues, time.time() - start_time)
        
        # Update statistics
        self.validation_stats['total_validations'] += 1
        if summary.overall_result == ValidationResult.PASS:
            self.validation_stats['passed_validations'] += 1
        else:
            self.validation_stats['failed_validations'] += 1
            
        # Cache result
        if enable_cache and cache_key:
            self.validation_cache[cache_key] = summary
            # Cleanup old cache entries if needed
            if len(self.validation_cache) > self.config.validation_cache_size:
                oldest_key = min(self.validation_cache.keys())
                del self.validation_cache[oldest_key]
                
        return summary
        
    def validate_protein_sequence(
        self,
        sequence: str,
        level: ValidationLevel = None
    ) -> ValidationSummary:
        """Validate protein sequence."""
        return self.validate_data(sequence, DataType.PROTEIN_SEQUENCE, level)
        
    def validate_structure_pdb(
        self,
        pdb_data: str,
        level: ValidationLevel = None
    ) -> ValidationSummary:
        """Validate PDB structure data."""
        return self.validate_data(pdb_data, DataType.STRUCTURE_PDB, level)
        
    def validate_binding_affinity(
        self,
        affinity_data: Dict[str, Any],
        level: ValidationLevel = None
    ) -> ValidationSummary:
        """Validate binding affinity data."""
        return self.validate_data(affinity_data, DataType.BINDING_AFFINITY, level)
        
    def validate_generation_parameters(
        self,
        parameters: Dict[str, Any],
        level: ValidationLevel = None
    ) -> ValidationSummary:
        """Validate protein generation parameters."""
        return self.validate_data(parameters, DataType.GENERATION_PARAMETERS, level)
        
    def validate_batch(
        self,
        data_items: List[Tuple[Any, DataType]],
        level: ValidationLevel = None
    ) -> List[ValidationSummary]:
        """Validate multiple data items."""
        results = []
        
        for data, data_type in data_items:
            summary = self.validate_data(data, data_type, level)
            results.append(summary)
            
        return results
        
    def _create_validation_summary(self, issues: List[ValidationIssue], validation_time: float) -> ValidationSummary:
        """Create validation summary from issues."""
        total_checks = len(issues) + 1  # +1 for basic format check
        passed_checks = 0
        warning_checks = 0
        failed_checks = 0
        error_checks = 0
        
        for issue in issues:
            if issue.result == ValidationResult.PASS:
                passed_checks += 1
            elif issue.result == ValidationResult.WARNING:
                warning_checks += 1
            elif issue.result == ValidationResult.FAIL:
                failed_checks += 1
            elif issue.result == ValidationResult.ERROR:
                error_checks += 1
                
        # If no issues, we have one passing check
        if not issues:
            passed_checks = 1
            
        # Determine overall result
        if error_checks > 0:
            overall_result = ValidationResult.ERROR
        elif failed_checks > 0:
            overall_result = ValidationResult.FAIL
        elif warning_checks > 0:
            overall_result = ValidationResult.WARNING
        else:
            overall_result = ValidationResult.PASS
            
        # Calculate data integrity score (0-100)
        if total_checks > 0:
            integrity_score = ((passed_checks + (warning_checks * 0.5)) / total_checks) * 100
        else:
            integrity_score = 100
            
        return ValidationSummary(
            total_checks=total_checks,
            passed_checks=passed_checks,
            warning_checks=warning_checks,
            failed_checks=failed_checks,
            error_checks=error_checks,
            issues=issues,
            validation_time_ms=validation_time * 1000,
            data_integrity_score=integrity_score,
            overall_result=overall_result
        )
        
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation system statistics."""
        return {
            'validation_stats': self.validation_stats.copy(),
            'cache_size': len(self.validation_cache),
            'cache_limit': self.config.validation_cache_size,
            'validators_available': list(self.validators.keys()),
            'config': {
                'default_level': self.config.default_level.value,
                'security_validation_enabled': self.config.enable_security_validation,
                'biological_validation_enabled': self.config.enable_biological_validation,
                'checksum_validation_enabled': self.config.enable_checksum_validation
            }
        }
        
    def clear_validation_cache(self):
        """Clear validation cache."""
        self.validation_cache.clear()
        logger.info("Validation cache cleared")


# Demo and testing functions
def demo_validation_system():
    """Demonstrate the validation system capabilities."""
    config = ValidationConfig(
        default_level=ValidationLevel.NORMAL,
        enable_security_validation=True,
        enable_biological_validation=True
    )
    
    validation_system = ComprehensiveValidationSystem(config)
    
    print("=== Comprehensive Validation System Demo ===")
    
    # Test protein sequence validation
    test_sequences = [
        "MKLLVLLVLLVLGGGHHHHHHH",  # Valid sequence
        "MKLLVLLVLLVLGGGHHHHHHX",  # Invalid character
        "M",  # Too short
        "",   # Empty
        "PPPPPPPPPPPPPPPPPPPPPP",  # High proline content
    ]
    
    print("\n--- Protein Sequence Validation ---")
    for i, sequence in enumerate(test_sequences):
        print(f"\nTest {i+1}: '{sequence}'")
        summary = validation_system.validate_protein_sequence(sequence)
        print(f"Result: {summary.overall_result.value}")
        print(f"Integrity Score: {summary.data_integrity_score:.1f}")
        if summary.issues:
            for issue in summary.issues:
                print(f"  - {issue.result.value.upper()}: {issue.message}")
                
    # Test generation parameters validation
    test_params = [
        {"temperature": 0.8, "num_proteins": 100, "motif": "HELIX_SHEET_HELIX"},  # Valid
        {"temperature": -1, "num_proteins": 0},  # Invalid values
        {"temperature": "invalid", "num_proteins": "not_a_number"},  # Invalid types
    ]
    
    print("\n--- Generation Parameters Validation ---")
    for i, params in enumerate(test_params):
        print(f"\nTest {i+1}: {params}")
        summary = validation_system.validate_generation_parameters(params)
        print(f"Result: {summary.overall_result.value}")
        if summary.issues:
            for issue in summary.issues:
                print(f"  - {issue.result.value.upper()}: {issue.message}")
                
    # Test security validation
    test_security_data = [
        "Normal protein sequence",
        "SELECT * FROM proteins; DROP TABLE users;",  # SQL injection
        "<script>alert('xss')</script>",  # XSS attempt
        "../../../etc/passwd",  # Path traversal
    ]
    
    print("\n--- Security Validation ---")
    for i, data in enumerate(test_security_data):
        print(f"\nTest {i+1}: '{data}'")
        summary = validation_system.validate_data(data, DataType.USER_INPUT)
        print(f"Result: {summary.overall_result.value}")
        if summary.issues:
            for issue in summary.issues:
                print(f"  - {issue.result.value.upper()}: {issue.message}")
                
    # Show statistics
    stats = validation_system.get_validation_statistics()
    print(f"\n=== Validation Statistics ===")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    demo_validation_system()