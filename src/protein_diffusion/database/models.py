"""
Database models for protein diffusion design lab.

This module defines the data models for proteins, structures, experiments,
and results with validation and serialization capabilities.
"""

import json
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict, field
from enum import Enum


class ExperimentStatus(Enum):
    """Status of an experiment."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PredictionMethod(Enum):
    """Structure prediction methods."""
    ESMFOLD = "esmfold"
    COLABFOLD = "colabfold"
    ALPHAFOLD = "alphafold"
    CHIMERAX = "chimeraX"
    UNKNOWN = "unknown"


@dataclass
class ProteinModel:
    """
    Data model for protein sequences.
    
    Represents a protein sequence with metadata and computed properties.
    """
    id: Optional[int] = None
    sequence: str = ""
    sequence_hash: str = field(default="", init=False)
    length: int = field(default=0, init=False)
    created_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Compute derived fields."""
        if self.sequence:
            self.sequence_hash = self._compute_hash()
            self.length = len(self.sequence)
        
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of the sequence."""
        return hashlib.sha256(self.sequence.encode()).hexdigest()
    
    def add_metadata(self, key: str, value: Any):
        """Add metadata field."""
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata field."""
        return self.metadata.get(key, default)
    
    def validate(self) -> bool:
        """Validate protein sequence."""
        if not self.sequence:
            return False
        
        # Check for valid amino acids
        valid_aa = set("ACDEFGHIKLMNPQRSTVWYUOX")
        sequence_aa = set(self.sequence.upper())
        
        return sequence_aa.issubset(valid_aa)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        data = asdict(self)
        data['metadata'] = json.dumps(self.metadata)
        if isinstance(data['created_at'], datetime):
            data['created_at'] = data['created_at'].isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProteinModel':
        """Create from dictionary."""
        if 'metadata' in data and isinstance(data['metadata'], str):
            data['metadata'] = json.loads(data['metadata'])
        
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        return cls(**data)
    
    def get_sequence_properties(self) -> Dict[str, float]:
        """Calculate basic sequence properties."""
        if not self.sequence:
            return {}
        
        sequence = self.sequence.upper()
        total_aa = len(sequence)
        
        # Amino acid composition
        aa_counts = {aa: sequence.count(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"}
        
        # Hydrophobicity (Kyte-Doolittle scale)
        hydrophobicity_scale = {
            'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
            'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
            'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
            'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
        }
        
        hydrophobicity = sum(hydrophobicity_scale.get(aa, 0) for aa in sequence) / total_aa
        
        # Charge at pH 7
        charge_scale = {'D': -1, 'E': -1, 'K': 1, 'R': 1, 'H': 0.1}
        net_charge = sum(charge_scale.get(aa, 0) for aa in sequence)
        
        return {
            "hydrophobicity": hydrophobicity,
            "net_charge": net_charge,
            "charge_density": net_charge / total_aa,
            "glycine_content": aa_counts.get('G', 0) / total_aa,
            "proline_content": aa_counts.get('P', 0) / total_aa,
            "aromatic_content": sum(aa_counts.get(aa, 0) for aa in "FWY") / total_aa,
            "molecular_weight": self._calculate_molecular_weight(),
        }
    
    def _calculate_molecular_weight(self) -> float:
        """Calculate approximate molecular weight in Da."""
        # Amino acid molecular weights (average)
        aa_weights = {
            'A': 89.1, 'C': 121.0, 'D': 133.1, 'E': 147.1, 'F': 165.2,
            'G': 75.1, 'H': 155.2, 'I': 131.2, 'K': 146.2, 'L': 131.2,
            'M': 149.2, 'N': 132.1, 'P': 115.1, 'Q': 146.2, 'R': 174.2,
            'S': 105.1, 'T': 119.1, 'V': 117.1, 'W': 204.2, 'Y': 181.2
        }
        
        weight = sum(aa_weights.get(aa, 110.0) for aa in self.sequence.upper())
        # Subtract water molecules for peptide bonds
        weight -= (len(self.sequence) - 1) * 18.015
        
        return weight


@dataclass
class StructureModel:
    """
    Data model for protein structures.
    
    Represents predicted or experimental protein structures with quality metrics.
    """
    id: Optional[int] = None
    protein_id: int = 0
    pdb_content: str = ""
    confidence: float = 0.0
    structure_quality: float = 0.0
    prediction_method: PredictionMethod = PredictionMethod.UNKNOWN
    created_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize timestamps."""
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def add_quality_metric(self, metric_name: str, value: float):
        """Add a quality metric."""
        if 'quality_metrics' not in self.metadata:
            self.metadata['quality_metrics'] = {}
        self.metadata['quality_metrics'][metric_name] = value
    
    def get_quality_metric(self, metric_name: str) -> Optional[float]:
        """Get a quality metric."""
        return self.metadata.get('quality_metrics', {}).get(metric_name)
    
    def has_structure(self) -> bool:
        """Check if structure data is available."""
        return bool(self.pdb_content)
    
    def get_structure_summary(self) -> Dict[str, Any]:
        """Get summary of structure properties."""
        summary = {
            "has_structure": self.has_structure(),
            "confidence": self.confidence,
            "structure_quality": self.structure_quality,
            "prediction_method": self.prediction_method.value,
        }
        
        if self.metadata.get('quality_metrics'):
            summary.update(self.metadata['quality_metrics'])
        
        return summary
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        data = asdict(self)
        data['metadata'] = json.dumps(self.metadata)
        data['prediction_method'] = self.prediction_method.value
        if isinstance(data['created_at'], datetime):
            data['created_at'] = data['created_at'].isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StructureModel':
        """Create from dictionary."""
        if 'metadata' in data and isinstance(data['metadata'], str):
            data['metadata'] = json.loads(data['metadata'])
        
        if 'prediction_method' in data and isinstance(data['prediction_method'], str):
            data['prediction_method'] = PredictionMethod(data['prediction_method'])
        
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        return cls(**data)


@dataclass
class ExperimentModel:
    """
    Data model for experiments.
    
    Represents a protein design experiment with parameters and status.
    """
    id: Optional[int] = None
    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    status: ExperimentStatus = ExperimentStatus.PENDING
    
    def __post_init__(self):
        """Initialize timestamps."""
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def set_parameter(self, key: str, value: Any):
        """Set experiment parameter."""
        self.parameters[key] = value
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get experiment parameter."""
        return self.parameters.get(key, default)
    
    def update_status(self, status: ExperimentStatus):
        """Update experiment status."""
        self.status = status
    
    def is_completed(self) -> bool:
        """Check if experiment is completed."""
        return self.status == ExperimentStatus.COMPLETED
    
    def is_running(self) -> bool:
        """Check if experiment is running."""
        return self.status == ExperimentStatus.RUNNING
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        data = asdict(self)
        data['parameters'] = json.dumps(self.parameters)
        data['status'] = self.status.value
        if isinstance(data['created_at'], datetime):
            data['created_at'] = data['created_at'].isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentModel':
        """Create from dictionary."""
        if 'parameters' in data and isinstance(data['parameters'], str):
            data['parameters'] = json.loads(data['parameters'])
        
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = ExperimentStatus(data['status'])
        
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        return cls(**data)


@dataclass
class ResultModel:
    """
    Data model for experiment results.
    
    Represents evaluation results for proteins in experiments.
    """
    id: Optional[int] = None
    experiment_id: int = 0
    protein_id: int = 0
    binding_affinity: Optional[float] = None
    composite_score: float = 0.0
    ranking: Optional[int] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize timestamps."""
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def add_metric(self, metric_name: str, value: Any):
        """Add evaluation metric."""
        self.metrics[metric_name] = value
    
    def get_metric(self, metric_name: str, default: Any = None) -> Any:
        """Get evaluation metric."""
        return self.metrics.get(metric_name, default)
    
    def has_binding_data(self) -> bool:
        """Check if binding affinity data is available."""
        return self.binding_affinity is not None
    
    def get_result_summary(self) -> Dict[str, Any]:
        """Get summary of results."""
        return {
            "binding_affinity": self.binding_affinity,
            "composite_score": self.composite_score,
            "ranking": self.ranking,
            "metrics_count": len(self.metrics),
            "has_binding_data": self.has_binding_data(),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        data = asdict(self)
        data['metrics'] = json.dumps(self.metrics)
        if isinstance(data['created_at'], datetime):
            data['created_at'] = data['created_at'].isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResultModel':
        """Create from dictionary."""
        if 'metrics' in data and isinstance(data['metrics'], str):
            data['metrics'] = json.loads(data['metrics'])
        
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        return cls(**data)


# Model registry for type checking and validation
MODEL_REGISTRY = {
    'protein': ProteinModel,
    'structure': StructureModel,
    'experiment': ExperimentModel,
    'result': ResultModel,
}


def create_model(model_type: str, **kwargs) -> Any:
    """Factory function to create models."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return MODEL_REGISTRY[model_type](**kwargs)


def validate_model(model: Any) -> bool:
    """Validate a model instance."""
    if hasattr(model, 'validate'):
        return model.validate()
    return True