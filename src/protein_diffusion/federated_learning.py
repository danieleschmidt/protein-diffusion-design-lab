"""
Federated Learning Framework for Protein Diffusion

Advanced federated learning system for distributed training across
multiple institutions while preserving data privacy.
"""

import asyncio
import json
import time
import uuid
import hashlib
import logging
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pickle
import base64
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cryptography
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

try:
    import websockets
    import aiohttp
    NETWORK_AVAILABLE = True
except ImportError:
    NETWORK_AVAILABLE = False


logger = logging.getLogger(__name__)


class FederatedLearningRole(Enum):
    """Roles in federated learning system."""
    COORDINATOR = "coordinator"
    PARTICIPANT = "participant"
    AGGREGATOR = "aggregator"
    VALIDATOR = "validator"


class TrainingPhase(Enum):
    """Training phases."""
    INITIALIZATION = "initialization"
    LOCAL_TRAINING = "local_training"
    MODEL_SHARING = "model_sharing"
    AGGREGATION = "aggregation"
    VALIDATION = "validation"
    COMPLETION = "completion"


class PrivacyLevel(Enum):
    """Privacy preservation levels."""
    NONE = "none"
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    HOMOMORPHIC_ENCRYPTION = "homomorphic_encryption"
    SECURE_MULTIPARTY = "secure_multiparty"
    FEDERATED_AVERAGING = "federated_averaging"


@dataclass
class FederatedParticipant:
    """Federated learning participant information."""
    participant_id: str
    name: str
    institution: str
    role: FederatedLearningRole
    host: str
    port: int
    public_key: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    data_statistics: Dict[str, Any] = field(default_factory=dict)
    trust_score: float = 1.0
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "active"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "participant_id": self.participant_id,
            "name": self.name,
            "institution": self.institution,
            "role": self.role.value,
            "host": self.host,
            "port": self.port,
            "public_key": self.public_key,
            "capabilities": self.capabilities,
            "data_statistics": self.data_statistics,
            "trust_score": self.trust_score,
            "last_activity": self.last_activity.isoformat(),
            "status": self.status
        }


@dataclass
class FederatedModel:
    """Federated model representation."""
    model_id: str
    version: int
    architecture: str
    parameters: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: Optional[str] = None
    encrypted: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    contributor_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "architecture": self.architecture,
            "parameters": self.parameters,
            "metadata": self.metadata,
            "checksum": self.checksum,
            "encrypted": self.encrypted,
            "created_at": self.created_at.isoformat(),
            "contributor_id": self.contributor_id
        }
    
    def compute_checksum(self) -> str:
        """Compute model checksum for integrity verification."""
        model_bytes = json.dumps(self.parameters, sort_keys=True).encode()
        return hashlib.sha256(model_bytes).hexdigest()


@dataclass
class TrainingRound:
    """Federated training round information."""
    round_id: str
    round_number: int
    coordinator_id: str
    participants: List[str]
    global_model_version: int
    phase: TrainingPhase
    started_at: datetime
    completed_at: Optional[datetime] = None
    results: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_id": self.round_id,
            "round_number": self.round_number,
            "coordinator_id": self.coordinator_id,
            "participants": self.participants,
            "global_model_version": self.global_model_version,
            "phase": self.phase.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "results": self.results,
            "metrics": self.metrics
        }


class PrivacyPreserver:
    """Privacy preservation mechanisms."""
    
    def __init__(self, privacy_level: PrivacyLevel = PrivacyLevel.FEDERATED_AVERAGING):
        self.privacy_level = privacy_level
        self.noise_multiplier = 1.0
        self.l2_norm_clip = 1.0
        
    def add_noise(self, gradients: Dict[str, Any], sensitivity: float = 1.0) -> Dict[str, Any]:
        """Add differential privacy noise to gradients."""
        if self.privacy_level != PrivacyLevel.DIFFERENTIAL_PRIVACY:
            return gradients
        
        noisy_gradients = {}
        for key, grad in gradients.items():
            if isinstance(grad, (list, tuple)):
                # Convert to numpy for processing
                grad_array = np.array(grad)
                noise = np.random.normal(0, self.noise_multiplier * sensitivity, grad_array.shape)
                noisy_gradients[key] = (grad_array + noise).tolist()
            else:
                noisy_gradients[key] = grad
                
        return noisy_gradients
    
    def clip_gradients(self, gradients: Dict[str, Any]) -> Dict[str, Any]:
        """Clip gradients for privacy preservation."""
        clipped_gradients = {}
        
        for key, grad in gradients.items():
            if isinstance(grad, (list, tuple)):
                grad_array = np.array(grad)
                grad_norm = np.linalg.norm(grad_array)
                
                if grad_norm > self.l2_norm_clip:
                    clipped_gradients[key] = (grad_array * self.l2_norm_clip / grad_norm).tolist()
                else:
                    clipped_gradients[key] = grad
            else:
                clipped_gradients[key] = grad
                
        return clipped_gradients
    
    def secure_aggregate(self, model_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform secure aggregation of model updates."""
        if not model_updates:
            return {}
        
        # Initialize aggregated updates
        aggregated = {}
        
        # Get all parameter keys from first update
        first_update = model_updates[0]
        for key in first_update:
            aggregated[key] = []
        
        # Collect all updates for each parameter
        for update in model_updates:
            for key, value in update.items():
                if key in aggregated:
                    if isinstance(value, (list, tuple)):
                        aggregated[key].append(np.array(value))
                    else:
                        aggregated[key].append(value)
        
        # Average the updates
        final_aggregated = {}
        for key, values in aggregated.items():
            if values and isinstance(values[0], np.ndarray):
                final_aggregated[key] = np.mean(values, axis=0).tolist()
            elif values and isinstance(values[0], (int, float)):
                final_aggregated[key] = np.mean(values)
            else:
                final_aggregated[key] = values[0] if values else None
        
        return final_aggregated


class FederatedCoordinator:
    """Coordinates federated learning across participants."""
    
    def __init__(self, coordinator_id: str, name: str = "FL-Coordinator"):
        self.coordinator_id = coordinator_id
        self.name = name
        self.participants: Dict[str, FederatedParticipant] = {}
        self.global_model: Optional[FederatedModel] = None
        self.current_round: Optional[TrainingRound] = None
        self.training_history: List[TrainingRound] = []
        self.privacy_preserver = PrivacyPreserver()
        self._lock = asyncio.Lock()
        
    async def register_participant(self, participant: FederatedParticipant) -> bool:
        """Register a new participant."""
        async with self._lock:
            try:
                self.participants[participant.participant_id] = participant
                logger.info(f"Registered participant {participant.name} ({participant.participant_id})")
                return True
            except Exception as e:
                logger.error(f"Failed to register participant: {e}")
                return False
    
    async def initialize_global_model(
        self,
        architecture: str,
        initial_parameters: Dict[str, Any],
        metadata: Dict[str, Any] = None
    ) -> str:
        """Initialize the global model."""
        model_id = str(uuid.uuid4())
        
        self.global_model = FederatedModel(
            model_id=model_id,
            version=0,
            architecture=architecture,
            parameters=initial_parameters,
            metadata=metadata or {},
            contributor_id=self.coordinator_id
        )
        
        self.global_model.checksum = self.global_model.compute_checksum()
        
        logger.info(f"Initialized global model {model_id}")
        return model_id
    
    async def start_training_round(self, selected_participants: Optional[List[str]] = None) -> str:
        """Start a new training round."""
        if not self.global_model:
            raise ValueError("Global model not initialized")
        
        async with self._lock:
            round_id = str(uuid.uuid4())
            round_number = len(self.training_history) + 1
            
            # Select participants (all active by default)
            if selected_participants is None:
                active_participants = [
                    p.participant_id for p in self.participants.values()
                    if p.status == "active" and p.role == FederatedLearningRole.PARTICIPANT
                ]
            else:
                active_participants = selected_participants
            
            self.current_round = TrainingRound(
                round_id=round_id,
                round_number=round_number,
                coordinator_id=self.coordinator_id,
                participants=active_participants,
                global_model_version=self.global_model.version,
                phase=TrainingPhase.INITIALIZATION,
                started_at=datetime.now(timezone.utc)
            )
            
            logger.info(f"Started training round {round_number} with {len(active_participants)} participants")
            
            # Distribute global model to participants
            await self._distribute_global_model(active_participants)
            
            return round_id
    
    async def _distribute_global_model(self, participant_ids: List[str]):
        """Distribute global model to selected participants."""
        self.current_round.phase = TrainingPhase.MODEL_SHARING
        
        distribution_tasks = []
        for participant_id in participant_ids:
            if participant_id in self.participants:
                participant = self.participants[participant_id]
                task = self._send_model_to_participant(participant, self.global_model)
                distribution_tasks.append(task)
        
        # Wait for all distributions to complete
        results = await asyncio.gather(*distribution_tasks, return_exceptions=True)
        
        successful_distributions = sum(1 for r in results if r is True)
        logger.info(f"Successfully distributed model to {successful_distributions}/{len(participant_ids)} participants")
    
    async def _send_model_to_participant(
        self,
        participant: FederatedParticipant,
        model: FederatedModel
    ) -> bool:
        """Send model to a specific participant."""
        try:
            # Simulate network communication
            await asyncio.sleep(0.1)
            
            logger.debug(f"Sent model to participant {participant.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send model to {participant.name}: {e}")
            return False
    
    async def collect_model_updates(self, timeout: float = 300.0) -> List[Dict[str, Any]]:
        """Collect model updates from participants."""
        if not self.current_round:
            raise ValueError("No active training round")
        
        self.current_round.phase = TrainingPhase.LOCAL_TRAINING
        
        # Simulate waiting for participant updates
        await asyncio.sleep(2.0)  # Simulate training time
        
        # Generate mock updates from participants
        updates = []
        for participant_id in self.current_round.participants:
            if participant_id in self.participants:
                participant = self.participants[participant_id]
                
                # Generate mock model update
                mock_update = self._generate_mock_update(participant_id)
                updates.append({
                    "participant_id": participant_id,
                    "update": mock_update,
                    "training_samples": np.random.randint(100, 1000),
                    "training_loss": np.random.uniform(0.1, 1.0),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
        
        logger.info(f"Collected updates from {len(updates)} participants")
        return updates
    
    def _generate_mock_update(self, participant_id: str) -> Dict[str, Any]:
        """Generate mock model update for simulation."""
        if not self.global_model:
            return {}
        
        # Generate small random updates to simulate training
        mock_update = {}
        for key, value in self.global_model.parameters.items():
            if isinstance(value, (list, tuple)):
                # Add small random changes
                original = np.array(value)
                noise = np.random.normal(0, 0.01, original.shape)
                mock_update[key] = (original + noise).tolist()
            else:
                mock_update[key] = value
        
        return mock_update
    
    async def aggregate_updates(self, model_updates: List[Dict[str, Any]]) -> FederatedModel:
        """Aggregate model updates using federated averaging."""
        if not model_updates:
            raise ValueError("No model updates to aggregate")
        
        self.current_round.phase = TrainingPhase.AGGREGATION
        
        # Extract updates and weights
        updates = [update["update"] for update in model_updates]
        weights = [update.get("training_samples", 1) for update in model_updates]
        
        # Apply privacy preservation
        privacy_updates = []
        for update in updates:
            clipped = self.privacy_preserver.clip_gradients(update)
            noisy = self.privacy_preserver.add_noise(clipped)
            privacy_updates.append(noisy)
        
        # Weighted federated averaging
        aggregated_params = self._weighted_average(privacy_updates, weights)
        
        # Create new global model version
        new_version = self.global_model.version + 1
        new_model = FederatedModel(
            model_id=self.global_model.model_id,
            version=new_version,
            architecture=self.global_model.architecture,
            parameters=aggregated_params,
            metadata={
                **self.global_model.metadata,
                "round_number": self.current_round.round_number,
                "participants": len(model_updates),
                "aggregation_method": "federated_averaging"
            },
            contributor_id=self.coordinator_id
        )
        
        new_model.checksum = new_model.compute_checksum()
        
        # Update global model
        self.global_model = new_model
        
        logger.info(f"Aggregated model to version {new_version}")
        return new_model
    
    def _weighted_average(self, updates: List[Dict[str, Any]], weights: List[float]) -> Dict[str, Any]:
        """Compute weighted average of model updates."""
        if not updates or not weights:
            return {}
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Initialize aggregated parameters
        aggregated = {}
        first_update = updates[0]
        
        for key in first_update:
            weighted_values = []
            
            for i, update in enumerate(updates):
                if key in update:
                    value = update[key]
                    weight = normalized_weights[i]
                    
                    if isinstance(value, (list, tuple)):
                        weighted_value = np.array(value) * weight
                        weighted_values.append(weighted_value)
                    elif isinstance(value, (int, float)):
                        weighted_values.append(value * weight)
            
            if weighted_values:
                if isinstance(weighted_values[0], np.ndarray):
                    aggregated[key] = np.sum(weighted_values, axis=0).tolist()
                else:
                    aggregated[key] = sum(weighted_values)
        
        return aggregated
    
    async def evaluate_global_model(self) -> Dict[str, float]:
        """Evaluate the global model performance."""
        if not self.current_round:
            raise ValueError("No active training round")
        
        self.current_round.phase = TrainingPhase.VALIDATION
        
        # Simulate model evaluation
        await asyncio.sleep(1.0)
        
        # Generate mock evaluation metrics
        metrics = {
            "accuracy": np.random.uniform(0.85, 0.95),
            "loss": np.random.uniform(0.1, 0.5),
            "f1_score": np.random.uniform(0.80, 0.90),
            "perplexity": np.random.uniform(10, 50),
            "validation_samples": np.random.randint(1000, 5000)
        }
        
        self.current_round.metrics = metrics
        
        logger.info(f"Global model evaluation: accuracy={metrics['accuracy']:.3f}, loss={metrics['loss']:.3f}")
        return metrics
    
    async def complete_training_round(self) -> Dict[str, Any]:
        """Complete the current training round."""
        if not self.current_round:
            raise ValueError("No active training round")
        
        self.current_round.phase = TrainingPhase.COMPLETION
        self.current_round.completed_at = datetime.now(timezone.utc)
        
        # Store results
        self.current_round.results = {
            "global_model_version": self.global_model.version,
            "participants_count": len(self.current_round.participants),
            "aggregation_successful": True,
            "evaluation_metrics": self.current_round.metrics
        }
        
        # Move to history
        self.training_history.append(self.current_round)
        
        # Clear current round
        completed_round = self.current_round
        self.current_round = None
        
        logger.info(f"Completed training round {completed_round.round_number}")
        
        return completed_round.to_dict()
    
    async def get_training_status(self) -> Dict[str, Any]:
        """Get current training status."""
        return {
            "coordinator_id": self.coordinator_id,
            "global_model_version": self.global_model.version if self.global_model else 0,
            "active_participants": len([p for p in self.participants.values() if p.status == "active"]),
            "current_round": self.current_round.to_dict() if self.current_round else None,
            "completed_rounds": len(self.training_history),
            "last_evaluation": self.training_history[-1].metrics if self.training_history else {}
        }


class FederatedParticipantClient:
    """Client for participating in federated learning."""
    
    def __init__(self, participant_info: FederatedParticipant):
        self.participant_info = participant_info
        self.local_model: Optional[FederatedModel] = None
        self.local_dataset = None
        self.training_config = {}
        self.coordinator_connection = None
        
    async def connect_to_coordinator(self, coordinator_host: str, coordinator_port: int) -> bool:
        """Connect to federated learning coordinator."""
        try:
            # Simulate connection
            await asyncio.sleep(0.1)
            
            logger.info(f"Connected to coordinator at {coordinator_host}:{coordinator_port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to coordinator: {e}")
            return False
    
    async def register_with_coordinator(self) -> bool:
        """Register with the coordinator."""
        try:
            # Simulate registration
            await asyncio.sleep(0.1)
            
            logger.info(f"Registered participant {self.participant_info.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register with coordinator: {e}")
            return False
    
    def load_local_dataset(self, dataset_path: str, preprocessing_config: Dict[str, Any] = None):
        """Load and preprocess local dataset."""
        # Simulate dataset loading
        mock_dataset = {
            "samples": np.random.randint(500, 2000),
            "features": 256,
            "sequences": ["MKTVRQ..." for _ in range(100)],  # Mock protein sequences
            "labels": np.random.randint(0, 2, 100).tolist()
        }
        
        self.local_dataset = mock_dataset
        
        # Update participant statistics
        self.participant_info.data_statistics = {
            "total_samples": mock_dataset["samples"],
            "feature_dimensions": mock_dataset["features"],
            "class_distribution": {"class_0": 60, "class_1": 40},
            "data_quality": 0.92
        }
        
        logger.info(f"Loaded local dataset with {mock_dataset['samples']} samples")
    
    async def receive_global_model(self, global_model: FederatedModel) -> bool:
        """Receive global model from coordinator."""
        try:
            # Verify model integrity
            computed_checksum = global_model.compute_checksum()
            if global_model.checksum != computed_checksum:
                logger.warning("Model checksum mismatch - potential corruption")
                return False
            
            self.local_model = global_model
            logger.info(f"Received global model version {global_model.version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to receive global model: {e}")
            return False
    
    async def train_local_model(self, epochs: int = 1, batch_size: int = 32) -> Dict[str, Any]:
        """Train the local model on local data."""
        if not self.local_model or not self.local_dataset:
            raise ValueError("Local model or dataset not available")
        
        logger.info(f"Starting local training for {epochs} epochs")
        
        # Simulate training process
        training_metrics = []
        
        for epoch in range(epochs):
            # Simulate epoch training
            await asyncio.sleep(0.5)  # Simulate training time
            
            epoch_loss = np.random.uniform(0.1, 1.0) * (0.9 ** epoch)  # Decreasing loss
            epoch_accuracy = min(0.95, 0.6 + (epoch * 0.05))  # Increasing accuracy
            
            training_metrics.append({
                "epoch": epoch + 1,
                "loss": epoch_loss,
                "accuracy": epoch_accuracy,
                "samples_processed": self.local_dataset["samples"]
            })
            
            logger.debug(f"Epoch {epoch + 1}: loss={epoch_loss:.3f}, accuracy={epoch_accuracy:.3f}")
        
        # Generate model update (difference from original)
        model_update = self._compute_model_update()
        
        training_result = {
            "participant_id": self.participant_info.participant_id,
            "training_samples": self.local_dataset["samples"],
            "training_epochs": epochs,
            "final_loss": training_metrics[-1]["loss"],
            "final_accuracy": training_metrics[-1]["accuracy"],
            "model_update": model_update,
            "training_time": epochs * 0.5,  # Simulated training time
            "metrics": training_metrics
        }
        
        logger.info(f"Completed local training: final_loss={training_result['final_loss']:.3f}")
        
        return training_result
    
    def _compute_model_update(self) -> Dict[str, Any]:
        """Compute model update (gradient/parameter changes)."""
        if not self.local_model:
            return {}
        
        # Generate mock parameter updates
        model_update = {}
        
        for key, value in self.local_model.parameters.items():
            if isinstance(value, (list, tuple)):
                # Add small random updates to simulate training changes
                original = np.array(value)
                update = np.random.normal(0, 0.01, original.shape)
                model_update[key] = update.tolist()
            else:
                model_update[key] = np.random.normal(0, 0.01)
        
        return model_update
    
    async def send_model_update(self, training_result: Dict[str, Any]) -> bool:
        """Send model update to coordinator."""
        try:
            # Simulate sending update
            await asyncio.sleep(0.2)
            
            logger.info("Sent model update to coordinator")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send model update: {e}")
            return False
    
    async def participate_in_round(self, training_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Participate in a complete federated learning round."""
        config = training_config or {"epochs": 1, "batch_size": 32}
        
        try:
            # Wait for global model
            logger.info("Waiting for global model...")
            await asyncio.sleep(1.0)  # Simulate waiting
            
            # Train local model
            training_result = await self.train_local_model(
                epochs=config.get("epochs", 1),
                batch_size=config.get("batch_size", 32)
            )
            
            # Send update to coordinator
            success = await self.send_model_update(training_result)
            
            return {
                "success": success,
                "training_result": training_result,
                "participant_id": self.participant_info.participant_id
            }
            
        except Exception as e:
            logger.error(f"Error participating in round: {e}")
            return {
                "success": False,
                "error": str(e),
                "participant_id": self.participant_info.participant_id
            }


class FederatedLearningOrchestrator:
    """High-level orchestrator for federated learning experiments."""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.experiment_id = str(uuid.uuid4())
        self.coordinator: Optional[FederatedCoordinator] = None
        self.participants: List[FederatedParticipantClient] = []
        self.experiment_config = {}
        self.results_history: List[Dict[str, Any]] = []
        
    def setup_coordinator(self, coordinator_name: str = "FL-Coordinator") -> FederatedCoordinator:
        """Setup federated learning coordinator."""
        coordinator_id = f"coordinator_{self.experiment_id[:8]}"
        
        self.coordinator = FederatedCoordinator(coordinator_id, coordinator_name)
        
        logger.info(f"Setup coordinator {coordinator_name}")
        return self.coordinator
    
    def add_participant(
        self,
        name: str,
        institution: str,
        host: str = "localhost",
        port: int = 0,
        capabilities: List[str] = None
    ) -> FederatedParticipantClient:
        """Add a participant to the federation."""
        participant_id = f"participant_{len(self.participants)}_{self.experiment_id[:8]}"
        
        participant_info = FederatedParticipant(
            participant_id=participant_id,
            name=name,
            institution=institution,
            role=FederatedLearningRole.PARTICIPANT,
            host=host,
            port=port or (8000 + len(self.participants)),
            capabilities=capabilities or ["protein_diffusion", "local_training"]
        )
        
        participant_client = FederatedParticipantClient(participant_info)
        self.participants.append(participant_client)
        
        logger.info(f"Added participant {name} from {institution}")
        return participant_client
    
    async def initialize_federation(
        self,
        model_architecture: str,
        initial_parameters: Dict[str, Any],
        experiment_config: Dict[str, Any] = None
    ):
        """Initialize the federated learning experiment."""
        if not self.coordinator:
            raise ValueError("Coordinator not setup")
        
        self.experiment_config = experiment_config or {
            "max_rounds": 10,
            "min_participants": 2,
            "convergence_threshold": 0.001,
            "privacy_level": "federated_averaging"
        }
        
        # Initialize global model
        await self.coordinator.initialize_global_model(
            architecture=model_architecture,
            initial_parameters=initial_parameters,
            metadata={
                "experiment_name": self.experiment_name,
                "experiment_id": self.experiment_id,
                "total_participants": len(self.participants)
            }
        )
        
        # Register all participants
        for participant in self.participants:
            await self.coordinator.register_participant(participant.participant_info)
            
            # Load mock datasets for participants
            participant.load_local_dataset(f"data/{participant.participant_info.name}")
        
        logger.info(f"Initialized federation with {len(self.participants)} participants")
    
    async def run_federated_training(self, max_rounds: Optional[int] = None) -> Dict[str, Any]:
        """Run complete federated training experiment."""
        if not self.coordinator:
            raise ValueError("Federation not initialized")
        
        max_rounds = max_rounds or self.experiment_config.get("max_rounds", 10)
        convergence_threshold = self.experiment_config.get("convergence_threshold", 0.001)
        
        experiment_start = datetime.now(timezone.utc)
        round_results = []
        
        previous_loss = float('inf')
        converged = False
        
        for round_num in range(1, max_rounds + 1):
            logger.info(f"Starting federated learning round {round_num}/{max_rounds}")
            
            try:
                # Start training round
                round_id = await self.coordinator.start_training_round()
                
                # Participants perform local training
                participant_tasks = []
                for participant in self.participants:
                    task = participant.participate_in_round({
                        "epochs": 2,
                        "batch_size": 32,
                        "learning_rate": 0.001
                    })
                    participant_tasks.append(task)
                
                # Wait for all participants to complete
                participant_results = await asyncio.gather(*participant_tasks, return_exceptions=True)
                
                # Collect successful results
                successful_results = [
                    result for result in participant_results 
                    if isinstance(result, dict) and result.get("success", False)
                ]
                
                logger.info(f"Collected results from {len(successful_results)}/{len(self.participants)} participants")
                
                # Aggregate model updates
                if successful_results:
                    model_updates = [result["training_result"] for result in successful_results]
                    aggregated_model = await self.coordinator.aggregate_updates(model_updates)
                    
                    # Evaluate global model
                    evaluation_metrics = await self.coordinator.evaluate_global_model()
                    
                    # Complete training round
                    round_summary = await self.coordinator.complete_training_round()
                    
                    round_results.append({
                        "round_number": round_num,
                        "round_id": round_id,
                        "participants": len(successful_results),
                        "metrics": evaluation_metrics,
                        "model_version": aggregated_model.version,
                        "summary": round_summary
                    })
                    
                    # Check for convergence
                    current_loss = evaluation_metrics.get("loss", 0)
                    if abs(previous_loss - current_loss) < convergence_threshold:
                        logger.info(f"Converged after {round_num} rounds (loss change: {abs(previous_loss - current_loss):.6f})")
                        converged = True
                        break
                    
                    previous_loss = current_loss
                    
                    logger.info(f"Round {round_num} completed: loss={current_loss:.4f}, accuracy={evaluation_metrics.get('accuracy', 0):.4f}")
                
                else:
                    logger.warning(f"No successful participants in round {round_num}")
                    break
                    
            except Exception as e:
                logger.error(f"Error in round {round_num}: {e}")
                break
        
        experiment_end = datetime.now(timezone.utc)
        experiment_duration = (experiment_end - experiment_start).total_seconds()
        
        experiment_results = {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "total_rounds": len(round_results),
            "converged": converged,
            "final_metrics": round_results[-1]["metrics"] if round_results else {},
            "experiment_duration": experiment_duration,
            "participants_count": len(self.participants),
            "round_results": round_results,
            "completed_at": experiment_end.isoformat()
        }
        
        self.results_history.append(experiment_results)
        
        logger.info(f"Federated training experiment completed in {experiment_duration:.2f} seconds")
        logger.info(f"Final metrics: {experiment_results['final_metrics']}")
        
        return experiment_results


# Example usage and testing
async def example_federated_learning():
    """Example federated learning experiment."""
    
    # Create orchestrator
    orchestrator = FederatedLearningOrchestrator("Protein-Diffusion-FL-v1")
    
    # Setup coordinator
    coordinator = orchestrator.setup_coordinator("Protein-FL-Coordinator")
    
    # Add participants from different institutions
    participants_config = [
        {"name": "MIT-Lab", "institution": "MIT", "host": "localhost", "port": 8001},
        {"name": "Stanford-Lab", "institution": "Stanford", "host": "localhost", "port": 8002},
        {"name": "UCSF-Lab", "institution": "UCSF", "host": "localhost", "port": 8003},
        {"name": "DeepMind-Lab", "institution": "DeepMind", "host": "localhost", "port": 8004}
    ]
    
    for config in participants_config:
        orchestrator.add_participant(**config)
    
    # Initialize federation with mock protein diffusion model
    mock_model_params = {
        "embedding_weights": np.random.randn(1000, 256).tolist(),
        "transformer_weights": np.random.randn(512, 512).tolist(),
        "output_projection": np.random.randn(256, 20).tolist(),  # 20 amino acids
        "bias_terms": np.random.randn(256).tolist()
    }
    
    await orchestrator.initialize_federation(
        model_architecture="protein_diffusion_transformer",
        initial_parameters=mock_model_params,
        experiment_config={
            "max_rounds": 5,
            "convergence_threshold": 0.01,
            "privacy_level": "differential_privacy"
        }
    )
    
    # Run federated training
    results = await orchestrator.run_federated_training()
    
    print("Federated Learning Results:")
    print(f"- Experiment: {results['experiment_name']}")
    print(f"- Rounds completed: {results['total_rounds']}")
    print(f"- Converged: {results['converged']}")
    print(f"- Final accuracy: {results['final_metrics'].get('accuracy', 0):.4f}")
    print(f"- Duration: {results['experiment_duration']:.2f} seconds")
    
    return results


if __name__ == "__main__":
    asyncio.run(example_federated_learning())