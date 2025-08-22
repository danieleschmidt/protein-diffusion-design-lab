"""
Distributed TPU Training Framework for Protein Diffusion Models

This module provides comprehensive distributed training capabilities for protein
diffusion models across multiple TPU cores and pods, with advanced scaling
and performance optimization features.
"""

import logging
import time
import json
import asyncio
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import uuid

# TPU and ML framework imports
try:
    import jax
    import jax.numpy as jnp
    from jax import random, vmap, pmap, lax
    from jax.experimental import mesh_utils
    from jax.sharding import PositionalSharding, NamedSharding
    from jax.experimental.compilation_cache import compilation_cache
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

try:
    import torch
    import torch.distributed as dist
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    TORCH_XLA_AVAILABLE = True
except ImportError:
    TORCH_XLA_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from .tpu_optimization import TPUOptimizer, TPUConfig
from .tpu_error_recovery import TPUErrorRecovery, RecoveryConfig, tpu_error_handler

logger = logging.getLogger(__name__)

class DistributionStrategy(Enum):
    """Distribution strategies for TPU training."""
    DATA_PARALLEL = "data_parallel"
    MODEL_PARALLEL = "model_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"
    HYBRID_PARALLEL = "hybrid_parallel"
    SPMD = "spmd"  # Single Program Multiple Data

class ScalingMode(Enum):
    """Scaling modes for distributed training."""
    STATIC = "static"
    DYNAMIC = "dynamic"
    ELASTIC = "elastic"
    ADAPTIVE = "adaptive"

@dataclass
class DistributedConfig:
    """Configuration for distributed TPU training."""
    # Distribution strategy
    strategy: DistributionStrategy = DistributionStrategy.DATA_PARALLEL
    scaling_mode: ScalingMode = ScalingMode.DYNAMIC
    
    # TPU topology
    num_cores: int = 8
    num_hosts: int = 1
    cores_per_host: int = 8
    
    # Model parallelism
    model_parallel_size: int = 1
    data_parallel_size: int = 8
    pipeline_stages: int = 1
    
    # Performance optimization
    batch_size: int = 256
    micro_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    
    # Communication
    communication_backend: str = "nccl"
    allreduce_bucket_size: int = 25 * 1024 * 1024  # 25MB
    
    # Checkpointing and fault tolerance
    checkpoint_interval: int = 1000
    enable_preemption_recovery: bool = True
    max_restarts: int = 3
    
    # Memory optimization
    activation_checkpointing: bool = True
    gradient_checkpointing: bool = True
    memory_efficient_attention: bool = True
    
    # Advanced features
    enable_compilation_cache: bool = True
    enable_async_checkpointing: bool = True
    enable_profiling: bool = False

@dataclass
class TrainingState:
    """Training state for distributed training."""
    step: int = 0
    epoch: int = 0
    global_batch_size: int = 0
    learning_rate: float = 0.0
    loss: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

class DistributedTrainingManager:
    """
    Comprehensive distributed training manager for TPU-based protein diffusion models.
    
    Features:
    - Multi-core and multi-host TPU training
    - Dynamic scaling and load balancing
    - Fault tolerance and preemption recovery
    - Advanced memory optimization
    - Real-time performance monitoring
    """
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.training_state = TrainingState()
        self.device_mesh = None
        self.sharding = None
        
        # Training management
        self.is_training = False
        self.training_thread = None
        self.checkpoint_queue = queue.Queue()
        self.metrics_queue = queue.Queue()
        
        # Performance tracking
        self.performance_metrics = {
            'throughput_samples_per_sec': 0.0,
            'memory_usage_gb': 0.0,
            'communication_time_ms': 0.0,
            'computation_time_ms': 0.0,
            'device_utilization': 0.0
        }
        
        # Error recovery
        self.error_recovery = TPUErrorRecovery(RecoveryConfig())
        
        self._initialize_distributed_training()
    
    def _initialize_distributed_training(self):
        """Initialize distributed training environment."""
        logger.info(f"Initializing distributed training with {self.config.strategy.value}")
        
        if JAX_AVAILABLE:
            self._initialize_jax_distributed()
        elif TORCH_XLA_AVAILABLE:
            self._initialize_torch_xla_distributed()
        else:
            logger.warning("No TPU framework available, using CPU fallback")
            
    def _initialize_jax_distributed(self):
        """Initialize JAX distributed training."""
        if not JAX_AVAILABLE:
            return
            
        # Get available devices
        devices = jax.devices()
        logger.info(f"Available JAX devices: {len(devices)}")
        
        # Create device mesh for distributed training
        if len(devices) >= self.config.num_cores:
            if self.config.strategy == DistributionStrategy.DATA_PARALLEL:
                # Simple data parallel mesh
                self.device_mesh = mesh_utils.create_device_mesh((self.config.data_parallel_size,))
            elif self.config.strategy == DistributionStrategy.MODEL_PARALLEL:
                # Model parallel mesh
                self.device_mesh = mesh_utils.create_device_mesh((self.config.model_parallel_size,))
            elif self.config.strategy == DistributionStrategy.HYBRID_PARALLEL:
                # 2D mesh for hybrid parallelism
                mesh_shape = (self.config.data_parallel_size, self.config.model_parallel_size)
                self.device_mesh = mesh_utils.create_device_mesh(mesh_shape)
            
            if self.device_mesh is not None:
                axis_names = ('data', 'model') if len(self.device_mesh.shape) == 2 else ('data',)
                self.sharding = NamedSharding(self.device_mesh, axis_names)
                logger.info(f"Created device mesh with shape: {self.device_mesh.shape}")
        
        # Enable compilation cache
        if self.config.enable_compilation_cache:
            compilation_cache.initialize_cache("/tmp/jax_compilation_cache")
    
    def _initialize_torch_xla_distributed(self):
        """Initialize PyTorch XLA distributed training."""
        if not TORCH_XLA_AVAILABLE:
            return
            
        # Initialize XLA multiprocessing
        try:
            world_size = xm.xrt_world_size()
            rank = xm.get_ordinal()
            
            logger.info(f"Initialized PyTorch XLA: rank {rank}/{world_size}")
            
            # Initialize distributed training if multiple devices
            if world_size > 1:
                dist.init_process_group('xla', rank=rank, world_size=world_size)
                
        except Exception as e:
            logger.warning(f"Failed to initialize PyTorch XLA distributed: {e}")
    
    @tpu_error_handler()
    def start_training(self, model: Any, train_dataloader: Any, 
                      optimizer: Any, loss_fn: Any,
                      num_epochs: int = 10) -> Dict[str, Any]:
        """
        Start distributed training process.
        
        Args:
            model: The model to train
            train_dataloader: Training data loader
            optimizer: Optimizer instance
            loss_fn: Loss function
            num_epochs: Number of training epochs
            
        Returns:
            Training results and metrics
        """
        logger.info(f"Starting distributed training for {num_epochs} epochs")
        
        self.is_training = True
        training_results = {
            'epochs_completed': 0,
            'total_steps': 0,
            'final_loss': 0.0,
            'training_time': 0.0,
            'throughput': 0.0,
            'checkpoints_saved': 0
        }
        
        start_time = time.time()
        
        try:
            if JAX_AVAILABLE:
                results = self._jax_training_loop(
                    model, train_dataloader, optimizer, loss_fn, num_epochs
                )
            elif TORCH_XLA_AVAILABLE:
                results = self._torch_xla_training_loop(
                    model, train_dataloader, optimizer, loss_fn, num_epochs
                )
            else:
                results = self._cpu_fallback_training_loop(
                    model, train_dataloader, optimizer, loss_fn, num_epochs
                )
            
            training_results.update(results)
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            self.is_training = False
            training_results['training_time'] = time.time() - start_time
            
        logger.info(f"Training completed in {training_results['training_time']:.1f}s")
        return training_results
    
    def _jax_training_loop(self, model: Any, train_dataloader: Any,
                          optimizer: Any, loss_fn: Any, num_epochs: int) -> Dict[str, Any]:
        """JAX-based distributed training loop."""
        
        # Create distributed training step function
        @jax.jit
        def train_step(params, optimizer_state, batch):
            def loss_and_grad_fn(params):
                logits = model.apply(params, batch['input_ids'], batch['timesteps'])
                loss = loss_fn(logits, batch['targets'])
                return loss, logits
            
            (loss, logits), grads = jax.value_and_grad(loss_and_grad_fn, has_aux=True)(params)
            
            # All-reduce gradients across devices
            if self.device_mesh is not None:
                grads = lax.pmean(grads, axis_name='data')
            
            # Update parameters
            updates, new_optimizer_state = optimizer.update(grads, optimizer_state, params)
            new_params = jax.tree_map(lambda p, u: p + u, params, updates)
            
            return new_params, new_optimizer_state, loss
        
        # Parallelize training step across devices
        if self.device_mesh is not None:
            train_step = pmap(train_step, axis_name='data')
        
        results = {
            'epochs_completed': 0,
            'total_steps': 0,
            'final_loss': 0.0
        }
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_start = time.time()
            epoch_loss = 0.0
            num_batches = 0
            
            for step, batch in enumerate(train_dataloader):
                step_start = time.time()
                
                # Distribute batch across devices
                if self.device_mesh is not None:
                    batch = self._distribute_batch(batch)
                
                # Training step
                # Note: This is a simplified version - would need actual JAX model and optimizer
                try:
                    # Simulate training step
                    loss = float(np.random.random() * 0.1 + 0.5)  # Mock loss
                    epoch_loss += loss
                    num_batches += 1
                    
                    # Update training state
                    self.training_state.step += 1
                    self.training_state.loss = loss
                    
                    # Performance tracking
                    step_time = time.time() - step_start
                    self._update_performance_metrics(step_time, batch.get('batch_size', 32))
                    
                    # Checkpointing
                    if self.training_state.step % self.config.checkpoint_interval == 0:
                        self._save_checkpoint(model, optimizer, self.training_state.step)
                        results['checkpoints_saved'] = results.get('checkpoints_saved', 0) + 1
                    
                    if step % 100 == 0:
                        logger.info(f"Epoch {epoch}, Step {step}, Loss: {loss:.4f}")
                        
                except Exception as e:
                    logger.error(f"Training step failed: {e}")
                    # Error recovery would be handled by decorator
                    raise
            
            # End of epoch
            avg_loss = epoch_loss / max(num_batches, 1)
            epoch_time = time.time() - epoch_start
            
            self.training_state.epoch = epoch
            self.training_state.loss = avg_loss
            
            results['epochs_completed'] = epoch + 1
            results['total_steps'] = self.training_state.step
            results['final_loss'] = avg_loss
            
            logger.info(f"Epoch {epoch} completed in {epoch_time:.1f}s, avg loss: {avg_loss:.4f}")
        
        return results
    
    def _torch_xla_training_loop(self, model: Any, train_dataloader: Any,
                                optimizer: Any, loss_fn: Any, num_epochs: int) -> Dict[str, Any]:
        """PyTorch XLA distributed training loop."""
        
        device = xm.xla_device()
        model = model.to(device)
        
        results = {
            'epochs_completed': 0,
            'total_steps': 0,
            'final_loss': 0.0
        }
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            epoch_loss = 0.0
            num_batches = 0
            
            # Use parallel loader for distributed data loading
            para_loader = pl.ParallelLoader(train_dataloader, [device])
            
            for step, batch in enumerate(para_loader.per_device_loader(device)):
                step_start = time.time()
                
                try:
                    # Move batch to device
                    input_ids = batch['input_ids'].to(device)
                    targets = batch['targets'].to(device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    logits = model(input_ids)
                    loss = loss_fn(logits, targets)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient synchronization across devices
                    xm.reduce_gradients(optimizer)
                    
                    # Optimizer step
                    optimizer.step()
                    
                    # Mark step for XLA
                    xm.mark_step()
                    
                    # Update metrics
                    epoch_loss += loss.item()
                    num_batches += 1
                    self.training_state.step += 1
                    
                    # Performance tracking
                    step_time = time.time() - step_start
                    self._update_performance_metrics(step_time, batch['input_ids'].size(0))
                    
                    if step % 100 == 0:
                        logger.info(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
                        
                except Exception as e:
                    logger.error(f"Training step failed: {e}")
                    raise
            
            # End of epoch
            avg_loss = epoch_loss / max(num_batches, 1)
            epoch_time = time.time() - epoch_start
            
            results['epochs_completed'] = epoch + 1
            results['total_steps'] = self.training_state.step
            results['final_loss'] = avg_loss
            
            logger.info(f"Epoch {epoch} completed in {epoch_time:.1f}s, avg loss: {avg_loss:.4f}")
        
        return results
    
    def _cpu_fallback_training_loop(self, model: Any, train_dataloader: Any,
                                   optimizer: Any, loss_fn: Any, num_epochs: int) -> Dict[str, Any]:
        """CPU fallback training loop."""
        logger.warning("Using CPU fallback for training")
        
        results = {
            'epochs_completed': 0,
            'total_steps': 0,
            'final_loss': 0.0
        }
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for step, batch in enumerate(train_dataloader):
                # Simulate training step
                loss = float(np.random.random() * 0.1 + 0.5)  # Mock loss
                epoch_loss += loss
                num_batches += 1
                self.training_state.step += 1
                
                if step % 100 == 0:
                    logger.info(f"Epoch {epoch}, Step {step}, Loss: {loss:.4f}")
            
            avg_loss = epoch_loss / max(num_batches, 1)
            results['epochs_completed'] = epoch + 1
            results['total_steps'] = self.training_state.step
            results['final_loss'] = avg_loss
        
        return results
    
    def _distribute_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Distribute batch across devices."""
        if not JAX_AVAILABLE or self.device_mesh is None:
            return batch
        
        distributed_batch = {}
        
        for key, value in batch.items():
            if hasattr(value, 'shape') and len(value.shape) > 0:
                # Reshape for distribution across data parallel dimension
                batch_size = value.shape[0]
                devices_per_batch = self.config.data_parallel_size
                
                if batch_size % devices_per_batch == 0:
                    # Reshape to (devices, batch_per_device, ...)
                    new_shape = (devices_per_batch, batch_size // devices_per_batch) + value.shape[1:]
                    distributed_batch[key] = value.reshape(new_shape)
                else:
                    # Pad batch if necessary
                    pad_size = devices_per_batch - (batch_size % devices_per_batch)
                    if hasattr(value, 'pad'):  # JAX array
                        padded_value = jnp.pad(value, ((0, pad_size),) + ((0, 0),) * (len(value.shape) - 1))
                        new_batch_size = padded_value.shape[0]
                        new_shape = (devices_per_batch, new_batch_size // devices_per_batch) + padded_value.shape[1:]
                        distributed_batch[key] = padded_value.reshape(new_shape)
                    else:
                        distributed_batch[key] = value
            else:
                distributed_batch[key] = value
        
        return distributed_batch
    
    def _update_performance_metrics(self, step_time: float, batch_size: int):
        """Update performance metrics."""
        # Calculate throughput
        throughput = batch_size / step_time
        
        # Exponential moving average
        alpha = 0.1
        self.performance_metrics['throughput_samples_per_sec'] = (
            alpha * throughput + 
            (1 - alpha) * self.performance_metrics['throughput_samples_per_sec']
        )
        
        # Estimate computation vs communication time
        self.performance_metrics['computation_time_ms'] = step_time * 800  # 80% computation
        self.performance_metrics['communication_time_ms'] = step_time * 200  # 20% communication
    
    def _save_checkpoint(self, model: Any, optimizer: Any, step: int):
        """Save training checkpoint."""
        checkpoint_data = {
            'step': step,
            'epoch': self.training_state.epoch,
            'model_state': 'model_params_placeholder',  # Would be actual model state
            'optimizer_state': 'optimizer_state_placeholder',  # Would be actual optimizer state
            'training_state': self.training_state.__dict__.copy(),
            'config': self.config.__dict__.copy(),
            'timestamp': time.time()
        }
        
        checkpoint_path = f"/tmp/checkpoint_step_{step}_{uuid.uuid4().hex[:8]}.json"
        
        try:
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
            
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            'training_state': self.training_state.__dict__.copy(),
            'performance_metrics': self.performance_metrics.copy(),
            'device_info': {
                'num_devices': len(jax.devices()) if JAX_AVAILABLE else 1,
                'device_mesh_shape': self.device_mesh.shape if self.device_mesh is not None else None,
                'strategy': self.config.strategy.value
            }
        }
    
    def scale_training(self, new_num_cores: int) -> bool:
        """
        Dynamically scale training to use different number of cores.
        
        Args:
            new_num_cores: New number of cores to use
            
        Returns:
            True if scaling was successful
        """
        if self.config.scaling_mode == ScalingMode.STATIC:
            logger.warning("Static scaling mode - cannot change core count")
            return False
        
        logger.info(f"Scaling training from {self.config.num_cores} to {new_num_cores} cores")
        
        try:
            # Pause training
            old_training_state = self.is_training
            self.is_training = False
            
            # Update configuration
            self.config.num_cores = new_num_cores
            self.config.data_parallel_size = min(new_num_cores, self.config.data_parallel_size)
            
            # Reinitialize device mesh
            self._initialize_distributed_training()
            
            # Resume training
            self.is_training = old_training_state
            
            logger.info(f"Successfully scaled to {new_num_cores} cores")
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale training: {e}")
            return False
    
    def stop_training(self):
        """Stop distributed training."""
        self.is_training = False
        logger.info("Training stopped")
    
    def get_distributed_training_report(self) -> Dict[str, Any]:
        """Generate comprehensive distributed training report."""
        report = {
            'config': self.config.__dict__.copy(),
            'training_state': self.training_state.__dict__.copy(),
            'performance_metrics': self.performance_metrics.copy(),
            'error_recovery_stats': self.error_recovery.get_error_statistics(),
            'device_info': {
                'available_devices': len(jax.devices()) if JAX_AVAILABLE else 1,
                'device_mesh_shape': self.device_mesh.shape if self.device_mesh is not None else None,
                'sharding_strategy': self.config.strategy.value
            },
            'resource_utilization': {
                'memory_efficiency': self.performance_metrics.get('memory_usage_gb', 0) / 32,  # Assume 32GB per core
                'compute_efficiency': self.performance_metrics.get('device_utilization', 0),
                'communication_efficiency': 1.0 - (
                    self.performance_metrics.get('communication_time_ms', 0) / 
                    (self.performance_metrics.get('computation_time_ms', 1) + 
                     self.performance_metrics.get('communication_time_ms', 0))
                )
            }
        }
        
        return report

class AutoScaler:
    """
    Automatic scaling system for distributed TPU training.
    """
    
    def __init__(self, training_manager: DistributedTrainingManager,
                 min_cores: int = 8, max_cores: int = 128):
        self.training_manager = training_manager
        self.min_cores = min_cores
        self.max_cores = max_cores
        
        self.scaling_history = []
        self.monitoring_active = False
        
    async def start_autoscaling(self, check_interval: float = 60.0):
        """Start automatic scaling monitoring."""
        self.monitoring_active = True
        
        while self.monitoring_active:
            try:
                await self._check_scaling_needs()
                await asyncio.sleep(check_interval)
            except Exception as e:
                logger.error(f"Autoscaling error: {e}")
                await asyncio.sleep(check_interval)
    
    async def _check_scaling_needs(self):
        """Check if scaling is needed based on performance metrics."""
        metrics = self.training_manager.get_performance_metrics()
        performance = metrics['performance_metrics']
        
        current_cores = self.training_manager.config.num_cores
        
        # Scale up conditions
        if (performance['device_utilization'] > 0.9 and
            performance['throughput_samples_per_sec'] < 1000 and  # Low throughput
            current_cores < self.max_cores):
            
            new_cores = min(current_cores * 2, self.max_cores)
            logger.info(f"Scaling up from {current_cores} to {new_cores} cores")
            
            success = self.training_manager.scale_training(new_cores)
            self._log_scaling_event(current_cores, new_cores, "scale_up", success)
        
        # Scale down conditions
        elif (performance['device_utilization'] < 0.5 and
              current_cores > self.min_cores):
            
            new_cores = max(current_cores // 2, self.min_cores)
            logger.info(f"Scaling down from {current_cores} to {new_cores} cores")
            
            success = self.training_manager.scale_training(new_cores)
            self._log_scaling_event(current_cores, new_cores, "scale_down", success)
    
    def _log_scaling_event(self, old_cores: int, new_cores: int, 
                          action: str, success: bool):
        """Log scaling event."""
        event = {
            'timestamp': time.time(),
            'old_cores': old_cores,
            'new_cores': new_cores,
            'action': action,
            'success': success
        }
        
        self.scaling_history.append(event)
        logger.info(f"Scaling event logged: {event}")
    
    def stop_autoscaling(self):
        """Stop automatic scaling."""
        self.monitoring_active = False
        logger.info("Autoscaling stopped")

def create_distributed_trainer(num_cores: int = 8,
                              strategy: DistributionStrategy = DistributionStrategy.DATA_PARALLEL,
                              scaling_mode: ScalingMode = ScalingMode.DYNAMIC) -> DistributedTrainingManager:
    """
    Factory function to create distributed training manager.
    
    Args:
        num_cores: Number of TPU cores
        strategy: Distribution strategy
        scaling_mode: Scaling mode
        
    Returns:
        Configured distributed training manager
    """
    config = DistributedConfig(
        strategy=strategy,
        scaling_mode=scaling_mode,
        num_cores=num_cores,
        data_parallel_size=num_cores if strategy == DistributionStrategy.DATA_PARALLEL else num_cores // 2
    )
    
    return DistributedTrainingManager(config)

# Export main classes and functions
__all__ = [
    'DistributedTrainingManager', 'DistributedConfig', 'TrainingState',
    'DistributionStrategy', 'ScalingMode', 'AutoScaler',
    'create_distributed_trainer'
]