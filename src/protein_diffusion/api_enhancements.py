"""
Enhanced Public API for Protein Diffusion Design Lab

This module provides optimized, user-friendly interfaces including:
- Simplified high-level API for common use cases
- Streaming generation with real-time feedback
- Batch processing with progress tracking
- Template-based generation for specific protein types
- Integration helpers for common workflows
"""

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    from . import mock_torch as torch
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    class MockNumpy:
        @staticmethod
        def mean(arr): return 0.5
        @staticmethod
        def array(data): return data
    np = MockNumpy()
    NUMPY_AVAILABLE = False

from typing import List, Dict, Optional, Union, Iterator, Callable, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging
import time
import json
from abc import ABC, abstractmethod

from .diffuser import ProteinDiffuser, ProteinDiffuserConfig
from .ranker import AffinityRanker, AffinityRankerConfig
from .advanced_generation import AdvancedGenerationPipeline, AdvancedGenerationConfig

logger = logging.getLogger(__name__)


@dataclass
class GenerationRequest:
    """Simplified generation request structure."""
    # Core parameters
    target_type: str = "general"  # "antibody", "enzyme", "binder", "general"
    sequence_length: Optional[int] = None
    motif: Optional[str] = None
    
    # Generation settings
    num_candidates: int = 10
    quality_filter: bool = True
    diversity_boost: bool = True
    
    # Advanced options
    temperature: float = 1.0
    guidance_scale: float = 1.0
    sampling_method: str = "ddpm"
    
    # Output preferences
    return_confidence: bool = True
    return_structure_prediction: bool = False
    return_binding_prediction: bool = False
    
    # Constraints
    constraints: Dict[str, Any] = field(default_factory=dict)
    target_properties: Dict[str, float] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Standardized generation result."""
    sequence: str
    confidence: float
    generation_time: float
    
    # Optional fields
    structure_quality: Optional[float] = None
    binding_affinity: Optional[float] = None
    diversity_score: Optional[float] = None
    novelty_score: Optional[float] = None
    
    # Metadata
    generation_method: str = "diffusion"
    model_version: str = "1.0"
    request_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sequence": self.sequence,
            "confidence": self.confidence,
            "generation_time": self.generation_time,
            "structure_quality": self.structure_quality,
            "binding_affinity": self.binding_affinity,
            "diversity_score": self.diversity_score,
            "novelty_score": self.novelty_score,
            "generation_method": self.generation_method,
            "model_version": self.model_version,
            "request_id": self.request_id,
        }


class ProteinTemplate:
    """Template for specific protein types with optimized parameters."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
    
    @staticmethod
    def get_antibody_template() -> "ProteinTemplate":
        """Template for antibody generation."""
        config = {
            "sequence_length": 120,
            "motif": "VARIABLE_CONSTANT",
            "temperature": 0.8,
            "guidance_scale": 1.2,
            "constraints": {
                "cysteine_pairs": True,
                "hydrophobic_core": True,
                "charge_distribution": "balanced"
            },
            "target_properties": {
                "binding_specificity": 0.9,
                "stability": 0.8
            }
        }
        return ProteinTemplate("antibody", config)
    
    @staticmethod
    def get_enzyme_template() -> "ProteinTemplate":
        """Template for enzyme generation."""
        config = {
            "sequence_length": 200,
            "motif": "ACTIVE_SITE_CENTERED",
            "temperature": 0.9,
            "guidance_scale": 1.0,
            "constraints": {
                "active_site_geometry": True,
                "substrate_binding": True,
                "catalytic_triad": True
            },
            "target_properties": {
                "catalytic_efficiency": 0.85,
                "substrate_specificity": 0.75
            }
        }
        return ProteinTemplate("enzyme", config)
    
    @staticmethod
    def get_binder_template() -> "ProteinTemplate":
        """Template for binding protein generation."""
        config = {
            "sequence_length": 80,
            "motif": "BINDING_INTERFACE",
            "temperature": 1.0,
            "guidance_scale": 1.5,
            "constraints": {
                "interface_residues": True,
                "binding_affinity": True
            },
            "target_properties": {
                "binding_strength": 0.9,
                "selectivity": 0.8
            }
        }
        return ProteinTemplate("binder", config)
    
    @staticmethod
    def get_scaffold_template() -> "ProteinTemplate":
        """Template for scaffold protein generation."""
        config = {
            "sequence_length": 150,
            "motif": "STRUCTURED_CORE",
            "temperature": 1.1,
            "guidance_scale": 0.9,
            "constraints": {
                "secondary_structure": True,
                "stability": True
            },
            "target_properties": {
                "foldability": 0.9,
                "thermostability": 0.7
            }
        }
        return ProteinTemplate("scaffold", config)


class StreamingGenerator:
    """Streaming protein generation with real-time feedback."""
    
    def __init__(self, diffuser: ProteinDiffuser):
        self.diffuser = diffuser
        self.progress_callbacks: List[Callable] = []
        self.quality_callbacks: List[Callable] = []
    
    def add_progress_callback(self, callback: Callable[[int, int, str], None]):
        """Add callback for progress updates: callback(current, total, stage)."""
        self.progress_callbacks.append(callback)
    
    def add_quality_callback(self, callback: Callable[[float, str], None]):
        """Add callback for quality updates: callback(quality_score, sequence)."""
        self.quality_callbacks.append(callback)
    
    def generate_stream(
        self,
        request: GenerationRequest,
        stream_quality_checks: bool = True
    ) -> Iterator[GenerationResult]:
        """
        Generate proteins with streaming updates.
        
        Args:
            request: Generation request
            stream_quality_checks: Perform real-time quality assessment
            
        Yields:
            GenerationResult objects as they are completed
        """
        start_time = time.time()
        
        # Notify progress callbacks
        for callback in self.progress_callbacks:
            callback(0, request.num_candidates, "starting")
        
        # Generate in batches for streaming
        batch_size = min(5, request.num_candidates)
        total_generated = 0
        
        while total_generated < request.num_candidates:
            current_batch_size = min(batch_size, request.num_candidates - total_generated)
            
            # Update progress
            for callback in self.progress_callbacks:
                callback(total_generated, request.num_candidates, "generating")
            
            # Generate batch
            batch_results = self.diffuser.generate(
                motif=request.motif,
                num_samples=current_batch_size,
                max_length=request.sequence_length,
                temperature=request.temperature,
                guidance_scale=request.guidance_scale,
                sampling_method=request.sampling_method,
                progress=False  # We handle progress externally
            )
            
            # Process and stream results
            for i, result in enumerate(batch_results):
                if "error" in result:
                    continue
                
                # Create standardized result
                gen_result = GenerationResult(
                    sequence=result["sequence"],
                    confidence=result.get("confidence", 0.0),
                    generation_time=time.time() - start_time,
                    generation_method="streaming_diffusion",
                    request_id=f"stream_{total_generated + i}"
                )
                
                # Real-time quality assessment
                if stream_quality_checks:
                    try:
                        quality_score = self._assess_quality(gen_result.sequence)
                        gen_result.structure_quality = quality_score
                        
                        # Notify quality callbacks
                        for callback in self.quality_callbacks:
                            callback(quality_score, gen_result.sequence)
                    except Exception as e:
                        logger.warning(f"Quality assessment failed: {e}")
                
                yield gen_result
                total_generated += 1
        
        # Final progress update
        for callback in self.progress_callbacks:
            callback(total_generated, request.num_candidates, "completed")
    
    def _assess_quality(self, sequence: str) -> float:
        """Quick quality assessment for streaming."""
        # Simplified quality metrics
        length_score = 1.0 if 20 <= len(sequence) <= 500 else 0.5
        
        # Basic composition checks
        aa_counts = {aa: sequence.count(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"}
        total_aa = len(sequence)
        
        # Check for reasonable amino acid distribution
        composition_score = 1.0
        for aa, count in aa_counts.items():
            frequency = count / total_aa
            if frequency > 0.3:  # No single AA should dominate
                composition_score *= 0.8
        
        # Check for stop codons or unusual patterns
        pattern_score = 1.0
        if "XX" in sequence or "***" in sequence:
            pattern_score = 0.1
        
        return length_score * composition_score * pattern_score


class BatchProcessor:
    """Efficient batch processing for multiple generation requests."""
    
    def __init__(self, diffuser: ProteinDiffuser, ranker: Optional[AffinityRanker] = None):
        self.diffuser = diffuser
        self.ranker = ranker
        self.processing_stats = {
            "total_requests": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "average_generation_time": 0.0
        }
    
    def process_batch(
        self,
        requests: List[GenerationRequest],
        max_concurrent: int = 5,
        enable_ranking: bool = True,
        progress_callback: Optional[Callable] = None
    ) -> List[List[GenerationResult]]:
        """
        Process multiple generation requests efficiently.
        
        Args:
            requests: List of generation requests
            max_concurrent: Maximum concurrent requests
            enable_ranking: Enable ranking of results
            progress_callback: Optional progress callback
            
        Returns:
            List of generation results for each request
        """
        logger.info(f"Processing batch of {len(requests)} requests")
        start_time = time.time()
        
        all_results = []
        self.processing_stats["total_requests"] = len(requests)
        
        # Process requests in batches
        for i in range(0, len(requests), max_concurrent):
            batch_requests = requests[i:i + max_concurrent]
            
            if progress_callback:
                progress_callback(i, len(requests), "processing")
            
            # Process batch
            batch_results = self._process_request_batch(batch_requests, enable_ranking)
            all_results.extend(batch_results)
            
            # Update progress
            if progress_callback:
                progress_callback(min(i + max_concurrent, len(requests)), len(requests), "processing")
        
        # Update statistics
        total_time = time.time() - start_time
        successful = sum(1 for results in all_results for r in results if r.confidence > 0)
        self.processing_stats["successful_generations"] = successful
        self.processing_stats["average_generation_time"] = total_time / len(requests)
        
        logger.info(f"Batch processing complete: {successful} successful generations in {total_time:.2f}s")
        
        return all_results
    
    def _process_request_batch(
        self,
        requests: List[GenerationRequest],
        enable_ranking: bool
    ) -> List[List[GenerationResult]]:
        """Process a batch of requests concurrently."""
        batch_results = []
        
        for request in requests:
            try:
                # Generate sequences
                start_time = time.time()
                
                generation_results = self.diffuser.generate(
                    motif=request.motif,
                    num_samples=request.num_candidates,
                    max_length=request.sequence_length,
                    temperature=request.temperature,
                    guidance_scale=request.guidance_scale,
                    sampling_method=request.sampling_method,
                    progress=False
                )
                
                generation_time = time.time() - start_time
                
                # Convert to standardized results
                standardized_results = []
                for i, result in enumerate(generation_results):
                    if "error" in result:
                        continue
                    
                    gen_result = GenerationResult(
                        sequence=result["sequence"],
                        confidence=result.get("confidence", 0.0),
                        generation_time=generation_time / len(generation_results),
                        request_id=f"batch_{id(request)}_{i}"
                    )
                    
                    # Add optional evaluations
                    if request.return_structure_prediction:
                        gen_result.structure_quality = self._predict_structure_quality(result["sequence"])
                    
                    if request.return_binding_prediction:
                        gen_result.binding_affinity = self._predict_binding_affinity(result["sequence"])
                    
                    standardized_results.append(gen_result)
                
                # Ranking if enabled
                if enable_ranking and self.ranker and standardized_results:
                    standardized_results = self._rank_results(standardized_results)
                
                batch_results.append(standardized_results)
                
            except Exception as e:
                logger.error(f"Request processing failed: {e}")
                batch_results.append([])
        
        return batch_results
    
    def _predict_structure_quality(self, sequence: str) -> float:
        """Predict structure quality (placeholder)."""
        # Simple heuristic - in practice, use trained model
        return min(1.0, len(sequence) / 100.0 * 0.8 + 0.2)
    
    def _predict_binding_affinity(self, sequence: str) -> float:
        """Predict binding affinity (placeholder)."""
        # Simple heuristic - in practice, use trained model
        hydrophobic = sum(1 for aa in sequence if aa in "AILMFPWV")
        return -5.0 - (hydrophobic / len(sequence)) * 8.0
    
    def _rank_results(self, results: List[GenerationResult]) -> List[GenerationResult]:
        """Rank results using the affinity ranker."""
        if not self.ranker:
            return results
        
        try:
            sequences = [r.sequence for r in results]
            rankings = self.ranker.rank(sequences, return_detailed=False)
            
            # Update results with ranking scores
            for result, ranking in zip(results, rankings):
                result.binding_affinity = ranking.get("binding_affinity", result.binding_affinity)
                result.structure_quality = ranking.get("structure_quality", result.structure_quality)
            
            # Sort by composite score
            results.sort(key=lambda x: x.confidence, reverse=True)
            
        except Exception as e:
            logger.warning(f"Ranking failed: {e}")
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.processing_stats.copy()


class ProteinDesignWorkflow:
    """High-level workflow for common protein design tasks."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[ProteinDiffuserConfig] = None,
        device: str = "auto"
    ):
        # Auto-detect device
        if device == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        # Initialize core components
        if config is None:
            config = ProteinDiffuserConfig(device=device)
        
        if model_path:
            self.diffuser = ProteinDiffuser.from_pretrained(model_path, config)
        else:
            self.diffuser = ProteinDiffuser(config)
        
        self.ranker = AffinityRanker()
        self.batch_processor = BatchProcessor(self.diffuser, self.ranker)
        self.streaming_generator = StreamingGenerator(self.diffuser)
        
        # Templates
        self.templates = {
            "antibody": ProteinTemplate.get_antibody_template(),
            "enzyme": ProteinTemplate.get_enzyme_template(),
            "binder": ProteinTemplate.get_binder_template(),
            "scaffold": ProteinTemplate.get_scaffold_template()
        }
        
        logger.info(f"ProteinDesignWorkflow initialized on {device}")
    
    def design_antibody(
        self,
        target_antigen: Optional[str] = None,
        num_candidates: int = 20,
        **kwargs
    ) -> List[GenerationResult]:
        """Design antibody sequences for a target antigen."""
        template = self.templates["antibody"]
        request = self._create_request_from_template(template, num_candidates, **kwargs)
        
        if target_antigen:
            request.constraints["target_antigen"] = target_antigen
        
        return self._execute_workflow(request)
    
    def design_enzyme(
        self,
        substrate: Optional[str] = None,
        reaction_type: Optional[str] = None,
        num_candidates: int = 15,
        **kwargs
    ) -> List[GenerationResult]:
        """Design enzyme sequences for catalyzing specific reactions."""
        template = self.templates["enzyme"]
        request = self._create_request_from_template(template, num_candidates, **kwargs)
        
        if substrate:
            request.constraints["substrate"] = substrate
        if reaction_type:
            request.constraints["reaction_type"] = reaction_type
        
        return self._execute_workflow(request)
    
    def design_binder(
        self,
        target_protein: str,
        binding_site: Optional[str] = None,
        num_candidates: int = 25,
        **kwargs
    ) -> List[GenerationResult]:
        """Design protein binders for a target protein."""
        template = self.templates["binder"]
        request = self._create_request_from_template(template, num_candidates, **kwargs)
        
        request.constraints["target_protein"] = target_protein
        if binding_site:
            request.constraints["binding_site"] = binding_site
        
        # Enable binding prediction for binders
        request.return_binding_prediction = True
        
        return self._execute_workflow(request)
    
    def design_scaffold(
        self,
        secondary_structure: Optional[str] = None,
        num_candidates: int = 10,
        **kwargs
    ) -> List[GenerationResult]:
        """Design scaffold proteins with specified structure."""
        template = self.templates["scaffold"]
        request = self._create_request_from_template(template, num_candidates, **kwargs)
        
        if secondary_structure:
            request.motif = secondary_structure
        
        # Enable structure prediction for scaffolds
        request.return_structure_prediction = True
        
        return self._execute_workflow(request)
    
    def design_custom(
        self,
        target_type: str,
        custom_template: Optional[ProteinTemplate] = None,
        **kwargs
    ) -> List[GenerationResult]:
        """Design proteins with custom parameters."""
        if custom_template:
            template = custom_template
        else:
            template = self.templates.get(target_type, self.templates["scaffold"])
        
        request = self._create_request_from_template(template, **kwargs)
        return self._execute_workflow(request)
    
    def _create_request_from_template(
        self,
        template: ProteinTemplate,
        num_candidates: int = 10,
        **overrides
    ) -> GenerationRequest:
        """Create generation request from template."""
        # Start with template config
        config = template.config.copy()
        config.update(overrides)
        
        return GenerationRequest(
            target_type=template.name,
            num_candidates=num_candidates,
            sequence_length=config.get("sequence_length"),
            motif=config.get("motif"),
            temperature=config.get("temperature", 1.0),
            guidance_scale=config.get("guidance_scale", 1.0),
            constraints=config.get("constraints", {}),
            target_properties=config.get("target_properties", {}),
        )
    
    def _execute_workflow(self, request: GenerationRequest) -> List[GenerationResult]:
        """Execute the generation workflow."""
        logger.info(f"Executing {request.target_type} design workflow")
        
        # Use batch processor for single request
        results = self.batch_processor.process_batch([request], enable_ranking=True)
        
        if results:
            return results[0]
        else:
            return []
    
    def get_available_templates(self) -> List[str]:
        """Get list of available protein templates."""
        return list(self.templates.keys())
    
    def add_custom_template(self, name: str, template: ProteinTemplate):
        """Add a custom protein template."""
        self.templates[name] = template
        logger.info(f"Added custom template: {name}")
    
    def generate_stream(
        self,
        request: GenerationRequest,
        progress_callback: Optional[Callable] = None,
        quality_callback: Optional[Callable] = None
    ) -> Iterator[GenerationResult]:
        """Generate proteins with streaming updates."""
        if progress_callback:
            self.streaming_generator.add_progress_callback(progress_callback)
        if quality_callback:
            self.streaming_generator.add_quality_callback(quality_callback)
        
        yield from self.streaming_generator.generate_stream(request)
    
    def health_check(self) -> Dict[str, Any]:
        """Perform system health check."""
        return self.diffuser.health_check()
    
    def save_results(
        self,
        results: List[GenerationResult],
        output_path: str,
        format: str = "json"
    ):
        """Save generation results to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            data = {
                "timestamp": time.time(),
                "results": [r.to_dict() for r in results],
                "statistics": {
                    "total_sequences": len(results),
                    "avg_confidence": np.mean([r.confidence for r in results]),
                    "avg_length": np.mean([len(r.sequence) for r in results])
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif format == "fasta":
            with open(output_path, 'w') as f:
                for i, result in enumerate(results):
                    f.write(f">sequence_{i}_confidence_{result.confidence:.3f}\n")
                    f.write(f"{result.sequence}\n")
        
        logger.info(f"Results saved to {output_path}")


# Convenience functions for quick access
def design_antibody(target_antigen: str = None, num_candidates: int = 20, **kwargs) -> List[GenerationResult]:
    """Quick antibody design function."""
    workflow = ProteinDesignWorkflow()
    return workflow.design_antibody(target_antigen, num_candidates, **kwargs)


def design_enzyme(substrate: str = None, num_candidates: int = 15, **kwargs) -> List[GenerationResult]:
    """Quick enzyme design function."""
    workflow = ProteinDesignWorkflow()
    return workflow.design_enzyme(substrate, num_candidates=num_candidates, **kwargs)


def design_binder(target_protein: str, num_candidates: int = 25, **kwargs) -> List[GenerationResult]:
    """Quick binder design function."""
    workflow = ProteinDesignWorkflow()
    return workflow.design_binder(target_protein, num_candidates=num_candidates, **kwargs)


def quick_generate(
    sequence_type: str = "general",
    length: int = 100,
    num_sequences: int = 5,
    **kwargs
) -> List[str]:
    """Ultra-simple generation function returning just sequences."""
    workflow = ProteinDesignWorkflow()
    
    request = GenerationRequest(
        target_type=sequence_type,
        sequence_length=length,
        num_candidates=num_sequences,
        **kwargs
    )
    
    results = workflow._execute_workflow(request)
    return [r.sequence for r in results]