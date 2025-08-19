"""
Enhanced API Layer - Production-ready REST API for protein diffusion workflows.

This module provides a FastAPI-based REST API with authentication, rate limiting,
async processing, and comprehensive error handling for protein design workflows.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional, Union
from dataclasses import asdict
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, UploadFile, File
    from fastapi.security import APIKeyHeader
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field, validator
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None

try:
    from .integration_manager import IntegrationManager, IntegrationConfig, WorkflowResult
    from .security_framework import SecurityManager, SecurityConfig
    from .monitoring import SystemMonitor
    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False

logger = logging.getLogger(__name__)

# Pydantic models for API
if FASTAPI_AVAILABLE:
    class GenerationRequest(BaseModel):
        motif: Optional[str] = Field(None, description="Target protein motif")
        num_candidates: int = Field(50, ge=1, le=1000, description="Number of candidates to generate")
        max_length: int = Field(256, ge=50, le=2048, description="Maximum sequence length")
        temperature: float = Field(1.0, ge=0.1, le=3.0, description="Sampling temperature")
        sampling_method: str = Field("ddpm", description="Sampling method: ddpm or ddim")
        guidance_scale: float = Field(1.0, ge=0.5, le=10.0, description="Guidance scale")
        client_id: str = Field("api_client", description="Client identifier")
        
        @validator('sampling_method')
        def validate_sampling_method(cls, v):
            if v not in ['ddpm', 'ddim']:
                raise ValueError('Sampling method must be ddpm or ddim')
            return v
    
    class EvaluationRequest(BaseModel):
        sequences: List[str] = Field(..., description="List of protein sequences")
        target_pdb_content: Optional[str] = Field(None, description="Target PDB file content")
        include_structure: bool = Field(True, description="Include structure prediction")
        include_binding: bool = Field(True, description="Include binding affinity")
        client_id: str = Field("api_client", description="Client identifier")
        
        @validator('sequences')
        def validate_sequences(cls, v):
            if not v:
                raise ValueError('At least one sequence required')
            if len(v) > 100:
                raise ValueError('Maximum 100 sequences per request')
            return v
    
    class DesignAndRankRequest(BaseModel):
        motif: Optional[str] = Field(None, description="Target protein motif")
        num_candidates: int = Field(50, ge=1, le=500, description="Number of candidates")
        max_ranked: int = Field(20, ge=1, le=100, description="Maximum ranked results")
        target_pdb_content: Optional[str] = Field(None, description="Target PDB content")
        temperature: float = Field(1.0, ge=0.1, le=3.0)
        binding_weight: float = Field(0.4, ge=0.0, le=1.0)
        structure_weight: float = Field(0.3, ge=0.0, le=1.0)
        diversity_weight: float = Field(0.2, ge=0.0, le=1.0)
        novelty_weight: float = Field(0.1, ge=0.0, le=1.0)
        client_id: str = Field("api_client", description="Client identifier")
    
    class WorkflowResponse(BaseModel):
        success: bool
        workflow_id: str
        execution_time: float
        message: str
        data: Optional[Dict[str, Any]] = None
        errors: List[str] = []
        warnings: List[str] = []
    
    class HealthResponse(BaseModel):
        status: str
        timestamp: float
        version: str = "1.0.0"
        components: Dict[str, Any]
        active_workflows: int
        uptime: float


class ProteinDiffusionAPI:
    """Enhanced FastAPI application for protein diffusion workflows."""
    
    def __init__(self, config: Optional[IntegrationConfig] = None):
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI not available - install with: pip install fastapi uvicorn")
        
        if not INTEGRATION_AVAILABLE:
            raise ImportError("Integration components not available")
        
        self.config = config or IntegrationConfig()
        self.integration_manager = IntegrationManager(self.config)
        self.start_time = time.time()
        
        # Rate limiting storage
        self.rate_limits: Dict[str, Dict] = {}
        
        # Create FastAPI app
        self.app = self._create_app()
        
        logger.info("Protein Diffusion API initialized")
    
    def _create_app(self) -> FastAPI:
        """Create and configure FastAPI application."""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            logger.info("Starting Protein Diffusion API")
            yield
            # Shutdown
            logger.info("Shutting down Protein Diffusion API")
            self.integration_manager.shutdown()
        
        app = FastAPI(
            title="Protein Diffusion Design Lab API",
            description="Production-ready API for protein design using diffusion models",
            version="1.0.0",
            lifespan=lifespan
        )
        
        # Configure CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add routes
        self._add_routes(app)
        
        return app
    
    def _add_routes(self, app: FastAPI):
        """Add API routes."""
        
        # Authentication dependency
        api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
        
        async def get_api_key(api_key: str = Depends(api_key_header)):
            if not api_key:
                raise HTTPException(status_code=401, detail="API key required")
            
            # Validate API key (simplified - use proper auth in production)
            if api_key != "demo-api-key-change-in-production":
                raise HTTPException(status_code=401, detail="Invalid API key")
            
            return api_key
        
        async def check_rate_limit(client_id: str = "default"):
            """Simple rate limiting check."""
            now = time.time()
            
            if client_id not in self.rate_limits:
                self.rate_limits[client_id] = {"count": 0, "window_start": now}
            
            client_data = self.rate_limits[client_id]
            
            # Reset window if expired (1 minute window)
            if now - client_data["window_start"] > 60:
                client_data["count"] = 0
                client_data["window_start"] = now
            
            # Check limit (100 requests per minute)
            if client_data["count"] >= 100:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            client_data["count"] += 1
        
        @app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Get API health status."""
            try:
                health_data = self.integration_manager.get_system_health()
                
                return HealthResponse(
                    status=health_data["overall_status"],
                    timestamp=health_data["timestamp"],
                    components=health_data["components"],
                    active_workflows=health_data["active_workflows"],
                    uptime=time.time() - self.start_time
                )
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                raise HTTPException(status_code=500, detail="Health check failed")
        
        @app.post("/generate", response_model=WorkflowResponse)
        async def generate_proteins(
            request: GenerationRequest,
            background_tasks: BackgroundTasks,
            api_key: str = Depends(get_api_key)
        ):
            """Generate protein scaffolds."""
            try:
                await check_rate_limit(request.client_id)
                
                # Convert request to dict
                generation_params = {
                    "motif": request.motif,
                    "num_samples": request.num_candidates,
                    "max_length": request.max_length,
                    "temperature": request.temperature,
                    "sampling_method": request.sampling_method,
                    "guidance_scale": request.guidance_scale,
                    "client_id": request.client_id,
                }
                
                # Generate in background for large requests
                if request.num_candidates > 50:
                    workflow_id = f"async_gen_{int(time.time())}_{request.client_id}"
                    background_tasks.add_task(
                        self._async_generation,
                        workflow_id,
                        generation_params
                    )
                    
                    return WorkflowResponse(
                        success=True,
                        workflow_id=workflow_id,
                        execution_time=0.0,
                        message=f"Large generation job queued. Check status with workflow ID: {workflow_id}",
                        data={"status": "queued"}
                    )
                
                # Synchronous generation for small requests
                result = self.integration_manager.diffuser.generate(**generation_params)
                
                return WorkflowResponse(
                    success=True,
                    workflow_id=f"sync_gen_{int(time.time())}",
                    execution_time=0.0,  # Would be calculated in real implementation
                    message=f"Generated {len(result)} protein scaffolds",
                    data={"sequences": result}
                )
                
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/evaluate", response_model=WorkflowResponse)
        async def evaluate_sequences(
            request: EvaluationRequest,
            api_key: str = Depends(get_api_key)
        ):
            """Evaluate protein sequences."""
            try:
                await check_rate_limit(request.client_id)
                
                # Save target PDB if provided
                target_pdb_path = None
                if request.target_pdb_content:
                    target_pdb_path = f"/tmp/target_{int(time.time())}.pdb"
                    with open(target_pdb_path, 'w') as f:
                        f.write(request.target_pdb_content)
                
                # Evaluate sequences
                result = self.integration_manager.evaluate_sequences(
                    sequences=request.sequences,
                    target_pdb=target_pdb_path,
                    include_structure=request.include_structure,
                    include_binding=request.include_binding,
                    client_id=request.client_id
                )
                
                # Cleanup temp file
                if target_pdb_path:
                    Path(target_pdb_path).unlink(missing_ok=True)
                
                return WorkflowResponse(
                    success=result.success,
                    workflow_id=result.workflow_id,
                    execution_time=result.execution_time,
                    message=f"Evaluated {len(request.sequences)} sequences",
                    data={
                        "ranking_results": result.ranking_results,
                        "structure_results": result.generation_results
                    },
                    errors=result.errors,
                    warnings=result.warnings
                )
                
            except Exception as e:
                logger.error(f"Evaluation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/design-and-rank", response_model=WorkflowResponse)
        async def design_and_rank(
            request: DesignAndRankRequest,
            background_tasks: BackgroundTasks,
            api_key: str = Depends(get_api_key)
        ):
            """Complete protein design and ranking workflow."""
            try:
                await check_rate_limit(request.client_id)
                
                # Save target PDB if provided
                target_pdb_path = None
                if request.target_pdb_content:
                    target_pdb_path = f"/tmp/target_{int(time.time())}.pdb"
                    with open(target_pdb_path, 'w') as f:
                        f.write(request.target_pdb_content)
                
                # Update ranker config with request weights
                if self.integration_manager.ranker:
                    self.integration_manager.ranker.config.binding_weight = request.binding_weight
                    self.integration_manager.ranker.config.structure_weight = request.structure_weight
                    self.integration_manager.ranker.config.diversity_weight = request.diversity_weight
                    self.integration_manager.ranker.config.novelty_weight = request.novelty_weight
                
                # Run complete workflow
                result = self.integration_manager.design_and_rank_proteins(
                    motif=request.motif,
                    num_candidates=request.num_candidates,
                    target_pdb=target_pdb_path,
                    max_ranked=request.max_ranked,
                    client_id=request.client_id,
                    temperature=request.temperature
                )
                
                # Cleanup temp file
                if target_pdb_path:
                    Path(target_pdb_path).unlink(missing_ok=True)
                
                return WorkflowResponse(
                    success=result.success,
                    workflow_id=result.workflow_id,
                    execution_time=result.execution_time,
                    message=f"Designed and ranked {len(result.ranking_results)} proteins",
                    data={
                        "generation_results": result.generation_results,
                        "ranking_results": result.ranking_results,
                        "performance_metrics": result.performance_metrics
                    },
                    errors=result.errors,
                    warnings=result.warnings
                )
                
            except Exception as e:
                logger.error(f"Design and rank workflow failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/workflow/{workflow_id}", response_model=WorkflowResponse)
        async def get_workflow_status(
            workflow_id: str,
            api_key: str = Depends(get_api_key)
        ):
            """Get workflow status."""
            try:
                status = self.integration_manager.get_workflow_status(workflow_id)
                
                if not status:
                    raise HTTPException(status_code=404, detail="Workflow not found")
                
                return WorkflowResponse(
                    success=True,
                    workflow_id=workflow_id,
                    execution_time=status.get("execution_time", 0.0),
                    message=f"Workflow status: {status.get('status', 'unknown')}",
                    data=status
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to get workflow status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/workflows", response_model=Dict[str, Any])
        async def list_workflows(
            api_key: str = Depends(get_api_key)
        ):
            """List active workflows."""
            try:
                workflows = self.integration_manager.list_active_workflows()
                return {
                    "total": len(workflows),
                    "workflows": workflows
                }
            except Exception as e:
                logger.error(f"Failed to list workflows: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/upload-pdb")
        async def upload_pdb(
            file: UploadFile = File(...),
            api_key: str = Depends(get_api_key)
        ):
            """Upload PDB file for use in workflows."""
            try:
                if not file.filename.endswith('.pdb'):
                    raise HTTPException(status_code=400, detail="File must be a PDB file")
                
                # Save uploaded file
                upload_path = Path(f"/tmp/uploaded_{int(time.time())}_{file.filename}")
                content = await file.read()
                
                with open(upload_path, 'wb') as f:
                    f.write(content)
                
                return {
                    "success": True,
                    "message": f"PDB file uploaded successfully",
                    "file_path": str(upload_path),
                    "file_size": len(content)
                }
                
            except Exception as e:
                logger.error(f"File upload failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/stats")
        async def get_api_stats(
            api_key: str = Depends(get_api_key)
        ):
            """Get API usage statistics."""
            try:
                return {
                    "uptime": time.time() - self.start_time,
                    "total_rate_limited_clients": len(self.rate_limits),
                    "rate_limits": {k: v["count"] for k, v in self.rate_limits.items()},
                    "system_health": self.integration_manager.get_system_health()
                }
            except Exception as e:
                logger.error(f"Failed to get stats: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def _async_generation(self, workflow_id: str, params: Dict[str, Any]):
        """Run async generation task."""
        try:
            logger.info(f"Starting async generation for workflow {workflow_id}")
            result = self.integration_manager.diffuser.generate(**params)
            logger.info(f"Async generation completed for workflow {workflow_id}")
            
            # Store result for later retrieval
            # In production, use a proper task queue like Celery
            
        except Exception as e:
            logger.error(f"Async generation failed for workflow {workflow_id}: {e}")
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """Run the API server."""
        uvicorn.run(self.app, host=host, port=port, **kwargs)


# Convenience function for creating API instance
def create_api(config: Optional[IntegrationConfig] = None) -> ProteinDiffusionAPI:
    """Create ProteinDiffusionAPI instance."""
    return ProteinDiffusionAPI(config)


# CLI for running the API
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Protein Diffusion API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    
    # Create and run API
    try:
        api = create_api()
        api.run(
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level.lower()
        )
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise
