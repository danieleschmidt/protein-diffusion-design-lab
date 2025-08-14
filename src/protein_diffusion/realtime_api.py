"""
Real-Time Protein Diffusion API

Advanced async API with WebSocket streaming, Server-Sent Events,
and microservices-ready architecture for real-time protein generation.
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Any, Optional, AsyncGenerator, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import logging
from enum import Enum
import weakref

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

try:
    import fastapi
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
    from fastapi.responses import StreamingResponse
    from sse_starlette.sse import EventSourceResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


logger = logging.getLogger(__name__)


class StreamEventType(Enum):
    """Types of streaming events."""
    GENERATION_START = "generation_start"
    GENERATION_PROGRESS = "generation_progress"
    GENERATION_COMPLETE = "generation_complete"
    VALIDATION_START = "validation_start"
    VALIDATION_COMPLETE = "validation_complete"
    FOLDING_START = "folding_start"
    FOLDING_COMPLETE = "folding_complete"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


@dataclass
class StreamEvent:
    """Real-time stream event."""
    event_type: StreamEventType
    session_id: str
    timestamp: datetime
    data: Dict[str, Any]
    progress: float = 0.0
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "progress": self.progress,
            "message": self.message
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class GenerationSession:
    """Active generation session tracking."""
    session_id: str
    user_id: Optional[str]
    start_time: datetime
    last_activity: datetime
    parameters: Dict[str, Any]
    status: str = "active"
    progress: float = 0.0
    results: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.results is None:
            self.results = []


class ConnectionManager:
    """WebSocket connection manager for real-time updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.session_connections: Dict[str, List[WebSocket]] = {}
        self.user_connections: Dict[str, List[WebSocket]] = {}
        
    async def connect(self, websocket: WebSocket, session_id: str, user_id: Optional[str] = None):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        if session_id not in self.session_connections:
            self.session_connections[session_id] = []
        self.session_connections[session_id].append(websocket)
        
        if user_id:
            if user_id not in self.user_connections:
                self.user_connections[user_id] = []
            self.user_connections[user_id].append(websocket)
            
        logger.info(f"WebSocket connected: session={session_id}, user={user_id}")
        
    def disconnect(self, websocket: WebSocket, session_id: str, user_id: Optional[str] = None):
        """Remove WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            
        if session_id in self.session_connections:
            if websocket in self.session_connections[session_id]:
                self.session_connections[session_id].remove(websocket)
                
        if user_id and user_id in self.user_connections:
            if websocket in self.user_connections[user_id]:
                self.user_connections[user_id].remove(websocket)
                
        logger.info(f"WebSocket disconnected: session={session_id}, user={user_id}")
    
    async def send_to_session(self, session_id: str, message: str):
        """Send message to all connections in a session."""
        if session_id in self.session_connections:
            connections = self.session_connections[session_id].copy()
            for websocket in connections:
                try:
                    await websocket.send_text(message)
                except Exception as e:
                    logger.error(f"Error sending to session {session_id}: {e}")
                    self.session_connections[session_id].remove(websocket)
    
    async def send_to_user(self, user_id: str, message: str):
        """Send message to all user connections."""
        if user_id in self.user_connections:
            connections = self.user_connections[user_id].copy()
            for websocket in connections:
                try:
                    await websocket.send_text(message)
                except Exception as e:
                    logger.error(f"Error sending to user {user_id}: {e}")
                    self.user_connections[user_id].remove(websocket)
    
    async def broadcast(self, message: str):
        """Broadcast message to all active connections."""
        connections = self.active_connections.copy()
        for websocket in connections:
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting: {e}")
                self.active_connections.remove(websocket)


class RealTimeProteinAPI:
    """Real-time protein diffusion API with streaming capabilities."""
    
    def __init__(self):
        self.sessions: Dict[str, GenerationSession] = {}
        self.connection_manager = ConnectionManager()
        self.event_callbacks: Dict[StreamEventType, List[Callable]] = {}
        self.background_tasks: List[asyncio.Task] = []
        
        # Initialize heartbeat task
        self._start_heartbeat()
        
    def _start_heartbeat(self):
        """Start background heartbeat task."""
        async def heartbeat():
            while True:
                await asyncio.sleep(30)  # 30 second heartbeat
                event = StreamEvent(
                    event_type=StreamEventType.HEARTBEAT,
                    session_id="system",
                    timestamp=datetime.now(timezone.utc),
                    data={"status": "healthy"},
                    message="System heartbeat"
                )
                await self.connection_manager.broadcast(event.to_json())
        
        if asyncio.get_event_loop().is_running():
            task = asyncio.create_task(heartbeat())
            self.background_tasks.append(task)
    
    def register_callback(self, event_type: StreamEventType, callback: Callable):
        """Register event callback."""
        if event_type not in self.event_callbacks:
            self.event_callbacks[event_type] = []
        self.event_callbacks[event_type].append(callback)
    
    async def _emit_event(self, event: StreamEvent):
        """Emit stream event to connected clients."""
        message = event.to_json()
        
        # Send to session-specific connections
        await self.connection_manager.send_to_session(event.session_id, message)
        
        # Call registered callbacks
        if event.event_type in self.event_callbacks:
            for callback in self.event_callbacks[event.event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                except Exception as e:
                    logger.error(f"Error in event callback: {e}")
    
    def create_session(self, user_id: Optional[str] = None, parameters: Optional[Dict[str, Any]] = None) -> str:
        """Create new generation session."""
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        
        session = GenerationSession(
            session_id=session_id,
            user_id=user_id,
            start_time=now,
            last_activity=now,
            parameters=parameters or {}
        )
        
        self.sessions[session_id] = session
        logger.info(f"Created session {session_id} for user {user_id}")
        return session_id
    
    async def generate_protein_stream(
        self,
        session_id: str,
        motif: str,
        num_samples: int = 1,
        temperature: float = 0.8,
        guidance_scale: float = 1.0
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream protein generation with real-time updates."""
        
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        session.last_activity = datetime.now(timezone.utc)
        session.status = "generating"
        
        try:
            # Emit generation start event
            start_event = StreamEvent(
                event_type=StreamEventType.GENERATION_START,
                session_id=session_id,
                timestamp=datetime.now(timezone.utc),
                data={
                    "motif": motif,
                    "num_samples": num_samples,
                    "temperature": temperature,
                    "guidance_scale": guidance_scale
                },
                message=f"Starting generation of {num_samples} proteins"
            )
            await self._emit_event(start_event)
            yield start_event
            
            # Simulate generation with progress updates
            for i in range(num_samples):
                # Simulate generation steps
                for step in range(10):
                    progress = ((i * 10) + step + 1) / (num_samples * 10)
                    
                    progress_event = StreamEvent(
                        event_type=StreamEventType.GENERATION_PROGRESS,
                        session_id=session_id,
                        timestamp=datetime.now(timezone.utc),
                        data={
                            "sample": i + 1,
                            "step": step + 1,
                            "total_steps": 10
                        },
                        progress=progress,
                        message=f"Generating protein {i+1}/{num_samples} - Step {step+1}/10"
                    )
                    await self._emit_event(progress_event)
                    yield progress_event
                    
                    await asyncio.sleep(0.1)  # Simulate processing time
                
                # Generate mock protein result
                protein_result = {
                    "sequence": f"MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG_{i}",
                    "confidence": 0.85 + (i * 0.02),
                    "structure_prediction": f"structure_{session_id}_{i}.pdb",
                    "binding_affinity": -8.5 - (i * 0.3),
                    "generated_at": datetime.now(timezone.utc).isoformat()
                }
                session.results.append(protein_result)
            
            # Emit generation complete event
            complete_event = StreamEvent(
                event_type=StreamEventType.GENERATION_COMPLETE,
                session_id=session_id,
                timestamp=datetime.now(timezone.utc),
                data={
                    "results": session.results,
                    "total_generated": len(session.results)
                },
                progress=1.0,
                message=f"Generated {num_samples} proteins successfully"
            )
            await self._emit_event(complete_event)
            yield complete_event
            
            session.status = "completed"
            session.progress = 1.0
            
        except Exception as e:
            error_event = StreamEvent(
                event_type=StreamEventType.ERROR,
                session_id=session_id,
                timestamp=datetime.now(timezone.utc),
                data={"error": str(e)},
                message=f"Generation failed: {str(e)}"
            )
            await self._emit_event(error_event)
            yield error_event
            
            session.status = "failed"
    
    async def validate_protein_stream(
        self,
        session_id: str,
        protein_sequence: str
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream protein validation with real-time updates."""
        
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        session.last_activity = datetime.now(timezone.utc)
        
        try:
            start_event = StreamEvent(
                event_type=StreamEventType.VALIDATION_START,
                session_id=session_id,
                timestamp=datetime.now(timezone.utc),
                data={"sequence_length": len(protein_sequence)},
                message="Starting protein validation"
            )
            await self._emit_event(start_event)
            yield start_event
            
            # Simulate validation steps
            validation_steps = [
                "Checking sequence format",
                "Validating amino acid composition", 
                "Analyzing secondary structure",
                "Predicting folding stability",
                "Computing binding properties"
            ]
            
            results = {}
            for i, step in enumerate(validation_steps):
                progress = (i + 1) / len(validation_steps)
                
                progress_event = StreamEvent(
                    event_type=StreamEventType.VALIDATION_PROGRESS,
                    session_id=session_id,
                    timestamp=datetime.now(timezone.utc),
                    data={"current_step": step},
                    progress=progress,
                    message=f"Validation step {i+1}/{len(validation_steps)}: {step}"
                )
                await self._emit_event(progress_event)
                yield progress_event
                
                # Simulate validation result
                results[step.lower().replace(" ", "_")] = {
                    "status": "valid",
                    "score": 0.8 + (i * 0.04),
                    "details": f"Passed {step.lower()}"
                }
                
                await asyncio.sleep(0.2)
            
            complete_event = StreamEvent(
                event_type=StreamEventType.VALIDATION_COMPLETE,
                session_id=session_id,
                timestamp=datetime.now(timezone.utc),
                data={"validation_results": results},
                progress=1.0,
                message="Protein validation completed successfully"
            )
            await self._emit_event(complete_event)
            yield complete_event
            
        except Exception as e:
            error_event = StreamEvent(
                event_type=StreamEventType.ERROR,
                session_id=session_id,
                timestamp=datetime.now(timezone.utc),
                data={"error": str(e)},
                message=f"Validation failed: {str(e)}"
            )
            await self._emit_event(error_event)
            yield error_event
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current session status."""
        if session_id not in self.sessions:
            return None
            
        session = self.sessions[session_id]
        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "status": session.status,
            "progress": session.progress,
            "start_time": session.start_time.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "results_count": len(session.results),
            "parameters": session.parameters
        }
    
    async def cleanup_sessions(self, max_age_hours: int = 24):
        """Clean up old sessions."""
        now = datetime.now(timezone.utc)
        cutoff = now.timestamp() - (max_age_hours * 3600)
        
        sessions_to_remove = []
        for session_id, session in self.sessions.items():
            if session.last_activity.timestamp() < cutoff:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.sessions[session_id]
            logger.info(f"Cleaned up session {session_id}")
    
    def get_active_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get all active session statuses."""
        return {
            session_id: self.get_session_status(session_id)
            for session_id in self.sessions.keys()
        }


# FastAPI integration
if FASTAPI_AVAILABLE:
    
    class FastAPIRealTimeProteinAPI:
        """FastAPI integration for real-time protein API."""
        
        def __init__(self):
            self.api = RealTimeProteinAPI()
            self.app = FastAPI(
                title="Real-Time Protein Diffusion API",
                description="Advanced async API for real-time protein generation",
                version="1.0.0"
            )
            self._setup_routes()
        
        def _setup_routes(self):
            """Setup FastAPI routes."""
            
            @self.app.websocket("/ws/{session_id}")
            async def websocket_endpoint(websocket: WebSocket, session_id: str, user_id: str = None):
                await self.api.connection_manager.connect(websocket, session_id, user_id)
                try:
                    while True:
                        data = await websocket.receive_text()
                        message = json.loads(data)
                        
                        if message.get("type") == "ping":
                            await websocket.send_text(json.dumps({"type": "pong"}))
                        
                except WebSocketDisconnect:
                    self.api.connection_manager.disconnect(websocket, session_id, user_id)
            
            @self.app.post("/sessions")
            async def create_session(user_id: str = None, parameters: Dict[str, Any] = None):
                session_id = self.api.create_session(user_id, parameters)
                return {"session_id": session_id}
            
            @self.app.get("/sessions/{session_id}")
            async def get_session_status(session_id: str):
                status = self.api.get_session_status(session_id)
                if status is None:
                    return {"error": "Session not found"}, 404
                return status
            
            @self.app.post("/generate/{session_id}")
            async def generate_protein_streaming(
                session_id: str,
                motif: str,
                num_samples: int = 1,
                temperature: float = 0.8,
                guidance_scale: float = 1.0
            ):
                async def event_stream():
                    async for event in self.api.generate_protein_stream(
                        session_id, motif, num_samples, temperature, guidance_scale
                    ):
                        yield f"data: {event.to_json()}\n\n"
                
                return EventSourceResponse(event_stream())
            
            @self.app.post("/validate/{session_id}")
            async def validate_protein_streaming(session_id: str, protein_sequence: str):
                async def event_stream():
                    async for event in self.api.validate_protein_stream(session_id, protein_sequence):
                        yield f"data: {event.to_json()}\n\n"
                
                return EventSourceResponse(event_stream())


# Global instance
realtime_api = RealTimeProteinAPI()

# FastAPI app instance (if available)
if FASTAPI_AVAILABLE:
    fastapi_realtime_api = FastAPIRealTimeProteinAPI()
    app = fastapi_realtime_api.app
else:
    app = None