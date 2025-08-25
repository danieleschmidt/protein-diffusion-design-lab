"""
Real-Time Streaming API for Protein Diffusion Design Lab

This module provides WebSocket-based real-time streaming capabilities for
protein generation, structure prediction, and analysis workflows.
"""

import json
import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque
import logging

# Mock imports for environments without full dependencies
try:
    import websockets
    import websockets.server
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    websockets = None
    WEBSOCKETS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class StreamEventType(Enum):
    """Types of streaming events."""
    CONNECTION_ESTABLISHED = "connection_established"
    SUBSCRIPTION_CONFIRMED = "subscription_confirmed"
    PROTEIN_GENERATED = "protein_generated"
    STRUCTURE_PREDICTED = "structure_predicted"
    BINDING_CALCULATED = "binding_calculated"
    PROGRESS_UPDATE = "progress_update"
    ERROR_OCCURRED = "error_occurred"
    WORKFLOW_COMPLETED = "workflow_completed"
    SYSTEM_STATUS = "system_status"
    BATCH_STARTED = "batch_started"
    BATCH_COMPLETED = "batch_completed"
    REAL_TIME_METRICS = "real_time_metrics"


class SubscriptionType(Enum):
    """Types of data streams clients can subscribe to."""
    PROTEIN_GENERATION = "protein_generation"
    STRUCTURE_PREDICTION = "structure_prediction"
    BINDING_AFFINITY = "binding_affinity"
    WORKFLOW_PROGRESS = "workflow_progress"
    SYSTEM_METRICS = "system_metrics"
    ERROR_NOTIFICATIONS = "error_notifications"
    ALL_EVENTS = "all_events"


@dataclass
class StreamEvent:
    """Represents a streaming event."""
    event_type: StreamEventType
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    subscription_type: SubscriptionType = SubscriptionType.ALL_EVENTS
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(asdict(self), default=str)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'StreamEvent':
        """Create event from JSON string."""
        data = json.loads(json_str)
        return cls(
            event_type=StreamEventType(data['event_type']),
            timestamp=data['timestamp'],
            event_id=data['event_id'],
            subscription_type=SubscriptionType(data['subscription_type']),
            data=data['data'],
            metadata=data['metadata']
        )


@dataclass
class ClientConnection:
    """Represents a connected WebSocket client."""
    client_id: str
    websocket: Any  # websockets.WebSocketServerProtocol
    subscriptions: Set[SubscriptionType] = field(default_factory=set)
    connected_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    message_count: int = 0
    is_active: bool = True


@dataclass
class StreamingConfig:
    """Configuration for the streaming API server."""
    host: str = "localhost"
    port: int = 8765
    max_connections: int = 1000
    heartbeat_interval: float = 30.0
    message_queue_size: int = 1000
    enable_compression: bool = True
    enable_authentication: bool = False
    auth_token_header: str = "Authorization"
    log_level: str = "INFO"
    enable_metrics: bool = True
    metrics_interval: float = 60.0


class EventBuffer:
    """Buffer for streaming events with size limits and retention."""
    
    def __init__(self, max_size: int = 10000, retention_seconds: float = 3600.0):
        self.max_size = max_size
        self.retention_seconds = retention_seconds
        self.events: deque = deque(maxlen=max_size)
        self._lock = asyncio.Lock()
    
    async def add_event(self, event: StreamEvent):
        """Add an event to the buffer."""
        async with self._lock:
            self.events.append(event)
            await self._cleanup_old_events()
    
    async def get_events(
        self,
        subscription_type: Optional[SubscriptionType] = None,
        since_timestamp: Optional[float] = None,
        limit: Optional[int] = None
    ) -> List[StreamEvent]:
        """Get events from the buffer with filtering."""
        async with self._lock:
            filtered_events = []
            
            for event in self.events:
                # Filter by subscription type
                if subscription_type and event.subscription_type != subscription_type:
                    if subscription_type != SubscriptionType.ALL_EVENTS:
                        continue
                
                # Filter by timestamp
                if since_timestamp and event.timestamp < since_timestamp:
                    continue
                
                filtered_events.append(event)
                
                # Apply limit
                if limit and len(filtered_events) >= limit:
                    break
            
            return filtered_events
    
    async def _cleanup_old_events(self):
        """Remove events older than retention period."""
        cutoff_time = time.time() - self.retention_seconds
        
        # Remove old events from the front of the deque
        while self.events and self.events[0].timestamp < cutoff_time:
            self.events.popleft()


class ProteinGenerationStreamer:
    """Streams protein generation results in real-time."""
    
    def __init__(self, event_buffer: EventBuffer):
        self.event_buffer = event_buffer
        self.active_generations: Dict[str, Dict[str, Any]] = {}
    
    async def start_protein_generation(
        self,
        generation_id: str,
        num_proteins: int = 100,
        temperature: float = 0.8,
        motif: str = "HELIX_SHEET_HELIX"
    ):
        """Start streaming protein generation process."""
        # Store generation info
        self.active_generations[generation_id] = {
            'num_proteins': num_proteins,
            'temperature': temperature,
            'motif': motif,
            'generated_count': 0,
            'start_time': time.time()
        }
        
        # Send start event
        start_event = StreamEvent(
            event_type=StreamEventType.BATCH_STARTED,
            subscription_type=SubscriptionType.PROTEIN_GENERATION,
            data={
                'generation_id': generation_id,
                'num_proteins': num_proteins,
                'temperature': temperature,
                'motif': motif
            }
        )
        await self.event_buffer.add_event(start_event)
        
        # Generate proteins with streaming updates
        for i in range(num_proteins):
            # Simulate generation time
            await asyncio.sleep(0.01 + (i % 5) * 0.002)
            
            # Generate mock protein
            protein = {
                'id': f'{generation_id}_protein_{i}',
                'sequence': f'MKLLVLLVLL{motif}GGGHHHHHHH{i:03d}',
                'length': 50 + i,
                'confidence': 0.85 + (i % 10) * 0.01,
                'temperature': temperature,
                'generation_method': 'Real-Time Diffusion'
            }
            
            # Send protein generated event
            protein_event = StreamEvent(
                event_type=StreamEventType.PROTEIN_GENERATED,
                subscription_type=SubscriptionType.PROTEIN_GENERATION,
                data={
                    'generation_id': generation_id,
                    'protein': protein,
                    'progress': (i + 1) / num_proteins,
                    'generated_count': i + 1,
                    'remaining_count': num_proteins - i - 1
                }
            )
            await self.event_buffer.add_event(protein_event)
            
            # Update generation info
            self.active_generations[generation_id]['generated_count'] = i + 1
            
            # Send progress update every 10 proteins
            if (i + 1) % 10 == 0 or (i + 1) == num_proteins:
                progress_event = StreamEvent(
                    event_type=StreamEventType.PROGRESS_UPDATE,
                    subscription_type=SubscriptionType.WORKFLOW_PROGRESS,
                    data={
                        'generation_id': generation_id,
                        'progress': (i + 1) / num_proteins,
                        'generated_count': i + 1,
                        'total_count': num_proteins,
                        'elapsed_time': time.time() - self.active_generations[generation_id]['start_time']
                    }
                )
                await self.event_buffer.add_event(progress_event)
        
        # Send completion event
        completion_event = StreamEvent(
            event_type=StreamEventType.BATCH_COMPLETED,
            subscription_type=SubscriptionType.PROTEIN_GENERATION,
            data={
                'generation_id': generation_id,
                'total_generated': num_proteins,
                'total_time': time.time() - self.active_generations[generation_id]['start_time'],
                'avg_confidence': 0.90,  # Mock average
                'success': True
            }
        )
        await self.event_buffer.add_event(completion_event)
        
        # Clean up
        del self.active_generations[generation_id]


class StructurePredictionStreamer:
    """Streams structure prediction results in real-time."""
    
    def __init__(self, event_buffer: EventBuffer):
        self.event_buffer = event_buffer
        self.active_predictions: Dict[str, Dict[str, Any]] = {}
    
    async def predict_structures(
        self,
        prediction_id: str,
        sequences: List[str]
    ):
        """Stream structure prediction results."""
        # Store prediction info
        self.active_predictions[prediction_id] = {
            'sequences': sequences,
            'predicted_count': 0,
            'start_time': time.time()
        }
        
        # Send start event
        start_event = StreamEvent(
            event_type=StreamEventType.BATCH_STARTED,
            subscription_type=SubscriptionType.STRUCTURE_PREDICTION,
            data={
                'prediction_id': prediction_id,
                'sequence_count': len(sequences),
                'method': 'ESMFold-Stream'
            }
        )
        await self.event_buffer.add_event(start_event)
        
        # Process each sequence
        for i, sequence in enumerate(sequences):
            # Simulate prediction time
            await asyncio.sleep(0.05 + len(sequence) * 0.001)
            
            # Generate mock structure
            structure = {
                'sequence': sequence,
                'pdb_content': f'MOCK_PDB_STRUCTURE_FOR_{sequence[:10]}...',
                'confidence': 0.88 + (i % 5) * 0.02,
                'tm_score': 0.82 + (i % 3) * 0.03,
                'method': 'ESMFold-Stream',
                'prediction_time': 0.05 + len(sequence) * 0.001
            }
            
            # Send structure predicted event
            structure_event = StreamEvent(
                event_type=StreamEventType.STRUCTURE_PREDICTED,
                subscription_type=SubscriptionType.STRUCTURE_PREDICTION,
                data={
                    'prediction_id': prediction_id,
                    'structure': structure,
                    'progress': (i + 1) / len(sequences),
                    'predicted_count': i + 1
                }
            )
            await self.event_buffer.add_event(structure_event)
            
            # Update prediction info
            self.active_predictions[prediction_id]['predicted_count'] = i + 1
        
        # Send completion event
        completion_event = StreamEvent(
            event_type=StreamEventType.WORKFLOW_COMPLETED,
            subscription_type=SubscriptionType.STRUCTURE_PREDICTION,
            data={
                'prediction_id': prediction_id,
                'total_predicted': len(sequences),
                'total_time': time.time() - self.active_predictions[prediction_id]['start_time'],
                'avg_confidence': 0.92,
                'success': True
            }
        )
        await self.event_buffer.add_event(completion_event)
        
        # Clean up
        del self.active_predictions[prediction_id]


class SystemMetricsStreamer:
    """Streams real-time system metrics and status."""
    
    def __init__(self, event_buffer: EventBuffer):
        self.event_buffer = event_buffer
        self.is_streaming = False
        
    async def start_metrics_streaming(self, interval: float = 5.0):
        """Start streaming system metrics."""
        self.is_streaming = True
        
        while self.is_streaming:
            try:
                # Collect system metrics
                metrics = await self._collect_system_metrics()
                
                # Send metrics event
                metrics_event = StreamEvent(
                    event_type=StreamEventType.REAL_TIME_METRICS,
                    subscription_type=SubscriptionType.SYSTEM_METRICS,
                    data=metrics
                )
                await self.event_buffer.add_event(metrics_event)
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error streaming metrics: {e}")
                await asyncio.sleep(interval)
                
    def stop_metrics_streaming(self):
        """Stop streaming system metrics."""
        self.is_streaming = False
        
    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        current_time = time.time()
        
        # Mock system metrics
        cpu_usage = 30.0 + 20.0 * (0.5 + 0.3 * (current_time % 60) / 60)
        memory_usage = 60.0 + 15.0 * (0.5 + 0.4 * (current_time % 45) / 45)
        gpu_usage = 45.0 + 25.0 * (0.5 + 0.5 * (current_time % 30) / 30)
        
        if NUMPY_AVAILABLE and np:
            cpu_usage += np.random.normal(0, 2)
            memory_usage += np.random.normal(0, 1.5)
            gpu_usage += np.random.normal(0, 3)
        
        return {
            'timestamp': current_time,
            'cpu_usage_percent': max(0, min(100, cpu_usage)),
            'memory_usage_percent': max(0, min(100, memory_usage)),
            'gpu_usage_percent': max(0, min(100, gpu_usage)),
            'active_connections': len(getattr(self, '_server_connections', {})),
            'events_per_minute': 150,  # Mock value
            'system_status': 'healthy',
            'uptime_seconds': current_time - (current_time - 3600),  # Mock 1 hour uptime
        }


class RealTimeStreamingServer:
    """
    Real-Time Streaming API Server
    
    Provides WebSocket-based real-time streaming for protein design workflows.
    """
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.event_buffer = EventBuffer(max_size=config.message_queue_size)
        self.connections: Dict[str, ClientConnection] = {}
        self.server = None
        self.is_running = False
        
        # Initialize streamers
        self.protein_streamer = ProteinGenerationStreamer(self.event_buffer)
        self.structure_streamer = StructurePredictionStreamer(self.event_buffer)
        self.metrics_streamer = SystemMetricsStreamer(self.event_buffer)
        
        logger.info(f"Real-Time Streaming Server initialized on {config.host}:{config.port}")
        
    async def start_server(self):
        """Start the WebSocket server."""
        if not WEBSOCKETS_AVAILABLE:
            logger.error("WebSockets not available. Install with: pip install websockets")
            return
            
        self.is_running = True
        
        # Start background tasks
        metrics_task = asyncio.create_task(self._start_background_tasks())
        
        try:
            # Start WebSocket server
            self.server = await websockets.serve(
                self._handle_client,
                self.config.host,
                self.config.port,
                compression="deflate" if self.config.enable_compression else None,
                max_size=2**20,  # 1MB max message size
                max_queue=128
            )
            
            logger.info(f"Real-Time Streaming Server started on ws://{self.config.host}:{self.config.port}")
            
            # Keep server running
            await self.server.wait_closed()
            
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            self.is_running = False
            metrics_task.cancel()
            
    async def stop_server(self):
        """Stop the WebSocket server."""
        self.is_running = False
        
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            
        # Disconnect all clients
        for client in list(self.connections.values()):
            await client.websocket.close()
            
        self.connections.clear()
        self.metrics_streamer.stop_metrics_streaming()
        
        logger.info("Real-Time Streaming Server stopped")
        
    async def _handle_client(self, websocket, path):
        """Handle individual client connections."""
        client_id = str(uuid.uuid4())
        client = ClientConnection(
            client_id=client_id,
            websocket=websocket,
            metadata={'path': path, 'remote_address': websocket.remote_address}
        )
        
        self.connections[client_id] = client
        
        try:
            logger.info(f"Client {client_id} connected from {websocket.remote_address}")
            
            # Send connection confirmation
            welcome_event = StreamEvent(
                event_type=StreamEventType.CONNECTION_ESTABLISHED,
                data={'client_id': client_id, 'server_time': time.time()}
            )
            await websocket.send(welcome_event.to_json())
            
            # Handle client messages
            async for message in websocket:
                try:
                    await self._process_client_message(client, message)
                    client.last_activity = time.time()
                    client.message_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing message from {client_id}: {e}")
                    error_event = StreamEvent(
                        event_type=StreamEventType.ERROR_OCCURRED,
                        data={'error': str(e), 'client_id': client_id}
                    )
                    await websocket.send(error_event.to_json())
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} disconnected")
        except Exception as e:
            logger.error(f"Client {client_id} error: {e}")
        finally:
            # Clean up client connection
            if client_id in self.connections:
                del self.connections[client_id]
                
    async def _process_client_message(self, client: ClientConnection, message: str):
        """Process messages from clients."""
        try:
            data = json.loads(message)
            command = data.get('command')
            
            if command == 'subscribe':
                await self._handle_subscription(client, data)
            elif command == 'unsubscribe':
                await self._handle_unsubscription(client, data)
            elif command == 'start_generation':
                await self._handle_start_generation(client, data)
            elif command == 'predict_structures':
                await self._handle_predict_structures(client, data)
            elif command == 'get_history':
                await self._handle_get_history(client, data)
            elif command == 'ping':
                await self._handle_ping(client, data)
            else:
                raise ValueError(f"Unknown command: {command}")
                
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON message")
        except Exception as e:
            raise ValueError(f"Message processing error: {e}")
            
    async def _handle_subscription(self, client: ClientConnection, data: Dict[str, Any]):
        """Handle client subscription requests."""
        subscription_types = data.get('subscription_types', [])
        
        for sub_type in subscription_types:
            try:
                subscription = SubscriptionType(sub_type)
                client.subscriptions.add(subscription)
            except ValueError:
                logger.warning(f"Invalid subscription type: {sub_type}")
                
        # Send confirmation
        confirmation = StreamEvent(
            event_type=StreamEventType.SUBSCRIPTION_CONFIRMED,
            data={
                'client_id': client.client_id,
                'subscriptions': [sub.value for sub in client.subscriptions]
            }
        )
        await client.websocket.send(confirmation.to_json())
        
        logger.info(f"Client {client.client_id} subscribed to: {[s.value for s in client.subscriptions]}")
        
    async def _handle_unsubscription(self, client: ClientConnection, data: Dict[str, Any]):
        """Handle client unsubscription requests."""
        subscription_types = data.get('subscription_types', [])
        
        for sub_type in subscription_types:
            try:
                subscription = SubscriptionType(sub_type)
                client.subscriptions.discard(subscription)
            except ValueError:
                logger.warning(f"Invalid subscription type: {sub_type}")
                
        logger.info(f"Client {client.client_id} unsubscribed from: {subscription_types}")
        
    async def _handle_start_generation(self, client: ClientConnection, data: Dict[str, Any]):
        """Handle protein generation start requests."""
        generation_id = data.get('generation_id', str(uuid.uuid4()))
        num_proteins = data.get('num_proteins', 50)
        temperature = data.get('temperature', 0.8)
        motif = data.get('motif', 'HELIX_SHEET_HELIX')
        
        # Start protein generation in background
        asyncio.create_task(self.protein_streamer.start_protein_generation(
            generation_id, num_proteins, temperature, motif
        ))
        
        # Send acknowledgment
        ack_event = StreamEvent(
            event_type=StreamEventType.BATCH_STARTED,
            data={
                'generation_id': generation_id,
                'status': 'started',
                'client_id': client.client_id
            }
        )
        await client.websocket.send(ack_event.to_json())
        
    async def _handle_predict_structures(self, client: ClientConnection, data: Dict[str, Any]):
        """Handle structure prediction requests."""
        prediction_id = data.get('prediction_id', str(uuid.uuid4()))
        sequences = data.get('sequences', [])
        
        if not sequences:
            raise ValueError("No sequences provided for structure prediction")
            
        # Start structure prediction in background
        asyncio.create_task(self.structure_streamer.predict_structures(
            prediction_id, sequences
        ))
        
        # Send acknowledgment
        ack_event = StreamEvent(
            event_type=StreamEventType.BATCH_STARTED,
            data={
                'prediction_id': prediction_id,
                'status': 'started',
                'sequence_count': len(sequences),
                'client_id': client.client_id
            }
        )
        await client.websocket.send(ack_event.to_json())
        
    async def _handle_get_history(self, client: ClientConnection, data: Dict[str, Any]):
        """Handle event history requests."""
        subscription_type = data.get('subscription_type')
        since_timestamp = data.get('since_timestamp')
        limit = data.get('limit', 100)
        
        if subscription_type:
            try:
                sub_type = SubscriptionType(subscription_type)
            except ValueError:
                sub_type = None
        else:
            sub_type = None
            
        # Get historical events
        events = await self.event_buffer.get_events(
            subscription_type=sub_type,
            since_timestamp=since_timestamp,
            limit=limit
        )
        
        # Send historical events
        for event in events:
            if self._should_send_event(client, event):
                await client.websocket.send(event.to_json())
                
    async def _handle_ping(self, client: ClientConnection, data: Dict[str, Any]):
        """Handle ping/pong for connection health."""
        pong_event = StreamEvent(
            event_type=StreamEventType.SYSTEM_STATUS,
            data={
                'type': 'pong',
                'client_id': client.client_id,
                'server_time': time.time(),
                'client_uptime': time.time() - client.connected_at
            }
        )
        await client.websocket.send(pong_event.to_json())
        
    def _should_send_event(self, client: ClientConnection, event: StreamEvent) -> bool:
        """Determine if an event should be sent to a client."""
        if not client.is_active:
            return False
            
        # Check if client is subscribed to this event type
        if event.subscription_type in client.subscriptions:
            return True
        if SubscriptionType.ALL_EVENTS in client.subscriptions:
            return True
            
        return False
        
    async def _start_background_tasks(self):
        """Start background tasks for the server."""
        # Start metrics streaming
        if self.config.enable_metrics:
            asyncio.create_task(self.metrics_streamer.start_metrics_streaming(
                self.config.metrics_interval
            ))
            
        # Start connection cleanup task
        asyncio.create_task(self._connection_cleanup_task())
        
        # Start event broadcasting task
        asyncio.create_task(self._event_broadcasting_task())
        
    async def _connection_cleanup_task(self):
        """Clean up inactive connections."""
        while self.is_running:
            try:
                current_time = time.time()
                inactive_clients = []
                
                for client_id, client in self.connections.items():
                    # Check for inactive connections (no activity for 5 minutes)
                    if (current_time - client.last_activity) > 300:
                        inactive_clients.append(client_id)
                        
                # Remove inactive clients
                for client_id in inactive_clients:
                    if client_id in self.connections:
                        client = self.connections[client_id]
                        try:
                            await client.websocket.close()
                        except Exception:
                            pass
                        del self.connections[client_id]
                        logger.info(f"Cleaned up inactive client {client_id}")
                        
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Connection cleanup error: {e}")
                await asyncio.sleep(60)
                
    async def _event_broadcasting_task(self):
        """Broadcast events to subscribed clients."""
        last_broadcast_time = time.time()
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Get recent events
                events = await self.event_buffer.get_events(
                    since_timestamp=last_broadcast_time,
                    limit=1000
                )
                
                # Broadcast to subscribed clients
                for event in events:
                    clients_to_notify = [
                        client for client in self.connections.values()
                        if self._should_send_event(client, event)
                    ]
                    
                    # Send event to clients
                    for client in clients_to_notify:
                        try:
                            await client.websocket.send(event.to_json())
                        except Exception as e:
                            logger.error(f"Error sending event to client {client.client_id}: {e}")
                            client.is_active = False
                            
                last_broadcast_time = current_time
                await asyncio.sleep(0.1)  # Check for new events frequently
                
            except Exception as e:
                logger.error(f"Event broadcasting error: {e}")
                await asyncio.sleep(1)
                
    def get_server_status(self) -> Dict[str, Any]:
        """Get current server status."""
        return {
            'is_running': self.is_running,
            'total_connections': len(self.connections),
            'active_connections': len([c for c in self.connections.values() if c.is_active]),
            'total_events_buffered': len(self.event_buffer.events),
            'server_config': {
                'host': self.config.host,
                'port': self.config.port,
                'max_connections': self.config.max_connections,
                'compression_enabled': self.config.enable_compression
            },
            'connection_details': [
                {
                    'client_id': client.client_id,
                    'connected_at': client.connected_at,
                    'last_activity': client.last_activity,
                    'message_count': client.message_count,
                    'subscriptions': [sub.value for sub in client.subscriptions],
                    'is_active': client.is_active
                }
                for client in self.connections.values()
            ]
        }


# Convenience functions
async def start_streaming_server(
    host: str = "localhost",
    port: int = 8765,
    max_connections: int = 100
) -> RealTimeStreamingServer:
    """Start a real-time streaming server with default configuration."""
    config = StreamingConfig(
        host=host,
        port=port,
        max_connections=max_connections,
        enable_metrics=True,
        enable_compression=True
    )
    
    server = RealTimeStreamingServer(config)
    
    # Start server in background task
    server_task = asyncio.create_task(server.start_server())
    
    # Give server time to start
    await asyncio.sleep(1)
    
    return server


async def demo_streaming_server():
    """Demonstrate the streaming server capabilities."""
    print("Starting Real-Time Streaming Server demo...")
    
    if not WEBSOCKETS_AVAILABLE:
        print("WebSockets not available. Install with: pip install websockets")
        return
        
    config = StreamingConfig(
        host="localhost",
        port=8765,
        max_connections=10,
        enable_metrics=True,
        metrics_interval=5.0
    )
    
    server = RealTimeStreamingServer(config)
    
    try:
        # Start server
        server_task = asyncio.create_task(server.start_server())
        
        # Run for demo duration
        await asyncio.sleep(30)
        
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        await server.stop_server()
        
    print("Demo completed.")


if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_streaming_server())