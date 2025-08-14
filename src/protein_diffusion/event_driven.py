"""
Event-Driven Architecture for Protein Diffusion Platform

Advanced event-driven system with message queues, event sourcing,
and reactive processing capabilities.
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Callable, Union, AsyncGenerator
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor
import logging
from collections import defaultdict, deque

try:
    import aiokafka
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

try:
    import aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import pika
    RABBITMQ_AVAILABLE = True
except ImportError:
    RABBITMQ_AVAILABLE = False


logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of system events."""
    PROTEIN_GENERATION_REQUESTED = "protein.generation.requested"
    PROTEIN_GENERATION_STARTED = "protein.generation.started"
    PROTEIN_GENERATION_COMPLETED = "protein.generation.completed"
    PROTEIN_GENERATION_FAILED = "protein.generation.failed"
    
    PROTEIN_VALIDATION_REQUESTED = "protein.validation.requested"
    PROTEIN_VALIDATION_COMPLETED = "protein.validation.completed"
    PROTEIN_VALIDATION_FAILED = "protein.validation.failed"
    
    STRUCTURE_PREDICTION_REQUESTED = "structure.prediction.requested"
    STRUCTURE_PREDICTION_COMPLETED = "structure.prediction.completed"
    STRUCTURE_PREDICTION_FAILED = "structure.prediction.failed"
    
    BINDING_ANALYSIS_REQUESTED = "binding.analysis.requested"
    BINDING_ANALYSIS_COMPLETED = "binding.analysis.completed"
    
    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_COMPLETED = "workflow.completed"
    WORKFLOW_FAILED = "workflow.failed"
    
    USER_SESSION_CREATED = "user.session.created"
    USER_SESSION_ENDED = "user.session.ended"
    
    SYSTEM_HEALTH_CHECK = "system.health.check"
    SYSTEM_ALERT = "system.alert"
    SYSTEM_METRIC_UPDATED = "system.metric.updated"


@dataclass
class Event:
    """Base event structure."""
    event_id: str
    event_type: EventType
    aggregate_id: str
    aggregate_type: str
    payload: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = 1
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "aggregate_id": self.aggregate_id,
            "aggregate_type": self.aggregate_type,
            "payload": self.payload,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        return cls(
            event_id=data["event_id"],
            event_type=EventType(data["event_type"]),
            aggregate_id=data["aggregate_id"],
            aggregate_type=data["aggregate_type"],
            payload=data["payload"],
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            version=data.get("version", 1),
            correlation_id=data.get("correlation_id"),
            causation_id=data.get("causation_id")
        )


@dataclass
class Command:
    """Command structure for CQRS pattern."""
    command_id: str
    command_type: str
    aggregate_id: str
    payload: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expected_version: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "command_id": self.command_id,
            "command_type": self.command_type,
            "aggregate_id": self.aggregate_id,
            "payload": self.payload,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "expected_version": self.expected_version
        }


class EventStore:
    """Event store for event sourcing."""
    
    def __init__(self):
        self._events: Dict[str, List[Event]] = defaultdict(list)
        self._snapshots: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
    
    async def append_event(self, event: Event) -> bool:
        """Append event to the event store."""
        async with self._lock:
            try:
                self._events[event.aggregate_id].append(event)
                logger.debug(f"Appended event {event.event_id} to aggregate {event.aggregate_id}")
                
                # Notify subscribers
                await self._notify_subscribers(event)
                return True
                
            except Exception as e:
                logger.error(f"Failed to append event {event.event_id}: {e}")
                return False
    
    async def get_events(
        self, 
        aggregate_id: str, 
        from_version: int = 0,
        to_version: Optional[int] = None
    ) -> List[Event]:
        """Get events for an aggregate."""
        async with self._lock:
            events = self._events.get(aggregate_id, [])
            
            filtered_events = [
                event for event in events
                if event.version > from_version and (to_version is None or event.version <= to_version)
            ]
            
            return sorted(filtered_events, key=lambda e: e.version)
    
    async def get_events_by_type(
        self, 
        event_type: EventType,
        from_timestamp: Optional[datetime] = None,
        to_timestamp: Optional[datetime] = None
    ) -> List[Event]:
        """Get events by type and time range."""
        async with self._lock:
            all_events = []
            for events in self._events.values():
                all_events.extend(events)
            
            filtered_events = [
                event for event in all_events
                if event.event_type == event_type
            ]
            
            if from_timestamp:
                filtered_events = [e for e in filtered_events if e.timestamp >= from_timestamp]
            if to_timestamp:
                filtered_events = [e for e in filtered_events if e.timestamp <= to_timestamp]
            
            return sorted(filtered_events, key=lambda e: e.timestamp)
    
    async def save_snapshot(self, aggregate_id: str, version: int, state: Dict[str, Any]):
        """Save aggregate snapshot."""
        async with self._lock:
            self._snapshots[aggregate_id] = {
                "version": version,
                "state": state,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def get_snapshot(self, aggregate_id: str) -> Optional[Dict[str, Any]]:
        """Get latest snapshot for aggregate."""
        return self._snapshots.get(aggregate_id)
    
    def subscribe(self, event_type: EventType, handler: Callable[[Event], None]):
        """Subscribe to events of a specific type."""
        self._subscribers[event_type].append(handler)
    
    async def _notify_subscribers(self, event: Event):
        """Notify event subscribers."""
        subscribers = self._subscribers.get(event.event_type, [])
        for subscriber in subscribers:
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    await subscriber(event)
                else:
                    subscriber(event)
            except Exception as e:
                logger.error(f"Error in event subscriber: {e}")


class MessageQueue(ABC):
    """Abstract message queue interface."""
    
    @abstractmethod
    async def publish(self, topic: str, message: str, key: Optional[str] = None) -> bool:
        """Publish message to topic."""
        pass
    
    @abstractmethod
    async def subscribe(self, topic: str, handler: Callable[[str, str], None]) -> bool:
        """Subscribe to topic with message handler."""
        pass
    
    @abstractmethod
    async def close(self):
        """Close queue connections."""
        pass


class InMemoryMessageQueue(MessageQueue):
    """In-memory message queue for development/testing."""
    
    def __init__(self):
        self._topics: Dict[str, deque] = defaultdict(deque)
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
    
    async def start(self):
        """Start message processing."""
        self._running = True
        self._processor_task = asyncio.create_task(self._process_messages())
    
    async def stop(self):
        """Stop message processing."""
        self._running = False
        if self._processor_task:
            await self._processor_task
    
    async def publish(self, topic: str, message: str, key: Optional[str] = None) -> bool:
        """Publish message to topic."""
        async with self._lock:
            self._topics[topic].append((message, key))
            return True
    
    async def subscribe(self, topic: str, handler: Callable[[str, str], None]) -> bool:
        """Subscribe to topic with message handler."""
        self._subscribers[topic].append(handler)
        return True
    
    async def _process_messages(self):
        """Process queued messages."""
        while self._running:
            try:
                async with self._lock:
                    for topic, queue in self._topics.items():
                        if queue and topic in self._subscribers:
                            message, key = queue.popleft()
                            
                            for handler in self._subscribers[topic]:
                                try:
                                    if asyncio.iscoroutinefunction(handler):
                                        await handler(topic, message)
                                    else:
                                        handler(topic, message)
                                except Exception as e:
                                    logger.error(f"Error in message handler: {e}")
                
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Error processing messages: {e}")
    
    async def close(self):
        """Close queue connections."""
        await self.stop()


class KafkaMessageQueue(MessageQueue):
    """Kafka-based message queue."""
    
    def __init__(self, bootstrap_servers: str = "localhost:9092"):
        self.bootstrap_servers = bootstrap_servers
        self.producer: Optional[AIOKafkaProducer] = None
        self.consumers: Dict[str, AIOKafkaConsumer] = {}
        self.consumer_tasks: Dict[str, asyncio.Task] = {}
    
    async def start(self):
        """Start Kafka connections."""
        if KAFKA_AVAILABLE:
            self.producer = AIOKafkaProducer(bootstrap_servers=self.bootstrap_servers)
            await self.producer.start()
    
    async def publish(self, topic: str, message: str, key: Optional[str] = None) -> bool:
        """Publish message to Kafka topic."""
        if not self.producer:
            return False
        
        try:
            await self.producer.send_and_wait(
                topic,
                message.encode('utf-8'),
                key=key.encode('utf-8') if key else None
            )
            return True
        except Exception as e:
            logger.error(f"Failed to publish to Kafka: {e}")
            return False
    
    async def subscribe(self, topic: str, handler: Callable[[str, str], None]) -> bool:
        """Subscribe to Kafka topic."""
        if not KAFKA_AVAILABLE:
            return False
        
        try:
            consumer = AIOKafkaConsumer(
                topic,
                bootstrap_servers=self.bootstrap_servers,
                auto_offset_reset='earliest',
                group_id=f"protein-diffusion-{topic}"
            )
            await consumer.start()
            self.consumers[topic] = consumer
            
            # Start consumer task
            task = asyncio.create_task(self._consume_messages(topic, consumer, handler))
            self.consumer_tasks[topic] = task
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to Kafka topic {topic}: {e}")
            return False
    
    async def _consume_messages(
        self, 
        topic: str, 
        consumer: AIOKafkaConsumer, 
        handler: Callable[[str, str], None]
    ):
        """Consume messages from Kafka topic."""
        try:
            async for msg in consumer:
                try:
                    message = msg.value.decode('utf-8')
                    if asyncio.iscoroutinefunction(handler):
                        await handler(topic, message)
                    else:
                        handler(topic, message)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    
        except Exception as e:
            logger.error(f"Error in Kafka consumer: {e}")
    
    async def close(self):
        """Close Kafka connections."""
        for task in self.consumer_tasks.values():
            task.cancel()
        
        for consumer in self.consumers.values():
            await consumer.stop()
        
        if self.producer:
            await self.producer.stop()


class EventBus:
    """Event bus for handling domain events."""
    
    def __init__(self, message_queue: MessageQueue, event_store: EventStore):
        self.message_queue = message_queue
        self.event_store = event_store
        self.handlers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._started = False
    
    async def start(self):
        """Start the event bus."""
        if hasattr(self.message_queue, 'start'):
            await self.message_queue.start()
        
        # Subscribe to event topics
        for event_type in EventType:
            await self.message_queue.subscribe(
                f"events.{event_type.value}",
                self._handle_message
            )
        
        self._started = True
        logger.info("Event bus started")
    
    async def stop(self):
        """Stop the event bus."""
        if hasattr(self.message_queue, 'stop'):
            await self.message_queue.stop()
        await self.message_queue.close()
        self._started = False
        logger.info("Event bus stopped")
    
    async def publish_event(self, event: Event) -> bool:
        """Publish event to the event bus."""
        try:
            # Store event in event store
            await self.event_store.append_event(event)
            
            # Publish to message queue
            topic = f"events.{event.event_type.value}"
            return await self.message_queue.publish(topic, event.to_json(), event.aggregate_id)
            
        except Exception as e:
            logger.error(f"Failed to publish event {event.event_id}: {e}")
            return False
    
    def register_handler(self, event_type: EventType, handler: Callable[[Event], None]):
        """Register event handler."""
        self.handlers[event_type].append(handler)
        logger.info(f"Registered handler for {event_type.value}")
    
    async def _handle_message(self, topic: str, message: str):
        """Handle incoming message from queue."""
        try:
            event_data = json.loads(message)
            event = Event.from_dict(event_data)
            
            # Call registered handlers
            handlers = self.handlers.get(event.event_type, [])
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    logger.error(f"Error in event handler: {e}")
                    
        except Exception as e:
            logger.error(f"Error handling message: {e}")


class ProteinGenerationAggregate:
    """Protein generation aggregate for event sourcing."""
    
    def __init__(self, aggregate_id: str):
        self.aggregate_id = aggregate_id
        self.version = 0
        self.status = "pending"
        self.parameters = {}
        self.results = []
        self.created_at = None
        self.completed_at = None
        self.error_message = None
    
    def apply_event(self, event: Event):
        """Apply event to update aggregate state."""
        if event.event_type == EventType.PROTEIN_GENERATION_REQUESTED:
            self._handle_generation_requested(event)
        elif event.event_type == EventType.PROTEIN_GENERATION_STARTED:
            self._handle_generation_started(event)
        elif event.event_type == EventType.PROTEIN_GENERATION_COMPLETED:
            self._handle_generation_completed(event)
        elif event.event_type == EventType.PROTEIN_GENERATION_FAILED:
            self._handle_generation_failed(event)
        
        self.version = event.version
    
    def _handle_generation_requested(self, event: Event):
        self.status = "requested"
        self.parameters = event.payload
        self.created_at = event.timestamp
    
    def _handle_generation_started(self, event: Event):
        self.status = "generating"
    
    def _handle_generation_completed(self, event: Event):
        self.status = "completed"
        self.results = event.payload.get("results", [])
        self.completed_at = event.timestamp
    
    def _handle_generation_failed(self, event: Event):
        self.status = "failed"
        self.error_message = event.payload.get("error", "Unknown error")
        self.completed_at = event.timestamp
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "aggregate_id": self.aggregate_id,
            "version": self.version,
            "status": self.status,
            "parameters": self.parameters,
            "results": self.results,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message
        }


class ProteinGenerationService:
    """Event-driven protein generation service."""
    
    def __init__(self, event_bus: EventBus, event_store: EventStore):
        self.event_bus = event_bus
        self.event_store = event_store
        self._register_handlers()
    
    def _register_handlers(self):
        """Register event handlers."""
        self.event_bus.register_handler(
            EventType.PROTEIN_GENERATION_REQUESTED,
            self._handle_generation_requested
        )
    
    async def request_generation(
        self,
        motif: str,
        num_samples: int = 1,
        temperature: float = 0.8,
        user_id: Optional[str] = None
    ) -> str:
        """Request protein generation."""
        
        aggregate_id = str(uuid.uuid4())
        correlation_id = str(uuid.uuid4())
        
        event = Event(
            event_id=str(uuid.uuid4()),
            event_type=EventType.PROTEIN_GENERATION_REQUESTED,
            aggregate_id=aggregate_id,
            aggregate_type="protein_generation",
            payload={
                "motif": motif,
                "num_samples": num_samples,
                "temperature": temperature,
                "user_id": user_id
            },
            correlation_id=correlation_id,
            version=1
        )
        
        await self.event_bus.publish_event(event)
        return aggregate_id
    
    async def _handle_generation_requested(self, event: Event):
        """Handle generation request."""
        logger.info(f"Processing generation request {event.aggregate_id}")
        
        try:
            # Emit generation started event
            started_event = Event(
                event_id=str(uuid.uuid4()),
                event_type=EventType.PROTEIN_GENERATION_STARTED,
                aggregate_id=event.aggregate_id,
                aggregate_type="protein_generation",
                payload={"started_by": "generation_service"},
                correlation_id=event.correlation_id,
                causation_id=event.event_id,
                version=event.version + 1
            )
            await self.event_bus.publish_event(started_event)
            
            # Simulate generation process
            await asyncio.sleep(2.0)  # Simulate processing time
            
            # Generate results
            num_samples = event.payload.get("num_samples", 1)
            results = []
            
            for i in range(num_samples):
                sequence = f"MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG_{i}"
                results.append({
                    "sequence": sequence,
                    "confidence": 0.85 + (i * 0.02),
                    "generated_at": datetime.now(timezone.utc).isoformat()
                })
            
            # Emit generation completed event
            completed_event = Event(
                event_id=str(uuid.uuid4()),
                event_type=EventType.PROTEIN_GENERATION_COMPLETED,
                aggregate_id=event.aggregate_id,
                aggregate_type="protein_generation",
                payload={
                    "results": results,
                    "generation_time": 2.0
                },
                correlation_id=event.correlation_id,
                causation_id=event.event_id,
                version=event.version + 2
            )
            await self.event_bus.publish_event(completed_event)
            
        except Exception as e:
            # Emit generation failed event
            failed_event = Event(
                event_id=str(uuid.uuid4()),
                event_type=EventType.PROTEIN_GENERATION_FAILED,
                aggregate_id=event.aggregate_id,
                aggregate_type="protein_generation",
                payload={
                    "error": str(e),
                    "failed_at": datetime.now(timezone.utc).isoformat()
                },
                correlation_id=event.correlation_id,
                causation_id=event.event_id,
                version=event.version + 2
            )
            await self.event_bus.publish_event(failed_event)
    
    async def get_generation_status(self, aggregate_id: str) -> Optional[Dict[str, Any]]:
        """Get generation status by rebuilding aggregate from events."""
        events = await self.event_store.get_events(aggregate_id)
        
        if not events:
            return None
        
        aggregate = ProteinGenerationAggregate(aggregate_id)
        for event in events:
            aggregate.apply_event(event)
        
        return aggregate.to_dict()


class WorkflowOrchestrator:
    """Orchestrates complex workflows using events."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self._register_handlers()
    
    def _register_handlers(self):
        """Register workflow event handlers."""
        self.event_bus.register_handler(
            EventType.PROTEIN_GENERATION_COMPLETED,
            self._handle_generation_completed
        )
        self.event_bus.register_handler(
            EventType.PROTEIN_VALIDATION_COMPLETED,
            self._handle_validation_completed
        )
    
    async def start_workflow(
        self,
        workflow_type: str,
        parameters: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> str:
        """Start a new workflow."""
        
        workflow_id = str(uuid.uuid4())
        correlation_id = str(uuid.uuid4())
        
        workflow_data = {
            "workflow_id": workflow_id,
            "workflow_type": workflow_type,
            "parameters": parameters,
            "user_id": user_id,
            "status": "started",
            "steps_completed": [],
            "current_step": None,
            "correlation_id": correlation_id,
            "created_at": datetime.now(timezone.utc)
        }
        
        self.active_workflows[workflow_id] = workflow_data
        
        # Emit workflow started event
        event = Event(
            event_id=str(uuid.uuid4()),
            event_type=EventType.WORKFLOW_STARTED,
            aggregate_id=workflow_id,
            aggregate_type="workflow",
            payload={
                "workflow_type": workflow_type,
                "parameters": parameters,
                "user_id": user_id
            },
            correlation_id=correlation_id
        )
        
        await self.event_bus.publish_event(event)
        
        # Start first step based on workflow type
        if workflow_type == "protein_design_pipeline":
            await self._start_protein_generation_step(workflow_id, parameters)
        
        return workflow_id
    
    async def _start_protein_generation_step(self, workflow_id: str, parameters: Dict[str, Any]):
        """Start protein generation step in workflow."""
        workflow = self.active_workflows[workflow_id]
        workflow["current_step"] = "protein_generation"
        
        # Request protein generation
        generation_event = Event(
            event_id=str(uuid.uuid4()),
            event_type=EventType.PROTEIN_GENERATION_REQUESTED,
            aggregate_id=str(uuid.uuid4()),  # New aggregate for generation
            aggregate_type="protein_generation",
            payload=parameters,
            correlation_id=workflow["correlation_id"],
            metadata={"workflow_id": workflow_id}
        )
        
        await self.event_bus.publish_event(generation_event)
    
    async def _handle_generation_completed(self, event: Event):
        """Handle protein generation completion in workflow context."""
        workflow_id = event.metadata.get("workflow_id")
        if not workflow_id or workflow_id not in self.active_workflows:
            return
        
        workflow = self.active_workflows[workflow_id]
        workflow["steps_completed"].append("protein_generation")
        workflow["generation_results"] = event.payload["results"]
        
        # Move to next step - validation
        await self._start_validation_step(workflow_id, event.payload["results"])
    
    async def _start_validation_step(self, workflow_id: str, generation_results: List[Dict[str, Any]]):
        """Start validation step for generated proteins."""
        workflow = self.active_workflows[workflow_id]
        workflow["current_step"] = "protein_validation"
        
        # Validate each generated protein
        for result in generation_results:
            validation_event = Event(
                event_id=str(uuid.uuid4()),
                event_type=EventType.PROTEIN_VALIDATION_REQUESTED,
                aggregate_id=str(uuid.uuid4()),
                aggregate_type="protein_validation",
                payload={"sequence": result["sequence"]},
                correlation_id=workflow["correlation_id"],
                metadata={"workflow_id": workflow_id}
            )
            
            await self.event_bus.publish_event(validation_event)
    
    async def _handle_validation_completed(self, event: Event):
        """Handle protein validation completion in workflow context."""
        workflow_id = event.metadata.get("workflow_id")
        if not workflow_id or workflow_id not in self.active_workflows:
            return
        
        workflow = self.active_workflows[workflow_id]
        
        if "validation_results" not in workflow:
            workflow["validation_results"] = []
        
        workflow["validation_results"].append(event.payload)
        
        # Check if all validations are complete
        generation_count = len(workflow.get("generation_results", []))
        validation_count = len(workflow["validation_results"])
        
        if validation_count >= generation_count:
            # All validations complete - finish workflow
            await self._complete_workflow(workflow_id)
    
    async def _complete_workflow(self, workflow_id: str):
        """Complete workflow execution."""
        workflow = self.active_workflows[workflow_id]
        workflow["status"] = "completed"
        workflow["completed_at"] = datetime.now(timezone.utc)
        
        # Emit workflow completed event
        event = Event(
            event_id=str(uuid.uuid4()),
            event_type=EventType.WORKFLOW_COMPLETED,
            aggregate_id=workflow_id,
            aggregate_type="workflow",
            payload={
                "workflow_type": workflow["workflow_type"],
                "generation_results": workflow["generation_results"],
                "validation_results": workflow["validation_results"],
                "execution_time": (workflow["completed_at"] - workflow["created_at"]).total_seconds()
            },
            correlation_id=workflow["correlation_id"]
        )
        
        await self.event_bus.publish_event(event)
        
        # Clean up workflow from memory
        del self.active_workflows[workflow_id]
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get current workflow status."""
        workflow = self.active_workflows.get(workflow_id)
        if not workflow:
            return None
        
        return {
            "workflow_id": workflow_id,
            "status": workflow["status"],
            "current_step": workflow["current_step"],
            "steps_completed": workflow["steps_completed"],
            "created_at": workflow["created_at"].isoformat(),
            "progress": len(workflow["steps_completed"]) / 3  # Assuming 3 total steps
        }


# Global instances
event_store = EventStore()
message_queue = InMemoryMessageQueue()  # Can be switched to KafkaMessageQueue
event_bus = EventBus(message_queue, event_store)
generation_service = ProteinGenerationService(event_bus, event_store)
workflow_orchestrator = WorkflowOrchestrator(event_bus)


async def example_event_driven_usage():
    """Example usage of the event-driven architecture."""
    
    # Start the event bus
    await event_bus.start()
    
    try:
        # Request protein generation
        aggregate_id = await generation_service.request_generation(
            motif="HELIX_SHEET_HELIX",
            num_samples=2,
            temperature=0.8,
            user_id="user123"
        )
        
        print(f"Generation requested: {aggregate_id}")
        
        # Wait for generation to complete
        await asyncio.sleep(3)
        
        # Check generation status
        status = await generation_service.get_generation_status(aggregate_id)
        print(f"Generation status: {status['status']}")
        
        # Start a complex workflow
        workflow_id = await workflow_orchestrator.start_workflow(
            "protein_design_pipeline",
            {
                "motif": "ALPHA_HELIX",
                "num_samples": 1,
                "temperature": 0.7
            },
            user_id="user123"
        )
        
        print(f"Workflow started: {workflow_id}")
        
        # Monitor workflow progress
        for i in range(10):
            await asyncio.sleep(1)
            workflow_status = workflow_orchestrator.get_workflow_status(workflow_id)
            if workflow_status:
                print(f"Workflow progress: {workflow_status['progress']:.2f}")
                if workflow_status["status"] == "completed":
                    break
        
    finally:
        await event_bus.stop()


if __name__ == "__main__":
    asyncio.run(example_event_driven_usage())