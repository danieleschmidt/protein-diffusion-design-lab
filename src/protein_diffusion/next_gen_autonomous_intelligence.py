"""
Next-Generation Autonomous Intelligence System for Protein Design Lab.

This module implements self-evolving AI capabilities that autonomously improve
the protein design process through continuous learning, meta-optimization,
and intelligent decision-making.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timezone
from pathlib import Path
import logging
import hashlib
import pickle
from abc import ABC, abstractmethod

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
    np = MockNumpy()
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class AutonomousIntelligenceConfig:
    """Configuration for autonomous intelligence system."""
    # Learning parameters
    learning_rate: float = 0.001
    adaptation_threshold: float = 0.1
    confidence_threshold: float = 0.8
    exploration_rate: float = 0.2
    
    # Meta-optimization
    meta_learning_enabled: bool = True
    hyperparameter_tuning: bool = True
    architecture_search: bool = True
    
    # Self-improvement
    continuous_learning: bool = True
    performance_tracking: bool = True
    feedback_integration: bool = True
    
    # Autonomous decision making
    autonomous_optimization: bool = True
    intelligent_scaling: bool = True
    predictive_maintenance: bool = True
    
    # Safety and control
    human_oversight_required: bool = False
    safety_bounds_checking: bool = True
    rollback_on_degradation: bool = True
    
    # Storage and persistence
    knowledge_base_path: str = "./autonomous_intelligence"
    checkpoint_interval: int = 3600  # 1 hour
    max_memory_size: int = 1000000000  # 1GB


class KnowledgeNode:
    """Individual knowledge node in the autonomous intelligence system."""
    
    def __init__(self, node_id: str, knowledge_type: str, content: Dict[str, Any]):
        self.node_id = node_id
        self.knowledge_type = knowledge_type
        self.content = content
        self.created_at = datetime.now(timezone.utc)
        self.last_accessed = self.created_at
        self.usage_count = 0
        self.confidence_score = 1.0
        self.connections = set()
        
    def access(self):
        """Record access to this knowledge node."""
        self.last_accessed = datetime.now(timezone.utc)
        self.usage_count += 1
        
    def update_confidence(self, success: bool, feedback_score: float = None):
        """Update confidence based on usage outcomes."""
        if success:
            self.confidence_score = min(1.0, self.confidence_score * 1.01)
        else:
            self.confidence_score = max(0.1, self.confidence_score * 0.95)
        
        if feedback_score is not None:
            self.confidence_score = (self.confidence_score + feedback_score) / 2
            
    def connect(self, other_node_id: str):
        """Create connection to another knowledge node."""
        self.connections.add(other_node_id)
        
    def to_dict(self) -> Dict[str, Any]:
        """Serialize knowledge node to dictionary."""
        return {
            'node_id': self.node_id,
            'knowledge_type': self.knowledge_type,
            'content': self.content,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'usage_count': self.usage_count,
            'confidence_score': self.confidence_score,
            'connections': list(self.connections)
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeNode':
        """Create knowledge node from dictionary."""
        node = cls(data['node_id'], data['knowledge_type'], data['content'])
        node.created_at = datetime.fromisoformat(data['created_at'])
        node.last_accessed = datetime.fromisoformat(data['last_accessed'])
        node.usage_count = data['usage_count']
        node.confidence_score = data['confidence_score']
        node.connections = set(data['connections'])
        return node


class KnowledgeGraph:
    """Self-organizing knowledge graph for autonomous learning."""
    
    def __init__(self, config: AutonomousIntelligenceConfig):
        self.config = config
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.knowledge_types = {
            'protein_structure': 'Protein structural knowledge',
            'generation_strategy': 'Protein generation strategies', 
            'optimization_method': 'Optimization methodologies',
            'performance_pattern': 'Performance patterns and insights',
            'failure_case': 'Failure cases and solutions',
            'user_preference': 'User preferences and patterns',
            'research_insight': 'Research insights and discoveries'
        }
        
    def add_knowledge(self, knowledge_type: str, content: Dict[str, Any]) -> str:
        """Add new knowledge to the graph."""
        node_id = self._generate_node_id(knowledge_type, content)
        
        if node_id not in self.nodes:
            self.nodes[node_id] = KnowledgeNode(node_id, knowledge_type, content)
            logger.info(f"Added new knowledge node: {node_id}")
        else:
            # Update existing node
            self.nodes[node_id].content.update(content)
            logger.info(f"Updated knowledge node: {node_id}")
            
        return node_id
        
    def get_knowledge(self, node_id: str) -> Optional[KnowledgeNode]:
        """Retrieve knowledge node by ID."""
        if node_id in self.nodes:
            self.nodes[node_id].access()
            return self.nodes[node_id]
        return None
        
    def search_knowledge(self, knowledge_type: str, query: Dict[str, Any]) -> List[KnowledgeNode]:
        """Search for relevant knowledge nodes."""
        relevant_nodes = []
        
        for node in self.nodes.values():
            if node.knowledge_type == knowledge_type:
                similarity = self._calculate_similarity(query, node.content)
                if similarity > 0.5:  # Threshold for relevance
                    relevant_nodes.append((node, similarity))
                    
        # Sort by similarity and confidence
        relevant_nodes.sort(key=lambda x: x[1] * x[0].confidence_score, reverse=True)
        return [node for node, _ in relevant_nodes[:10]]  # Return top 10
        
    def create_connections(self, node_id: str, related_nodes: List[str]):
        """Create connections between knowledge nodes."""
        if node_id in self.nodes:
            for related_id in related_nodes:
                if related_id in self.nodes:
                    self.nodes[node_id].connect(related_id)
                    self.nodes[related_id].connect(node_id)
                    
    def prune_knowledge(self, min_confidence: float = 0.3, max_age_days: int = 30):
        """Prune low-quality or outdated knowledge."""
        current_time = datetime.now(timezone.utc)
        nodes_to_remove = []
        
        for node_id, node in self.nodes.items():
            age_days = (current_time - node.created_at).days
            
            if (node.confidence_score < min_confidence and 
                node.usage_count < 5 and 
                age_days > max_age_days):
                nodes_to_remove.append(node_id)
                
        for node_id in nodes_to_remove:
            del self.nodes[node_id]
            logger.info(f"Pruned knowledge node: {node_id}")
            
    def _generate_node_id(self, knowledge_type: str, content: Dict[str, Any]) -> str:
        """Generate unique ID for knowledge node."""
        content_str = json.dumps(content, sort_keys=True)
        hash_object = hashlib.md5(f"{knowledge_type}:{content_str}".encode())
        return hash_object.hexdigest()[:16]
        
    def _calculate_similarity(self, query: Dict[str, Any], content: Dict[str, Any]) -> float:
        """Calculate similarity between query and content."""
        # Simple similarity based on common keys and values
        common_keys = set(query.keys()) & set(content.keys())
        if not common_keys:
            return 0.0
            
        matches = sum(1 for key in common_keys if query[key] == content[key])
        return matches / len(common_keys)
        
    def export_knowledge(self) -> Dict[str, Any]:
        """Export knowledge graph for persistence."""
        return {
            'nodes': {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            'metadata': {
                'total_nodes': len(self.nodes),
                'knowledge_types': list(self.knowledge_types.keys()),
                'exported_at': datetime.now(timezone.utc).isoformat()
            }
        }
        
    def import_knowledge(self, data: Dict[str, Any]):
        """Import knowledge graph from persistence."""
        self.nodes = {
            node_id: KnowledgeNode.from_dict(node_data)
            for node_id, node_data in data['nodes'].items()
        }
        logger.info(f"Imported {len(self.nodes)} knowledge nodes")


class MetaLearningOptimizer:
    """Meta-learning system for automatic hyperparameter optimization."""
    
    def __init__(self, config: AutonomousIntelligenceConfig):
        self.config = config
        self.optimization_history: List[Dict[str, Any]] = []
        self.current_best_params: Dict[str, Any] = {}
        self.parameter_bounds: Dict[str, Tuple[float, float]] = {
            'temperature': (0.1, 2.0),
            'guidance_scale': (1.0, 10.0),
            'num_samples': (1, 1000),
            'max_length': (50, 1000),
            'learning_rate': (1e-6, 1e-2),
            'dropout': (0.0, 0.5)
        }
        
    def suggest_parameters(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest optimal parameters based on task context and history."""
        if not self.optimization_history:
            return self._get_default_parameters()
            
        # Find similar tasks in history
        similar_tasks = self._find_similar_tasks(task_context)
        
        if similar_tasks:
            # Use best performing parameters from similar tasks
            best_task = max(similar_tasks, key=lambda x: x['performance_score'])
            suggested_params = best_task['parameters'].copy()
            
            # Add exploration noise
            if self.config.exploration_rate > 0:
                suggested_params = self._add_exploration_noise(suggested_params)
                
            return suggested_params
        else:
            # Use Bayesian optimization or random search
            return self._bayesian_optimize(task_context)
            
    def record_performance(self, parameters: Dict[str, Any], 
                          performance_score: float, 
                          task_context: Dict[str, Any],
                          additional_metrics: Dict[str, float] = None):
        """Record performance of parameter configuration."""
        record = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'parameters': parameters,
            'performance_score': performance_score,
            'task_context': task_context,
            'additional_metrics': additional_metrics or {}
        }
        
        self.optimization_history.append(record)
        
        # Update current best parameters
        if (not self.current_best_params or 
            performance_score > self._get_best_score()):
            self.current_best_params = parameters.copy()
            
        # Prune history to prevent unbounded growth
        if len(self.optimization_history) > 10000:
            self.optimization_history = self.optimization_history[-5000:]
            
    def _find_similar_tasks(self, task_context: Dict[str, Any], threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find similar tasks in optimization history."""
        similar_tasks = []
        
        for record in self.optimization_history:
            similarity = self._calculate_task_similarity(task_context, record['task_context'])
            if similarity >= threshold:
                similar_tasks.append(record)
                
        return similar_tasks
        
    def _calculate_task_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """Calculate similarity between task contexts."""
        common_keys = set(context1.keys()) & set(context2.keys())
        if not common_keys:
            return 0.0
            
        matches = 0
        for key in common_keys:
            if context1[key] == context2[key]:
                matches += 1
            elif isinstance(context1[key], (int, float)) and isinstance(context2[key], (int, float)):
                # Numerical similarity
                diff = abs(context1[key] - context2[key])
                max_val = max(abs(context1[key]), abs(context2[key]), 1)
                similarity = 1 - (diff / max_val)
                matches += max(0, similarity)
                
        return matches / len(common_keys)
        
    def _get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for initial suggestions."""
        return {
            'temperature': 1.0,
            'guidance_scale': 1.0,
            'num_samples': 10,
            'max_length': 256,
            'learning_rate': 1e-4,
            'dropout': 0.1
        }
        
    def _add_exploration_noise(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Add exploration noise to parameters."""
        noisy_params = parameters.copy()
        
        for param, value in parameters.items():
            if param in self.parameter_bounds:
                min_val, max_val = self.parameter_bounds[param]
                noise_scale = (max_val - min_val) * self.config.exploration_rate * 0.1
                noise = np.random.normal(0, noise_scale)
                noisy_params[param] = max(min_val, min(max_val, value + noise))
                
        return noisy_params
        
    def _bayesian_optimize(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """Simple Bayesian optimization for parameter suggestion."""
        # For simplicity, use random search with bias toward good regions
        suggested_params = {}
        
        for param, (min_val, max_val) in self.parameter_bounds.items():
            if self.current_best_params and param in self.current_best_params:
                # Sample around current best with some exploration
                center = self.current_best_params[param]
                std = (max_val - min_val) * 0.1
                value = np.random.normal(center, std)
                suggested_params[param] = max(min_val, min(max_val, value))
            else:
                # Random sampling
                suggested_params[param] = np.random.uniform(min_val, max_val)
                
        return suggested_params
        
    def _get_best_score(self) -> float:
        """Get the best performance score from history."""
        if not self.optimization_history:
            return 0.0
        return max(record['performance_score'] for record in self.optimization_history)


class AutonomousDecisionEngine:
    """Autonomous decision engine for intelligent system management."""
    
    def __init__(self, config: AutonomousIntelligenceConfig):
        self.config = config
        self.decision_history: List[Dict[str, Any]] = []
        self.active_decisions: Dict[str, Any] = {}
        
    async def make_decision(self, decision_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Make autonomous decision based on context and learned patterns."""
        decision_id = f"{decision_type}_{int(time.time())}"
        
        # Analyze context and historical patterns
        analysis = await self._analyze_context(decision_type, context)
        
        # Generate decision options
        options = self._generate_options(decision_type, context, analysis)
        
        # Select best option
        selected_option = self._select_best_option(options, analysis)
        
        # Create decision record
        decision = {
            'decision_id': decision_id,
            'decision_type': decision_type,
            'context': context,
            'analysis': analysis,
            'options': options,
            'selected_option': selected_option,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'confidence': analysis.get('confidence', 0.5),
            'expected_outcome': selected_option.get('expected_outcome', 'unknown')
        }
        
        self.active_decisions[decision_id] = decision
        self.decision_history.append(decision)
        
        logger.info(f"Made autonomous decision: {decision_type} -> {selected_option['action']}")
        
        return decision
        
    async def _analyze_context(self, decision_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze context for decision making."""
        analysis = {
            'urgency': self._assess_urgency(context),
            'complexity': self._assess_complexity(context),
            'risk_level': self._assess_risk(context),
            'resource_availability': self._check_resources(context),
            'historical_success_rate': self._get_historical_success_rate(decision_type),
            'confidence': 0.7  # Default confidence
        }
        
        # Adjust confidence based on analysis
        if analysis['historical_success_rate'] > 0.8:
            analysis['confidence'] *= 1.2
        if analysis['risk_level'] > 0.7:
            analysis['confidence'] *= 0.8
            
        analysis['confidence'] = min(1.0, analysis['confidence'])
        
        return analysis
        
    def _generate_options(self, decision_type: str, context: Dict[str, Any], analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate decision options based on context and analysis."""
        options = []
        
        if decision_type == 'scaling':
            options = [
                {
                    'action': 'scale_up',
                    'description': 'Increase computational resources',
                    'expected_outcome': 'improved_performance',
                    'cost': analysis.get('resource_availability', 0.5),
                    'risk': 0.2,
                    'benefit': 0.8
                },
                {
                    'action': 'maintain',
                    'description': 'Keep current resource allocation', 
                    'expected_outcome': 'stable_performance',
                    'cost': 0.1,
                    'risk': 0.1,
                    'benefit': 0.4
                },
                {
                    'action': 'optimize',
                    'description': 'Optimize current resource usage',
                    'expected_outcome': 'efficient_performance',
                    'cost': 0.3,
                    'risk': 0.3,
                    'benefit': 0.7
                }
            ]
        elif decision_type == 'model_update':
            options = [
                {
                    'action': 'update_immediately',
                    'description': 'Apply model update immediately',
                    'expected_outcome': 'improved_accuracy',
                    'cost': 0.4,
                    'risk': 0.5,
                    'benefit': 0.9
                },
                {
                    'action': 'staged_rollout',
                    'description': 'Gradual model update rollout',
                    'expected_outcome': 'safe_improvement',
                    'cost': 0.6,
                    'risk': 0.2,
                    'benefit': 0.8
                },
                {
                    'action': 'delay_update',
                    'description': 'Delay update for more testing',
                    'expected_outcome': 'maintained_stability',
                    'cost': 0.2,
                    'risk': 0.1,
                    'benefit': 0.3
                }
            ]
        else:
            # Default options for unknown decision types
            options = [
                {
                    'action': 'conservative',
                    'description': 'Take conservative approach',
                    'expected_outcome': 'stable',
                    'cost': 0.2,
                    'risk': 0.1,
                    'benefit': 0.4
                },
                {
                    'action': 'aggressive',
                    'description': 'Take aggressive approach',
                    'expected_outcome': 'high_reward',
                    'cost': 0.7,
                    'risk': 0.6,
                    'benefit': 0.9
                }
            ]
            
        return options
        
    def _select_best_option(self, options: List[Dict[str, Any]], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Select the best option based on analysis and preferences."""
        scored_options = []
        
        for option in options:
            # Calculate utility score
            benefit_weight = 0.4
            cost_weight = 0.3
            risk_weight = 0.3
            
            utility_score = (
                option['benefit'] * benefit_weight -
                option['cost'] * cost_weight -
                option['risk'] * risk_weight * analysis['risk_level']
            )
            
            scored_options.append((option, utility_score))
            
        # Sort by utility score and return best option
        scored_options.sort(key=lambda x: x[1], reverse=True)
        return scored_options[0][0]
        
    def _assess_urgency(self, context: Dict[str, Any]) -> float:
        """Assess urgency of decision."""
        urgency_indicators = [
            'error_rate' in context,
            'performance_degradation' in context,
            'resource_exhaustion' in context,
            context.get('timeout_approaching', False)
        ]
        return sum(urgency_indicators) / len(urgency_indicators)
        
    def _assess_complexity(self, context: Dict[str, Any]) -> float:
        """Assess complexity of decision."""
        complexity_factors = [
            len(context.get('dependencies', [])),
            context.get('num_affected_components', 1),
            len(context.get('constraints', []))
        ]
        return min(1.0, sum(complexity_factors) / 10.0)
        
    def _assess_risk(self, context: Dict[str, Any]) -> float:
        """Assess risk level of decision."""
        risk_factors = [
            context.get('production_impact', False),
            context.get('irreversible_action', False),
            context.get('data_loss_risk', False),
            context.get('security_implications', False)
        ]
        return sum(risk_factors) / len(risk_factors)
        
    def _check_resources(self, context: Dict[str, Any]) -> float:
        """Check resource availability."""
        return context.get('available_resources', 0.7)
        
    def _get_historical_success_rate(self, decision_type: str) -> float:
        """Get historical success rate for decision type."""
        relevant_decisions = [d for d in self.decision_history if d['decision_type'] == decision_type]
        if not relevant_decisions:
            return 0.5  # Default neutral success rate
            
        # Simplified success calculation - in reality would track outcomes
        return 0.75  # Placeholder success rate


class AutonomousIntelligenceSystem:
    """Main autonomous intelligence system coordinating all components."""
    
    def __init__(self, config: AutonomousIntelligenceConfig = None):
        self.config = config or AutonomousIntelligenceConfig()
        self.knowledge_graph = KnowledgeGraph(self.config)
        self.meta_optimizer = MetaLearningOptimizer(self.config)
        self.decision_engine = AutonomousDecisionEngine(self.config)
        
        self.performance_metrics: Dict[str, List[float]] = {}
        self.adaptation_state: Dict[str, Any] = {}
        self.last_checkpoint = time.time()
        
        # Initialize storage
        self.storage_path = Path(self.config.knowledge_base_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing knowledge if available
        self._load_persistent_state()
        
    async def learn_from_experience(self, experience: Dict[str, Any]):
        """Learn from experience and update knowledge base."""
        experience_type = experience.get('type', 'general')
        
        # Extract knowledge from experience
        if experience_type == 'generation_task':
            await self._learn_from_generation(experience)
        elif experience_type == 'ranking_task':
            await self._learn_from_ranking(experience)
        elif experience_type == 'optimization_task':
            await self._learn_from_optimization(experience)
        else:
            await self._learn_from_general_experience(experience)
            
        # Update meta-learning optimizer
        if 'parameters' in experience and 'performance_score' in experience:
            self.meta_optimizer.record_performance(
                experience['parameters'],
                experience['performance_score'],
                experience.get('task_context', {}),
                experience.get('additional_metrics', {})
            )
            
        # Periodic checkpoint
        if time.time() - self.last_checkpoint > self.config.checkpoint_interval:
            await self._checkpoint()
            
    async def optimize_parameters(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize parameters for a given task using meta-learning."""
        suggested_params = self.meta_optimizer.suggest_parameters(task_context)
        
        # Add knowledge-based adjustments
        relevant_knowledge = self.knowledge_graph.search_knowledge('optimization_method', task_context)
        
        for knowledge_node in relevant_knowledge[:3]:  # Top 3 relevant nodes
            if 'parameter_adjustments' in knowledge_node.content:
                adjustments = knowledge_node.content['parameter_adjustments']
                for param, adjustment in adjustments.items():
                    if param in suggested_params:
                        # Weighted adjustment based on confidence
                        weight = knowledge_node.confidence_score
                        suggested_params[param] = (
                            suggested_params[param] * (1 - weight) + 
                            adjustment * weight
                        )
                        
        logger.info(f"Optimized parameters for task: {task_context.get('task_type', 'unknown')}")
        return suggested_params
        
    async def make_autonomous_decision(self, decision_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Make autonomous decision with knowledge graph consultation."""
        # Consult knowledge graph for relevant insights
        relevant_knowledge = self.knowledge_graph.search_knowledge('failure_case', context)
        
        # Add knowledge insights to context
        enhanced_context = context.copy()
        enhanced_context['knowledge_insights'] = [
            {
                'content': node.content,
                'confidence': node.confidence_score,
                'usage_count': node.usage_count
            }
            for node in relevant_knowledge[:5]
        ]
        
        # Make decision using enhanced context
        decision = await self.decision_engine.make_decision(decision_type, enhanced_context)
        
        # Record decision as knowledge
        decision_knowledge = {
            'decision_type': decision_type,
            'context_features': self._extract_context_features(context),
            'selected_action': decision['selected_option']['action'],
            'confidence': decision['confidence']
        }
        
        self.knowledge_graph.add_knowledge('decision_pattern', decision_knowledge)
        
        return decision
        
    async def adapt_system(self, performance_feedback: Dict[str, float]):
        """Adapt system based on performance feedback."""
        # Update performance metrics
        for metric, value in performance_feedback.items():
            if metric not in self.performance_metrics:
                self.performance_metrics[metric] = []
            self.performance_metrics[metric].append(value)
            
            # Keep only recent metrics
            if len(self.performance_metrics[metric]) > 1000:
                self.performance_metrics[metric] = self.performance_metrics[metric][-500:]
                
        # Detect performance degradation
        degradation_detected = await self._detect_performance_degradation()
        
        if degradation_detected:
            logger.warning("Performance degradation detected, initiating adaptation")
            await self._initiate_adaptive_response(performance_feedback)
            
        # Update adaptation state
        self.adaptation_state['last_adaptation'] = datetime.now(timezone.utc).isoformat()
        self.adaptation_state['performance_trend'] = self._calculate_performance_trend()
        
    async def _learn_from_generation(self, experience: Dict[str, Any]):
        """Learn from protein generation experience."""
        generation_knowledge = {
            'motif': experience.get('motif'),
            'parameters': experience.get('parameters', {}),
            'success_rate': experience.get('success_rate', 0.5),
            'quality_scores': experience.get('quality_scores', []),
            'generation_time': experience.get('generation_time', 0),
            'sequence_diversity': experience.get('sequence_diversity', 0.5)
        }
        
        self.knowledge_graph.add_knowledge('generation_strategy', generation_knowledge)
        
    async def _learn_from_ranking(self, experience: Dict[str, Any]):
        """Learn from ranking experience."""
        ranking_knowledge = {
            'ranking_criteria': experience.get('ranking_criteria', {}),
            'top_sequences': experience.get('top_sequences', []),
            'ranking_accuracy': experience.get('ranking_accuracy', 0.5),
            'user_feedback': experience.get('user_feedback', {})
        }
        
        self.knowledge_graph.add_knowledge('ranking_strategy', ranking_knowledge)
        
    async def _learn_from_optimization(self, experience: Dict[str, Any]):
        """Learn from optimization experience."""
        optimization_knowledge = {
            'optimization_method': experience.get('method'),
            'starting_performance': experience.get('starting_performance'),
            'final_performance': experience.get('final_performance'),
            'optimization_steps': experience.get('steps', 0),
            'convergence_rate': experience.get('convergence_rate', 0.5)
        }
        
        self.knowledge_graph.add_knowledge('optimization_method', optimization_knowledge)
        
    async def _learn_from_general_experience(self, experience: Dict[str, Any]):
        """Learn from general experience."""
        general_knowledge = {
            'context': experience.get('context', {}),
            'outcome': experience.get('outcome'),
            'lessons_learned': experience.get('lessons_learned', []),
            'success': experience.get('success', False)
        }
        
        self.knowledge_graph.add_knowledge('general_insight', general_knowledge)
        
    def _extract_context_features(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant features from context."""
        features = {}
        
        # Numerical features
        for key, value in context.items():
            if isinstance(value, (int, float)):
                features[f"numeric_{key}"] = value
            elif isinstance(value, bool):
                features[f"boolean_{key}"] = value
            elif isinstance(value, str):
                features[f"categorical_{key}"] = value
                
        return features
        
    async def _detect_performance_degradation(self) -> bool:
        """Detect if system performance is degrading."""
        if not self.performance_metrics:
            return False
            
        for metric, values in self.performance_metrics.items():
            if len(values) < 10:  # Need enough data points
                continue
                
            # Compare recent performance to historical average
            recent_avg = np.mean(values[-5:])
            historical_avg = np.mean(values[:-5])
            
            if recent_avg < historical_avg * 0.9:  # 10% degradation threshold
                return True
                
        return False
        
    async def _initiate_adaptive_response(self, feedback: Dict[str, float]):
        """Initiate adaptive response to performance issues."""
        # Analyze the type of degradation
        degradation_type = self._analyze_degradation_type(feedback)
        
        # Make autonomous decision on corrective action
        correction_decision = await self.make_autonomous_decision(
            'performance_correction',
            {
                'degradation_type': degradation_type,
                'current_metrics': feedback,
                'system_state': self.adaptation_state
            }
        )
        
        logger.info(f"Initiated adaptive response: {correction_decision['selected_option']['action']}")
        
    def _analyze_degradation_type(self, feedback: Dict[str, float]) -> str:
        """Analyze the type of performance degradation."""
        # Simple heuristic classification
        if feedback.get('accuracy', 1.0) < 0.8:
            return 'accuracy_degradation'
        elif feedback.get('speed', 1.0) < 0.8:
            return 'speed_degradation'
        elif feedback.get('memory_usage', 0.0) > 0.9:
            return 'memory_degradation'
        else:
            return 'general_degradation'
            
    def _calculate_performance_trend(self) -> str:
        """Calculate overall performance trend."""
        if not self.performance_metrics:
            return 'stable'
            
        trends = []
        for metric, values in self.performance_metrics.items():
            if len(values) >= 5:
                recent = np.mean(values[-3:])
                older = np.mean(values[-6:-3]) if len(values) >= 6 else np.mean(values[:-3])
                
                if recent > older * 1.05:
                    trends.append('improving')
                elif recent < older * 0.95:
                    trends.append('degrading')
                else:
                    trends.append('stable')
                    
        # Majority vote
        if not trends:
            return 'stable'
            
        trend_counts = {'improving': 0, 'degrading': 0, 'stable': 0}
        for trend in trends:
            trend_counts[trend] += 1
            
        return max(trend_counts.items(), key=lambda x: x[1])[0]
        
    async def _checkpoint(self):
        """Save system state to persistent storage."""
        try:
            # Save knowledge graph
            knowledge_data = self.knowledge_graph.export_knowledge()
            knowledge_file = self.storage_path / f"knowledge_{int(time.time())}.json"
            
            with open(knowledge_file, 'w') as f:
                json.dump(knowledge_data, f, indent=2)
                
            # Save meta-optimizer state
            optimizer_data = {
                'optimization_history': self.meta_optimizer.optimization_history[-1000:],  # Keep recent
                'current_best_params': self.meta_optimizer.current_best_params
            }
            
            optimizer_file = self.storage_path / f"optimizer_{int(time.time())}.json"
            with open(optimizer_file, 'w') as f:
                json.dump(optimizer_data, f, indent=2)
                
            # Save system state
            system_data = {
                'performance_metrics': {k: v[-100:] for k, v in self.performance_metrics.items()},
                'adaptation_state': self.adaptation_state,
                'checkpoint_time': datetime.now(timezone.utc).isoformat()
            }
            
            system_file = self.storage_path / f"system_{int(time.time())}.json"
            with open(system_file, 'w') as f:
                json.dump(system_data, f, indent=2)
                
            self.last_checkpoint = time.time()
            logger.info(f"System checkpoint saved at {datetime.now()}")
            
            # Cleanup old checkpoints (keep only latest 10)
            self._cleanup_old_checkpoints()
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            
    def _cleanup_old_checkpoints(self):
        """Clean up old checkpoint files."""
        try:
            for pattern in ['knowledge_*.json', 'optimizer_*.json', 'system_*.json']:
                files = list(self.storage_path.glob(pattern))
                files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                
                # Keep only latest 10 files
                for old_file in files[10:]:
                    old_file.unlink()
                    
        except Exception as e:
            logger.warning(f"Failed to cleanup old checkpoints: {e}")
            
    def _load_persistent_state(self):
        """Load system state from persistent storage."""
        try:
            # Load latest knowledge graph
            knowledge_files = list(self.storage_path.glob('knowledge_*.json'))
            if knowledge_files:
                latest_knowledge = max(knowledge_files, key=lambda x: x.stat().st_mtime)
                with open(latest_knowledge, 'r') as f:
                    knowledge_data = json.load(f)
                self.knowledge_graph.import_knowledge(knowledge_data)
                
            # Load latest optimizer state
            optimizer_files = list(self.storage_path.glob('optimizer_*.json'))
            if optimizer_files:
                latest_optimizer = max(optimizer_files, key=lambda x: x.stat().st_mtime)
                with open(latest_optimizer, 'r') as f:
                    optimizer_data = json.load(f)
                self.meta_optimizer.optimization_history = optimizer_data['optimization_history']
                self.meta_optimizer.current_best_params = optimizer_data['current_best_params']
                
            # Load latest system state
            system_files = list(self.storage_path.glob('system_*.json'))
            if system_files:
                latest_system = max(system_files, key=lambda x: x.stat().st_mtime)
                with open(latest_system, 'r') as f:
                    system_data = json.load(f)
                self.performance_metrics = system_data['performance_metrics']
                self.adaptation_state = system_data['adaptation_state']
                
            logger.info("Loaded persistent state successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load persistent state: {e}")
            
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'knowledge_base': {
                'total_nodes': len(self.knowledge_graph.nodes),
                'knowledge_types': list(self.knowledge_graph.knowledge_types.keys()),
                'avg_confidence': np.mean([node.confidence_score for node in self.knowledge_graph.nodes.values()]) if self.knowledge_graph.nodes else 0.0
            },
            'meta_learning': {
                'optimization_records': len(self.meta_optimizer.optimization_history),
                'best_performance': self.meta_optimizer._get_best_score(),
                'current_best_params': self.meta_optimizer.current_best_params
            },
            'decision_engine': {
                'active_decisions': len(self.decision_engine.active_decisions),
                'total_decisions': len(self.decision_engine.decision_history)
            },
            'performance': {
                'metrics_tracked': list(self.performance_metrics.keys()),
                'performance_trend': self._calculate_performance_trend(),
                'last_adaptation': self.adaptation_state.get('last_adaptation', 'never')
            },
            'system_health': {
                'last_checkpoint': datetime.fromtimestamp(self.last_checkpoint).isoformat(),
                'config': {
                    'autonomous_optimization': self.config.autonomous_optimization,
                    'continuous_learning': self.config.continuous_learning,
                    'meta_learning_enabled': self.config.meta_learning_enabled
                }
            }
        }