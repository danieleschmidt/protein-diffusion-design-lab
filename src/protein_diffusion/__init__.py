"""
Protein Diffusion Design Lab

A plug-and-play diffusion pipeline for protein scaffolds that rivals commercial suites.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "your.email@example.com"

# Core API imports
from .diffuser import ProteinDiffuser, ProteinDiffuserConfig
from .ranker import AffinityRanker, AffinityRankerConfig
from .models import DiffusionTransformer, DDPM, DiffusionTransformerConfig, DDPMConfig
from .tokenization.selfies_tokenizer import SELFIESTokenizer, TokenizerConfig
from .tokenization.protein_embeddings import ProteinEmbeddings, EmbeddingConfig
from .folding.structure_predictor import StructurePredictor, StructurePredictorConfig

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    # Main interfaces
    "ProteinDiffuser",
    "AffinityRanker",
    # Core models
    "DiffusionTransformer",
    "DDPM",
    # Tokenization
    "SELFIESTokenizer", 
    "ProteinEmbeddings",
    # Structure prediction
    "StructurePredictor",
    # Configurations
    "ProteinDiffuserConfig",
    "AffinityRankerConfig",
    "DiffusionTransformerConfig",
    "DDPMConfig",
    "TokenizerConfig",
    "EmbeddingConfig",
    "StructurePredictorConfig",
    # Integration and Automation (Generation 1 additions)
    "IntegrationManager",
    "IntegrationConfig",
    "WorkflowResult",
    "WorkflowAutomation",
    "AutomationConfig",
    "ProteinDiffusionAPI",
    # Generation 2 components (Robustness)
    "ErrorRecoveryManager",
    "RecoveryConfig",
    "ErrorType",
    "RecoveryAction",
    "SystemMonitor",
    "MetricsCollector", 
    "PerformanceProfiler",
    "SecurityManager",
    "SecurityConfig",
    "SecurityLevel",
    # Generation 3 components (Scalability)
    "DistributedProcessingManager",
    "PerformanceOptimizer",
    "AdaptiveScalingManager",
]

# Import new Generation 1 components
try:
    from .integration_manager import IntegrationManager, IntegrationConfig, WorkflowResult
    from .workflow_automation import WorkflowAutomation, AutomationConfig
    from .enhanced_api import ProteinDiffusionAPI
    GENERATION_1_AVAILABLE = True
except ImportError:
    GENERATION_1_AVAILABLE = False
    IntegrationManager = None
    IntegrationConfig = None
    WorkflowResult = None
    WorkflowAutomation = None
    AutomationConfig = None
    ProteinDiffusionAPI = None

# Import new Generation 2 components
try:
    from .robust_error_recovery import ErrorRecoveryManager, RecoveryConfig, ErrorType, RecoveryAction
    # Use existing monitoring module instead of advanced_monitoring
    from .monitoring import SystemMonitor, MetricsCollector, PerformanceProfiler
    from .enterprise_security import SecurityManager, SecurityConfig, SecurityLevel, User, Session
    GENERATION_2_AVAILABLE = True
except ImportError:
    GENERATION_2_AVAILABLE = False
    ErrorRecoveryManager = None
    RecoveryConfig = None
    ErrorType = None  
    RecoveryAction = None
    SystemMonitor = None
    MetricsCollector = None
    PerformanceProfiler = None
    SecurityManager = None
    SecurityConfig = None
    SecurityLevel = None
    User = None
    Session = None

# Import new Generation 3 components
try:
    from .distributed_processing import DistributedProcessingManager, DistributedConfig
    from .performance_optimization import PerformanceOptimizer, PerformanceConfig
    from .adaptive_scaling import AdaptiveScalingManager, ScalingConfig
    GENERATION_3_AVAILABLE = True
except ImportError:
    GENERATION_3_AVAILABLE = False
    DistributedProcessingManager = None
    DistributedConfig = None
    PerformanceOptimizer = None
    PerformanceConfig = None
    AdaptiveScalingManager = None
    ScalingConfig = None