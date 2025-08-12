"""
Health checks and system diagnostics for protein diffusion models.

This module provides comprehensive health monitoring and diagnostic
capabilities for ensuring system reliability in production.
"""

import time
import logging
import json
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import platform
import sys

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    from . import mock_torch as torch
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class HealthStatus:
    """Health check result."""
    component: str
    status: str  # 'healthy', 'warning', 'critical'
    message: str
    timestamp: float
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}

@dataclass
class SystemDiagnostics:
    """System diagnostic information."""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    gpu_available: bool
    gpu_memory: Optional[float]
    python_version: str
    platform_info: str
    uptime: float

class HealthChecker:
    """Comprehensive health checking system."""
    
    def __init__(self):
        self.start_time = time.time()
        self.last_check = {}
        self.check_interval = 60  # seconds
        
    def check_system_health(self) -> List[HealthStatus]:
        """Perform comprehensive system health checks."""
        health_checks = []
        
        # Check system resources
        health_checks.extend(self._check_system_resources())
        
        # Check dependencies
        health_checks.extend(self._check_dependencies())
        
        # Check model components
        health_checks.extend(self._check_model_components())
        
        # Check storage and I/O
        health_checks.extend(self._check_storage())
        
        return health_checks
    
    def _check_system_resources(self) -> List[HealthStatus]:
        """Check system resource availability."""
        checks = []
        current_time = time.time()
        
        # CPU check
        if PSUTIL_AVAILABLE:
            cpu_usage = psutil.cpu_percent(interval=1)
            if cpu_usage > 90:
                status = "critical"
                message = f"High CPU usage: {cpu_usage:.1f}%"
            elif cpu_usage > 70:
                status = "warning"
                message = f"Moderate CPU usage: {cpu_usage:.1f}%"
            else:
                status = "healthy"
                message = f"CPU usage normal: {cpu_usage:.1f}%"
        else:
            status = "warning"
            message = "CPU monitoring unavailable (psutil not installed)"
            cpu_usage = 0
        
        checks.append(HealthStatus(
            component="cpu",
            status=status,
            message=message,
            timestamp=current_time,
            details={"usage_percent": cpu_usage}
        ))
        
        # Memory check
        if PSUTIL_AVAILABLE:
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            if memory_usage > 90:
                status = "critical"
                message = f"High memory usage: {memory_usage:.1f}%"
            elif memory_usage > 75:
                status = "warning"
                message = f"Moderate memory usage: {memory_usage:.1f}%"
            else:
                status = "healthy"
                message = f"Memory usage normal: {memory_usage:.1f}%"
            
            details = {
                "usage_percent": memory_usage,
                "available_gb": memory.available / (1024**3),
                "total_gb": memory.total / (1024**3)
            }
        else:
            status = "warning"
            message = "Memory monitoring unavailable"
            details = {}
        
        checks.append(HealthStatus(
            component="memory",
            status=status,
            message=message,
            timestamp=current_time,
            details=details
        ))
        
        # GPU check
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                gpu_usage = (gpu_allocated / gpu_memory) * 100
                
                if gpu_usage > 90:
                    status = "critical"
                    message = f"High GPU memory usage: {gpu_usage:.1f}%"
                elif gpu_usage > 75:
                    status = "warning"
                    message = f"Moderate GPU memory usage: {gpu_usage:.1f}%"
                else:
                    status = "healthy"
                    message = f"GPU memory usage normal: {gpu_usage:.1f}%"
                
                details = {
                    "usage_percent": gpu_usage,
                    "allocated_gb": gpu_allocated,
                    "total_gb": gpu_memory,
                    "device_name": torch.cuda.get_device_name(0)
                }
            except Exception as e:
                status = "warning"
                message = f"GPU monitoring error: {e}"
                details = {}
        else:
            status = "warning"
            message = "GPU not available or not detected"
            details = {"cuda_available": False}
        
        checks.append(HealthStatus(
            component="gpu",
            status=status,
            message=message,
            timestamp=current_time,
            details=details
        ))
        
        return checks
    
    def _check_dependencies(self) -> List[HealthStatus]:
        """Check critical dependencies."""
        checks = []
        current_time = time.time()
        
        dependencies = {
            "torch": ("torch", "PyTorch for deep learning"),
            "numpy": ("numpy", "NumPy for numerical computing"),
            "esm": ("esm", "ESM for protein embeddings"),
            "biopython": ("Bio", "BioPython for structure analysis"),
            "streamlit": ("streamlit", "Streamlit for web interface"),
            "plotly": ("plotly", "Plotly for visualizations"),
        }
        
        for dep_name, (module_name, description) in dependencies.items():
            try:
                __import__(module_name)
                status = "healthy"
                message = f"{description} available"
            except ImportError:
                if dep_name in ["torch", "numpy"]:
                    status = "warning"
                    message = f"{description} not available (using fallback)"
                else:
                    status = "warning"
                    message = f"Optional dependency '{description}' not available"
            
            checks.append(HealthStatus(
                component=f"dependency_{dep_name}",
                status=status,
                message=message,
                timestamp=current_time,
                details={"module": module_name, "required": dep_name in ["torch", "numpy"]}
            ))
        
        return checks
    
    def _check_model_components(self) -> List[HealthStatus]:
        """Check model component health."""
        checks = []
        current_time = time.time()
        
        # Test basic model initialization
        try:
            from .diffuser import ProteinDiffuser, ProteinDiffuserConfig
            config = ProteinDiffuserConfig()
            diffuser = ProteinDiffuser(config)
            
            status = "healthy"
            message = "Model initialization successful"
            details = {"components_loaded": True}
        except Exception as e:
            status = "critical"
            message = f"Model initialization failed: {e}"
            details = {"error": str(e), "error_type": type(e).__name__}
        
        checks.append(HealthStatus(
            component="model_initialization",
            status=status,
            message=message,
            timestamp=current_time,
            details=details
        ))
        
        # Test basic generation capability
        try:
            test_result = diffuser.generate(motif="TEST", num_samples=1, max_length=10)
            if test_result and len(test_result) > 0:
                status = "healthy"
                message = "Generation capability verified"
                details = {"test_generation": True, "results_count": len(test_result)}
            else:
                status = "warning"
                message = "Generation produced no results"
                details = {"test_generation": False}
        except Exception as e:
            status = "warning"
            message = f"Generation test failed: {e}"
            details = {"error": str(e), "test_generation": False}
        
        checks.append(HealthStatus(
            component="generation_capability",
            status=status,
            message=message,
            timestamp=current_time,
            details=details
        ))
        
        return checks
    
    def _check_storage(self) -> List[HealthStatus]:
        """Check storage and I/O health."""
        checks = []
        current_time = time.time()
        
        # Check disk space
        if PSUTIL_AVAILABLE:
            try:
                disk_usage = psutil.disk_usage('/')
                usage_percent = (disk_usage.used / disk_usage.total) * 100
                
                if usage_percent > 95:
                    status = "critical"
                    message = f"Critical disk usage: {usage_percent:.1f}%"
                elif usage_percent > 85:
                    status = "warning"
                    message = f"High disk usage: {usage_percent:.1f}%"
                else:
                    status = "healthy"
                    message = f"Disk usage normal: {usage_percent:.1f}%"
                
                details = {
                    "usage_percent": usage_percent,
                    "free_gb": disk_usage.free / (1024**3),
                    "total_gb": disk_usage.total / (1024**3)
                }
            except Exception as e:
                status = "warning"
                message = f"Disk monitoring error: {e}"
                details = {}
        else:
            status = "warning"
            message = "Disk monitoring unavailable"
            details = {}
        
        checks.append(HealthStatus(
            component="disk_space",
            status=status,
            message=message,
            timestamp=current_time,
            details=details
        ))
        
        return checks
    
    def get_system_diagnostics(self) -> SystemDiagnostics:
        """Get comprehensive system diagnostics."""
        if PSUTIL_AVAILABLE:
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            disk_usage = psutil.disk_usage('/').percent
        else:
            cpu_usage = 0.0
            memory_usage = 0.0
            disk_usage = 0.0
        
        # GPU information
        gpu_available = TORCH_AVAILABLE and torch.cuda.is_available()
        gpu_memory = None
        if gpu_available:
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            except:
                gpu_memory = None
        
        return SystemDiagnostics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            gpu_available=gpu_available,
            gpu_memory=gpu_memory,
            python_version=sys.version,
            platform_info=platform.platform(),
            uptime=time.time() - self.start_time
        )
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        health_checks = self.check_system_health()
        diagnostics = self.get_system_diagnostics()
        
        # Categorize checks
        healthy_count = sum(1 for check in health_checks if check.status == "healthy")
        warning_count = sum(1 for check in health_checks if check.status == "warning")
        critical_count = sum(1 for check in health_checks if check.status == "critical")
        
        # Overall status
        if critical_count > 0:
            overall_status = "critical"
        elif warning_count > 0:
            overall_status = "warning"
        else:
            overall_status = "healthy"
        
        return {
            "timestamp": time.time(),
            "overall_status": overall_status,
            "summary": {
                "healthy": healthy_count,
                "warning": warning_count,
                "critical": critical_count,
                "total": len(health_checks)
            },
            "health_checks": [asdict(check) for check in health_checks],
            "system_diagnostics": asdict(diagnostics),
            "recommendations": self._generate_recommendations(health_checks)
        }
    
    def _generate_recommendations(self, health_checks: List[HealthStatus]) -> List[str]:
        """Generate actionable recommendations based on health checks."""
        recommendations = []
        
        for check in health_checks:
            if check.status == "critical":
                if check.component == "cpu":
                    recommendations.append("Consider reducing computational load or upgrading CPU")
                elif check.component == "memory":
                    recommendations.append("Close unnecessary applications or add more RAM")
                elif check.component == "gpu":
                    recommendations.append("Free GPU memory or upgrade GPU")
                elif check.component == "disk_space":
                    recommendations.append("Free disk space immediately - system may become unstable")
                elif "model" in check.component:
                    recommendations.append("Check model files and dependencies")
            
            elif check.status == "warning":
                if "dependency" in check.component:
                    dep_name = check.component.split("_")[-1]
                    recommendations.append(f"Consider installing {dep_name} for enhanced functionality")
                elif check.component in ["cpu", "memory", "gpu"]:
                    recommendations.append(f"Monitor {check.component} usage - approaching limits")
        
        return list(set(recommendations))  # Remove duplicates

# Global health checker instance
health_checker = HealthChecker()

def get_health_status() -> Dict[str, Any]:
    """Get current system health status."""
    return health_checker.generate_health_report()