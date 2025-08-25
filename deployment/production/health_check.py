#!/usr/bin/env python3
"""
Production Health Check Script for Protein Diffusion Design Lab

This script performs comprehensive health checks for all system components
and reports the overall system health status.
"""

import os
import sys
import time
import json
import requests
import psutil
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

# Add src to path for imports
sys.path.append('/app/src')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    component: str
    status: HealthStatus
    message: str
    response_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class HealthChecker:
    """Comprehensive health checker for all system components."""
    
    def __init__(self):
        self.checks = []
        self.timeout = 10  # seconds
        
    def add_check(self, check_func, name: str):
        """Add a health check function."""
        self.checks.append((check_func, name))
        
    def run_all_checks(self) -> List[HealthCheckResult]:
        """Run all registered health checks."""
        results = []
        
        for check_func, name in self.checks:
            try:
                start_time = time.time()
                result = check_func()
                result.response_time_ms = (time.time() - start_time) * 1000
                results.append(result)
            except Exception as e:
                result = HealthCheckResult(
                    component=name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed: {str(e)}",
                    response_time_ms=0
                )
                results.append(result)
                
        return results
        
    def check_api_health(self) -> HealthCheckResult:
        """Check main API health."""
        try:
            response = requests.get(
                "http://localhost:8000/health",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return HealthCheckResult(
                    component="api",
                    status=HealthStatus.HEALTHY,
                    message="API is responding normally",
                    metadata={"status_code": response.status_code}
                )
            else:
                return HealthCheckResult(
                    component="api",
                    status=HealthStatus.WARNING,
                    message=f"API returned status {response.status_code}",
                    metadata={"status_code": response.status_code}
                )
                
        except requests.exceptions.ConnectionError:
            return HealthCheckResult(
                component="api",
                status=HealthStatus.CRITICAL,
                message="Cannot connect to API server"
            )
        except requests.exceptions.Timeout:
            return HealthCheckResult(
                component="api",
                status=HealthStatus.WARNING,
                message="API response timeout"
            )
        except Exception as e:
            return HealthCheckResult(
                component="api",
                status=HealthStatus.CRITICAL,
                message=f"API health check error: {str(e)}"
            )
            
    def check_database_health(self) -> HealthCheckResult:
        """Check PostgreSQL database health."""
        try:
            import psycopg2
            from urllib.parse import urlparse
            
            db_url = os.getenv('DATABASE_URL')
            if not db_url:
                return HealthCheckResult(
                    component="database",
                    status=HealthStatus.CRITICAL,
                    message="DATABASE_URL not configured"
                )
                
            parsed = urlparse(db_url)
            conn = psycopg2.connect(
                host=parsed.hostname,
                port=parsed.port or 5432,
                database=parsed.path[1:],  # Remove leading /
                user=parsed.username,
                password=parsed.password,
                connect_timeout=self.timeout
            )
            
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            if result:
                return HealthCheckResult(
                    component="database",
                    status=HealthStatus.HEALTHY,
                    message="Database connection successful"
                )
            else:
                return HealthCheckResult(
                    component="database",
                    status=HealthStatus.WARNING,
                    message="Database query returned unexpected result"
                )
                
        except ImportError:
            return HealthCheckResult(
                component="database",
                status=HealthStatus.WARNING,
                message="psycopg2 not available for database check"
            )
        except Exception as e:
            return HealthCheckResult(
                component="database",
                status=HealthStatus.CRITICAL,
                message=f"Database connection failed: {str(e)}"
            )
            
    def check_redis_health(self) -> HealthCheckResult:
        """Check Redis cache health."""
        try:
            import redis
            from urllib.parse import urlparse
            
            redis_url = os.getenv('REDIS_URL', 'redis://redis:6379/0')
            parsed = urlparse(redis_url)
            
            r = redis.Redis(
                host=parsed.hostname or 'localhost',
                port=parsed.port or 6379,
                db=int(parsed.path[1:]) if parsed.path and len(parsed.path) > 1 else 0,
                socket_timeout=self.timeout,
                socket_connect_timeout=self.timeout
            )
            
            # Test basic operations
            test_key = "health_check_test"
            r.set(test_key, "test_value", ex=60)
            value = r.get(test_key)
            r.delete(test_key)
            
            if value == b"test_value":
                return HealthCheckResult(
                    component="redis",
                    status=HealthStatus.HEALTHY,
                    message="Redis is working correctly"
                )
            else:
                return HealthCheckResult(
                    component="redis",
                    status=HealthStatus.WARNING,
                    message="Redis test operation failed"
                )
                
        except ImportError:
            return HealthCheckResult(
                component="redis",
                status=HealthStatus.WARNING,
                message="redis package not available for check"
            )
        except Exception as e:
            return HealthCheckResult(
                component="redis",
                status=HealthStatus.CRITICAL,
                message=f"Redis connection failed: {str(e)}"
            )
            
    def check_gpu_health(self) -> HealthCheckResult:
        """Check GPU availability and health."""
        try:
            if not os.getenv('GPU_ENABLED', 'false').lower() == 'true':
                return HealthCheckResult(
                    component="gpu",
                    status=HealthStatus.HEALTHY,
                    message="GPU not enabled, running in CPU mode"
                )
                
            import torch
            
            if not torch.cuda.is_available():
                return HealthCheckResult(
                    component="gpu",
                    status=HealthStatus.WARNING,
                    message="CUDA not available"
                )
                
            device_count = torch.cuda.device_count()
            gpu_info = []
            
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
                
                gpu_info.append({
                    'device_id': i,
                    'name': device_name,
                    'memory_allocated_gb': round(memory_allocated, 2),
                    'memory_total_gb': round(memory_total, 2),
                    'memory_utilization_pct': round((memory_allocated / memory_total) * 100, 1)
                })
                
            return HealthCheckResult(
                component="gpu",
                status=HealthStatus.HEALTHY,
                message=f"{device_count} GPU(s) available and accessible",
                metadata={'devices': gpu_info}
            )
            
        except ImportError:
            return HealthCheckResult(
                component="gpu",
                status=HealthStatus.WARNING,
                message="PyTorch not available for GPU check"
            )
        except Exception as e:
            return HealthCheckResult(
                component="gpu",
                status=HealthStatus.WARNING,
                message=f"GPU health check failed: {str(e)}"
            )
            
    def check_system_resources(self) -> HealthCheckResult:
        """Check system resource utilization."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Network stats
            network = psutil.net_io_counters()
            
            # Load average
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else (0, 0, 0)
            
            # Determine overall resource health
            status = HealthStatus.HEALTHY
            messages = []
            
            if cpu_percent > 90:
                status = HealthStatus.CRITICAL
                messages.append(f"High CPU usage: {cpu_percent}%")
            elif cpu_percent > 80:
                status = HealthStatus.WARNING
                messages.append(f"Elevated CPU usage: {cpu_percent}%")
                
            if memory_percent > 95:
                status = HealthStatus.CRITICAL
                messages.append(f"Critical memory usage: {memory_percent}%")
            elif memory_percent > 85:
                status = HealthStatus.WARNING
                messages.append(f"High memory usage: {memory_percent}%")
                
            if disk_percent > 95:
                status = HealthStatus.CRITICAL
                messages.append(f"Critical disk usage: {disk_percent}%")
            elif disk_percent > 85:
                status = HealthStatus.WARNING
                messages.append(f"High disk usage: {disk_percent}%")
                
            if not messages:
                messages.append("System resources are healthy")
                
            return HealthCheckResult(
                component="system_resources",
                status=status,
                message="; ".join(messages),
                metadata={
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'memory_available_gb': round(memory.available / 1024**3, 2),
                    'disk_percent': disk_percent,
                    'disk_free_gb': round(disk.free / 1024**3, 2),
                    'load_average': load_avg,
                    'network_bytes_sent': network.bytes_sent if network else 0,
                    'network_bytes_recv': network.bytes_recv if network else 0
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="system_resources",
                status=HealthStatus.WARNING,
                message=f"Resource check failed: {str(e)}"
            )
            
    def check_protein_diffusion_modules(self) -> HealthCheckResult:
        """Check if core protein diffusion modules can be imported."""
        try:
            # Test core module imports
            from protein_diffusion import ProteinDiffuser, AffinityRanker
            from protein_diffusion.nextgen_orchestration import NextGenOrchestrationEngine
            from protein_diffusion.nextgen_performance_optimizer import NextGenPerformanceOptimizer
            from protein_diffusion.comprehensive_validation_system import ComprehensiveValidationSystem
            
            # Test basic instantiation
            diffuser_config = type('Config', (), {})()
            # Don't actually instantiate to avoid loading models
            
            return HealthCheckResult(
                component="protein_modules",
                status=HealthStatus.HEALTHY,
                message="All core protein diffusion modules importable",
                metadata={
                    'modules_checked': [
                        'ProteinDiffuser', 'AffinityRanker', 'NextGenOrchestrationEngine',
                        'NextGenPerformanceOptimizer', 'ComprehensiveValidationSystem'
                    ]
                }
            )
            
        except ImportError as e:
            return HealthCheckResult(
                component="protein_modules",
                status=HealthStatus.CRITICAL,
                message=f"Failed to import core modules: {str(e)}"
            )
        except Exception as e:
            return HealthCheckResult(
                component="protein_modules",
                status=HealthStatus.WARNING,
                message=f"Module check failed: {str(e)}"
            )
            
    def check_streaming_api(self) -> HealthCheckResult:
        """Check streaming API health."""
        try:
            import socket
            
            # Check if streaming port is open
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            result = sock.connect_ex(('localhost', 8765))
            sock.close()
            
            if result == 0:
                return HealthCheckResult(
                    component="streaming_api",
                    status=HealthStatus.HEALTHY,
                    message="Streaming API port is accessible"
                )
            else:
                return HealthCheckResult(
                    component="streaming_api",
                    status=HealthStatus.WARNING,
                    message="Streaming API port not accessible"
                )
                
        except Exception as e:
            return HealthCheckResult(
                component="streaming_api",
                status=HealthStatus.WARNING,
                message=f"Streaming API check failed: {str(e)}"
            )


def main():
    """Main health check execution."""
    checker = HealthChecker()
    
    # Register all health checks
    checker.add_check(checker.check_api_health, "api")
    checker.add_check(checker.check_database_health, "database")
    checker.add_check(checker.check_redis_health, "redis")
    checker.add_check(checker.check_gpu_health, "gpu")
    checker.add_check(checker.check_system_resources, "system_resources")
    checker.add_check(checker.check_protein_diffusion_modules, "protein_modules")
    checker.add_check(checker.check_streaming_api, "streaming_api")
    
    # Run all checks
    print("🧬 Protein Diffusion Design Lab - Health Check")
    print("=" * 50)
    
    results = checker.run_all_checks()
    
    # Print results
    overall_status = HealthStatus.HEALTHY
    critical_count = 0
    warning_count = 0
    
    for result in results:
        status_icon = {
            HealthStatus.HEALTHY: "✅",
            HealthStatus.WARNING: "⚠️",
            HealthStatus.CRITICAL: "❌",
            HealthStatus.UNKNOWN: "❓"
        }.get(result.status, "❓")
        
        print(f"{status_icon} {result.component}: {result.message}")
        if result.response_time_ms > 0:
            print(f"   Response time: {result.response_time_ms:.1f}ms")
            
        if result.status == HealthStatus.CRITICAL:
            critical_count += 1
            overall_status = HealthStatus.CRITICAL
        elif result.status == HealthStatus.WARNING and overall_status != HealthStatus.CRITICAL:
            warning_count += 1
            overall_status = HealthStatus.WARNING
            
    print("=" * 50)
    
    # Overall status
    overall_icon = {
        HealthStatus.HEALTHY: "✅",
        HealthStatus.WARNING: "⚠️",
        HealthStatus.CRITICAL: "❌"
    }.get(overall_status, "❓")
    
    print(f"{overall_icon} Overall Status: {overall_status.value.upper()}")
    print(f"📊 Summary: {len(results)} checks, {critical_count} critical, {warning_count} warnings")
    
    # JSON output for programmatic use
    if "--json" in sys.argv:
        health_data = {
            "overall_status": overall_status.value,
            "timestamp": time.time(),
            "checks": [
                {
                    "component": r.component,
                    "status": r.status.value,
                    "message": r.message,
                    "response_time_ms": r.response_time_ms,
                    "metadata": r.metadata
                }
                for r in results
            ],
            "summary": {
                "total_checks": len(results),
                "critical_count": critical_count,
                "warning_count": warning_count
            }
        }
        print(json.dumps(health_data, indent=2))
        
    # Exit codes for container health checks
    if overall_status == HealthStatus.CRITICAL:
        sys.exit(1)
    elif overall_status == HealthStatus.WARNING:
        sys.exit(2)  # Warning status
    else:
        sys.exit(0)  # Healthy


if __name__ == "__main__":
    main()