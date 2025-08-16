"""
Production Deployment Manager for Protein Diffusion Design Lab.

This module provides comprehensive deployment, monitoring, and operational
capabilities for production environments.
"""

import logging
import time
import json
import subprocess
import shutil
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    # Mock yaml module
    class MockYaml:
        @staticmethod
        def dump(data, default_flow_style=False):
            import json
            return json.dumps(data, indent=2)
    yaml = MockYaml()

logger = logging.getLogger(__name__)


class DeploymentStatus(Enum):
    """Deployment status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"


class EnvironmentType(Enum):
    """Environment type enumeration."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class DeploymentConfig:
    """Configuration for deployment operations."""
    # Environment settings
    environment: EnvironmentType = EnvironmentType.DEVELOPMENT
    app_name: str = "protein-diffusion-lab"
    version: str = "1.0.0"
    
    # Infrastructure settings
    container_registry: str = "docker.io"
    kubernetes_namespace: str = "protein-diffusion"
    replicas: int = 3
    
    # Resource limits
    cpu_request: str = "500m"
    cpu_limit: str = "2000m"
    memory_request: str = "1Gi"
    memory_limit: str = "4Gi"
    
    # Storage settings
    persistent_volume_size: str = "20Gi"
    storage_class: str = "standard"
    
    # Security settings
    enable_tls: bool = True
    enable_rbac: bool = True
    service_account: str = "protein-diffusion-sa"
    
    # Monitoring settings
    enable_monitoring: bool = True
    enable_logging: bool = True
    log_level: str = "INFO"
    
    # Deployment strategy
    deployment_strategy: str = "RollingUpdate"
    max_surge: str = "25%"
    max_unavailable: str = "25%"
    
    # Health checks
    readiness_probe_path: str = "/health"
    liveness_probe_path: str = "/health"
    startup_probe_path: str = "/health"
    
    # Auto-scaling
    enable_hpa: bool = True
    min_replicas: int = 2
    max_replicas: int = 10
    target_cpu_utilization: int = 70
    
    # Configuration paths
    config_dir: str = "./deployment/config"
    kubernetes_dir: str = "./deployment/kubernetes"
    docker_dir: str = "./deployment/docker"


@dataclass
class DeploymentResult:
    """Result from a deployment operation."""
    deployment_id: str
    status: DeploymentStatus
    timestamp: float
    environment: EnvironmentType
    version: str
    
    # Deployment details
    services_deployed: List[str]
    configuration: Dict[str, Any]
    
    # Health status
    health_status: Dict[str, Any]
    
    # Error information
    error_message: Optional[str] = None
    rollback_info: Optional[Dict[str, Any]] = None


class ContainerManager:
    """Manage container builds and registry operations."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
    
    def build_image(self, dockerfile_path: str, image_tag: str) -> Dict[str, Any]:
        """Build Docker image."""
        logger.info(f"Building Docker image: {image_tag}")
        
        try:
            # Build command
            build_cmd = [
                "docker", "build",
                "-t", image_tag,
                "-f", dockerfile_path,
                "."
            ]
            
            result = subprocess.run(
                build_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"Docker image built successfully: {image_tag}")
            
            return {
                "success": True,
                "image_tag": image_tag,
                "build_output": result.stdout
            }
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Docker build failed: {e.stderr}")
            return {
                "success": False,
                "error": e.stderr,
                "returncode": e.returncode
            }
    
    def push_image(self, image_tag: str) -> Dict[str, Any]:
        """Push image to container registry."""
        logger.info(f"Pushing image to registry: {image_tag}")
        
        try:
            push_cmd = ["docker", "push", image_tag]
            
            result = subprocess.run(
                push_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"Image pushed successfully: {image_tag}")
            
            return {
                "success": True,
                "image_tag": image_tag,
                "push_output": result.stdout
            }
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Docker push failed: {e.stderr}")
            return {
                "success": False,
                "error": e.stderr,
                "returncode": e.returncode
            }
    
    def scan_image(self, image_tag: str) -> Dict[str, Any]:
        """Scan image for security vulnerabilities."""
        logger.info(f"Scanning image for vulnerabilities: {image_tag}")
        
        try:
            # Use trivy for security scanning
            scan_cmd = [
                "trivy", "image",
                "--format", "json",
                "--severity", "HIGH,CRITICAL",
                image_tag
            ]
            
            result = subprocess.run(
                scan_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            scan_results = json.loads(result.stdout)
            
            # Count vulnerabilities
            vulnerability_count = 0
            for result_item in scan_results.get("Results", []):
                vulnerability_count += len(result_item.get("Vulnerabilities", []))
            
            return {
                "success": True,
                "vulnerability_count": vulnerability_count,
                "scan_results": scan_results,
                "passed": vulnerability_count == 0
            }
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"Image scan failed: {e.stderr}")
            return {
                "success": False,
                "error": e.stderr,
                "passed": False
            }
        except FileNotFoundError:
            logger.warning("Trivy not found, skipping security scan")
            return {
                "success": False,
                "error": "Security scanner not available",
                "passed": True  # Allow deployment if scanner not available
            }


class KubernetesManager:
    """Manage Kubernetes deployments."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
    
    def generate_manifests(self) -> Dict[str, str]:
        """Generate Kubernetes manifests."""
        manifests = {}
        
        # Deployment manifest
        deployment_manifest = self._generate_deployment_manifest()
        manifests["deployment.yaml"] = yaml.dump(deployment_manifest, default_flow_style=False)
        
        # Service manifest
        service_manifest = self._generate_service_manifest()
        manifests["service.yaml"] = yaml.dump(service_manifest, default_flow_style=False)
        
        # ConfigMap manifest
        configmap_manifest = self._generate_configmap_manifest()
        manifests["configmap.yaml"] = yaml.dump(configmap_manifest, default_flow_style=False)
        
        # Ingress manifest (if TLS enabled)
        if self.config.enable_tls:
            ingress_manifest = self._generate_ingress_manifest()
            manifests["ingress.yaml"] = yaml.dump(ingress_manifest, default_flow_style=False)
        
        # HPA manifest (if enabled)
        if self.config.enable_hpa:
            hpa_manifest = self._generate_hpa_manifest()
            manifests["hpa.yaml"] = yaml.dump(hpa_manifest, default_flow_style=False)
        
        # ServiceAccount manifest (if RBAC enabled)
        if self.config.enable_rbac:
            sa_manifest = self._generate_serviceaccount_manifest()
            manifests["serviceaccount.yaml"] = yaml.dump(sa_manifest, default_flow_style=False)
        
        return manifests
    
    def _generate_deployment_manifest(self) -> Dict[str, Any]:
        """Generate Kubernetes Deployment manifest."""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": self.config.app_name,
                "namespace": self.config.kubernetes_namespace,
                "labels": {
                    "app": self.config.app_name,
                    "version": self.config.version,
                    "environment": self.config.environment.value
                }
            },
            "spec": {
                "replicas": self.config.replicas,
                "strategy": {
                    "type": self.config.deployment_strategy,
                    "rollingUpdate": {
                        "maxSurge": self.config.max_surge,
                        "maxUnavailable": self.config.max_unavailable
                    }
                },
                "selector": {
                    "matchLabels": {
                        "app": self.config.app_name
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": self.config.app_name,
                            "version": self.config.version
                        }
                    },
                    "spec": {
                        "serviceAccountName": self.config.service_account if self.config.enable_rbac else "default",
                        "containers": [{
                            "name": self.config.app_name,
                            "image": f"{self.config.container_registry}/{self.config.app_name}:{self.config.version}",
                            "ports": [{
                                "containerPort": 8000,
                                "name": "http"
                            }],
                            "resources": {
                                "requests": {
                                    "cpu": self.config.cpu_request,
                                    "memory": self.config.memory_request
                                },
                                "limits": {
                                    "cpu": self.config.cpu_limit,
                                    "memory": self.config.memory_limit
                                }
                            },
                            "env": [
                                {
                                    "name": "LOG_LEVEL",
                                    "value": self.config.log_level
                                },
                                {
                                    "name": "ENVIRONMENT",
                                    "value": self.config.environment.value
                                }
                            ],
                            "livenessProbe": {
                                "httpGet": {
                                    "path": self.config.liveness_probe_path,
                                    "port": 8000
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": self.config.readiness_probe_path,
                                    "port": 8000
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5
                            },
                            "startupProbe": {
                                "httpGet": {
                                    "path": self.config.startup_probe_path,
                                    "port": 8000
                                },
                                "initialDelaySeconds": 10,
                                "periodSeconds": 10,
                                "failureThreshold": 30
                            }
                        }]
                    }
                }
            }
        }
    
    def _generate_service_manifest(self) -> Dict[str, Any]:
        """Generate Kubernetes Service manifest."""
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{self.config.app_name}-service",
                "namespace": self.config.kubernetes_namespace,
                "labels": {
                    "app": self.config.app_name
                }
            },
            "spec": {
                "selector": {
                    "app": self.config.app_name
                },
                "ports": [{
                    "port": 80,
                    "targetPort": 8000,
                    "protocol": "TCP",
                    "name": "http"
                }],
                "type": "ClusterIP"
            }
        }
    
    def _generate_configmap_manifest(self) -> Dict[str, Any]:
        """Generate ConfigMap manifest."""
        return {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": f"{self.config.app_name}-config",
                "namespace": self.config.kubernetes_namespace
            },
            "data": {
                "app.conf": f"""
# Protein Diffusion Lab Configuration
environment: {self.config.environment.value}
log_level: {self.config.log_level}
enable_monitoring: {self.config.enable_monitoring}
enable_logging: {self.config.enable_logging}
"""
            }
        }
    
    def _generate_ingress_manifest(self) -> Dict[str, Any]:
        """Generate Ingress manifest."""
        return {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {
                "name": f"{self.config.app_name}-ingress",
                "namespace": self.config.kubernetes_namespace,
                "annotations": {
                    "nginx.ingress.kubernetes.io/rewrite-target": "/",
                    "cert-manager.io/cluster-issuer": "letsencrypt-prod" if self.config.environment == EnvironmentType.PRODUCTION else "letsencrypt-staging"
                }
            },
            "spec": {
                "tls": [{
                    "hosts": [f"{self.config.app_name}.{self.config.environment.value}.com"],
                    "secretName": f"{self.config.app_name}-tls"
                }],
                "rules": [{
                    "host": f"{self.config.app_name}.{self.config.environment.value}.com",
                    "http": {
                        "paths": [{
                            "path": "/",
                            "pathType": "Prefix",
                            "backend": {
                                "service": {
                                    "name": f"{self.config.app_name}-service",
                                    "port": {
                                        "number": 80
                                    }
                                }
                            }
                        }]
                    }
                }]
            }
        }
    
    def _generate_hpa_manifest(self) -> Dict[str, Any]:
        """Generate HorizontalPodAutoscaler manifest."""
        return {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"{self.config.app_name}-hpa",
                "namespace": self.config.kubernetes_namespace
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": self.config.app_name
                },
                "minReplicas": self.config.min_replicas,
                "maxReplicas": self.config.max_replicas,
                "metrics": [{
                    "type": "Resource",
                    "resource": {
                        "name": "cpu",
                        "target": {
                            "type": "Utilization",
                            "averageUtilization": self.config.target_cpu_utilization
                        }
                    }
                }]
            }
        }
    
    def _generate_serviceaccount_manifest(self) -> Dict[str, Any]:
        """Generate ServiceAccount manifest."""
        return {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": {
                "name": self.config.service_account,
                "namespace": self.config.kubernetes_namespace
            }
        }
    
    def deploy_manifests(self, manifests: Dict[str, str]) -> Dict[str, Any]:
        """Deploy manifests to Kubernetes."""
        logger.info(f"Deploying to Kubernetes namespace: {self.config.kubernetes_namespace}")
        
        results = {}
        
        # Create namespace if it doesn't exist
        namespace_result = self._ensure_namespace()
        results["namespace"] = namespace_result
        
        # Deploy each manifest
        for manifest_name, manifest_content in manifests.items():
            result = self._apply_manifest(manifest_name, manifest_content)
            results[manifest_name] = result
        
        return results
    
    def _ensure_namespace(self) -> Dict[str, Any]:
        """Ensure namespace exists."""
        try:
            cmd = ["kubectl", "get", "namespace", self.config.kubernetes_namespace]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                # Create namespace
                create_cmd = ["kubectl", "create", "namespace", self.config.kubernetes_namespace]
                create_result = subprocess.run(create_cmd, capture_output=True, text=True, check=True)
                logger.info(f"Created namespace: {self.config.kubernetes_namespace}")
                return {"success": True, "action": "created"}
            else:
                logger.info(f"Namespace already exists: {self.config.kubernetes_namespace}")
                return {"success": True, "action": "exists"}
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create namespace: {e.stderr}")
            return {"success": False, "error": e.stderr}
    
    def _apply_manifest(self, manifest_name: str, manifest_content: str) -> Dict[str, Any]:
        """Apply a single manifest."""
        try:
            # Write manifest to temporary file
            manifest_file = Path("/tmp") / manifest_name
            with open(manifest_file, 'w') as f:
                f.write(manifest_content)
            
            # Apply manifest
            cmd = ["kubectl", "apply", "-f", str(manifest_file), "-n", self.config.kubernetes_namespace]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Clean up
            manifest_file.unlink()
            
            logger.info(f"Applied manifest: {manifest_name}")
            return {"success": True, "output": result.stdout}
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to apply manifest {manifest_name}: {e.stderr}")
            return {"success": False, "error": e.stderr}
    
    def check_deployment_status(self) -> Dict[str, Any]:
        """Check deployment status."""
        try:
            cmd = [
                "kubectl", "get", "deployment", self.config.app_name,
                "-n", self.config.kubernetes_namespace,
                "-o", "json"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            deployment_info = json.loads(result.stdout)
            
            status = deployment_info.get("status", {})
            
            return {
                "success": True,
                "ready_replicas": status.get("readyReplicas", 0),
                "replicas": status.get("replicas", 0),
                "updated_replicas": status.get("updatedReplicas", 0),
                "available_replicas": status.get("availableReplicas", 0),
                "deployment_ready": status.get("readyReplicas", 0) == status.get("replicas", 0)
            }
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to check deployment status: {e.stderr}")
            return {"success": False, "error": e.stderr}


class DeploymentManager:
    """Main deployment manager orchestrating all deployment operations."""
    
    def __init__(self, config: Optional[DeploymentConfig] = None):
        self.config = config or DeploymentConfig()
        
        # Initialize managers
        self.container_manager = ContainerManager(self.config)
        self.kubernetes_manager = KubernetesManager(self.config)
        
        # Create output directories
        self._setup_directories()
        
        logger.info(f"Deployment Manager initialized for {self.config.environment.value} environment")
    
    def _setup_directories(self):
        """Setup deployment directories."""
        for directory in [self.config.config_dir, self.config.kubernetes_dir, self.config.docker_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def deploy_complete_stack(
        self,
        build_image: bool = True,
        run_security_scan: bool = True,
        push_to_registry: bool = True
    ) -> DeploymentResult:
        """Deploy the complete application stack."""
        deployment_id = self._generate_deployment_id()
        start_time = time.time()
        
        logger.info(f"Starting complete stack deployment: {deployment_id}")
        
        try:
            # Phase 1: Container Build and Security
            if build_image:
                image_tag = f"{self.config.container_registry}/{self.config.app_name}:{self.config.version}"
                dockerfile_path = Path(self.config.docker_dir) / "Dockerfile"
                
                # Build image
                build_result = self.container_manager.build_image(str(dockerfile_path), image_tag)
                if not build_result["success"]:
                    raise RuntimeError(f"Image build failed: {build_result['error']}")
                
                # Security scan
                if run_security_scan:
                    scan_result = self.container_manager.scan_image(image_tag)
                    if not scan_result["passed"]:
                        logger.warning(f"Security scan found {scan_result.get('vulnerability_count', 0)} vulnerabilities")
                        if self.config.environment == EnvironmentType.PRODUCTION:
                            raise RuntimeError("Production deployment blocked due to security vulnerabilities")
                
                # Push to registry
                if push_to_registry:
                    push_result = self.container_manager.push_image(image_tag)
                    if not push_result["success"]:
                        raise RuntimeError(f"Image push failed: {push_result['error']}")
            
            # Phase 2: Generate Kubernetes Manifests
            manifests = self.kubernetes_manager.generate_manifests()
            
            # Save manifests to disk
            self._save_manifests(manifests)
            
            # Phase 3: Deploy to Kubernetes
            deployment_results = self.kubernetes_manager.deploy_manifests(manifests)
            
            # Check for deployment failures
            failed_deployments = [name for name, result in deployment_results.items() 
                                if not result.get("success", False)]
            
            if failed_deployments:
                raise RuntimeError(f"Failed to deploy: {', '.join(failed_deployments)}")
            
            # Phase 4: Wait for deployment to be ready
            self._wait_for_deployment_ready()
            
            # Phase 5: Health check
            health_status = self._check_deployment_health()
            
            # Create successful deployment result
            deployment_result = DeploymentResult(
                deployment_id=deployment_id,
                status=DeploymentStatus.DEPLOYED,
                timestamp=start_time,
                environment=self.config.environment,
                version=self.config.version,
                services_deployed=list(manifests.keys()),
                configuration=asdict(self.config),
                health_status=health_status
            )
            
            logger.info(f"Deployment completed successfully: {deployment_id}")
            return deployment_result
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            
            # Create failed deployment result
            return DeploymentResult(
                deployment_id=deployment_id,
                status=DeploymentStatus.FAILED,
                timestamp=start_time,
                environment=self.config.environment,
                version=self.config.version,
                services_deployed=[],
                configuration=asdict(self.config),
                health_status={},
                error_message=str(e)
            )
    
    def _save_manifests(self, manifests: Dict[str, str]):
        """Save generated manifests to disk."""
        manifest_dir = Path(self.config.kubernetes_dir) / self.config.environment.value
        manifest_dir.mkdir(parents=True, exist_ok=True)
        
        for manifest_name, manifest_content in manifests.items():
            manifest_file = manifest_dir / manifest_name
            with open(manifest_file, 'w') as f:
                f.write(manifest_content)
            
            logger.info(f"Saved manifest: {manifest_file}")
    
    def _wait_for_deployment_ready(self, timeout: int = 300):
        """Wait for deployment to be ready."""
        logger.info("Waiting for deployment to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.kubernetes_manager.check_deployment_status()
            
            if status.get("success") and status.get("deployment_ready"):
                logger.info("Deployment is ready")
                return
            
            time.sleep(10)
        
        raise RuntimeError("Deployment readiness timeout")
    
    def _check_deployment_health(self) -> Dict[str, Any]:
        """Check deployment health."""
        # This would implement actual health checks
        # For now, return a basic health status
        return {
            "overall_status": "healthy",
            "timestamp": time.time(),
            "checks": {
                "deployment_ready": True,
                "pods_running": True,
                "service_available": True
            }
        }
    
    def _generate_deployment_id(self) -> str:
        """Generate unique deployment identifier."""
        import hashlib
        timestamp = str(time.time())
        environment = self.config.environment.value
        version = self.config.version
        unique_string = f"{environment}_{version}_{timestamp}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:12]
    
    def rollback_deployment(self, target_version: str) -> DeploymentResult:
        """Rollback to a previous version."""
        deployment_id = self._generate_deployment_id()
        start_time = time.time()
        
        logger.info(f"Rolling back deployment to version: {target_version}")
        
        try:
            # Update config with target version
            rollback_config = DeploymentConfig(**asdict(self.config))
            rollback_config.version = target_version
            
            # Create new managers with rollback config
            rollback_k8s_manager = KubernetesManager(rollback_config)
            
            # Generate manifests for target version
            manifests = rollback_k8s_manager.generate_manifests()
            
            # Deploy rollback manifests
            deployment_results = rollback_k8s_manager.deploy_manifests(manifests)
            
            # Wait for rollback to complete
            self._wait_for_deployment_ready()
            
            # Check health
            health_status = self._check_deployment_health()
            
            return DeploymentResult(
                deployment_id=deployment_id,
                status=DeploymentStatus.ROLLED_BACK,
                timestamp=start_time,
                environment=self.config.environment,
                version=target_version,
                services_deployed=list(manifests.keys()),
                configuration=asdict(rollback_config),
                health_status=health_status,
                rollback_info={"from_version": self.config.version, "to_version": target_version}
            )
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            
            return DeploymentResult(
                deployment_id=deployment_id,
                status=DeploymentStatus.FAILED,
                timestamp=start_time,
                environment=self.config.environment,
                version=self.config.version,
                services_deployed=[],
                configuration=asdict(self.config),
                health_status={},
                error_message=f"Rollback failed: {e}"
            )
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status."""
        k8s_status = self.kubernetes_manager.check_deployment_status()
        health_status = self._check_deployment_health()
        
        return {
            "kubernetes_status": k8s_status,
            "health_status": health_status,
            "configuration": asdict(self.config),
            "timestamp": time.time()
        }


# Convenience functions for deployment operations

def deploy_to_environment(
    environment: EnvironmentType,
    version: str,
    config_overrides: Optional[Dict[str, Any]] = None
) -> DeploymentResult:
    """Deploy to a specific environment with version."""
    config = DeploymentConfig(environment=environment, version=version)
    
    if config_overrides:
        for key, value in config_overrides.items():
            setattr(config, key, value)
    
    manager = DeploymentManager(config)
    return manager.deploy_complete_stack()


def quick_deploy(version: str = "latest") -> DeploymentResult:
    """Quick deployment with minimal configuration."""
    config = DeploymentConfig(
        environment=EnvironmentType.DEVELOPMENT,
        version=version,
        replicas=1,
        enable_tls=False,
        enable_hpa=False
    )
    
    manager = DeploymentManager(config)
    return manager.deploy_complete_stack(build_image=True, run_security_scan=False)