#!/usr/bin/env python3
"""
Terragon Autonomous Value Discovery Engine
Continuously discovers, scores, and prioritizes SDLC improvement tasks
"""

import json
import subprocess
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml


class ValueDiscoveryEngine:
    """Autonomous value discovery and prioritization system"""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.config_path = self.repo_path / ".terragon" / "config.yaml"
        self.metrics_path = self.repo_path / ".terragon" / "value-metrics.json"
        self.backlog_path = self.repo_path / "BACKLOG.md"
        
        self.config = self._load_config()
        self.metrics = self._load_metrics()
        
    def _load_config(self) -> Dict:
        """Load Terragon configuration"""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        return self._default_config()
    
    def _load_metrics(self) -> Dict:
        """Load historical metrics"""
        if self.metrics_path.exists():
            with open(self.metrics_path) as f:
                return json.load(f)
        return {"executionHistory": [], "backlogMetrics": {}}
    
    def _default_config(self) -> Dict:
        """Default configuration for new repositories"""
        return {
            "scoring": {
                "weights": {"wsjf": 0.5, "ice": 0.2, "technicalDebt": 0.2, "security": 0.1},
                "thresholds": {"minScore": 10, "maxRisk": 0.8}
            }
        }
    
    def discover_value_items(self) -> List[Dict]:
        """Discover all potential value items across multiple sources"""
        items = []
        
        # Git history analysis
        items.extend(self._discover_from_git_history())
        
        # Static analysis
        items.extend(self._discover_from_static_analysis())
        
        # Security vulnerabilities
        items.extend(self._discover_security_items())
        
        # Dependency updates
        items.extend(self._discover_dependency_updates())
        
        # Documentation gaps
        items.extend(self._discover_documentation_gaps())
        
        # Infrastructure improvements
        items.extend(self._discover_infrastructure_gaps())
        
        return items
    
    def _discover_from_git_history(self) -> List[Dict]:
        """Extract TODO/FIXME items from git history and current code"""
        items = []
        
        try:
            # Find TODO/FIXME/HACK comments in current code
            result = subprocess.run(
                ["git", "grep", "-n", "-i", r"TODO\|FIXME\|HACK\|XXX"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            for line in result.stdout.split('\n'):
                if line.strip():
                    parts = line.split(':')
                    if len(parts) >= 3:
                        file_path = parts[0]
                        line_num = parts[1]
                        comment = ':'.join(parts[2:]).strip()
                        
                        items.append({
                            "id": f"code-debt-{len(items)}",
                            "title": f"Address code comment in {file_path}:{line_num}",
                            "description": comment,
                            "category": "technical-debt",
                            "type": "code-improvement",
                            "file": file_path,
                            "effort": self._estimate_effort(comment),
                            "impact": self._estimate_impact(comment, "technical-debt")
                        })
        except subprocess.CalledProcessError:
            pass
        
        return items
    
    def _discover_from_static_analysis(self) -> List[Dict]:
        """Run static analysis tools to discover code quality issues"""
        items = []
        
        # Python-specific analysis
        if (self.repo_path / "pyproject.toml").exists():
            # Flake8 issues
            items.extend(self._run_flake8_analysis())
            
            # MyPy issues
            items.extend(self._run_mypy_analysis())
            
            # Security issues (Bandit)
            items.extend(self._run_bandit_analysis())
        
        return items
    
    def _discover_security_items(self) -> List[Dict]:
        """Discover security-related improvements"""
        items = []
        
        # Check for missing security files
        security_files = [
            ("SECURITY.md", "security-policy"),
            (".github/dependabot.yml", "dependency-automation"),
            ("bandit.yaml", "security-scanning"),
            (".safety-policy.json", "vulnerability-policy")
        ]
        
        for file_path, category in security_files:
            if not (self.repo_path / file_path).exists():
                items.append({
                    "id": f"security-{category}",
                    "title": f"Add missing {file_path}",
                    "description": f"Security enhancement: add {file_path} for {category}",
                    "category": "security",
                    "type": "security-enhancement",
                    "effort": 2,
                    "impact": 8,
                    "priority": "high"
                })
        
        return items
    
    def _discover_dependency_updates(self) -> List[Dict]:
        """Check for outdated dependencies"""
        items = []
        
        try:
            # Check Python dependencies if requirements files exist
            for req_file in ["requirements.txt", "requirements-dev.txt"]:
                req_path = self.repo_path / req_file
                if req_path.exists():
                    # This would integrate with pip-audit or safety in a real implementation
                    items.append({
                        "id": f"deps-{req_file}",
                        "title": f"Update dependencies in {req_file}",
                        "description": f"Check and update outdated packages in {req_file}",
                        "category": "dependencies",
                        "type": "dependency-update",
                        "effort": 3,
                        "impact": 6,
                        "priority": "medium"
                    })
        except Exception:
            pass
        
        return items
    
    def _discover_documentation_gaps(self) -> List[Dict]:
        """Find documentation that needs updating or creation"""
        items = []
        
        # Check for missing essential docs
        essential_docs = [
            ("CHANGELOG.md", "changelog"),
            ("docs/API.md", "api-documentation"),
            ("docs/ARCHITECTURE.md", "architecture-docs"),
            ("docs/PERFORMANCE.md", "performance-docs")
        ]
        
        for doc_path, doc_type in essential_docs:
            if not (self.repo_path / doc_path).exists():
                items.append({
                    "id": f"docs-{doc_type}",
                    "title": f"Create {doc_path}",
                    "description": f"Add missing documentation: {doc_path}",
                    "category": "documentation",
                    "type": "doc-creation",
                    "effort": 4,
                    "impact": 5,
                    "priority": "low"
                })
        
        return items
    
    def _discover_infrastructure_gaps(self) -> List[Dict]:
        """Find infrastructure and DevOps improvements"""
        items = []
        
        # Check for missing CI/CD
        if not (self.repo_path / ".github" / "workflows").exists():
            items.append({
                "id": "infra-github-actions",
                "title": "Set up GitHub Actions CI/CD",
                "description": "Create comprehensive CI/CD pipeline with testing, security scans, and deployment",
                "category": "infrastructure",
                "type": "ci-cd-setup",
                "effort": 8,
                "impact": 9,
                "priority": "high"
            })
        
        # Check for monitoring setup
        if not (self.repo_path / "monitoring").exists():
            items.append({
                "id": "infra-monitoring",
                "title": "Set up application monitoring",
                "description": "Add comprehensive monitoring with metrics, logs, and alerts",
                "category": "infrastructure", 
                "type": "monitoring-setup",
                "effort": 6,
                "impact": 7,
                "priority": "medium"
            })
        
        return items
    
    def calculate_scores(self, items: List[Dict]) -> List[Dict]:
        """Calculate WSJF, ICE, and composite scores for all items"""
        scored_items = []
        
        for item in items:
            # WSJF calculation
            wsjf = self._calculate_wsjf(item)
            
            # ICE calculation  
            ice = self._calculate_ice(item)
            
            # Technical debt score
            tech_debt = self._calculate_technical_debt_score(item)
            
            # Composite score
            weights = self.config["scoring"]["weights"]
            composite = (
                weights["wsjf"] * self._normalize_score(wsjf, 0, 100) +
                weights["ice"] * self._normalize_score(ice, 0, 1000) +
                weights["technicalDebt"] * self._normalize_score(tech_debt, 0, 100) +
                weights["security"] * (2.0 if item["category"] == "security" else 1.0)
            )
            
            # Apply category boosts
            if item["category"] == "security":
                composite *= 2.0
            elif item["category"] == "infrastructure":
                composite *= 1.5
            
            item.update({
                "scores": {
                    "wsjf": wsjf,
                    "ice": ice,
                    "technicalDebt": tech_debt,
                    "composite": composite
                }
            })
            
            scored_items.append(item)
        
        return sorted(scored_items, key=lambda x: x["scores"]["composite"], reverse=True)
    
    def _calculate_wsjf(self, item: Dict) -> float:
        """Calculate Weighted Shortest Job First score"""
        # Cost of Delay components
        user_value = item.get("impact", 5)  # 1-10 scale
        time_criticality = self._get_time_criticality(item)
        risk_reduction = self._get_risk_reduction(item)
        opportunity = self._get_opportunity_enablement(item)
        
        cost_of_delay = user_value + time_criticality + risk_reduction + opportunity
        job_size = item.get("effort", 3)  # story points
        
        return cost_of_delay / max(job_size, 1)
    
    def _calculate_ice(self, item: Dict) -> float:
        """Calculate Impact, Confidence, Ease score"""
        impact = item.get("impact", 5)  # 1-10
        confidence = self._get_confidence(item)
        ease = 11 - item.get("effort", 5)  # Inverse of effort
        
        return impact * confidence * ease
    
    def _calculate_technical_debt_score(self, item: Dict) -> float:
        """Calculate technical debt specific scoring"""
        if item.get("category") != "technical-debt":
            return 0
        
        debt_impact = item.get("impact", 5) * 10
        debt_interest = self._calculate_debt_interest(item)
        hotspot_multiplier = self._get_hotspot_multiplier(item)
        
        return (debt_impact + debt_interest) * hotspot_multiplier
    
    def generate_backlog(self, scored_items: List[Dict]) -> str:
        """Generate markdown backlog with prioritized items"""
        now = datetime.now()
        
        # Select next best item
        next_item = scored_items[0] if scored_items else None
        
        # Generate backlog markdown
        backlog_md = f"""# 📊 Autonomous Value Backlog

Last Updated: {now.isoformat()}
Repository Maturity: {self.config.get('repository', {}).get('maturity', 'unknown')}

## 🎯 Next Best Value Item
"""
        
        if next_item:
            scores = next_item["scores"]
            backlog_md += f"""**[{next_item['id'].upper()}] {next_item['title']}**
- **Composite Score**: {scores['composite']:.1f}
- **WSJF**: {scores['wsjf']:.1f} | **ICE**: {scores['ice']:.0f} | **Tech Debt**: {scores['technicalDebt']:.0f}
- **Estimated Effort**: {next_item.get('effort', 'Unknown')} hours
- **Category**: {next_item['category'].title()}
- **Expected Impact**: {next_item.get('description', 'No description')}

"""
        
        backlog_md += """## 📋 Top 10 Backlog Items

| Rank | ID | Title | Score | Category | Est. Hours | Priority |
|------|-----|--------|---------|----------|------------|----------|
"""
        
        for i, item in enumerate(scored_items[:10], 1):
            score = item["scores"]["composite"]
            category = item["category"].title()
            effort = item.get("effort", "?")
            priority = item.get("priority", "medium")
            title = item["title"][:50] + "..." if len(item["title"]) > 50 else item["title"]
            
            backlog_md += f"| {i} | {item['id']} | {title} | {score:.1f} | {category} | {effort} | {priority} |\n"
        
        # Add metrics section
        backlog_md += f"""

## 📈 Value Discovery Metrics
- **Total Items Discovered**: {len(scored_items)}
- **High Priority Items**: {len([i for i in scored_items if i.get('priority') == 'high'])}
- **Security Items**: {len([i for i in scored_items if i['category'] == 'security'])}
- **Technical Debt Items**: {len([i for i in scored_items if i['category'] == 'technical-debt'])}
- **Infrastructure Items**: {len([i for i in scored_items if i['category'] == 'infrastructure'])}

## 🔄 Discovery Sources
- Code Analysis (TODO/FIXME): {len([i for i in scored_items if i.get('type') == 'code-improvement'])}
- Security Scans: {len([i for i in scored_items if i['category'] == 'security'])}
- Dependency Analysis: {len([i for i in scored_items if i['category'] == 'dependencies'])}
- Documentation Gaps: {len([i for i in scored_items if i['category'] == 'documentation'])}
- Infrastructure Gaps: {len([i for i in scored_items if i['category'] == 'infrastructure'])}

---
*Generated by Terragon Autonomous SDLC Enhancement System*
"""
        
        return backlog_md
    
    def save_backlog(self, backlog_content: str):
        """Save the generated backlog to file"""
        with open(self.backlog_path, 'w') as f:
            f.write(backlog_content)
    
    def save_metrics(self):
        """Save updated metrics to file"""
        self.metrics_path.parent.mkdir(exist_ok=True)
        with open(self.metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    # Helper methods for scoring calculations
    def _estimate_effort(self, comment: str) -> int:
        """Estimate effort based on comment complexity"""
        if any(word in comment.lower() for word in ["simple", "quick", "easy"]):
            return 1
        elif any(word in comment.lower() for word in ["refactor", "rewrite", "major"]):
            return 8
        else:
            return 3
    
    def _estimate_impact(self, comment: str, category: str) -> int:
        """Estimate impact based on comment content and category"""
        if category == "security":
            return 9
        elif any(word in comment.lower() for word in ["performance", "critical", "bug"]):
            return 8
        elif any(word in comment.lower() for word in ["improvement", "optimize"]):
            return 6
        else:
            return 4
    
    def _get_time_criticality(self, item: Dict) -> float:
        """Calculate time criticality component"""
        if item["category"] == "security":
            return 8.0
        elif item["category"] == "infrastructure":
            return 6.0
        else:
            return 3.0
    
    def _get_risk_reduction(self, item: Dict) -> float:
        """Calculate risk reduction component"""
        if item["category"] == "security":
            return 9.0
        elif item.get("type") == "ci-cd-setup":
            return 7.0
        else:
            return 2.0
    
    def _get_opportunity_enablement(self, item: Dict) -> float:
        """Calculate opportunity enablement component"""
        if item["category"] == "infrastructure":
            return 8.0
        elif item["category"] == "documentation":
            return 4.0
        else:
            return 3.0
    
    def _get_confidence(self, item: Dict) -> float:
        """Get confidence level for item execution"""
        if item["category"] in ["documentation", "security"]:
            return 9.0
        elif item["category"] == "infrastructure":
            return 7.0
        else:
            return 6.0
    
    def _calculate_debt_interest(self, item: Dict) -> float:
        """Calculate future cost if technical debt not addressed"""
        return item.get("effort", 3) * 0.5  # Debt grows at 50% per cycle
    
    def _get_hotspot_multiplier(self, item: Dict) -> float:
        """Get hotspot multiplier based on file activity"""
        # In a full implementation, this would analyze git log for file churn
        return 1.5  # Default multiplier
    
    def _normalize_score(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize score to 0-1 range"""
        if max_val == min_val:
            return 0.5
        return (value - min_val) / (max_val - min_val)
    
    def _run_flake8_analysis(self) -> List[Dict]:
        """Run flake8 and extract issues"""
        # Placeholder - would run actual flake8 analysis
        return []
    
    def _run_mypy_analysis(self) -> List[Dict]:
        """Run mypy and extract type issues"""
        # Placeholder - would run actual mypy analysis
        return []
    
    def _run_bandit_analysis(self) -> List[Dict]:
        """Run bandit security analysis"""
        # Placeholder - would run actual bandit analysis
        return []


def main():
    """Main execution function"""
    engine = ValueDiscoveryEngine()
    
    print("🔍 Discovering value items...")
    items = engine.discover_value_items()
    
    print(f"📊 Scoring {len(items)} items...")
    scored_items = engine.calculate_scores(items)
    
    print("📝 Generating backlog...")
    backlog_content = engine.generate_backlog(scored_items)
    engine.save_backlog(backlog_content)
    
    print("💾 Saving metrics...")
    engine.save_metrics()
    
    print(f"✅ Complete! Generated backlog with {len(scored_items)} items")
    if scored_items:
        next_item = scored_items[0]
        print(f"🎯 Next best value item: {next_item['title']} (Score: {next_item['scores']['composite']:.1f})")


if __name__ == "__main__":
    main()