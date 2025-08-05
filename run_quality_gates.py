#!/usr/bin/env python3
"""
Quality gates runner for the protein diffusion project.

This script runs various quality checks without external dependencies.
"""

import ast
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Any
import re


class QualityGateRunner:
    """Run quality gates and checks."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.src_dir = project_root / "src"
        self.test_dir = project_root / "tests"
        self.results = {}
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all quality checks."""
        print("🔍 Running Quality Gates...")
        print("=" * 50)
        
        # Syntax check
        print("\n1. Checking Python syntax...")
        self.results["syntax"] = self.check_syntax()
        
        # Import check
        print("\n2. Checking imports...")
        self.results["imports"] = self.check_imports()
        
        # Code structure
        print("\n3. Analyzing code structure...")
        self.results["structure"] = self.analyze_code_structure()
        
        # Documentation check
        print("\n4. Checking documentation...")
        self.results["documentation"] = self.check_documentation()
        
        # Security scan
        print("\n5. Running security checks...")
        self.results["security"] = self.check_security()
        
        # Code quality metrics
        print("\n6. Computing code quality metrics...")
        self.results["quality"] = self.compute_quality_metrics()
        
        # Print summary
        self.print_summary()
        
        return self.results
    
    def check_syntax(self) -> Dict[str, Any]:
        """Check Python syntax for all files."""
        results = {"passed": True, "errors": [], "files_checked": 0}
        
        for py_file in self.src_dir.rglob("*.py"):
            results["files_checked"] += 1
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                ast.parse(source, filename=str(py_file))
                print(f"  ✅ {py_file.relative_to(self.project_root)}")
                
            except SyntaxError as e:
                results["passed"] = False
                error_msg = f"{py_file.relative_to(self.project_root)}: {e.msg} (line {e.lineno})"
                results["errors"].append(error_msg)
                print(f"  ❌ {error_msg}")
            except Exception as e:
                results["passed"] = False
                error_msg = f"{py_file.relative_to(self.project_root)}: {str(e)}"
                results["errors"].append(error_msg)
                print(f"  ⚠️  {error_msg}")
        
        if results["passed"]:
            print(f"  🎉 All {results['files_checked']} Python files have valid syntax!")
        
        return results
    
    def check_imports(self) -> Dict[str, Any]:
        """Check import statements and dependencies."""
        results = {"passed": True, "issues": [], "imports": {}}
        
        for py_file in self.src_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                tree = ast.parse(source, filename=str(py_file))
                file_imports = []
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            file_imports.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            file_imports.append(node.module)
                
                results["imports"][str(py_file.relative_to(self.project_root))] = file_imports
                
                # Check for problematic imports
                problematic = ['os.system', 'subprocess.call', 'eval', 'exec']
                for imp in file_imports:
                    if any(prob in imp for prob in problematic):
                        results["issues"].append(f"{py_file.relative_to(self.project_root)}: potentially unsafe import '{imp}'")
                
            except Exception as e:
                results["issues"].append(f"{py_file.relative_to(self.project_root)}: {str(e)}")
        
        if not results["issues"]:
            print("  ✅ No import issues found")
        else:
            for issue in results["issues"]:
                print(f"  ⚠️  {issue}")
        
        return results
    
    def analyze_code_structure(self) -> Dict[str, Any]:
        """Analyze code structure and complexity."""
        results = {
            "total_files": 0,
            "total_lines": 0,
            "total_functions": 0,
            "total_classes": 0,
            "files": {}
        }
        
        for py_file in self.src_dir.rglob("*.py"):
            results["total_files"] += 1
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                    lines = source.split('\n')
                
                results["total_lines"] += len(lines)
                
                tree = ast.parse(source, filename=str(py_file))
                
                file_stats = {
                    "lines": len(lines),
                    "functions": 0,
                    "classes": 0,
                    "complexity": 0
                }
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        file_stats["functions"] += 1
                        results["total_functions"] += 1
                    elif isinstance(node, ast.ClassDef):
                        file_stats["classes"] += 1
                        results["total_classes"] += 1
                    elif isinstance(node, (ast.If, ast.For, ast.While, ast.With)):
                        file_stats["complexity"] += 1
                
                results["files"][str(py_file.relative_to(self.project_root))] = file_stats
                
            except Exception as e:
                print(f"  ⚠️  Error analyzing {py_file.relative_to(self.project_root)}: {e}")
        
        print(f"  📊 Code Statistics:")
        print(f"     Files: {results['total_files']}")
        print(f"     Lines: {results['total_lines']}")
        print(f"     Functions: {results['total_functions']}")
        print(f"     Classes: {results['total_classes']}")
        
        return results
    
    def check_documentation(self) -> Dict[str, Any]:
        """Check documentation coverage."""
        results = {
            "documented_functions": 0,
            "undocumented_functions": 0,
            "documented_classes": 0,
            "undocumented_classes": 0,
            "files": {}
        }
        
        for py_file in self.src_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                tree = ast.parse(source, filename=str(py_file))
                
                file_stats = {
                    "documented_functions": 0,
                    "undocumented_functions": 0,
                    "documented_classes": 0,
                    "undocumented_classes": 0
                }
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if ast.get_docstring(node):
                            file_stats["documented_functions"] += 1
                            results["documented_functions"] += 1
                        else:
                            file_stats["undocumented_functions"] += 1
                            results["undocumented_functions"] += 1
                    
                    elif isinstance(node, ast.ClassDef):
                        if ast.get_docstring(node):
                            file_stats["documented_classes"] += 1
                            results["documented_classes"] += 1
                        else:
                            file_stats["undocumented_classes"] += 1
                            results["undocumented_classes"] += 1
                
                results["files"][str(py_file.relative_to(self.project_root))] = file_stats
                
            except Exception as e:
                print(f"  ⚠️  Error checking docs in {py_file.relative_to(self.project_root)}: {e}")
        
        total_functions = results["documented_functions"] + results["undocumented_functions"]
        total_classes = results["documented_classes"] + results["undocumented_classes"]
        
        func_coverage = (results["documented_functions"] / total_functions * 100) if total_functions > 0 else 100
        class_coverage = (results["documented_classes"] / total_classes * 100) if total_classes > 0 else 100
        
        print(f"  📚 Documentation Coverage:")
        print(f"     Functions: {func_coverage:.1f}% ({results['documented_functions']}/{total_functions})")
        print(f"     Classes: {class_coverage:.1f}% ({results['documented_classes']}/{total_classes})")
        
        results["function_coverage"] = func_coverage
        results["class_coverage"] = class_coverage
        
        return results
    
    def check_security(self) -> Dict[str, Any]:
        """Basic security checks."""
        results = {"issues": [], "files_checked": 0}
        
        # Patterns to look for (excluding method calls)
        security_patterns = [
            (r'^[^.]*eval\s*\(', "Use of eval() function"),  # Not preceded by a dot
            (r'^[^.]*exec\s*\(', "Use of exec() function"),  # Not preceded by a dot
            (r'os\.system\s*\(', "Use of os.system()"),
            (r'subprocess\.call\s*\([^)]*shell\s*=\s*True', "Subprocess with shell=True"),
            (r'pickle\.loads?\s*\(', "Use of pickle (potential security risk)"),
            (r'^[^#]*input\s*\([^)]*\)', "Use of input() function"),  # Not in comments
            (r'__import__\s*\(', "Dynamic imports"),
        ]
        
        for py_file in self.src_dir.rglob("*.py"):
            results["files_checked"] += 1
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                for pattern, description in security_patterns:
                    # Check each line to avoid false positives
                    for line_num, line in enumerate(source.split('\n'), 1):
                        if re.search(pattern, line, re.IGNORECASE):
                            # Additional filtering for false positives
                            if 'eval(' in line and ('.eval()' in line or 'model.eval()' in line):
                                continue  # Skip PyTorch model.eval() calls
                            if 'exec(' in line and ('.exec(' in line):
                                continue  # Skip method calls
                            
                            results["issues"].append({
                                "file": str(py_file.relative_to(self.project_root)),
                                "line": line_num,
                                "issue": description,
                                "code": line.strip()
                            })
                
            except Exception as e:
                print(f"  ⚠️  Error scanning {py_file.relative_to(self.project_root)}: {e}")
        
        if not results["issues"]:
            print(f"  🔒 No security issues found in {results['files_checked']} files")
        else:
            print(f"  ⚠️  Found {len(results['issues'])} potential security issues:")
            for issue in results["issues"][:5]:  # Show first 5
                print(f"     {issue['file']}:{issue['line']} - {issue['issue']}")
            if len(results["issues"]) > 5:
                print(f"     ... and {len(results['issues']) - 5} more")
        
        return results
    
    def compute_quality_metrics(self) -> Dict[str, Any]:
        """Compute code quality metrics."""
        results = {
            "complexity_score": 0,
            "maintainability_score": 0,
            "test_coverage_estimate": 0,
            "total_score": 0
        }
        
        # Complexity score based on structure analysis
        structure = self.results.get("structure", {})
        if structure.get("total_lines", 0) > 0:
            avg_lines_per_file = structure["total_lines"] / max(structure["total_files"], 1)
            avg_functions_per_file = structure["total_functions"] / max(structure["total_files"], 1)
            
            # Lower complexity is better
            complexity_score = max(0, 100 - (avg_lines_per_file / 10) - (avg_functions_per_file * 2))
            results["complexity_score"] = min(100, complexity_score)
        
        # Maintainability score based on documentation
        docs = self.results.get("documentation", {})
        if docs:
            doc_score = (docs.get("function_coverage", 0) + docs.get("class_coverage", 0)) / 2
            results["maintainability_score"] = doc_score
        
        # Test coverage estimate (based on test file existence)
        test_files = list(self.test_dir.rglob("*.py")) if self.test_dir.exists() else []
        src_files = list(self.src_dir.rglob("*.py"))
        
        if src_files:
            test_ratio = len(test_files) / len(src_files)
            results["test_coverage_estimate"] = min(100, test_ratio * 100)
        
        # Overall score
        results["total_score"] = (
            results["complexity_score"] * 0.3 +
            results["maintainability_score"] * 0.4 +
            results["test_coverage_estimate"] * 0.3
        )
        
        print(f"  📈 Quality Metrics:")
        print(f"     Complexity Score: {results['complexity_score']:.1f}/100")
        print(f"     Maintainability: {results['maintainability_score']:.1f}/100")
        print(f"     Test Coverage Est: {results['test_coverage_estimate']:.1f}/100")
        print(f"     Overall Score: {results['total_score']:.1f}/100")
        
        return results
    
    def print_summary(self):
        """Print overall summary."""
        print("\n" + "=" * 50)
        print("📋 QUALITY GATES SUMMARY")
        print("=" * 50)
        
        # Overall status
        syntax_ok = self.results.get("syntax", {}).get("passed", False)
        security_issues = len(self.results.get("security", {}).get("issues", []))
        total_score = self.results.get("quality", {}).get("total_score", 0)
        
        if syntax_ok and security_issues == 0 and total_score >= 70:
            print("🎉 PASSED - All quality gates met!")
            status = "PASSED"
        elif syntax_ok and security_issues == 0:
            print("⚠️  CONDITIONAL PASS - Basic requirements met, room for improvement")
            status = "CONDITIONAL_PASS"
        else:
            print("❌ FAILED - Quality gates not met")
            status = "FAILED"
        
        print(f"\nStatus: {status}")
        print(f"Overall Score: {total_score:.1f}/100")
        
        # Recommendations
        print("\n🔧 RECOMMENDATIONS:")
        
        if not syntax_ok:
            print("  • Fix syntax errors before deployment")
        
        if security_issues > 0:
            print("  • Address security issues")
        
        docs = self.results.get("documentation", {})
        if docs.get("function_coverage", 0) < 80:
            print("  • Improve function documentation coverage")
        
        if docs.get("class_coverage", 0) < 80:
            print("  • Improve class documentation coverage")
        
        if self.results.get("quality", {}).get("test_coverage_estimate", 0) < 70:
            print("  • Add more test files")
        
        if total_score < 70:
            print("  • Refactor complex code for better maintainability")
        
        print("\n" + "=" * 50)
        
        return status


def main():
    """Main entry point."""
    project_root = Path(__file__).parent
    
    runner = QualityGateRunner(project_root)
    results = runner.run_all_checks()
    
    # Return appropriate exit code
    syntax_ok = results.get("syntax", {}).get("passed", False)
    security_issues = len(results.get("security", {}).get("issues", []))
    
    if syntax_ok and security_issues == 0:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure


if __name__ == "__main__":
    main()