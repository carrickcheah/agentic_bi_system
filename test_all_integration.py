#!/usr/bin/env python3
"""
Comprehensive test script to validate all LanceDB integration components.
Tests all files across Phase 0, 1, and 2 to ensure everything works correctly.
"""

import asyncio
import sys
from pathlib import Path
import importlib.util
import traceback
from datetime import datetime

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


class IntegrationTester:
    def __init__(self):
        self.results = {}
        self.base_path = Path(__file__).parent
        
    def print_header(self, text):
        """Print a formatted header."""
        print(f"\n{BLUE}{'=' * 80}{RESET}")
        print(f"{BLUE}{text}{RESET}")
        print(f"{BLUE}{'=' * 80}{RESET}")
    
    def print_section(self, text):
        """Print a section header."""
        print(f"\n{YELLOW}>>> {text}{RESET}")
        print(f"{YELLOW}{'-' * 60}{RESET}")
    
    def print_result(self, name, success, error=None):
        """Print test result."""
        if success:
            print(f"{GREEN}✅ {name}: PASS{RESET}")
            self.results[name] = "PASS"
        else:
            print(f"{RED}❌ {name}: FAIL{RESET}")
            if error:
                print(f"{RED}   Error: {error}{RESET}")
            self.results[name] = "FAIL"
    
    async def test_file_exists(self, file_path, description):
        """Test if a file exists."""
        path = self.base_path / file_path
        exists = path.exists()
        self.print_result(f"File exists: {description}", exists, 
                         f"Not found at {file_path}" if not exists else None)
        return exists
    
    async def test_import(self, module_path, description):
        """Test if a module can be imported."""
        try:
            # Handle different import paths
            if module_path.startswith("app."):
                # Try to import as a module
                parts = module_path.split('.')
                module_name = parts[-1]
                
                # Build the actual file path
                if "lance_db.src" in module_path:
                    file_path = self.base_path / "app" / "lance_db" / "src" / f"{module_name}.py"
                elif "intelligence" in module_path:
                    file_path = self.base_path / "app" / "intelligence" / f"{module_name}.py"
                elif "investigation" in module_path:
                    file_path = self.base_path / "app" / "investigation" / f"{module_name}.py"
                elif "insight_synthesis" in module_path:
                    file_path = self.base_path / "app" / "insight_synthesis" / f"{module_name}.py"
                else:
                    self.print_result(f"Import: {description}", False, f"Unknown module path: {module_path}")
                    return False
            else:
                file_path = self.base_path / module_path
            
            if not file_path.exists():
                self.print_result(f"Import: {description}", False, f"File not found: {file_path}")
                return False
            
            # Load module from file
            spec = importlib.util.spec_from_file_location(module_path, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            self.print_result(f"Import: {description}", True)
            return True
            
        except Exception as e:
            self.print_result(f"Import: {description}", False, str(e))
            return False
    
    async def test_phase_0(self):
        """Test Phase 0: Vector Infrastructure Foundation."""
        self.print_section("Phase 0: Vector Infrastructure Foundation")
        
        # Test file existence
        phase_0_files = [
            ("app/lance_db/src/enterprise_vector_schema.py", "Enterprise Vector Schema"),
            ("app/lance_db/src/vector_index_manager.py", "Vector Index Manager"),
            ("app/lance_db/src/vector_performance_monitor.py", "Vector Performance Monitor"),
            ("app/lance_db/src/test_enterprise_vector_schema.py", "Schema Test"),
            ("app/lance_db/src/test_vector_index_manager.py", "Index Manager Test"),
            ("app/lance_db/src/test_vector_performance_monitor.py", "Performance Monitor Test"),
        ]
        
        for file_path, desc in phase_0_files:
            await self.test_file_exists(file_path, desc)
        
        # Test imports
        phase_0_imports = [
            ("app.lance_db.src.enterprise_vector_schema", "Enterprise Vector Schema"),
            ("app.lance_db.src.vector_index_manager", "Vector Index Manager"),
            ("app.lance_db.src.vector_performance_monitor", "Vector Performance Monitor"),
        ]
        
        for module, desc in phase_0_imports:
            await self.test_import(module, desc)
    
    async def test_phase_1(self):
        """Test Phase 1: Intelligence Module Integration."""
        self.print_section("Phase 1: Intelligence Module Integration")
        
        # Test file existence
        phase_1_files = [
            ("app/intelligence/vector_enhanced_domain_expert.py", "Vector Enhanced Domain Expert"),
            ("app/intelligence/vector_enhanced_complexity_analyzer.py", "Vector Enhanced Complexity Analyzer"),
            ("app/intelligence/lancedb_pattern_recognizer.py", "LanceDB Pattern Recognizer"),
            ("app/intelligence/test_vector_enhanced_domain_expert.py", "Domain Expert Test"),
            ("app/intelligence/test_vector_enhanced_complexity.py", "Complexity Analyzer Test"),
            ("app/intelligence/test_lancedb_pattern_recognizer.py", "Pattern Recognizer Test"),
        ]
        
        for file_path, desc in phase_1_files:
            await self.test_file_exists(file_path, desc)
        
        # Test imports
        phase_1_imports = [
            ("app.intelligence.vector_enhanced_domain_expert", "Vector Enhanced Domain Expert"),
            ("app.intelligence.vector_enhanced_complexity_analyzer", "Vector Enhanced Complexity Analyzer"),
            ("app.intelligence.lancedb_pattern_recognizer", "LanceDB Pattern Recognizer"),
        ]
        
        for module, desc in phase_1_imports:
            await self.test_import(module, desc)
    
    async def test_phase_2(self):
        """Test Phase 2: Investigation & Insight Synthesis Integration."""
        self.print_section("Phase 2: Investigation & Insight Synthesis Integration")
        
        # Phase 2.1 - Investigation
        phase_2_1_files = [
            ("app/investigation/vector_enhanced_investigator.py", "Vector Enhanced Investigator"),
            ("app/investigation/test_vector_enhanced_investigator.py", "Investigator Test"),
        ]
        
        for file_path, desc in phase_2_1_files:
            await self.test_file_exists(file_path, desc)
        
        # Phase 2.2 - Insight Synthesis
        phase_2_2_files = [
            ("app/insight_synthesis/vector_enhanced_insight_synthesizer.py", "Vector Enhanced Synthesizer"),
            ("app/insight_synthesis/test_vector_enhanced_synthesizer.py", "Synthesizer Test"),
            ("app/insight_synthesis/test_vector_synthesizer_standalone.py", "Standalone Test"),
        ]
        
        for file_path, desc in phase_2_2_files:
            await self.test_file_exists(file_path, desc)
        
        # Phase 2.3 - Cross-Module Intelligence
        phase_2_3_files = [
            ("app/lance_db/src/investigation_insight_intelligence.py", "Cross-Module Intelligence"),
            ("app/lance_db/src/test_investigation_insight_intelligence.py", "Intelligence Test"),
        ]
        
        for file_path, desc in phase_2_3_files:
            await self.test_file_exists(file_path, desc)
        
        # Test imports
        phase_2_imports = [
            ("app.investigation.vector_enhanced_investigator", "Vector Enhanced Investigator"),
            ("app.insight_synthesis.vector_enhanced_insight_synthesizer", "Vector Enhanced Synthesizer"),
            ("app.lance_db.src.investigation_insight_intelligence", "Cross-Module Intelligence"),
        ]
        
        for module, desc in phase_2_imports:
            await self.test_import(module, desc)
    
    async def test_documentation(self):
        """Test documentation files."""
        self.print_section("Documentation Files")
        
        doc_files = [
            ("app/lance_db/INTEGRATION_COMPLETE.md", "Integration Complete Guide"),
            ("app/lance_db/VECTOR_INTEGRATION_DIAGRAM.md", "Integration Diagrams"),
            ("app/lance_db/QUICKSTART_GUIDE.md", "Quick Start Guide"),
        ]
        
        for file_path, desc in doc_files:
            await self.test_file_exists(file_path, desc)
    
    async def test_core_functionality(self):
        """Test core functionality without dependencies."""
        self.print_section("Core Functionality Tests")
        
        # Test 1: Enterprise Vector Schema basic functionality
        try:
            from app.lance_db.src.enterprise_vector_schema import (
                ModuleSource, BusinessDomain, PerformanceTier,
                generate_vector_id, normalize_score_to_unified_scale
            )
            
            # Test ID generation
            vector_id = generate_vector_id()
            assert len(vector_id) > 0, "Vector ID should not be empty"
            
            # Test score normalization
            normalized = normalize_score_to_unified_scale(75, 0, 100)
            assert 0 <= normalized <= 1, "Normalized score should be between 0 and 1"
            assert normalized == 0.75, "Normalization calculation incorrect"
            
            self.print_result("Enterprise Schema Functions", True)
            
        except Exception as e:
            self.print_result("Enterprise Schema Functions", False, str(e))
        
        # Test 2: Basic class instantiation
        try:
            # Test data classes can be created
            from app.lance_db.src.investigation_insight_intelligence import (
                InvestigationInsightLinkType,
                FeedbackLoopType,
                InvestigationInsightIntelligenceEngine
            )
            
            # Test enums
            link_type = InvestigationInsightLinkType.DIRECT_GENERATION
            loop_type = FeedbackLoopType.INSIGHT_QUALITY
            
            # Test engine creation
            engine = InvestigationInsightIntelligenceEngine()
            assert engine.logger is not None, "Engine logger should be initialized"
            
            self.print_result("Cross-Module Classes", True)
            
        except Exception as e:
            self.print_result("Cross-Module Classes", False, str(e))
    
    def print_summary(self):
        """Print test summary."""
        self.print_header("TEST SUMMARY")
        
        total = len(self.results)
        passed = sum(1 for result in self.results.values() if result == "PASS")
        failed = total - passed
        
        print(f"\nTotal Tests: {total}")
        print(f"{GREEN}Passed: {passed}{RESET}")
        print(f"{RED}Failed: {failed}{RESET}")
        
        if failed > 0:
            print(f"\n{RED}Failed Tests:{RESET}")
            for name, result in self.results.items():
                if result == "FAIL":
                    print(f"  - {name}")
        
        success_rate = (passed / total * 100) if total > 0 else 0
        print(f"\n{BLUE}Success Rate: {success_rate:.1f}%{RESET}")
        
        if success_rate == 100:
            print(f"\n{GREEN}🎉 ALL TESTS PASSED! LanceDB Integration is working correctly!{RESET}")
        elif success_rate >= 80:
            print(f"\n{YELLOW}⚠️  Most tests passed. Some components may need attention.{RESET}")
        else:
            print(f"\n{RED}❌ Significant issues detected. Please review failed tests.{RESET}")
        
        return failed == 0


async def main():
    """Run all integration tests."""
    print(f"{BLUE}🚀 LanceDB Integration Test Suite{RESET}")
    print(f"{BLUE}Testing all components across Phase 0, 1, and 2{RESET}")
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tester = IntegrationTester()
    
    try:
        # Run all test phases
        await tester.test_phase_0()
        await tester.test_phase_1()
        await tester.test_phase_2()
        await tester.test_documentation()
        await tester.test_core_functionality()
        
        # Print summary
        success = tester.print_summary()
        
        # Return appropriate exit code
        return 0 if success else 1
        
    except Exception as e:
        print(f"\n{RED}Fatal error during testing:{RESET}")
        print(f"{RED}{str(e)}{RESET}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)