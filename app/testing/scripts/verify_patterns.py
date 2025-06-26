#!/usr/bin/env python3
"""
Simple Pattern Verification Script

Verifies that all 150 manufacturing patterns are correctly structured
and can be loaded from JSON files.

Usage:
    python testing/scripts/verify_patterns.py
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

# Add app to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))


def verify_pattern_files():
    """Verify all pattern JSON files are valid and well-structured."""
    
    print("🔍 Verifying Business Intelligence Pattern Library...")
    print("=" * 60)
    
    # Pattern file organization by business domain
    pattern_files = {
        # Manufacturing domains
        "production": "production_operations.json",
        "quality": "quality_management.json", 
        "supply_chain": "supply_chain_inventory.json",
        "cost_management": "cost_management.json",
        "asset_management": "asset_equipment.json",
        "safety": "safety_compliance.json",
        "customer": "customer_demand.json",
        "planning": "planning_scheduling.json",
        "hr": "hr_workforce.json",
        # Business domains
        "sales": "sales_revenue.json",
        "product": "product_management.json",
        "marketing": "marketing_campaigns.json",
        "operations": "operations_efficiency.json",
        "finance": "finance_budgeting.json"
    }
    
    expected_counts = {
        # Manufacturing domains
        "production": 30,
        "quality": 25,
        "supply_chain": 25,
        "cost_management": 20,
        "asset_management": 15,
        "safety": 10,
        "customer": 10,
        "planning": 10,
        "hr": 5,
        # Business domains
        "sales": 20,
        "product": 15,
        "marketing": 15,
        "operations": 10,
        "finance": 10
    }
    
    pattern_dir = Path(__file__).parent.parent.parent / "app" / "data" / "patterns"
    
    total_patterns = 0
    domain_stats = defaultdict(int)
    complexity_stats = defaultdict(int)
    success_rate_stats = defaultdict(int)
    
    required_fields = [
        "information", "metadata"
    ]
    
    required_metadata_fields = [
        "pattern", "user_roles", "business_domain", "timeframe", 
        "complexity", "success_rate", "confidence_indicators", 
        "expected_deliverables", "data_source", "sample_size"
    ]
    
    all_patterns = {}
    validation_errors = []
    
    for domain, filename in pattern_files.items():
        file_path = pattern_dir / filename
        
        print(f"\n📂 Loading {filename}...")
        
        if not file_path.exists():
            validation_errors.append(f"❌ File not found: {filename}")
            continue
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                domain_patterns = json.load(f)
            
            # Validate pattern count
            actual_count = len(domain_patterns)
            expected_count = expected_counts[domain]
            
            if actual_count == expected_count:
                print(f"   ✅ Pattern count: {actual_count}/{expected_count}")
            else:
                print(f"   ⚠️  Pattern count: {actual_count}/{expected_count}")
                validation_errors.append(f"Pattern count mismatch in {filename}: {actual_count}/{expected_count}")
            
            # Validate pattern structure
            valid_patterns = 0
            for i, pattern in enumerate(domain_patterns, 1):
                pattern_id = f"{domain}_{i:03d}"
                
                # Check required fields
                if all(field in pattern for field in required_fields):
                    metadata = pattern.get("metadata", {})
                    if all(field in metadata for field in required_metadata_fields):
                        valid_patterns += 1
                        
                        # Add pattern to collection with generated ID
                        pattern["pattern_id"] = pattern_id
                        all_patterns[pattern_id] = pattern
                        
                        # Collect statistics
                        domain_stats[metadata["business_domain"]] += 1
                        complexity_stats[metadata["complexity"]] += 1
                        
                        # Group success rates
                        success_rate = metadata["success_rate"]
                        if success_rate >= 0.6:
                            success_rate_stats["high (≥0.6)"] += 1
                        elif success_rate >= 0.5:
                            success_rate_stats["moderate (0.5-0.6)"] += 1
                        else:
                            success_rate_stats["low (<0.5)"] += 1
                        
                    else:
                        missing_fields = [field for field in required_metadata_fields if field not in metadata]
                        validation_errors.append(f"Pattern {pattern_id} missing metadata fields: {missing_fields}")
                else:
                    missing_fields = [field for field in required_fields if field not in pattern]
                    validation_errors.append(f"Pattern {pattern_id} missing required fields: {missing_fields}")
            
            print(f"   📊 Valid patterns: {valid_patterns}/{actual_count}")
            total_patterns += valid_patterns
            
        except json.JSONDecodeError as e:
            validation_errors.append(f"❌ JSON parsing error in {filename}: {e}")
        except Exception as e:
            validation_errors.append(f"❌ Error loading {filename}: {e}")
    
    # Print comprehensive summary
    print("\n" + "=" * 60)
    print("📊 PATTERN LIBRARY VERIFICATION SUMMARY")
    print("=" * 60)
    
    print(f"Total Patterns Loaded: {total_patterns}/220")
    print(f"Validation Errors: {len(validation_errors)}")
    
    if len(validation_errors) == 0:
        print("✅ All patterns loaded successfully!")
    else:
        print("\n❌ Validation Errors:")
        for error in validation_errors:
            print(f"   {error}")
    
    # Domain distribution
    print(f"\n📈 Business Domain Distribution:")
    for domain, count in sorted(domain_stats.items()):
        print(f"   {domain:15}: {count:3d} patterns")
    
    # Complexity distribution
    print(f"\n🎯 Complexity Distribution:")
    for complexity, count in sorted(complexity_stats.items()):
        print(f"   {complexity:10}: {count:3d} patterns")
    
    # Success rate distribution
    print(f"\n📊 Success Rate Distribution:")
    for rate_range, count in sorted(success_rate_stats.items()):
        print(f"   {rate_range:15}: {count:3d} patterns")
    
    # Sample patterns
    print(f"\n🔍 Sample Patterns by Domain:")
    domains_shown = set()
    for pattern_id, pattern in list(all_patterns.items())[:5]:
        domain = pattern["metadata"]["business_domain"]
        if domain not in domains_shown:
            print(f"   [{domain}] {pattern['information'][:60]}...")
            domains_shown.add(domain)
    
    # Bootstrap success rate analysis
    print(f"\n🎲 Bootstrap Success Rate Analysis:")
    bootstrap_patterns = [p for p in all_patterns.values() 
                         if p["metadata"]["data_source"] == "bootstrap_estimate"]
    
    if bootstrap_patterns:
        success_rates = [p["metadata"]["success_rate"] for p in bootstrap_patterns]
        avg_success_rate = sum(success_rates) / len(success_rates)
        min_rate = min(success_rates)
        max_rate = max(success_rates)
        
        print(f"   Bootstrap patterns: {len(bootstrap_patterns)}")
        print(f"   Average success rate: {avg_success_rate:.3f}")
        print(f"   Success rate range: {min_rate:.3f} - {max_rate:.3f}")
    
    print("\n" + "=" * 60)
    
    # Pattern matching preview
    if total_patterns > 0:
        print("🔍 Pattern Matching Preview:")
        sample_queries = [
            "What is our daily production efficiency?",
            "How can we reduce defect rates?", 
            "What are our inventory optimization opportunities?"
        ]
        
        for query in sample_queries:
            print(f"\n   Query: '{query}'")
            # Simple keyword matching for preview
            keywords = query.lower().split()
            
            matches = []
            for pattern_id, pattern in all_patterns.items():
                info = pattern["information"].lower()
                indicators = [ind.lower() for ind in pattern["metadata"]["confidence_indicators"]]
                
                score = 0
                for keyword in keywords:
                    if any(keyword in text for text in [info] + indicators):
                        score += 1
                
                if score > 0:
                    matches.append((pattern_id, pattern, score))
            
            # Sort by score and show top match
            matches.sort(key=lambda x: x[2], reverse=True)
            if matches:
                best_match = matches[0]
                print(f"   → Best match: {best_match[1]['information'][:50]}...")
                print(f"   → Domain: {best_match[1]['metadata']['business_domain']}")
                print(f"   → Success rate: {best_match[1]['metadata']['success_rate']:.3f}")
            else:
                print("   → No matches found")
    
    return len(validation_errors) == 0, total_patterns


def main():
    """Main verification function."""
    print("🚀 Manufacturing Pattern Library Verification")
    print("   Checking 220 business intelligence patterns (manufacturing + business domains)...")
    
    success, pattern_count = verify_pattern_files()
    
    if success and pattern_count == 220:
        print("\n🎉 SUCCESS: All 220 patterns verified and ready!")
        print("\n📝 Pattern Library Status:")
        print("   ✅ JSON structure validation passed")
        print("   ✅ Metadata schema compliance verified")
        print("   ✅ Bootstrap success rates configured")
        print("   ✅ Business domain coverage complete")
        print("   ✅ Ready for Qdrant indexing and MCP integration")
        
        print("\n🚀 Next Steps:")
        print("   1. Run full test suite: python testing/scripts/test_pattern_library.py")
        print("   2. Initialize Qdrant collection and index patterns")
        print("   3. Create PostgreSQL tables for success tracking")
        print("   4. Integrate with QueryProcessor for pattern-aware processing")
        
        return 0
    else:
        print(f"\n❌ VALIDATION FAILED: {pattern_count}/220 patterns loaded")
        print("   Please fix validation errors before proceeding.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)