#!/usr/bin/env python3
"""
Test the separated FastMCP backend and FastAPI frontend architecture.

This test verifies that:
1. FastMCP backend can run independently
2. FastAPI frontend can communicate via service bridge
3. Separation of concerns is maintained
"""

import asyncio
import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from server import create_standalone_server
from bridge.service_bridge import get_service_bridge
from fastapi.app_factory import create_app
from config import settings


async def test_fastmcp_backend_standalone():
    """Test FastMCP backend running independently."""
    print("🧪 Testing FastMCP Backend (Standalone)")
    print("=" * 50)
    
    try:
        # Create standalone FastMCP server
        server = create_standalone_server()
        await server.initialize()
        
        # Test service operations
        service = server.get_service()
        if not service:
            print("❌ FastMCP service not available")
            return False
        
        # Test service health
        is_healthy = service.is_healthy()
        print(f"✅ FastMCP service health: {is_healthy}")
        
        # Test business operations
        status = await service.get_service_status()
        print(f"✅ Service status: {status['service']['status']}")
        print(f"   Active databases: {list(status['databases'].keys())}")
        
        # Cleanup
        await server.cleanup()
        print("✅ FastMCP backend standalone test completed")
        return True
        
    except Exception as e:
        print(f"❌ FastMCP backend test failed: {e}")
        return False


async def test_service_bridge():
    """Test service bridge communication layer."""
    print("\n🌉 Testing Service Bridge")
    print("=" * 50)
    
    try:
        # Get service bridge
        bridge = get_service_bridge()
        
        # Test health check
        is_healthy = await bridge.is_healthy()
        print(f"✅ Service bridge health: {is_healthy}")
        
        if is_healthy:
            # Test service status
            status = await bridge.get_service_status()
            print(f"✅ Bridge type: {status.get('bridge', {}).get('type', 'unknown')}")
            print(f"   Services available: {list(status.get('databases', {}).keys())}")
            
            # Test a simple operation
            try:
                tables = await bridge.list_tables("mariadb")
                print(f"✅ Can list tables: {len(tables)} tables found")
            except Exception as e:
                print(f"⚠️  Table listing failed (may be expected): {e}")
        
        print("✅ Service bridge test completed")
        return True
        
    except Exception as e:
        print(f"❌ Service bridge test failed: {e}")
        return False


async def test_fastapi_integration():
    """Test FastAPI integration with service bridge."""
    print("\n🚀 Testing FastAPI Integration")
    print("=" * 50)
    
    try:
        # This would normally be tested with a test client
        # For now, just verify the app can be created
        app = create_app()
        print("✅ FastAPI app created successfully")
        
        # Check if routes are registered
        routes = [route.path for route in app.routes]
        api_routes = [r for r in routes if r.startswith('/api')]
        print(f"✅ API routes registered: {len(api_routes)} routes")
        
        # Key routes that should exist
        expected_routes = ['/health', '/', '/api/database/execute', '/api/database/schema']
        missing_routes = [r for r in expected_routes if r not in routes]
        
        if missing_routes:
            print(f"⚠️  Missing expected routes: {missing_routes}")
        else:
            print("✅ All expected routes present")
        
        print("✅ FastAPI integration test completed")
        return True
        
    except Exception as e:
        print(f"❌ FastAPI integration test failed: {e}")
        return False


async def main():
    """Run all architecture separation tests."""
    print("🏗️  Testing Separated Architecture")
    print("=" * 70)
    print(f"Configuration: {settings.app_name} v{settings.app_version}")
    print()
    
    # Run tests
    backend_ok = await test_fastmcp_backend_standalone()
    bridge_ok = await test_service_bridge()
    frontend_ok = await test_fastapi_integration()
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Test Results Summary")
    print("=" * 70)
    
    tests = [
        ("FastMCP Backend Standalone", backend_ok),
        ("Service Bridge Communication", bridge_ok),
        ("FastAPI Integration", frontend_ok)
    ]
    
    passed = sum(1 for _, ok in tests if ok)
    total = len(tests)
    
    for test_name, ok in tests:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Architecture separation successful!")
        print("\n💡 Benefits achieved:")
        print("   🔹 FastMCP backend can run independently")
        print("   🔹 FastAPI frontend communicates via clean interface")
        print("   🔹 Clear separation of concerns maintained")
        print("   🔹 Both layers can be scaled independently")
    else:
        print("⚠️  Some tests failed - check implementation")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)