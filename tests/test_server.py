#!/usr/bin/env python3
"""Test script to verify MCP server functionality."""

import sys
import subprocess
from pathlib import Path

def test_server_import():
    """Test if the server can be imported successfully."""
    try:
        sys.path.insert(0, 'src')
        import server
        print("✅ Server imports successfully")
        return True
    except Exception as e:
        print(f"❌ Server import failed: {e}")
        return False

def test_job_manager():
    """Test job manager functionality."""
    try:
        sys.path.insert(0, 'src')
        from jobs.manager import job_manager

        # Test list jobs (should be empty initially)
        result = job_manager.list_jobs()
        if result.get("status") == "success":
            print("✅ Job manager working")
            return True
        else:
            print(f"❌ Job manager failed: {result}")
            return False
    except Exception as e:
        print(f"❌ Job manager test failed: {e}")
        return False

def test_example_data():
    """Test example data availability."""
    examples_dir = Path("examples/data")
    if examples_dir.exists():
        files = list(examples_dir.iterdir())
        print(f"✅ Example data found: {len(files)} files")
        for f in files:
            print(f"   - {f.name}")
        return True
    else:
        print("❌ Example data directory not found")
        return False

def test_fastmcp_dev():
    """Test if fastmcp dev command works."""
    try:
        # Use subprocess to check if fastmcp can start the server
        cmd = ["env/bin/python", "-m", "fastmcp", "dev", "src/server.py", "--help"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        if result.returncode == 0 or "fastmcp" in result.stdout.lower():
            print("✅ FastMCP dev command available")
            return True
        else:
            print(f"❌ FastMCP dev command failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ FastMCP dev test failed: {e}")
        return False

def main():
    print("🧪 Testing MCP Server Setup")
    print("=" * 50)

    tests = [
        test_server_import,
        test_job_manager,
        test_example_data,
        test_fastmcp_dev
    ]

    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 50)
    print(f"📊 Results: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("🎉 All tests passed! MCP server is ready.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")

if __name__ == "__main__":
    main()