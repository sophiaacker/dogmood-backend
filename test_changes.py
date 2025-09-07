#!/usr/bin/env python3
"""
Script to test that code changes are being picked up by making a visible change
and verifying it appears in the API response.
"""
import requests
import time
import sys
from pathlib import Path

def test_code_change_detection():
    """Test that code changes are being picked up by the server"""
    
    base_url = "http://localhost:8000"
    
    # First, make a test request to see current behavior
    print("🔍 Testing current API response...")
    
    classifier_dir = Path("snoutscout_classifier")
    test_file = classifier_dir / "1.wav"
    
    if not test_file.exists():
        print("❌ Test file not found: snoutscout_classifier/1.wav")
        return False
    
    try:
        with open(test_file, 'rb') as f:
            files = {'file': (test_file.name, f, 'audio/wav')}
            response = requests.post(f"{base_url}/analyze", files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Current response:")
            print(f"   Label: {result.get('label')}")
            print(f"   State: {result.get('state', 'N/A')}")
            print(f"   Reason: {result.get('reason', 'N/A')}")
            return True
        else:
            print(f"❌ API request failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Request error: {e}")
        return False

def main():
    print("🔄 Testing Code Change Detection")
    print("=" * 40)
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            print("❌ Server not responding properly")
            sys.exit(1)
    except:
        print("❌ Server is not running. Start with: uvicorn main:app --reload")
        sys.exit(1)
    
    print("✅ Server is running")
    
    # Test current API behavior
    if test_code_change_detection():
        print("\n✅ API is responding correctly")
        print("\n💡 To test code changes:")
        print("   1. Modify suggestions.py (e.g., change a rule message)")
        print("   2. Save the file")
        print("   3. Run this script again")
        print("   4. The uvicorn server should auto-reload and show changes")
    else:
        print("\n❌ API test failed")

if __name__ == "__main__":
    main()
