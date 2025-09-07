#!/usr/bin/env python3
"""
Test script for DogMood API to verify the /analyze endpoint is working
and picking up code changes.
"""
import requests
import os
import sys
from pathlib import Path

def test_health_endpoint(base_url="http://localhost:8000"):
    """Test the health endpoint first"""
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"✅ Health check: {response.status_code}")
        print(f"   Response: {response.json()}")
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_analyze_endpoint(base_url="http://localhost:8000"):
    """Test the analyze endpoint with a sample audio file"""
    
    # Look for training WAV files to use as test input
    classifier_dir = Path("snoutscout_classifier")
    test_files = []
    
    for i in range(1, 7):  # Check 1.wav through 6.wav
        wav_file = classifier_dir / f"{i}.wav"
        if wav_file.exists():
            test_files.append(wav_file)
    
    if not test_files:
        print("❌ No test WAV files found in snoutscout_classifier/")
        return False
    
    # Use the first available test file
    test_file = test_files[0]
    print(f"🎵 Testing with: {test_file}")
    
    try:
        with open(test_file, 'rb') as f:
            files = {'file': (test_file.name, f, 'audio/wav')}
            response = requests.post(f"{base_url}/analyze", files=files, timeout=30)
        
        print(f"📊 Analyze response: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Analysis successful!")
            print(f"   Label: {result.get('label')}")
            print(f"   Confidence: {result.get('confidence', 0):.2f}")
            print(f"   State: {result.get('state', 'N/A')[:100]}...")
            print(f"   Suggestion: {result.get('suggestion', 'N/A')[:100]}...")
            if result.get('probs'):
                print(f"   Probabilities: {result['probs']}")
            return True
        else:
            print(f"❌ Analysis failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    print("🐶 DogMood API Test Script")
    print("=" * 40)
    
    base_url = "http://localhost:8000"
    
    # Test health endpoint
    if not test_health_endpoint(base_url):
        print("\n❌ Server is not running or not responding")
        print("💡 Start the server with: uvicorn main:app --reload")
        sys.exit(1)
    
    print()
    
    # Test analyze endpoint
    if test_analyze_endpoint(base_url):
        print("\n🎉 All tests passed! The API is working correctly.")
    else:
        print("\n❌ Analysis endpoint test failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
