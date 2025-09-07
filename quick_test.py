#!/usr/bin/env python3
"""
Quick test script to verify your DogMood API is working
"""
import requests
from pathlib import Path

def quick_test():
    print("🐶 Quick DogMood API Test")
    print("=" * 25)
    
    # Health check
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running")
        else:
            print("❌ Server health check failed")
            return
    except:
        print("❌ Server not responding. Run: ./start_dev.sh")
        return
    
    # Test analyze endpoint
    test_file = Path("snoutscout_classifier/1.wav")
    if not test_file.exists():
        print("❌ Test file not found")
        return
    
    print(f"🎵 Testing with: {test_file}")
    
    try:
        with open(test_file, 'rb') as f:
            files = {'file': (test_file.name, f, 'audio/wav')}
            response = requests.post("http://localhost:8000/analyze", files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Analysis successful!")
            print(f"   🏷️  Label: {result.get('label')}")
            print(f"   📊 Confidence: {result.get('confidence', 0):.3f}")
            print(f"   🎭 State: {result.get('state', 'N/A')}")
            print(f"   💡 Suggestion: {result.get('suggestion', 'N/A')}")
            
            # Show products if available
            products = result.get('products', [])
            if products:
                print(f"   🛍️  Products: {', '.join(products)}")
            
            # Check if LLM is being used
            reason = result.get('reason', '')
            if len(reason) > 50:
                print("   🧠 LLM is working!")
            else:
                print("   📋 Using rule fallback")
                
        else:
            print(f"❌ Analysis failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    quick_test()
