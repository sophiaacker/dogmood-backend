#!/usr/bin/env python3
"""
Test script for the unified DogMood API (bark + skin analysis)
"""
import requests
import os

API_BASE = "http://localhost:8000"

def test_health():
    """Test the health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed")
            print(f"   Version: {data.get('version')}")
            print(f"   Bark classifier: {data.get('bark_classifier')}")
            print(f"   Skin classifier: {data.get('skin_classifier')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_bark_analysis():
    """Test bark analysis with audio file"""
    print("\n🎵 Testing bark analysis...")
    audio_file = "snoutscout_classifier/1.wav"
    
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return False
    
    try:
        with open(audio_file, 'rb') as f:
            files = {'file': (audio_file, f, 'audio/wav')}
            response = requests.post(f"{API_BASE}/analyze", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Bark analysis successful!")
            print(f"   Analysis type: {result.get('analysis_type')}")
            print(f"   Label: {result.get('label')}")
            print(f"   Confidence: {result.get('confidence', 0):.3f}")
            print(f"   State: {result.get('state')}")
            print(f"   Suggestion: {result.get('suggestion')}")
            products = result.get('products', [])
            if products:
                print(f"   Products: {', '.join(products)}")
            return True
        else:
            print(f"❌ Bark analysis failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Bark analysis error: {e}")
        return False

def test_skin_analysis():
    """Test skin analysis with image file"""
    print("\n🔬 Testing skin analysis...")
    image_file = "dog_skin_classifier/test_lick.png"
    
    if not os.path.exists(image_file):
        print(f"❌ Image file not found: {image_file}")
        return False
    
    try:
        with open(image_file, 'rb') as f:
            files = {'file': (image_file, f, 'image/png')}
            response = requests.post(f"{API_BASE}/analyze", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Skin analysis successful!")
            print(f"   Analysis type: {result.get('analysis_type')}")
            print(f"   Label: {result.get('label')}")
            print(f"   Confidence: {result.get('confidence', 0):.3f}")
            print(f"   State: {result.get('state')}")
            print(f"   Suggestion: {result.get('suggestion')}")
            products = result.get('products', [])
            if products:
                print(f"   Products: {', '.join(products)}")
            return True
        else:
            print(f"❌ Skin analysis failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Skin analysis error: {e}")
        return False

def test_unsupported_file():
    """Test with unsupported file type"""
    print("\n❓ Testing unsupported file type...")
    try:
        # Create a temporary text file
        with open("temp_test.txt", "w") as f:
            f.write("This is a test file")
        
        with open("temp_test.txt", 'rb') as f:
            files = {'file': ("test.txt", f, 'text/plain')}
            response = requests.post(f"{API_BASE}/analyze", files=files)
        
        if response.status_code == 400:
            print(f"✅ Correctly rejected unsupported file type")
            return True
        else:
            print(f"❌ Should have rejected unsupported file: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Unsupported file test error: {e}")
        return False
    finally:
        # Cleanup
        try:
            os.unlink("temp_test.txt")
        except:
            pass

def main():
    print("🐶 DogMood Unified API Test")
    print("=" * 40)
    
    # Test health endpoint
    if not test_health():
        print("❌ Server not available. Make sure it's running on port 8000")
        return
    
    # Test bark analysis
    bark_success = test_bark_analysis()
    
    # Test skin analysis
    skin_success = test_skin_analysis()
    
    # Test error handling
    error_success = test_unsupported_file()
    
    print("\n" + "=" * 40)
    print("📊 Test Summary:")
    print(f"   Bark Analysis: {'✅' if bark_success else '❌'}")
    print(f"   Skin Analysis: {'✅' if skin_success else '❌'}")
    print(f"   Error Handling: {'✅' if error_success else '❌'}")
    
    if bark_success and skin_success and error_success:
        print("\n🎉 All tests passed! The unified API is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()
