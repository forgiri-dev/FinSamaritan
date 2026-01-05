"""
Quick Backend Testing Script
Tests all major endpoints and functionality
"""
import requests
import json
import sys
import time

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health check endpoint"""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed")
            print(f"   Status: {data.get('status')}")
            print(f"   Cache size: {data.get('cache_size')}")
            print(f"   Gemini configured: {data.get('gemini_configured')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        print("   Make sure the backend server is running!")
        return False

def test_agent_simple():
    """Test simple agent query"""
    print("\n🔍 Testing agent endpoint (simple query)...")
    try:
        response = requests.post(
            f"{BASE_URL}/agent",
            json={"text": "What is the current price of RELIANCE.NS?"},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print(f"✅ Agent query successful")
                print(f"   Response preview: {data.get('response', '')[:100]}...")
                return True
            else:
                print(f"❌ Agent query failed: {data}")
                return False
        else:
            print(f"❌ Agent query failed: {response.status_code}")
            print(f"   {response.text}")
            return False
    except Exception as e:
        print(f"❌ Agent query failed: {e}")
        return False

def test_portfolio_management():
    """Test portfolio management"""
    print("\n🔍 Testing portfolio management...")
    try:
        # Add stock
        response = requests.post(
            f"{BASE_URL}/agent",
            json={"text": "I bought 100 shares of RELIANCE.NS at 2400"},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print(f"✅ Portfolio add successful")
                print(f"   Response: {data.get('response', '')[:150]}...")
                
                # Check portfolio
                time.sleep(1)
                response2 = requests.post(
                    f"{BASE_URL}/agent",
                    json={"text": "Show me my portfolio"},
                    timeout=30
                )
                if response2.status_code == 200:
                    data2 = response2.json()
                    if data2.get("success"):
                        print(f"✅ Portfolio view successful")
                        return True
                
                return True
            else:
                print(f"❌ Portfolio add failed: {data}")
                return False
        else:
            print(f"❌ Portfolio add failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Portfolio management failed: {e}")
        return False

def test_screener():
    """Test stock screener"""
    print("\n🔍 Testing stock screener...")
    try:
        response = requests.post(
            f"{BASE_URL}/agent",
            json={"text": "Show me stocks with PE ratio less than 20"},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print(f"✅ Screener query successful")
                print(f"   Response preview: {data.get('response', '')[:150]}...")
                return True
            else:
                print(f"❌ Screener query failed: {data}")
                return False
        else:
            print(f"❌ Screener query failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Screener test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("FinSamaritan Backend Test Suite")
    print("=" * 50)
    print(f"Testing server at: {BASE_URL}\n")
    
    results = []
    
    # Test 1: Health check
    results.append(("Health Check", test_health()))
    
    if not results[0][1]:
        print("\n❌ Backend server is not running or not accessible!")
        print("   Please start the server first: uvicorn main:app --reload")
        sys.exit(1)
    
    # Test 2: Simple agent query
    results.append(("Agent Query", test_agent_simple()))
    
    # Test 3: Portfolio management
    results.append(("Portfolio Management", test_portfolio_management()))
    
    # Test 4: Screener
    results.append(("Stock Screener", test_screener()))
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n⚠️ Some tests failed. Check the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()


