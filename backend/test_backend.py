"""
Simple test script to check backend setup
Run this to diagnose common issues
"""
import os
import sys

print("=" * 60)
print("FinSamaritan Backend Diagnostic Test")
print("=" * 60)
print()

# Test 1: Check Python version
print("1. Checking Python version...")
print(f"   Python: {sys.version}")
if sys.version_info < (3, 8):
    print("   ❌ ERROR: Python 3.8+ required")
    sys.exit(1)
else:
    print("   ✅ Python version OK")
print()

# Test 2: Check dependencies
print("2. Checking dependencies...")
try:
    import fastapi
    print("   ✅ fastapi")
except ImportError:
    print("   ❌ fastapi not installed")
    sys.exit(1)

try:
    import pandas
    print("   ✅ pandas")
except ImportError:
    print("   ❌ pandas not installed")
    sys.exit(1)

try:
    import google.generativeai
    print("   ✅ google-generativeai")
except ImportError:
    print("   ❌ google-generativeai not installed")
    sys.exit(1)

try:
    import numpy
    print("   ✅ numpy")
except ImportError:
    print("   ❌ numpy not installed")
    sys.exit(1)

try:
    import yfinance
    print("   ✅ yfinance")
except ImportError:
    print("   ❌ yfinance not installed")
    sys.exit(1)

print("   ✅ All dependencies OK")
print()

# Test 3: Check .env file
print("3. Checking .env file...")
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    print("   ✅ .env file exists")
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        print(f"   ✅ GEMINI_API_KEY found (length: {len(api_key)} characters)")
    else:
        print("   ❌ GEMINI_API_KEY not found in .env file")
        print("   Add: GEMINI_API_KEY=your_key_here")
        sys.exit(1)
else:
    print("   ❌ .env file not found")
    print("   Create .env file with: GEMINI_API_KEY=your_key_here")
    sys.exit(1)
print()

# Test 4: Check stock_data.csv
print("4. Checking stock_data.csv...")
csv_path = os.path.join(os.path.dirname(__file__), 'stock_data.csv')
if os.path.exists(csv_path):
    print("   ✅ stock_data.csv exists")
    import pandas as pd
    try:
        df = pd.read_csv(csv_path)
        print(f"   ✅ CSV readable ({len(df)} rows)")
    except Exception as e:
        print(f"   ❌ Error reading CSV: {e}")
        sys.exit(1)
else:
    print("   ⚠️  stock_data.csv not found")
    print("   Run: python stock_data_generator.py")
print()

# Test 5: Test Gemini API connection
print("5. Testing Gemini API connection...")
try:
    import google.generativeai as genai
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("   ❌ GEMINI_API_KEY not found in environment")
        sys.exit(1)
    
    print(f"   API Key found (first 10 chars: {api_key[:10]}...)")
    genai.configure(api_key=api_key)
    
    # First, try to list available models
    print("   Fetching available models...")
    try:
        models = list(genai.list_models())
        print(f"   ✅ Found {len(models)} models")
        
        # Filter models that support generateContent
        available_models = []
        for m in models:
            if 'generateContent' in m.supported_generation_methods:
                model_name = m.name.replace('models/', '')
                available_models.append(model_name)
                if len(available_models) <= 5:  # Show first 5
                    print(f"      - {model_name}")
        
        if len(available_models) > 5:
            print(f"      ... and {len(available_models) - 5} more")
        
        # Use the first available model for testing
        if available_models:
            test_model_name = available_models[0]
            print(f"\n   Testing with model: {test_model_name}")
            model = genai.GenerativeModel(test_model_name)
            response = model.generate_content("Say 'test' if you can read this.")
            print(f"   ✅ Model '{test_model_name}' works!")
            print(f"   Response: {response.text[:50]}")
            print(f"\n   💡 Recommended model for main.py: {test_model_name}")
        else:
            raise Exception("No models with generateContent support found")
            
    except Exception as list_error:
        error_msg = str(list_error)
        print(f"   ⚠️  Could not list models: {error_msg[:100]}")
        
        # Fallback: try common model names
        print("   Trying common model names...")
        model_names_to_try = [
            'gemini-2.5-flash',
            'gemini-2.0-flash',
            'gemini-2.5-pro',
            'gemini-flash-latest',
            'gemini-pro-latest',
            'gemini-pro',
        ]
        
        model = None
        working_model_name = None
        
        for model_name in model_names_to_try:
            try:
                test_model = genai.GenerativeModel(model_name)
                response = test_model.generate_content("Say 'test' if you can read this.")
                model = test_model
                working_model_name = model_name
                print(f"   ✅ Model '{model_name}' works!")
                print(f"   Response: {response.text[:50]}")
                break
            except Exception as model_error:
                error_msg = str(model_error)
                if "404" in error_msg or "not found" in error_msg.lower():
                    continue
                elif "403" in error_msg or "permission" in error_msg.lower():
                    print(f"   ❌ Permission denied for {model_name}")
                    print("   This usually means your API key is invalid or doesn't have access")
                    raise
                else:
                    print(f"   ⚠️  {model_name}: {error_msg[:80]}")
                    continue
        
        if model is None:
            print("   ❌ No working Gemini model found.")
            print("\n   Possible issues:")
            print("   1. API key is invalid or expired")
            print("   2. API key doesn't have access to Gemini models")
            print("   3. Network/firewall blocking API access")
            print("   4. Model names have changed - try updating:")
            print("      pip install --upgrade google-generativeai")
            print("   5. Check available models at: https://ai.google.dev/models")
            raise Exception("Could not connect to any Gemini model")
        
        print(f"   ✅ Using model: {working_model_name}")
    
    print("   ✅ Gemini API connection OK")
except Exception as e:
    print(f"   ❌ Gemini API error: {e}")
    print("\n   Troubleshooting steps:")
    print("   1. Verify API key at: https://makersuite.google.com/app/apikey")
    print("   2. Make sure API key is correct in .env file")
    print("   3. Try updating: pip install --upgrade google-generativeai")
    print("   4. Check if your API key has access to Gemini models")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# Test 6: Test agent_tools
print("6. Testing agent_tools module...")
try:
    from agent_tools import load_stock_data, search_stocks
    print("   ✅ agent_tools imports OK")
    
    # Try loading stock data
    df = load_stock_data()
    print(f"   ✅ Stock data loaded ({len(df)} rows)")
    
    # Try a simple search
    result = search_stocks("Sector == 'IT'")
    if result["success"]:
        print(f"   ✅ Search function works ({result['count']} results)")
    else:
        print(f"   ⚠️  Search returned error: {result.get('error')}")
except Exception as e:
    print(f"   ❌ Error testing agent_tools: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

print("=" * 60)
print("✅ All tests passed! Backend should work.")
print("=" * 60)
print()
print("Next steps:")
print("1. Run: python main.py")
print("2. Test in browser: http://localhost:8000")
print("3. Test API: http://localhost:8000/docs")

