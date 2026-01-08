"""
Quick diagnostic script to check Gemini API connection
Run this to see what models are available and test your API key
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ ERROR: GEMINI_API_KEY not found in .env file")
    print("Create .env file with: GEMINI_API_KEY=your_key_here")
    sys.exit(1)

print(f"✓ API Key found (length: {len(api_key)} characters)")
print(f"  First 10 chars: {api_key[:10]}...")
print()

try:
    import google.generativeai as genai
    print(f"✓ google-generativeai package imported")
    
    # Check package version
    try:
        import pkg_resources
        version = pkg_resources.get_distribution('google-generativeai').version
        print(f"✓ Package version: {version}")
    except:
        print("⚠ Could not determine package version")
    
    print()
    print("=" * 60)
    print("Configuring API...")
    genai.configure(api_key=api_key)
    
    print("=" * 60)
    print("Fetching available models...")
    print("=" * 60)
    
    try:
        models = list(genai.list_models())
        print(f"✓ Found {len(models)} total models\n")
        
        # Filter models that support generateContent
        available_models = []
        for m in models:
            if 'generateContent' in m.supported_generation_methods:
                model_name = m.name.replace('models/', '')
                available_models.append(model_name)
        
        print(f"✓ Found {len(available_models)} models with generateContent support:\n")
        
        # Show all available models
        for i, model_name in enumerate(available_models, 1):
            print(f"  {i}. {model_name}")
        
        if not available_models:
            print("  ❌ No models found with generateContent support!")
            print("  This might indicate:")
            print("    - API key is invalid")
            print("    - API key doesn't have access to Gemini")
            print("    - Network/firewall issue")
            sys.exit(1)
        
        print()
        print("=" * 60)
        print(f"Testing with first available model: {available_models[0]}")
        print("=" * 60)
        
        model = genai.GenerativeModel(available_models[0])
        response = model.generate_content("Say 'Hello' if you can read this.")
        
        print(f"✓ Model '{available_models[0]}' works!")
        print(f"✓ Response: {response.text}")
        print()
        print("=" * 60)
        print("✅ SUCCESS! Your API is working correctly.")
        print("=" * 60)
        print()
        print("Recommended models for main.py:")
        print(f"  Agent model: {available_models[0]}")
        if len(available_models) > 1:
            # Try to find a pro model for vision
            pro_models = [m for m in available_models if 'pro' in m.lower()]
            if pro_models:
                print(f"  Vision model: {pro_models[0]}")
            else:
                print(f"  Vision model: {available_models[0]} (or {available_models[1] if len(available_models) > 1 else available_models[0]})")
        else:
            print(f"  Vision model: {available_models[0]}")
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error listing models: {error_msg}")
        print()
        
        if "403" in error_msg or "permission" in error_msg.lower():
            print("This is a PERMISSION error. Possible causes:")
            print("  1. API key is invalid or expired")
            print("  2. API key doesn't have access to Gemini models")
            print("  3. Check your API key at: https://makersuite.google.com/app/apikey")
        elif "401" in error_msg or "unauthorized" in error_msg.lower():
            print("This is an AUTHENTICATION error:")
            print("  1. API key is incorrect")
            print("  2. API key has been revoked")
            print("  3. Get a new API key at: https://makersuite.google.com/app/apikey")
        elif "404" in error_msg:
            print("This is a NOT FOUND error:")
            print("  1. API endpoint might have changed")
            print("  2. Try updating: pip install --upgrade google-generativeai")
        else:
            print("Unknown error. Full details:")
            import traceback
            traceback.print_exc()
        
        sys.exit(1)
        
except ImportError:
    print("❌ ERROR: google-generativeai package not installed")
    print("Install it with: pip install google-generativeai")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

