#!/usr/bin/env python3
"""
Quick test for API migration to uu.ci endpoint
Tests the OpenAI-compatible API with chatgpt-4o-latest model
"""

import requests
import json

# Migration settings
API_BASE_URL = "https://uu.ci/v1/chat/completions"
API_KEY = "sk-PfyoxkCG9FTdx2khC9857162F0Ed43658f9fD45eE7251bC9"
MODEL = "chatgpt-4o-latest"

def test_chat_completion():
    """Test basic chat completion with the migrated API endpoint"""
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": "Hello! Please respond with 'API test successful' if you receive this message."
            }
        ],
        "max_tokens": 50,
        "temperature": 0.7
    }
    
    print(f"Testing API endpoint: {API_BASE_URL}")
    print(f"Using model: {MODEL}")
    print("-" * 50)
    
    try:
        response = requests.post(
            API_BASE_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        # Print response status
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS!")
            print(f"\nResponse: {json.dumps(result, indent=2)}")
            
            # Extract the message content
            if "choices" in result and len(result["choices"]) > 0:
                message_content = result["choices"][0].get("message", {}).get("content", "")
                print(f"\n📝 Assistant Response: {message_content}")
                
        else:
            print("❌ FAILED!")
            print(f"Error: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    print("=" * 50)
    print("API Migration Test - uu.ci Endpoint")
    print("=" * 50)
    test_chat_completion()
