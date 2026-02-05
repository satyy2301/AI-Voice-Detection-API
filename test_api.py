import requests
import base64
import json
import sys

# Read audio file
audio_path = r"C:\Users\satyy\OneDrive\Desktop\Recording (3).wav"
try:
    with open(audio_path, 'rb') as f:
        audio_data = f.read()
    audio_base64 = base64.b64encode(audio_data).decode('utf-8')
    print(f"✓ Loaded audio file ({len(audio_data)} bytes)")
except Exception as e:
    print(f"✗ Failed to read audio: {e}")
    sys.exit(1)

# Prepare request
url = "http://localhost:8000/detect"
headers = {
    "x-api-key": "hackathon-secret-key",
    "Content-Type": "application/json"
}
payload = {
    "language": "english",
    "audioFormat": "wav",
    "audioBase64": audio_base64
}

print("\n" + "="*60)
print("🧪 TESTING /detect ENDPOINT")
print("="*60)
print(f"URL: {url}")
print(f"API Key: {headers['x-api-key']}")
print(f"Audio Size: {len(audio_data)} bytes ({len(audio_base64)} base64 chars)")
print(f"Language: {payload['language']}")
print("="*60)

try:
    print("\n📤 Sending request...")
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    print(f"✓ Response received: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\n" + "="*60)
        print("✅ INFERENCE SUCCESSFUL!")
        print("="*60)
        print(f"Classification: {result['classification']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Language: {result['language']}")
        print("="*60)
    else:
        print(f"\n✗ Error: {response.status_code}")
        print(f"Response: {response.text}")
except Exception as e:
    print(f"\n✗ Request failed: {e}")
    sys.exit(1)
