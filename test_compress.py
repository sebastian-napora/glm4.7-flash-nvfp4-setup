"""Test compression callback with very long conversation"""
import requests
import json

# Need 50k tokens minimum with at least 10 messages
# My previous test showed ~400 tokens per message
# 125 messages * 400 = 50k tokens

long_messages = []
for i in range(125):
    long_messages.append({"role": "user", "content": f"Message {i}. " + "Hello world " * 500})

print(f"Testing with {len(long_messages)} messages...")

response = requests.post(
    "http://localhost:11111/v1/chat/completions",
    json={
        "model": "glm-4.7-flash-nvfp4",
        "messages": long_messages,
        "max_tokens": 100,
    },
    timeout=300
)
print(f"Status: {response.status_code}")
if response.ok:
    data = response.json()
    content = data['choices'][0]['message']['content']
    print(f"Response: {content[:500]}...")
    print(f"Usage: {data.get('usage', {})}")
    # Check if compression happened
    if '[CONTEXT COMPRESSED' in content:
        print("✓ Compression triggered!")
    else:
        print("✗ No compression")
else:
    print(f"Error: {response.text[:500]}")
