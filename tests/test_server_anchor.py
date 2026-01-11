#!/usr/bin/env python3
"""
Test AnchorConnector via vLLM server API.

Run the server first:
  vllm serve moonshotai/Kimi-Linear-48B-A3B-Instruct \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --kv-connector AnchorConnector \
    --kv-connector-config '{"storage_path": "/tmp/anchors"}'

Then run this test:
  python tests/test_server_anchor.py
"""

import requests
import json

SERVER_URL = "http://localhost:8000"

def generate(prompt: str, max_tokens: int = 50) -> str:
    """Send generation request to vLLM server."""
    response = requests.post(
        f"{SERVER_URL}/v1/completions",
        json={
            "model": "moonshotai/Kimi-Linear-48B-A3B-Instruct",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.1,
        }
    )
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return ""
    return response.json()["choices"][0]["text"]

def chat(messages: list, max_tokens: int = 50) -> str:
    """Send chat request to vLLM server."""
    response = requests.post(
        f"{SERVER_URL}/v1/chat/completions",
        json={
            "model": "moonshotai/Kimi-Linear-48B-A3B-Instruct",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.1,
        }
    )
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return ""
    return response.json()["choices"][0]["message"]["content"]

def main():
    print("="*60)
    print("Testing vLLM Server with AnchorConnector")
    print("="*60)

    # Check server health
    try:
        health = requests.get(f"{SERVER_URL}/health")
        print(f"\nServer health: {health.status_code}")
    except requests.exceptions.ConnectionError:
        print("\nERROR: Cannot connect to server!")
        print("Make sure vLLM server is running on port 8000")
        return

    # Context for testing
    context = """Alice is a software engineer from Tokyo who loves hiking.
She works at a startup building AI systems. Her favorite hiking spot is Mount Takao.
She has a dog named Mochi who sometimes joins her on easy trails.
Alice is 28 years old and has been coding since she was 15."""

    # Test 1: Chat with context
    print("\n" + "="*60)
    print("TEST 1: Chat with context")
    print("="*60)

    messages = [
        {"role": "user", "content": f"{context}\n\nWhat is Alice's dog's name?"}
    ]
    answer = chat(messages)
    print(f"Q: What is Alice's dog's name?")
    print(f"A: {answer}")

    # Test 2: Follow-up question (tests if context is maintained)
    print("\n" + "="*60)
    print("TEST 2: Follow-up questions")
    print("="*60)

    questions = [
        "Where does Alice work?",
        "How old is Alice?",
        "What is her favorite hiking spot?",
    ]

    for q in questions:
        messages = [
            {"role": "user", "content": f"{context}\n\n{q}"}
        ]
        answer = chat(messages, max_tokens=30)
        print(f"Q: {q}")
        print(f"A: {answer}\n")

    # Test 3: Without context (baseline)
    print("="*60)
    print("TEST 3: Without context (baseline)")
    print("="*60)

    messages = [
        {"role": "user", "content": "What is Alice's dog's name?"}
    ]
    answer = chat(messages)
    print(f"Q: What is Alice's dog's name?")
    print(f"A: {answer}")

    # Check anchor storage
    print("\n" + "="*60)
    print("Checking anchor storage...")
    print("="*60)

    import os
    storage_path = "/tmp/anchors"
    if os.path.exists(storage_path):
        anchors = os.listdir(storage_path)
        print(f"Anchors in {storage_path}: {anchors}")
    else:
        print(f"No anchors saved yet (storage path doesn't exist)")

    print("\n" + "="*60)
    print("Tests completed!")
    print("="*60)

if __name__ == "__main__":
    main()
