#!/usr/bin/env python3
"""
Example: Extending vLLM API with Anchor Support

This shows how to add anchor registration to the vLLM server.
Two approaches are demonstrated:

1. Custom endpoint approach - Add /register_anchor endpoint
2. Token-based approach - Parse anchor tokens from prompts

For production use, you'd integrate this into vLLM's api_server.py
"""

from typing import Optional
import hashlib


# =============================================================================
# Approach 1: Custom Endpoint
# =============================================================================

"""
Add this to vllm/entrypoints/openai/api_server.py:

from fastapi import Request
from pydantic import BaseModel

class AnchorRegistration(BaseModel):
    request_id: str
    anchor_id: str

@router.post("/v1/anchors/register")
async def register_anchor(registration: AnchorRegistration):
    \"\"\"Register a semantic anchor for a request.\"\"\"
    # Get connector from engine
    connector = engine.kv_connector
    if connector and hasattr(connector, 'register_anchor'):
        connector.register_anchor(
            registration.request_id,
            registration.anchor_id
        )
        return {"status": "registered", "anchor_id": registration.anchor_id}
    return {"status": "error", "message": "Connector not available"}

@router.get("/v1/anchors")
async def list_anchors():
    \"\"\"List all stored anchors.\"\"\"
    connector = engine.kv_connector
    if connector and hasattr(connector, 'list_anchors'):
        return {"anchors": connector.list_anchors()}
    return {"anchors": []}

@router.delete("/v1/anchors/{anchor_id}")
async def delete_anchor(anchor_id: str):
    \"\"\"Delete an anchor.\"\"\"
    connector = engine.kv_connector
    if connector and hasattr(connector, 'delete_anchor'):
        success = connector.delete_anchor(anchor_id)
        return {"status": "deleted" if success else "not_found"}
    return {"status": "error"}
"""


# =============================================================================
# Approach 2: Token-Based Anchors
# =============================================================================

def extract_anchor_from_prompt(prompt: str) -> Optional[str]:
    """
    Extract anchor ID from prompt using special token format.

    Format: <|anchor:ANCHOR_ID|>
    Example: <|anchor:alice-software-tokyo|>

    Usage in prompt:
        "Context: <|anchor:alice-software-tokyo|>
         Alice is a software engineer from Tokyo..."
    """
    import re
    match = re.search(r'<\|anchor:([^|]+)\|>', prompt)
    if match:
        return f"<{match.group(1)}/>"
    return None


def generate_semantic_anchor(context: str, num_keywords: int = 5) -> str:
    """
    Auto-generate a semantic anchor from context.

    In practice, you'd use:
    - Named entity recognition (NER)
    - Keyword extraction
    - Topic modeling
    - Or an LLM to summarize

    This is a simple hash-based placeholder.
    """
    # Simple: hash first N words
    words = context.lower().split()[:20]
    content = " ".join(words)
    hash_val = hashlib.md5(content.encode()).hexdigest()[:8]
    return f"<auto-{hash_val}/>"


# =============================================================================
# Example: Modified Connector with Auto-Detection
# =============================================================================

MODIFIED_CONNECTOR_CODE = '''
# Add to AnchorConnector class:

def _extract_anchor_from_tokens(self, token_ids: list[int], tokenizer) -> Optional[str]:
    """Extract anchor from special tokens in prompt."""
    # Decode and look for anchor pattern
    text = tokenizer.decode(token_ids[:100])  # Check first 100 tokens

    import re
    match = re.search(r'<\|anchor:([^|]+)\|>', text)
    if match:
        return f"<{match.group(1)}/>"
    return None

def get_num_new_matched_tokens(
    self,
    request: "Request",
    num_computed_tokens: int,
) -> tuple[int | None, bool]:
    """Check if anchor exists, auto-detecting from prompt if needed."""
    request_id = request.request_id

    # First check registry
    if request_id in self._anchor_registry:
        anchor_id = self._anchor_registry[request_id]
    else:
        # Try to extract from prompt
        anchor_id = self._extract_anchor_from_tokens(
            request.prompt_token_ids,
            request.tokenizer  # Need tokenizer reference
        )
        if anchor_id:
            self._anchor_registry[request_id] = anchor_id

    if not anchor_id:
        return 0, False

    anchor_path = self._get_anchor_path(anchor_id)
    if not os.path.exists(anchor_path):
        return 0, False

    # TODO: Return actual cached token count
    logger.info(f"Anchor hit for {anchor_id}")
    return 0, False
'''


# =============================================================================
# Demo: Using Anchors via Client
# =============================================================================

def demo_client_with_anchors():
    """
    Demonstrate how a client would use anchors.

    This assumes the custom endpoint approach is implemented.
    """
    import requests

    SERVER_URL = "http://localhost:8000"

    # Step 1: Prepare context with anchor
    context = """Alice is a software engineer from Tokyo who loves hiking.
She works at a startup building AI systems. Her favorite hiking spot is Mount Takao.
She has a dog named Mochi who sometimes joins her on easy trails."""

    anchor_id = "<alice-software-tokyo/>"

    # Step 2: Register anchor (hypothetical endpoint)
    print(f"Registering anchor: {anchor_id}")
    # requests.post(f"{SERVER_URL}/v1/anchors/register", json={
    #     "request_id": "req-001",
    #     "anchor_id": anchor_id
    # })

    # Step 3: First request - will save state
    print("First request (saves state)...")
    response = requests.post(
        f"{SERVER_URL}/v1/chat/completions",
        json={
            "model": "moonshotai/Kimi-Linear-48B-A3B-Instruct",
            "messages": [
                {"role": "user", "content": f"{context}\n\nWhat is Alice's dog's name?"}
            ],
            "max_tokens": 50,
        }
    )
    print(f"Response: {response.json()['choices'][0]['message']['content']}")

    # Step 4: Subsequent requests - will load cached state
    print("\nSubsequent request (loads cached state)...")
    # Same anchor would load the cached KDA state instead of recomputing

    return True


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Anchor API Extension Examples")
    print("=" * 60)

    # Test anchor extraction
    test_prompts = [
        "Hello <|anchor:alice-software-tokyo|> Alice is a software engineer...",
        "No anchor in this prompt",
        "<|anchor:bob-engineer-seattle|> Bob works in Seattle",
    ]

    print("\n--- Anchor Extraction ---")
    for prompt in test_prompts:
        anchor = extract_anchor_from_prompt(prompt)
        print(f"  Prompt: {prompt[:40]}...")
        print(f"  Anchor: {anchor}\n")

    # Test auto-generation
    print("--- Auto-Generated Anchors ---")
    contexts = [
        "Alice is a software engineer from Tokyo",
        "Bob is a data scientist in New York",
        "Alice is a software engineer from Tokyo",  # Same as first
    ]

    for ctx in contexts:
        anchor = generate_semantic_anchor(ctx)
        print(f"  Context: {ctx[:40]}...")
        print(f"  Anchor: {anchor}\n")

    print("=" * 60)
    print("See code comments for API integration details")
    print("=" * 60)
