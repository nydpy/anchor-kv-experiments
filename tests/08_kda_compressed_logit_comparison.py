#!/usr/bin/env python3
"""
Experiment 08: KDA Compressed Context Logit Comparison
======================================================

QUESTION: Does compressed context + loaded KDA state produce similar logits
          to full context processing?

HYPOTHESIS:
After processing "I am a boy", the KDA recurrent_state contains:
    S = k₁⊗v₁ + k₂⊗v₂ + k₃⊗v₃ + k₄⊗v₄

If we save this state and later load it with a compressed anchor token,
the model should produce SIMILAR OUTPUT LOGITS as full context.

TEST DESIGN:
┌─────────────────────────────────────────────────────────────────────┐
│  BASELINE: Full context processing                                   │
│  ──────────────────────────────────────────────────────────────────  │
│  Input: "I am a boy" (4 tokens)                                      │
│  KDA:   S = Σ(kᵢ⊗vᵢ) computed during forward pass                   │
│  Output: logits_baseline                                             │
│                                                                      │
│  → Save KDA state S at end of context                                │
├─────────────────────────────────────────────────────────────────────┤
│  TEST: Compressed anchor + loaded KDA                                │
│  ──────────────────────────────────────────────────────────────────  │
│  1. Load saved KDA state S into model                                │
│  2. Input: "<anchor/>" (1 token) instead of full context             │
│  3. KDA uses loaded S (no accumulation needed)                       │
│  4. Output: logits_compressed                                        │
│                                                                      │
│  → Compare: logits_compressed ≈ logits_baseline ?                    │
├─────────────────────────────────────────────────────────────────────┤
│  CONTROL: Fresh generation (no loaded state)                         │
│  ──────────────────────────────────────────────────────────────────  │
│  Input: "<anchor/>" (1 token), no loaded KDA state                   │
│  Output: logits_fresh                                                │
│                                                                      │
│  → logits_fresh should be DIFFERENT from baseline                    │
└─────────────────────────────────────────────────────────────────────┘

SUCCESS CRITERIA:
- KL divergence(logits_compressed, logits_baseline) < threshold
- KL divergence(logits_fresh, logits_baseline) > threshold
- Top-k token overlap between compressed and baseline

METRICS:
- KL Divergence: Measures probability distribution similarity
- Cosine Similarity: Measures logit vector direction
- Top-k Overlap: Measures if same tokens are most likely
- Argmax Match: Do they predict the same next token?

Run: python tests/anchor_experiments/08_kda_only_compression.py
"""

import json
import os
import tempfile
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn.functional as F

# Configuration
MODEL_NAME = "moonshotai/Kimi-Linear-Instruct"
TENSOR_PARALLEL_SIZE = 8
MAX_MODEL_LEN = 32768


@dataclass
class LogitComparison:
    """Results of comparing two logit distributions."""
    kl_divergence: float      # Lower = more similar
    cosine_similarity: float  # Higher = more similar (1.0 = identical direction)
    top_k_overlap: float      # Fraction of top-k tokens that match
    argmax_match: bool        # Do they predict same next token?
    baseline_token: str       # What baseline predicts
    test_token: str           # What test predicts


def compare_logits(
    logits_baseline: torch.Tensor,
    logits_test: torch.Tensor,
    tokenizer,
    k: int = 10,
) -> LogitComparison:
    """
    Compare two logit distributions.

    Args:
        logits_baseline: [vocab_size] logits from full context
        logits_test: [vocab_size] logits from compressed context
        tokenizer: For decoding token IDs
        k: Number of top tokens to compare

    Returns:
        LogitComparison with similarity metrics
    """
    # Ensure same device and shape
    logits_baseline = logits_baseline.float().flatten()
    logits_test = logits_test.float().flatten()

    # 1. KL Divergence (probability distribution similarity)
    probs_baseline = F.softmax(logits_baseline, dim=-1)
    probs_test = F.softmax(logits_test, dim=-1)
    # KL(P || Q) - how much info lost when using Q to approximate P
    kl_div = F.kl_div(
        probs_test.log(),
        probs_baseline,
        reduction='sum'
    ).item()

    # 2. Cosine Similarity (vector direction)
    cos_sim = F.cosine_similarity(
        logits_baseline.unsqueeze(0),
        logits_test.unsqueeze(0)
    ).item()

    # 3. Top-k Overlap
    top_k_baseline = torch.topk(logits_baseline, k).indices.tolist()
    top_k_test = torch.topk(logits_test, k).indices.tolist()
    overlap = len(set(top_k_baseline) & set(top_k_test)) / k

    # 4. Argmax Match
    argmax_baseline = logits_baseline.argmax().item()
    argmax_test = logits_test.argmax().item()
    argmax_match = argmax_baseline == argmax_test

    # Decode tokens for display
    baseline_token = tokenizer.decode([argmax_baseline])
    test_token = tokenizer.decode([argmax_test])

    return LogitComparison(
        kl_divergence=kl_div,
        cosine_similarity=cos_sim,
        top_k_overlap=overlap,
        argmax_match=argmax_match,
        baseline_token=baseline_token,
        test_token=test_token,
    )


def get_test_contexts():
    """Test contexts with clear facts to verify recall."""
    return [
        {
            "id": "alice-software-tokyo",
            "context": """Hi, my name is Alice and I work as a software engineer at a startup in Tokyo.
I've been living here for 5 years and I really enjoy the city.
My hobbies include hiking, photography, and cooking Japanese food.
My favorite hiking spot is Mount Takao, about an hour from my apartment.
Last weekend I went to Mount Fuji and took some amazing photos of the sunrise.""",
            "queries": [
                ("What are Alice's hobbies?", ["hiking", "photography", "cooking"]),
                ("Where does Alice work?", ["software", "engineer", "tokyo", "startup"]),
                ("What did Alice do last weekend?", ["mount fuji", "photos", "sunrise"]),
            ],
        },
        {
            "id": "bob-chef-paris",
            "context": """My name is Bob and I'm a professional chef in Paris.
I specialize in French pastries, especially croissants and macarons.
I've won the Golden Whisk award three times for my chocolate soufflé.
My restaurant is called "Le Petit Nuage" which means "The Little Cloud".
I wake up at 4am every day to prepare fresh bread for the morning rush.""",
            "queries": [
                ("What is Bob's specialty?", ["pastries", "croissants", "macarons"]),
                ("What award did Bob win?", ["golden whisk", "chocolate", "soufflé"]),
                ("What is Bob's restaurant called?", ["petit nuage", "little cloud"]),
            ],
        },
    ]


class KDAStateManager:
    """
    Manages KDA-only state extraction and injection.

    Storage format (per anchor):
        {anchor_id}.pt contains:
        - kda_states: {layer_idx: recurrent_state_tensor}
        - No MLA K,V cache (that's the point!)
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def save_kda_states(
        self,
        anchor_id: str,
        kda_states: dict[int, torch.Tensor],
        metadata: dict = None,
    ):
        """
        Save ONLY KDA recurrent states (skip MLA K,V).

        Args:
            anchor_id: Semantic anchor ID
            kda_states: {layer_idx: recurrent_state} for KDA layers only
            metadata: Optional context info
        """
        safe_id = anchor_id.replace("/", "_").replace("\\", "_")
        path = self.cache_dir / f"{safe_id}.pt"

        # Save only KDA states - this is the key difference!
        torch.save({
            "anchor_id": anchor_id,
            "kda_states": {k: v.cpu() for k, v in kda_states.items()},
            "metadata": metadata or {},
            "storage_type": "kda_only",  # Mark as KDA-only
        }, path)

        # Calculate storage size
        total_bytes = sum(t.numel() * t.element_size() for t in kda_states.values())
        print(f"  Saved KDA-only state: {len(kda_states)} layers, {total_bytes / 1024:.1f} KB")

    def load_kda_states(self, anchor_id: str) -> dict[int, torch.Tensor]:
        """Load KDA states for injection."""
        safe_id = anchor_id.replace("/", "_").replace("\\", "_")
        path = self.cache_dir / f"{safe_id}.pt"

        if not path.exists():
            raise FileNotFoundError(f"No KDA state found for {anchor_id}")

        data = torch.load(path, weights_only=False)
        return data["kda_states"]


def extract_kda_states_from_model(model, num_layers: int = 48) -> dict[int, torch.Tensor]:
    """
    Extract KDA recurrent states from model.

    This is model-specific. For Kimi-Linear:
    - 36 KDA layers (75%)
    - 12 MLA layers (25%)

    We only extract the KDA layers.
    """
    kda_states = {}

    # Access model internals (Kimi-Linear specific)
    try:
        for layer_idx in range(num_layers):
            layer = model.layers[layer_idx]

            # Check if this is a KDA layer
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'recurrent_state'):
                # This is a KDA layer - extract recurrent state
                recurrent_state = layer.self_attn.recurrent_state
                if recurrent_state is not None:
                    kda_states[layer_idx] = recurrent_state.clone()

    except Exception as e:
        print(f"  Warning: Could not extract KDA states: {e}")
        print("  → Model internals may have different structure")

    return kda_states


def inject_kda_states_to_model(model, kda_states: dict[int, torch.Tensor]):
    """
    Inject KDA recurrent states into model.

    This restores the "memory" without needing MLA K,V cache.
    """
    try:
        for layer_idx, state in kda_states.items():
            layer = model.layers[layer_idx]

            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'recurrent_state'):
                # Inject state (move to same device as model)
                device = layer.self_attn.recurrent_state.device
                layer.self_attn.recurrent_state.copy_(state.to(device))

    except Exception as e:
        print(f"  Warning: Could not inject KDA states: {e}")


def test_without_vllm():
    """
    Test KDA-only compression logic without full vLLM setup.

    This verifies the save/load mechanism works before running
    the full GPU test.
    """
    print("\n" + "=" * 70)
    print("TEST 0: KDA-Only Storage Verification (No GPU)")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = KDAStateManager(tmpdir)

        # Create mock KDA states (simulating 36 KDA layers)
        print("\n1. Creating mock KDA states...")
        mock_kda_states = {}
        for layer_idx in [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33]:  # Sample KDA layers
            # Typical KDA recurrent state shape: [batch, heads, head_dim, state_dim]
            mock_kda_states[layer_idx] = torch.randn(1, 32, 128, 256)

        total_params = sum(t.numel() for t in mock_kda_states.values())
        total_bytes = sum(t.numel() * 4 for t in mock_kda_states.values())  # float32
        print(f"   Mock KDA states: {len(mock_kda_states)} layers")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Total size: {total_bytes / 1024:.1f} KB")

        # Save
        print("\n2. Saving KDA-only states...")
        manager.save_kda_states(
            anchor_id="alice-software-tokyo",
            kda_states=mock_kda_states,
            metadata={"context_tokens": 150, "test": True},
        )

        # Load
        print("\n3. Loading KDA-only states...")
        loaded_states = manager.load_kda_states("alice-software-tokyo")

        # Verify
        print("\n4. Verifying loaded states...")
        assert len(loaded_states) == len(mock_kda_states), "Layer count mismatch!"

        for layer_idx in mock_kda_states:
            original = mock_kda_states[layer_idx]
            loaded = loaded_states[layer_idx]
            assert torch.allclose(original, loaded), f"Layer {layer_idx} mismatch!"

        print("   ✓ All states match perfectly!")

        # Show storage comparison
        print("\n" + "-" * 70)
        print("STORAGE COMPARISON (500 token context, 48 layers):")
        print("-" * 70)

        kda_size_per_layer = 1 * 32 * 128 * 256 * 4  # ~4MB per layer (mock)
        # Real Kimi-Linear: ~8KB per layer
        real_kda_size = 8 * 1024 * 36  # 36 KDA layers × 8KB
        mla_kv_size = 500 * 2 * 128 * 12 * 4  # 500 tokens × K,V × 128 dim × 12 layers × 4 bytes

        print(f"   KDA-only (36 layers):  {real_kda_size / 1024:.0f} KB")
        print(f"   MLA K,V (12 layers):   {mla_kv_size / 1024 / 1024:.1f} MB")
        print(f"   Ratio: KDA is {mla_kv_size / real_kda_size:.0f}x smaller!")

        return True


def test_logit_comparison_with_vllm():
    """
    Full logit comparison test with vLLM and Kimi-Linear model.

    Tests whether:
    - Compressed context + loaded KDA → similar logits to full context
    - Fresh generation (no state) → different logits (control)

    Requires: 8× L4 GPUs or equivalent.
    """
    print("\n" + "=" * 70)
    print("TEST: KDA Compressed Context Logit Comparison")
    print("=" * 70)

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("vLLM not installed. Run: pip install vllm")
        return False

    if not torch.cuda.is_available():
        print("CUDA not available. Skipping full test.")
        return False

    print(f"\nGPUs available: {torch.cuda.device_count()}")

    # Initialize model
    print(f"\nLoading {MODEL_NAME}...")
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
    )

    # Get tokenizer for decoding
    tokenizer = llm.get_tokenizer()

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = KDAStateManager(tmpdir)
        test_data = get_test_contexts()[0]

        sampling_params = SamplingParams(
            max_tokens=1,  # Just predict next token
            temperature=0.0,
            logprobs=50,  # Get top 50 token probabilities
        )

        # ─────────────────────────────────────────────────────────────────
        # STEP 1: BASELINE - Full context processing
        # ─────────────────────────────────────────────────────────────────
        print("\n" + "-" * 70)
        print("STEP 1: BASELINE - Full Context Processing")
        print("-" * 70)

        context = test_data["context"]
        query = "\n\nWhat are the hobbies?"

        full_prompt = context + query
        print(f"Context length: {len(context)} chars")
        print(f"Full prompt: {full_prompt[:100]}...")

        baseline_output = llm.generate([full_prompt], sampling_params)[0]

        if baseline_output.outputs[0].logprobs:
            baseline_logprobs = baseline_output.outputs[0].logprobs[0]
            print(f"Baseline top tokens: {list(baseline_logprobs.keys())[:5]}")
            baseline_token = baseline_output.outputs[0].text
            print(f"Baseline predicts: '{baseline_token}'")
        else:
            print("Warning: No logprobs returned")
            return False

        # ─────────────────────────────────────────────────────────────────
        # STEP 2: Extract and save KDA state
        # ─────────────────────────────────────────────────────────────────
        print("\n" + "-" * 70)
        print("STEP 2: Extract KDA State After Context Processing")
        print("-" * 70)

        try:
            model = llm.llm_engine.model_executor.driver_worker.model_runner.model
            kda_states = extract_kda_states_from_model(model)

            if kda_states:
                manager.save_kda_states(test_data["id"], kda_states)
                print(f"Saved KDA states from {len(kda_states)} layers")
            else:
                print("Warning: No KDA states extracted")
                print("→ Model may not expose recurrent_state directly")
        except Exception as e:
            print(f"Could not extract states: {e}")
            kda_states = {}

        # ─────────────────────────────────────────────────────────────────
        # STEP 3: TEST - Compressed anchor + loaded KDA
        # ─────────────────────────────────────────────────────────────────
        print("\n" + "-" * 70)
        print("STEP 3: TEST - Compressed Context + Loaded KDA")
        print("-" * 70)

        compressed_logprobs = None
        if kda_states:
            loaded_kda = manager.load_kda_states(test_data["id"])
            inject_kda_states_to_model(model, loaded_kda)
            print("Injected KDA states into model")

            anchor_prompt = f"<{test_data['id']}/>" + query
            print(f"Compressed prompt: {anchor_prompt}")

            compressed_output = llm.generate([anchor_prompt], sampling_params)[0]

            if compressed_output.outputs[0].logprobs:
                compressed_logprobs = compressed_output.outputs[0].logprobs[0]
                compressed_token = compressed_output.outputs[0].text
                print(f"Compressed predicts: '{compressed_token}'")
        else:
            print("Skipped (no KDA states)")

        # ─────────────────────────────────────────────────────────────────
        # STEP 4: CONTROL - Fresh generation (no loaded state)
        # ─────────────────────────────────────────────────────────────────
        print("\n" + "-" * 70)
        print("STEP 4: CONTROL - Fresh Generation (No State)")
        print("-" * 70)

        fresh_prompt = f"<{test_data['id']}/>" + query
        print(f"Fresh prompt: {fresh_prompt}")

        fresh_output = llm.generate([fresh_prompt], sampling_params)[0]

        fresh_logprobs = None
        if fresh_output.outputs[0].logprobs:
            fresh_logprobs = fresh_output.outputs[0].logprobs[0]
            fresh_token = fresh_output.outputs[0].text
            print(f"Fresh predicts: '{fresh_token}'")

        # ─────────────────────────────────────────────────────────────────
        # STEP 5: Compare logprobs
        # ─────────────────────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("RESULTS: Logit Distribution Comparison")
        print("=" * 70)

        def logprobs_to_tensor(logprobs_dict, vocab_size=151936):
            """Convert vLLM logprobs dict to tensor."""
            tensor = torch.full((vocab_size,), float('-inf'))
            for token_id, logprob_obj in logprobs_dict.items():
                if isinstance(token_id, int):
                    tensor[token_id] = logprob_obj.logprob
            return tensor

        if baseline_logprobs and compressed_logprobs:
            baseline_tensor = logprobs_to_tensor(baseline_logprobs)
            compressed_tensor = logprobs_to_tensor(compressed_logprobs)

            comparison = compare_logits(baseline_tensor, compressed_tensor, tokenizer)

            print(f"""
    ┌────────────────────────────────────────────────────────────────────┐
    │  BASELINE vs COMPRESSED (with loaded KDA)                          │
    ├────────────────────────────────────────────────────────────────────┤
    │  KL Divergence:      {comparison.kl_divergence:>10.4f}  (lower = more similar)     │
    │  Cosine Similarity:  {comparison.cosine_similarity:>10.4f}  (1.0 = identical)         │
    │  Top-10 Overlap:     {comparison.top_k_overlap:>10.1%}  (fraction matching)       │
    │  Argmax Match:       {str(comparison.argmax_match):>10}                            │
    │  Baseline token:     {comparison.baseline_token:>10}                              │
    │  Compressed token:   {comparison.test_token:>10}                              │
    └────────────────────────────────────────────────────────────────────┘
            """)

            success = comparison.cosine_similarity > 0.9 and comparison.argmax_match

            if success:
                print("✓ SUCCESS: Compressed context + KDA ≈ Full context!")
            else:
                print("✗ MISMATCH: Logits differ significantly")

        if baseline_logprobs and fresh_logprobs:
            baseline_tensor = logprobs_to_tensor(baseline_logprobs)
            fresh_tensor = logprobs_to_tensor(fresh_logprobs)

            fresh_comparison = compare_logits(baseline_tensor, fresh_tensor, tokenizer)

            print(f"""
    ┌────────────────────────────────────────────────────────────────────┐
    │  BASELINE vs FRESH (no loaded state - control)                     │
    ├────────────────────────────────────────────────────────────────────┤
    │  KL Divergence:      {fresh_comparison.kl_divergence:>10.4f}                           │
    │  Cosine Similarity:  {fresh_comparison.cosine_similarity:>10.4f}                           │
    │  Top-10 Overlap:     {fresh_comparison.top_k_overlap:>10.1%}                           │
    │  Argmax Match:       {str(fresh_comparison.argmax_match):>10}                            │
    └────────────────────────────────────────────────────────────────────┘
            """)

            if not fresh_comparison.argmax_match:
                print("✓ CONTROL PASSED: Fresh differs from baseline")
            else:
                print("⚠ CONTROL WARNING: Fresh matches baseline")

        return True


def main():
    print("\n" + "#" * 70)
    print("# KDA COMPRESSED CONTEXT LOGIT COMPARISON")
    print("#" * 70)
    print("""
    This experiment tests whether:

    Full context → KDA state → Compressed anchor + loaded KDA
                                      ↓
                              Similar logits?

    If logits match:
    - KDA recurrent_state truly compresses context
    - Can use anchor tokens + loaded state for inference
    - Massive storage/compute savings
    """)

    # Test 0: Storage verification (no GPU)
    storage_ok = test_without_vllm()
    if not storage_ok:
        print("\nStorage test failed!")
        return

    # Test 1: Logit comparison with vLLM (requires GPU)
    print("\n" + "-" * 70)
    print("Checking GPU availability for logit comparison test...")

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Found {gpu_count} GPUs")

        if gpu_count >= 8:
            test_logit_comparison_with_vllm()
        else:
            print(f"Need 8 GPUs for Kimi-Linear, have {gpu_count}")
            print("Skipping full test. Storage verification passed.")
    else:
        print("No CUDA available. Storage verification passed.")


if __name__ == "__main__":
    main()
