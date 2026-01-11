#!/usr/bin/env python3
"""
Experiment 07: Kimi-Linear KDA State Extraction
================================================

QUESTION: Does Kimi-Linear's KDA state actually compress context?

WHY THIS MATTERS:
Our previous experiments (01-06) showed that NoPE without KDA cannot
compress context - K,V injection provides no accuracy benefit.

Kimi-Linear has KDA (Key-Driven Attention / linear attention) which
accumulates context into a recurrent state. This state SHOULD contain
compressed context information.

ARCHITECTURE:
Kimi-Linear is a hybrid model:
- KDA layers: Linear attention with recurrent state S
- MLA layers: Standard attention with K,V cache

MODEL: moonshotai/Kimi-Linear-48B-A3B-Instruct
- 48B total params, 3B activated (MoE)
- 1M context support
- 75% KV cache reduction

WHAT WE TEST:
1. Full context generation (baseline)
2. Extract KDA state after context processing
3. Restore KDA state and generate (should match baseline)
4. Compare with fresh generation (no state)

REQUIREMENTS:
- 8× L4 GPUs (or 2× A100 40GB)
- vLLM with Kimi-Linear support

Run: python tests/anchor_experiments/07_kimi_linear_kda.py
"""

import torch
from typing import Optional, Tuple, List, Dict, Any

# Configuration
MODEL_NAME = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
TENSOR_PARALLEL_SIZE = 8  # For 8× L4
MAX_MODEL_LEN = 32768  # Start small, increase as needed


def setup_vllm_engine():
    """Initialize vLLM with Kimi-Linear."""
    from vllm import LLM, SamplingParams

    print(f"\nLoading {MODEL_NAME}...")
    print(f"Tensor Parallel: {TENSOR_PARALLEL_SIZE}")

    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
    )

    print("Model loaded successfully!")
    return llm


def test_basic_generation(llm):
    """Test basic generation works."""
    from vllm import SamplingParams

    print("\n" + "=" * 70)
    print("TEST 1: Basic Generation")
    print("=" * 70)

    prompt = "Hello, my name is Alice and I live in Tokyo."
    sampling_params = SamplingParams(
        max_tokens=20,
        temperature=0.0,  # Deterministic
    )

    outputs = llm.generate([prompt], sampling_params)

    print(f"\nPrompt: {prompt}")
    print(f"Output: {outputs[0].outputs[0].text}")

    return outputs[0]


def test_kda_state_extraction(llm):
    """
    Test KDA state extraction and restoration.

    This is the key experiment - can we:
    1. Process context and extract KDA state
    2. Save the state
    3. Restore and continue generation
    4. Get the same output as full context?
    """
    from vllm import SamplingParams

    print("\n" + "=" * 70)
    print("TEST 2: KDA State Extraction & Restoration")
    print("=" * 70)

    # Context and query
    context = """Hi, my name is Alice and I work as a software engineer at a startup in Tokyo.
I've been living here for 5 years and I really enjoy the city.
My hobbies include hiking, photography, and cooking Japanese food.
Last weekend I went to Mount Fuji and took some amazing photos."""

    query = "\n\nUser: What are your hobbies?\nAssistant:"

    full_prompt = context + query

    sampling_params = SamplingParams(
        max_tokens=50,
        temperature=0.0,
    )

    # 1. Full context generation (baseline)
    print("\n1. FULL CONTEXT (baseline):")
    print(f"   Context length: {len(context)} chars")

    full_outputs = llm.generate([full_prompt], sampling_params)
    full_response = full_outputs[0].outputs[0].text
    print(f"   Response: {full_response[:100]}...")

    # 2. Try to access internal state
    # Note: This depends on vLLM's internal API
    print("\n2. Attempting KDA state extraction...")

    # The actual implementation depends on vLLM's Kimi-Linear integration
    # We need to check if vLLM exposes the KDA state

    try:
        # This is exploratory - check what's available
        model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        print(f"   Model type: {type(model)}")

        # Check for KDA-related attributes
        if hasattr(model, 'get_kda_state'):
            print("   ✓ Model has get_kda_state method")
        else:
            print("   ✗ Model does not expose get_kda_state directly")
            print("   → Need to check Kimi-Linear specific API")

    except Exception as e:
        print(f"   Error accessing model internals: {e}")

    return {
        'full_response': full_response,
        'context_length': len(context),
    }


def test_anchor_with_kimi(llm):
    """
    Test anchor-based compression with Kimi-Linear.

    Even without direct KDA state access, we can test if:
    1. Anchor text + fresh computation matches full context
    2. The model understands anchor format
    """
    from vllm import SamplingParams

    print("\n" + "=" * 70)
    print("TEST 3: Anchor-Based Compression")
    print("=" * 70)

    # Full context
    context = """Hi, my name is Alice and I work as a software engineer at a startup in Tokyo.
I've been living here for 5 years and I really enjoy the city.
My hobbies include hiking, photography, and cooking Japanese food."""

    # Semantic anchor (5-word summary)
    anchor = "<alice-software-engineer-tokyo-hiking/>"

    # Instruction prefix
    instruction = """[COMPRESSED CONTEXT: The following anchor contains key information from previous conversation.]
Previous: """

    query = "\n\nUser: What do you do for work?\nAssistant:"

    sampling_params = SamplingParams(
        max_tokens=50,
        temperature=0.0,
    )

    # 1. Full context
    print("\n1. FULL CONTEXT:")
    full_prompt = context + query
    full_out = llm.generate([full_prompt], sampling_params)[0].outputs[0].text
    print(f"   {full_out[:100]}...")

    # 2. Anchor only
    print("\n2. ANCHOR ONLY:")
    anchor_prompt = instruction + anchor + query
    anchor_out = llm.generate([anchor_prompt], sampling_params)[0].outputs[0].text
    print(f"   {anchor_out[:100]}...")

    # 3. Compare
    print("\n3. COMPARISON:")
    print(f"   Full context mentions 'software engineer': {'software' in full_out.lower()}")
    print(f"   Anchor mentions 'software engineer': {'software' in anchor_out.lower()}")

    return {
        'full_output': full_out,
        'anchor_output': anchor_out,
    }


def explore_kda_api(llm):
    """
    Explore what KDA-related APIs are available in vLLM's Kimi-Linear.
    """
    print("\n" + "=" * 70)
    print("EXPLORATION: KDA API Discovery")
    print("=" * 70)

    try:
        # Get model reference
        engine = llm.llm_engine

        print("\n1. Engine attributes:")
        for attr in dir(engine):
            if 'kv' in attr.lower() or 'kda' in attr.lower() or 'state' in attr.lower():
                print(f"   - {attr}")

        # Check model config
        print("\n2. Model config:")
        if hasattr(engine, 'model_config'):
            config = engine.model_config
            print(f"   Model: {config.model}")
            print(f"   Max len: {config.max_model_len}")

        # Check for KV cache manager
        print("\n3. KV Cache manager:")
        if hasattr(engine, 'cache_config'):
            cache = engine.cache_config
            print(f"   Block size: {cache.block_size}")
            print(f"   GPU blocks: {cache.num_gpu_blocks}")

    except Exception as e:
        print(f"Error during exploration: {e}")


def main():
    print("\n" + "#" * 70)
    print("# KIMI-LINEAR KDA STATE EXPERIMENT")
    print("#" * 70)
    print("""
    This experiment tests Kimi-Linear's KDA (linear attention) state.

    Key hypothesis: KDA state contains compressed context that can be
    saved and restored, unlike standard K,V cache which requires all
    positions.

    Requirements:
    - 8× L4 GPUs (192GB total)
    - vLLM with Kimi-Linear support
    """)

    # Check if we can import vLLM
    try:
        from vllm import LLM
        print("✓ vLLM imported successfully")
    except ImportError:
        print("✗ vLLM not installed")
        print("  Run: pip install vllm")
        return

    # Check GPU availability
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✓ CUDA available: {gpu_count} GPUs")
        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {name} ({mem:.1f}GB)")
    else:
        print("✗ CUDA not available - need GPUs for Kimi-Linear")
        return

    # Initialize model
    print("\n" + "-" * 70)
    print("Initializing Kimi-Linear...")
    print("-" * 70)

    try:
        llm = setup_vllm_engine()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nMake sure you have:")
        print("  1. Enough GPU memory (8× L4 or 2× A100)")
        print("  2. vLLM with Kimi-Linear support")
        print("  3. Model access (may need HF token)")
        return

    # Run experiments
    results = {}

    # Test 1: Basic generation
    results['basic'] = test_basic_generation(llm)

    # Test 2: KDA state extraction
    results['kda'] = test_kda_state_extraction(llm)

    # Test 3: Anchor compression
    results['anchor'] = test_anchor_with_kimi(llm)

    # Exploration
    explore_kda_api(llm)

    # Summary
    print("\n" + "#" * 70)
    print("# SUMMARY")
    print("#" * 70)
    print("""
    Next steps based on findings:
    1. If KDA state is accessible → implement state save/restore
    2. If not directly accessible → work with vLLM team or modify source
    3. Anchor-based compression works regardless → test at scale
    """)


if __name__ == "__main__":
    main()
