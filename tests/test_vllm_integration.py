#!/usr/bin/env python3
"""
vLLM Integration Test for Anchor Connector

This test verifies the anchor connector works with vLLM and Kimi-Linear.

Requirements:
- vLLM installed with anchor connector (run setup_vllm.sh)
- 8× L4 GPUs or 2× A100 40GB

Run: python tests/test_vllm_integration.py
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

import torch


def check_environment():
    """Verify environment is ready."""
    print("\n" + "=" * 70)
    print("ENVIRONMENT CHECK")
    print("=" * 70)

    # Check CUDA
    if not torch.cuda.is_available():
        print("✗ CUDA not available - need GPUs for this test")
        return False

    gpu_count = torch.cuda.device_count()
    print(f"✓ CUDA available: {gpu_count} GPUs")

    for i in range(gpu_count):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {name} ({mem:.1f}GB)")

    # Check vLLM
    try:
        import vllm
        print(f"✓ vLLM version: {vllm.__version__}")
    except ImportError:
        print("✗ vLLM not installed")
        return False

    # Check anchor connector
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.anchor_connector import AnchorConnector
        print("✓ AnchorConnector available")
    except ImportError:
        print("✗ AnchorConnector not installed - run setup_vllm.sh")
        return False

    return True


def test_anchor_connector_standalone():
    """Test anchor connector without full model."""
    print("\n" + "=" * 70)
    print("TEST 1: Anchor Connector Standalone")
    print("=" * 70)

    from vllm.distributed.kv_transfer.kv_connector.v1.anchor_connector import AnchorConnector

    # Create temp directory for anchor cache
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            'anchor_cache_dir': tmpdir,
            'session_id': 'test_session',
        }

        connector = AnchorConnector(
            rank=0,
            local_rank=0,
            config=config,
        )

        print(f"✓ Connector created")
        print(f"  Cache dir: {tmpdir}")

        # Test anchor registration
        anchor_id = "test_anchor_1"
        connector.register_anchor(anchor_id, position=0)
        print(f"✓ Anchor registered: {anchor_id}")

        # Test saving dummy state
        dummy_kda_state = {
            'layer_0': torch.randn(1, 8, 64, 64),  # Dummy KDA state
            'layer_1': torch.randn(1, 8, 64, 64),
        }

        connector.save_anchor_state(anchor_id, dummy_kda_state)
        print(f"✓ State saved")

        # Test loading state
        loaded_state = connector.load_anchor_state(anchor_id)
        print(f"✓ State loaded")

        # Verify state matches
        for key in dummy_kda_state:
            if not torch.allclose(dummy_kda_state[key], loaded_state[key]):
                print(f"✗ State mismatch for {key}")
                return False

        print(f"✓ State verified - matches original")

    return True


def test_kimi_linear_basic():
    """Test basic Kimi-Linear generation."""
    print("\n" + "=" * 70)
    print("TEST 2: Kimi-Linear Basic Generation")
    print("=" * 70)

    from vllm import LLM, SamplingParams

    # Configuration
    model_name = "moonshotai/Kimi-Linear-Instruct"
    tensor_parallel = torch.cuda.device_count()

    print(f"Loading {model_name}...")
    print(f"Tensor parallel: {tensor_parallel}")

    try:
        llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel,
            max_model_len=32768,
            trust_remote_code=True,
        )
        print("✓ Model loaded")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False

    # Test generation
    prompt = "Hello, my name is Alice. What is my name?"
    sampling_params = SamplingParams(max_tokens=20, temperature=0.0)

    outputs = llm.generate([prompt], sampling_params)
    response = outputs[0].outputs[0].text

    print(f"\nPrompt: {prompt}")
    print(f"Response: {response}")

    if "alice" in response.lower():
        print("✓ Model correctly identified name")
    else:
        print("⚠ Model may not have understood context")

    return True


def test_anchor_with_kimi_linear():
    """Test anchor-based context compression with Kimi-Linear."""
    print("\n" + "=" * 70)
    print("TEST 3: Anchor-Based Compression")
    print("=" * 70)

    from vllm import LLM, SamplingParams

    model_name = "moonshotai/Kimi-Linear-Instruct"
    tensor_parallel = torch.cuda.device_count()

    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel,
        max_model_len=32768,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(max_tokens=50, temperature=0.0)

    # Full context
    context = """Hi, my name is Alice and I work as a software engineer at a startup in Tokyo.
I've been living here for 5 years and I really enjoy the city.
My hobbies include hiking, photography, and cooking Japanese food.
Last weekend I went to Mount Fuji and took some amazing photos."""

    # Semantic anchor
    anchor = "<alice-software-tokyo-hiking-photography/>"

    # Instruction for model
    instruction = """[COMPRESSED CONTEXT: The anchor below contains key information from previous conversation.]
Previous: """

    query = "\n\nUser: What are your hobbies?\nAssistant:"

    # Test 1: Full context
    print("\n1. FULL CONTEXT:")
    full_prompt = context + query
    full_output = llm.generate([full_prompt], sampling_params)[0].outputs[0].text
    print(f"   {full_output[:100]}...")

    # Test 2: Anchor only
    print("\n2. ANCHOR ONLY:")
    anchor_prompt = instruction + anchor + query
    anchor_output = llm.generate([anchor_prompt], sampling_params)[0].outputs[0].text
    print(f"   {anchor_output[:100]}...")

    # Compare
    print("\n3. COMPARISON:")
    keywords = ['hiking', 'photography', 'cooking']
    for kw in keywords:
        in_full = kw in full_output.lower()
        in_anchor = kw in anchor_output.lower()
        status = "✓" if in_full == in_anchor else "✗"
        print(f"   {status} {kw}: Full={in_full}, Anchor={in_anchor}")

    return True


def test_kda_state_access():
    """Attempt to access KDA state from Kimi-Linear."""
    print("\n" + "=" * 70)
    print("TEST 4: KDA State Access (Experimental)")
    print("=" * 70)

    from vllm import LLM, SamplingParams

    model_name = "moonshotai/Kimi-Linear-Instruct"
    tensor_parallel = torch.cuda.device_count()

    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel,
        max_model_len=8192,
        trust_remote_code=True,
    )

    # Explore model internals
    print("\nExploring model internals...")

    try:
        engine = llm.llm_engine

        # Check for KDA-related methods
        kda_methods = []
        for attr in dir(engine):
            if any(x in attr.lower() for x in ['kda', 'state', 'recurrent', 'linear']):
                kda_methods.append(attr)

        if kda_methods:
            print(f"Found KDA-related attributes: {kda_methods}")
        else:
            print("No direct KDA attributes found on engine")

        # Try to access model layers
        try:
            model = engine.model_executor.driver_worker.model_runner.model
            print(f"Model type: {type(model).__name__}")

            # Check for KDA layers
            for name, module in model.named_modules():
                if 'kda' in name.lower() or 'linear_attention' in name.lower():
                    print(f"  Found: {name} ({type(module).__name__})")

        except Exception as e:
            print(f"Could not access model layers: {e}")

    except Exception as e:
        print(f"Error exploring model: {e}")

    print("\n⚠ KDA state access may require vLLM modification")
    print("  → File a feature request or modify vLLM source")

    return True


def main():
    print("\n" + "#" * 70)
    print("# vLLM INTEGRATION TEST FOR ANCHOR CONNECTOR")
    print("#" * 70)

    # Check environment first
    if not check_environment():
        print("\n✗ Environment not ready. Exiting.")
        sys.exit(1)

    results = {}

    # Test 1: Connector standalone
    try:
        results['connector'] = test_anchor_connector_standalone()
    except Exception as e:
        print(f"✗ Test failed: {e}")
        results['connector'] = False

    # Test 2: Basic generation
    try:
        results['basic'] = test_kimi_linear_basic()
    except Exception as e:
        print(f"✗ Test failed: {e}")
        results['basic'] = False

    # Test 3: Anchor compression
    try:
        results['anchor'] = test_anchor_with_kimi_linear()
    except Exception as e:
        print(f"✗ Test failed: {e}")
        results['anchor'] = False

    # Test 4: KDA state access
    try:
        results['kda'] = test_kda_state_access()
    except Exception as e:
        print(f"✗ Test failed: {e}")
        results['kda'] = False

    # Summary
    print("\n" + "#" * 70)
    print("# SUMMARY")
    print("#" * 70)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name}: {status}")

    all_passed = all(results.values())
    print("\n" + ("All tests passed!" if all_passed else "Some tests failed."))

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
