#!/usr/bin/env python3
"""
Test actual KDA state extraction from Kimi-Linear.

This test:
1. Runs inference and captures the KDA recurrent state
2. Saves it to disk
3. Verifies we can load it back
"""

import torch
import os
import safetensors.torch
from vllm import LLM, SamplingParams

def main():
    print(f"GPUs available: {torch.cuda.device_count()}")

    # Storage path for anchors
    storage_path = "/tmp/anchor_test"
    os.makedirs(storage_path, exist_ok=True)

    # Load model
    print("\nLoading Kimi-Linear...")
    llm = LLM(
        model="moonshotai/Kimi-Linear-48B-A3B-Instruct",
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.85,
        max_model_len=8192,
        trust_remote_code=True,
    )
    print("Model loaded!")

    # Access the model internals
    print("\n" + "="*60)
    print("Exploring model structure for KDA layers...")
    print("="*60)

    # Get the model from the engine
    try:
        # vLLM v1 structure
        model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    except AttributeError:
        try:
            # Alternative path
            model = llm.llm_engine.driver_worker.model_runner.model
        except AttributeError:
            print("Could not access model internals directly.")
            print("Trying alternative approach...")
            model = None

    if model is not None:
        # Find KDA layers
        print(f"\nModel type: {type(model).__name__}")

        kda_layers = []
        mla_layers = []

        for name, module in model.named_modules():
            module_name = type(module).__name__
            if 'KDA' in module_name or 'Delta' in module_name:
                kda_layers.append((name, module))
            elif 'MLA' in module_name:
                mla_layers.append((name, module))

        print(f"\nFound {len(kda_layers)} KDA layers")
        print(f"Found {len(mla_layers)} MLA layers")

        # Check for kv_cache attribute
        if kda_layers:
            name, layer = kda_layers[0]
            print(f"\nFirst KDA layer: {name}")
            print(f"  Type: {type(layer).__name__}")

            if hasattr(layer, 'kv_cache'):
                kv_cache = layer.kv_cache
                print(f"  kv_cache type: {type(kv_cache)}")
                if isinstance(kv_cache, (list, tuple)):
                    print(f"  kv_cache length: {len(kv_cache)}")
                    if len(kv_cache) > 0:
                        first = kv_cache[0]
                        if isinstance(first, (list, tuple)):
                            print(f"  First element is tuple of length: {len(first)}")
                            for i, t in enumerate(first):
                                if hasattr(t, 'shape'):
                                    print(f"    [{i}] shape: {t.shape}")
                        elif hasattr(first, 'shape'):
                            print(f"  First element shape: {first.shape}")
            else:
                print("  No kv_cache attribute found")
                # List all attributes
                attrs = [a for a in dir(layer) if not a.startswith('_')]
                print(f"  Available attributes: {attrs[:20]}...")

    # Run inference to populate KV cache
    print("\n" + "="*60)
    print("Running inference to populate KV cache...")
    print("="*60)

    context = """Alice is a software engineer from Tokyo who loves hiking.
She works at a startup building AI systems. Her favorite hiking spot is Mount Takao.
She has a dog named Mochi who sometimes joins her on easy trails."""

    prompt = f"{context}\n\nQuestion: What is Alice's dog's name?\nAnswer:"
    sampling_params = SamplingParams(temperature=0.1, max_tokens=20)

    outputs = llm.generate([prompt], sampling_params)
    print(f"Generated: {outputs[0].outputs[0].text}")

    # Try to extract state after inference
    print("\n" + "="*60)
    print("Attempting to extract KDA state after inference...")
    print("="*60)

    if model is not None and kda_layers:
        name, layer = kda_layers[0]

        if hasattr(layer, 'kv_cache'):
            kv_cache = layer.kv_cache

            if isinstance(kv_cache, (list, tuple)) and len(kv_cache) > 0:
                first = kv_cache[0]

                if isinstance(first, (list, tuple)) and len(first) == 4:
                    conv_q, conv_k, conv_v, recurrent = first

                    print(f"KDA state extracted from {name}:")
                    print(f"  conv_state_q shape: {conv_q.shape}")
                    print(f"  conv_state_k shape: {conv_k.shape}")
                    print(f"  conv_state_v shape: {conv_v.shape}")
                    print(f"  recurrent_state shape: {recurrent.shape}")

                    # Save to disk
                    save_path = os.path.join(storage_path, "alice_anchor.safetensors")
                    tensors = {
                        "conv_state_q": conv_q.cpu(),
                        "conv_state_k": conv_k.cpu(),
                        "conv_state_v": conv_v.cpu(),
                        "recurrent_state": recurrent.cpu(),
                    }
                    safetensors.torch.save_file(tensors, save_path)
                    print(f"\nSaved KDA state to: {save_path}")

                    # Load it back
                    loaded = safetensors.torch.load_file(save_path)
                    print(f"Loaded back successfully!")
                    print(f"  Keys: {list(loaded.keys())}")

                    # Verify shapes match
                    for key in tensors:
                        assert loaded[key].shape == tensors[key].shape
                    print("  All shapes match!")

                else:
                    print(f"Unexpected kv_cache structure: {type(first)}, len={len(first) if hasattr(first, '__len__') else 'N/A'}")
            else:
                print(f"kv_cache is empty or not a list/tuple")
        else:
            print("No kv_cache found on layer")
    else:
        print("Could not access model layers")

    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)

if __name__ == "__main__":
    main()
