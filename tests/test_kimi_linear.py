#!/usr/bin/env python3
"""Test Kimi-Linear with AnchorConnector on vLLM."""

from vllm import LLM, SamplingParams
import torch

def main():
    print(f"GPUs available: {torch.cuda.device_count()}")

    # Load Kimi-Linear
    print("\nLoading Kimi-Linear-48B-A3B-Instruct...")
    llm = LLM(
        model="moonshotai/Kimi-Linear-48B-A3B-Instruct",
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.85,
        max_model_len=8192,
        trust_remote_code=True,
    )
    print("Model loaded successfully!")

    # Test basic generation
    prompts = [
        "Alice is a software engineer from Tokyo who loves hiking.",
        "What is the capital of France?",
    ]
    sampling_params = SamplingParams(temperature=0.7, max_tokens=100)

    print("\n" + "="*50)
    print("Testing basic generation:")
    print("="*50)

    outputs = llm.generate(prompts, sampling_params)
    for i, output in enumerate(outputs):
        print(f"\nPrompt {i+1}: {output.prompt[:60]}...")
        print(f"Output: {output.outputs[0].text}")

    # Test AnchorConnector import
    print("\n" + "="*50)
    print("Testing AnchorConnector:")
    print("="*50)

    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.anchor_connector import (
            AnchorConnector,
            AnchorConnectorMetadata,
        )
        print("AnchorConnector imported successfully!")

        # Test metadata creation
        meta = AnchorConnectorMetadata()
        meta.add_anchor(
            anchor_id="<alice-software-tokyo/>",
            block_ids=[0, 1, 2],
            block_size=16,
            num_tokens=40,
            is_store=True,
        )
        print(f"Created anchor metadata with {len(meta.anchors)} anchor(s)")
        print(f"  - anchor_id: {meta.anchors[0].anchor_id}")
        print(f"  - num_tokens: {meta.anchors[0].num_tokens}")
        print(f"  - slot_mapping shape: {meta.anchors[0].slot_mapping.shape}")

    except Exception as e:
        print(f"AnchorConnector test failed: {e}")

    print("\n" + "="*50)
    print("All tests completed!")
    print("="*50)

if __name__ == "__main__":
    main()
