#!/usr/bin/env python3
"""Test KDA state save/restore with AnchorConnector."""

from vllm import LLM, SamplingParams
import torch
import os

def main():
    print(f"GPUs available: {torch.cuda.device_count()}")

    # Load Kimi-Linear
    print("\nLoading Kimi-Linear...")
    llm = LLM(
        model="moonshotai/Kimi-Linear-48B-A3B-Instruct",
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.85,
        max_model_len=8192,
        trust_remote_code=True,
    )
    print("Model loaded!")

    # Context to save
    context = """Alice is a software engineer from Tokyo who loves hiking.
She works at a startup building AI systems. Her favorite hiking spot is Mount Takao.
She has a dog named Mochi who sometimes joins her on easy trails.
Alice is 28 years old and has been coding since she was 15."""

    # Test 1: Generate with full context
    print("\n" + "="*60)
    print("TEST 1: Generate with full context")
    print("="*60)

    prompt1 = f"{context}\n\nQuestion: What is Alice's dog's name?"
    sampling_params = SamplingParams(temperature=0.1, max_tokens=50)

    outputs = llm.generate([prompt1], sampling_params)
    answer1 = outputs[0].outputs[0].text
    print(f"Q: What is Alice's dog's name?")
    print(f"A: {answer1}")

    # Test 2: Generate without context (should fail or give generic answer)
    print("\n" + "="*60)
    print("TEST 2: Generate WITHOUT context (baseline)")
    print("="*60)

    prompt2 = "Question: What is Alice's dog's name?"
    outputs = llm.generate([prompt2], sampling_params)
    answer2 = outputs[0].outputs[0].text
    print(f"Q: What is Alice's dog's name?")
    print(f"A: {answer2}")

    # Test 3: Test AnchorConnector save/load simulation
    print("\n" + "="*60)
    print("TEST 3: AnchorConnector metadata simulation")
    print("="*60)

    from vllm.distributed.kv_transfer.kv_connector.v1.anchor_connector import (
        AnchorConnector,
        AnchorConnectorMetadata,
    )

    # Create anchor for Alice context
    anchor_id = "<alice-software-tokyo-mochi/>"
    meta = AnchorConnectorMetadata()

    # Simulate block allocation (16 tokens per block, ~80 tokens in context)
    num_tokens = len(context.split()) * 2  # rough estimate
    num_blocks = (num_tokens + 15) // 16
    block_ids = list(range(num_blocks))

    meta.add_anchor(
        anchor_id=anchor_id,
        block_ids=block_ids,
        block_size=16,
        num_tokens=num_tokens,
        is_store=True,
    )

    print(f"Created anchor: {anchor_id}")
    print(f"  - Estimated tokens: {num_tokens}")
    print(f"  - Blocks allocated: {num_blocks}")
    print(f"  - Slot mapping shape: {meta.anchors[0].slot_mapping.shape}")

    # Test 4: Verify context matters
    print("\n" + "="*60)
    print("TEST 4: Different questions about context")
    print("="*60)

    questions = [
        "Where does Alice work?",
        "How old is Alice?",
        "What is Alice's favorite hiking spot?",
    ]

    for q in questions:
        prompt = f"{context}\n\nQuestion: {q}"
        outputs = llm.generate([prompt], sampling_params)
        print(f"Q: {q}")
        print(f"A: {outputs[0].outputs[0].text.strip()}\n")

    print("="*60)
    print("All tests completed!")
    print("="*60)
    print("\nNOTE: Full anchor save/restore requires vLLM integration.")
    print("The AnchorConnector is ready but needs to be wired into")
    print("vLLM's KV transfer pipeline to actually save/restore state.")

if __name__ == "__main__":
    main()
