#!/usr/bin/env python3
"""
Direct model loading to access KDA state.
Bypasses vLLM's multi-process architecture to access model internals.
"""

import torch
import os
import safetensors.torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    print(f"GPUs available: {torch.cuda.device_count()}")

    storage_path = "/tmp/anchor_test"
    os.makedirs(storage_path, exist_ok=True)

    model_name = "moonshotai/Kimi-Linear-48B-A3B-Instruct"

    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"\nLoading model directly (this may take a while)...")
    print("Note: Using device_map='auto' for multi-GPU distribution")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print("Model loaded!")

    # Explore model structure
    print("\n" + "="*60)
    print("Exploring model structure...")
    print("="*60)

    print(f"Model type: {type(model).__name__}")

    # Find layers with KDA or MLA
    kda_layers = []
    mla_layers = []
    all_layers = []

    for name, module in model.named_modules():
        module_name = type(module).__name__
        if 'Attention' in module_name or 'KDA' in module_name or 'MLA' in module_name or 'Delta' in module_name:
            all_layers.append((name, module_name))
            if 'KDA' in module_name or 'Delta' in module_name or 'Linear' in module_name:
                kda_layers.append((name, module))
            elif 'MLA' in module_name:
                mla_layers.append((name, module))

    print(f"\nFound {len(all_layers)} attention-related layers")
    print(f"Sample layers:")
    for name, mod_type in all_layers[:10]:
        print(f"  {name}: {mod_type}")

    # Check for recurrent state
    print("\n" + "="*60)
    print("Looking for recurrent/KV state...")
    print("="*60)

    for name, module in list(model.named_modules())[:50]:
        # Check for state tensors
        if hasattr(module, 'kv_cache'):
            print(f"Found kv_cache in {name}")
        if hasattr(module, 'recurrent_state'):
            print(f"Found recurrent_state in {name}")
        if hasattr(module, 'conv_state'):
            print(f"Found conv_state in {name}")
        if hasattr(module, 'state'):
            print(f"Found state in {name}")

    # Run inference
    print("\n" + "="*60)
    print("Running inference...")
    print("="*60)

    context = "Alice is a software engineer from Tokyo. She has a dog named Mochi."
    prompt = f"<|im_start|>user\n{context}\n\nWhat is Alice's dog's name?<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print(f"Input tokens: {inputs['input_ids'].shape}")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"Response: {response}")

    # Check model state after inference
    print("\n" + "="*60)
    print("Checking model state after inference...")
    print("="*60)

    # Look for any cached state
    for name, module in model.named_modules():
        for attr in ['kv_cache', 'cache', 'state', 'recurrent_state', 'conv_state_q', 'conv_state_k', 'conv_state_v']:
            if hasattr(module, attr):
                val = getattr(module, attr)
                if val is not None:
                    if isinstance(val, torch.Tensor):
                        print(f"{name}.{attr}: tensor shape {val.shape}")
                    elif isinstance(val, (list, tuple)) and len(val) > 0:
                        print(f"{name}.{attr}: {type(val).__name__} of length {len(val)}")
                        if hasattr(val[0], 'shape'):
                            print(f"  First element shape: {val[0].shape}")

    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)

if __name__ == "__main__":
    main()
