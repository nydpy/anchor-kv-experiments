#!/usr/bin/env python3
"""
Experiment 02: Keyword K,V Extraction Hypothesis
================================================

QUESTION: Does extracting K,V only at keyword positions preserve meaning?

HYPOTHESIS:
If we extract K,V at keyword positions (e.g., [0, 3, 7]), the compressed
K,V might preserve semantic meaning because causal attention accumulates
context into later positions.

WHAT IT TESTS:
- Extract keywords from text (stop-word removal)
- Find token positions of keywords
- Extract K,V only at those positions (~30% compression)
- Compare output with full K,V

MODEL: McGill-NLP/codellm_1b_nope (NoPE, no KDA)

EXPECTED FINDING:
NO - keyword-only K,V produces garbage output. Without KDA, standard
attention needs ALL K,V positions. Each position N only "knows"
tokens 0..N, so removing positions loses information.

Run: python tests/anchor_experiments/02_keyword_kv_hypothesis.py
"""

import sys
from pathlib import Path
import torch

# Globals
MODEL = None
TOKENIZER = None
DEVICE = "cpu"


def load_model():
    """Load the NoPE model (only once)."""
    global MODEL, TOKENIZER, DEVICE

    if MODEL is not None:
        return MODEL, TOKENIZER

    print("\n" + "=" * 70)
    print("LOADING MODEL")
    print("=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import time

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")

    model_name = "McGill-NLP/codellm_1b_nope"
    print(f"Loading: {model_name}")

    start = time.time()

    TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    MODEL = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
    )

    if DEVICE == "cpu":
        MODEL = MODEL.to(DEVICE)

    print(f"Loaded in {time.time() - start:.1f}s")
    return MODEL, TOKENIZER


def tokenize_and_show(text):
    """Tokenize text and show each token with position."""
    model, tokenizer = load_model()

    # Tokenize
    tokens = tokenizer.tokenize(text)
    input_ids = tokenizer.encode(text, add_special_tokens=False)

    print(f"\n  Original text: \"{text}\"")
    print(f"\n  Tokens ({len(tokens)}):")
    print(f"  {'Pos':<5} {'Token':<15} {'ID':<10}")
    print(f"  {'-'*5} {'-'*15} {'-'*10}")

    for i, (tok, tok_id) in enumerate(zip(tokens, input_ids)):
        print(f"  {i:<5} {repr(tok):<15} {tok_id:<10}")

    return tokens, input_ids


def find_keyword_positions(tokens, keywords):
    """Find positions of keywords in token list."""
    positions = {}

    for keyword in keywords:
        keyword_lower = keyword.lower()
        for i, tok in enumerate(tokens):
            # Clean token (remove special chars like Ġ for space)
            clean_tok = tok.replace('Ġ', '').replace('▁', '').lower()
            if clean_tok == keyword_lower or keyword_lower in clean_tok:
                if keyword not in positions:
                    positions[keyword] = i
                    break

    return positions


def extract_full_kv(text):
    """Extract K,V for all positions."""
    model, tokenizer = load_model()

    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True, return_dict=True)

    past_kv = outputs.past_key_values

    # Convert to dict: {layer: (K, V)}
    # K, V shape: [batch, heads, seq_len, head_dim]
    kv_cache = {}
    for layer_idx, (keys, values) in enumerate(past_kv):
        # Keep as [seq_len, heads, head_dim]
        k = keys[0].transpose(0, 1).cpu()  # [seq_len, heads, head_dim]
        v = values[0].transpose(0, 1).cpu()
        kv_cache[layer_idx] = (k, v)

    return kv_cache, input_ids.shape[1]


def extract_keyword_kv(full_kv_cache, keyword_positions):
    """Extract K,V only at keyword positions."""
    keyword_kv = {}

    positions = sorted(keyword_positions.values())

    for layer_idx, (k, v) in full_kv_cache.items():
        # k, v shape: [seq_len, heads, head_dim]
        # Extract only keyword positions
        k_keyword = k[positions]  # [num_keywords, heads, head_dim]
        v_keyword = v[positions]
        keyword_kv[layer_idx] = (k_keyword, v_keyword)

    return keyword_kv, positions


def compare_kv_caches(full_kv, keyword_kv, keyword_positions):
    """Compare full K,V vs keyword-only K,V."""
    print(f"\n  K,V Cache Comparison:")
    print(f"  {'-'*60}")

    # Full K,V
    k_full, v_full = full_kv[0]
    print(f"\n  Full K,V:")
    print(f"    Shape: K={list(k_full.shape)}, V={list(v_full.shape)}")
    print(f"    Seq len: {k_full.shape[0]}")
    total_full = sum(k.numel() + v.numel() for k, v in full_kv.values()) * 4
    print(f"    Total size: {total_full / 1024:.2f} KB")

    # Keyword K,V
    k_kw, v_kw = keyword_kv[0]
    print(f"\n  Keyword K,V:")
    print(f"    Shape: K={list(k_kw.shape)}, V={list(v_kw.shape)}")
    print(f"    Seq len: {k_kw.shape[0]} (positions: {sorted(keyword_positions.values())})")
    total_kw = sum(k.numel() + v.numel() for k, v in keyword_kv.values()) * 4
    print(f"    Total size: {total_kw / 1024:.2f} KB")

    # Compression ratio
    ratio = total_full / total_kw
    print(f"\n  Compression: {ratio:.1f}x ({100 - 100/ratio:.1f}% reduction)")


def test_generation_with_kv(prompt, past_key_values, label):
    """Test generation with injected K,V."""
    model, tokenizer = load_model()

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

    print(f"\n  {label}:")
    print(f"    Prompt: \"{prompt}\"")

    model.eval()
    with torch.no_grad():
        # Generate with past_key_values
        generated = model.generate(
            input_ids,
            past_key_values=past_key_values,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    output = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"    Output: \"{output}\"")

    return output


def convert_to_past_key_values(kv_cache):
    """Convert our kv_cache format to HuggingFace past_key_values format."""
    past_key_values = []

    for layer_idx in sorted(kv_cache.keys()):
        k, v = kv_cache[layer_idx]
        # Our format: [seq_len, heads, head_dim]
        # HF format: [batch, heads, seq_len, head_dim]
        k = k.transpose(0, 1).unsqueeze(0).to(DEVICE)
        v = v.transpose(0, 1).unsqueeze(0).to(DEVICE)
        past_key_values.append((k, v))

    return tuple(past_key_values)


def run_test(text, keywords):
    """Run the full test."""
    print("\n" + "=" * 70)
    print("TEST: KEYWORD K,V EXTRACTION")
    print("=" * 70)

    # Step 1: Tokenize and show positions
    print("\n" + "-" * 70)
    print("STEP 1: TOKENIZATION")
    print("-" * 70)

    tokens, input_ids = tokenize_and_show(text)

    # Step 2: Find keyword positions
    print("\n" + "-" * 70)
    print("STEP 2: FIND KEYWORD POSITIONS")
    print("-" * 70)

    print(f"\n  Keywords to find: {keywords}")

    keyword_positions = find_keyword_positions(tokens, keywords)

    print(f"\n  Found positions:")
    for kw, pos in sorted(keyword_positions.items(), key=lambda x: x[1]):
        print(f"    {kw:<15} → position {pos:<3} (token: {repr(tokens[pos])})")

    if len(keyword_positions) != len(keywords):
        missing = set(keywords) - set(keyword_positions.keys())
        print(f"\n  WARNING: Could not find: {missing}")

    # Step 3: Extract full K,V
    print("\n" + "-" * 70)
    print("STEP 3: EXTRACT FULL K,V CACHE")
    print("-" * 70)

    full_kv, seq_len = extract_full_kv(text)

    print(f"\n  Extracted K,V for {seq_len} tokens")
    k, v = full_kv[0]
    print(f"  Layer 0 shape: K={list(k.shape)}, V={list(v.shape)}")
    print(f"  Total layers: {len(full_kv)}")

    # Step 4: Extract keyword-only K,V
    print("\n" + "-" * 70)
    print("STEP 4: EXTRACT KEYWORD-ONLY K,V")
    print("-" * 70)

    keyword_kv, positions = extract_keyword_kv(full_kv, keyword_positions)

    print(f"\n  Extracted K,V at positions: {positions}")
    k_kw, v_kw = keyword_kv[0]
    print(f"  Layer 0 shape: K={list(k_kw.shape)}, V={list(v_kw.shape)}")

    # Step 5: Compare
    print("\n" + "-" * 70)
    print("STEP 5: COMPARE K,V CACHES")
    print("-" * 70)

    compare_kv_caches(full_kv, keyword_kv, keyword_positions)

    # Step 6: Show what each K,V "knows"
    print("\n" + "-" * 70)
    print("STEP 6: WHAT EACH K,V POSITION 'KNOWS' (CAUSAL)")
    print("-" * 70)

    print(f"\n  Causal attention means K,V[i] has seen tokens 0..i")
    print()
    for kw, pos in sorted(keyword_positions.items(), key=lambda x: x[1]):
        context_tokens = tokens[:pos+1]
        print(f"  K,V[{pos}] '{kw}' knows: {context_tokens}")

    # Step 7: Test generation (if model supports it)
    print("\n" + "-" * 70)
    print("STEP 7: GENERATION COMPARISON")
    print("-" * 70)

    continuation = " result ="

    # Fresh generation (no K,V)
    print(f"\n  Testing continuation: \"{continuation}\"")

    test_generation_with_kv(continuation, None, "Fresh (no K,V)")

    # With full K,V
    full_past = convert_to_past_key_values(full_kv)
    test_generation_with_kv(continuation, full_past, "With FULL K,V")

    # With keyword K,V
    keyword_past = convert_to_past_key_values(keyword_kv)
    test_generation_with_kv(continuation, keyword_past, "With KEYWORD K,V")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


def main():
    # Test case 1: Code
    print("\n" + "#" * 70)
    print("# TEST CASE 1: CODE")
    print("#" * 70)

    text1 = "def square(x):\n    return x * x"
    keywords1 = ["def", "square", "return"]
    run_test(text1, keywords1)

    input("\nPress Enter for next test...")

    # Test case 2: Natural language
    print("\n" + "#" * 70)
    print("# TEST CASE 2: NATURAL LANGUAGE")
    print("#" * 70)

    text2 = "Alice walked through the garden and admired the beautiful roses"
    keywords2 = ["Alice", "garden", "roses"]
    run_test(text2, keywords2)

    input("\nPress Enter for next test...")

    # Test case 3: Longer code
    print("\n" + "#" * 70)
    print("# TEST CASE 3: LONGER CODE")
    print("#" * 70)

    text3 = """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)"""
    keywords3 = ["fibonacci", "return", "recursive"]
    run_test(text3, keywords3)


if __name__ == "__main__":
    main()
