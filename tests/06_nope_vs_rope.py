#!/usr/bin/env python3
"""
Experiment 06: NoPE vs RoPE Model Comparison
============================================

QUESTION: How do different position encoding schemes affect K,V extraction?

MODELS COMPARED:

NoPE (No Position Encoding): codellm_1b_nope
  - K,V has no position info baked in
  - K,V can theoretically be loaded at any position
  - But without KDA, context relationships are lost

RoPE (Rotary Position Encoding): GPT-2
  - K,V has position info baked in
  - K,V from position N "knows" it's at position N
  - Position mismatch occurs if loaded elsewhere

WHAT IT TESTS:
- Extract K,V at keyword positions for both models
- Compare output quality (cosine similarity, token matches)
- Test if position encoding affects compression viability

EXPECTED FINDING:
Both models struggle with keyword-only K,V extraction.
Architecture (NoPE vs RoPE) matters less than having KDA for context
accumulation. Neither can compress effectively without linear attention.

Run: python tests/anchor_experiments/06_nope_vs_rope.py
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F

MODELS = {}
TOKENIZERS = {}
DEVICE = "cpu"


def load_model(model_type):
    """Load model by type: 'nope' or 'rope'."""
    global MODELS, TOKENIZERS, DEVICE

    if model_type in MODELS:
        return MODELS[model_type], TOKENIZERS[model_type]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if model_type == "nope":
        model_name = "McGill-NLP/codellm_1b_nope"
        print(f"\nLoading NoPE model: {model_name}")
    else:  # rope
        model_name = "gpt2"
        print(f"\nLoading RoPE model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # Use float32 for CPU compatibility
    )
    model = model.to(DEVICE)
    model.eval()

    # GPT-2 needs pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    MODELS[model_type] = model
    TOKENIZERS[model_type] = tokenizer

    print(f"  Loaded on {DEVICE}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    return model, tokenizer


def extract_full_kv(text, model_type):
    """Extract full K,V cache from text."""
    model, tokenizer = load_model(model_type)

    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids, use_cache=True, return_dict=True)

    return outputs.past_key_values, input_ids.shape[1]


def extract_kv_at_positions(full_kv, positions):
    """Extract K,V only at specified positions."""
    extracted = []
    for keys, values in full_kv:
        # keys/values shape: [batch, heads, seq_len, head_dim]
        k = keys[:, :, positions, :]
        v = values[:, :, positions, :]
        extracted.append((k, v))
    return tuple(extracted)


def generate_with_kv(query, past_kv, model_type, n_tokens=3):
    """Generate tokens with injected K,V."""
    model, tokenizer = load_model(model_type)

    input_ids = tokenizer(query, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)

    all_logits = []
    all_tokens = []
    current_kv = past_kv

    with torch.no_grad():
        for step in range(n_tokens):
            outputs = model(
                input_ids,
                past_key_values=current_kv,
                use_cache=True,
                return_dict=True,
            )

            logits = outputs.logits[:, -1, :]
            all_logits.append(logits.cpu())

            next_token = torch.argmax(logits, dim=-1)
            all_tokens.append(next_token.item())

            input_ids = next_token.unsqueeze(0)
            current_kv = outputs.past_key_values

    return all_logits, all_tokens


def find_keyword_positions(text, keywords, tokenizer):
    """Find token positions of keywords."""
    tokens = tokenizer.tokenize(text)
    positions = []

    for keyword in keywords:
        kw_lower = keyword.lower()
        for i, tok in enumerate(tokens):
            # Handle different tokenizer formats
            clean = tok.replace('Ġ', '').replace('▁', '').replace('Ċ', '').lower()
            if kw_lower in clean or clean in kw_lower:
                positions.append(i)  # Token position (0-indexed)
                break

    # Always include last position (len-1 for 0-indexed)
    last_pos = len(tokens) - 1
    if last_pos not in positions and last_pos >= 0:
        positions.append(last_pos)

    return sorted(set(positions))


def run_comparison(context, keywords, query, model_type):
    """Run comparison for one model type."""
    model, tokenizer = load_model(model_type)

    print(f"\n{'='*60}")
    print(f"MODEL: {model_type.upper()} ({'No Position Encoding' if model_type == 'nope' else 'Rotary Position Encoding'})")
    print(f"{'='*60}")

    # Tokenize and show
    tokens = tokenizer.tokenize(context)
    print(f"\nContext: \"{context[:50]}...\"")
    print(f"Tokens: {len(tokens)}")
    print(f"Keywords: {keywords}")

    # Find keyword positions
    kw_positions = find_keyword_positions(context, keywords, tokenizer)
    print(f"Keyword positions: {kw_positions}")

    # Extract K,V
    full_kv, seq_len = extract_full_kv(context, model_type)
    keyword_kv = extract_kv_at_positions(full_kv, kw_positions)

    print(f"\nK,V shapes:")
    k_full, _ = full_kv[0]
    k_kw, _ = keyword_kv[0]
    print(f"  Full K,V:    {list(k_full.shape)} ({seq_len} positions)")
    print(f"  Keyword K,V: {list(k_kw.shape)} ({len(kw_positions)} positions)")

    # Generate with full K,V
    print(f"\nQuery: \"{query}\"")
    print(f"\nGenerating 3 tokens...")

    full_logits, full_tokens = generate_with_kv(query, full_kv, model_type, n_tokens=3)
    kw_logits, kw_tokens = generate_with_kv(query, keyword_kv, model_type, n_tokens=3)

    full_output = tokenizer.decode(full_tokens)
    kw_output = tokenizer.decode(kw_tokens)

    print(f"\n  Full K,V output:    \"{full_output}\" (tokens: {full_tokens})")
    print(f"  Keyword K,V output: \"{kw_output}\" (tokens: {kw_tokens})")

    # Compare logits
    cos_sims = []
    for i, (f_logits, k_logits) in enumerate(zip(full_logits, kw_logits)):
        cos = F.cosine_similarity(f_logits.float(), k_logits.float(), dim=-1).item()
        cos_sims.append(cos)

    print(f"\n  Cosine similarity (Full vs Keyword K,V):")
    for i, cos in enumerate(cos_sims):
        match = "✓" if full_tokens[i] == kw_tokens[i] else "✗"
        print(f"    Step {i+1}: {cos:.4f} {match}")

    avg_cos = sum(cos_sims) / len(cos_sims)
    matches = sum(1 for f, k in zip(full_tokens, kw_tokens) if f == k)

    print(f"\n  Average cosine: {avg_cos:.4f}")
    print(f"  Token matches: {matches}/3")
    print(f"  Output match: {'✓' if full_output == kw_output else '✗'}")

    return {
        'avg_cos': avg_cos,
        'matches': matches,
        'output_match': full_output == kw_output,
        'full_output': full_output,
        'kw_output': kw_output,
    }


def main():
    print("\n" + "#" * 70)
    print("# NoPE vs RoPE: KEYWORD K,V EXTRACTION COMPARISON")
    print("#" * 70)
    print("""
    Hypothesis:
    - NoPE: K,V has no position info, but needs KDA to accumulate context
    - RoPE: K,V has position info, might preserve relationships better

    Test: Extract K,V at keyword positions only, compare output similarity
    """)

    # Test cases
    test_cases = [
        {
            "context": "My name is Alice and I work at Google as a software engineer.",
            "keywords": ["Alice", "Google", "software"],
            "query": " Nice to meet you! What do you",
        },
        {
            "context": "Yesterday I went to the park and saw beautiful flowers and birds.",
            "keywords": ["yesterday", "park", "flowers", "birds"],
            "query": " That sounds lovely! Did you",
        },
        {
            "context": "The recipe needs flour, sugar, eggs and butter to make a cake.",
            "keywords": ["recipe", "flour", "sugar", "cake"],
            "query": " How long does it take to",
        },
    ]

    results = {"nope": [], "rope": []}

    for i, tc in enumerate(test_cases):
        print(f"\n\n{'#'*70}")
        print(f"# TEST CASE {i+1}")
        print(f"{'#'*70}")

        for model_type in ["nope", "rope"]:
            try:
                r = run_comparison(
                    tc["context"],
                    tc["keywords"],
                    tc["query"],
                    model_type
                )
                results[model_type].append(r)
            except Exception as e:
                print(f"\n  ERROR with {model_type}: {e}")
                results[model_type].append(None)

    # Final summary
    print(f"\n\n{'#'*70}")
    print("# FINAL SUMMARY")
    print(f"{'#'*70}")

    print(f"\n  {'Model':<10} {'Avg Cosine':<12} {'Token Match':<12} {'Output Match':<12}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*12}")

    for model_type in ["nope", "rope"]:
        valid = [r for r in results[model_type] if r is not None]
        if valid:
            avg_cos = sum(r['avg_cos'] for r in valid) / len(valid)
            avg_match = sum(r['matches'] for r in valid) / len(valid)
            out_match = sum(1 for r in valid if r['output_match']) / len(valid)
            print(f"  {model_type.upper():<10} {avg_cos:.4f}       {avg_match:.1f}/3         {out_match*100:.0f}%")

    print(f"\n  Analysis:")
    print(f"    - Higher cosine = keyword K,V better preserves full context meaning")
    print(f"    - NoPE should allow position-independent K,V loading")
    print(f"    - RoPE has position baked in, might cause issues when positions change")


if __name__ == "__main__":
    main()
