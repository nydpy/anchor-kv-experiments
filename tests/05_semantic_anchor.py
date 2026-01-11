#!/usr/bin/env python3
"""
Experiment 05: Semantic Anchor with K,V Injection
=================================================

QUESTION: Can 5-word semantic anchors + K,V injection match full context?

SEMANTIC ANCHOR FORMAT:
<alice-lives-tokyo-software-engineer/>
- 5 keywords joined with hyphens
- Wrapped in XML-like tag format

WHAT IT COMPARES:
1. FULL CONTEXT (baseline)
2. FULL CONTEXT + ANCHOR at end (reference)
3. ANCHOR ONLY + K,V injection (from full context)
4. ANCHOR ONLY (fresh compute, no K,V)

KEY COMPARISON:
Direct cosine similarity between #3 (injected) and #4 (recomputed)
to see if injection provides any accuracy benefit.

MODEL: McGill-NLP/codellm_1b_nope (NoPE, no KDA)

EXPECTED FINDING:
- Recomputed vs Injected: ~0.98 cosine similarity, 83% token match
- They produce NEARLY IDENTICAL outputs
- K,V injection doesn't improve accuracy - only saves computation

Run: python tests/anchor_experiments/05_semantic_anchor.py
"""

import torch
import torch.nn.functional as F

MODEL = None
TOKENIZER = None
DEVICE = "cpu"


def load_model():
    global MODEL, TOKENIZER, DEVICE

    if MODEL is not None:
        return MODEL, TOKENIZER

    print("\nLoading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "McGill-NLP/codellm_1b_nope"

    TOKENIZER = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    MODEL = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    MODEL = MODEL.to(DEVICE)
    MODEL.eval()

    print(f"Loaded on {DEVICE}\n")
    return MODEL, TOKENIZER


def create_semantic_anchor(context):
    """Create a 5-word semantic summary anchor from context."""
    import re

    # Extract key words (nouns, verbs, names)
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                  'could', 'should', 'may', 'might', 'must', 'to', 'of', 'in',
                  'for', 'on', 'with', 'at', 'by', 'from', 'as', 'and', 'but',
                  'or', 'if', 'then', 'so', 'i', 'you', 'he', 'she', 'it', 'we',
                  'they', 'my', 'your', 'his', 'her', 'its', 'our', 'their',
                  'this', 'that', 'these', 'those', 'am', 'not', 'hi', 'hello'}

    words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9]*\b', context.lower())
    keywords = []
    seen = set()
    for word in words:
        if word not in stop_words and word not in seen and len(word) > 2:
            keywords.append(word)
            seen.add(word)
            if len(keywords) >= 5:
                break

    # Create anchor tag format
    anchor_name = "-".join(keywords)
    anchor_tag = f"<{anchor_name}/>"

    return anchor_tag, keywords


def generate_fresh(text, query, n_tokens=3):
    """Generate with fresh computation."""
    model, tokenizer = load_model()

    full_input = text + query
    input_ids = tokenizer(full_input, return_tensors="pt").input_ids.to(DEVICE)

    all_logits = []
    all_tokens = []

    with torch.no_grad():
        outputs = model(input_ids, use_cache=True, return_dict=True)
        current_kv = outputs.past_key_values

        logits = outputs.logits[:, -1, :]
        all_logits.append(logits.cpu())
        next_token = torch.argmax(logits, dim=-1)
        all_tokens.append(next_token.item())

        current_input = next_token.unsqueeze(0)
        for _ in range(n_tokens - 1):
            outputs = model(current_input, past_key_values=current_kv, use_cache=True, return_dict=True)
            current_kv = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            all_logits.append(logits.cpu())
            next_token = torch.argmax(logits, dim=-1)
            all_tokens.append(next_token.item())
            current_input = next_token.unsqueeze(0)

    return all_logits, all_tokens


def generate_with_kv(text, query, injected_kv, n_tokens=3):
    """Generate with text + injected K,V."""
    model, tokenizer = load_model()

    full_input = text + query
    input_ids = tokenizer(full_input, return_tensors="pt").input_ids.to(DEVICE)

    all_logits = []
    all_tokens = []

    with torch.no_grad():
        outputs = model(input_ids, past_key_values=injected_kv, use_cache=True, return_dict=True)
        current_kv = outputs.past_key_values

        logits = outputs.logits[:, -1, :]
        all_logits.append(logits.cpu())
        next_token = torch.argmax(logits, dim=-1)
        all_tokens.append(next_token.item())

        current_input = next_token.unsqueeze(0)
        for _ in range(n_tokens - 1):
            outputs = model(current_input, past_key_values=current_kv, use_cache=True, return_dict=True)
            current_kv = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            all_logits.append(logits.cpu())
            next_token = torch.argmax(logits, dim=-1)
            all_tokens.append(next_token.item())
            current_input = next_token.unsqueeze(0)

    return all_logits, all_tokens


def extract_kv(text):
    """Extract K,V from text."""
    model, tokenizer = load_model()
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids, use_cache=True, return_dict=True)

    return outputs.past_key_values, input_ids.shape[1]


def run_test(context, query, description):
    """Run comparison test."""
    model, tokenizer = load_model()

    print("\n" + "=" * 70)
    print(f"TEST: {description}")
    print("=" * 70)

    # Create semantic anchor
    anchor_tag, keywords = create_semantic_anchor(context)

    print(f"\nCONTEXT: \"{context}\"")
    print(f"SEMANTIC ANCHOR: {anchor_tag}")
    print(f"KEYWORDS: {keywords}")
    print(f"QUERY: \"{query}\"")

    # Count tokens
    context_tokens = len(tokenizer.tokenize(context))
    anchor_tokens = len(tokenizer.tokenize(anchor_tag))

    print(f"\nTOKEN COUNT:")
    print(f"  Full context: {context_tokens} tokens")
    print(f"  Anchor only:  {anchor_tokens} tokens")
    print(f"  Compression:  {context_tokens / anchor_tokens:.1f}x")

    # Extract K,V from full context
    full_kv, full_seq_len = extract_kv(context)

    # Also extract K,V from full context + anchor (for reference)
    full_with_anchor = context + " " + anchor_tag
    full_anchor_kv, _ = extract_kv(full_with_anchor)

    print(f"\n" + "-" * 70)
    print("GENERATING 3 TOKENS...")
    print("-" * 70)

    results = {}

    # 1. FULL CONTEXT (baseline)
    full_logits, full_tokens = generate_fresh(context, query, n_tokens=3)
    full_output = tokenizer.decode(full_tokens)
    results['full'] = {'logits': full_logits, 'tokens': full_tokens, 'output': full_output}
    print(f"\n1. FULL CONTEXT (baseline):")
    print(f"   Input: \"{context[:40]}...\"")
    print(f"   Output: \"{full_output}\" {full_tokens}")

    # 2. FULL CONTEXT + ANCHOR at end
    full_anchor_logits, full_anchor_tokens = generate_fresh(full_with_anchor, query, n_tokens=3)
    full_anchor_output = tokenizer.decode(full_anchor_tokens)
    results['full_anchor'] = {'logits': full_anchor_logits, 'tokens': full_anchor_tokens, 'output': full_anchor_output}
    print(f"\n2. FULL CONTEXT + ANCHOR:")
    print(f"   Input: \"{context[:30]}... {anchor_tag}\"")
    print(f"   Output: \"{full_anchor_output}\" {full_anchor_tokens}")

    # 3. ANCHOR ONLY + K,V injection (from full context)
    anchor_kv_logits, anchor_kv_tokens = generate_with_kv(anchor_tag, query, full_kv, n_tokens=3)
    anchor_kv_output = tokenizer.decode(anchor_kv_tokens)
    results['anchor_kv'] = {'logits': anchor_kv_logits, 'tokens': anchor_kv_tokens, 'output': anchor_kv_output}
    print(f"\n3. ANCHOR ONLY + K,V INJECTION:")
    print(f"   Input: \"{anchor_tag}\"")
    print(f"   K,V injected: {full_seq_len} positions from full context")
    print(f"   Output: \"{anchor_kv_output}\" {anchor_kv_tokens}")

    # 4. ANCHOR ONLY (no K,V, fresh compute)
    anchor_only_logits, anchor_only_tokens = generate_fresh(anchor_tag, query, n_tokens=3)
    anchor_only_output = tokenizer.decode(anchor_only_tokens)
    results['anchor_only'] = {'logits': anchor_only_logits, 'tokens': anchor_only_tokens, 'output': anchor_only_output}
    print(f"\n4. ANCHOR ONLY (no K,V):")
    print(f"   Input: \"{anchor_tag}\"")
    print(f"   Output: \"{anchor_only_output}\" {anchor_only_tokens}")

    # Compare
    print(f"\n" + "-" * 70)
    print("COMPARISON (vs FULL CONTEXT baseline)")
    print("-" * 70)

    print(f"\n{'Approach':<30} {'Cosine':<10} {'Match':<10} {'Output':<20}")
    print(f"{'-'*30} {'-'*10} {'-'*10} {'-'*20}")

    for name, data in results.items():
        if name == 'full':
            continue

        cos_sims = [F.cosine_similarity(full_logits[i].float(), data['logits'][i].float(), dim=-1).item()
                    for i in range(3)]
        avg_cos = sum(cos_sims) / 3

        matches = sum(1 for f, g in zip(full_tokens, data['tokens']) if f == g)
        match_str = f"{matches}/3" + (" ✓" if matches == 3 else "")

        print(f"{name:<30} {avg_cos:<10.4f} {match_str:<10} {data['output']:<20}")

    # K,V injection benefit
    cos_anchor_only = sum(F.cosine_similarity(full_logits[i].float(), results['anchor_only']['logits'][i].float(), dim=-1).item() for i in range(3)) / 3
    cos_anchor_kv = sum(F.cosine_similarity(full_logits[i].float(), results['anchor_kv']['logits'][i].float(), dim=-1).item() for i in range(3)) / 3
    benefit = cos_anchor_kv - cos_anchor_only

    match_anchor_only = sum(1 for f, g in zip(full_tokens, results['anchor_only']['tokens']) if f == g)
    match_anchor_kv = sum(1 for f, g in zip(full_tokens, results['anchor_kv']['tokens']) if f == g)

    # Direct comparison: recomputed vs injected K,V
    cos_recompute_vs_inject = sum(F.cosine_similarity(
        results['anchor_only']['logits'][i].float(),
        results['anchor_kv']['logits'][i].float(),
        dim=-1).item() for i in range(3)) / 3

    match_recompute_vs_inject = sum(1 for a, b in zip(results['anchor_only']['tokens'], results['anchor_kv']['tokens']) if a == b)

    print(f"\n  K,V INJECTION BENEFIT (vs full context):")
    print(f"    Cosine: {cos_anchor_only:.4f} → {cos_anchor_kv:.4f} ({'+' if benefit > 0 else ''}{benefit:.4f})")
    print(f"    Matches: {match_anchor_only}/3 → {match_anchor_kv}/3")

    print(f"\n  RECOMPUTED vs INJECTED K,V (direct comparison):")
    print(f"    Cosine similarity: {cos_recompute_vs_inject:.4f}")
    print(f"    Token matches: {match_recompute_vs_inject}/3")

    return {
        'compression': context_tokens / anchor_tokens,
        'cos_anchor_only': cos_anchor_only,
        'cos_anchor_kv': cos_anchor_kv,
        'benefit': benefit,
        'match_anchor_only': match_anchor_only,
        'match_anchor_kv': match_anchor_kv,
        'cos_recompute_vs_inject': cos_recompute_vs_inject,
        'match_recompute_vs_inject': match_recompute_vs_inject,
        'full_output': full_output,
        'anchor_kv_output': anchor_kv_output,
        'anchor_only_output': anchor_only_output,
    }


def main():
    load_model()

    print("\n" + "#" * 70)
    print("# SEMANTIC ANCHOR TEST")
    print("#" * 70)
    print("""
    Semantic Anchor = 5-word summary like "<alice-lives-tokyo-software/>"

    Comparing:
    1. FULL CONTEXT - baseline
    2. FULL CONTEXT + ANCHOR - reference (anchor at end)
    3. ANCHOR + K,V - compressed text with K,V from full context
    4. ANCHOR ONLY - compressed text, no K,V (fresh compute)

    Question: Does K,V injection make the anchor work like full context?
    """)

    tests = [
        ("Hi my name is Alice and I live in Tokyo as a software engineer.",
         " Nice to meet you! What",
         "Introduction"),

        ("Yesterday I went hiking in the mountains and saw beautiful waterfalls.",
         " That sounds amazing! Did you",
         "Activity"),

        ("My friend Bob is having a birthday party next Saturday with cake and games.",
         " Are you going to",
         "Event"),

        ("I made delicious pasta with tomato sauce and fresh basil from my garden.",
         " Yum! Was the",
         "Food"),
    ]

    all_results = []

    for context, query, desc in tests:
        r = run_test(context, query, desc)
        all_results.append((desc, r))

    # Summary
    print("\n" + "#" * 70)
    print("# FINAL SUMMARY")
    print("#" * 70)

    print(f"\n{'Test':<15} {'Compress':<10} {'Anchor Only':<15} {'Anchor+K,V':<15} {'Recomp vs Inject':<18}")
    print(f"{'-'*15} {'-'*10} {'-'*15} {'-'*15} {'-'*18}")

    for desc, r in all_results:
        ao_str = f"{r['cos_anchor_only']:.3f} ({r['match_anchor_only']}/3)"
        akv_str = f"{r['cos_anchor_kv']:.3f} ({r['match_anchor_kv']}/3)"
        rvi_str = f"{r['cos_recompute_vs_inject']:.4f} ({r['match_recompute_vs_inject']}/3)"
        print(f"{desc:<15} {r['compression']:<10.1f}x {ao_str:<15} {akv_str:<15} {rvi_str:<18}")

    # Averages
    avg_compress = sum(r['compression'] for _, r in all_results) / len(all_results)
    avg_ao = sum(r['cos_anchor_only'] for _, r in all_results) / len(all_results)
    avg_akv = sum(r['cos_anchor_kv'] for _, r in all_results) / len(all_results)
    avg_benefit = sum(r['benefit'] for _, r in all_results) / len(all_results)
    avg_rvi = sum(r['cos_recompute_vs_inject'] for _, r in all_results) / len(all_results)

    avg_match_ao = sum(r['match_anchor_only'] for _, r in all_results) / len(all_results)
    avg_match_akv = sum(r['match_anchor_kv'] for _, r in all_results) / len(all_results)
    avg_match_rvi = sum(r['match_recompute_vs_inject'] for _, r in all_results) / len(all_results)

    print(f"{'-'*15} {'-'*10} {'-'*15} {'-'*15} {'-'*18}")
    print(f"{'AVERAGE':<15} {avg_compress:<10.1f}x {avg_ao:.3f} ({avg_match_ao:.1f}/3)  {avg_akv:.3f} ({avg_match_akv:.1f}/3)  {avg_rvi:.4f} ({avg_match_rvi:.1f}/3)")

    print("\n" + "=" * 70)
    print("CONCLUSIONS:")
    print(f"\n1. Recomputed vs Injected K,V similarity: {avg_rvi:.4f}")
    if avg_rvi > 0.99:
        print("   → Nearly IDENTICAL outputs - injection saves compute, same accuracy")
    elif avg_rvi > 0.95:
        print("   → Very SIMILAR outputs - minimal difference between approaches")
    else:
        print("   → DIFFERENT outputs - approaches diverge")

    print(f"\n2. K,V injection benefit (vs full context): {'+' if avg_benefit > 0 else ''}{avg_benefit:.4f}")
    if avg_benefit > 0.05:
        print("   → K,V injection SIGNIFICANTLY helps")
    elif avg_benefit > 0:
        print("   → K,V injection MARGINALLY helps")
    else:
        print("   → K,V injection provides NO benefit")

    print(f"\n3. Average compression: {avg_compress:.1f}x")
    print(f"   Average token matches: {avg_match_ao:.1f}/3 (recomputed) vs {avg_match_akv:.1f}/3 (injected)")
    print("=" * 70)


if __name__ == "__main__":
    main()
