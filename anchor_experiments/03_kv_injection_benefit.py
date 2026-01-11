#!/usr/bin/env python3
"""
Experiment 03: K,V Injection Benefit Test
=========================================

QUESTION: Does K,V injection provide ANY benefit over keyword text alone?

WHAT IT COMPARES:
1. FULL CONTEXT - Original text, fresh K,V compute (baseline)
2. KEYWORD TEXT ONLY - Just keywords, fresh K,V compute (no injection)
3. KEYWORD TEXT + K,V - Keywords with K,V injected from original positions

KEY METRIC:
- If #3 > #2 → K,V injection helps
- If #3 ≈ #2 → K,V injection provides no benefit

MODEL: McGill-NLP/codellm_1b_nope (NoPE, no KDA)

EXPECTED FINDING:
Marginal benefit (~+0.02 cosine). K,V injection and fresh computation
produce nearly identical outputs. The only benefit is SAVED COMPUTE -
no accuracy improvement.

Run: python tests/anchor_experiments/03_kv_injection_benefit.py
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


def extract_keywords_simple(text):
    """Extract keywords from text."""
    import re
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                  'could', 'should', 'may', 'might', 'must', 'to', 'of', 'in',
                  'for', 'on', 'with', 'at', 'by', 'from', 'as', 'and', 'but',
                  'or', 'if', 'then', 'so', 'i', 'you', 'he', 'she', 'it', 'we',
                  'they', 'my', 'your', 'his', 'her', 'its', 'our', 'their',
                  'this', 'that', 'these', 'those', 'am', 'not'}

    words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9]*\b', text)
    keywords = []
    seen = set()
    for word in words:
        if word.lower() not in stop_words and word.lower() not in seen and len(word) > 2:
            keywords.append(word)
            seen.add(word.lower())
    return keywords[:6]


def find_keyword_token_positions(context, keywords, tokenizer):
    """Find token positions of keywords in the tokenized context."""
    tokens = tokenizer.tokenize(context)
    positions = []

    for keyword in keywords:
        kw_lower = keyword.lower()
        for i, tok in enumerate(tokens):
            clean = tok.replace('Ġ', '').replace('▁', '').replace('Ċ', '').lower()
            if kw_lower == clean or kw_lower in clean:
                positions.append(i)
                break

    # Include last token
    if len(tokens) - 1 not in positions:
        positions.append(len(tokens) - 1)

    return sorted(set(positions)), tokens


def generate_fresh(text, query, n_tokens=3):
    """Generate with fresh computation (context + query)."""
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


def generate_with_kv_injection(keyword_text, query, injected_kv, n_tokens=3):
    """Generate with keyword text + injected K,V."""
    model, tokenizer = load_model()

    # Tokenize keyword text + query
    full_input = keyword_text + query
    input_ids = tokenizer(full_input, return_tensors="pt").input_ids.to(DEVICE)

    all_logits = []
    all_tokens = []

    with torch.no_grad():
        # First pass with injected K,V
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


def extract_kv_at_positions(context, positions):
    """Extract K,V from context at specified positions."""
    model, tokenizer = load_model()

    input_ids = tokenizer(context, return_tensors="pt").input_ids.to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids, use_cache=True, return_dict=True)

    full_kv = outputs.past_key_values

    # Extract only specified positions
    extracted = []
    for keys, values in full_kv:
        k = keys[:, :, positions, :]
        v = values[:, :, positions, :]
        extracted.append((k, v))

    return tuple(extracted), full_kv


def run_test(context, query, description):
    """Run comparison test."""
    model, tokenizer = load_model()

    print("\n" + "=" * 70)
    print(f"TEST: {description}")
    print("=" * 70)

    # Extract keywords
    keywords = extract_keywords_simple(context)
    keyword_text = " ".join(keywords)

    print(f"\nCONTEXT: \"{context}\"")
    print(f"KEYWORDS: {keywords}")
    print(f"KEYWORD TEXT: \"{keyword_text}\"")
    print(f"QUERY: \"{query}\"")

    # Find keyword positions
    positions, tokens = find_keyword_token_positions(context, keywords, tokenizer)
    print(f"\nTOKEN POSITIONS: {positions} (out of {len(tokens)} tokens)")

    # Extract K,V at keyword positions
    keyword_kv, full_kv = extract_kv_at_positions(context, positions)

    k_full, _ = full_kv[0]
    k_kw, _ = keyword_kv[0]
    print(f"FULL K,V shape: {list(k_full.shape)}")
    print(f"KEYWORD K,V shape: {list(k_kw.shape)}")

    # Generate with 3 approaches
    print(f"\n" + "-" * 70)
    print("GENERATING 3 TOKENS...")
    print("-" * 70)

    # 1. Full context (baseline)
    full_logits, full_tokens = generate_fresh(context, query, n_tokens=3)
    full_output = tokenizer.decode(full_tokens)
    print(f"\n1. FULL CONTEXT (baseline):")
    print(f"   Input: \"{context[:40]}...\" + query")
    print(f"   Output: \"{full_output}\" {full_tokens}")

    # 2. Keyword text only (no K,V injection)
    kw_logits, kw_tokens = generate_fresh(keyword_text, query, n_tokens=3)
    kw_output = tokenizer.decode(kw_tokens)
    print(f"\n2. KEYWORD TEXT ONLY (fresh compute):")
    print(f"   Input: \"{keyword_text}\" + query")
    print(f"   Output: \"{kw_output}\" {kw_tokens}")

    # 3. Keyword text + K,V injection
    kw_kv_logits, kw_kv_tokens = generate_with_kv_injection(keyword_text, query, keyword_kv, n_tokens=3)
    kw_kv_output = tokenizer.decode(kw_kv_tokens)
    print(f"\n3. KEYWORD TEXT + K,V INJECTION:")
    print(f"   Input: \"{keyword_text}\" + query")
    print(f"   K,V injected: {len(positions)} positions from original context")
    print(f"   Output: \"{kw_kv_output}\" {kw_kv_tokens}")

    # Compare
    print(f"\n" + "-" * 70)
    print("COMPARISON (cosine similarity to FULL CONTEXT)")
    print("-" * 70)

    cos_kw = []
    cos_kw_kv = []
    for i in range(3):
        c1 = F.cosine_similarity(full_logits[i].float(), kw_logits[i].float(), dim=-1).item()
        c2 = F.cosine_similarity(full_logits[i].float(), kw_kv_logits[i].float(), dim=-1).item()
        cos_kw.append(c1)
        cos_kw_kv.append(c2)

    print(f"\n{'Step':<8} {'KW Only':<12} {'KW + K,V':<12} {'Benefit':<12}")
    print(f"{'-'*8} {'-'*12} {'-'*12} {'-'*12}")
    for i in range(3):
        benefit = cos_kw_kv[i] - cos_kw[i]
        sign = "+" if benefit > 0 else ""
        print(f"Step {i+1}   {cos_kw[i]:.4f}       {cos_kw_kv[i]:.4f}       {sign}{benefit:.4f}")

    avg_kw = sum(cos_kw) / 3
    avg_kw_kv = sum(cos_kw_kv) / 3
    avg_benefit = avg_kw_kv - avg_kw

    print(f"{'-'*8} {'-'*12} {'-'*12} {'-'*12}")
    sign = "+" if avg_benefit > 0 else ""
    print(f"AVG      {avg_kw:.4f}       {avg_kw_kv:.4f}       {sign}{avg_benefit:.4f}")

    # Token matches
    match_kw = sum(1 for f, k in zip(full_tokens, kw_tokens) if f == k)
    match_kw_kv = sum(1 for f, k in zip(full_tokens, kw_kv_tokens) if f == k)

    print(f"\nToken matches with FULL:")
    print(f"  KW Only:  {match_kw}/3")
    print(f"  KW + K,V: {match_kw_kv}/3")

    # Verdict
    print(f"\n" + "-" * 70)
    if avg_benefit > 0.05:
        print(f"VERDICT: K,V injection HELPS (+{avg_benefit:.4f} cosine)")
    elif avg_benefit > 0:
        print(f"VERDICT: K,V injection has SLIGHT benefit (+{avg_benefit:.4f} cosine)")
    elif avg_benefit > -0.05:
        print(f"VERDICT: K,V injection has NO significant effect ({avg_benefit:.4f} cosine)")
    else:
        print(f"VERDICT: K,V injection HURTS ({avg_benefit:.4f} cosine)")
    print("-" * 70)

    return {
        'avg_kw': avg_kw,
        'avg_kw_kv': avg_kw_kv,
        'benefit': avg_benefit,
        'match_kw': match_kw,
        'match_kw_kv': match_kw_kv,
    }


def main():
    load_model()

    print("\n" + "#" * 70)
    print("# DOES K,V INJECTION PROVIDE ANY BENEFIT?")
    print("#" * 70)
    print("""
    Comparing:
    1. FULL CONTEXT - Original text (baseline)
    2. KEYWORD TEXT ONLY - Just keywords, fresh compute
    3. KEYWORD TEXT + K,V - Keywords with K,V from original positions

    Question: Does injecting K,V improve over just using keyword text?
    """)

    results = []

    # Test cases
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

    for context, query, desc in tests:
        r = run_test(context, query, desc)
        results.append((desc, r))

    # Final summary
    print("\n" + "#" * 70)
    print("# FINAL SUMMARY")
    print("#" * 70)

    print(f"\n{'Test':<15} {'KW Only':<10} {'KW+K,V':<10} {'Benefit':<10}")
    print(f"{'-'*15} {'-'*10} {'-'*10} {'-'*10}")

    for desc, r in results:
        sign = "+" if r['benefit'] > 0 else ""
        print(f"{desc:<15} {r['avg_kw']:.4f}     {r['avg_kw_kv']:.4f}     {sign}{r['benefit']:.4f}")

    avg_benefit = sum(r['benefit'] for _, r in results) / len(results)
    print(f"{'-'*15} {'-'*10} {'-'*10} {'-'*10}")
    sign = "+" if avg_benefit > 0 else ""
    print(f"{'OVERALL':<15} {'--':^10} {'--':^10} {sign}{avg_benefit:.4f}")

    print(f"\n" + "=" * 70)
    if avg_benefit > 0.05:
        print("CONCLUSION: K,V INJECTION PROVIDES BENEFIT")
    elif avg_benefit > 0:
        print("CONCLUSION: K,V INJECTION HAS MARGINAL BENEFIT")
    else:
        print("CONCLUSION: K,V INJECTION PROVIDES NO BENEFIT")
    print("=" * 70)


if __name__ == "__main__":
    main()
