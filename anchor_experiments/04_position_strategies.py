#!/usr/bin/env python3
"""
Experiment 04: Position Selection Strategies
============================================

QUESTION: What position selection strategy works best for K,V compression?

STRATEGIES TESTED:
- keywords_only (~30%): Just keyword positions + last token
- every_2nd (50%): Every other token
- every_2nd_plus_kw (~55%): Every other + keywords
- first_half (50%): First 50% of tokens + last
- last_half (50%): Last 50% of tokens
- spread_50 (50%): Evenly distributed

WHAT IT MEASURES:
- Cosine similarity to full context baseline
- Token match rate (how many generated tokens match)
- Compression ratio achieved

MODEL: McGill-NLP/codellm_1b_nope (NoPE, no KDA)

EXPECTED FINDING:
every_2nd (50%) achieves best results (3/4 perfect matches).
keywords_only (30%) is too aggressive - only 1/4 perfect matches.
Position DISTRIBUTION matters more than keyword SELECTION.

Run: python tests/anchor_experiments/04_position_strategies.py
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


def extract_keywords(text):
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
    return keywords


def find_keyword_positions(tokens, keywords):
    """Find token positions of keywords."""
    positions = []
    for keyword in keywords:
        kw_lower = keyword.lower()
        for i, tok in enumerate(tokens):
            clean = tok.replace('Ġ', '').replace('▁', '').replace('Ċ', '').lower()
            if kw_lower == clean or kw_lower in clean:
                positions.append(i)
                break
    return positions


def get_position_strategies(num_tokens, tokens, keywords):
    """Get different position selection strategies."""
    strategies = {}

    # 1. Keywords only (+ last)
    kw_pos = find_keyword_positions(tokens, keywords)
    if num_tokens - 1 not in kw_pos:
        kw_pos.append(num_tokens - 1)
    strategies['keywords_only'] = sorted(set(kw_pos))

    # 2. Every 2nd token (50%)
    every_2nd = list(range(0, num_tokens, 2))
    if num_tokens - 1 not in every_2nd:
        every_2nd.append(num_tokens - 1)
    strategies['every_2nd'] = sorted(set(every_2nd))

    # 3. Every 2nd + keywords
    combined = set(every_2nd + kw_pos)
    strategies['every_2nd_plus_kw'] = sorted(combined)

    # 4. First 50% + last
    half = num_tokens // 2
    first_half = list(range(half))
    first_half.append(num_tokens - 1)
    strategies['first_half'] = sorted(set(first_half))

    # 5. Last 50%
    last_half = list(range(half, num_tokens))
    strategies['last_half'] = sorted(set(last_half))

    # 6. Evenly spread 50%
    step = max(1, num_tokens // (num_tokens // 2))
    spread = list(range(0, num_tokens, step))[:num_tokens // 2]
    if num_tokens - 1 not in spread:
        spread.append(num_tokens - 1)
    strategies['spread_50'] = sorted(set(spread))

    return strategies


def extract_full_kv(context):
    """Extract full K,V from context."""
    model, tokenizer = load_model()
    input_ids = tokenizer(context, return_tensors="pt").input_ids.to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids, use_cache=True, return_dict=True)

    return outputs.past_key_values, input_ids.shape[1]


def extract_kv_at_positions(full_kv, positions):
    """Extract K,V at specified positions."""
    extracted = []
    for keys, values in full_kv:
        k = keys[:, :, positions, :]
        v = values[:, :, positions, :]
        extracted.append((k, v))
    return tuple(extracted)


def generate_with_kv(context_text, query, injected_kv, n_tokens=3):
    """Generate with context text + injected K,V."""
    model, tokenizer = load_model()

    full_input = context_text + query
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


def build_compressed_text(tokens, positions, tokenizer):
    """Build compressed text from selected token positions."""
    selected_tokens = [tokens[i] for i in positions if i < len(tokens)]
    # Join tokens, handling special prefixes
    text = tokenizer.convert_tokens_to_string(selected_tokens)
    return text.strip()


def run_test(context, query, description):
    """Run test with different position strategies."""
    model, tokenizer = load_model()

    print("\n" + "=" * 70)
    print(f"TEST: {description}")
    print("=" * 70)

    print(f"\nCONTEXT: \"{context}\"")
    print(f"QUERY: \"{query}\"")

    # Tokenize
    tokens = tokenizer.tokenize(context)
    num_tokens = len(tokens)
    keywords = extract_keywords(context)

    print(f"\nTOKENS: {num_tokens}")
    print(f"KEYWORDS: {keywords}")

    # Get strategies
    strategies = get_position_strategies(num_tokens, tokens, keywords)

    # Extract full K,V
    full_kv, seq_len = extract_full_kv(context)

    # Generate baseline (full context)
    full_logits, full_tokens = generate_fresh(context, query, n_tokens=3)
    full_output = tokenizer.decode(full_tokens)

    print(f"\n" + "-" * 70)
    print(f"BASELINE (FULL CONTEXT): \"{full_output}\" {full_tokens}")
    print("-" * 70)

    results = {}

    print(f"\n{'Strategy':<25} {'Positions':<15} {'%':<8} {'Output':<25} {'Match':<8} {'Cosine':<8}")
    print(f"{'-'*25} {'-'*15} {'-'*8} {'-'*25} {'-'*8} {'-'*8}")

    for name, positions in strategies.items():
        # Extract K,V at positions
        partial_kv = extract_kv_at_positions(full_kv, positions)

        # Build compressed text
        compressed_text = build_compressed_text(tokens, positions, tokenizer)

        # Generate with compressed text + K,V
        logits, gen_tokens = generate_with_kv(compressed_text, query, partial_kv, n_tokens=3)
        output = tokenizer.decode(gen_tokens)

        # Calculate metrics
        pct = len(positions) / num_tokens * 100
        matches = sum(1 for f, g in zip(full_tokens, gen_tokens) if f == g)

        cos_sims = [F.cosine_similarity(full_logits[i].float(), logits[i].float(), dim=-1).item()
                    for i in range(3)]
        avg_cos = sum(cos_sims) / 3

        results[name] = {
            'positions': len(positions),
            'pct': pct,
            'output': output,
            'matches': matches,
            'cos': avg_cos,
            'tokens': gen_tokens,
        }

        match_str = f"{matches}/3" + (" ✓" if matches == 3 else "")
        print(f"{name:<25} {len(positions):<15} {pct:<8.1f} {output:<25} {match_str:<8} {avg_cos:.4f}")

    return results, full_tokens, full_output


def main():
    load_model()

    print("\n" + "#" * 70)
    print("# K,V INJECTION WITH 50% TOKEN POSITIONS")
    print("#" * 70)
    print("""
    Testing different position selection strategies:
    - keywords_only: Just keywords + last token (~20%)
    - every_2nd: Every 2nd token (50%)
    - every_2nd_plus_kw: Every 2nd + keywords (~50-60%)
    - first_half: First 50% of tokens + last
    - last_half: Last 50% of tokens
    - spread_50: Evenly spread 50%

    Each uses: compressed text (selected tokens) + K,V from those positions
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
        results, full_tokens, full_output = run_test(context, query, desc)
        all_results.append((desc, results, full_tokens))

    # Summary
    print("\n" + "#" * 70)
    print("# SUMMARY: AVERAGE ACROSS ALL TESTS")
    print("#" * 70)

    strategy_names = list(all_results[0][1].keys())

    print(f"\n{'Strategy':<25} {'Avg %':<10} {'Avg Match':<12} {'Avg Cosine':<12} {'Perfect':<10}")
    print(f"{'-'*25} {'-'*10} {'-'*12} {'-'*12} {'-'*10}")

    for strat in strategy_names:
        avg_pct = sum(r[1][strat]['pct'] for r in all_results) / len(all_results)
        avg_match = sum(r[1][strat]['matches'] for r in all_results) / len(all_results)
        avg_cos = sum(r[1][strat]['cos'] for r in all_results) / len(all_results)
        perfect = sum(1 for r in all_results if r[1][strat]['matches'] == 3)

        print(f"{strat:<25} {avg_pct:<10.1f} {avg_match:<12.2f} {avg_cos:<12.4f} {perfect}/4")

    # Find best
    print("\n" + "=" * 70)
    best_strat = max(strategy_names, key=lambda s: sum(r[1][s]['matches'] for r in all_results))
    best_matches = sum(r[1][best_strat]['matches'] for r in all_results) / len(all_results)
    best_pct = sum(r[1][best_strat]['pct'] for r in all_results) / len(all_results)

    print(f"BEST STRATEGY: {best_strat}")
    print(f"  Average token match: {best_matches:.2f}/3")
    print(f"  Average positions: {best_pct:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
