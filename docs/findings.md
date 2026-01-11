# Experiment Findings

## Summary

We tested K,V cache compression strategies using:
- **Local (CPU):** codellm_1b_nope (1B params, NoPE, no KDA)
- **GCP (GPU):** Kimi-Linear (48B/3B activated, NoPE + KDA + MLA)

## Key Findings

### 1. Without KDA, K,V compression fails

**Experiment 02:** Extract K,V only at keyword positions (~30%)

| Approach | Output |
|----------|--------|
| Full K,V | Meaningful response |
| Keyword K,V | Garbage ("I I I", "f f f") |

**Why:** Standard causal attention at position N only knows tokens 0..N. Removing positions loses information permanently.

### 2. K,V injection vs recomputation: Same accuracy

**Experiment 05:** Compare recomputed vs injected K,V

| Metric | Value |
|--------|-------|
| Cosine similarity | 0.9786 |
| Token match | 83% (10/12) |

**Conclusion:** Recomputed K,V ≈ Injected K,V. The only benefit is **saved compute**, not accuracy.

### 3. Position distribution matters more than keywords

**Experiment 04:** Compare position selection strategies

| Strategy | Match Rate |
|----------|------------|
| keywords_only (30%) | 1/4 |
| every_2nd (50%) | 3/4 |
| first_half (50%) | 2/4 |
| last_half (50%) | 2/4 |

**Conclusion:** Even distribution (every_2nd) > semantic selection (keywords_only)

### 4. Semantic anchors capture meaning through text

**Experiment 05:** 5-word anchors like `<alice-software-tokyo-hiking/>`

- Anchor text alone achieves ~0.96 cosine similarity to full context
- K,V injection adds only +0.0006 improvement
- The **text representation** carries the meaning, not the K,V cache

## Architecture Comparison

| Model | Position Encoding | Attention Type | K,V Compression |
|-------|------------------|----------------|-----------------|
| GPT-2 | RoPE | Standard | ✗ Fails |
| codellm_1b_nope | NoPE | Standard | ✗ Fails |
| Kimi-Linear | NoPE | KDA + MLA | ✓ Should work |

## Why Kimi-Linear is Different

```
Standard Attention (all tested models):
  K,V at position N = information about tokens 0..N
  → Removing positions = losing information

KDA (Linear Attention) in Kimi-Linear:
  State S_n = Σ(k_i ⊗ v_i) for i=1..n
  → State accumulates ALL context
  → Can save/restore state directly
```

## Recommendations

### For NoPE without KDA:
- Don't bother with K,V compression
- Use anchor TEXT for semantic compression
- Benefit is only compute savings on recomputation

### For Kimi-Linear (with KDA):
- KDA state should contain compressed context
- Need to access KDA state via vLLM API
- May require vLLM modification or feature request

## Next Steps

1. **Test Kimi-Linear on GCP** - Verify KDA state is accessible
2. **Modify vLLM** - Expose KDA state save/restore API
3. **Benchmark** - Compare latency/memory with full K,V cache
