# Anchor K,V Cache Experiments

This directory contains experiments testing K,V cache compression strategies for context preservation in LLMs.

## Model Used

**McGill-NLP/codellm_1b_nope** - A 1B parameter code model with:
- **NoPE** (No Position Encoding) - K,V has no position info baked in
- **No KDA** (No Key-Driven Attention) - Standard causal attention, not linear attention

## Experiment Progression

### 01_basic_kv_flow.py
**Question:** Can we extract, store, and inject K,V cache with a NoPE model?

**What it tests:**
- K,V extraction from text
- K,V storage (tensor shapes)
- K,V injection into new generation
- Position independence (NoPE property)

**Finding:** Yes, basic K,V flow works. NoPE allows position-independent K,V loading.

---

### 02_keyword_kv_hypothesis.py
**Question:** Does extracting K,V only at keyword positions preserve meaning?

**Hypothesis:** If we extract K,V at positions [0, 3, 7] (keywords), the compressed K,V might preserve the semantic meaning of the full context.

**What it tests:**
- Extract keywords from text
- Find token positions of keywords
- Extract K,V only at those positions
- Compare output with full K,V

**Finding:** NO. Without KDA, keyword-only K,V produces garbage output. Standard attention needs all K,V positions because each position only "knows" tokens 0..N at position N.

---

### 03_kv_injection_benefit.py
**Question:** Does K,V injection provide ANY benefit over just using keyword text?

**Compares:**
1. Full context (baseline)
2. Keyword text only (fresh K,V computation)
3. Keyword text + K,V injection (pre-computed K,V)

**Finding:** Marginal benefit (~+0.02 cosine). K,V injection and fresh computation produce nearly identical outputs. The only benefit is **saved compute** - no accuracy improvement.

---

### 04_position_strategies.py
**Question:** What position selection strategy works best for K,V compression?

**Strategies tested:**
- `keywords_only` (~30%) - Just keyword positions
- `every_2nd` (50%) - Every other token
- `every_2nd_plus_kw` (~55%) - Every other + keywords
- `first_half` (50%) - First half of tokens
- `last_half` (50%) - Last half of tokens
- `spread_50` (50%) - Evenly spread

**Finding:** `every_2nd` (50%) achieves best results (3/4 perfect matches). Keywords-only (30%) is too aggressive - only 1/4 perfect matches. Position distribution matters more than keyword selection.

---

### 05_semantic_anchor.py
**Question:** Can 5-word semantic anchors + K,V injection match full context?

**Semantic anchor format:** `<alice-lives-tokyo-software-engineer/>`

**Compares:**
1. Full context (baseline)
2. Full context + anchor at end
3. Anchor only + K,V injection (from full context)
4. Anchor only (fresh compute, no K,V)

**Key metric:** Direct comparison between recomputed vs injected K,V

**Finding:**
- Recomputed vs Injected K,V: ~0.98 cosine similarity, 83% token match
- They produce **nearly identical outputs**
- K,V injection doesn't improve accuracy - only saves computation

---

### 06_nope_vs_rope.py
**Question:** How does NoPE compare to RoPE (Rotary Position Encoding) for K,V extraction?

**Models:**
- NoPE: codellm_1b_nope (no position encoding)
- RoPE: GPT-2 (rotary position encoding)

**Finding:** Both struggle with keyword-only K,V extraction. The architecture (NoPE vs RoPE) matters less than having KDA for context accumulation.

---

### 07_kimi_linear_kda.py (GPU Required)
**Question:** Does Kimi-Linear's KDA state actually compress context?

**Model:** moonshotai/Kimi-Linear-Instruct
- 48B total params, 3B activated (MoE)
- Has KDA (linear attention) + MLA (standard attention)
- 1M context support, 75% KV reduction

**What it tests:**
- Full context vs anchor-only generation
- KDA state extraction (if accessible via vLLM API)
- Long context compression viability

**Requirements:** 8× L4 GPUs or 2× A100 40GB

---

### kimi_linear_notebook.ipynb
Jupyter notebook version for easy GCP execution.

---

## Key Conclusions

### What We Learned

1. **Without KDA, K,V compression doesn't preserve meaning**
   - Standard attention needs all K,V positions
   - Each position N only knows tokens 0..N
   - No mechanism to "summarize" context into fewer positions

2. **K,V injection vs recomputation: Same accuracy, different compute**
   - Recomputed K,V ≈ Injected K,V (~0.98 cosine)
   - The only benefit is skipping forward pass
   - No accuracy improvement from injection

3. **Position selection matters for partial K,V**
   - 50% (every_2nd) >> 30% (keywords_only)
   - Distribution matters more than semantic selection

4. **For true compression, need KDA (linear attention)**
   - Models like Kimi-Linear accumulate context into recurrent state
   - State S_n contains compressed representation of all context
   - Can restore state without reprocessing all tokens

### What This Means for Anchor System

For NoPE without KDA:
- Anchor text captures semantic meaning through text representation
- K,V cache is redundant - no accuracy benefit
- Only use case: skip recomputation for efficiency

For models with KDA (future):
- KDA state S_n IS the compressed context
- Can save and restore state directly
- True O(1) context restoration vs O(N) recomputation

## Running the Tests

### Local (CPU) - Experiments 01-06
```bash
# Run individual tests
python tests/anchor_experiments/01_basic_kv_flow.py
python tests/anchor_experiments/02_keyword_kv_hypothesis.py
python tests/anchor_experiments/03_kv_injection_benefit.py
python tests/anchor_experiments/04_position_strategies.py
python tests/anchor_experiments/05_semantic_anchor.py
python tests/anchor_experiments/06_nope_vs_rope.py

# Run all (takes ~5-10 min on CPU)
for f in tests/anchor_experiments/0[1-6]*.py; do python $f; done
```

### GCP (8× L4 GPU) - Experiment 07
```bash
# Option 1: Python script
python tests/anchor_experiments/07_kimi_linear_kda.py

# Option 2: Jupyter notebook
jupyter notebook tests/anchor_experiments/kimi_linear_notebook.ipynb
```

## Model Requirements

### For experiments 01-06 (local):
```bash
pip install torch transformers
```
The 1B NoPE model (~6.8GB) will be downloaded on first run.

### For experiment 07 (GCP):
```bash
pip install vllm torch transformers
```
Kimi-Linear requires 8× L4 or 2× A100 40GB GPUs.
