# Anchor K,V Cache Experiments

Testing K,V cache compression strategies for LLMs, specifically for Kimi-Linear's KDA (Key-Driven Attention).

## Project Structure

```
anchor-kv-experiments/
├── src/
│   └── anchor_connector.py      # vLLM K,V connector implementation
├── tests/
│   ├── 01-06_*.py               # Proof-of-concept tests (transformers)
│   └── test_vllm_integration.py # vLLM integration test
├── notebooks/
│   └── kimi_linear.ipynb        # GCP notebook
├── docs/
│   └── findings.md              # Experiment findings
├── requirements.txt
├── setup_vllm.sh                # vLLM installation script
└── README.md
```

## Quick Start

### 1. Install vLLM with Anchor Connector

```bash
# Clone and setup
git clone https://github.com/YOUR_USERNAME/anchor-kv-experiments.git
cd anchor-kv-experiments
./setup_vllm.sh
```

### 2. Run Proof-of-Concept Tests (CPU)

```bash
# These use transformers directly, no GPU needed
pip install torch transformers
python tests/01_basic_kv_flow.py
python tests/05_semantic_anchor.py
```

### 3. Run vLLM Integration Test (8× L4 GPU)

```bash
# Requires GPU
python tests/test_vllm_integration.py
```

## Key Findings

### Without KDA (standard attention):
- K,V cache compression doesn't preserve meaning
- Keyword-only K,V produces garbage output
- Injection vs recomputation: same accuracy, only saves compute

### With KDA (Kimi-Linear):
- KDA state accumulates context into recurrent state
- State can theoretically be saved/restored
- 75% K,V cache reduction built-in

## Hardware Requirements

| Test | Hardware |
|------|----------|
| 01-06 (proof-of-concept) | CPU only |
| 07+ (Kimi-Linear) | 8× L4 or 2× A100 40GB |

## License

Apache 2.0 (same as vLLM)
