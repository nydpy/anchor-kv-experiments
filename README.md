# Anchor K,V Cache Experiments

Semantic anchor-based KV cache compression for Kimi-Linear models using vLLM.

## Overview

This project implements an `AnchorConnector` for vLLM that enables:
- Save/restore KDA recurrent state (linear attention compressed context)
- Save/restore MLA K,V cache (standard attention)
- Use semantic anchors as cache keys for efficient lookup

## GCP Setup (4× A100 40GB)

### 1. Create GCP Instance

- **Machine**: `a2-highgpu-4g` (4× A100 40GB)
- **Disk**: 300GB+ (Kimi-Linear model is ~100GB)
- **Image**: Deep Learning VM with CUDA

### 2. Install NVIDIA Drivers

```bash
# Install driver (if not already installed)
sudo /opt/deeplearning/install-driver.sh
# or
sudo apt-get install -y nvidia-driver-535-server && sudo reboot

# Verify
nvidia-smi
```

### 3. Create Virtual Environment

```bash
python -m venv ~/vllm-env
source ~/vllm-env/bin/activate
pip install vllm
```

### 4. Clone Repositories

```bash
# Clone vLLM fork with AnchorConnector
git clone https://github.com/nydpy/vllm.git ~/vllm-source
cd ~/vllm-source
git remote add nydpy https://github.com/nydpy/vllm.git
git fetch nydpy
git checkout nydpy/feature/anchor-connector -- vllm/distributed/kv_transfer/kv_connector/v1/anchor_connector.py
git checkout nydpy/feature/anchor-connector -- vllm/distributed/kv_transfer/kv_connector/factory.py

# Clone experiments repo
git clone https://github.com/nydpy/anchor-kv-experiments.git ~/anchor-kv-experiments
```

### 5. Copy AnchorConnector to Installed vLLM

```bash
cd ~

# Copy anchor_connector.py
cp ~/vllm-source/vllm/distributed/kv_transfer/kv_connector/v1/anchor_connector.py \
   ~/vllm-env/lib/python3.10/site-packages/vllm/distributed/kv_transfer/kv_connector/v1/

# Copy factory.py (with AnchorConnector registration)
cp ~/vllm-source/vllm/distributed/kv_transfer/kv_connector/factory.py \
   ~/vllm-env/lib/python3.10/site-packages/vllm/distributed/kv_transfer/kv_connector/

# Verify registration
python -c "from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory; print('AnchorConnector' in KVConnectorFactory._registry)"
# Should print: True
```

### 6. Run vLLM Server with AnchorConnector

```bash
vllm serve moonshotai/Kimi-Linear-48B-A3B-Instruct \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --trust-remote-code \
  --kv-transfer-config '{"kv_connector": "AnchorConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"storage_path": "/tmp/anchors"}}'
```

### 7. Test the Server

In another terminal:

```bash
source ~/vllm-env/bin/activate
cd ~/anchor-kv-experiments
python tests/test_server_anchor.py
```

## Tests

| Test | Description |
|------|-------------|
| `test_kimi_linear.py` | Basic Kimi-Linear generation test |
| `test_anchor_save_restore.py` | Context-dependent generation test |
| `test_server_anchor.py` | Test via vLLM server API |
| `test_programmatic_connector.py` | Check connector configuration |

## Key Findings

### Without KDA (standard attention):
- K,V cache compression doesn't preserve meaning
- Keyword-only K,V produces garbage output
- Injection vs recomputation: same accuracy, only saves compute

### With KDA (Kimi-Linear):
- KDA state accumulates context into recurrent state
- State can theoretically be saved/restored
- 75% K,V cache reduction built-in

## Architecture

### AnchorConnector

The `AnchorConnector` handles both:
- **KDA layers**: Save/restore recurrent state (conv_state_q, conv_state_k, conv_state_v, recurrent_state)
- **MLA layers**: Save/restore K,V cache

### Semantic Anchors

Instead of full token hash, use semantic anchors like:
- `<alice-software-tokyo/>` - 5-word context summary
- Enables context sharing across similar prompts

## Repositories

- **vLLM Fork**: https://github.com/nydpy/vllm (branch: `feature/anchor-connector`)
- **Experiments**: https://github.com/nydpy/anchor-kv-experiments

## Troubleshooting

### "AnchorConnector NOT registered in factory"
Copy the updated `factory.py` from the vLLM fork (see step 5).

### "No module named 'vllm._C'"
Run Python from outside the `vllm-source` folder (run from `~`).

### Out of disk space
Kimi-Linear needs ~100GB. Resize disk to 300GB+ in GCP Console.

### No GPU detected
Install NVIDIA drivers: `sudo apt-get install nvidia-driver-535-server && sudo reboot`

## License

Apache 2.0 (same as vLLM)
