#!/bin/bash
# Setup vLLM with Anchor Connector for Kimi-Linear

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Setting up vLLM with Anchor Connector"
echo "=========================================="

# Check if vLLM folder exists
if [ -d "$SCRIPT_DIR/vllm" ]; then
    echo ""
    echo "1. Installing vLLM from local fork..."
    pip install -e "$SCRIPT_DIR/vllm"
else
    echo ""
    echo "1. vLLM folder not found. Cloning..."
    cd "$SCRIPT_DIR"
    git clone https://github.com/vllm-project/vllm.git
    cd vllm
    git checkout -b feature/anchor-connector

    # Copy our anchor connector
    echo ""
    echo "2. Installing anchor connector..."
    cp "$SCRIPT_DIR/src/anchor_connector.py" \
       "$SCRIPT_DIR/vllm/vllm/distributed/kv_transfer/kv_connector/v1/"

    # Install
    pip install -e "$SCRIPT_DIR/vllm"
fi

echo ""
echo "3. Verifying installation..."
python -c "
from vllm.distributed.kv_transfer.kv_connector.v1.anchor_connector import AnchorConnector
print('   ✓ AnchorConnector imported successfully')
"

echo ""
echo "=========================================="
echo "Setup complete!"
echo ""
echo "Usage with Kimi-Linear:"
echo "  python -c \"from vllm import LLM; llm = LLM('moonshotai/Kimi-Linear-48B-A3B-Instruct', ...)\""
echo ""
echo "With anchor connector:"
echo "  vllm serve model --kv-connector AnchorConnector --kv-connector-config '{\"storage_path\": \"/tmp/anchors\"}'"
echo "=========================================="
