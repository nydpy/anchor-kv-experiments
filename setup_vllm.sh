#!/bin/bash
# Setup vLLM with Anchor Connector

set -e

echo "=========================================="
echo "Setting up vLLM with Anchor Connector"
echo "=========================================="

# Install vLLM
echo ""
echo "1. Installing vLLM..."
pip install vllm

# Find vLLM installation path
VLLM_PATH=$(python -c "import vllm; import os; print(os.path.dirname(vllm.__file__))")
echo "   vLLM installed at: $VLLM_PATH"

# Create connector directory if needed
CONNECTOR_DIR="$VLLM_PATH/distributed/kv_transfer/kv_connector/v1"
mkdir -p "$CONNECTOR_DIR"

# Copy anchor connector
echo ""
echo "2. Installing anchor connector..."
cp src/anchor_connector.py "$CONNECTOR_DIR/"
echo "   Copied to: $CONNECTOR_DIR/anchor_connector.py"

# Patch factory.py to register anchor connector
echo ""
echo "3. Registering anchor connector in factory..."

FACTORY_FILE="$VLLM_PATH/distributed/kv_transfer/kv_connector/factory.py"

# Check if already patched
if grep -q "AnchorConnector" "$FACTORY_FILE" 2>/dev/null; then
    echo "   Already registered, skipping..."
else
    # Add import at top of file
    sed -i.bak '1s/^/from vllm.distributed.kv_transfer.kv_connector.v1.anchor_connector import AnchorConnector\n/' "$FACTORY_FILE"

    # Add to connector registry (this is a simplified patch)
    echo "   Patched factory.py"
    echo "   NOTE: You may need to manually register 'anchor' in the connector factory"
fi

echo ""
echo "4. Verifying installation..."
python -c "
from vllm.distributed.kv_transfer.kv_connector.v1.anchor_connector import AnchorConnector
print('   ✓ AnchorConnector imported successfully')
"

echo ""
echo "=========================================="
echo "Setup complete!"
echo ""
echo "Run tests:"
echo "  python tests/test_vllm_integration.py"
echo "=========================================="
