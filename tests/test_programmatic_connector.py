#!/usr/bin/env python3
"""
Test AnchorConnector via programmatic vLLM configuration.
Bypasses CLI to configure KV connector directly.
"""

import torch
import os

def main():
    print(f"GPUs available: {torch.cuda.device_count()}")

    # Try to configure connector programmatically
    print("\n" + "="*60)
    print("Attempting programmatic connector configuration...")
    print("="*60)

    try:
        from vllm import LLM, SamplingParams
        from vllm.config import KVTransferConfig

        # Check if KVTransferConfig exists and what it accepts
        print(f"KVTransferConfig available: {KVTransferConfig}")
        import inspect
        sig = inspect.signature(KVTransferConfig.__init__)
        print(f"KVTransferConfig params: {list(sig.parameters.keys())}")

    except ImportError as e:
        print(f"KVTransferConfig not available: {e}")
        print("Trying alternative approach...")

    # Try with EngineArgs
    print("\n" + "="*60)
    print("Checking EngineArgs for KV connector support...")
    print("="*60)

    try:
        from vllm.engine.arg_utils import EngineArgs
        import inspect
        sig = inspect.signature(EngineArgs.__init__)
        params = list(sig.parameters.keys())

        kv_params = [p for p in params if 'kv' in p.lower() or 'connector' in p.lower() or 'transfer' in p.lower()]
        print(f"KV-related params in EngineArgs: {kv_params}")

        if kv_params:
            print("\nKV connector IS supported in this vLLM version!")
        else:
            print("\nKV connector NOT in EngineArgs - checking other configs...")

    except Exception as e:
        print(f"Error checking EngineArgs: {e}")

    # Try to find any kv_transfer related code
    print("\n" + "="*60)
    print("Checking vLLM for KV transfer support...")
    print("="*60)

    try:
        import vllm
        vllm_path = vllm.__path__[0]
        kv_transfer_path = os.path.join(vllm_path, "distributed", "kv_transfer")

        if os.path.exists(kv_transfer_path):
            print(f"KV transfer module exists at: {kv_transfer_path}")
            contents = os.listdir(kv_transfer_path)
            print(f"Contents: {contents}")

            connector_path = os.path.join(kv_transfer_path, "kv_connector")
            if os.path.exists(connector_path):
                print(f"\nConnector path exists: {connector_path}")
                v1_path = os.path.join(connector_path, "v1")
                if os.path.exists(v1_path):
                    print(f"V1 connectors: {os.listdir(v1_path)}")
        else:
            print(f"KV transfer module NOT found at: {kv_transfer_path}")

    except Exception as e:
        print(f"Error: {e}")

    # Check if our AnchorConnector is there
    print("\n" + "="*60)
    print("Checking for AnchorConnector...")
    print("="*60)

    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.anchor_connector import AnchorConnector
        print("AnchorConnector found and importable!")

        # Check its methods
        print(f"Methods: {[m for m in dir(AnchorConnector) if not m.startswith('_')]}")

    except ImportError as e:
        print(f"AnchorConnector not found: {e}")

    # Check factory registration
    print("\n" + "="*60)
    print("Checking connector factory...")
    print("="*60)

    try:
        from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
        print(f"Factory registry: {list(KVConnectorFactory._registry.keys())}")

        if "AnchorConnector" in KVConnectorFactory._registry:
            print("AnchorConnector IS registered!")
        else:
            print("AnchorConnector NOT registered in factory")
            print("We need to register it or use module path")

    except Exception as e:
        print(f"Error checking factory: {e}")

    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print("""
The AnchorConnector code exists, but the pre-built vLLM
doesn't expose KV connector via CLI args.

Options:
1. Build vLLM from source (enables --kv-connector CLI)
2. Modify vLLM's LLMEngine code to enable connector
3. Test connector logic in isolation (unit tests)
    """)

if __name__ == "__main__":
    main()
