# SPDX-License-Identifier: Apache-2.0
"""
Anchor-based State Storage for Kimi-Linear (NoPE) models

================================================================================
SYSTEM OVERVIEW
================================================================================

Each anchor represents a TEXT CHUNK (not cumulative):

    [text1]<anchor1/>[text2]<anchor2/>[text3]<anchor3/>
    └─────┬────────┘└─────┬────────┘└─────┬────────┘
       anchor1.pt      anchor2.pt      anchor3.pt

Storage per anchor: K,V for that chunk only (~500 tokens = ~24MB)
60 anchors × 500 tokens = ~1.4 GB total (manageable)

================================================================================
BACKGROUND JOB - When new message comes in
================================================================================

Three options for processing new text:

OPTION A: Full text input + loaded K,V
    Input:  [text1]<anchor1/>[text2]<anchor2/>[text3]<anchor3/>[text4]<anchor4/>
    K,V:    └─────load─────┘└─────load─────┘└─────load─────┘└───compute───┘
    - Full text in input
    - K,V loaded from disk for old chunks
    - New chunk computed and saved

OPTION B: Compressed input + loaded K,V
    Input:  <anchor1/><anchor2/><anchor3/>[text4]<anchor4/>
    K,V:    └──load──┘└──load──┘└──load──┘└───compute───┘
                ↓         ↓         ↓
           (K,V for   (K,V for   (K,V for
            text1)     text2)     text3)
    - Input is compressed (short)
    - K,V cache still has full text1, text2, text3
    - NoPE allows this mismatch - model attends to full K,V
    - More efficient (shorter input)

OPTION C: Selective expansion (rule-based) + loaded K,V
    Input:  <a1/><a2/>...<a56/>[text57][text58][text59][text60][text_new]<a_new/>
            └── compressed ──┘└───── recent 4: always expanded ────┘└─ compute ─┘
    K,V:    └─────────────── load relevant (BM25) ──────────────┘└── compute ──┘
    - Text expansion: RULE-BASED (recent N messages expanded)
    - K,V loading: BM25-BASED (load relevant anchors up to limit)
    - Most practical for production

EXAMPLE - Background Job Flow:
    New message arrives. Process and save K,V for new anchor.

    Input to model (WITH anchor tokens - need K,V at anchor positions):
    ┌──────────────────────────────────────────────────────────────────┐
    │ <a1/><a2/>...<a56/>                                              │  ← Old: compressed
    │ Bob said he would meet everyone at noon.<a57/>                   │  ← Recent: WITH anchor
    │ Charlie brought snacks for the picnic.<a58/>                     │  ← Recent: WITH anchor
    │ Diana checked the weather forecast.<a59/>                        │  ← Recent: WITH anchor
    │ Eve reminded everyone about the dress code.<a60/>                │  ← Recent: WITH anchor
    │ [NEW] Frank suggested visiting the museum tomorrow.<a61/>        │  ← NEW: compute K,V
    └──────────────────────────────────────────────────────────────────┘

    K,V loaded from disk: a1-a56 (compressed anchors)
    K,V computed fresh: a57-a61 (recent + new)

    At anchor position <a61/>:
        - Extract KDA recurrent_state (36 layers × 8KB = 288KB)
        - Extract MLA K,V for text61 chunk (12 layers × 500 tokens = 12MB)
        - Save to disk: anchor_cache/session_123/a61.pt

    Key difference from inference:
        - Background: INCLUDE anchor tokens (need to save K,V at those positions)
        - Inference:  EXCLUDE anchor tokens (faster, not saving anything)

================================================================================
INFERENCE - When user queries
================================================================================

Flow:
    1. User sends message
    2. Build compressed context (old anchors as tags, recent expanded)
    3. Extract keywords (fast LLM) → BM25 search
    4. Load K,V for relevant anchors from disk (up to limit)
    5. Inject K,V into model
    6. Generate response (main LLM)

Retrieval limits (from memory-context-builder.ts):
    0-100 anchors:   30% (min 3, max 30)
    100-300 anchors: 20 + 20% above 100
    300+ anchors:    40 + 10% above 300 (cap 60)

EXAMPLE - Inference Flow:
    Context has 60 anchors. User asks: "What did Alice say about the garden?"

    Step 1-2: Build compressed context
        Input to model (NO anchor tokens - faster inference):
        ┌──────────────────────────────────────────────────────────────────┐
        │ <a1/><a2/>...<a56/>                                              │  ← Old: compressed
        │ Bob said he would meet everyone at noon.                         │  ← Recent: text57
        │ Charlie brought snacks for the picnic.                           │  ← Recent: text58
        │ Diana checked the weather forecast.                              │  ← Recent: text59
        │ Eve reminded everyone about the dress code.                      │  ← Recent: text60
        │ [USER] What did Alice say about the garden?                      │  ← Query
        └──────────────────────────────────────────────────────────────────┘

    Step 3: Extract keywords (fast LLM)
        Keywords: ["Alice", "garden", "said", "flowers", "planting"]

    Step 4: BM25 search → Load relevant anchors
        BM25 returns: [a3, a12, a28, a45] (anchors mentioning Alice/garden)
        Load K,V from disk: ~100MB into GPU (4 anchors × 25MB each)

    Step 5: Inject K,V into model
        KDA layers (36): Inject recurrent_state for [a3, a12, a28, a45]
        MLA layers (12): Inject K,V cache for [a3, a12, a28, a45]

        Model now "remembers" content from those anchors even though
        input only shows <a3/><a12/><a28/><a45/> tags!

    Step 6: Generate response

        TEXT INPUT (what model processes as tokens):
        ┌─────────────────────────────────────────────────────────────┐
        │ <a1/><a2/>...<a56/>                                         │  ← ~60 tokens
        │ Bob said he would meet everyone at noon.                    │  ← ~10 tokens
        │ Charlie brought snacks for the picnic.                      │  ← ~8 tokens
        │ Diana checked the weather forecast.                         │  ← ~7 tokens
        │ Eve reminded everyone about the dress code.                 │  ← ~9 tokens
        │ [USER] What did Alice say about the garden?                 │  ← ~10 tokens
        └─────────────────────────────────────────────────────────────┘
                            Total: ~104 tokens (SHORT!)

        INJECTED K,V CACHE (what model attends to):
        ┌─────────────────────────────────────────────────────────────┐
        │ a3:  "Alice walked through the garden admiring roses..."    │  ← 500 tokens K,V
        │ a12: "Alice told Bob about her garden plans..."             │  ← 500 tokens K,V
        │ a28: "Bob asked Alice about the soil quality..."            │  ← 500 tokens K,V
        │ a45: "Alice showed everyone her garden sketch..."           │  ← 500 tokens K,V
        └─────────────────────────────────────────────────────────────┘
                            Total: ~2000 tokens of K,V (RICH CONTEXT!)

        MODEL ATTENTION:
        ┌─────────────────────────────────────────────────────────────┐
        │   Text tokens (104) ──┐                                     │
        │                       ▼                                     │
        │              ┌────────────────┐                             │
        │   K,V ──────▶│   ATTENTION    │───▶ Response                │
        │   (2000)     └────────────────┘                             │
        │                                                             │
        │   NoPE: K,V has no position info, loadable anywhere!        │
        └─────────────────────────────────────────────────────────────┘

        OUTPUT: "Alice mentioned she wanted to plant roses in the
        garden (anchor 12) and discussed soil quality with Bob
        (anchor 28)..."

    Key insight: Input is SHORT (compressed tags + recent + query)
                 but model attends to FULL K,V from loaded anchors.
                 NoPE allows this position mismatch!

================================================================================
STORAGE MODEL - Kimi-Linear hybrid efficiency
================================================================================

Kimi-Linear uses NoPE (No Position Encoding), making K,V position-independent.

KDA Layers (36/48 = 75%):
    - Linear attention with recurrent state
    - Storage: recurrent_state only (fixed ~8KB per layer)
    - O(1) memory regardless of chunk size
    - Recurrent state compresses context

MLA Layers (12/48 = 25%):
    - Standard attention with K,V cache (NoPE)
    - Storage: K,V for chunk tokens
    - O(chunk_size) per layer
    - NoPE allows loading at any position

Storage example (500 tokens/chunk, 60 anchors):
    KDA: 36 layers × 8KB = 288KB per anchor → 17MB total
    MLA: 12 layers × 500 × 2KB = 12MB per anchor → 720MB total (dominates)
    Combined: ~1.4GB total on disk

At inference (20 relevant anchors loaded):
    ~470MB into GPU (manageable)
================================================================================
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

# Try to import vllm logger, fall back to standard logging
try:
    from vllm.logger import init_logger
    logger = init_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


@dataclass
class AnchorMetadata:
    """Metadata for an anchor (from LLM compression)."""
    seq_len: int  # Number of tokens in this anchor's text chunk
    accessible_to: list[str] = field(default_factory=lambda: ["*"])  # ["*"] = all, ["alice"] = only alice
    block_tag: str = ""  # Character name or content type (e.g., "alice", "scenario")
    context: dict = field(default_factory=dict)  # {timestamp, location, characters_present}


@dataclass
class AnchorState:
    """
    State data for a single semantic anchor across all layers.

    Anchor naming convention:
        - kebab-case with 5 meaningful words
        - e.g., "alice-admired-roses-plant-flowers"
        - NOT sequential like "a1", "a2", "a3"

    For KDA layers (linear attention):
        - Stores: recurrent_state tensor
        - Shape: [num_heads, head_dim, state_dim] or similar
        - Size: O(1) fixed, ~8KB per layer
        - This is a COMPRESSED representation of context

    For MLA layers (standard attention with NoPE):
        - Stores: K,V cache for this anchor's text chunk
        - Shape: keys [seq_len, num_heads, head_dim], values [seq_len, num_heads, head_dim]
        - Size: O(chunk_size) per layer, ~500 tokens = ~12MB for 12 layers
        - NoPE allows these to be loaded at different positions during inference

    Example storage for anchor "alice-admired-roses-plant-flowers" (~500 tokens):
        KDA (36 layers): 36 × 8KB = 288KB
        MLA (12 layers): 12 × 500 × 2KB = 12MB
        Total: ~12MB per anchor
    """
    anchor_id: str  # Semantic name (e.g., "alice-admired-roses-plant-flowers")

    # Metadata from LLM compression
    metadata: AnchorMetadata = field(default_factory=lambda: AnchorMetadata(seq_len=0))

    # KDA layers: {layer_idx: recurrent_state_tensor}
    # - recurrent_state is O(1) fixed size
    kda_states: dict[int, torch.Tensor] = field(default_factory=dict)

    # MLA layers: {layer_idx: (keys, values)}
    # - keys/values contain K,V for this anchor's chunk only
    mla_kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)

    # Track which layers are which type
    layer_types: dict[int, str] = field(default_factory=dict)  # {layer_idx: "kda" | "mla"}


class AnchorStorage:
    """
    Simple storage backend for semantic anchor states.

    Directory structure:
        cache_dir/
        └── {session_id}/
            ├── index.json                              # Anchor index with metadata
            └── {anchor_id}.pt                          # State for one anchor (all layers)

    Example anchor IDs (semantic, not sequential):
        - "alice-admired-roses-plant-flowers"
        - "bob-shares-weekend-plans-hiking"
        - "charlie-jokes-about-weather-rain"

    Usage:
        storage = AnchorStorage("/path/to/cache")

        # Background (Step 1): save anchor states after inference
        storage.save_anchor_state(session_id, "alice-admired-roses-plant-flowers", state)

        # Inference (Step 2.6): load relevant anchors from BM25
        states = storage.load_anchor_states(session_id, [
            "alice-admired-roses-plant-flowers",
            "alice-garden-plans-vegetable-section",
        ])
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _session_dir(self, session_id: str) -> Path:
        return self.cache_dir / session_id

    def _anchor_path(self, session_id: str, anchor_id: str) -> Path:
        # Sanitize anchor_id for filesystem
        safe_id = anchor_id.replace("/", "_").replace("\\", "_")
        return self._session_dir(session_id) / f"{safe_id}.pt"

    def _index_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "index.json"

    def has_session(self, session_id: str) -> bool:
        """Check if we have cached states for this session"""
        return self._index_path(session_id).exists()

    def get_anchor_ids(self, session_id: str) -> list[str]:
        """Get list of cached anchor IDs for a session"""
        index_path = self._index_path(session_id)
        if not index_path.exists():
            return []
        with open(index_path, "r") as f:
            return json.load(f).get("anchor_ids", [])

    def save_all_anchor_states(
        self,
        session_id: str,
        anchor_states: dict[str, AnchorState],
        metadata: dict | None = None,
    ):
        """
        Save states for all anchors in a session (background job).

        Args:
            session_id: Chat/session identifier
            anchor_states: {anchor_id: AnchorState}
            metadata: Optional metadata (model_id, timestamp, etc.)
        """
        session_dir = self._session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)

        # Save each anchor state
        for anchor_id, state in anchor_states.items():
            anchor_path = self._anchor_path(session_id, anchor_id)
            torch.save({
                "anchor_id": state.anchor_id,
                "metadata": {
                    "seq_len": state.metadata.seq_len,
                    "accessible_to": state.metadata.accessible_to,
                    "block_tag": state.metadata.block_tag,
                    "context": state.metadata.context,
                },
                "kda_states": state.kda_states,      # O(1) per layer
                "mla_kv_cache": state.mla_kv_cache,  # O(n) per layer
                "layer_types": state.layer_types,
            }, anchor_path)

        # Save index
        index_data = {
            "anchor_ids": list(anchor_states.keys()),
            "metadata": metadata or {},
        }
        with open(self._index_path(session_id), "w") as f:
            json.dump(index_data, f, indent=2)

        logger.info(f"Saved {len(anchor_states)} anchor states for session {session_id}")

    def save_anchor_state(
        self,
        session_id: str,
        anchor_id: str,
        state: AnchorState,
    ):
        """Save state for a single semantic anchor (Step 1: Background Job)."""
        session_dir = self._session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)

        # Save anchor state
        anchor_path = self._anchor_path(session_id, anchor_id)
        torch.save({
            "anchor_id": state.anchor_id,
            "metadata": {
                "seq_len": state.metadata.seq_len,
                "accessible_to": state.metadata.accessible_to,
                "block_tag": state.metadata.block_tag,
                "context": state.metadata.context,
            },
            "kda_states": state.kda_states,      # O(1) per layer
            "mla_kv_cache": state.mla_kv_cache,  # O(n) per layer
            "layer_types": state.layer_types,
        }, anchor_path)

        # Update index
        anchor_ids = self.get_anchor_ids(session_id)
        if anchor_id not in anchor_ids:
            anchor_ids.append(anchor_id)
            index_data = {"anchor_ids": anchor_ids, "metadata": {}}
            with open(self._index_path(session_id), "w") as f:
                json.dump(index_data, f, indent=2)

    def load_anchor_states(
        self,
        session_id: str,
        anchor_ids: list[str],
    ) -> dict[str, AnchorState]:
        """
        Load states for specific semantic anchors (Step 2.6: after BM25 retrieval).

        Args:
            session_id: Chat/session identifier
            anchor_ids: List of semantic anchor IDs from BM25 results
                       e.g., ["alice-admired-roses-plant-flowers", "bob-shares-plans"]

        Returns:
            {anchor_id: AnchorState} for found anchors
        """
        result = {}
        for anchor_id in anchor_ids:
            anchor_path = self._anchor_path(session_id, anchor_id)
            if not anchor_path.exists():
                logger.warning(f"Anchor state not found: {anchor_id}")
                continue

            data = torch.load(anchor_path, weights_only=False)

            # Load metadata
            meta_dict = data.get("metadata", {})
            metadata = AnchorMetadata(
                seq_len=meta_dict.get("seq_len", 0),
                accessible_to=meta_dict.get("accessible_to", ["*"]),
                block_tag=meta_dict.get("block_tag", ""),
                context=meta_dict.get("context", {}),
            )

            result[anchor_id] = AnchorState(
                anchor_id=data["anchor_id"],
                metadata=metadata,
                kda_states=data.get("kda_states", {}),      # O(1) per layer
                mla_kv_cache=data.get("mla_kv_cache", {}),  # O(n) per layer
                layer_types=data.get("layer_types", {}),
            )

        logger.info(f"Loaded {len(result)}/{len(anchor_ids)} anchor states")
        return result

    def delete_session(self, session_id: str):
        """Delete all cached states for a session"""
        import shutil
        session_dir = self._session_dir(session_id)
        if session_dir.exists():
            shutil.rmtree(session_dir)
            logger.info(f"Deleted session cache: {session_id}")


class AnchorStateExtractor:
    """
    Extracts states at anchor positions during background inference.

    For KDA layers: Extract recurrent_state (O(1) fixed size)
    For MLA layers: Extract full K,V cache up to anchor position (O(n))
    """

    def __init__(self, anchor_ids: list[str], anchor_token_positions: dict[str, int]):
        """
        Args:
            anchor_ids: List of anchor IDs to extract
            anchor_token_positions: {anchor_id: token_position}
        """
        self.anchor_ids = anchor_ids
        self.anchor_positions = anchor_token_positions
        self.states: dict[str, AnchorState] = {
            aid: AnchorState(anchor_id=aid) for aid in anchor_ids
        }

    def extract_kda_state(
        self,
        layer_idx: int,
        recurrent_state: torch.Tensor,
        anchor_id: str,
    ):
        """
        Extract KDA recurrent state for an anchor.

        Args:
            layer_idx: Layer index
            recurrent_state: Recurrent state tensor (O(1) fixed size)
            anchor_id: Which anchor this is for
        """
        if anchor_id not in self.states:
            return
        self.states[anchor_id].kda_states[layer_idx] = recurrent_state.clone().cpu()
        self.states[anchor_id].layer_types[layer_idx] = "kda"

    def extract_mla_kv(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        anchor_id: str,
    ):
        """
        Extract MLA K,V cache for an anchor.

        Note: keys/values should contain ALL tokens up to anchor position (O(n)).

        Args:
            layer_idx: Layer index
            keys: Key tensor [seq_len, num_heads, head_dim] - all tokens up to anchor
            values: Value tensor [seq_len, num_heads, head_dim] - all tokens up to anchor
            anchor_id: Which anchor this is for
        """
        if anchor_id not in self.states:
            return
        k = keys.clone().cpu()
        v = values.clone().cpu()
        self.states[anchor_id].mla_kv_cache[layer_idx] = (k, v)
        self.states[anchor_id].layer_types[layer_idx] = "mla"

    def get_all_states(self) -> dict[str, AnchorState]:
        """Get all extracted states"""
        return self.states


class AnchorStateInjector:
    """
    Injects pre-computed states into model during inference (Step 2.6-2.7).

    Called after BM25 retrieval to load relevant anchor states.

    For KDA layers: Inject recurrent_state to restore "memory" of full context
    For MLA layers: Inject full K,V cache so attention can see all tokens

    Performance optimization:
        Use get_concatenated_mla_kv() for pre-allocated tensor concatenation
        instead of multiple torch.cat() calls.
    """

    def __init__(self, anchor_states: dict[str, AnchorState]):
        """
        Args:
            anchor_states: {anchor_id: AnchorState} from storage.load_anchor_states()
                          e.g., {"alice-admired-roses": AnchorState(...), ...}
        """
        self.anchor_states = anchor_states
        self._total_seq_len: int | None = None

    def get_total_seq_len(self) -> int:
        """Get total sequence length across all loaded anchors."""
        if self._total_seq_len is None:
            self._total_seq_len = sum(
                state.metadata.seq_len for state in self.anchor_states.values()
            )
        return self._total_seq_len

    def get_kda_state(self, layer_idx: int, anchor_id: str) -> Optional[torch.Tensor]:
        """Get KDA recurrent state for injection (O(1) fixed size)"""
        if anchor_id not in self.anchor_states:
            return None
        state = self.anchor_states[anchor_id]
        return state.kda_states.get(layer_idx)

    def get_mla_kv(
        self, layer_idx: int, anchor_id: str
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Get MLA K,V cache for injection (O(n) - all tokens up to anchor)"""
        if anchor_id not in self.anchor_states:
            return None
        state = self.anchor_states[anchor_id]
        return state.mla_kv_cache.get(layer_idx)

    def get_all_kda_states_for_layer(self, layer_idx: int) -> dict[str, torch.Tensor]:
        """Get all KDA recurrent states for a specific layer"""
        result = {}
        for anchor_id, state in self.anchor_states.items():
            if layer_idx in state.kda_states:
                result[anchor_id] = state.kda_states[layer_idx]
        return result

    def get_all_mla_kv_for_layer(
        self, layer_idx: int
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Get all MLA K,V caches for a specific layer"""
        result = {}
        for anchor_id, state in self.anchor_states.items():
            if layer_idx in state.mla_kv_cache:
                result[anchor_id] = state.mla_kv_cache[layer_idx]
        return result

    def get_layer_type(self, layer_idx: int) -> Optional[str]:
        """Get the layer type (kda or mla) for a layer index"""
        for state in self.anchor_states.values():
            if layer_idx in state.layer_types:
                return state.layer_types[layer_idx]
        return None

    def get_concatenated_mla_kv(
        self,
        layer_idx: int,
        device: str = "cuda",
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """
        Get concatenated K,V for all anchors with pre-allocation (FAST).

        Instead of multiple torch.cat() calls, this:
        1. Calculates total sequence length from metadata
        2. Pre-allocates one big tensor
        3. Fills in each anchor's K,V directly

        Args:
            layer_idx: Layer index
            device: Target device ("cuda" or "cpu")

        Returns:
            (keys, values) tensors with shape [total_seq_len, num_heads, head_dim]
            or None if no MLA data for this layer
        """
        # Collect all K,V for this layer
        kv_list = []
        for anchor_id, state in self.anchor_states.items():
            if layer_idx in state.mla_kv_cache:
                k, v = state.mla_kv_cache[layer_idx]
                kv_list.append((anchor_id, k, v, state.metadata.seq_len))

        if not kv_list:
            return None

        # Get shape from first K,V
        _, first_k, first_v, _ = kv_list[0]
        num_heads = first_k.shape[1]
        head_dim = first_k.shape[2]
        total_seq_len = sum(seq_len for _, _, _, seq_len in kv_list)

        # Pre-allocate (FAST - single allocation)
        keys = torch.zeros(total_seq_len, num_heads, head_dim, device=device)
        values = torch.zeros(total_seq_len, num_heads, head_dim, device=device)

        # Fill in directly (no torch.cat overhead)
        offset = 0
        for anchor_id, k, v, seq_len in kv_list:
            actual_len = k.shape[0]  # Use actual tensor size
            keys[offset:offset + actual_len] = k.to(device)
            values[offset:offset + actual_len] = v.to(device)
            offset += actual_len

        return keys, values

    def get_combined_kda_state(
        self,
        layer_idx: int,
        device: str = "cuda",
    ) -> Optional[torch.Tensor]:
        """
        Combine KDA recurrent states from all anchors.

        For KDA layers, recurrent states need to be combined (typically summed
        or averaged) to represent the combined context from all loaded anchors.

        Args:
            layer_idx: Layer index
            device: Target device

        Returns:
            Combined recurrent state tensor or None if no KDA data
        """
        states = []
        for state in self.anchor_states.values():
            if layer_idx in state.kda_states:
                states.append(state.kda_states[layer_idx])

        if not states:
            return None

        # Stack and sum (or could use other combination strategies)
        stacked = torch.stack([s.to(device) for s in states], dim=0)
        return stacked.sum(dim=0)  # Combine recurrent states


# =============================================================================
# High-level API
# =============================================================================

def create_storage(cache_dir: str = "/tmp/anchor_kv_cache") -> AnchorStorage:
    """Create storage instance"""
    return AnchorStorage(cache_dir)


def save_session_states(
    storage: AnchorStorage,
    session_id: str,
    anchor_states: dict[str, AnchorState],
    metadata: dict | None = None,
):
    """
    Save all anchor states for a session (background job).

    Args:
        storage: AnchorStorage instance
        session_id: Chat/session ID
        anchor_states: {anchor_id: AnchorState}
        metadata: Optional metadata
    """
    storage.save_all_anchor_states(session_id, anchor_states, metadata)


def load_relevant_states(
    storage: AnchorStorage,
    session_id: str,
    anchor_ids: list[str],
) -> AnchorStateInjector:
    """
    Load states for relevant anchors (after BM25).

    Args:
        storage: AnchorStorage instance
        session_id: Chat/session ID
        anchor_ids: List of anchor IDs from BM25 retrieval

    Returns:
        AnchorStateInjector ready for inference
    """
    states = storage.load_anchor_states(session_id, anchor_ids)
    return AnchorStateInjector(states)
