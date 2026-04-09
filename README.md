# Engram LiteLLM Provider

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green)
![Tests: 53 passing](https://img.shields.io/badge/tests-53%20passing-brightgreen)

**Stateful inference for LiteLLM — drop-in backwards compatibility for stateless apps.**

---

## Overview

Every LLM conversation today is stateless. Your application sends the *entire* message history on every turn. The model re-reads everything from scratch, you pay for all those tokens again, and latency grows with every message.

The Engram LiteLLM Provider fixes this. It wraps [Engram](https://github.com/Clarit-AI/Engram)'s snapshot-based persistent state system as a [LiteLLM](https://github.com/BerriAI/litellm) provider, so any application already using LiteLLM can get stateful inference benefits — **without changing a single line of application code**.

The provider automatically detects redundant context in your messages, restores a saved model state snapshot (~2ms), and only processes the new tokens. On a 50-turn conversation, this can mean **93.8% fewer tokens processed**.

For applications that want explicit control, the provider also exposes all 5 Engram snapshot endpoints (save, restore, list, info, delete) as first-class methods.

---

## Features

- **Automatic backwards compatibility** — Stateless apps get stateful benefits with zero code changes
- **Three operating modes** — `auto` (default), `stateless`, and `explicit`
- **Full snapshot management** — Save, restore, list, inspect, and delete snapshots
- **Streaming support** — Sync restore before stream, async save after completion
- **Thread-safe** — Lock-protected conversation tracking for concurrent requests
- **Graceful degradation** — Restore failures fall back to full prefill silently
- **Distributed deployment awareness** — Warns when process-local tracking won't work

---

## Prerequisites

1. **An Engram server** running with snapshot persistence enabled:
   ```bash
   pip install -e "python/"  # from the Engram repo

   python -m sglang.launch_server \
     --model-path ibm-granite/granite-4.0-h-tiny \
     --enable-snapshot-persistence \
     --snapshot-dir ./snapshots \
     --mamba-scheduler-strategy no_buffer \
     --disable-radix-cache \
     --port 30000
   ```
   See the [Engram Quick Start](https://github.com/Clarit-AI/Engram#quick-start) for full setup instructions.

2. **A compatible Mamba/Mamba2 hybrid model**:

   | Model | Architecture | Status |
   |-------|-------------|--------|
   | IBM Granite 4.0-H-tiny (4B) | Mamba2+Attention hybrid | PASS |
   | IBM Granite 4.0-H-small (8B) | Mamba2+Attention hybrid | PASS |
   | NVIDIA Nemotron-Cascade-2-30B | MoE Mamba2 hybrid | PASS |
   | Alibaba Qwen3-Next-80B | Mamba2+Attn+MoE | PASS |
   | Codestral 7B | Pure Mamba2 | PASS |
   | NVIDIA Nemotron-3-Super-120B FP8 | LatentMoE Mamba2 hybrid | BLOCKED (SM89+) |

---

## Installation

```bash
# From source
pip install git+https://github.com/Clarit-AI/Engram-LiteLLM-Provider.git

# Or clone and install locally
git clone https://github.com/Clarit-AI/Engram-LiteLLM-Provider.git
cd Engram-LiteLLM-Provider
pip install -e .
```

Dependencies: `httpx`, and optionally `transformers` (for accurate token estimates).

---

## Quick Start

### 1. Backwards Compatible Mode (Zero Code Changes)

If your app already uses LiteLLM, just change the model prefix to `engram/`:

```python
from litellm.llms.engram import EngramChatConfig

config = EngramChatConfig()

# Your existing stateless code — sends full history every turn
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is quantum computing?"},
]

# First call — standard completion + auto-save snapshot
request = config.transform_request("engram/granite-4.0-h-tiny", messages, {}, {})
response, metadata = await config.async_completion(
    model="engram/granite-4.0-h-tiny",
    messages=messages,
    api_base="http://localhost:30000",
    api_key=None,
    headers={},
    optional_params={},
    request_data=request,
)

# Second call — provider detects prefix match, restores snapshot (~2ms),
# only processes the new message. Your app doesn't need to change.
messages.append({"role": "assistant", "content": response.json()["choices"][0]["message"]["content"]})
messages.append({"role": "user", "content": "Can you explain qubits?"})

request = config.transform_request("engram/granite-4.0-h-tiny", messages, {}, {})
response, metadata = await config.async_completion(
    model="engram/granite-4.0-h-tiny",
    messages=messages,
    api_base="http://localhost:30000",
    api_key=None,
    headers={},
    optional_params={},
    request_data=request,
)

print(f"Tokens saved: {metadata.tokens_saved}")
print(f"Restore time: {metadata.restore_time_ms}ms")
```

### 2. Explicit Stateful Mode

For production apps with session management:

```python
config = EngramChatConfig()

# Provide your own conversation ID for reliable tracking
messages = [{"role": "user", "content": "Hello"}]
request = config.transform_request(
    "engram/granite-4.0-h-tiny",
    messages,
    {"extra_body": {"conversation_id": "session-abc-123"}},
    {},
)

response, metadata = await config.async_completion(
    model="engram/granite-4.0-h-tiny",
    messages=messages,
    api_base="http://localhost:30000",
    api_key=None,
    headers={},
    optional_params={},
    request_data=request,
)

# Later — restore from a specific turn
request = config.transform_request(
    "engram/granite-4.0-h-tiny",
    [{"role": "user", "content": "Continue from where we left off"}],
    {"extra_body": {
        "conversation_id": "session-abc-123",
        "restore_from": "session-abc-123:3",  # Restore turn 3
    }},
    {},
)
```

### 3. Snapshot Management

```python
from litellm.llms.engram.snapshot.handler import SnapshotClient

client = SnapshotClient(api_base="http://localhost:30000")

# Save
await client.save_snapshot("session-abc", turn_number=5)

# List all snapshots for a conversation
snapshots = await client.list_snapshots("session-abc")
for s in snapshots:
    print(f"Turn {s.turn_number}: {s.snapshot_id} ({s.size_bytes} bytes, tier: {s.tier})")

# Get info on a specific snapshot
info = await client.get_snapshot_info("session-abc", turn_number=3)

# Restore
await client.restore_snapshot("session-abc", turn_number=3)

# Delete
await client.delete_snapshot("session-abc", turn_number=1)
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGRAM_BASE_URL` | `http://localhost:30000` | Engram server URL |
| `ENGRAM_API_KEY` | `None` | API key for authenticated servers |
| `ENGRAM_AUTO_SAVE` | `true` | Save snapshot after every turn |
| `ENGRAM_STATEFUL_MODE` | `auto` | Default operating mode (`auto`, `stateless`, `explicit`) |
| `ENGRAM_TOKENIZER_PATH` | `None` | Local tokenizer path (for air-gapped deployments) |

### Per-Call Parameters

Pass via `extra_body` in `optional_params`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `conversation_id` | `str` | Auto-generated | Stable session identifier (recommended for production) |
| `restore_from` | `str` | `None` | Restore target as `"conv_id:turn_number"` |
| `auto_save` | `bool` | `True` | Save snapshot after this turn |
| `stateful_mode` | `str` | `"auto"` | Mode for this call: `auto`, `stateless`, `explicit` |
| `branch_name` | `str` | `None` | Optional branch name for snapshot |

---

## Operating Modes

### Auto (Default)

The provider automatically:
1. Generates a pseudo-ID from the first 2 messages (or uses your `conversation_id`)
2. On subsequent calls, detects if the incoming messages are a prefix match with known state
3. If matched, restores the snapshot (~2ms) and only sends new messages for generation
4. Saves a new snapshot after the response

This is the **zero-config** mode. Your stateless app gets stateful benefits automatically.

### Stateless

Standard OpenAI-compatible pass-through. No snapshots, no state tracking. Useful for testing or when you want to bypass the stateful system entirely.

```python
optional_params = {"extra_body": {"stateful_mode": "stateless"}}
```

### Explicit

You manage snapshots directly via `conversation_id` and `restore_from`. The provider won't auto-detect prefixes — it only restores when you tell it to.

```python
optional_params = {"extra_body": {
    "stateful_mode": "explicit",
    "conversation_id": "my-session",
    "restore_from": "my-session:5",
}}
```

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Your Application                  │
│         (stateless, sends full message history)      │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│                 EngramChatConfig                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │   Tracker   │  │ ContextDiff │  │ SnapshotCli │ │
│  │  (conv IDs) │  │  (prefix)   │  │ (5 methods) │ │
│  └─────────────┘  └─────────────┘  └─────────────┘ │
│  ┌─────────────┐  ┌─────────────┐                   │
│  │ Tokenizer   │  │ StreamWrap  │                   │
│  │ (accounting)│  │ (async save)│                   │
│  └─────────────┘  └─────────────┘                   │
└─────────────────────┬───────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ restore_snapshot│     │ chat/completions│
│    (~2ms)       │     │  (generation)   │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
          ┌─────────────────┐
          │  save_snapshot   │
          │ (auto or manual) │
          └─────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│                   Engram Server                      │
│     (SGLang + Mamba + Snapshot Persistence)          │
└─────────────────────────────────────────────────────┘
```

### Component Overview

| Component | File | Purpose |
|-----------|------|---------|
| `EngramChatConfig` | `chat/transformation.py` | Main provider class, request/response transformation, async_completion override |
| `ConversationTracker` | `state/tracker.py` | Thread-safe conversation state tracking with pseudo-ID fingerprinting |
| `ContextDiffer` | `state/differ.py` | Prefix matching algorithm (strips up to 5 messages from tail) |
| `SnapshotClient` | `snapshot/handler.py` | Async HTTP client for all 5 Engram snapshot endpoints |
| `TokenizerClient` | `state/tokenizer.py` | Lazy-loaded HuggingFace tokenizer for token savings estimates |
| `EngramStreamWrapper` | `streaming/wrapper.py` | Context-aware stream wrapper with fire-and-forget save |

---

## How Backwards Compatibility Works

### The Problem

A typical multi-turn chat app sends the full message history on every turn:

```
Turn 1: [system, user1]                              → 50 tokens
Turn 2: [system, user1, asst1, user2]                → 120 tokens
Turn 3: [system, user1, asst1, user2, asst2, user3]  → 210 tokens
...
Turn 20: [system, user1, ..., user20]                 → 5,000 tokens
```

Total tokens processed across 20 turns: **~25,000 tokens**. Most of that is redundant re-processing.

### The Solution

The Engram provider detects that each new request is an extension of a known conversation:

```
Turn 1: Process [system, user1] → 50 tokens, save snapshot
Turn 2: Restore snapshot (2ms), process [asst1, user2] → 70 tokens, save
Turn 3: Restore snapshot (2ms), process [asst2, user3] → 90 tokens, save
...
Turn 20: Restore snapshot (2ms), process [asst19, user20] → ~250 tokens
```

Total tokens processed: **~1,550 tokens** — a 93.8% reduction.

### Prefix Detection Algorithm

1. Hash the stored message history for each known conversation
2. On each incoming request, strip 1 to 5 messages from the tail
3. Hash each stripped prefix and compare against known hashes
4. If a match is found: restore that snapshot, process only the delta messages
5. If no match: standard full-prefill completion

### Pseudo-ID Fingerprinting

When no `conversation_id` is provided, the provider generates a deterministic pseudo-ID from the first 2 messages (typically system prompt + first user message). This scopes prefix matching to prevent cross-conversation contamination. If a fingerprint collision is detected, the provider skips stateful optimization for that call and logs a warning.

---

## Streaming

The provider handles streaming with restore-before and save-after semantics:

1. **Restore** — Synchronous, before the stream starts (~2ms)
2. **Stream** — SSE chunks proxied through to your application
3. **Save** — Asynchronous, after the `[DONE]` chunk (fire-and-forget)

If your application cancels the stream mid-generation (`GeneratorExit`), the save is **skipped** — partial-turn state is not saved.

The stream wrapper automatically detects sync vs async context:
- **Async context** (event loop running): Uses `asyncio.create_task()` for the save
- **Sync context** (no event loop): Spawns a daemon thread

---

## Error Handling

| Failure | Behavior |
|---------|----------|
| Restore HTTP error (4xx/5xx) | Log warning, fall back to full prefill, set `metadata.restore_failed = True` |
| Snapshot not found (404) | Same as above — fallback to full prefill |
| Save failure after completion | Log warning silently, do not surface to caller |
| Save failure after stream | Log warning silently, stream is already complete |
| Tokenizer load failure | Skip `tokens_saved` metadata, continue normally |
| Fingerprint collision | Skip prefix match, fall back to full prefill, log warning |

### Distributed Deployment

The `ConversationTracker` is process-local. In multi-worker deployments (gunicorn, uvicorn with multiple workers), each worker maintains independent state. Cross-worker requests will silently fall back to full prefill.

**Recommendations for distributed deployments:**
- Always provide an explicit `conversation_id` via `extra_body`
- Use `stateful_mode: "explicit"` for predictable behavior
- Future: configure `ENGRAM_TRACKER_REDIS_URL` for shared state (not yet implemented)

---

## Response Metadata

Every response includes Engram-specific metadata:

```python
response, metadata = await config.async_completion(...)

metadata.conversation_id   # str — conversation identifier used
metadata.turn_number       # int — current turn number
metadata.snapshot_id       # str — snapshot ID if auto-saved
metadata.tokens_saved      # int — estimated tokens not reprocessed
metadata.restore_time_ms   # float — snapshot restore latency in ms
metadata.restore_failed    # bool — True if restore fell back to full prefill
metadata.restore_error     # str — error message if restore failed
metadata.auto_save         # bool — whether auto-save was enabled
```

---

## Development

### Running Tests

```bash
# Install test dependencies
pip install httpx pytest pytest-asyncio

# Run all tests (53 tests, no GPU required)
pytest tests/ -v
```

### Test Structure

```
tests/test_litellm/llms/engram/
├── test_transformation.py      # EngramChatConfig, side-channel, request/response
├── test_tracker.py             # ConversationTracker, thread safety, collision
├── test_differ.py              # ContextDiffer prefix matching algorithm
├── test_streaming.py           # StreamWrapper sync/async, cancellation
├── test_snapshot_methods.py    # SnapshotClient CRUD operations
├── test_restore_fallback.py    # Restore failure → full prefill fallback
├── test_collision.py           # Fingerprint collision detection
└── fixtures/
    └── mock_engram_server.py   # httpx transport mock for Engram endpoints
```

### Source Structure

```
litellm/llms/engram/
├── chat/
│   └── transformation.py      # EngramChatConfig (main provider class)
├── snapshot/
│   └── handler.py             # SnapshotClient (async HTTP client)
├── state/
│   ├── tracker.py             # ConversationTracker (thread-safe)
│   ├── differ.py              # ContextDiffer (prefix matching)
│   └── tokenizer.py           # TokenizerClient (HF lazy-load)
├── streaming/
│   └── wrapper.py             # EngramStreamWrapper
├── errors.py                  # RestoreError, SaveError, SnapshotError
└── types.py                   # Dataclasses (ConversationState, PrefixMatch, etc.)
```

---

## Requirements

- Python 3.9+
- [httpx](https://www.python-httpx.org/) (HTTP client)
- [LiteLLM](https://github.com/BerriAI/litellm) (for integration into the LiteLLM ecosystem)
- [transformers](https://github.com/huggingface/transformers) (optional — for accurate token estimates; falls back to character-based approximation)

---

## License

Apache-2.0 — same license as [Engram](https://github.com/Clarit-AI/Engram) and [LiteLLM](https://github.com/BerriAI/litellm).

---

## Links

- **Engram** — https://github.com/Clarit-AI/Engram
- **LiteLLM** — https://github.com/BerriAI/litellm
- **Engram API Guide** — https://github.com/Clarit-AI/Engram/blob/main/docs/stateful_mamba/api_guide.md
- **LiteLLM Provider Registration** — https://docs.litellm.ai/docs/provider_registration/
- **Clarit AI** — https://clarit.ai

---

## Acknowledgments

Built on [Engram](https://github.com/Clarit-AI/Engram) by [Clarit AI](https://clarit.ai), which extends [SGLang](https://github.com/sgl-project/sglang) with persistent Mamba state management. Designed for the [LiteLLM](https://github.com/BerriAI/litellm) ecosystem by [BerriAI](https://github.com/BerriAI).

The Mamba architecture was developed by Albert Gu and Tri Dao. Mamba2 was developed by Tri Dao and Albert Gu at Carnegie Mellon University.
