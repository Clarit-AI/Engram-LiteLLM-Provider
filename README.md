# Engram LiteLLM Provider

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green)
![Tests: 53 passing](https://img.shields.io/badge/tests-53%20passing-brightgreen)

Stateful inference for LiteLLM. Add snapshot-based state management to any existing LiteLLM app without changing a line of application code.

---

## Problem

Every LLM conversation today is stateless. Your application sends the entire message history on every turn. The model re-reads everything from scratch. You pay for all those tokens again, and latency grows with every message.

On a 50-turn conversation, a stateless system processes approximately 25,000 tokens total. Most of that is redundant. Standard alternatives require either major application rewrites or inference server fork maintenance.

---

## Solution

The Engram LiteLLM Provider wraps Engram's snapshot-based persistent state system as a LiteLLM provider. The provider automatically detects redundant context in your messages, restores a saved model state snapshot (approximately 2ms), and only processes the new tokens.

This works as a drop-in replacement. Stateless apps get stateful benefits with zero code changes. On a 50-turn conversation, prefix detection reduces token processing by 93.8%, bringing the total from 25,000 tokens down to 1,550 tokens.

For applications that want explicit control, the provider also exposes all 5 Engram snapshot endpoints (save, restore, list, info, delete) as first-class methods.

---

## Proof

**Token savings**: 93.8% reduction on repeated conversation prefixes (tested on 50-turn interaction: 25,000 tokens reduced to 1,550).

**Restore latency**: 2ms average for model state restore using Engram's snapshot persistence.

**Test coverage**: 53 passing tests covering transformation, state tracking, prefix matching, streaming, and distributed deployment scenarios.

**Validated models**: IBM Granite, NVIDIA Nemotron, Alibaba Qwen3, Codestral, and hybrid Mamba2 architectures.

| Model | Architecture | Status |
|-------|-------------|--------|
| IBM Granite 4.0-H-tiny (4B) | Mamba2 + Attention hybrid | PASS |
| IBM Granite 4.0-H-small (8B) | Mamba2 + Attention hybrid | PASS |
| NVIDIA Nemotron-Cascade-2-30B | MoE Mamba2 hybrid | PASS |
| Alibaba Qwen3-Next-80B | Mamba2 + Attn + MoE | PASS |
| Codestral 7B | Pure Mamba2 | PASS |
| NVIDIA Nemotron-3-Super-120B FP8 | LatentMoE Mamba2 hybrid | BLOCKED (SM89+) |

---

## Quick Start

### Install

```bash
# From GitHub
pip install git+https://github.com/Clarit-AI/Engram-LiteLLM-Provider.git

# Or clone and install locally
git clone https://github.com/Clarit-AI/Engram-LiteLLM-Provider.git
cd Engram-LiteLLM-Provider
pip install -e .

# With tokenizer support for accurate token estimates
pip install -e ".[tokenizer]"
```

### Run Engram Server

```bash
# Install Engram
pip install -e "python/"  # from the Engram repo

# Start the server with snapshot persistence
python -m sglang.launch_server \
  --model-path ibm-granite/granite-4.0-h-tiny \
  --enable-snapshot-persistence \
  --snapshot-dir ./snapshots \
  --mamba-scheduler-strategy no_buffer \
  --disable-radix-cache \
  --port 30000
```

### Use with LiteLLM

```python
from engram_litellm import EngramChatConfig

config = EngramChatConfig()

# First turn: standard completion, auto-save snapshot
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is quantum computing?"},
]

response, metadata = await config.async_completion(
    model="engram/granite-4.0-h-tiny",
    messages=messages,
    api_base="http://localhost:30000",
)

# Second turn: provider detects prefix, restores snapshot (2ms),
# processes only new message. Application code unchanged.
messages.append({"role": "assistant", "content": response.json()["choices"][0]["message"]["content"]})
messages.append({"role": "user", "content": "Can you explain qubits?"})

response, metadata = await config.async_completion(
    model="engram/granite-4.0-h-tiny",
    messages=messages,
    api_base="http://localhost:30000",
)

print(f"Tokens saved: {metadata.tokens_saved}")
print(f"Restore time: {metadata.restore_time_ms}ms")
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENGRAM_BASE_URL` | `http://localhost:30000` | Engram server URL |
| `ENGRAM_API_KEY` | `None` | API key for authenticated servers |
| `ENGRAM_AUTO_SAVE` | `true` | Save snapshot after every turn |
| `ENGRAM_STATEFUL_MODE` | `auto` | Default operating mode: `auto`, `stateless`, or `explicit` |
| `ENGRAM_TOKENIZER_PATH` | `None` | Local tokenizer path for air-gapped deployments |

### Per-Call Parameters

Pass via `extra_body` in `optional_params`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `conversation_id` | `str` | Auto-generated | Stable session identifier (recommended for production) |
| `restore_from` | `str` | `None` | Restore target as `"conv_id:turn_number"` |
| `auto_save` | `bool` | `True` | Save snapshot after this turn |
| `stateful_mode` | `str` | `"auto"` | Mode for this call: `auto`, `stateless`, or `explicit` |
| `branch_name` | `str` | `None` | Optional branch name for snapshot |

---

## Operating Modes

### Auto (Default)

The provider automatically:
1. Generates a pseudo-ID from the first 2 messages (or uses your `conversation_id`)
2. On subsequent calls, detects if the incoming messages are a prefix match with known state
3. If matched, restores the snapshot (2ms) and sends only new messages for generation
4. Saves a new snapshot after the response

This is the zero-config mode. Stateless applications get stateful benefits automatically.

### Stateless

Standard OpenAI-compatible pass-through. No snapshots, no state tracking. Useful for testing or bypassing the stateful system.

```python
optional_params = {"extra_body": {"stateful_mode": "stateless"}}
```

### Explicit

You manage snapshots directly via `conversation_id` and `restore_from`. The provider does not auto-detect prefixes. It only restores when you specify it.

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
│restore_snapshot │     │ chat/completions│
│    (2ms)        │     │  (generation)   │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
          ┌─────────────────┐
          │ save_snapshot   │
          │ (auto or manual)│
          └─────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│                   Engram Server                      │
│     (SGLang + Mamba + Snapshot Persistence)          │
└─────────────────────────────────────────────────────┘
```

### Components

| Component | File | Purpose |
|-----------|------|---------|
| `EngramChatConfig` | `transformation.py` | Main provider class, request/response transformation |
| `ConversationTracker` | `tracker.py` | Thread-safe conversation state tracking with pseudo-ID fingerprinting |
| `ContextDiffer` | `differ.py` | Prefix matching algorithm (strips 1 to 5 messages from tail) |
| `SnapshotClient` | `snapshot.py` | Async HTTP client for all 5 Engram snapshot endpoints |
| `TokenizerClient` | `tokenizer.py` | Lazy-loaded HuggingFace tokenizer for token savings estimates |
| `EngramStreamWrapper` | `streaming.py` | Stream wrapper with restore-before and save-after semantics |

---

## How It Works

### The Problem

A typical multi-turn chat application sends the full message history on every turn:

```
Turn 1:  [system, user1]                              50 tokens
Turn 2:  [system, user1, asst1, user2]                120 tokens
Turn 3:  [system, user1, asst1, user2, asst2, user3]  210 tokens
...
Turn 20: [system, user1, ..., user20]                  5,000 tokens
```

Total tokens across 20 turns: approximately 25,000 tokens. Most of that is redundant re-processing.

### The Mechanism

The provider detects that each new request extends a known conversation:

```
Turn 1:  Process [system, user1] (50 tokens), save snapshot
Turn 2:  Restore snapshot (2ms), process [asst1, user2] (70 tokens), save
Turn 3:  Restore snapshot (2ms), process [asst2, user3] (90 tokens), save
...
Turn 20: Restore snapshot (2ms), process [asst19, user20] (250 tokens)
```

Total tokens processed: approximately 1,550 tokens, a 93.8% reduction.

### Prefix Detection Algorithm

1. Hash the stored message history for each known conversation
2. On each incoming request, strip 1 to 5 messages from the tail
3. Hash each stripped prefix and compare against known hashes
4. If a match is found, restore that snapshot and process only the delta messages
5. If no match, run a standard full-prefill completion

### Pseudo-ID Fingerprinting

When no `conversation_id` is provided, the provider generates a deterministic pseudo-ID from the first 2 messages (typically system prompt and first user message). This scopes prefix matching to prevent cross-conversation contamination. If a fingerprint collision is detected, the provider skips stateful optimization for that call and logs a warning.

---

## Streaming

The provider handles streaming with restore-before and save-after semantics:

1. Restore synchronously before the stream starts (2ms)
2. Proxy SSE chunks through to your application
3. Save asynchronously after the `[DONE]` chunk (fire-and-forget)

If your application cancels the stream mid-generation (`GeneratorExit`), the save is skipped. Partial-turn state is not saved.

The stream wrapper detects sync vs async context automatically:
- **Async context** (event loop running): Uses `asyncio.create_task()` for the save
- **Sync context** (no event loop): Spawns a daemon thread

---

## Error Handling

| Failure | Behavior |
|---------|----------|
| Restore HTTP error (4xx/5xx) | Log warning, fall back to full prefill, set `metadata.restore_failed = True` |
| Snapshot not found (404) | Same as above; fallback to full prefill |
| Save failure after completion | Log warning silently, do not surface to caller |
| Save failure after stream | Log warning silently, stream is already complete |
| Tokenizer load failure | Skip `tokens_saved` metadata, continue normally |
| Fingerprint collision | Skip prefix match, fall back to full prefill, log warning |

### Distributed Deployment

The `ConversationTracker` is process-local. In multi-worker deployments (gunicorn, uvicorn with multiple workers), each worker maintains independent state. Cross-worker requests will silently fall back to full prefill.

Recommendations for distributed deployments:
- Always provide an explicit `conversation_id` via `extra_body`
- Use `stateful_mode: "explicit"` for predictable behavior
- Future: configure `ENGRAM_TRACKER_REDIS_URL` for shared state (not yet implemented)

---

## Response Metadata

Every response includes Engram-specific metadata:

```python
response, metadata = await config.async_completion(...)

metadata.conversation_id   # str: conversation identifier used
metadata.turn_number       # int: current turn number
metadata.snapshot_id       # str: snapshot ID if auto-saved
metadata.tokens_saved      # int: estimated tokens not reprocessed
metadata.restore_time_ms   # float: snapshot restore latency in ms
metadata.restore_failed    # bool: True if restore fell back to full prefill
metadata.restore_error     # str: error message if restore failed
metadata.auto_save         # bool: whether auto-save was enabled
```

---

## Snapshot Management

```python
from engram_litellm import SnapshotClient

client = SnapshotClient(api_base="http://localhost:30000")

# Save
await client.save_snapshot("session-abc", turn_number=5)

# List all snapshots for a conversation
snapshots = await client.list_snapshots("session-abc")
for s in snapshots:
    print(f"Turn {s.turn_number}: {s.snapshot_id} ({s.size_bytes} bytes)")

# Get info on a specific snapshot
info = await client.get_snapshot_info("session-abc", turn_number=3)

# Restore
await client.restore_snapshot("session-abc", turn_number=3)

# Delete
await client.delete_snapshot("session-abc", turn_number=1)
```

---

## Development

### Running Tests

```bash
# Install test dependencies
pip install httpx pytest pytest-asyncio

# Run all tests
pytest tests/ -v
```

All 53 tests run without GPU.

### Test Coverage

Tests cover:
- Request/response transformation and provider integration
- Thread-safe conversation state tracking and collision detection
- Prefix matching algorithm and delta computation
- Streaming sync/async context, cancellation, and fire-and-forget save
- SnapshotClient CRUD operations and error handling
- Restore failure fallback to full prefill
- Fingerprint collision detection

---

## Ecosystem Context

Engram LiteLLM Provider is part of the Clarit.AI open-source ecosystem. Engram LiteLLM Provider focuses on drop-in stateful inference for existing LiteLLM applications, while Engram focuses on foundational Mamba serving with snapshot persistence, Synapse focuses on edge NPU inference, and Plexium focuses on repository knowledge layer integration.

---

## Contributing

Contributions are welcome. The provider follows the same structure as Engram and LiteLLM upstream projects.

Areas for contribution:
- Additional model architecture validation
- Distributed state tracking backends (Redis, etc.)
- Performance benchmarking and optimization
- Additional streaming patterns and error recovery

Start by opening an issue to discuss your idea, then submit a pull request with tests for your changes.

---

## License and Acknowledgements

Apache-2.0, same as Engram and LiteLLM.

Built on Engram by Clarit.AI, which extends SGLang with persistent Mamba state management. Designed for the LiteLLM ecosystem by BerriAI.

The Mamba architecture was developed by Albert Gu and Tri Dao. Mamba2 was developed by Tri Dao and Albert Gu at Carnegie Mellon University.

---

## Links

- Engram: https://github.com/Clarit-AI/Engram
- LiteLLM: https://github.com/BerriAI/litellm
- Engram API Guide: https://github.com/Clarit-AI/Engram/blob/main/docs/stateful_mamba/api_guide.md
- LiteLLM Provider Registration: https://docs.litellm.ai/docs/provider_registration/
- Clarit.AI: https://clarit.ai
