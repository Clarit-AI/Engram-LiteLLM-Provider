# Backwards Compatibility Mode for Stateful LLM Providers

> **Purpose:** This document explains how the Engram LiteLLM Provider enables *existing stateless applications* to benefit from snapshot-based stateful inference **without any application code changes**. It serves as a reference for engineers replicating this pattern across other providers and languages (TypeScript, Go, Rust, Python, etc.).

---

## Table of Contents

1. [The Core Problem](#1-the-core-problem)
2. [The Solution at a Glance](#2-the-solution-at-a-glance)
3. [Architecture Overview](#3-architecture-overview)
4. [The Compatibility Layer: How It Works](#4-the-compatibility-layer-how-it-works)
5. [Three Operating Modes](#5-three-operating-modes)
6. [Key Design Patterns](#6-key-design-patterns)
7. [Data Flow Walkthrough](#7-data-flow-walkthrough)
8. [Replicating This Pattern in Other Languages](#8-replicating-this-pattern-in-other-languages)
9. [Critical Implementation Details](#9-critical-implementation-details)
10. [Pitfalls and Edge Cases](#10-pitfalls-and-edge-cases)

---

## 1. The Core Problem

Stateless LLM applications send the **full message history** on every turn:

```
Turn 1:  [system, user_1]                                 ~50 tokens
Turn 2:  [system, user_1, assistant_1, user_2]            ~120 tokens
Turn 3:  [system, user_1, assistant_1, user_2, ...]       ~210 tokens
...
Turn 50: [system, user_1, ..., assistant_49, user_50]     ~5,000 tokens
```

Across 50 turns, approximately **25,000 tokens** are processed — the vast majority being redundant re-processing of messages the model has already seen.

The alternative — stateful inference with snapshot restore — requires applications to:
- Manage snapshot IDs and turn numbers
- Explicitly call restore endpoints before generation
- Track conversation state across requests
- Refactor their entire message-sending pattern

**This is the barrier the backwards compatibility mode eliminates.**

---

## 2. The Solution at a Glance

The provider sits between the application and the inference server as a **transparent proxy** that:

1. **Intercepts** every request the application makes (no code changes needed — just swap the model prefix).
2. **Detects** redundant context by comparing incoming messages against stored conversation state.
3. **Restores** a saved model state snapshot (~2ms) so only *new* tokens need processing.
4. **Saves** a new snapshot after the response completes.

The application continues sending full message history exactly as before. The provider handles the optimization silently.

**Result:** 93.8% token reduction on a 50-turn conversation (25,000 → 1,550 tokens), zero application code changes.

---

## 3. Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                     Your Application                          │
│        (stateless, sends full message history every turn)     │
└────────────────────────┬─────────────────────────────────────┘
                         │  Full messages, unchanged
                         ▼
┌──────────────────────────────────────────────────────────────┐
│              Backwards Compatibility Provider                 │
│                                                               │
│  ┌─────────────────┐  ┌──────────────────┐  ┌──────────────┐│
│  │ Conversation    │  │ Context Differ   │  │ Snapshot     ││
│  │ Tracker         │  │ (Prefix Match)   │  │ Client       ││
│  │                 │  │                  │  │              ││
│  │ • Fingerprint   │  │ • Strip 1-5 from │  │ • save       ││
│  │   messages[:2]  │  │   tail, hash     │  │ • restore    ││
│  │ • SHA-256 hash  │  │ • Compare vs     │  │ • list       ││
│  │ • In-memory     │  │   stored state   │  │ • get_info   ││
│  │   dictionary    │  │ • Return delta   │  │ • delete     ││
│  └────────┬────────┘  └────────┬─────────┘  └──────┬───────┘│
│           │                    │                     │        │
│           ▼                    ▼                     ▼        │
│  ┌──────────────────────────────────────────────────────────┐│
│  │           Request Transformation Pipeline                ││
│  │                                                          ││
│  │  1. Extract stateful params from extra_body              ││
│  │  2. Resolve conversation_id (user-provided > fingerprint ││
│  │     > UUID)                                               ││
│  │  3. Find prefix match → identify new messages             ││
│  │  4. Store state in side-channel (request_id keyed)        ││
│  │  5. Build standard OpenAI-compatible request body         ││
│  └────────────────────────┬─────────────────────────────────┘│
│                           │                                   │
│                           ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐│
│  │           async_completion / async_streaming             ││
│  │                                                          ││
│  │  1. Retrieve state from side-channel via request_id      ││
│  │  2. If prefix matched → restore snapshot (~2ms)          ││
│  │  3. On restore failure → fall back to full prefill        ││
│  │  4. Send request to inference server                      ││
│  │  5. Record turn + auto-save snapshot after response       ││
│  └────────────────────────┬─────────────────────────────────┘│
└───────────────────────────┼─────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
     ┌─────────────────┐        ┌──────────────────┐
     │ restore_snapshot│        │ chat/completions │
     │     (~2ms)      │        │   (generation)   │
     └────────┬────────┘        └────────┬─────────┘
              │                          │
              └─────────────┬────────────┘
                            ▼
                   ┌─────────────────┐
                   │  save_snapshot  │
                   │  (auto/manual)  │
                   └────────┬────────┘
                            ▼
┌───────────────────────────────────────────────────────────────┐
│                    Inference Server                            │
│         (SGLang + Mamba + Snapshot Persistence)               │
└───────────────────────────────────────────────────────────────┘
```

### Component Summary

| Component | Responsibility |
|-----------|---------------|
| **ConversationTracker** | Thread-safe, process-local dictionary mapping conversation IDs to their state (turn number, message hash, recent messages). Generates deterministic pseudo-IDs from the first 2 messages when no explicit `conversation_id` is provided. |
| **ContextDiffer** | Prefix matching algorithm. Strips 1 to 5 messages from the tail of incoming messages, hashes each stripped prefix, and compares against stored hashes to find the longest matching prefix. |
| **SnapshotClient** | Async HTTP client for the 5 snapshot endpoints: `save_snapshot`, `restore_snapshot`, `list_snapshots`, `get_snapshot_info`, `delete_snapshot`. |
| **TokenizerClient** | Lazy-loaded tokenizer for estimating token savings. Falls back to character-based approximation (~4 chars/token) if unavailable. |
| **StreamWrapper** | Wraps streaming responses with fire-and-forget async save after completion. Detects sync vs async context. Skips save on mid-stream cancellation. |
| **Request Transformer** | The main orchestration layer. Extracts params, resolves conversation IDs, performs prefix detection, stores state in a side-channel, and builds the outgoing request. |

---

## 4. The Compatibility Layer: How It Works

### 4.1 Transparent Interception

The provider implements the **LiteLLM provider interface**, which defines specific hooks:

```
validate_environment()  → Set up auth headers
get_complete_url()      → Route to correct endpoint
transform_request()     → Intercept and modify outgoing requests
transform_response()    → Attach metadata to responses
async_completion()      → Override with restore-before-generate logic
async_streaming()       → Override with restore-before-stream logic
```

**Key insight:** The application calls the provider exactly like any other LiteLLM provider. The provider intercepts at the transformation layer, injects stateful behavior, and returns a response that looks identical to what the application expects — plus metadata.

### 4.2 The Side-Channel Pattern

One of the most critical design decisions: **state is passed between `transform_request()` and `async_completion()` via an in-memory side-channel, NOT via the HTTP request body.**

```python
# In transform_request():
request_id = str(uuid.uuid4())
self._pending_state[request_id] = {
    "conversation_id": conv_id,
    "turn_number": current_turn,
    "auto_save": auto_save,
    "_prefix_match": match,           # The prefix match result
    "_restore_target": "conv:turn",   # What to restore
    "_messages": messages,            # Original messages
    "_created_at": time.time(),
}
request["_engram_request_id"] = request_id  # Only this goes in the body

# In async_completion():
request_id = request_data.pop("_engram_request_id")
state = self._pending_state.pop(request_id)  # Retrieved, not deserialized
```

**Why this matters:**
- Internal state (prefix match data, message arrays, match objects) never gets serialized into JSON.
- No risk of leaking implementation details into the HTTP payload.
- The application's request body remains clean and OpenAI-compatible.
- TTL-based cleanup (300 seconds) prevents memory leaks from orphaned entries.

**Replication note:** In languages without a shared in-memory dictionary (e.g., distributed Go services), this pattern requires a local cache layer (e.g., `sync.Map` in Go, `Map` in TypeScript with cleanup timers) scoped to the process handling the request.

### 4.3 Conversation ID Resolution Priority

When the application doesn't provide a `conversation_id`, the provider resolves one deterministically:

```
1. User-provided conversation_id (via extra_body) → use as-is
2. ≥2 messages available → SHA-256 hash of messages[:2] → "auto-{hex[:16]}"
3. Fallback → generate random UUID
```

The **pseudo-ID fingerprint** (option 2) is the key to zero-config backwards compatibility. It ensures that the same application sending the same system prompt and first user message will always get the same conversation ID, enabling automatic prefix detection without any configuration.

### 4.4 Collision Detection

When a pseudo-ID fingerprint matches an existing conversation, the provider verifies it's a **continuation** (prefix match), not a **collision** (different conversation that happens to share the same first 2 messages):

```python
def _is_prefix_or_match(incoming, stored):
    stored_msgs = stored.last_messages
    if len(incoming) < len(stored_msgs):
        return False  # Can't be a continuation
    # Verify stored messages are a prefix of incoming
    for i, msg in enumerate(stored_msgs):
        if _hash_message(incoming[i]) != _hash_message(msg):
            return False  # Different conversation
    return True
```

If a collision is detected (same first 2 messages, but different subsequent messages), the provider **skips the stateful optimization** for that call and falls back to full prefill. This prevents cross-conversation contamination.

---

## 5. Three Operating Modes

### 5.1 Auto Mode (Default — Zero Config)

The provider handles everything automatically. The application code remains completely unchanged.

```
Application sends:  [system, user_1, assistant_1, user_2]  (full history)
Provider intercepts:
  1. Fingerprint → "auto-a1b2c3d4..."
  2. Prefix match → stored state found for turn 1
  3. Restore snapshot for turn 1 (~2ms)
  4. Send only [assistant_1, user_2] to server
  5. Receive response
  6. Auto-save snapshot for turn 2
Application receives: Normal response + metadata
```

**What the application sees:** Identical API. Just faster and cheaper.

### 5.2 Stateless Mode (Opt-Out)

Standard OpenAI-compatible pass-through. No state tracking, no snapshots.

```python
optional_params = {"extra_body": {"stateful_mode": "stateless"}}
```

Useful for testing, debugging, or bypassing the stateful system for specific calls.

### 5.3 Explicit Mode (Opt-In Control)

The application provides `conversation_id` and `restore_from` explicitly. The provider does not auto-detect prefixes — it only restores when told to.

```python
optional_params = {"extra_body": {
    "stateful_mode": "explicit",
    "conversation_id": "my-session",
    "restore_from": "my-session:5",
}}
```

For applications that want to manage their own snapshot lifecycle while still benefiting from the provider's restore/save infrastructure.

---

## 6. Key Design Patterns

### 6.1 Prefix Detection with Tail Stripping

The `ContextDiffer` does not compare messages one-by-one sequentially. Instead, it uses a **hash-and-compare** approach:

```
Incoming messages: [sys, u1, a1, u2, a2, u3]  (6 messages)

Strip 1 from tail: [sys, u1, a1, u2, a2] → hash → compare vs stored
Strip 2 from tail: [sys, u1, a1, u2]      → hash → compare vs stored
Strip 3 from tail: [sys, u1, a1]          → hash → compare vs stored
...
Strip N from tail: until MAX_LOOKAHEAD (5) or prefix is empty
```

The first hash match identifies the longest matching prefix. The delta (messages after the prefix) is what gets sent for generation.

**Why hash-based?** 
- O(1) comparison per strip level (vs O(N) for sequential message comparison).
- Canonical JSON serialization (sorted keys, ASCII-only) ensures deterministic hashes.
- No need to transmit or store full message bodies for comparison — just hashes.

### 6.2 Graceful Degradation

Every failure mode falls back to full prefill:

| Failure | Fallback |
|---------|----------|
| Restore HTTP error (4xx/5xx) | Full message prefill, `metadata.restore_failed = True` |
| Snapshot not found (404) | Full message prefill |
| Fingerprint collision | Full message prefill, warning logged |
| Tracker has no state for this ID | Full message prefill (normal first-turn behavior) |
| Tokenizer load failure | Continue without `tokens_saved` estimate |
| Save failure | Warning logged, response already delivered |

**The application never sees an error from a failed optimization.** It always gets a valid response — just without the token savings for that turn.

### 6.3 Fire-and-Forget Async Save

After a response is delivered, the snapshot save happens asynchronously and independently:

**Streaming:**
```
1. Restore synchronously before stream starts (~2ms)
2. Proxy SSE chunks to application
3. On [DONE]: fire save task (non-blocking)
4. If stream cancelled mid-way (GeneratorExit): skip save
```

**Non-streaming:**
```
1. Restore synchronously before generation
2. Generate response
3. Record turn, then save snapshot synchronously
```

The stream wrapper detects the execution context:
- **Async** (event loop running): `asyncio.create_task(save_and_log())`
- **Sync** (no event loop): `threading.Thread(target=asyncio.run, daemon=True)`

### 6.4 Thread-Safe Process-Local State

The `ConversationTracker` uses:
- A `threading.Lock()` for thread safety within a process.
- An in-memory `Dict[str, ConversationState]` for storage.
- Worker detection (`GUNICORN_WORKERS`, `WEB_CONCURRENCY`) to warn about multi-worker limitations.

**Limitation:** State is not shared across processes or machines. Cross-worker requests fall back to full prefill silently. For distributed deployments, users are advised to provide explicit `conversation_id` and use `stateful_mode: "explicit"`.

---

## 7. Data Flow Walkthrough

Here's the complete journey of a single request in **auto mode**:

```
┌─ APPLICATION ─────────────────────────────────────────────────────────┐
│                                                                       │
│  messages = [system, user_1, assistant_1, user_2]                     │
│                                                                       │
│  response, metadata = await config.async_completion(                  │
│      model="engram/granite-4.0-h-tiny",                               │
│      messages=messages,                                               │
│      api_base="http://localhost:30000",                               │
│  )                                                                    │
│                                                                       │
└─┬─────────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─ TRANSFORM_REQUEST ───────────────────────────────────────────────────┐
│                                                                       │
│  1. Extract stateful params from optional_params/extra_body            │
│     → conversation_id, restore_from, auto_save, stateful_mode          │
│                                                                       │
│  2. Resolve conversation_id:                                           │
│     → Not provided → fingerprint(messages[:2]) → "auto-a1b2c3d4"      │
│                                                                       │
│  3. Check for collision:                                               │
│     → "auto-a1b2c3d4" not in tracker → no collision                   │
│                                                                       │
│  4. Auto mode → find prefix match:                                     │
│     → Stored: turn 1, messages = [system, user_1]                     │
│     → Strip 2 from tail: [system, user_1] → hash MATCH                 │
│     → PrefixMatch: turn_number=1, new_messages=[assistant_1, user_2]  │
│       tokens_saved=50                                                  │
│                                                                       │
│  5. Store in side-channel:                                             │
│     _pending_state["uuid-xyz"] = {                                     │
│         "conversation_id": "auto-a1b2c3d4",                           │
│         "turn_number": 2,                                             │
│         "_prefix_match": PrefixMatch(...),                            │
│         "_restore_target": "auto-a1b2c3d4:1",                         │
│         "_messages": [...original messages...],                       │
│     }                                                                  │
│                                                                       │
│  6. Build request body:                                                │
│     { "model": "granite-4.0-h-tiny", "messages": [...],               │
│       "_engram_request_id": "uuid-xyz" }                              │
│                                                                       │
└─┬─────────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─ ASYNC_COMPLETION ────────────────────────────────────────────────────┐
│                                                                       │
│  1. Retrieve state: _pending_state.pop("uuid-xyz")                     │
│                                                                       │
│  2. Restore target exists → call restore_snapshot(                     │
│         conversation_id="auto-a1b2c3d4", turn_number=1)               │
│     → ~2ms, server loads model state from disk/memory                  │
│                                                                       │
│  3. POST to /v1/chat/completions with full messages                    │
│     (server uses restored state to skip prefix tokens)                 │
│                                                                       │
│  4. Receive response                                                   │
│                                                                       │
│  5. Record turn: tracker.record_turn("auto-a1b2c3d4", messages, 2)    │
│                                                                       │
│  6. Auto-save: save_snapshot(conversation_id="auto-a1b2c3d4", turn=2) │
│                                                                       │
│  7. Attach metadata to response:                                       │
│     metadata.tokens_saved = 50                                        │
│     metadata.restore_time_ms = 2.1                                    │
│     metadata.turn_number = 2                                          │
│     metadata.snapshot_id = "snap-xyz"                                 │
│                                                                       │
└─┬─────────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─ APPLICATION ─────────────────────────────────────────────────────────┐
│                                                                       │
│  Receives: response.choices[0].message.content = "..."                │
│  Plus:    metadata.tokens_saved, metadata.restore_time_ms, etc.       │
│                                                                       │
│  Application code: ZERO CHANGES                                       │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 8. Replicating This Pattern in Other Languages

### 8.1 Universal Requirements

Any implementation of this backwards compatibility layer needs:

1. **Provider/Adapter Interface** — A way to intercept requests before they reach the inference server and responses before they reach the application.
2. **Conversation State Storage** — Process-local storage for conversation metadata (ID → {turn, hash, messages}).
3. **HTTP Client** — For calling snapshot endpoints (save, restore, list, info, delete).
4. **Hashing** — SHA-256 (or equivalent) for canonical message hashing.
5. **Async Runtime** — For non-blocking snapshot operations.

### 8.2 TypeScript / Node.js

```typescript
// Key patterns to replicate:

// 1. Side-channel (use a Map with TTL cleanup)
class SideChannel {
  private state = new Map<string, StateEntry>();
  
  set(requestId: string, state: StateEntry) {
    this.state.set(requestId, { ...state, createdAt: Date.now() });
    this.scheduleCleanup();
  }
  
  get(requestId: string): StateEntry | undefined {
    return this.state.get(requestId);
  }
  
  private scheduleCleanup() {
    // setInterval or setTimeout to prune entries > 300s old
  }
}

// 2. ConversationTracker (use Map, process-local)
class ConversationTracker {
  private state = new Map<string, ConversationState>();
  
  getOrCreate(messages: Message[], userId?: string): [string, boolean] {
    if (userId) return [userId, false];
    if (messages.length >= 2) {
      const fingerprint = this.fingerprint(messages.slice(0, 2));
      // Check collision, prefix match...
    }
    return [crypto.randomUUID(), false];
  }
}

// 3. ContextDiffer (same algorithm, different syntax)
class ContextDiffer {
  static findPrefixMatch(incoming: Message[], stored: ConversationState) {
    const maxStrip = Math.min(MAX_LOOKAHEAD, incoming.length - 1);
    for (let n = 1; n <= maxStrip; n++) {
      const prefix = incoming.slice(0, -n);
      if (hashMessages(prefix) === stored.messagesHash) {
        return { newMessages: incoming.slice(-n), tokensSaved: ... };
      }
    }
    return null;
  }
}

// 4. Interception point (depends on framework)
// For OpenAI SDK: wrap the chat.completions.create method
// For LiteLLM TypeScript: implement the provider transform interface
```

### 8.3 Go

```go
// Key patterns to replicate:

// 1. Side-channel (use sync.Map with periodic cleanup)
type SideChannel struct {
    mu    sync.RWMutex
    state map[string]*StateEntry
}

func (sc *SideChannel) Set(requestID string, state *StateEntry) {
    sc.mu.Lock()
    sc.state[requestID] = &StateEntry{
        State:     state,
        CreatedAt: time.Now(),
    }
    sc.mu.Unlock()
}

func (sc *SideChannel) Get(requestID string) *StateEntry {
    sc.mu.RLock()
    defer sc.mu.RUnlock()
    entry := sc.state[requestID]
    if time.Since(entry.CreatedAt) > 300*time.Second {
        delete(sc.state, requestID)
        return nil
    }
    return entry.State
}

// 2. ConversationTracker (use sync.Map for thread safety)
type ConversationTracker struct {
    mu    sync.RWMutex
    state map[string]*ConversationState
}

// 3. ContextDiffer (same hash-and-compare algorithm)
func FindPrefixMatch(incoming []Message, stored *ConversationState) *PrefixMatch {
    maxStrip := min(5, len(incoming)-1)
    for n := 1; n <= maxStrip; n++ {
        prefix := incoming[:len(incoming)-n]
        if hashMessages(prefix) == stored.MessagesHash {
            return &PrefixMatch{
                NewMessages: incoming[len(incoming)-n:],
                TokensSaved: estimateTokens(prefix),
            }
        }
    }
    return nil
}

// 4. HTTP middleware pattern for interception
// Use http.RoundTripper or middleware to intercept before/after
// the actual inference server call
```

### 8.4 Rust

```rust
// Key patterns to replicate:

// 1. Side-channel (use DashMap or tokio::sync::RwLock<HashMap>)
use dashmap::DashMap;
use std::time::{Duration, Instant};

struct SideChannel {
    state: DashMap<String, StateEntry>,
}

struct StateEntry {
    data: StateData,
    created_at: Instant,
    ttl: Duration,
}

impl SideChannel {
    fn set(&self, request_id: String, state: StateData) {
        self.state.insert(request_id, StateEntry {
            data: state,
            created_at: Instant::now(),
            ttl: Duration::from_secs(300),
        });
    }
    
    fn get(&self, request_id: &str) -> Option<StateData> {
        self.state.get(request_id)
            .filter(|e| e.created_at.elapsed() < e.ttl)
            .map(|e| e.data.clone())
    }
}

// 2. ConversationTracker (use Arc<RwLock<HashMap>> or DashMap)
// 3. ContextDiffer (same algorithm, use sha2 crate for hashing)
// 4. Tower middleware or hyper service for interception
```

### 8.5 Language-Agnostonic Implementation Checklist

| Component | What to Build | Critical Details |
|-----------|--------------|------------------|
| **Provider Interface** | Interceptor/wrapper around the LLM client's chat completion method | Must hook both request transformation and response transformation |
| **ConversationTracker** | In-memory map: `conversation_id → {turn_number, messages_hash, last_messages}` | Thread-safe; detect multi-worker deployments; warn about distributed limitations |
| **Pseudo-ID Generator** | Deterministic hash of `messages[:2]` | Use SHA-256, truncate to 16 hex chars, prefix with `"auto-"` |
| **ContextDiffer** | Strip 1-5 messages from tail, hash each prefix, compare | Canonical JSON serialization (sorted keys, consistent encoding) |
| **Side-Channel** | In-memory map: `request_id → state_dict` with TTL cleanup | Never serialize internal state into HTTP body; clean up orphaned entries |
| **SnapshotClient** | HTTP client with 5 methods: save, restore, list, info, delete | Async/non-blocking where possible; reuse HTTP connections |
| **StreamWrapper** | Wrapper around streaming responses with post-completion save | Detect sync vs async context; skip save on cancellation |
| **TokenizerClient** | Token counter for `tokens_saved` metadata | Can be approximate (chars/4); real tokenizer optional |
| **Error Handling** | All failures fall back to full prefill | Never surface optimization errors to the application |

---

## 9. Critical Implementation Details

### 9.1 Canonical JSON for Hashing

Message hashing **must** be deterministic. Use canonical JSON serialization:

```python
canonical = json.dumps(messages, sort_keys=True, ensure_ascii=True)
hash = hashlib.sha256(canonical.encode()).hexdigest()
```

In other languages:
- **TypeScript:** `JSON.stringify(messages)` — ensure consistent key ordering (sort keys before stringify)
- **Go:** `json.Marshal()` with sorted struct keys, or use a canonical JSON library
- **Rust:** `serde_json` with `preserve_order` disabled and keys sorted

### 9.2 Token Estimation

The `tokens_saved` metric is an **estimate** for response metadata only. It does not affect the actual request:

```
Approximation: total_chars // 4 (with minimum of 1)
Better:        Use HuggingFace tokenizer (Python), tiktoken (TypeScript/Go/Rust)
```

The actual token savings happen server-side during restore-and-generate. This client-side estimate is purely for observability.

### 9.3 Request Body Hygiene

Only `request_id` is injected into the request body. All stateful data flows through the side-channel:

```python
# CORRECT: Only the request_id travels in the HTTP body
request["_engram_request_id"] = request_id

# INCORRECT: Never do this — it pollutes the request
request["_prefix_match"] = match.to_dict()
request["_stored_messages"] = messages
```

### 9.4 MAX_LOOKAHEAD_TURNS = 5

The prefix detection algorithm only strips up to 5 messages from the tail. This is a performance/coverage tradeoff:

- **Too few:** Misses valid prefix matches when applications add 3+ new messages between calls.
- **Too many:** O(N) hash computations per request, diminishing returns.

5 covers the common pattern: `[new_assistant_response, new_user_message]` (2 messages), with buffer for tool calls and multi-message turns.

### 9.5 State TTL = 300 Seconds

The side-channel entries expire after 300 seconds. This is generous enough for slow inference calls but prevents unbounded memory growth. Implement periodic cleanup:

```python
def _cleanup_orphaned_state(self):
    now = time.time()
    orphaned = [rid for rid, s in self._pending_state.items()
                if now - s.get("_created_at", 0) > 300]
    for rid in orphaned:
        del self._pending_state[rid]
```

### 9.6 Environment Variable Configuration

All operational parameters should be configurable via environment variables with sensible defaults:

```
ENGRAM_BASE_URL          →  http://localhost:30000
ENGRAM_AUTO_SAVE         →  true
ENGRAM_STATEFUL_MODE     →  auto
ENGRAM_TOKENIZER_PATH    →  (none, lazy load)
ENGRAM_TRACKER_REDIS_URL →  (none, future: distributed state)
```

---

## 10. Pitfalls and Edge Cases

### 10.1 Cross-Worker State Loss

**Problem:** In multi-process deployments (gunicorn with 4 workers), Worker A handles turn 1, Worker B handles turn 2. Worker B has no state for this conversation ID → silent fallback to full prefill.

**Mitigation:**
- Warn users when multiple workers are detected.
- Recommend explicit `conversation_id` + `stateful_mode: "explicit"` for distributed deployments.
- Future: Redis/shared cache backend for the tracker.

### 10.2 Mid-Stream Cancellation

**Problem:** The application cancels a streaming request mid-generation. If the provider saves the partial turn, the next request will restore an incomplete state.

**Solution:** Detect `GeneratorExit` in the stream wrapper and skip the save:

```python
async def __aiter__(self):
    try:
        async for chunk in self._stream:
            yield chunk
    except GeneratorExit:
        self._cancelled = True
        raise
    finally:
        if not self._cancelled:
            self._fire_save()  # Only save if stream completed normally
```

### 10.3 Fingerprint Collision on Different Conversations

**Problem:** Two different users send the same system prompt and first message. The pseudo-ID fingerprint is identical.

**Solution:** The `_is_prefix_or_match` check verifies that the stored messages are actually a prefix of the incoming messages. If the third message differs, it's treated as a collision, and the stateful optimization is skipped with a warning.

**For production:** Always use explicit `conversation_id` per user session to avoid any collision risk.

### 10.4 Snapshot Save Failure After Successful Generation

**Problem:** The response is delivered to the application, but the snapshot save fails (network error, disk full, etc.).

**Solution:** Log a warning silently. The next request will simply re-process the full prefix (no snapshot to restore). No data is lost — the conversation continues, just without the optimization benefit until a successful save.

### 10.5 Mode Switching Mid-Conversation

**Problem:** Turn 1 uses `auto` mode, turn 2 uses `stateless` mode. The tracker still records turn 1's state, but turn 2 doesn't use it.

**Behavior:** This is allowed and handled gracefully. Turn 2 processes full messages (stateless), but the tracker still has turn 1's state. Turn 3 (back to `auto`) can still restore from turn 1's snapshot.

### 10.6 Message Order Sensitivity

**Problem:** The application sends messages in a slightly different order or format between turns (e.g., adding whitespace, reordering content blocks).

**Impact:** The hash-based comparison will detect this as a different prefix → full prefill fallback. The conversation continues normally, just without optimization.

**Mitigation:** Ensure canonical message formatting in the hashing layer (sorted keys, normalized whitespace).

---

## Summary

The backwards compatibility mode works by **intercepting stateless requests, detecting redundant context through prefix matching, restoring cached model state, and only processing the delta** — all transparently to the application.

The key architectural decisions that make this possible:

1. **Side-channel state passing** — Internal state never pollutes the HTTP request body.
2. **Deterministic pseudo-ID fingerprinting** — Enables zero-config conversation tracking.
3. **Hash-based prefix detection** — Efficient O(1) comparison per strip level.
4. **Graceful degradation** — Every failure falls back to full prefill; the application never sees an error.
5. **Three operating modes** — Auto (zero config), stateless (opt-out), and explicit (opt-in control) cover all use cases.

To replicate this pattern in any language:
- Implement the provider/adapter interface for your framework.
- Build process-local conversation tracking with thread-safe storage.
- Implement hash-based prefix detection with tail stripping.
- Create an HTTP client for snapshot endpoints.
- Wire everything together with a side-channel for inter-stage communication.
- Ensure all failures degrade gracefully to standard stateless behavior.
