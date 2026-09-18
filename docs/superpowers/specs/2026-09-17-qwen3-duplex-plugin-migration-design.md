# Qwen3-Omni Duplex Plugin Migration

## Context

The Qwen3-Omni duplex branch predates the upstream refactor that moved duplex
session ownership into `DuplexSessionRunner` and combined the old engine
runtime extension and serving runtime adapter into one `DuplexModelPlugin`.
The merge currently has six unresolved paths. Qwen3 still points its pipeline
at the removed `duplex_serving_adapter` contract and its existing serving path
uses a generic chat-completion fallback.

Qwen3-Omni does not expose the native incremental listen/speak data plane used
by MiniCPM-o. Its supported behavior is turn-based: the engine collects a
committed audio turn, the ordinary Qwen3 chat pipeline produces a streamed text
or audio response, and the Realtime transport projects that response.

## Goals

- Register Qwen3-Omni through the current `DuplexModelPlugin` contract.
- Preserve the Qwen3 turn-policy prompts, private runtime configuration, input
  buffering, and chat-completion fallback behavior.
- Keep the canonical conversation, response, playback, epoch, cancellation,
  and resume state in the engine-resident session.
- Preserve upstream Realtime, Server VAD, backpressure, and stale-output
  behavior.
- Resolve the merge without restoring the deleted legacy full-duplex runtime.

## Non-goals

- Do not implement a native Qwen3 incremental data plane.
- Do not maintain a second serving-side conversation or audio ledger.
- Do not change Qwen3 model weights, stage topology, or ordinary chat request
  semantics beyond what is required for the fallback bridge.
- Do not change capabilities that Qwen3 does not support, including multi-session
  native state or session resume, unless an existing upstream contract requires
  the value to be reported explicitly.

## Architecture

### Qwen3 model plugin

Add a Qwen3 plugin in the model-owned duplex package that implements every
abstract method of `DuplexModelPlugin`:

- `configure_sampling_params`, `plan_append`, and `decide_output` provide the
  engine policy surface required by the new orchestrator contract.
- The plugin-owned session state implements the current
  `DuplexModelSessionState` interface and retains Qwen3 interruption state.
- The plugin-owned PCM buffer retains committed input until the engine has
  emitted the commit and started the fallback request.
- Runtime configuration validation and updates retain the existing private
  `auto_commit_silence_ms` rule.
- Capabilities report turn-commit-only operation, no native input append, no
  native model turn policy, and chat completions enabled.
- The data-plane object remains a contract implementation that rejects native
  projection if called. The live Qwen3 path never calls it.

The pipeline changes from `duplex_serving_adapter` to `duplex_plugin`. The old
adapter, old runtime bridge, and old serving session objects are not used by
the Qwen3 pipeline after migration.

### Engine/API fallback bridge

The engine and API process communicate through an internal, non-Realtime
fallback request/output channel:

1. The engine runner receives and normalizes audio using the upstream input and
   VAD flow.
2. On a valid committed Qwen3 turn, the runner commits the input into the
   engine session history, creates the response lifecycle state, and emits an
   internal fallback request containing the session id, current epoch/fence,
   response configuration, and a snapshot of canonical history.
3. The API-side duplex handler receives that internal request and invokes the
   existing `OmniOpenAIServingChat` service. This keeps ordinary chat generation
   off the orchestrator loop and avoids submitting a chat request from the
   engine loop that owns the same stage resources.
4. The handler parses streamed chat chunks and sends model-neutral fallback
   output messages back to the engine. The projection logic is extracted from
   `ChatFallbackProjectorMixin`; it does not mutate session state directly.
5. The engine applies each fallback output through `ModelChannel` and
   `SessionEmitter`, so the engine remains the only owner of response ids,
   Realtime event ordering, assistant history, playback accounting, metrics,
   and terminal response state.

Fallback request/output messages are internal engine messages and are never
serialized as client-visible Realtime events. Only the normal typed response,
error, cancellation, and session events leave the engine.

### Cancellation and stale output

Every fallback request is bound to the session epoch and response id. A
barge-in, response cancel, output clear, close, or expiry advances or invalidates
that identity before cancelling the API-side fallback task. The engine drops
any output whose identity no longer matches the active response. The API-side
task also stops consuming the chat stream after cancellation. The existing
failed-response sequence is used for provider errors:

```text
error -> response.done(status=failed, committed=false)
```

Successful fallback streams finish through the engine's normal response output
and response-done path, including history commit rules and playback metadata.

## Merge Resolution

The upstream engine-resident implementation is the base for:

- `realtime_input.py` and `serving.py` transport behavior;
- `runtime_bridge.py` native-runtime eligibility and control behavior;
- `session_runner.py` Server VAD, append ordering, backpressure, and fences;
- typed engine commands, events, and session state.

Qwen-specific behavior is reintroduced only through the plugin and the generic
fallback bridge. Existing Qwen regression tests are updated to assert the new
plugin path rather than the removed adapter path. No conflict resolution keeps
two parallel session implementations.

## Error Handling

- Invalid client configuration is rejected by the plugin before a session is
  admitted.
- Unsupported native data-plane use fails closed with an internal runtime
  error; it is not silently treated as a successful response.
- `ErrorResponse`, malformed fallback chunks, unsupported response modalities,
  and provider exceptions use the existing Realtime error codes and failed
  response lifecycle.
- Fallback output from a closed, cancelled, or stale epoch is discarded without
  producing client-visible output.

## Testing

Tests are written before production changes where the behavior is new:

- plugin loading, abstract-contract compliance, capabilities, runtime config,
  Qwen policy state, and PCM buffer behavior;
- fallback request identity and history snapshot construction;
- text/audio/error chunk projection without direct session mutation;
- runner response lifecycle, cancellation, stale output, and close behavior;
- pipeline registration through `duplex_plugin` and absence of the legacy
  adapter path;
- upstream duplex handler, Server VAD, input-format, and session tests.

The local environment currently lacks `torch`, so tests requiring the full vLLM
import graph may be unavailable. Static checks, bytecode compilation, focused
pure-Python tests, and the exact dependency failure are recorded separately.

## Acceptance Criteria

- `git diff --name-only --diff-filter=U` produces no output.
- No conflict markers remain under `vllm_omni` or `tests`.
- Qwen3's pipeline resolves a concrete `DuplexModelPlugin`.
- A committed Qwen3 turn reaches the ordinary chat service through the internal
  fallback bridge and returns text/audio through engine-owned Realtime events.
- Barge-in and cancellation prevent stale fallback output from reaching the
  client.
- Ruff, formatting, static checks, and all runnable focused tests pass; missing
  runtime dependencies are reported exactly.
- The resulting history is a normal merge commit with no squash or rebase.
