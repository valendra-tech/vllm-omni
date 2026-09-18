# Qwen3-Omni Duplex Plugin Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate Qwen3-Omni from the deleted serving-adapter contract to the engine-resident `DuplexModelPlugin` framework while preserving its turn-based chat-completion fallback and resolving the current upstream merge.

**Architecture:** Upstream owns the duplex session, Realtime projection, fences, cancellation, and response history. A Qwen3 plugin owns Qwen-specific policy and committed-input buffering. An internal fallback request/output bridge lets the API process call `OmniOpenAIServingChat` without running chat generation on the orchestrator loop; the engine applies projected output through `ModelChannel`.

**Tech Stack:** Python 3.11+, dataclasses, asyncio, vLLM-Omni duplex engine, OpenAI Realtime protocol, pytest, Ruff, Git merge, GitHub CLI.

---

## Implementation Constraints

- Work only in `/Users/jose/code/valendra/vllm-omni/.worktrees/VAL-351-qwen3-omni-conflict-resolution-v3`.
- The worktree is already in a non-fast-forward merge from `upstream/main`; do not reset, abort, squash, rebase, or discard unrelated staged upstream changes.
- Do not create child commits while the index contains unmerged paths. Stage the design, implementation, tests, and resolved merge paths together in the final normal merge commit.
- Use upstream's `realtime_input.py`, `serving.py`, `runtime_bridge.py`, and Server VAD behavior as the base. Qwen-specific behavior must use the new plugin/fallback seams.
- Do not import `vllm_omni.entrypoints.duplex.protocol`, `runtime_adapter`, `runtime_bridge`, or the deleted legacy full-duplex runtime from the Qwen model plugin.
- The local environment does not provide `torch`; record the exact import/test failure and do not claim unavailable tests passed.

## File Map

### New files

- `vllm_omni/engine/duplex/fallback.py`: immutable engine/API fallback request and output DTOs.
- `vllm_omni/model_executor/models/qwen3_omni/duplex/__init__.py`: Qwen3 duplex exports.
- `vllm_omni/model_executor/models/qwen3_omni/duplex/data_plane.py`: fail-closed compatibility data plane.
- `vllm_omni/model_executor/models/qwen3_omni/duplex/input.py`: committed-turn PCM buffer and reservations.
- `vllm_omni/model_executor/models/qwen3_omni/duplex/plugin.py`: Qwen3 `DuplexModelPlugin` implementation.
- `vllm_omni/model_executor/models/qwen3_omni/duplex/policy.py`: Qwen3 turn and interruption prompts.
- `vllm_omni/model_executor/models/qwen3_omni/duplex/session.py`: Qwen3 `DuplexModelSessionState`.
- `tests/engine/duplex/test_duplex_fallback.py`: fallback DTO/message behavior.
- `tests/engine/duplex/test_qwen3_plugin.py`: Qwen3 plugin, state, buffer, and policy behavior.
- `tests/engine/duplex/test_qwen3_fallback_runner.py`: engine fallback lifecycle and cancellation.
- `tests/entrypoints/duplex/test_chat_fallback.py`: API-side request construction and pure chunk projection.

### Modified files

- `vllm_omni/engine/duplex/config.py`: expose fallback capability and configurable implementation/input mode metadata.
- `vllm_omni/engine/duplex/plugin.py`: add optional fallback/default-auto-response/barge-in hooks.
- `vllm_omni/engine/duplex/messages.py`: add internal fallback request, started, output, failed, and cancel messages.
- `vllm_omni/engine/duplex_omni_engine.py`: submit internal fallback messages from the API process.
- `vllm_omni/engine/duplex/session/context.py`: track the active fallback request identity.
- `vllm_omni/engine/duplex/session/manager.py`: route fallback messages and emit internal fallback requests/cancellation.
- `vllm_omni/engine/duplex/session/model_channel.py`: apply model-neutral fallback output through the existing response state path.
- `vllm_omni/engine/duplex/session/runner.py`: start fallback requests, accept Qwen text turns, handle fallback output/errors, and cancel stale fallback tasks.
- `vllm_omni/entrypoints/duplex_omni.py`: route internal fallback requests/cancels to the registered serving sink.
- `vllm_omni/entrypoints/duplex/chat_fallback.py`: replace the old `DuplexSession` mixin with API-side request/projection helpers.
- `vllm_omni/entrypoints/duplex/serving.py`: register the fallback sink and run/cancel API-side fallback tasks.
- `vllm_omni/entrypoints/openai/api_server.py`: initialize chat before the duplex handler and pass the chat service into it.
- `vllm_omni/model_executor/models/qwen3_omni/pipeline.py`: replace `duplex_serving_adapter` with `duplex_plugin`.
- `tests/entrypoints/duplex/test_duplex_serving.py`: cover fallback sink routing and cancellation without exposing internal messages.
- `tests/entrypoints/openai_api/test_duplex_handler.py`: resolve the upstream conflict and retain only tests compatible with the engine-resident API.

### Removed or no longer referenced

- `vllm_omni/experimental/fullduplex/qwen3omni/serving_adapter.py`
- `vllm_omni/experimental/fullduplex/qwen3omni/data_plane.py`
- `vllm_omni/experimental/fullduplex/qwen3omni/session.py`
- `vllm_omni/experimental/fullduplex/qwen3omni/policy.py`

Delete these only after all imports and tests have moved to the model-owned
duplex package. The old `runtime_bridge.py` and `runtime_adapter.py` are
upstream conflict content, not Qwen plugin dependencies.

## Task 1: Add the Internal Fallback Contract

**Files:**

- Create: `vllm_omni/engine/duplex/fallback.py`
- Modify: `vllm_omni/engine/duplex/config.py`
- Modify: `vllm_omni/engine/duplex/plugin.py`
- Modify: `vllm_omni/engine/duplex/messages.py`
- Test: `tests/engine/duplex/test_duplex_fallback.py`

- [ ] **Step 1: Write failing DTO and capability tests**

Add tests that require immutable identity-bearing DTOs and a capability that
distinguishes chat fallback from native duplex:

```python
def test_fallback_request_copies_session_identity_and_history() -> None:
    request = DuplexFallbackRequest(
        session_id="sid",
        request_id="fallback-sid-0-1",
        response_id="resp-sid-0-abcd",
        epoch=3,
        history=({"role": "user", "content": "hello"},),
        response_config={"model": "qwen", "modalities": ["audio"]},
        input_payload={"format": "pcm_f32le", "audio": "AAAA", "sample_rate_hz": 16_000},
        policy_messages=({"role": "system", "content": "policy"},),
    )

    assert request.session_id == "sid"
    assert request.epoch == 3
    assert request.history == (("role", "user"),)  # replace with the exact immutable DTO assertion
```

Use a complete assertion for the chosen DTO representation; the important
contract is that construction copies the input mappings and later mutation of
the caller's dictionaries cannot alter the request. Also assert:

```python
def test_chat_fallback_capability_is_false_by_default() -> None:
    assert DuplexCapabilities().supports_chat_fallback is False

def test_fallback_messages_are_engine_messages_not_realtime_events() -> None:
    message = DuplexSessionFallbackOutputMessage(
        session_id="sid",
        request_id="fallback-sid-0-1",
        response_id="resp",
        epoch=0,
        output={"text": "hello", "end_of_turn": True},
    )
    assert message.type == "duplex_session_fallback_output"
```

- [ ] **Step 2: Run the focused tests and confirm failure**

Run:

```bash
pytest -q tests/engine/duplex/test_duplex_fallback.py
```

Expected: collection fails because the DTO, capability field, and fallback
message classes do not exist yet.

- [ ] **Step 3: Implement the immutable DTOs and message classes**

Define `DuplexFallbackRequest` with these fields:

```python
@dataclass(frozen=True, slots=True)
class DuplexFallbackRequest:
    session_id: str
    request_id: str
    response_id: str
    epoch: int
    history: tuple[Mapping[str, object], ...]
    response_config: Mapping[str, object]
    input_payload: Mapping[str, object] | None
    policy_messages: tuple[Mapping[str, object], ...]
```

Copy each mapping in `__post_init__` and expose tuples so the engine snapshot
cannot be changed by the API-side task. Define matching output DTO/message
types for `started`, `output`, `failed`, and `cancelled`, each carrying the
request id, response id, and epoch. Add `supports_chat_fallback: bool = False`

```python
default_auto_response: bool = False

def fallback_policy_messages(self, state: DuplexModelSessionState) -> tuple[Mapping[str, object], ...]:
    del state
    return ()

def on_fallback_started(self, state: DuplexModelSessionState) -> None:
    del state

def on_barge_in(self, state: DuplexModelSessionState) -> None:
    del state
```

Keep these hooks concrete defaults so existing native plugins remain valid.
Make `DuplexCapabilities.as_dict()` use configurable `implementation_level` and
`input_modes` fields with the current native defaults, allowing Qwen to report
its turn-based mode without changing MiniCPM behavior.

- [ ] **Step 4: Run the tests and type/format checks**

Run:

```bash
pytest -q tests/engine/duplex/test_duplex_fallback.py
ruff check vllm_omni/engine/duplex/fallback.py vllm_omni/engine/duplex/config.py vllm_omni/engine/duplex/plugin.py vllm_omni/engine/duplex/messages.py tests/engine/duplex/test_duplex_fallback.py
ruff format --check vllm_omni/engine/duplex/fallback.py vllm_omni/engine/duplex/config.py vllm_omni/engine/duplex/plugin.py vllm_omni/engine/duplex/messages.py tests/engine/duplex/test_duplex_fallback.py
```

Expected: pure contract tests pass; the environment may still fail during
collection with the known missing `torch` dependency.

## Task 2: Implement the Qwen3 Model Plugin

**Files:**

- Create: `vllm_omni/model_executor/models/qwen3_omni/duplex/__init__.py`
- Create: `vllm_omni/model_executor/models/qwen3_omni/duplex/data_plane.py`
- Create: `vllm_omni/model_executor/models/qwen3_omni/duplex/input.py`
- Create: `vllm_omni/model_executor/models/qwen3_omni/duplex/plugin.py`
- Create: `vllm_omni/model_executor/models/qwen3_omni/duplex/policy.py`
- Create: `vllm_omni/model_executor/models/qwen3_omni/duplex/session.py`
- Test: `tests/engine/duplex/test_qwen3_plugin.py`

- [ ] **Step 1: Write failing plugin, state, and policy tests**

Add tests for the new dotted path and the Qwen-specific behavior:

```python
def test_qwen3_plugin_loads_through_the_unified_contract() -> None:
    plugin = load_duplex_plugin(
        "vllm_omni.model_executor.models.qwen3_omni.duplex.plugin.Qwen3OmniDuplexPlugin",
        _encode_audio,
    )
    capabilities = plugin.capabilities(max_sessions=1)
    assert capabilities.supports_chat_fallback is True
    assert capabilities.supports_input_append is False
    assert capabilities.supports_chat_completions is True
    assert capabilities.input_modes == ["turn_commit_only"]

def test_qwen3_fallback_policy_adds_interruption_note_until_request_starts() -> None:
    plugin = Qwen3OmniDuplexPlugin(_encode_audio)
    state = plugin.create_session_state()
    state.last_turn_interrupted = True

    messages = plugin.fallback_policy_messages(state)

    assert messages == (
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": INTERRUPTION_NOTE},
    )
    plugin.on_fallback_started(state)
    assert state.last_turn_interrupted is False

def test_qwen3_native_data_plane_fails_closed() -> None:
    plugin = Qwen3OmniDuplexPlugin(_encode_audio)
    with pytest.raises(RuntimeError, match="chat fallback"):
        list(plugin.data_plane.project({}, context=None))
```

Add buffer tests that append two PCM payloads, return one committed payload,
preserve its `pcm_f32le` format/sample rate, and clear all state after `clear()`.

- [ ] **Step 2: Run the tests and confirm failure**

Run:

```bash
pytest -q tests/engine/duplex/test_qwen3_plugin.py
```

Expected: collection or import failure because the model-owned duplex package
and unified plugin do not exist.

- [ ] **Step 3: Port the Qwen-specific policy and state**

Move the existing `SYSTEM_PROMPT` and `INTERRUPTION_NOTE` constants into
`duplex/policy.py`. Define a dataclass state based on
`DuplexModelSessionState`, with a committed-turn buffer, all required base
fields, and `last_turn_interrupted: bool = False`. Implement reservation
commit/rollback in `input.py` using the existing `PcmAppendBuffer` contract;
the buffer must retain the complete turn and must not emit native model chunks.

- [ ] **Step 4: Implement the Qwen3 plugin**

Implement `Qwen3OmniDuplexPlugin` with:

```python
plugin_id = "qwen3omni"
private_runtime_config_keys = frozenset({"auto_commit_silence_ms"})
default_auto_response = True

def configure_sampling_params(self, *, runtime_config, defaults):
    return tuple(defaults)

def plan_append(self, **kwargs):
    raise RuntimeError("Qwen3-Omni uses the chat fallback; native append is disabled")

def decide_output(self, **kwargs):
    return None
```

Set Qwen capabilities to chat fallback, turn-commit-only, no native input
append, no native model turn policy, Realtime endpoint enabled, single-session
and no resume. Preserve `validate_client_extra_body`,
`prepare_runtime_config`, and `runtime_config_for_update` for the private
silence setting. `fallback_policy_messages()` returns the Qwen policy and
optional interruption note. `on_fallback_started()` clears the interruption
marker, while `on_barge_in()` sets it.

- [ ] **Step 5: Run plugin tests and pipeline-independent static checks**

Run the Task 2 tests and:

```bash
ruff check vllm_omni/model_executor/models/qwen3_omni/duplex tests/engine/duplex/test_qwen3_plugin.py
ruff format --check vllm_omni/model_executor/models/qwen3_omni/duplex tests/engine/duplex/test_qwen3_plugin.py
```

Expected: all runnable pure plugin tests pass.

## Task 3: Add Engine-Resident Fallback Lifecycle

**Files:**

- Modify: `vllm_omni/engine/duplex/session/context.py`
- Modify: `vllm_omni/engine/duplex/session/manager.py`
- Modify: `vllm_omni/engine/duplex/session/model_channel.py`
- Modify: `vllm_omni/engine/duplex/session/runner.py`
- Modify: `vllm_omni/engine/duplex_omni_engine.py`
- Modify: `vllm_omni/engine/duplex/messages.py`
- Test: `tests/engine/duplex/test_qwen3_fallback_runner.py`

- [ ] **Step 1: Write failing runner lifecycle tests**

Use the existing `RecordingStagePort`/manager harness pattern from
`tests/engine/duplex/test_session_runner.py` and a Qwen plugin. Assert:

```python
async def test_qwen_commit_emits_internal_fallback_request_and_engine_owned_response() -> None:
    harness = await open_qwen_harness()
    try:
        await harness.run(append_qwen_audio())
        events = await harness.run(commands.Commit())
        request = await harness.next_fallback_request()

        assert request.session_id == SESSION_ID
        assert request.epoch == harness.session.epoch
        assert request.response_id == find(events, "response.created").response_id
        assert request.history[-1]["role"] == "user"
        assert harness.port.submissions == []
    finally:
        await close_harness(harness)

async def test_fallback_output_uses_normal_realtime_projection_and_history() -> None:
    harness = await open_qwen_harness()
    try:
        await harness.start_fallback_turn()
        response_id = harness.session.active_response_id
        await harness.send_fallback_started()
        events = await harness.send_fallback_output({
            "text": "hello",
            "data_plane_request_id": harness.fallback_request_id,
            "end_of_turn": True,
        })
        assert find(events, "response.text.delta").delta == "hello"
        assert find(events, "response.done").status == "completed"
        assert harness.session.active_response_id is None
        assert response_id is not None
    finally:
        await close_harness(harness)

async def test_barge_in_invalidates_and_cancels_fallback_output() -> None:
    harness = await open_qwen_harness()
    try:
        await harness.start_fallback_turn()
        request_id = harness.fallback_request_id
        old_epoch = harness.session.epoch
        events = await harness.run(commands.BargeIn())
        assert harness.session.epoch == old_epoch + 1
        assert harness.fallback_cancels == [request_id]
        assert await harness.send_fallback_output({
            "text": "late",
            "data_plane_request_id": request_id,
            "end_of_turn": True,
            "epoch": old_epoch,
        }) == []
    finally:
        await close_harness(harness)
```

The harness must assert that no native stage request is submitted for a Qwen
fallback turn and that the internal fallback request is not a public Realtime
event.

- [ ] **Step 2: Run the tests and confirm failure**

Run:

```bash
pytest -q tests/engine/duplex/test_qwen3_fallback_runner.py
```

Expected: failures because the manager, runner, and engine have no fallback
message routing or response path.

- [ ] **Step 3: Add fallback identity and manager routing**

Add `fallback_request_id: str | None` to `DuplexRunState`. Add the internal
message classes to the manager's accepted message tuple. Route started/output/
failed messages to the matching runner after checking the session id; enqueue
them on the runner mailbox so all session mutations remain ordered. Add manager
methods that put `DuplexSessionFallbackRequestMessage` and

- [ ] **Step 4: Start fallback responses from commit and response.create**

Add `_start_chat_fallback(input_payload)` to the runner. It must:

1. call `session.begin_response()`;
2. allocate `duplex-fallback-<session>-<epoch>-<input_commit_seq>`;
3. bind that id with `session.bind_request()` and `run.fallback_request_id`;
4. emit `response.created` using `ModelChannel.response_created_payload()`;
5. snapshot `session.history`, `session.response_config.as_dict()`, the committed
   audio payload, and `plugin.fallback_policy_messages(model_state)`;
6. ask the manager to emit the internal fallback request.

Use this method from the committed-turn path whenever
appending a user message and marking it unanswered. Permit a text-only
`response.create` to start fallback; retain the native text-only error for
plugins without chat fallback. Honor explicit `response_create=False` even when
the Qwen plugin's default auto-response is true.

- [ ] **Step 5: Apply started/output/failed messages through `ModelChannel`**

Add a model-channel method that accepts a model-neutral `dict[str, object]`
payload and passes it to `_send_one_model_output_event()` with the expected
fallback epoch. Every successful payload must contain
Realtime projection, response history, and `response.done`. Started messages
call `plugin.on_fallback_started(model_state)` and mark the compatibility data
plane request active. Failed messages call the existing model-error response
sequence with `error_code`, `error`, and the fallback request id.

- [ ] **Step 6: Integrate cancellation and cleanup**

When `_cancel_active_response()` sees `run.fallback_request_id`, emit an
internal fallback cancel message instead of trying to abort it as a native
stage request. Clear the fallback identity after invalidating the epoch. Call
`plugin.on_barge_in(model_state)` only when an active response was actually
cancelled. Close and expiry paths must emit the same cancel message before
tearing down the runner. Late fallback output must be ignored by epoch and
request-id checks.

- [ ] **Step 7: Run engine tests and static checks**

Run:

```bash
pytest -q tests/engine/duplex/test_duplex_fallback.py tests/engine/duplex/test_qwen3_plugin.py tests/engine/duplex/test_qwen3_fallback_runner.py tests/engine/duplex/test_session_runner.py
ruff check vllm_omni/engine/duplex vllm_omni/engine/duplex_omni_engine.py tests/engine/duplex
ruff format --check vllm_omni/engine/duplex vllm_omni/engine/duplex_omni_engine.py tests/engine/duplex
```

Expected: all runnable tests pass; full import collection may report the known
missing `torch` dependency.

## Task 4: Add the API-Side Chat Fallback Bridge

**Files:**

- Modify: `vllm_omni/entrypoints/duplex_omni.py`
- Modify: `vllm_omni/entrypoints/duplex/chat_fallback.py`
- Modify: `vllm_omni/entrypoints/duplex/serving.py`
- Modify: `vllm_omni/entrypoints/openai/api_server.py`
- Modify: `vllm_omni/engine/duplex_omni_engine.py`
- Create: `tests/entrypoints/duplex/test_chat_fallback.py`
- Modify: `tests/entrypoints/duplex/test_duplex_serving.py`

- [ ] **Step 1: Write failing pure request/projection tests**

Use a fake chat service and immutable fallback request. Test the following
without constructing an engine session:

```python
    request = DuplexFallbackRequest(
        session_id="sid",
        request_id="fallback-sid-0-1",
        response_id="resp",
        epoch=0,
        history=({
            "role": "user",
            "content": [{"type": "audio_url", "audio_url": {"url": "native-duplex:input-audio"}}],
        },),
        response_config={"model": "qwen", "modalities": ["audio"], "response_format": "wav"},
        input_payload={"audio": base64.b64encode(struct.pack("<2f", 0.1, 0.2)).decode(), "format": "pcm_f32le", "sample_rate_hz": 16_000},
        policy_messages=({"role": "system", "content": "policy"},),
    )

    chat_request = build_chat_request(request, model="qwen")

    assert chat_request.messages[0]["content"] == "policy"
    assert chat_request.messages[-1]["content"][0]["audio_url"]["url"].startswith("data:audio/wav;base64,")

def test_project_chat_payload_returns_model_neutral_text_output() -> None:
    outputs = project_chat_payload(
        {"choices": [{"delta": {"content": "hello"}, "finish_reason": None}]},
        request_id="fallback-sid-0-1",
    )
    assert outputs == [{"text": "hello", "data_plane_request_id": "fallback-sid-0-1"}]

def test_project_chat_error_returns_failed_result_without_resetting_policy() -> None:
    result = project_chat_payload(
        {"error": {"message": "rejected", "type": "BadRequestError"}},
        request_id="fallback-sid-0-1",
    )
    assert result == [{"error": "rejected", "error_code": "BadRequestError"}]
```

Also test audio chunks return per-chunk duration and preserve the configured
format, and that `[DONE]` becomes one final `end_of_turn` model-neutral output.

- [ ] **Step 2: Run the tests and confirm failure**

Run:

```bash
pytest -q tests/entrypoints/duplex/test_chat_fallback.py
```

Expected: collection fails because the new pure request/projection helpers do
not exist.

- [ ] **Step 3: Implement pure request construction and projection**

Replace the old mixin's dependency on `entrypoints.duplex.protocol.DuplexSession`
with helpers accepting `DuplexFallbackRequest`. Build `ChatCompletionRequest`
from the copied response config, policy messages, instructions, and history.
Convert an engine `pcm_f32le` payload to a PCM16 WAV data URI with
`pcm_f32le_payload_to_wav`. Strip Realtime-only extra-body keys and retain the
existing tools/tool-choice mapping.

Implement `project_chat_payload()` as a pure function. It must return model
neutral dictionaries using `text`, `audio`, `audio_format`, `sample_rate_hz`,
`audio_duration_ms`, `data_plane_request_id`, `error`, and `error_code`; it must
never mutate a session or playback cursor. Use `_audio_metadata()` for duration
calculation and emit a final `end_of_turn` dictionary for stream completion.

- [ ] **Step 4: Add engine routing and serving task ownership**

Add a fallback sink to `DuplexOmni`. `_route_engine_message()` must consume
internal fallback request/cancel messages without placing them in the public
`DuplexSessionHandle.events()` queue. Register a synchronous callback from
`OmniDuplexSessionHandler`; the handler stores one API task per session and
passes `DuplexFallbackRequest` to it.

The task must:

1. build the chat request;
2. await `create_chat_completion()`;
3. submit `started` only after the result is known not to be an
   `ErrorResponse`/generic error object;
4. drain async SSE chunks through `project_chat_payload()`;
5. submit every output to the engine;
6. submit one terminal output when the provider stream completes;
7. submit a failed message on provider error/exception; and
8. stop on cancellation without sending another output.

Add `submit_fallback_started_async`, `submit_fallback_output_async`, and
`submit_fallback_failed_async` to `DuplexOmniEngine`, using the same bounded
request queue/backpressure policy as session commands.

- [ ] **Step 5: Reorder API initialization**

In `_init_duplex_app_state()`, initialize `state.openai_serving_chat` before
constructing `OmniDuplexSessionHandler`, then pass the chat service into the
handler. Keep `/v1/chat/completions` on the same `DuplexOmni` engine and leave
ordinary chat generation outside the orchestrator loop.

- [ ] **Step 6: Run API bridge tests and static checks**

Run:

```bash
pytest -q tests/entrypoints/duplex/test_chat_fallback.py tests/entrypoints/duplex/test_duplex_serving.py
ruff check vllm_omni/entrypoints/duplex_omni.py vllm_omni/entrypoints/duplex/chat_fallback.py vllm_omni/entrypoints/duplex/serving.py vllm_omni/entrypoints/openai/api_server.py tests/entrypoints/duplex
ruff format --check vllm_omni/entrypoints/duplex_omni.py vllm_omni/entrypoints/duplex/chat_fallback.py vllm_omni/entrypoints/duplex/serving.py vllm_omni/entrypoints/openai/api_server.py tests/entrypoints/duplex
```

Expected: pure projection tests pass; runtime-dependent collection may be
blocked by `torch`.

## Task 5: Wire Qwen3 and Resolve the Six Merge Paths

**Files:**

- Modify: `vllm_omni/model_executor/models/qwen3_omni/pipeline.py`
- Resolve: `tests/entrypoints/openai_api/test_duplex_handler.py`
- Resolve: `vllm_omni/entrypoints/duplex/chat_fallback.py`
- Resolve: `vllm_omni/entrypoints/duplex/realtime_input.py`
- Resolve: `vllm_omni/entrypoints/duplex/runtime_bridge.py`
- Resolve: `vllm_omni/entrypoints/duplex/serving.py`
- Resolve: `vllm_omni/entrypoints/duplex/session_runner.py`
- Modify: Qwen3 and duplex tests that still import the deleted adapter package.

- [ ] **Step 1: Write the pipeline wiring regression test**

Add to `tests/engine/duplex/test_qwen3_plugin.py`:

```python
    assert QWEN3_OMNI_PIPELINE.duplex_plugin == (
        "vllm_omni.model_executor.models.qwen3_omni.duplex.plugin.Qwen3OmniDuplexPlugin"
    )
    assert QWEN3_OMNI_PIPELINE.duplex_serving_adapter is None
```

Run the test before changing the pipeline; it must fail against the current
legacy binding.

- [ ] **Step 2: Change the Qwen3 pipeline binding**

Replace the existing `duplex_serving_adapter` keyword with:

```python
duplex_plugin=(
    "vllm_omni.model_executor.models.qwen3_omni.duplex.plugin.Qwen3OmniDuplexPlugin"
),
```

Move all Qwen tests to the engine config/events/session imports and the new
model-owned plugin package. Delete the old experimental Qwen adapter files only
when `rg` reports no remaining imports.

- [ ] **Step 3: Resolve `realtime_input.py` and `serving.py` from upstream first**

Remove all conflict markers. Keep upstream's `RealtimeEnvelope`, typed command
translation, attachment registry, event pump, session resume rules, and
Server-VAD configuration. Do not restore the old serving-side session registry
or `DuplexSession` ledger. Keep the fallback sink/task additions from Task 4 in
the final `serving.py`.

- [ ] **Step 4: Resolve `runtime_bridge.py` without reintroducing Qwen control RPCs**

The new Qwen plugin does not use the old runtime bridge. Keep upstream's native
runtime bridge only for legacy/native adapter callers that still compile, but
ensure the new `DuplexOmni` path does not import or select it. Remove any
Qwen-specific `supports_runtime_control` or old adapter branches that refer to
deleted classes.

- [ ] **Step 5: Resolve `session_runner.py` from upstream and retain generic fallback hooks**

Keep upstream sample-rate validation, Server VAD input ordering, backpressure,
append tail, cancellation fences, and typed projection. Add only the generic
fallback branches from Task 3. Do not paste the old 2,500-line serving runner
or reintroduce its session state.

- [ ] **Step 6: Resolve chat fallback and handler tests**

Keep the upstream engine-owned error/event behavior and the new pure API-side
projection. Remove old tests that instantiate `DuplexSession` or
`Qwen3OmniServingRuntimeAdapter`; replace them with the Qwen plugin runner and
pure fallback tests. Ensure no test asserts the deleted
`duplex_serving_adapter` path.

- [ ] **Step 7: Check the merge index and marker-free source**

Run:

```bash
rg -n '^(<<<<<<<|=======|>>>>>>>)' vllm_omni tests
rg -n 'experimental\.fullduplex\.qwen3omni|duplex_serving_adapter|entrypoints\.duplex\.runtime_adapter' vllm_omni/model_executor/models/qwen3_omni tests/engine/duplex tests/entrypoints/duplex
```

Expected: the first two commands produce no output. The third command may
report unrelated legacy pipelines, but it must not report Qwen3's pipeline or
the new duplex plugin/tests.

## Task 6: Verify, Stage, and Create the Merge Commit

**Files:** all files listed above; no unrelated worktree files.

- [ ] **Step 1: Compile changed Python files without importing vLLM**

Run:

```bash
python3 -m compileall -q vllm_omni/engine/duplex vllm_omni/engine/duplex_omni_engine.py vllm_omni/entrypoints/duplex_omni.py vllm_omni/entrypoints/duplex vllm_omni/model_executor/models/qwen3_omni/duplex
```

Expected: exit 0 and no syntax errors.

- [ ] **Step 2: Run focused lint and format checks**

Run:

```bash
ruff check vllm_omni/engine/duplex vllm_omni/engine/duplex_omni_engine.py vllm_omni/entrypoints/duplex_omni.py vllm_omni/entrypoints/duplex vllm_omni/entrypoints/openai/api_server.py vllm_omni/model_executor/models/qwen3_omni/duplex tests/engine/duplex tests/entrypoints/duplex tests/entrypoints/openai_api/test_duplex_handler.py
ruff format --check vllm_omni/engine/duplex vllm_omni/engine/duplex_omni_engine.py vllm_omni/entrypoints/duplex_omni.py vllm_omni/entrypoints/duplex vllm_omni/entrypoints/openai/api_server.py vllm_omni/model_executor/models/qwen3_omni/duplex tests/engine/duplex tests/entrypoints/duplex tests/entrypoints/openai_api/test_duplex_handler.py
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 3: Run the focused test suite**

Run:

```bash
pytest -q tests/engine/duplex/test_duplex_fallback.py tests/engine/duplex/test_qwen3_plugin.py tests/engine/duplex/test_qwen3_fallback_runner.py tests/engine/duplex/test_session_runner.py tests/entrypoints/duplex/test_chat_fallback.py tests/entrypoints/duplex/test_duplex_serving.py tests/entrypoints/openai_api/test_duplex_handler.py
```

Expected: all runnable tests pass. If collection fails because `torch` is not
installed, record the exact `ModuleNotFoundError` and report which pure-Python
checks did pass.

- [ ] **Step 4: Inspect the merge and implementation diff**

Run:

```bash
git diff --check
```

Confirm the unresolved-path list is empty, the design/plan and intended code
files are present, and no user changes outside the task were staged.

- [ ] **Step 5: Stage all intended changes and create one normal merge commit**

Run:

```bash
git status --short
git commit --signoff -m "chore: migrate Qwen3 duplex to unified plugin"
```

The commit must have two parents, retain the upstream merge history, and not be
a squash or rebase. Do not stage unrelated pre-existing changes by path; inspect
the staged diff before committing and add any omitted intended file explicitly.

## Task 7: Publish and Open the Integration PR

- [ ] **Step 1: Verify the commit and remote tracking**

Run:

```bash
```

Expected: the merge commit is at `HEAD`, the worktree is clean apart from
explicitly documented unrelated changes, and the commit has the expected
upstream parent.

- [ ] **Step 2: Push the integration branch**

Run:

```bash
```

- [ ] **Step 3: Create the fork PR against the existing feature branch**

Run:

```bash
  --repo valendra-tech/vllm-omni \
  --base VAL-351-qwen3-omni-duplex-upstream \
  --head VAL-351-qwen3-omni-conflict-resolution \
  --title "chore: migrate Qwen3 duplex to unified plugin" \
  --body "## Summary
- Merge current upstream duplex changes into the Qwen3-Omni feature branch
- Migrate Qwen3 to DuplexModelPlugin
- Preserve turn-based chat fallback through an engine/API bridge
- Keep engine-owned history, cancellation, and stale-output filtering

## Validation
- No unresolved merge paths or conflict markers
- Ruff, format, compile, and runnable focused tests checked
- Missing torch dependency recorded if applicable

This integration PR updates the branch used by upstream PR #6372."
```

Return the PR URL and state the exact validation commands that passed or were
blocked by missing dependencies.
