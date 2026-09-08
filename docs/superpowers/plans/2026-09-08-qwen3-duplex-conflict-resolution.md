# Qwen3-Omni Duplex Conflict Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge the current `upstream/main` into the Qwen3-Omni feature branch, preserve both upstream duplex behavior and the Qwen3 chat-fallback behavior, and publish the result as a fork PR targeting `VAL-351-qwen3-omni-duplex-upstream`.

**Architecture:** Start a new integration branch from `origin/VAL-351-qwen3-omni-duplex-upstream`, merge `upstream/main`, and resolve only the five content-conflict files. Keep upstream's newer Server VAD and duplex session behavior as the base; reapply Qwen3-specific adapter hooks, runtime-control opt-out, private-config validation, and float32 input conversion at their generic extension points.

**Tech Stack:** Git merge workflow, Python 3.11/3.12, pytest, Ruff, vLLM-Omni duplex runtime, GitHub CLI.

---

## File Map

- Modify: `tests/entrypoints/openai_api/test_duplex_handler.py` to retain both upstream and Qwen3 regression tests.
- Modify: `vllm_omni/entrypoints/duplex/chat_fallback.py` to combine current error handling with the Qwen3 request-issued hook.
- Modify: `vllm_omni/entrypoints/duplex/runtime_bridge.py` to gate runtime control by both adapter eligibility and `supports_runtime_control`.
- Modify: `vllm_omni/entrypoints/duplex/serving.py` to validate adapter-owned client config before preparing runtime config.
- Modify: `vllm_omni/entrypoints/duplex/session_runner.py` to preserve current Server VAD handling while restoring Qwen3 float32 conversion.
- Do not modify: `docs/serving/realtime_duplex_api.md`, `recipes/Qwen/Qwen3-Omni.md`, or the Qwen3 adapter files unless verification exposes a concrete integration issue; they merge automatically.

## Task 1: Create the Integration Branch

**Files:**
- Branch: `VAL-351-qwen3-omni-conflict-resolution`
- Base: `origin/VAL-351-qwen3-omni-duplex-upstream`
- Merge source: `upstream/main`

- [ ] **Step 1: Refresh the two source refs**

Run:

```bash
git fetch origin VAL-351-qwen3-omni-duplex-upstream
git fetch upstream main
```

Expected: `origin/VAL-351-qwen3-omni-duplex-upstream` points at `2dcc56e3` and
`upstream/main` points at the current upstream main commit.

- [ ] **Step 2: Create the new branch from the existing feature branch**

Run:

```bash
git switch --detach origin/VAL-351-qwen3-omni-duplex-upstream
git switch -c VAL-351-qwen3-omni-conflict-resolution
```

Expected: the new branch contains the existing feature branch history and no
working-tree changes.

- [ ] **Step 3: Start a non-fast-forward merge without committing**

Run:

```bash
git merge --no-commit --no-ff upstream/main
```

Expected unresolved paths:

```text
tests/entrypoints/openai_api/test_duplex_handler.py
vllm_omni/entrypoints/duplex/chat_fallback.py
vllm_omni/entrypoints/duplex/runtime_bridge.py
vllm_omni/entrypoints/duplex/serving.py
vllm_omni/entrypoints/duplex/session_runner.py
```

## Task 2: Resolve Tests and Chat Fallback

**Files:**
- Modify: `tests/entrypoints/openai_api/test_duplex_handler.py`
- Modify: `vllm_omni/entrypoints/duplex/chat_fallback.py`

- [ ] **Step 1: Keep both conflicting handler tests**

Remove the conflict markers and retain these two complete test functions:

```python
def test_duplex_session_preserves_response_history_order_when_user_item_arrives_during_generation():
    session = DuplexSession(session_id="sid-history-order", config=DuplexSessionConfig())
    first_user = {"role": "user", "content": "first turn"}
    next_user = {
        "role": "user",
        "content": [{"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,AAAA"}}],
    }
    session.append_history_message(first_user)
    session.begin_response()
    session.append_assistant_text("first answer")
    session.append_history_message(next_user)

    assistant = session.end_response(commit_text=True)

    assert session.history == (first_user, assistant, next_user)


def test_serving_adapter_auto_respond_on_commit_supports_callable_and_boolean():
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(FakeEngineClient()),
        config_timeout_s=0.1,
        idle_timeout_s=1,
        serving_runtime_adapter=PersonaPlexServingRuntimeAdapter(lambda *_: None),
    )
    session = DuplexSession(session_id="sid-commit-hook", config=DuplexSessionConfig())
    state = object()
    calls: list[tuple[str, object]] = []

    def hook(session_id: str, session_state: object) -> bool:
        calls.append((session_id, session_state))
        return False

    handler._serving_runtime_adapter = SimpleNamespace(
        auto_respond_on_commit=hook,
        session_state=lambda session_id: state,
    )
    assert handler._serving_adapter_auto_respond_on_commit(session) is False
    assert calls == [(session.session_id, state)]

    handler._serving_runtime_adapter = SimpleNamespace(auto_respond_on_commit=True)
    assert handler._serving_adapter_auto_respond_on_commit(session) is True
```

- [ ] **Step 2: Preserve upstream error handling and Qwen3 request-issued state reset**

In `ChatFallbackProjectorMixin._run_response`, keep the upstream-compatible
error test and insert the adapter callback only after the result is known not
to be an error:

```python
error_info = getattr(result, "error", None)
if isinstance(result, ErrorResponse) or error_info is not None:
    await send_json(
        {
            "type": "error",
            "error": getattr(error_info, "message", None) or str(result),
            "code": getattr(error_info, "type", None) or "chat_error",
        }
    )
    session.end_response(commit_text=False)
    return
adapter = getattr(self, "_serving_runtime_adapter", None)
request_issued = getattr(adapter, "on_turn_request_issued", None)
if callable(request_issued):
    request_issued(session.session_id, adapter.session_state(session.session_id))
```

- [ ] **Step 3: Check the two files for unresolved markers**

Run:

```bash
rg -n '^(<<<<<<<|=======|>>>>>>>)' tests/entrypoints/openai_api/test_duplex_handler.py vllm_omni/entrypoints/duplex/chat_fallback.py
```

Expected: no output.

## Task 3: Resolve Runtime Bridge and Serving Adapter Validation

**Files:**
- Modify: `vllm_omni/entrypoints/duplex/runtime_bridge.py`
- Modify: `vllm_omni/entrypoints/duplex/serving.py`

- [ ] **Step 1: Preserve the Qwen3 runtime-control opt-out helper**

Keep this helper in `NativeRuntimeBridgeMixin`:

```python
def _serving_adapter_uses_runtime_control(self) -> bool:
    adapter = getattr(self, "_serving_runtime_adapter", None)
    return getattr(adapter, "supports_runtime_control", True) is not False
```

- [ ] **Step 2: Combine both runtime-control gates in all three lifecycle methods**

In `_open_runtime_session`, `_signal_runtime_session`, and
`_close_runtime_session`, use this guard before accessing engine runtime RPCs:

```python
if not self._uses_serving_runtime_adapter(session.config) or not self._serving_adapter_uses_runtime_control():
    return True
```

This keeps upstream's session eligibility check and prevents Qwen3's
chat-fallback adapter from opening, signaling, or closing native runtime
control sessions. Leave `_serving_adapter_auto_respond_on_commit` unchanged.

- [ ] **Step 3: Validate `extra_body` through the selected local adapter**

In the session-create path, retain the upstream local adapter variable and
validate before preparing runtime config:

```python
runtime_adapter = self._serving_runtime_adapter if self._uses_serving_runtime_adapter(config) else None
runtime_config: dict[str, object] = {}
if runtime_adapter is not None:
    try:
        runtime_adapter.validate_client_extra_body(config.extra_body)
        runtime_config = await runtime_adapter.prepare_runtime_config(
            config,
            model_config=getattr(self._chat_service, "model_config", None),
        )
```

Use `runtime_adapter` for the subsequent `capabilities` and
`replace_runtime_config` calls in this handshake.

- [ ] **Step 4: Check the two files for unresolved markers**

Run:

```bash
rg -n '^(<<<<<<<|=======|>>>>>>>)' vllm_omni/entrypoints/duplex/runtime_bridge.py vllm_omni/entrypoints/duplex/serving.py
```

Expected: no output.

## Task 4: Resolve Audio Input and Server VAD Integration

**File:**
- Modify: `vllm_omni/entrypoints/duplex/session_runner.py`

- [ ] **Step 1: Combine the audio imports**

Keep the current upstream sample-rate validator and add the Qwen3 float32
conversion helper:

```python
from vllm_omni.entrypoints.duplex.audio import (
    convert_input_audio_with_rate,
    pcm_f32le_payload_to_wav,
    validate_input_sample_rate_hz,
)
```

- [ ] **Step 2: Preserve current event format aliases**

Use the broader event field fallback before selecting the default:

```python
fmt = event.get("format") or event.get("input_audio_format") or event.get("audio_format") or "pcm16"
if not isinstance(fmt, str):
    fmt = "pcm16"
```

Keep the upstream `native_input`, `server_vad_config`, and
`turn_based_server_vad` variables immediately after sample-rate parsing.

- [ ] **Step 3: Keep the latest upstream Server VAD branch intact**

Retain the upstream behavior that skips generic audio conversion for
turn-based Server VAD, validates PCM16/sample-rate input, applies backpressure,
pushes audio through the VAD pipeline, and commits detected turns. Do not run
float32-to-WAV conversion in that branch; Server VAD must continue to receive
its validated PCM16 input.

- [ ] **Step 4: Add Qwen3 float32 conversion only to the generic non-native path**

After generic `convert_input_audio_with_rate` succeeds and before the PCM16
decode rejection, use:

```python
if (
    not native_input
    and not turn_based_server_vad
    and isinstance(audio, str)
    and isinstance(fmt, str)
    and fmt.lower() == "pcm_f32le"
):
    try:
        audio, fmt, sample_rate_hz = pcm_f32le_payload_to_wav(audio, sample_rate_hz)
    except ValueError as exc:
        await emit_event({"type": "error", "error": str(exc), "code": "bad_audio"})
        continue
```

Then retain the upstream PCM16 rejection and turn-detection locking rules.

- [ ] **Step 5: Check the file for unresolved markers**

Run:

```bash
rg -n '^(<<<<<<<|=======|>>>>>>>)' vllm_omni/entrypoints/duplex/session_runner.py
```

Expected: no output.

## Task 5: Verify the Merge and Focused Behavior

**Files:**
- Test: `tests/entrypoints/openai_api/test_duplex_handler.py`
- Test: `tests/entrypoints/openai_api/test_duplex_handler_qwen3omni.py`
- Test: `tests/entrypoints/openai_api/test_duplex_input_audio.py`
- Test: `tests/entrypoints/test_stream_finish_reason.py`
- Test: `tests/e2e/features/fullduplex/test_qwen3omni_data_plane.py`
- Test: `tests/e2e/features/fullduplex/test_qwen3omni_handler_integration.py`
- Test: `tests/e2e/features/fullduplex/test_qwen3omni_policy.py`
- Test: `tests/e2e/features/fullduplex/test_qwen3omni_serving_adapter.py`

- [ ] **Step 1: Confirm no unresolved merge state**

Run:

```bash
git diff --name-only --diff-filter=U
git status --short
rg -n '^(<<<<<<<|=======|>>>>>>>)' tests vllm_omni
```

Expected: the first and third commands produce no output; status shows only
the five resolved files staged for the merge.

- [ ] **Step 2: Run Ruff checks**

Run:

```bash
ruff check tests/entrypoints/openai_api/test_duplex_handler.py tests/entrypoints/openai_api/test_duplex_handler_qwen3omni.py tests/entrypoints/openai_api/test_duplex_input_audio.py tests/entrypoints/test_stream_finish_reason.py tests/e2e/features/fullduplex/test_qwen3omni_data_plane.py tests/e2e/features/fullduplex/test_qwen3omni_handler_integration.py tests/e2e/features/fullduplex/test_qwen3omni_policy.py tests/e2e/features/fullduplex/test_qwen3omni_serving_adapter.py vllm_omni/entrypoints/duplex/chat_fallback.py vllm_omni/entrypoints/duplex/runtime_bridge.py vllm_omni/entrypoints/duplex/serving.py vllm_omni/entrypoints/duplex/session_runner.py
ruff format --check tests/entrypoints/openai_api/test_duplex_handler.py tests/entrypoints/openai_api/test_duplex_handler_qwen3omni.py tests/entrypoints/openai_api/test_duplex_input_audio.py tests/entrypoints/test_stream_finish_reason.py tests/e2e/features/fullduplex/test_qwen3omni_data_plane.py tests/e2e/features/fullduplex/test_qwen3omni_handler_integration.py tests/e2e/features/fullduplex/test_qwen3omni_policy.py tests/e2e/features/fullduplex/test_qwen3omni_serving_adapter.py vllm_omni/entrypoints/duplex/chat_fallback.py vllm_omni/entrypoints/duplex/runtime_bridge.py vllm_omni/entrypoints/duplex/serving.py vllm_omni/entrypoints/duplex/session_runner.py
```

Expected: both commands exit 0.

- [ ] **Step 3: Run the focused duplex/Qwen3 tests**

Run:

```bash
pytest -q tests/entrypoints/openai_api/test_duplex_handler.py tests/entrypoints/openai_api/test_duplex_handler_qwen3omni.py tests/entrypoints/openai_api/test_duplex_input_audio.py tests/entrypoints/test_stream_finish_reason.py tests/e2e/features/fullduplex/test_qwen3omni_data_plane.py tests/e2e/features/fullduplex/test_qwen3omni_handler_integration.py tests/e2e/features/fullduplex/test_qwen3omni_policy.py tests/e2e/features/fullduplex/test_qwen3omni_serving_adapter.py
```

Expected: all collected tests pass. If the local environment lacks the
runtime dependencies, record the exact missing dependency and rely on the
corresponding CI check instead of claiming the suite passed.

- [ ] **Step 4: Inspect the integration diff**

Run:

```bash
git diff --stat origin/VAL-351-qwen3-omni-duplex-upstream...HEAD
git diff --check
```

Expected: the diff stat reflects the incoming upstream integration and the
five resolved hunks; `git diff --check` produces no whitespace errors. Because
this is a merge PR into the old feature branch, GitHub will also display the
upstream commits being integrated; no additional changes outside the incoming
upstream tree and the five resolved hunks should be authored by this branch.

- [ ] **Step 5: Commit the resolved merge**

Run:

```bash
git add tests/entrypoints/openai_api/test_duplex_handler.py vllm_omni/entrypoints/duplex/chat_fallback.py vllm_omni/entrypoints/duplex/runtime_bridge.py vllm_omni/entrypoints/duplex/serving.py vllm_omni/entrypoints/duplex/session_runner.py
git commit --signoff -m "chore: resolve upstream merge conflicts for Qwen3 duplex"
```

Expected: one merge-resolution commit is created on
`VAL-351-qwen3-omni-conflict-resolution`.

## Task 6: Publish the Fork Pull Request

- [ ] **Step 1: Push the new branch**

Run:

```bash
git push -u origin VAL-351-qwen3-omni-conflict-resolution
```

- [ ] **Step 2: Create the fork PR against the existing feature branch**

Run:

```bash
gh pr create \
  --repo valendra-tech/vllm-omni \
  --base VAL-351-qwen3-omni-duplex-upstream \
  --head VAL-351-qwen3-omni-conflict-resolution \
  --title "chore: resolve upstream merge conflicts for Qwen3 duplex" \
  --body "## Summary
- Merge the current upstream/main into the Qwen3-Omni duplex feature branch
- Preserve upstream Server VAD behavior and Qwen3 chat-fallback hooks
- Keep runtime-control opt-out and adapter extra_body validation

## Validation
- No unresolved merge paths or conflict markers
- Ruff check and format checks pass
- Focused duplex and Qwen3 tests run

This PR updates the branch used by upstream PR #6372 after it is merged."
```

Expected: GitHub returns the URL for the fork PR, with base
`VAL-351-qwen3-omni-duplex-upstream` and head
`VAL-351-qwen3-omni-conflict-resolution`.
