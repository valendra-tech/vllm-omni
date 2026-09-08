# Qwen3-Omni Duplex Conflict Resolution

## Context

PR #6372 is based on `valendra-tech/vllm-omni` branch
`VAL-351-qwen3-omni-duplex-upstream`. The branch is currently marked
`CONFLICTING` against the latest `vllm-project/vllm-omni/main`. The conflicts
are caused by upstream duplex and Server VAD changes that landed after the
branch was created, not by the Qwen3 documentation recipe.

The resolution will be delivered as a pull request in the fork, targeting
the existing feature branch. Merging that PR will update the head branch used
by upstream PR #6372 without force-pushing it.

## Goals

- Create a new fork branch from `VAL-351-qwen3-omni-duplex-upstream`.
- Merge the current `upstream/main` into that branch.
- Resolve all content conflicts while preserving both upstream behavior and
  Qwen3-Omni behavior.
- Keep the original feature branch and upstream PR history intact.
- Verify that no conflict markers remain and run focused duplex/Qwen3 tests.

## Non-goals

- No unrelated refactoring.
- No direct force-push to `VAL-351-qwen3-omni-duplex-upstream`.
- No new upstream PR; the resulting PR is an integration PR in the fork.

## Conflict Decisions

### `tests/entrypoints/openai_api/test_duplex_handler.py`

Keep both independent regression tests: the upstream response-history-order
test and the Qwen3 serving-adapter auto-response hook test.

### `vllm_omni/entrypoints/duplex/chat_fallback.py`

Keep the current upstream error-response handling and retain the Qwen3
`on_turn_request_issued` callback after a successful chat request has been
created. The callback must not run for an error response.

### `vllm_omni/entrypoints/duplex/runtime_bridge.py`

Retain the current upstream serving-adapter eligibility check and combine it
with the Qwen3 adapter's `supports_runtime_control = False` behavior. Generic
native adapters continue to use runtime control; Qwen3 remains on the chat
fallback path and must not emit native engine control RPCs.

### `vllm_omni/entrypoints/duplex/serving.py`

Keep the upstream local `runtime_adapter` selection and invoke
`validate_client_extra_body` before `prepare_runtime_config` for the selected
adapter. The validation must not run for sessions without a serving adapter.

### `vllm_omni/entrypoints/duplex/session_runner.py`

Keep the latest upstream Server VAD flow, including sample-rate validation,
turn-based VAD handling, backpressure, and commit behavior. Reapply the
Qwen3-compatible float32 input conversion only in the non-native, non-turn-
based-VAD path. Preserve the Qwen3 input format aliases and sample-rate
propagation without bypassing the newer VAD validation.

## Validation

- Confirm the merge has no unmerged paths or conflict markers.
- Run formatting and lint checks for changed Python files.
- Run the focused duplex handler tests, Qwen3 integration/adapter/policy/data
  plane tests, input-audio tests, and stream-finish/sample-rate tests.
- Inspect the final diff against
  `VAL-351-qwen3-omni-duplex-upstream` to ensure the integration PR contains
  only conflict-resolution changes.

## Delivery

Push the branch as `VAL-351-qwen3-omni-conflict-resolution` to
`valendra-tech/vllm-omni` and create a fork PR with base
`VAL-351-qwen3-omni-duplex-upstream`.
