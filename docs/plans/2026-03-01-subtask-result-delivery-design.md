# Subtask Result Delivery — Design

## Problem

Subtask results are stored in `heart.subtasks` but never enter the agent's memory system. The parent agent can only see results by calling `list_tasks` — and even then, the results don't become part of episodes or facts for future recall.

## Goal

Automatically deliver subtask results (success and failure) to the parent session so they:
1. Appear in the parent agent's context without requiring a tool call
2. Become part of the parent session's episode → summarized → facts extracted
3. Are discoverable via `recall_deep` in future conversations

## Design

### Principle: No duplication

Subtask sessions run with `skip_episode=True` — they use the full cognitive pipeline (recall, tools, framing) but produce no episodes or facts. Results flow to the parent session only, where the existing episode pipeline captures them naturally.

### Flow: Success

1. Worker calls `run_turn(skip_episode=True)`
2. LLM executes, result stored in subtasks table
3. Parent session's next `pre_turn` detects undelivered completed subtask
4. Result injected into parent's system prompt context
5. Subtask marked `delivered=True`
6. Parent episode captures the result when session ends

### Flow: Failure / Timeout

1. Worker catches error or timeout, stores in subtasks table
2. No episode created (skip_episode=True), no garbage
3. Parent session's next `pre_turn` detects undelivered failed subtask
4. Error injected into parent's system prompt context
5. Subtask marked `delivered=True`
6. Parent agent decides whether to retry

### Changes

| Component | Change |
|-----------|--------|
| `Subtask` model + migration | Add `delivered BOOLEAN DEFAULT FALSE` column |
| `runner.run_turn()` | Add `skip_episode: bool = False` param, pass to `pre_turn` |
| `cognitive.pre_turn()` | Accept `skip_episode`, skip episode creation when True |
| `subtask_worker._execute_subtask()` | Pass `skip_episode=True` |
| `cognitive.pre_turn()` or `context.build()` | Inject undelivered subtask results into context |
| `heart.subtasks` | Add `get_undelivered(session_id)` and `mark_delivered(ids)` methods |

### Context injection format

```
=== Completed Subtasks ===
[subtask-abc12345] Task: Research X
Result: Y and Z were found...

=== Failed Subtasks ===
[subtask-def45678] Task: Analyze Y
Error: TimeoutError: timed out after 120s
```

### What we're NOT doing

- No new event bus handler — results flow through existing episode pipeline
- No episode creation for subtask sessions — avoids duplication
- No separate fact extraction from subtask results — parent episode handles it
- No `end_session` call from worker — nothing to end
