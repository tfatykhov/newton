# F009: Async Subtask Engine

**Status:** Planned
**Priority:** P1
**Prerequisites:** F004 (Runtime), F006 (Event Bus)
**Estimated Effort:** 8-12 hours

## Summary

Give Nous the ability to decompose complex work into background subtasks that execute asynchronously without blocking the main conversation. Subtasks run as independent agent turns with their own sessions, report results back via events, and can execute in parallel.

## Motivation

Currently, Nous handles everything in a single synchronous tool loop. This works for simple queries (3-5 tool calls, ~30 seconds) but breaks down for:

- **Multi-step research** — "Research X from 5 sources and write a report" takes minutes
- **Parallel work** — searching web + querying memory + reading files simultaneously  
- **Long-running tasks** — code generation, analysis, data processing
- **User experience** — user is blocked waiting for the entire chain to complete

## Architecture

```mermaid
graph TB
    User -->|"Research longevity"| Chat[/chat endpoint]
    Chat --> Runner[AgentRunner]
    Runner -->|tool_use: create_subtask| TQ[Task Queue]
    Runner -->|immediate response| User
    
    TQ -->|dequeue| W1[Worker 1]
    TQ -->|dequeue| W2[Worker 2]
    
    W1 -->|web_search + web_fetch| Results1[Results]
    W2 -->|recall_deep + learn_fact| Results2[Results]
    
    Results1 -->|event: subtask_completed| EB[Event Bus]
    Results2 -->|event: subtask_completed| EB
    
    EB -->|notify| Chat
    Chat -->|push or poll| User
```

## Core Concepts

### Subtask
A self-contained unit of work with:
- **task description** — what to do (natural language prompt)
- **parent session** — which conversation spawned it
- **priority** — urgent, normal, low
- **timeout** — max execution time
- **status** — pending, running, completed, failed, cancelled
- **result** — output text when done

### Task Queue
Postgres-backed queue (no Redis/RabbitMQ needed at this scale):
- `nous_system.subtasks` table
- Workers poll for pending tasks
- At-most-once delivery via `SELECT ... FOR UPDATE SKIP LOCKED`
- Configurable concurrency (default: 2 parallel workers)

### Workers
Background asyncio tasks that:
1. Dequeue a subtask
2. Create a fresh AgentRunner turn with subtask prompt
3. Execute with full tool access (bash, web, memory)
4. Store result in DB
5. Emit `subtask_completed` event

## Database Schema

```sql
CREATE TABLE nous_system.subtasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR NOT NULL,
    parent_session_id VARCHAR NOT NULL,
    task TEXT NOT NULL,
    priority INTEGER DEFAULT 100,  -- lower = higher priority
    status VARCHAR NOT NULL DEFAULT 'pending',
    -- CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled'))
    result TEXT,
    error TEXT,
    worker_id VARCHAR,  -- which worker picked it up
    timeout_seconds INTEGER DEFAULT 120,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_subtasks_pending ON nous_system.subtasks (priority, created_at)
    WHERE status = 'pending';
CREATE INDEX idx_subtasks_parent ON nous_system.subtasks (parent_session_id);
```

## Tool Definition

```json
{
  "name": "create_subtask",
  "description": "Create a background subtask for async execution. Use for work that takes more than a few seconds, or when you want to do multiple things in parallel. Results will be available via recall or notification.",
  "input_schema": {
    "type": "object",
    "properties": {
      "task": {
        "type": "string",
        "description": "What to do (clear, specific instruction)"
      },
      "priority": {
        "type": "string",
        "enum": ["urgent", "normal", "low"],
        "description": "Execution priority (default: normal)"
      },
      "timeout": {
        "type": "integer",
        "description": "Max seconds to run (default: 120, max: 600)"
      }
    },
    "required": ["task"]
  }
}
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/subtasks` | List subtasks (filterable by session, status) |
| GET | `/subtasks/{id}` | Get subtask detail + result |
| DELETE | `/subtasks/{id}` | Cancel a pending/running subtask |

## Notification Flow

When a subtask completes:
1. Worker emits `subtask_completed` event via Event Bus
2. If Telegram bot is connected, push notification to user:
   ```
   ✅ Subtask completed: "Research longevity..."
   Result: [summary]
   ```
3. Result stored as episode in Heart (available via `recall_deep`)
4. Next conversation turn automatically has subtask results in context

## Worker Configuration

```python
# config.py
subtask_workers: int = 2          # Parallel worker count
subtask_poll_interval: float = 1.0  # Seconds between polls
subtask_default_timeout: int = 120  # Default timeout per subtask
subtask_max_timeout: int = 600      # Max allowed timeout
```

## Example Flow

**User:** "Research the top 3 AI agent frameworks, compare them, and write a summary"

**Nous (immediate response):** "I'll research that in the background. Creating subtasks..."

**Tool calls:**
```
create_subtask("Search for and summarize LangChain agent framework - features, pros, cons")
create_subtask("Search for and summarize CrewAI agent framework - features, pros, cons")  
create_subtask("Search for and summarize AutoGen agent framework - features, pros, cons")
```

**3 workers execute in parallel (~30s each instead of ~90s serial)**

**Telegram notifications:**
```
✅ Subtask 1 completed: LangChain research
✅ Subtask 2 completed: CrewAI research  
✅ Subtask 3 completed: AutoGen research
```

**User:** "Show me the comparison"
**Nous:** (recalls all 3 subtask results from memory, synthesizes comparison)

## Relationship to Other Features

- **F006 (Event Bus):** Subtask completion events flow through the bus
- **F003 (Cognitive Layer):** Subtask results stored as episodes, available in pre_turn context
- **F008 (Memory Lifecycle):** Old subtask results compress/archive over time
- **005.3 (Web Tools):** Subtasks can use web_search/web_fetch

## Implementation Phases

1. **Schema + config** — subtasks table, worker settings
2. **create_subtask tool** — enqueue work, return task ID
3. **Worker loop** — background asyncio task, dequeue + execute
4. **API endpoints** — list/get/cancel subtasks
5. **Telegram notifications** — push results to user
6. **Context integration** — subtask results in pre_turn context

## Open Questions

- Should subtasks have their own cognitive frames, or always run in `task` frame?
- Should subtask results auto-learn as facts/episodes, or just be available on demand?
- Rate limiting: max subtasks per session? Per hour?
