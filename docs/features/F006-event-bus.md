# F006: Event Bus & Automation Pipeline

**Status:** Planned
**Priority:** P0 — Makes everything automatic
**Detail:** See [012-automation-pipeline](../research/012-automation-pipeline.md)

## Summary

In-process async event bus that connects all Nous systems. Agent actions emit events, handlers react automatically. No cron jobs, no manual triggers. 27 of 29 agent actions are fully automatic.

## Architecture

```
Agent actions → Event Bus → Handlers → State updates → More events
```

## Events

### Agent Events (emitted by cognitive layer)
| Event | Trigger | Data |
|-------|---------|------|
| `message_received` | User sends message | message, session_id |
| `frame_selected` | Frame engine picks frame | frame_id, confidence |
| `context_assembled` | Context engine builds context | items_loaded, tokens, latency_ms |
| `censors_checked` | Censor check during pre-action | triggered[], passed[] |
| `decision_recorded` | Brain records a decision | decision_id |
| `tool_called` | LLM uses a tool | tool, args, result, success |
| `response_sent` | Response delivered to user | tokens, items_referenced |
| `turn_completed` | Full turn finished | episode_id, duration |
| `error_occurred` | Something went wrong | type, message, context |
| `session_ended` | Conversation over | session_id |

### System Events (emitted by handlers)
| Event | Trigger | Data |
|-------|---------|------|
| `episode_completed` | Episode summarized and closed | episode_id, outcome |
| `fact_learned` | New fact stored | fact_id, content |
| `decision_reviewed` | Outcome logged for decision | decision_id, outcome |
| `censor_created` | New censor from failure | censor_id, trigger |
| `censor_triggered` | Censor caught something | censor_id, action |
| `censor_escalated` | Censor severity increased | censor_id, from, to |
| `calibration_drift` | Brier score degrading | direction, current, previous |
| `growth_report_generated` | Weekly report ready | report_id |
| `growth_alert` | Metric crossed threshold | alert message |

### Tick Events (only scheduled component)
| Event | Frequency | Purpose |
|-------|-----------|---------|
| `daily_tick` | Once/day | Memory maintenance, health checks |
| `weekly_tick` | Sundays | Growth report generation |

## Seven Automated Handlers

| Handler | Listens To | Does |
|---------|-----------|------|
| **Episode Tracker** | message_received, session_ended | Creates/closes episodes, generates LLM summaries |
| **Fact Extractor** | episode_completed | Extracts facts from episode summaries, deduplicates |
| **Outcome Detector** | tool_called, episode_completed, error_occurred | Links results back to Brain decisions |
| **Censor Creator** | decision_reviewed (failure) | Generates censors from failed decisions |
| **Calibration Updater** | decision_reviewed (every 10th) | Snapshots metrics, detects drift |
| **Growth Reporter** | weekly_tick | Full growth report with recommendations |
| **Memory Maintainer** | daily_tick | Trims episodes, retires noisy censors, flags weak procedures |

## Interface

```python
from nous.events import EventBus, Event

bus = EventBus()

# Register handler
bus.on("episode_completed", my_handler)

# Emit event (non-blocking, queued)
await bus.emit(Event(
    type="message_received",
    agent_id="nous-1",
    session_id="sess-abc",
    data={"message": "Hello"}
))
```

## Error Isolation

Handlers never crash the bus. Errors are logged and the bus continues processing. One broken handler doesn't affect others.
