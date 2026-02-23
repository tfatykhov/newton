# 014: Acteon — Action Gateway for Agent Coordination

**Source:** https://github.com/penserai/acteon
**Date:** 2026-02-23
**Relevance:** Event bus architecture (F006), multi-agent coordination (F013), approval gates (v0.2.0)

## What It Is

Rust-based Action Gateway that intercepts raw actions/events and transforms them through a configurable pipeline: deduplication, throttling, routing, payload modification, grouping, and state machines.

Built by Penser AI. Open source. Production-grade with pluggable backends (Redis, Postgres, DynamoDB, ClickHouse, Elasticsearch).

## Key Capabilities

### Rule Engine
- YAML rules with CEL expression support
- Suppression, deduplication, throttling, rerouting, payload modification
- Hot reload without restarts

### Event Grouping
- Batch related events with configurable wait times and group sizes
- Consolidated notifications (e.g., 50 pod alerts → 1 grouped alert)
- Background flushing and cleanup

### State Machines
- Configurable states (e.g., firing → acknowledged → resolved)
- Automatic timeout transitions
- Fingerprint-based event correlation

### Inhibition
- Suppress dependent events when parent events are active
- Expression functions: `has_active_event()`, `get_event_state()`, `event_in_state()`
- Example: suppress pod alerts when cluster is down

### Multi-Agent Safety (from their Agent Swarm guide)
- Identity isolation per agent
- Permission control and RBAC
- Prompt injection defense at the gateway level
- Rate limiting per agent/tenant
- Approval workflows with human-in-the-loop
- Failure isolation between agents
- Full observability (Prometheus, audit trails)

## Relevance to Nous

### F006 Event Bus (v0.1.0) — Learn, Don't Adopt
Our in-process async EventBus is simpler by design (single agent, single process). But Acteon's patterns are worth studying:

- **Event grouping** — we could batch `fact_threshold_exceeded` events instead of emitting per-interval. Group by category, flush every N minutes.
- **State machines for episodes** — episode lifecycle (active → summarized → archived) is essentially a state machine. Acteon models this explicitly.
- **Inhibition** — our censor system is a form of inhibition ("suppress this action because of that condition"). Acteon's expression-based approach is cleaner.

### F013 Multi-Agent (v0.2.0) — Potential Integration
When multiple Nous agents share knowledge, Acteon could sit in front as:
- **Action deduplication** — prevent two agents from recording the same decision
- **Rate limiting** — prevent one agent from flooding the shared Brain
- **Approval gates** — human approval for cross-agent knowledge sharing
- **Audit trail** — who did what, when, to whom

### HITL Approval Gates (v0.2.0) — Pattern Reference
Their rule-based suppression model is more mature than our planned approach:
```yaml
rules:
  - name: require-approval-for-block-censors
    condition:
      field: action.metadata.severity
      eq: block
    action:
      type: suppress
      reason: "Block-severity censors require approval"
```
We could implement something similar in our guardrail engine without adopting the full gateway.

## What NOT to Do

- **Don't add Acteon as a dependency for v0.1.0** — we're a single agent, in-process event bus is sufficient
- **Don't over-engineer F006** to match Acteon's feature set — their use case (distributed systems, multi-tenant) is bigger than ours
- **Don't rewrite our event bus in Rust** — Python async is fine for our scale

## Key Architectural Insight

Acteon separates *what happens* (providers/executors) from *what's allowed to happen* (rules/gateway). This is the same separation we have with Brain (decisions) vs Guardrails (constraints), but Acteon applies it at the infrastructure level rather than the cognitive level.

Their state machine approach to event lifecycle is worth adopting conceptually for F008 Memory Lifecycle — model fact/episode/procedure states as explicit state machines with transition rules, not just if/else in code.

## Technical Notes

- Rust 1.88+, Axum HTTP server
- 20+ crates, well-organized under `crates/`
- SDKs: Rust, Python, Node.js/TypeScript, Go, Java
- Admin UI + Swagger docs included
- Simulation framework for testing (mock providers, failure injection)
- Distributed locks with consistency guarantees varying by backend
