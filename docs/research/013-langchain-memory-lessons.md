# 013: Lessons from LangChain Agent Builder's Memory System

**Source:** https://blog.langchain.com/how-we-built-agent-builders-memory-system
**Date:** 2026-02-22
**Relevance:** Direct comparison — they built production agent memory, we're building Nous

## Their Architecture

- Memory as virtual files (Postgres-backed, exposed as filesystem)
- COALA paper taxonomy: procedural (AGENTS.md), semantic (skills), episodic (skipped)
- Agent edits its own instructions through corrections — memory builds iteratively
- Human-in-the-loop for all memory writes
- Deep Agents harness handles summarization, tool offloading, planning

## What We Should Adopt

### 1. File-View for LLM Consumption

**Problem:** LLMs read markdown files better than structured JSON. Our Heart stores everything in normalized Postgres tables — correct for programmatic access, but the LLM consuming context doesn't need relational structure.

**Action:** ContextEngine should render Heart data as markdown sections, not JSON dumps. The `_format_*` methods in `004-cognitive-layer.md` already do this, but we should be intentional: the output format is a flat document, not a data structure. Test with actual LLM comprehension, not just "does it serialize."

### 2. Active Compaction, Not Just Lifecycle

**Problem:** Their hardest lesson — agents accumulate specific examples instead of generalizing. "My email assistant started listing all specific vendors to ignore instead of updating itself to ignore all cold outreach."

**Our gap:** F008 Memory Lifecycle has `trim` and `archive` but no `generalize`. We have `facts.supersede()` but nothing triggers it automatically.

**Action:** Add to F008 or Monitor engine:
- When fact count in a domain exceeds threshold (e.g., 10+ facts about same topic), trigger compaction
- Compaction = LLM call to merge N specific facts into 1-2 general rules
- This is an Event Bus handler (F006): `on_fact_threshold_exceeded → compact_facts`
- Track compaction events for growth metrics

### 3. Validation Feedback Loop on Memory Writes

**Problem:** Their agents generated invalid files (bad schemas, wrong formats). They added validation + error feedback to LLM.

**Our gap:** Heart.learn(), Heart.store_procedure() etc. validate Pydantic schemas but don't validate semantic quality. A fact like "Redis is a database" and "Redis is a cache" can both be stored without checking contradiction.

**Action:** 
- Heart.facts.learn() should check for contradicting existing facts (embedding similarity > 0.9 with different content)
- Surface contradictions back to the Cognitive Layer for resolution
- Heart.procedures.store() should check for duplicate procedures in same domain
- Log validation issues as events for Monitor to assess

### 4. End-of-Session Reflection

**Problem:** They explicitly prompt the agent to review and update memory at end of work. Passive episode recording misses learnings.

**Our gap:** CognitiveLayer.end_session() just closes the episode. No reflection.

**Action:** Add reflection step to end_session():
```python
async def end_session(self, agent_id, session_id):
    # 1. Close episode (existing)
    # 2. NEW: Generate reflection summary
    #    - What was the main task?
    #    - What went well / poorly?
    #    - What facts were learned?
    #    - Should any procedures be created/updated?
    # 3. Feed reflection to Heart for fact extraction
    # 4. Emit "session_reflected" event
```
This IS an LLM call (unlike the rest of the cognitive layer), but it's once per session, not per turn. Worth the cost.

### 5. Approval Gates for Behavioral Memory

**Problem:** They require human approval for all memory edits. Attack surface: prompt injection → agent writes malicious procedures.

**Our gap:** Zero approval gates. Heart accepts all writes.

**Action for v0.1.0:** 
- Censors with severity="block" require explicit confirmation (not auto-created)
- Procedures that modify agent behavior flagged for review
- Facts from external sources marked as `unconfirmed` (existing field, but not enforced)
- Full HITL is a v0.2.0 feature

## What We Already Do Better

| Aspect | LangChain | Nous |
|--------|-----------|------|
| Episodic memory | Skipped | Full implementation (Heart episodes) |
| Decision intelligence | None | Brain (record, deliberate, calibrate) |
| Cognitive frames | None | 6 frames with pattern matching |
| Censors/constraints | None | Active censor system with escalation |
| Compaction | Manual prompting | Automated lifecycle (F008) — but needs generalization |
| Memory types | 2 (procedural, semantic) | 5 (episodic, semantic, procedural, censors, working) |
| Self-assessment | None | Monitor engine with surprise levels |

## Key Quote

> "The hardest part of building an agent that could remember things is prompting. In almost all cases where the agent was not performing well, the solution was to improve the prompt."

This validates our approach: build the machinery (frames, monitor, lifecycle) so prompting handles less. But it's also a warning — our system prompt construction in ContextEngine is going to need a LOT of iteration.

## Impact on Specs

- **004-cognitive-layer:** Add reflection step to `end_session()` (LLM call, once per session)
- **005-runtime:** No changes needed
- **F006 Event Bus:** Add `fact_threshold_exceeded` event type for compaction triggers
- **F008 Memory Lifecycle:** Add `generalize` operation alongside trim/archive
- **Future:** HITL approval gates for behavioral memory writes (v0.2.0)
