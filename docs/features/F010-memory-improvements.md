# F010: Memory Improvements

**Status:** Spec Ready  
**Priority:** P1 — Improves memory quality, context richness, and multi-user clarity  
**Covers:** Episode Summaries, Clean Decision Descriptions, Proactive Fact Learning, User-Tagged Episodes

---

## Summary

Four targeted improvements to the Nous memory system, identified from real usage observations:

1. **Episode Summaries** — Store meaningful summaries of conversations, not just the opening prompt
3. **Clean Decision Descriptions** — Enforce concise, factual decision descriptions instead of storing full response blobs
4. **Proactive Fact Learning** — Actively extract and store useful facts during conversations
5. **User-Tagged Episodes** — Tag episodes with the user who initiated them for filtered recall

> ℹ️ Item 2 (episode deduplication / noisy episode filtering) is tracked separately.

---

## F010.1 — Episode Summaries

### Problem
Episodes currently only store the opening user message as their title/identifier. This means:
- `"Hello"` tells us nothing about what was discussed
- `"Say hello in one sentence."` repeated 4 times is pure noise
- Recall by episode content is nearly impossible
- The agent can't reflect meaningfully on past conversations

### Solution
At episode close, automatically generate a **structured summary** of the conversation and store it alongside the episode record.

### Summary Schema
```
episode.summary: {
  title: str           # Short descriptive title (5-10 words)
  summary: str         # 100-150 word prose summary of what happened
  key_points: [str]    # 3-5 bullet points of key outcomes/topics
  outcome: str         # "resolved" | "partial" | "unresolved" | "informational"
  topics: [str]        # Tags for topic-based recall (e.g., ["weather", "tools", "memory"])
}
```

### Trigger
- **On episode close** (end of conversation / session end)
- Cognitive Layer calls `heart.episodes.close(episode_id)` which triggers summary generation

### LLM Prompt (summary generation)
```
Given the following conversation transcript, generate a structured summary.

Transcript:
{full_transcript}

Return JSON:
{
  "title": "<5-10 word descriptive title>",
  "summary": "<100-150 word prose summary>",
  "key_points": ["<point 1>", "<point 2>", ...],
  "outcome": "<resolved|partial|unresolved|informational>",
  "topics": ["<topic1>", "<topic2>", ...]
}
```

### Recall Improvement
- `recall_deep` can now match on summary text, key points, and topics
- Episode search becomes semantically meaningful
- Reflection loops (F008) have richer input

### Implementation Notes
- Summary generated via a lightweight LLM call at session end
- Store as JSONB column on `heart.episodes` table: `summary JSONB`
- If transcript is empty/trivial (< 3 turns), skip summary or store a minimal one
- Max transcript length passed to LLM: 8,000 tokens (truncate from middle if needed)

---

## F010.3 — Clean Decision Descriptions

### Problem
`record_decision` is being called with full response blobs as the `description` field — e.g., an entire markdown status table. This means:
- Decision records are bloated and unreadable
- The Brain can't meaningfully index or search decisions
- Calibration and deliberation loops get polluted with noise
- Only 1 out of 15 stored decisions is actually a real decision

### Solution
Enforce a **strict format contract** on `record_decision.description`:

```
description: str
  - Max 120 characters
  - Must be a factual, present-tense statement of what was decided
  - NOT a question, NOT a full response, NOT a markdown block
  - Examples:
    ✅ "Use direct Anthropic API calls instead of Claude Agent SDK"
    ✅ "Store episode summaries as JSONB on close"
    ✅ "Tag all episodes with the initiating user"
    ❌ "Here's my current status! 🧠 ..."
    ❌ "Now let me write the full spec:"
```

### Enforcement Layers

#### Layer 1: Cognitive Layer Validation (pre-tool-call)
Before calling `record_decision`, the Cognitive Layer checks:
- `len(description) <= 120`
- Description does not start with phrases like `"Here's"`, `"Now let me"`, `"I'll"`, `"Let me"`
- If check fails → LLM rewrites description to a clean version before calling tool

#### Layer 2: Heart/Brain Validation (on write)
- `brain.decisions.record()` enforces max 200 chars on description (hard limit)
- Logs a warning if description looks like a response blob (heuristic: contains markdown headers or emoji at start)

#### Layer 3: Censor
- Create a censor with `trigger_pattern: "^(Here's|Now let me|I'll|Let me|Here is)"` on `decision_description`
- Action: `warn` — reminds the agent to rewrite before storing

### Rewrite Prompt (when description fails validation)
```
The following text was intended as a decision description but is not concise enough.
Rewrite it as a single factual statement of max 120 characters describing what was decided.

Original: {description}

Rewritten decision:
```

---

## F010.4 — Proactive Fact Learning

### Problem
Facts are rarely stored proactively. After many conversations, only 3 facts exist:
- Tim prefers Celsius
- GitHub PAT
- API call method

Useful context is being lost: user roles, project preferences, team structure, recurring patterns, tool preferences, etc.

### Solution
At the **end of every episode**, run a **fact extraction pass** over the conversation summary and transcript to identify learnable facts.

### Fact Extraction Trigger
- Runs after episode summary is generated (F010.1)
- Input: episode summary + key_points + transcript (last 2,000 tokens)
- Output: list of candidate facts

### LLM Prompt (fact extraction)
```
Review the following conversation summary and extract any facts worth remembering long-term.

Focus on:
- User preferences (tools, formats, units, communication style)
- Project/system facts (architecture, constraints, conventions)
- People facts (roles, names, relationships)
- Rules or recurring patterns observed

Summary: {summary}
Key Points: {key_points}

Return JSON array (empty array if nothing worth storing):
[
  {
    "subject": "<who/what the fact is about>",
    "content": "<the fact, stated clearly>",
    "category": "<preference|technical|person|tool|concept|rule>",
    "confidence": <0.0-1.0>,
    "tags": ["<tag1>", "<tag2>"]
  }
]

Only include facts that would be genuinely useful to remember across future conversations.
Skip transient, trivial, or already-known information.
```

### Deduplication
- Before storing each candidate fact, run `recall_deep` with the fact content
- If similarity > 0.90 with an existing fact → skip (confirm existing instead)
- If similarity 0.80–0.90 → LLM checks if semantically equivalent
- Below 0.80 → store as new fact

### Guardrail
- Max **5 new facts per episode** to prevent over-accumulation
- Facts with confidence < 0.6 are not auto-stored (flagged for review instead)
- Sensitive facts (PATs, passwords, PII) follow existing censor rules

### Example Facts That Should Have Been Stored
```
subject: "Tim"
content: "Tim is the primary user of the Nous agent"
category: "person"

subject: "Nous project"
content: "Emerson is the developer/creator of the Nous project"
category: "person"

subject: "Tim"
content: "Tim uses Silver Spring, MD as his location for weather queries"
category: "preference"

subject: "Nous memory"
content: "Episode deduplication is being handled separately from other memory improvements"
category: "technical"
```

---

## F010.5 — User-Tagged Episodes

### Problem
Episodes don't record *who* initiated the conversation. This means:
- Can't distinguish Tim's sessions from Emerson's sessions
- `recall_deep` can't filter episodes by user
- Multi-user context bleeds together
- No per-user memory isolation or personalization

### Solution
Tag every episode with the **initiating user's identifier** at episode creation.

### Schema Change
Add `user_id` to `heart.episodes`:
```sql
ALTER TABLE heart.episodes
  ADD COLUMN user_id VARCHAR(100),
  ADD COLUMN user_display_name VARCHAR(100);

CREATE INDEX idx_episodes_user_id ON heart.episodes(user_id);
```

### Episode Creation Update
When `heart.episodes.start()` is called, pass user context:
```python
episode_id = heart.episodes.start(
    opening_message=message,
    frame=frame_name,
    user_id=context.user_id,           # e.g. "tim", "emerson"
    user_display_name=context.user_name  # e.g. "Tim", "Emerson"
)
```

### Recall Filter
`recall_deep` gains an optional `user_id` filter:
```python
recall_deep(query="weather", user_id="tim")
# → returns only Tim's episodes matching "weather"
```

### Working Memory
Active user identity stored in working memory at session start:
```
key: "current_user_id"     value: "tim"
key: "current_user_name"   value: "Tim"
```

### Context Engine Integration
- When assembling episode context, prefer episodes from the **current user** (higher relevance score)
- Cross-user episodes still surfaced when highly relevant, but ranked lower
- User tag shown in episode recall results: `[Tim] Checked weather for Silver Spring, MD`

### Privacy Boundary
- Each user's facts/preferences remain accessible across users (shared agent knowledge)
- Episode *content* is user-tagged but not access-restricted (single-agent, trusted users)
- Future: per-user fact namespacing can be added in F013 (Multi-Agent)

---

## Implementation Order

| Step | Feature | Dependency |
|------|---------|------------|
| 1 | F010.5 — User tagging | None — pure schema + session plumbing |
| 2 | F010.1 — Episode summaries | F010.5 (user tag on summary) |
| 3 | F010.3 — Clean decision descriptions | None — validation + censor |
| 4 | F010.4 — Proactive fact learning | F010.1 (uses episode summary as input) |

---

## Database Changes Summary

```sql
-- F010.1: Episode summaries
ALTER TABLE heart.episodes
  ADD COLUMN summary JSONB;
  -- { title, summary, key_points, outcome, topics }

-- F010.5: User tagging
ALTER TABLE heart.episodes
  ADD COLUMN user_id VARCHAR(100),
  ADD COLUMN user_display_name VARCHAR(100);

CREATE INDEX idx_episodes_user_id ON heart.episodes(user_id);
CREATE INDEX idx_episodes_summary_gin ON heart.episodes USING GIN(summary);
```

---

## Acceptance Criteria

### F010.1 — Episode Summaries
- [ ] Episodes closed after a conversation have a non-null `summary` JSONB field
- [ ] Summary contains `title`, `summary`, `key_points`, `outcome`, `topics`
- [ ] `recall_deep` matches on summary text and topics
- [ ] Trivial sessions (< 3 turns) produce a minimal summary or skip gracefully

### F010.3 — Clean Decision Descriptions
- [ ] `record_decision` rejects descriptions > 200 chars at the API layer
- [ ] Cognitive Layer rewrites non-compliant descriptions before calling tool
- [ ] Censor fires on description patterns starting with response-blob phrases
- [ ] All new decisions stored have clean, factual descriptions ≤ 120 chars

### F010.4 — Proactive Fact Learning
- [ ] After each episode close, fact extraction runs automatically
- [ ] At least 1 new useful fact stored per substantive conversation
- [ ] Duplicate facts are not re-stored (deduplication working)
- [ ] Max 5 facts per episode enforced
- [ ] Facts with confidence < 0.6 not auto-stored

### F010.5 — User-Tagged Episodes
- [ ] All new episodes have `user_id` set at creation
- [ ] `recall_deep` supports `user_id` filter parameter
- [ ] Working memory contains `current_user_id` at session start
- [ ] Episode recall results display user tag in output
- [ ] Context Engine boosts relevance for current-user episodes

---

## Related Features
- **F002** — Heart Module (memory system this improves)
- **F003** — Cognitive Layer (triggers summary + fact extraction)
- **F006** — Event Bus (episode close event triggers F010.1 + F010.4)
- **F008** — Memory Lifecycle (episode archiving uses summaries from F010.1)
