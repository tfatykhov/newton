# F011 — Skill Discovery & Auto-Activation

> **Status:** Planned
> **Priority:** P1
> **Depends on:** F003 (Cognitive Layer), F002 (Heart — Procedures)
> **Estimated effort:** ~4-6 hours

## Problem

Nous has skills (code-review, serper, web tools) stored as files in the workspace, but:
- **No indexing** — skills aren't registered anywhere the cognitive loop can find them
- **No auto-activation** — the agent must manually recall skill existence from facts
- **No matching** — there's no mechanism to match incoming tasks to relevant skills
- **No loading** — skill context (instructions, tool paths, patterns) isn't injected into working memory

Currently, skill awareness depends entirely on facts learned via `learn_fact`. If the fact isn't recalled (wrong query, low similarity), the skill is invisible.

## Solution

Use the existing **Procedure** system in Heart as the skill registry. Procedures already have:
- `name`, `domain`, `description`
- `goals` (upper fringe — what it's for)
- `core_patterns`, `core_tools`, `core_concepts` (middle — transferable knowledge)
- `implementation_notes` (lower fringe — specific details)
- `activation_count`, `success_count`, `failure_count` (effectiveness tracking)
- `tags` and `embedding` for semantic search

Skills are procedures that come from the filesystem rather than being learned from experience.

## Architecture

```mermaid
graph TD
    A[User Message] --> B[SENSE]
    B --> C[FRAME Selection]
    C --> D[RECALL]
    D --> D1[Decisions]
    D --> D2[Facts]
    D --> D3["Skills/Procedures (NEW)"]
    D --> D4[Episodes]
    D3 --> E[Working Memory]
    E --> F[DELIBERATE]
    F --> G[ACT — with skill context]
```

### Skill Discovery happens at the RECALL stage

This is the natural fit because:
1. **FRAME** has already classified the task type (task, decision, debug, conversation, creative)
2. **RECALL** is where we search all memory types — adding procedures/skills is a natural extension
3. Skills load into **Working Memory** as items with high relevance
4. The **Context Engine** already has a procedures budget slot (priority 7)

### Registration Flow

```mermaid
sequenceDiagram
    participant FS as Filesystem
    participant Idx as Skill Indexer
    participant DB as Heart.Procedures
    participant CL as Cognitive Layer

    Note over FS: skills/serper/SKILL.md
    Note over FS: skills/code-review/SKILL.md

    FS->>Idx: Scan workspace/skills/
    Idx->>Idx: Parse SKILL.md (name, description, tools, triggers)
    Idx->>DB: store_procedure() for each skill
    Note over DB: Procedures table with embeddings

    Note over CL: During RECALL stage
    CL->>DB: search_procedures(query=user_input, frame=current_frame)
    DB-->>CL: Matched skills (by embedding similarity + frame domain)
    CL->>CL: Load into working memory + context
```

## Design

### 1. Skill Manifest (SKILL.md frontmatter)

Skills already have `SKILL.md` files. Add structured frontmatter:

```yaml
---
name: serper-search
description: Google search via Serper.dev API
domain: research          # maps to procedure domain
triggers:                 # keywords/phrases that should activate this skill
  - web search
  - google
  - find online
  - research
  - look up
frames:                   # which cognitive frames auto-activate this skill
  - task
  - debug
tools:                    # tools this skill provides/enhances
  - web_search
  - web_fetch
requires:                 # dependencies
  - SERPER_API_KEY
---
```

### 2. Skill Indexer (`nous/skills/indexer.py`)

New module that:
- Scans `NOUS_WORKSPACE_DIR/skills/` on startup
- Parses SKILL.md frontmatter
- Creates/updates procedure records in Heart via `store_procedure()`
- Generates embeddings from description + triggers for semantic matching
- Re-indexes on `skill_added` / `skill_removed` events

```python
class SkillIndexer:
    """Index workspace skills as procedures in Heart."""

    async def index_all(self, workspace_dir: str) -> list[ProcedureDetail]:
        """Scan skills directory and register as procedures."""
        skills_dir = Path(workspace_dir) / "skills"
        if not skills_dir.exists():
            return []

        results = []
        for skill_dir in skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue

            manifest = self._parse_manifest(skill_md)
            procedure = await self._register_skill(manifest)
            results.append(procedure)

        return results

    def _parse_manifest(self, path: Path) -> SkillManifest:
        """Parse SKILL.md frontmatter into SkillManifest."""
        ...

    async def _register_skill(self, manifest: SkillManifest) -> ProcedureDetail:
        """Create or update procedure from skill manifest."""
        input = ProcedureInput(
            name=manifest.name,
            domain=manifest.domain,
            description=manifest.description,
            goals=manifest.triggers,          # upper fringe = when to activate
            core_tools=manifest.tools,        # middle = what tools it provides
            core_patterns=manifest.triggers,  # middle = activation patterns
            implementation_notes=[str(manifest.path)],  # lower fringe = where to find it
            tags=["skill", "filesystem"] + manifest.frames,
        )
        return await self._heart.store_procedure(input)
```

### 3. Cognitive Layer Integration (RECALL stage)

In `CognitiveLayer.pre_turn()`, after frame selection:

```python
# During RECALL — search procedures/skills by input + frame
procedures = await self._heart.search_procedures(
    query=user_input,
    domain=frame.frame_id,  # frame-based filtering
    limit=3,
)

# Load matched skills into working memory
for proc in procedures:
    await self._heart.working_memory.load_item(
        session_id=session_id,
        item=WorkingMemoryItem(
            ref_id=proc.id,
            ref_type="procedure",
            content=proc.description,
            relevance=similarity_score,
        ),
    )
```

### 4. Context Engine (procedures budget)

The context engine already has a procedures slot at priority 7. Currently unused because no procedures exist. Once skills are indexed as procedures, they'll automatically flow into context:

```python
# context.py — already implemented, just needs data
# 7. Procedures (F11: skills auto-surfaced here)
if budget.procedures > 0 and "procedure" not in skip_types:
    procedures = await self._heart.search_procedures(...)
    ...
```

### 5. Frame-Based Auto-Activation

Map frames to skill domains:

| Frame | Auto-activate skills with domain |
|-------|----------------------------------|
| task | Any matching domain |
| debug | debugging, code-review |
| decision | architecture, process |
| conversation | (none — lightweight) |
| creative | writing, design |

Skills with matching `frames` in their manifest are boosted in recall.

### 6. Effectiveness Tracking

When a skill is activated and the turn succeeds/fails:
- `success_count++` or `failure_count++` via `record_procedure_outcome()`
- Skills with low effectiveness get deprioritized in recall
- Links to episodes via `episode_procedures` table (currently empty)

## Database Impact

**No new tables.** Uses existing:
- `heart.procedures` — skill registry (currently 0 rows)
- `heart.episode_procedures` — skill-episode links (currently 0 rows)
- `heart.working_memory.items` — loaded skill context (currently always empty)

All three tables are built and waiting for data.

## Files Changed

| File | Change |
|------|--------|
| `nous/skills/__init__.py` | New package |
| `nous/skills/indexer.py` | Skill indexer (~150 lines) |
| `nous/skills/manifest.py` | SKILL.md parser (~60 lines) |
| `nous/cognitive/layer.py` | Add procedure search to RECALL stage |
| `nous/main.py` | Run skill indexer on startup |
| `tests/test_skill_indexer.py` | Tests (~200 lines) |

**Estimated:** ~410 lines new code, ~50 lines modified

## Relationship to Other Features

- **F002 Heart (Procedures)** — Skills ARE procedures. Same storage, same API.
- **F003 Cognitive Layer** — Skill discovery slots into the existing RECALL stage.
- **F005 Context Engine** — Procedures budget already allocated at priority 7.
- **F008 Memory Lifecycle** — Skills from filesystem are "permanent" (re-indexed on startup). Learned procedures from experience follow normal lifecycle.
- **v0.2.0 F011 K-Line Learning** — Learned procedures complement filesystem skills. Over time, Nous discovers its own "skills" from repeated patterns.

## Two-Pass RECALL: Merge & Ranking

Skill discovery uses a two-pass RECALL with merged scoring:

### Pass 1: Semantic Search
Query procedures by embedding similarity to user input. Returns `(skill_id, semantic_score)`.

### Pass 2: Frame-Domain Filter
Query procedures where `domain` matches current frame or `tags` contain the frame ID. Returns `(skill_id, domain_match=true)`.

### Merge Strategy
```python
# Combine scores, deduplicate by name
combined = {}
for skill_id, score in semantic_results:
    combined[skill_id] = score
for skill_id, _ in domain_results:
    combined[skill_id] = combined.get(skill_id, 0.0) + 0.2  # frame boost

# Sort by score, take top 3
top_skills = sorted(combined.items(), key=lambda x: -x[1])[:3]
```

- Frame-domain match adds `+0.2` score boost
- Deduplicate by skill `name`
- Top K=3 (fits within procedures context budget)

## Working Memory Content Format

When a skill loads into working memory, use a **compact format** — not full SKILL.md:

```
[Skill: serper-search]
Domain: research
Tools: web_search, web_fetch
Triggers: web search, google, find online, research
---
{first H2 section from SKILL.md — the "when to use" prose}
```

Full SKILL.md stays on disk, fetchable via `read_file` if the agent needs implementation details. This keeps 3 simultaneous skills within ~300-400 tokens total.

## Monitor Stage: Effectiveness Wiring

The MONITOR stage (post-turn) evaluates skill effectiveness:

```python
# In CognitiveLayer.post_turn(), after outcome assessment
for skill_id in turn_context.activated_skills:
    if turn_succeeded:
        await self._heart.record_procedure_outcome(skill_id, success=True)
    elif turn_failed:
        await self._heart.record_procedure_outcome(skill_id, success=False)
```

This wires into the existing `success_count` / `failure_count` columns. Skills with `failure_count / (success_count + failure_count) > 0.6` after 5+ activations get a `-0.1` score penalty in RECALL.

## Re-indexing Strategy

**v1: Startup-only.** SkillIndexer runs once on application start. New skills added mid-session require a restart.

**Future options (not in v1):**
- Manual trigger via `/index-skills` Telegram command
- File watcher (fsnotify) on the skills directory
- Event-driven: `skill_added` bus event triggers re-index

This is an explicit design constraint, not an oversight.

## `requires` Field Semantics

The `requires` field lists **environment variables** that must be set for the skill to function:

```yaml
requires:
  - SERPER_API_KEY
  - BRAVE_SEARCH_API_KEY
```

At index time, SkillIndexer checks `os.environ` for each required var:
- **All present** → procedure stored with `active=True`
- **Any missing** → procedure stored with `active=False`, logged as warning
- Inactive procedures are excluded from RECALL search

This prevents the agent from loading a skill into working memory, attempting to use it, and failing because a required API key is missing.

## Open Questions

1. Should the agent be able to create new skill files from learned procedures (the reverse flow)?
2. How to handle skill conflicts (two skills claim the same domain/triggers)?
3. Should effectiveness penalties decay over time (skill improved but old failures drag it down)?
