# 004: Cognitive Layer — The Nous Loop

**Status:** Ready to Build
**Priority:** P0 — Orchestrator that makes Brain + Heart think together
**Estimated Effort:** 10-14 hours
**Prerequisites:** 001-postgres-scaffold (merged), 002-brain-module (merged), 003-heart-module (merged)
**Feature Specs:** [F003-cognitive-layer.md](../features/F003-cognitive-layer.md), [F005-context-engine.md](../features/F005-context-engine.md)

## Objective

Implement the Cognitive Layer — the orchestrator that wires Brain and Heart into a thinking loop. Seven phases per interaction: Sense → Frame → Recall → Deliberate → Act → Monitor → Learn.

This is NOT an LLM wrapper. It's a set of Python functions that hook into agent execution (via the Claude Agent SDK or any similar harness). The LLM handles the "Act" phase. The Cognitive Layer handles everything else.

After this phase, any agent runtime can:
```python
from nous.cognitive import CognitiveLayer

cognitive = CognitiveLayer(brain, heart, settings)
context = await cognitive.pre_turn(agent_id, session_id, user_input)
# ... LLM generates response using context.system_prompt ...
await cognitive.post_turn(agent_id, session_id, response, tool_results)
```

## Architecture

```
CognitiveLayer (nous/cognitive/layer.py)
├── FrameEngine (nous/cognitive/frames.py)
│   ├── select()          → Pattern-match cognitive frame from input
│   ├── get()             → Fetch frame by ID
│   └── list_frames()     → All frames for agent
│
├── ContextEngine (nous/cognitive/context.py)
│   ├── build()           → Assemble context within token budget
│   ├── refresh_needed()  → Check if context is stale
│   └── expand()          → Load full detail for a memory
│
├── DeliberationEngine (nous/cognitive/deliberation.py)
│   ├── start()           → Record intent, return decision_id
│   ├── think()           → Capture micro-thought
│   └── finalize()        → Update decision with outcome
│
├── MonitorEngine (nous/cognitive/monitor.py)
│   ├── assess()          → Evaluate action result
│   └── learn()           → Extract lessons, create censors if needed
│
└── Schemas (nous/cognitive/schemas.py)
    ├── TurnContext        → Output of pre_turn (system_prompt, frame, decision_id, metadata)
    ├── TurnResult         → Input to post_turn (response text, tool calls, errors)
    ├── Assessment         → Monitor output (surprise_level, censor_candidates)
    └── ContextBudget      → Token allocation per priority layer
```

## File-by-File Specification

### 1. `nous/cognitive/schemas.py` (~120 lines)

Pydantic models for the cognitive layer's inputs and outputs.

```python
from pydantic import BaseModel, Field
from enum import Enum

class FrameType(str, Enum):
    TASK = "task"
    QUESTION = "question"
    DECISION = "decision"
    CREATIVE = "creative"
    CONVERSATION = "conversation"
    DEBUG = "debug"

class FrameSelection(BaseModel):
    """Result of frame selection."""
    frame_id: str
    frame_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    match_method: str  # "pattern" or "default"
    description: str | None = None
    default_category: str | None = None
    default_stakes: str | None = None
    questions_to_ask: list[str] = Field(default_factory=list)

class ContextBudget(BaseModel):
    """Token allocation for context assembly."""
    total: int = 8000
    identity: int = 500
    censors: int = 300
    frame: int = 500
    working_memory: int = 700
    decisions: int = 2000
    facts: int = 1500
    procedures: int = 1500
    episodes: int = 1000

    @classmethod
    def for_frame(cls, frame_id: str) -> "ContextBudget":
        """Return frame-adaptive budget."""
        budgets = {
            "conversation": cls(total=3000, decisions=500, facts=500, procedures=0, episodes=0),
            "question": cls(total=6000, decisions=1000, facts=1500, procedures=500, episodes=500),
            "task": cls(total=8000),
            "decision": cls(total=12000, decisions=3000, facts=2000, procedures=2000, episodes=1000),
            "creative": cls(total=6000, censors=100, decisions=1000, facts=1500, procedures=500, episodes=500),
            "debug": cls(total=10000, decisions=1500, facts=1000, procedures=2500, episodes=1000),
        }
        return budgets.get(frame_id, cls())

class ContextSection(BaseModel):
    """A section of assembled context."""
    priority: int  # 1-8 (1=highest)
    label: str
    content: str
    token_estimate: int  # rough char/4 estimate

class TurnContext(BaseModel):
    """Output of pre_turn — everything the agent needs."""
    system_prompt: str
    frame: FrameSelection
    decision_id: str | None = None  # Set if frame is 'decision' or 'task'
    active_censors: list[str] = Field(default_factory=list)
    context_token_estimate: int = 0
    recalled_decision_ids: list[str] = Field(default_factory=list)
    recalled_fact_ids: list[str] = Field(default_factory=list)

class ToolResult(BaseModel):
    """Representation of a tool call result for post_turn."""
    tool_name: str
    arguments: dict = Field(default_factory=dict)
    result: str | None = None
    error: str | None = None
    duration_ms: int | None = None

class TurnResult(BaseModel):
    """Input to post_turn — what happened during the turn."""
    response_text: str
    tool_results: list[ToolResult] = Field(default_factory=list)
    error: str | None = None
    duration_ms: int | None = None

class Assessment(BaseModel):
    """Monitor engine output."""
    decision_id: str | None = None
    intended: str | None = None
    actual: str
    surprise_level: float = Field(ge=0.0, le=1.0, default=0.0)
    censor_candidates: list[str] = Field(default_factory=list)
    facts_extracted: int = 0
    episode_recorded: bool = False
```

### 2. `nous/cognitive/frames.py` (~130 lines)

Frame selection via pattern matching against the `nous_system.frames` table.

```python
class FrameEngine:
    """Selects cognitive frame for each interaction."""

    def __init__(self, database: Database, settings: Settings) -> None:
        self._db = database
        self._settings = settings
        self._default_frame_id = "conversation"

    async def select(
        self, agent_id: str, input_text: str, session: AsyncSession | None = None
    ) -> FrameSelection:
        """Select the best cognitive frame for this input.

        Algorithm:
        1. Tokenize input into lowercase words
        2. For each frame, count how many activation_patterns match any word
        3. Frame with most matches wins (ties broken by order: decision > task > debug > question > creative > conversation)
        4. If no matches, return default frame (conversation)

        No LLM call — this is pure pattern matching for speed.
        """

    async def get(
        self, frame_id: str, agent_id: str, session: AsyncSession | None = None
    ) -> FrameSelection:
        """Fetch a specific frame by ID."""

    async def list_frames(
        self, agent_id: str, session: AsyncSession | None = None
    ) -> list[FrameSelection]:
        """List all frames for an agent."""
```

**Implementation notes:**
- Query `nous_system.frames` WHERE `agent_id = ?`
- Match input words against `activation_patterns` array (case-insensitive)
- Priority tie-breaking: `decision(6) > debug(5) > task(4) > question(3) > creative(2) > conversation(1)`
- Increment `usage_count` on the selected frame
- Return `FrameSelection` with `match_method="pattern"` or `"default"`

### 3. `nous/cognitive/context.py` (~250 lines)

Context assembly engine. Builds system prompt within token budget.

```python
class ContextEngine:
    """Assembles context from Brain and Heart within token budgets."""

    # Rough chars-per-token estimate
    CHARS_PER_TOKEN = 4

    def __init__(
        self,
        brain: Brain,
        heart: Heart,
        settings: Settings,
        identity_prompt: str = "",
    ) -> None:
        self._brain = brain
        self._heart = heart
        self._settings = settings
        self._identity_prompt = identity_prompt

    async def build(
        self,
        agent_id: str,
        session_id: str,
        input_text: str,
        frame: FrameSelection,
        session: AsyncSession | None = None,
    ) -> tuple[str, list[ContextSection]]:
        """Build system prompt + context sections within budget.

        Returns (system_prompt_string, sections_list).

        Assembly order (by priority):
        1. Identity prompt (always included, static)
        2. Active censors from Heart (always, severity=block first)
        3. Frame description + questions_to_ask
        4. Working memory (current task + open threads)
        5. Similar decisions from Brain.query()
        6. Relevant facts from Heart.recall(type="fact")
        7. Relevant procedures from Heart.recall(type="procedure")
        8. Related episodes from Heart.recall(type="episode")

        Each section is truncated to its budget allocation.
        Sections are skipped entirely if budget is 0 for that layer.
        """

    def _estimate_tokens(self, text: str) -> int:
        """Rough token count: len(text) / CHARS_PER_TOKEN."""
        return max(1, len(text) // self.CHARS_PER_TOKEN)

    def _truncate_to_budget(self, text: str, token_budget: int) -> str:
        """Truncate text to fit within token budget."""
        max_chars = token_budget * self.CHARS_PER_TOKEN
        if len(text) <= max_chars:
            return text
        return text[:max_chars - 3] + "..."

    def _format_decisions(self, decisions: list) -> str:
        """Format decision summaries for context.

        Format per decision:
        - [{outcome or "pending"}] {description} (confidence: {confidence})
          Reasons: {comma-separated reason texts}
        """

    def _format_facts(self, facts: list) -> str:
        """Format facts for context.

        Format per fact:
        - {content} [confirmed {confirmation_count}x, {status}]
        """

    def _format_procedures(self, procedures: list) -> str:
        """Format procedures for context.

        Format per procedure:
        - **{name}** ({domain}): {core_patterns as bullets}
        """

    def _format_episodes(self, episodes: list) -> str:
        """Format episodes for context.

        Format per episode:
        - [{outcome}] {summary} ({started_at date})
        """

    def _format_censors(self, censors: list) -> str:
        """Format active censors.

        Format:
        - **{SEVERITY}:** {trigger_pattern} — {reason}
        """

    def _format_working_memory(self, wm) -> str:
        """Format working memory state.

        Includes current_task, open_threads, and high-relevance items.
        """

    async def refresh_needed(
        self,
        agent_id: str,
        session_id: str,
        new_input: str,
        current_frame: FrameSelection,
        session: AsyncSession | None = None,
    ) -> bool:
        """Check if context should be rebuilt.

        Returns True if:
        1. Working memory's current_frame differs from current_frame.frame_id
        2. No working memory exists for this session
        """

    async def expand(
        self,
        memory_type: str,
        memory_id: str,
        session: AsyncSession | None = None,
    ) -> str:
        """Load full detail for a specific memory.

        memory_type: "decision", "fact", "episode", "procedure"
        Routes to Brain.get() or Heart.{manager}.get() accordingly.
        """
```

**Implementation notes:**
- Use `ContextBudget.for_frame(frame.frame_id)` to get budget
- Call Brain.query(input_text, limit=5) for decisions
- Call Heart.recall(input_text, ...) for facts/procedures/episodes (recall is already implemented)
- Call Heart.censors.check(input_text) is NOT what we want — use `Heart.censors.search(agent_id)` or query active censors directly
- For censors: query `heart.censors` WHERE `agent_id = ?` AND `status = 'active'`, order by severity DESC
- Working memory: use `Heart.working_memory.get(agent_id, session_id)`
- Build sections in priority order; each section gets its budget slice
- Concatenate all sections into one system prompt string with markdown headers

### 4. `nous/cognitive/deliberation.py` (~100 lines)

Thin wrapper around Brain for deliberation lifecycle.

```python
class DeliberationEngine:
    """Manages the deliberation lifecycle for a single turn.

    Wraps Brain.record(), Brain.think(), Brain.update() into
    a clean start → think → finalize flow.
    """

    def __init__(self, brain: Brain) -> None:
        self._brain = brain

    async def start(
        self,
        agent_id: str,
        description: str,
        frame: FrameSelection,
        session: AsyncSession | None = None,
    ) -> str:
        """Begin deliberation — record intent, return decision_id.

        Creates a decision via Brain.record() with:
        - description: "Plan: {description}"
        - confidence: 0.5 (to be updated in finalize)
        - category: frame.default_category or "process"
        - stakes: frame.default_stakes or "low"
        - tags: [frame.frame_id]
        - pattern: None (to be filled in finalize)
        """

    async def think(
        self,
        decision_id: str,
        thought: str,
        agent_id: str,
        session: AsyncSession | None = None,
    ) -> None:
        """Capture a micro-thought during deliberation."""

    async def finalize(
        self,
        decision_id: str,
        description: str,
        confidence: float,
        context: str | None = None,
        pattern: str | None = None,
        tags: list[str] | None = None,
        session: AsyncSession | None = None,
    ) -> None:
        """Update decision with final outcome.

        Calls Brain.update() with the provided fields.
        Only updates fields that are not None.
        """

    async def should_deliberate(self, frame: FrameSelection) -> bool:
        """Should this frame trigger deliberation?

        Returns True for: decision, task, debug
        Returns False for: conversation, question, creative
        """
```

### 5. `nous/cognitive/monitor.py` (~180 lines)

Post-turn assessment and learning.

```python
class MonitorEngine:
    """Post-turn self-assessment and learning.

    After each turn:
    1. Assess: Was the outcome surprising? Did censors fire?
    2. Learn: Extract facts, record episode, create censors from failures.
    """

    def __init__(self, brain: Brain, heart: Heart, settings: Settings) -> None:
        self._brain = brain
        self._heart = heart
        self._settings = settings

    async def assess(
        self,
        agent_id: str,
        session_id: str,
        turn_result: TurnResult,
        decision_id: str | None = None,
        session: AsyncSession | None = None,
    ) -> Assessment:
        """Evaluate what happened during the turn.

        Steps:
        1. If decision_id exists, fetch the decision from Brain
        2. Calculate surprise_level:
           - 0.0 if no errors and no decision
           - 0.3 if tool errors occurred
           - 0.7 if response contains error indicators ("failed", "error", "couldn't")
           - 0.9 if the turn itself errored (turn_result.error is set)
        3. Generate censor_candidates from errors:
           - Each tool error becomes: "Avoid {tool_name} with {args pattern} — {error}"
           - Only if error is NOT transient (skip timeout, rate limit, network errors)
        4. Return Assessment
        """

    async def learn(
        self,
        agent_id: str,
        session_id: str,
        assessment: Assessment,
        turn_result: TurnResult,
        frame: FrameSelection,
        episode_id: str | None = None,
        session: AsyncSession | None = None,
    ) -> Assessment:
        """Post-assessment learning — update state and create artifacts.

        Steps:
        1. If surprise_level > 0.7 and censor_candidates exist:
           - Create censors via Heart.censors.add() for each candidate
           - Set severity="warn" (escalation happens via Heart's built-in logic)

        2. If decision_id exists and turn_result has no errors:
           - Record thought "Turn completed successfully" via Brain.think()

        3. If episode_id exists:
           - End episode via Heart.episodes.end() with:
             - outcome: "success" if no errors, "partial" if tool errors, "failure" if turn error
             - lessons: censor_candidates (things learned NOT to do)
             - summary: f"{frame.frame_name}: {turn_result.response_text[:200]}"

        4. Update Assessment with facts_extracted count and episode_recorded flag.

        Returns updated Assessment.
        """

    def _is_transient_error(self, error: str) -> bool:
        """Check if error is transient (shouldn't create censors).

        Transient patterns: timeout, rate limit, 429, 503, connection refused,
        network error, ECONNRESET, ETIMEDOUT.
        """

    def _error_to_censor_text(self, tool_result: ToolResult) -> str:
        """Convert a tool error to a censor trigger pattern.

        Format: "Avoid using {tool_name} when {simplified args description} — caused: {error[:100]}"
        """
```

### 6. `nous/cognitive/layer.py` (~200 lines)

The main orchestrator. This is the public API.

```python
class CognitiveLayer:
    """The Nous Loop — orchestrates Brain and Heart into cognition.

    Usage:
        cognitive = CognitiveLayer(brain, heart, settings)

        # Before LLM turn
        ctx = await cognitive.pre_turn(agent_id, session_id, user_input)
        # ctx.system_prompt contains full context
        # ctx.decision_id set if deliberation started

        # After LLM turn
        result = TurnResult(response_text=llm_output, tool_results=[...])
        assessment = await cognitive.post_turn(agent_id, session_id, result, ctx)

        # End of conversation
        await cognitive.end_session(agent_id, session_id)
    """

    def __init__(
        self,
        brain: Brain,
        heart: Heart,
        settings: Settings,
        identity_prompt: str = "",
    ) -> None:
        self._brain = brain
        self._heart = heart
        self._settings = settings
        self._frames = FrameEngine(brain._db, settings)
        self._context = ContextEngine(brain, heart, settings, identity_prompt)
        self._deliberation = DeliberationEngine(brain)
        self._monitor = MonitorEngine(brain, heart, settings)

        # Track active episodes per session
        self._active_episodes: dict[str, str] = {}  # session_id -> episode_id

    async def pre_turn(
        self,
        agent_id: str,
        session_id: str,
        user_input: str,
        session: AsyncSession | None = None,
    ) -> TurnContext:
        """SENSE → FRAME → RECALL → DELIBERATE — prepare for LLM turn.

        Steps:
        1. SENSE: Receive user_input (passed in)

        2. FRAME: Select cognitive frame via FrameEngine.select()

        3. RECALL: Build context via ContextEngine.build()
           - Also checks refresh_needed() — if yes, rebuilds

        4. DELIBERATE: If frame warrants it (decision/task/debug):
           - Start deliberation via DeliberationEngine.start()
           - Record decision_id on TurnContext

        5. EPISODE: If no active episode for this session:
           - Start one via Heart.episodes.start()
           - Track in self._active_episodes

        6. WORKING MEMORY: Update via Heart.working_memory.focus()
           with current task = user_input, current_frame = frame.frame_id

        Return TurnContext with system_prompt, frame, decision_id, metadata.
        """

    async def post_turn(
        self,
        agent_id: str,
        session_id: str,
        turn_result: TurnResult,
        turn_context: TurnContext,
        session: AsyncSession | None = None,
    ) -> Assessment:
        """ACT (done) → MONITOR → LEARN — process LLM output.

        Steps:
        1. MONITOR: Assess the turn via MonitorEngine.assess()
           - Pass decision_id from turn_context if present

        2. LEARN: Extract lessons via MonitorEngine.learn()
           - Pass episode_id from self._active_episodes

        3. DELIBERATION: If decision_id exists and no errors:
           - Finalize via DeliberationEngine.finalize()
           - Description: first 200 chars of response
           - Confidence: 0.8 if no errors, 0.5 if tool errors, 0.3 if turn error

        4. Emit event to nous_system.events:
           - event_type: "turn_completed"
           - data: {frame, surprise_level, decision_id, has_errors}

        Return Assessment.
        """

    async def end_session(
        self,
        agent_id: str,
        session_id: str,
        session: AsyncSession | None = None,
    ) -> None:
        """Clean up session state.

        1. If active episode exists for this session:
           - End it via Heart.episodes.end() with outcome="completed"
        2. Remove from self._active_episodes
        3. Emit "session_ended" event
        """

    async def _emit_event(
        self,
        agent_id: str,
        session_id: str,
        event_type: str,
        data: dict,
        session: AsyncSession | None = None,
    ) -> None:
        """Write event to nous_system.events table.

        Creates an Event ORM object and inserts it.
        If session is provided, uses it; otherwise creates new.
        """
```

### 7. `nous/cognitive/__init__.py` (~30 lines)

Public re-exports.

```python
"""Cognitive layer — The Nous Loop.

Orchestrates Brain (decisions) and Heart (memory) into a
thinking loop: Sense → Frame → Recall → Deliberate → Act → Monitor → Learn.
"""

from nous.cognitive.context import ContextEngine
from nous.cognitive.deliberation import DeliberationEngine
from nous.cognitive.frames import FrameEngine
from nous.cognitive.layer import CognitiveLayer
from nous.cognitive.monitor import MonitorEngine
from nous.cognitive.schemas import (
    Assessment,
    ContextBudget,
    ContextSection,
    FrameSelection,
    FrameType,
    ToolResult,
    TurnContext,
    TurnResult,
)

__all__ = [
    "CognitiveLayer",
    "ContextEngine",
    "DeliberationEngine",
    "FrameEngine",
    "MonitorEngine",
    "Assessment",
    "ContextBudget",
    "ContextSection",
    "FrameSelection",
    "FrameType",
    "ToolResult",
    "TurnContext",
    "TurnResult",
]
```

## Test Specification

### `tests/test_frames.py` (~120 lines)

```python
# test_frame_select_decision — input "should we use Redis?" → frame_id="decision"
# test_frame_select_task — input "build a REST API" → frame_id="task"
# test_frame_select_debug — input "error in deployment" → frame_id="debug"
# test_frame_select_conversation — input "hey how are you" → frame_id="conversation"
# test_frame_select_no_match — input "xyzzy foobar" → default "conversation"
# test_frame_select_tiebreak — input "should we fix this bug" → "decision" beats "debug" (higher priority)
# test_frame_list — returns all 6 seed frames
# test_frame_get — fetch specific frame by id
# test_frame_usage_count_increments — select bumps usage_count
```

### `tests/test_context.py` (~200 lines)

```python
# test_build_conversation_budget — conversation frame gets 3K budget
# test_build_decision_budget — decision frame gets 12K budget
# test_build_includes_identity — identity section always present
# test_build_includes_censors — active censors always in context
# test_build_truncates_to_budget — long content truncated, not omitted
# test_build_skips_zero_budget — conversation skips procedures (budget=0)
# test_build_with_decisions — Brain.query() results appear in context
# test_build_with_facts — Heart facts appear in context
# test_build_with_working_memory — current task in context
# test_refresh_needed_frame_change — True when frame changed
# test_refresh_needed_same_frame — False when frame unchanged
# test_expand_decision — loads full decision detail
# test_expand_fact — loads full fact detail

# Fixtures: pre-seed Brain with 3 decisions, Heart with 5 facts, 2 procedures, 1 censor
```

### `tests/test_deliberation.py` (~80 lines)

```python
# test_start_creates_decision — record called with "Plan: ..." prefix
# test_start_uses_frame_defaults — category/stakes from frame
# test_think_delegates_to_brain — thought recorded via Brain.think()
# test_finalize_updates_decision — Brain.update() called with new description + confidence
# test_should_deliberate_decision — True for decision frame
# test_should_deliberate_task — True for task frame
# test_should_deliberate_conversation — False for conversation frame
```

### `tests/test_monitor.py` (~150 lines)

```python
# test_assess_no_errors — surprise_level=0.0
# test_assess_tool_error — surprise_level=0.3, censor_candidate generated
# test_assess_turn_error — surprise_level=0.9
# test_assess_response_error_keywords — "failed" in response → 0.7
# test_transient_error_no_censor — timeout/429 don't create censor candidates
# test_learn_creates_censors — high surprise → Heart.censors.add() called
# test_learn_ends_episode_success — no errors → outcome="success"
# test_learn_ends_episode_failure — turn error → outcome="failure"
# test_error_to_censor_text — format matches expected pattern
```

### `tests/test_cognitive_layer.py` (~250 lines)

Integration tests for the full loop.

```python
# --- pre_turn ---
# test_pre_turn_selects_frame — returns TurnContext with correct frame
# test_pre_turn_builds_context — system_prompt is non-empty
# test_pre_turn_starts_deliberation — decision frame → decision_id set
# test_pre_turn_no_deliberation_conversation — conversation → decision_id is None
# test_pre_turn_starts_episode — first call creates episode
# test_pre_turn_reuses_episode — second call same session reuses episode
# test_pre_turn_updates_working_memory — working memory set with input

# --- post_turn ---
# test_post_turn_assesses — returns Assessment
# test_post_turn_finalizes_deliberation — decision_id → Brain.update() called
# test_post_turn_no_finalize_without_decision — no decision_id → no update
# test_post_turn_creates_censor_on_failure — tool error → censor created
# test_post_turn_emits_event — event in nous_system.events

# --- end_session ---
# test_end_session_closes_episode — episode ended with "completed"
# test_end_session_emits_event — session_ended event
# test_end_session_idempotent — calling twice doesn't error

# --- full loop ---
# test_full_loop_decision — pre_turn(decision) → post_turn(success) → end_session
# test_full_loop_with_error — pre_turn(task) → post_turn(tool_error) → censor created
# test_full_loop_conversation — lightweight, no deliberation, no censor

# Fixtures: pre-seed with decisions, facts, procedures, censors for realistic context
```

## Dependencies

No new external dependencies. Uses only:
- `sqlalchemy` (already installed)
- `pydantic` (already installed)
- Brain and Heart modules (already implemented)

## Key Design Decisions

### D1: No LLM calls in the cognitive layer
Frame selection is pattern matching, not classification. Context assembly is retrieval, not generation. This keeps the cognitive layer fast (<100ms) and deterministic. The LLM is only invoked by the runtime (F004), not by the cognitive layer.

### D2: Session-scoped episode tracking via in-memory dict
`self._active_episodes` maps session_id → episode_id. This is ephemeral — if the process restarts, episodes are orphaned (ended by F008 Memory Lifecycle later). This is acceptable for v0.1.0. The dict is safe because Python async is single-threaded.

### D3: Surprise level is heuristic, not semantic
We use simple string matching ("failed", "error") and structural checks (tool errors, turn errors) rather than LLM-based assessment. This is deliberately crude — v0.1.0 needs to ship. Semantic assessment is a future enhancement.

### D4: Censor creation is conservative
Only non-transient tool errors create censor candidates. Censors start at severity="warn" and escalate via Heart's built-in logic. We err on the side of not creating censors rather than over-censoring.

### D5: Context sections are concatenated, not interleaved
System prompt is built by concatenating priority-ordered sections with markdown headers. No interleaving of decisions with facts. Simple, predictable, easy to debug.

### D6: ContextEngine depends on both Brain and Heart
This is intentional — the Context Engine IS the bridge between the two organs. It's the only component that queries both.

### D7: Events table used for audit trail
Every pre_turn and post_turn writes to `nous_system.events`. This is the raw event stream that F006 (Event Bus) will consume later. For now, it's just INSERT — no event bus processing.

### D8: Deliberation auto-start based on frame type
Decision, task, and debug frames automatically start deliberation. Conversation, question, and creative don't. This follows the principle: "record decisions BEFORE acting." The agent doesn't choose whether to deliberate — the frame decides.

## Error Handling

- All public methods catch exceptions and log them rather than crashing the turn
- If frame selection fails → fall back to "conversation" frame
- If context build fails → return minimal system prompt (identity only)
- If deliberation start fails → set decision_id=None, continue without deliberation
- If event emission fails → log warning, don't block the turn
- If episode operations fail → log warning, remove from tracking dict

## Migration Notes

No schema changes required. All tables exist from 001-postgres-scaffold:
- `nous_system.frames` — already seeded with 6 frames
- `nous_system.events` — ready for event insertion
- `heart.working_memory` — used by Heart module
- All Brain and Heart tables — queried via their existing APIs
