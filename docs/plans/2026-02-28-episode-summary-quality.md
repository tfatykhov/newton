# 008.4 Episode Summary Quality — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve episode summary quality with better prompts, decision awareness, smart truncation, and fact extraction coordination.

**Architecture:** Two PRs. PR 1 replaces the summary prompt and wires candidate_facts through to the fact extractor (skip redundant LLM call). PR 2 adds Brain constructor injection for decision context and priority-based transcript truncation.

**Tech Stack:** Python 3.12+, httpx, pytest, AsyncMock

**Design doc:** `docs/plans/2026-02-28-episode-summary-quality-design.md`

---

## PR 1: Enhanced Prompt + Fact Coordination

### Task 1: Write failing test for new summary prompt fields

**Files:**
- Modify: `tests/test_event_bus.py` — add to `TestEpisodeSummarizer` class (after line 420)

**Step 1: Write the failing test**

Add this test to `TestEpisodeSummarizer` in `tests/test_event_bus.py`:

```python
@pytest.mark.asyncio
async def test_summary_includes_new_fields(self):
    """008.4: Summary includes outcome_rationale and candidate_facts."""
    summary_json = {
        "title": "Architecture Discussion",
        "summary": "Discussed project architecture and chose PostgreSQL.",
        "key_points": ["PostgreSQL chosen for pgvector support and unified storage"],
        "outcome": "resolved",
        "outcome_rationale": "User's question was fully answered with a concrete decision",
        "topics": ["architecture", "database"],
        "candidate_facts": ["Project uses PostgreSQL 17 with pgvector for embeddings"],
    }
    summarizer, heart, bus, http_client = self._make_summarizer()
    heart.get_episode = AsyncMock(
        return_value=MagicMock(summary="opening", structured_summary=None)
    )
    heart.update_episode_summary = AsyncMock()
    http_client.post = AsyncMock(
        return_value=_mock_httpx_response(200, _llm_response(json.dumps(summary_json)))
    )

    episode_id = str(uuid4())
    event = _make_event(
        "session_ended",
        data={
            "episode_id": episode_id,
            "transcript": "User: What database should we use?\n\nAssistant: I recommend PostgreSQL with pgvector." * 3,
        },
    )
    await summarizer.handle(event)

    # Verify summary stored with new fields
    heart.update_episode_summary.assert_called_once()
    stored_summary = heart.update_episode_summary.call_args[0][1]
    assert stored_summary["outcome_rationale"] == "User's question was fully answered with a concrete decision"
    assert stored_summary["candidate_facts"] == ["Project uses PostgreSQL 17 with pgvector for embeddings"]

    # Verify candidate_facts passed through in emitted event
    bus.emit.assert_called_once()
    emitted = bus.emit.call_args[0][0]
    assert emitted.data["candidate_facts"] == ["Project uses PostgreSQL 17 with pgvector for embeddings"]
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_event_bus.py::TestEpisodeSummarizer::test_summary_includes_new_fields -v`
Expected: FAIL — emitted event data doesn't contain `candidate_facts` key

**Step 3: Commit failing test**

```bash
git add tests/test_event_bus.py
git commit -m "test: 008.4 failing test for new summary prompt fields"
```

---

### Task 2: Replace summary prompt and wire candidate_facts through event

**Files:**
- Modify: `nous/handlers/episode_summarizer.py:26-38` — replace `_SUMMARY_PROMPT`
- Modify: `nous/handlers/episode_summarizer.py:89-97` — add `candidate_facts` to emitted event

**Step 1: Replace `_SUMMARY_PROMPT` (lines 26-38)**

Replace the entire `_SUMMARY_PROMPT` string with:

```python
_SUMMARY_PROMPT = """You are summarizing a conversation episode for an AI agent's long-term memory.

Context:
- Agent: Nous (cognitive agent framework)
- This summary will be used for: semantic search recall, context assembly, calibration

Transcript:
{transcript}

{decision_context}

Return ONLY valid JSON (no markdown, no explanation):
{{
  "title": "<5-10 word descriptive title focusing on WHAT WAS ACCOMPLISHED>",
  "summary": "<100-150 word prose summary emphasizing decisions made, problems solved, and outcomes>",
  "key_points": [
    "<lesson or reusable knowledge, not just event description>",
    "<pattern or insight that would help in similar future situations>"
  ],
  "outcome": "<resolved|partial|unresolved|informational>",
  "outcome_rationale": "<1 sentence explaining why this outcome classification>",
  "topics": ["<topic1>", "<topic2>"],
  "candidate_facts": [
    "<factual statement worth storing as long-term knowledge>"
  ]
}}

Outcome guidelines:
- resolved: The user's request was fully addressed, task completed, question answered
- partial: Work started but not finished, or only some requests addressed
- unresolved: Failed to complete the task, hit blockers
- informational: Casual chat, status check, no actionable work done

For key_points: Focus on WHAT WAS LEARNED, not what happened. Ask yourself:
"If this agent faces a similar situation, what from this episode would help?"

For candidate_facts: Extract concrete, reusable knowledge (tool configs, preferences,
architectural decisions, API behaviors) that should persist as standalone facts."""
```

Note: The `{decision_context}` placeholder is empty string for PR 1. PR 2 fills it in.

**Step 2: Update `_generate_summary` to pass empty decision_context**

In `_generate_summary()` (line 115), change the prompt format call:

```python
prompt = _SUMMARY_PROMPT.format(transcript=transcript, decision_context="")
```

**Step 3: Update `_generate_summary` max_tokens**

The new prompt asks for more structured output. In the API call (line 123), increase max_tokens:

```python
"max_tokens": 800,
```

**Step 4: Add `candidate_facts` to emitted event data (lines 89-97)**

Update the `bus.emit()` call in `handle()`:

```python
await self._bus.emit(Event(
    type="episode_summarized",
    agent_id=event.agent_id,
    session_id=event.session_id,
    data={
        "episode_id": episode_id,
        "summary": summary,
        "candidate_facts": summary.get("candidate_facts", []),
    },
))
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_event_bus.py::TestEpisodeSummarizer::test_summary_includes_new_fields -v`
Expected: PASS

**Step 6: Run all existing summarizer tests to verify no regressions**

Run: `uv run pytest tests/test_event_bus.py::TestEpisodeSummarizer -v`
Expected: All 7 tests PASS (6 existing + 1 new)

Note: Test 14 (`test_truncates_long_transcripts`) checks for `[... middle truncated ...]` in the prompt. This marker is still present in the current truncation code (line 113), so it should still pass.

**Step 7: Commit**

```bash
git add nous/handlers/episode_summarizer.py tests/test_event_bus.py
git commit -m "feat(008.4): enhanced summary prompt with outcome guidelines and candidate_facts"
```

---

### Task 3: Write failing test for fact extractor candidate_facts passthrough

**Files:**
- Modify: `tests/test_event_bus.py` — add tests to `TestFactExtractor` class (after line 623)

**Step 1: Write the failing tests**

Add these tests to `TestFactExtractor` in `tests/test_event_bus.py`:

```python
@pytest.mark.asyncio
async def test_uses_candidate_facts_skips_llm(self):
    """008.4: When candidate_facts present, store directly without LLM call."""
    extractor, heart, bus, http_client = self._make_extractor()
    heart.search_facts = AsyncMock(return_value=[])  # No duplicates
    heart.learn = AsyncMock()

    event = _make_event(
        "episode_summarized",
        data={
            "episode_id": str(uuid4()),
            "summary": {
                "summary": "Discussed architecture.",
                "key_points": ["chose PostgreSQL"],
            },
            "candidate_facts": [
                "Project uses PostgreSQL 17 with pgvector",
                "Tim prefers direct architecture decisions",
            ],
        },
    )
    await extractor.handle(event)

    # LLM should NOT be called
    http_client.post.assert_not_called()
    # Both facts should be stored
    assert heart.learn.call_count == 2
    stored_contents = [call[0][0].content for call in heart.learn.call_args_list]
    assert "Project uses PostgreSQL 17 with pgvector" in stored_contents
    assert "Tim prefers direct architecture decisions" in stored_contents

@pytest.mark.asyncio
async def test_candidate_facts_deduped(self):
    """008.4: candidate_facts are deduped against existing facts."""
    extractor, heart, bus, http_client = self._make_extractor()

    existing_fact = MagicMock(spec=FactSummary)
    existing_fact.score = 0.90  # Above 0.85 -> deduped
    heart.search_facts = AsyncMock(return_value=[existing_fact])
    heart.learn = AsyncMock()

    event = _make_event(
        "episode_summarized",
        data={
            "episode_id": str(uuid4()),
            "summary": {"summary": "Already known.", "key_points": []},
            "candidate_facts": ["Project uses PostgreSQL"],
        },
    )
    await extractor.handle(event)

    http_client.post.assert_not_called()
    heart.learn.assert_not_called()  # Deduped

@pytest.mark.asyncio
async def test_falls_back_to_llm_without_candidate_facts(self):
    """008.4: When no candidate_facts, falls back to LLM extraction."""
    facts_json = [
        {"subject": "user", "content": "User likes tests", "category": "preference", "confidence": 0.9},
    ]
    extractor, heart, bus, http_client = self._make_extractor()
    heart.search_facts = AsyncMock(return_value=[])
    heart.learn = AsyncMock()
    http_client.post = AsyncMock(
        return_value=_mock_httpx_response(200, _llm_response(json.dumps(facts_json)))
    )

    event = _make_event(
        "episode_summarized",
        data={
            "episode_id": str(uuid4()),
            "summary": {"summary": "User likes testing.", "key_points": ["testing"]},
            # No candidate_facts key at all
        },
    )
    await extractor.handle(event)

    # LLM SHOULD be called (fallback)
    http_client.post.assert_called_once()
    heart.learn.assert_called_once()

@pytest.mark.asyncio
async def test_candidate_facts_max_5(self):
    """008.4: candidate_facts respects max 5 limit."""
    extractor, heart, bus, http_client = self._make_extractor()
    heart.search_facts = AsyncMock(return_value=[])
    heart.learn = AsyncMock()

    event = _make_event(
        "episode_summarized",
        data={
            "episode_id": str(uuid4()),
            "summary": {"summary": "Many facts.", "key_points": []},
            "candidate_facts": [f"Fact number {i}" for i in range(8)],
        },
    )
    await extractor.handle(event)

    http_client.post.assert_not_called()
    assert heart.learn.call_count == 5  # Max 5
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_event_bus.py::TestFactExtractor::test_uses_candidate_facts_skips_llm -v`
Expected: FAIL — current code ignores `candidate_facts` in event data

**Step 3: Commit failing tests**

```bash
git add tests/test_event_bus.py
git commit -m "test: 008.4 failing tests for fact extractor candidate_facts passthrough"
```

---

### Task 4: Implement fact extractor candidate_facts passthrough

**Files:**
- Modify: `nous/handlers/fact_extractor.py:71-119` — update `handle()` method

**Step 1: Update `handle()` to check for candidate_facts first**

Replace the `handle()` method body (lines 71-119):

```python
async def handle(self, event: Event) -> None:
    """Handle episode_summarized — extract and store facts.

    008.4: If candidate_facts present in event data, store them directly
    without calling the LLM. Falls back to LLM extraction otherwise.
    """
    summary = event.data.get("summary", {})
    if not summary:
        return

    try:
        # 008.4: Use pre-extracted candidate_facts if available
        candidate_facts = event.data.get("candidate_facts", [])
        if candidate_facts:
            await self._store_candidate_facts(
                candidate_facts, event.data.get("episode_id", "?")
            )
            return

        # Fallback: LLM extraction (backward compatibility)
        candidates = await self._extract_facts(summary)
        if not candidates:
            return

        stored = 0
        for fact in candidates[:5]:  # Max 5 per episode
            confidence = fact.get("confidence", 0.7)
            if confidence < 0.6:
                logger.debug("Skipping low-confidence fact: %s", fact.get("content", "")[:50])
                continue

            content = fact.get("content", "")
            existing = await self._heart.search_facts(content, limit=1)
            if existing and existing[0].score is not None and existing[0].score > 0.85:
                logger.debug("Skipping duplicate fact: %s", content[:50])
                continue

            fact_input = FactInput(
                subject=fact.get("subject", "unknown"),
                content=content,
                source="fact_extractor",
                confidence=confidence,
                category=fact.get("category"),
            )
            await self._heart.learn(fact_input)
            stored += 1

        if stored:
            logger.info(
                "Extracted %d facts from episode %s",
                stored,
                event.data.get("episode_id", "?"),
            )

    except Exception:
        logger.exception("Fact extraction failed for episode %s", event.data.get("episode_id"))
```

**Step 2: Add `_store_candidate_facts()` method**

Add this method to `FactExtractor` class after `handle()`:

```python
async def _store_candidate_facts(self, candidates: list[str], episode_id: str) -> None:
    """008.4: Store pre-extracted candidate facts directly, with dedup."""
    stored = 0
    for fact_text in candidates[:5]:  # Max 5 per episode
        if not fact_text or not fact_text.strip():
            continue

        # Dedup against existing facts
        existing = await self._heart.search_facts(fact_text, limit=1)
        if existing and existing[0].score is not None and existing[0].score > 0.85:
            logger.debug("Skipping duplicate candidate fact: %s", fact_text[:50])
            continue

        fact_input = FactInput(
            content=fact_text,
            source="episode_summarizer",
            confidence=0.8,  # Default confidence for LLM-extracted candidates
        )
        await self._heart.learn(fact_input)
        stored += 1

    if stored:
        logger.info(
            "Stored %d candidate facts from episode %s",
            stored,
            episode_id,
        )
```

**Step 3: Run new tests to verify they pass**

Run: `uv run pytest tests/test_event_bus.py::TestFactExtractor::test_uses_candidate_facts_skips_llm tests/test_event_bus.py::TestFactExtractor::test_candidate_facts_deduped tests/test_event_bus.py::TestFactExtractor::test_falls_back_to_llm_without_candidate_facts tests/test_event_bus.py::TestFactExtractor::test_candidate_facts_max_5 -v`
Expected: All 4 PASS

**Step 4: Run all fact extractor tests for regressions**

Run: `uv run pytest tests/test_event_bus.py::TestFactExtractor -v`
Expected: All tests PASS (existing + 4 new)

**Step 5: Commit**

```bash
git add nous/handlers/fact_extractor.py tests/test_event_bus.py
git commit -m "feat(008.4): fact extractor uses candidate_facts, skips LLM call"
```

---

### Task 5: Run full test suite for PR 1

**Step 1: Run all event bus tests**

Run: `uv run pytest tests/test_event_bus.py -v`
Expected: All tests PASS

**Step 2: Run episode tests (008.3 regression check)**

Run: `uv run pytest tests/test_episodes.py -v`
Expected: All tests PASS

**Step 3: Create PR 1 branch and commit**

```bash
git checkout -b feat/008.4-prompt-quality
git push -u origin feat/008.4-prompt-quality
```

---

## PR 2: Brain Context + Smart Truncation

### Task 6: Write failing test for `_truncate_transcript`

**Files:**
- Modify: `tests/test_event_bus.py` — add tests to `TestEpisodeSummarizer` class

**Step 1: Write the failing tests**

Add these tests to `TestEpisodeSummarizer`:

```python
@pytest.mark.asyncio
async def test_truncate_noop_under_limit(self):
    """008.4: Short transcript returned unchanged."""
    from nous.handlers.episode_summarizer import EpisodeSummarizer

    summarizer, _, _, _ = self._make_summarizer()
    short = "User: Hello\n\nAssistant: Hi there"
    result = summarizer._truncate_transcript(short)
    assert result == short

@pytest.mark.asyncio
async def test_truncate_preserves_first_last(self):
    """008.4: First and last turns always kept."""
    from nous.handlers.episode_summarizer import EpisodeSummarizer

    summarizer, _, _, _ = self._make_summarizer()
    turns = ["User: First turn"] + [f"Assistant: Middle turn {i} " + "x" * 500 for i in range(20)] + ["User: Last turn"]
    transcript = "\n\n".join(turns)
    result = summarizer._truncate_transcript(transcript, max_chars=2000)
    assert result.startswith("User: First turn")
    assert result.endswith("User: Last turn")

@pytest.mark.asyncio
async def test_truncate_prioritizes_decisions(self):
    """008.4: Decision turns kept over tool output."""
    from nous.handlers.episode_summarizer import EpisodeSummarizer

    summarizer, _, _, _ = self._make_summarizer()
    decision_turn = "Assistant: We decided to use PostgreSQL because it supports pgvector natively."
    tool_turn = "Tool output:\n```\n" + "x" * 600 + "\n```"
    filler = "Assistant: " + "y" * 400

    turns = ["User: Start"] + [tool_turn] * 5 + [decision_turn] + [filler] * 5 + ["User: End"]
    transcript = "\n\n".join(turns)
    result = summarizer._truncate_transcript(transcript, max_chars=3000)

    # Decision turn should be preserved
    assert "decided to use PostgreSQL" in result
```

**Step 2: Run to verify failure**

Run: `uv run pytest tests/test_event_bus.py::TestEpisodeSummarizer::test_truncate_noop_under_limit -v`
Expected: FAIL — `_truncate_transcript` method doesn't exist

**Step 3: Commit failing tests**

```bash
git add tests/test_event_bus.py
git commit -m "test: 008.4 failing tests for smart transcript truncation"
```

---

### Task 7: Implement `_truncate_transcript`

**Files:**
- Modify: `nous/handlers/episode_summarizer.py` — add method, replace lines 111-113

**Step 1: Add `_truncate_transcript` method to `EpisodeSummarizer`**

Add this method to the class (after `_generate_summary`):

```python
def _truncate_transcript(self, transcript: str, max_chars: int = 8000) -> str:
    """008.4: Truncate transcript preserving high-value turns.

    Scores turns by information density: decision language and user turns
    score higher, long tool outputs score lower. Always keeps first and
    last turns. Fills middle by score within budget.
    """
    if len(transcript) <= max_chars:
        return transcript

    turns = transcript.split("\n\n")
    if len(turns) <= 2:
        # Only first/last — truncate the longer one
        return transcript[:max_chars]

    # Score each turn by information density
    scored: list[tuple[float, int, str]] = []
    for i, turn in enumerate(turns):
        score = 1.0
        lower = turn.lower()
        # Boost: decision language, conclusions
        if any(w in lower for w in ["decided", "chose", "because", "learned", "conclusion", "chosen"]):
            score += 2.0
        # Boost: user turns (directives, questions)
        if lower.startswith("user:") or lower.startswith("human:"):
            score += 1.0
        # Penalize: long tool outputs, raw data
        if len(turn) > 500 and ("```" in turn or turn.count("\n") > 10):
            score -= 1.0
        scored.append((score, i, turn))

    # Always keep first and last turns
    first = turns[0]
    last = turns[-1]
    budget = max_chars - len(first) - len(last) - 50  # buffer for separators

    if budget <= 0:
        return first[:max_chars // 2] + "\n\n" + last[:max_chars // 2]

    # Sort middle turns by score (descending), break ties by original order
    middle = sorted(scored[1:-1], key=lambda x: (-x[0], x[1]))
    kept_indices: set[int] = set()
    used = 0
    for score, idx, turn in middle:
        if used + len(turn) > budget:
            continue
        kept_indices.add(idx)
        used += len(turn)

    # Reconstruct in original order
    result = [first]
    for score, idx, turn in scored[1:-1]:
        if idx in kept_indices:
            result.append(turn)
    result.append(last)

    return "\n\n".join(result)
```

**Step 2: Replace the inline truncation in `_generate_summary` (lines 111-113)**

Replace:
```python
if len(transcript) > 8000:
    half = 3800
    transcript = transcript[:half] + "\n\n[... middle truncated ...]\n\n" + transcript[-half:]
```

With:
```python
transcript = self._truncate_transcript(transcript)
```

**Step 3: Run truncation tests**

Run: `uv run pytest tests/test_event_bus.py::TestEpisodeSummarizer::test_truncate_noop_under_limit tests/test_event_bus.py::TestEpisodeSummarizer::test_truncate_preserves_first_last tests/test_event_bus.py::TestEpisodeSummarizer::test_truncate_prioritizes_decisions -v`
Expected: All 3 PASS

**Step 4: Run all summarizer tests for regressions**

Run: `uv run pytest tests/test_event_bus.py::TestEpisodeSummarizer -v`
Expected: All PASS

Note: Test 14 (`test_truncates_long_transcripts`) uses a 10000-char string of "x" repeated. With the new truncation, this string has no `\n\n` separators so `turns` will be a single element. The `len(turns) <= 2` branch will fire, returning `transcript[:max_chars]`. This won't contain `[... middle truncated ...]`. **This test will break.** Update it:

```python
@pytest.mark.asyncio
async def test_truncates_long_transcripts(self):
    """14. Truncates long transcripts (>8000 chars)."""
    summarizer, heart, bus, http_client = self._make_summarizer()
    heart.get_episode = AsyncMock(
        return_value=MagicMock(summary="opening", structured_summary=None)
    )
    heart.update_episode_summary = AsyncMock()

    summary_json = {"title": "T", "summary": "S", "key_points": [], "outcome": "resolved", "topics": []}
    captured_prompts: list[str] = []

    async def capture_post(url, **kwargs):
        body = kwargs.get("json", {})
        msg_content = body.get("messages", [{}])[0].get("content", "")
        captured_prompts.append(msg_content)
        return _mock_httpx_response(200, _llm_response(json.dumps(summary_json)))

    http_client.post = capture_post

    # Use turns separated by \n\n so truncation can split them
    long_transcript = "\n\n".join([f"User: Turn {i} " + "x" * 400 for i in range(30)])
    event = _make_event(
        "session_ended",
        data={"episode_id": str(uuid4()), "transcript": long_transcript},
    )
    await summarizer.handle(event)

    # The prompt sent to LLM should be truncated (shorter than original)
    assert len(captured_prompts) == 1
    assert len(captured_prompts[0]) < len(long_transcript)
```

**Step 5: Commit**

```bash
git add nous/handlers/episode_summarizer.py tests/test_event_bus.py
git commit -m "feat(008.4): priority-based transcript truncation"
```

---

### Task 8: Write failing test for `_build_decision_context`

**Files:**
- Modify: `tests/test_event_bus.py` — add tests to `TestEpisodeSummarizer`

**Step 1: Write the failing tests**

Add these tests. Note: `_make_summarizer` needs updating for Brain param (done in Task 9).
For now, write the tests that will fail:

```python
@pytest.mark.asyncio
async def test_build_decision_context_with_decisions(self):
    """008.4: Decision context includes linked decisions."""
    from nous.handlers.episode_summarizer import EpisodeSummarizer

    brain = AsyncMock()
    brain.get = AsyncMock(return_value=MagicMock(
        description="Use PostgreSQL for storage",
        category="architecture",
        stakes="high",
        confidence=0.9,
    ))
    heart = AsyncMock()
    heart.get_episode = AsyncMock(return_value=MagicMock(
        decision_ids=[uuid4()],
    ))
    summarizer, _, _, _ = self._make_summarizer(heart=heart, brain=brain)

    result = await summarizer._build_decision_context(str(uuid4()))
    assert "Decisions made during this episode:" in result
    assert "Use PostgreSQL for storage" in result
    assert "architecture" in result

@pytest.mark.asyncio
async def test_build_decision_context_no_decisions(self):
    """008.4: Empty string when no decisions linked."""
    from nous.handlers.episode_summarizer import EpisodeSummarizer

    brain = AsyncMock()
    heart = AsyncMock()
    heart.get_episode = AsyncMock(return_value=MagicMock(decision_ids=[]))
    summarizer, _, _, _ = self._make_summarizer(heart=heart, brain=brain)

    result = await summarizer._build_decision_context(str(uuid4()))
    assert result == ""

@pytest.mark.asyncio
async def test_build_decision_context_error_returns_empty(self):
    """008.4: Returns empty string on Brain errors."""
    from nous.handlers.episode_summarizer import EpisodeSummarizer

    brain = AsyncMock()
    brain.get = AsyncMock(side_effect=Exception("Brain unavailable"))
    heart = AsyncMock()
    heart.get_episode = AsyncMock(return_value=MagicMock(decision_ids=[uuid4()]))
    summarizer, _, _, _ = self._make_summarizer(heart=heart, brain=brain)

    result = await summarizer._build_decision_context(str(uuid4()))
    assert result == ""
```

**Step 2: Run to verify failure**

Run: `uv run pytest tests/test_event_bus.py::TestEpisodeSummarizer::test_build_decision_context_with_decisions -v`
Expected: FAIL — `_make_summarizer` doesn't accept `brain`, and `_build_decision_context` doesn't exist

**Step 3: Commit failing tests**

```bash
git add tests/test_event_bus.py
git commit -m "test: 008.4 failing tests for decision context injection"
```

---

### Task 9: Implement Brain injection and `_build_decision_context`

**Files:**
- Modify: `nous/handlers/episode_summarizer.py:10-14` — add Brain import
- Modify: `nous/handlers/episode_summarizer.py:48-59` — add `brain` to constructor
- Modify: `nous/handlers/episode_summarizer.py:104-141` — add `_build_decision_context`, update `_generate_summary`
- Modify: `nous/main.py:109` — pass `brain` to constructor
- Modify: `tests/test_event_bus.py:260-270` — update `_make_summarizer` helper

**Step 1: Add Brain import**

In `episode_summarizer.py`, add after line 22:

```python
from nous.brain.brain import Brain
```

**Step 2: Update constructor to accept Brain**

Replace `__init__` (lines 48-59):

```python
def __init__(
    self,
    heart: Heart,
    brain: Brain | None,
    settings: Settings,
    bus: EventBus,
    http_client: httpx.AsyncClient | None = None,
):
    self._heart = heart
    self._brain = brain
    self._settings = settings
    self._bus = bus
    self._http = http_client
    bus.on("session_ended", self.handle)
```

**Step 3: Add `_build_decision_context` method**

Add after `_truncate_transcript`:

```python
async def _build_decision_context(self, episode_id: str) -> str:
    """008.4: Fetch decisions linked to this episode for richer summarization."""
    if not self._brain:
        return ""

    try:
        episode = await self._heart.get_episode(UUID(episode_id))
        if not episode or not episode.decision_ids:
            return ""

        lines = ["Decisions made during this episode:"]
        for decision_id in episode.decision_ids:
            d = await self._brain.get(decision_id)
            if d:
                lines.append(
                    f"- [{d.category}/{d.stakes}] {d.description} "
                    f"(confidence: {d.confidence})"
                )

        return "\n".join(lines) if len(lines) > 1 else ""
    except Exception:
        logger.debug("Failed to build decision context for episode %s", episode_id)
        return ""
```

**Step 4: Update `handle()` to build decision context and pass to `_generate_summary`**

In `handle()`, change line 81 from:
```python
summary = await self._generate_summary(transcript)
```
To:
```python
decision_context = await self._build_decision_context(episode_id)
summary = await self._generate_summary(transcript, decision_context)
```

**Step 5: Update `_generate_summary` signature and prompt formatting**

Change signature to accept `decision_context`:
```python
async def _generate_summary(self, transcript: str, decision_context: str = "") -> dict[str, Any] | None:
```

And update the prompt format line:
```python
prompt = _SUMMARY_PROMPT.format(transcript=transcript, decision_context=decision_context)
```

**Step 6: Update `main.py` constructor call (line 109)**

Change:
```python
EpisodeSummarizer(heart, settings, bus, handler_http)
```
To:
```python
EpisodeSummarizer(heart, brain, settings, bus, handler_http)
```

**Step 7: Update test helper `_make_summarizer`**

In `tests/test_event_bus.py`, update `TestEpisodeSummarizer._make_summarizer` (line 260):

```python
def _make_summarizer(self, heart=None, settings=None, bus=None, http_client=None, brain=None):
    from nous.handlers.episode_summarizer import EpisodeSummarizer

    heart = heart or AsyncMock()
    brain = brain or AsyncMock()
    brain.get = brain.get if hasattr(brain, 'get') and not isinstance(brain.get, MagicMock) else AsyncMock(return_value=None)
    settings = settings or _mock_settings()
    bus = bus or MagicMock(spec=EventBus)
    bus.on = MagicMock()
    bus.emit = AsyncMock()
    http_client = http_client or AsyncMock(spec=httpx.AsyncClient)
    summarizer = EpisodeSummarizer(heart, brain, settings, bus, http_client)
    return summarizer, heart, bus, http_client
```

**Step 8: Run decision context tests**

Run: `uv run pytest tests/test_event_bus.py::TestEpisodeSummarizer::test_build_decision_context_with_decisions tests/test_event_bus.py::TestEpisodeSummarizer::test_build_decision_context_no_decisions tests/test_event_bus.py::TestEpisodeSummarizer::test_build_decision_context_error_returns_empty -v`
Expected: All 3 PASS

**Step 9: Run all summarizer tests for regressions**

Run: `uv run pytest tests/test_event_bus.py::TestEpisodeSummarizer -v`
Expected: All PASS

**Step 10: Commit**

```bash
git add nous/handlers/episode_summarizer.py nous/main.py tests/test_event_bus.py
git commit -m "feat(008.4): inject decision context into episode summaries"
```

---

### Task 10: Run full test suite for PR 2

**Step 1: Run all event bus tests**

Run: `uv run pytest tests/test_event_bus.py -v`
Expected: All tests PASS

**Step 2: Run episode tests**

Run: `uv run pytest tests/test_episodes.py -v`
Expected: All tests PASS

**Step 3: Run full test suite**

Run: `uv run pytest tests/ -v --timeout=60`
Expected: All tests PASS

**Step 4: Create PR 2 branch**

```bash
git checkout -b feat/008.4-brain-context
git push -u origin feat/008.4-brain-context
```
