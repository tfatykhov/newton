# Research Note 001: Foundations

*What would an AI agent look like if Minsky designed it?*

## The Problem with Current Agents

Today's AI agents fall into two camps:

**Stateless Reactors** — Most chatbots and assistants. They process input, generate output, and forget. Each conversation starts from zero. No learning, no growth, no continuity.

**Memory-Augmented Reactors** — Agents with RAG, vector stores, or conversation history. Better, but fundamentally still reactive. They remember *what happened* but don't learn *how to think better*. Adding more memory doesn't make them smarter — it makes them more informed, which is a different thing entirely.

**What's missing is administration.** Per Minsky's Papert's Principle: "Some of the most crucial steps in mental growth are based not simply on acquiring new skills, but on acquiring new administrative ways to use what one already knows."

A child who knows that liquid volume doesn't change when poured into a different glass *still gets fooled by the tall glass* — because they lack the administrative agent that overrides the perceptual signal. No amount of additional knowledge fixes this. Only a better manager does.

## The Newton Hypothesis

**An AI agent that implements Minsky's Society of Mind principles — using Cognition Engines as its memory substrate — will demonstrate qualitatively different behavior from memory-augmented LLM agents.**

Specifically, it should:
1. **Improve its own decision-making over time** (not just accumulate information)
2. **Recognize and avoid past failure modes** (censors, not just memory)
3. **Switch cognitive strategies based on context** (frames, not just prompts)
4. **Maintain coherent identity across sessions** (slow-changing agencies)
5. **Know what it doesn't know** (calibrated confidence)

## Core Architecture Mapping

### Minsky → Newton → Implementation

**Agents (Ch 1-3)**
- Mind is a society of simple agents, none intelligent alone
- Newton: Each capability (search, code, email, memory) is an agent
- Implementation: Tool functions, sub-agent spawns, cron jobs

**K-Lines (Ch 8)**
- Memory works by re-activating the agents that were active during learning
- Newton: Context bundles with level-bands reconnect the agent to past mental states
- Implementation: Markdown files with upper/core/lower fringe zones, activated by semantic search

**Censors & Suppressors (Ch 9)**
- Learning from failure means building agents that *prevent* actions, not just agents that *perform* them
- Newton: Guardrails that block (not warn). Escalation path: warn → block → hard-block
- Implementation: Cognition Engines guardrail system with configurable severity

**Papert's Principle (Ch 10)**
- Growth is administrative. New capabilities are detours around old systems.
- Newton: Git hooks, pre-action protocols, quality gates — intercepts that add structure without replacing functionality
- Implementation: Pre-push hooks, mandatory deliberation, quality scoring on decisions

**Frames (Ch 25)**
- One interpretation at a time. Can't hold two simultaneously.
- Newton: Explicit frame selection at the start of each task. Frame-splitting via sub-agents for parallel perspectives.
- Implementation: Frame templates (Devil's Advocate, Optimist, Historian), spawn protocol

**B-Brains (Ch 6)**
- A system that watches itself think. Not too tightly coupled (instability risk).
- Newton: Deliberation traces + calibration + drift detection. Layered monitoring.
- Implementation: CSTP deliberation tracker (B-brain), calibration system (C-brain), drift detection (D-brain)

**Parallel Bundles (Ch 18)**
- Multiple independent reasons are more robust than one logical chain
- Newton: Decisions require diverse reason types. Quality scoring penalizes single-type reasoning.
- Implementation: ReasonType diversity in decision records, Brier scoring

**Polynemes (Ch 19)**
- One signal, many agencies, each interprets independently
- Newton: Tags are polynemes. "minsky" means something different to ChromaDB (similarity search), to category filters (knowledge management), to the pattern matcher (theoretical framework)
- Implementation: Tag system in Cognition Engines

**Nemes & Micronemes (Ch 20)**
- Tiny features that constrain search without being "meanings" themselves
- Newton: Bridge definitions (structure + function) act as micronemes. They narrow recall without being the recalled thing.
- Implementation: Bridge extractors in Cognition Engines, auto-generated from decision text

**Pronomes (Ch 21)**
- Separate what things are (assignment) from what to do with them (action)
- Newton: Sub-agent templates separate context-filling from behavior
- Implementation: Frame templates with slots for context vs fixed action patterns

## What Cognition Engines Already Provides

Newton doesn't build memory from scratch. It uses Cognition Engines as a substrate:

| Need | Cognition Engines Feature |
|------|--------------------------|
| "What did I decide before?" | Semantic decision search (hybrid: vector + keyword) |
| "Should I do this?" | Guardrail system with block/warn levels |
| "How confident should I be?" | Calibration with Brier scoring |
| "What was I thinking?" | Deliberation traces with micro-thoughts |
| "How do decisions relate?" | Graph store with auto-linking |
| "Am I getting worse?" | Drift detection |
| "What patterns repeat?" | Bridge definitions (structure/function) |

## What Newton Adds On Top

| Need | Newton Addition |
|------|----------------|
| "What kind of problem is this?" | Frame selection engine |
| "What context do I need?" | K-line activation with level-bands |
| "How should I think about this?" | Parallel frame-splitting protocol |
| "Am I growing?" | Administrative growth tracking |
| "What should I NOT do?" | Censor registry (beyond guardrails) |
| "Who am I?" | Identity continuity (slow-changing agencies) |
| "How do I manage myself?" | Meta-cognitive administration layer |

## Open Questions

### 1. The Bootstrap Problem
A Newton agent needs structure to think well, but the structure itself requires thought to build. How does an agent bootstrap its own cognitive architecture?

**Possible answer:** Start with a minimal "seed" architecture (basic K-lines, simple guardrails, default frames) and let it grow through Papert's Principle. The agent adds administrative structure as it encounters situations the seed can't handle.

### 2. The Rigidity Problem
Too much structure can make an agent rigid. Minsky warns about this — overly specific K-lines don't transfer to new situations. The upper and lower fringes exist specifically to allow flexibility.

**Possible answer:** Quality scoring should penalize both too-rigid and too-loose decisions. Monitor the "generality" of K-lines — do they activate in diverse contexts or only one?

### 3. The Measurement Problem
How do you measure "cognitive growth"? It's not just task completion rate. A Newton agent should be measurably better at:
- Confidence calibration over time
- Avoiding repeated mistakes
- Faster context activation for familiar domains
- Better frame selection (fewer wrong-frame-first situations)

### 4. The Transfer Problem
Can Newton's cognitive architecture transfer between different LLM backends? Does the architecture work with Claude, GPT, Gemini, Llama? Or is it tuned to specific model behaviors?

**Hypothesis:** The architecture should be model-agnostic because it operates at the prompt/tool level, not the weight level. But different models may need different "administrative agents" — e.g., GPT might need stronger censors around hallucination, Claude might need different frame-switching prompts.

### 5. The Multi-Agent Problem
If multiple Newton agents collaborate, do they share K-lines? Decision memories? Calibration data? How does a "society of societies" work?

**Possible answer:** Federated decision sharing (Cognition Engines F037) + K-line exchange protocols. Agents maintain independent identities but can subscribe to shared knowledge domains.

## Next Steps

1. Define the minimal viable architecture (what's the smallest Newton that's still Newton?)
2. Map existing Emerson implementation to Newton components (what's already built?)
3. Identify gaps between current state and Newton vision
4. Design the frame selection engine
5. Design the K-line activation API
6. Prototype with a simple agent and measure growth

---

*"Intelligence is our name for whichever of those wonderful things we admire but don't yet understand."* — Minsky, Ch 7
