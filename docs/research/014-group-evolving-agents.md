# 014: Group-Evolving Agents (GEA) — Open-Ended Self-Improvement via Experience Sharing

**Paper:** Weng et al. (2026), UCSB. [arXiv:2602.04837](https://arxiv.org/html/2602.04837v1)
**Date reviewed:** 2026-02-23
**Relevance:** High — directly applicable to Nous multi-agent evolution and censor learning

## Core Idea

Treat a **group** of agents as the fundamental evolutionary unit, not individuals. Enable explicit experience sharing and reuse within the group at each evolutionary iteration. This replaces tree-structured evolution (where branches are isolated) with group-structured evolution (where discoveries propagate).

## Key Insight

> "AI agents are not biological individuals; why should their evolution remain constrained by biological paradigms?"

Tree-structured evolution wastes diversity — promising discoveries stay trapped in isolated branches and die as "short-lived variants." Group evolution consolidates complementary discoveries into sustained progress.

## Method (3 stages per iteration)

### 1. Parent Group Selection (Performance-Novelty)
- Represent each agent as binary task-success vector `z ∈ {0,1}^D`
- Novelty = average cosine distance to K nearest neighbors
- Score = `performance × √novelty` (performance-primary, novelty as mild bias)
- Select top-K agents to form parent group

### 2. Experience Collection
For each agent in the group, collect evolutionary traces:
- Framework-level code patches applied
- Predicted task patches for unsolved tasks
- Execution logs (tool invocation, workflow)
- Evaluation outcomes (failure modes)

### 3. Group Evolution
- **Aggregate** all traces into shared experience pool `S = ∪ T_j`
- Each agent **reflects** on shared pool → generates evolution directives
- Each agent **evolves** independently but informed by group experience
- Offspring that compile and pass basic tests → added to archive

## Results

| Benchmark | GEA | Tree-based (DGM) | Best Human-Designed |
|-----------|-----|-------------------|---------------------|
| SWE-bench Verified | **71.0%** | 56.7% | 71.8% (OpenHands+GPT-5) |
| Polyglot | **88.3%** | 68.3% | 52.0% (Aider+GPT-5) |

- Best GEA agent integrates 17 unique ancestors (28.3%) vs 9 for DGM (15%)
- Worst of GEA's top-5 (58.3%) beats DGM's single best (56.7%)
- Bug repair: 1.4 iterations (GEA) vs 5.0 (DGM)
- Improvements transfer across GPT and Claude model families

## Key Findings

1. **Diversity consolidation > diversity generation**: Both methods generate similar diversity, but GEA converts it into sustained progress while tree-based lets it die in branches
2. **Tool/workflow improvements, not prompt hacks**: All performance gains come from framework-level changes (new tools, modified workflows), not model-specific prompt engineering → transferable
3. **Accelerating returns**: As archive grows, more diverse experience available → stronger improvement per iteration. Performance curve steepens in later iterations.
4. **Robustness from sharing**: Faulty agents get repaired by observing better-performing peers. Group context acts as error correction.
5. **Cost-equivalent**: Same number of evolved agents as tree-based, similar compute cost (~$13K for SWE-bench, ~$1.5K for Polyglot)

## Connections to Nous

### Direct Mappings

| GEA Concept | Nous Equivalent | Notes |
|-------------|-----------------|-------|
| Experience sharing pool | Heart (shared memory) | Heart's episode/fact storage already enables cross-frame experience sharing |
| Performance-Novelty selection | Calibration + Bridge search | Performance = calibration accuracy; Novelty = cosine distance in bridge embeddings |
| Evolutionary traces | Decision deliberation traces | CSTP's pre_action → thoughts → outcome chain |
| Archive of agents | Decision graph | Historical decisions as "ancestors" that inform future ones |
| Group reflection | Frame-splitting protocol | Multiple cognitive frames analyzing shared evidence |
| Bug repair via peer experience | Censor learning (M1.3) | Guardrails that learn from peer agent outcomes |

### Implications for M1.3 (Censor Learning)

The GEA paper validates the core idea behind issue #8: guardrails/censors should evolve based on collective group experience, not just individual FP/FN counts. Specific applications:

1. **Group-informed censor evolution**: When multiple Nous agents share a Brain (F013), a guardrail's effectiveness should be measured across ALL agents, not per-agent
2. **Performance-Novelty for guardrail selection**: Don't just pick best-performing guardrails. Select guardrails that balance effectiveness with coverage diversity
3. **Experience pool for censor creation**: New censors should be generated from aggregated failure traces across the group, not just individual agent failures
4. **Accelerating censor quality**: As decision archive grows, censors should improve faster (same accelerating returns pattern)

### Implications for Cognitive Layer (004)

The reflection → evolution → action pipeline maps directly to the Nous Loop:
- **Reflect** = pre_turn (analyze context, activate relevant memories)
- **Evolve** = frame selection + context assembly (generate cognitive strategy)
- **Act** = LLM call with assembled context

The group-level experience sharing suggests the Cognitive Layer should:
- Pull decision patterns from ALL similar past decisions (not just the agent's own)
- Use bridge search to find structurally similar solutions from different problem domains
- Weight recent decisions higher but never fully isolate evolutionary "branches"

### Novel Insight: Accelerating Returns from Memory

GEA's most interesting finding for us: **performance improvement accelerates as the archive grows**. This suggests Nous agents should get *better at getting better* over time — exactly what Minsky's Ch. 10 (Papert's Principle) predicts. The administrative skill of using past experience should itself improve with accumulated experience.

This validates our design of measuring "mistake repetition rate" as the key growth metric. If the rate isn't decreasing over time, the agent isn't achieving the accelerating returns that group evolution predicts.

## Limitations & Critiques

1. **Coding-only evaluation**: No evidence this generalizes to reasoning, planning, or multi-modal tasks
2. **Fixed group size (K=2)**: Small groups. Unclear how scaling to larger groups affects diversity-performance tradeoff
3. **No analysis of harmful evolution**: What if shared experience propagates bad patterns? No censor/safety mechanism in the evolution loop
4. **Expensive**: $13K per SWE-bench run. Not practical for real-time agent improvement. Nous needs a lightweight version (sample-efficient learning from decisions, not full benchmark evaluation)
5. **Binary task representation**: Crude. Bridge definitions (structure+function) would be richer than binary success/fail vectors

## Actionable Takeaways for Nous

1. **Build group-aware censor learning into M1.3** — aggregate FP/FN across all agents sharing a Brain
2. **Use Performance-Novelty scoring for decision retrieval** — not just similarity, also diversity of retrieved context
3. **Track "ancestor count" metric** — how many historical decisions contribute to current decision quality
4. **Implement accelerating returns measurement** — plot learning curve over time, expect steepening (not flattening)
5. **Don't isolate cognitive frames** — share experience across frames like GEA shares across agents
