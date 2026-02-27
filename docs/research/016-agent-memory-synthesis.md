# 016: LLM Agent Memory — A 2025–2026 Field Synthesis

**Papers:** 9 papers reviewed (see index below)
**Date reviewed:** 2026-02-27
**Relevance:** Critical — defines the state of the art across all dimensions of agent memory: representation, retrieval, compression, governance, and cognitive architecture

---

## Paper Index

| ID | Title | Source | Date |
|----|-------|--------|------|
| P1 | A-MEM: Agentic Memory for LLM Agents | arXiv:2502.12110 (NeurIPS 2025) | Feb 2025 |
| P2 | Episodic Memory is the Missing Piece for Long-Term LLM Agents | arXiv:2502.06975 | Feb 2025 |
| P3 | Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory | arXiv:2504.19413 | Apr 2025 |
| P4 | Procedural Memory Is Not All You Need | arXiv:2505.03434 (ACM UMAP '25) | May 2025 |
| P5 | MAP: Modular Agentic Planner | Nature Communications 2025 | 2025 |
| P6 | Structured Cognitive Loop (SCL) | arXiv:2511.17673 | Nov 2025 |
| P7 | Memory in the Age of AI Agents: A Survey | arXiv:2512.13564 | Dec 2025 |
| P8 | ACC: Agent Cognitive Compressor | arXiv:2601.11653 | Jan 2026 |
| P9 | SYNAPSE: A Unified Memory Architecture | arXiv:2601.02744 | Jan 2026 |

---

## 1. Paper Summaries

### P1 — A-MEM: Agentic Memory for LLM Agents
**arXiv:2502.12110 · Xu, Liang, Mei, Gao, Tan, Zhang · NeurIPS 2025**

A-MEM replaces flat memory stores with a **Zettelkasten-inspired dynamic knowledge network**. When a new memory is encoded, the agent generates a comprehensive note containing contextual descriptions, keywords, and tags, then actively searches historical memories for semantic connections. Linked memories are updated bidirectionally — a new fact can trigger re-evaluation of the context of older ones.

**Key findings:**
- Outperforms SOTA baselines across 6 foundation models on longitudinal task evaluation
- Memory evolution (retroactive linking) provides measurable gains over append-only approaches
- Agent-driven linking outperforms static embedding similarity for surface-level fuzzy recall

**Mechanism:** Three-step write path — (1) generate note with metadata and keywords, (2) search existing memories for connection candidates, (3) update linked memories' contextual representations. Retrieval becomes graph traversal rather than k-NN lookup.

---

### P2 — Episodic Memory is the Missing Piece for Long-Term LLM Agents
**arXiv:2502.06975 · Mathis Pink et al. (7 authors, UT Austin) · Feb 2025**

A position paper making the case that episodic memory — **single-shot learning of instance-specific experience** — is the missing capability that separates long-lived agents from session-scoped ones. Semantic memory (facts), procedural memory (skills), and working memory (context) are all well-represented in current architectures. Episodic memory is not.

**Five properties of episodic memory identified:**
1. **Temporally indexed** — memories are ordered in time, not just by relevance
2. **Instance-specific** — records what happened in this particular interaction, not the general case
3. **Single-shot encodable** — must be learnable from one exposure, not repeated reinforcement
4. **Inspectable** — agents should be able to introspect and describe their own past experiences
5. **Compositional** — episodes can be combined to reason about multi-step, longitudinal patterns

**Key claim:** Without episodic memory, agents reset at every session. They can accumulate semantic knowledge but cannot reason about their own history — they cannot answer "the last time I tried this approach, it failed because…"

---

### P3 — Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory
**arXiv:2504.19413 · Chhikara, Khant, Aryan, Singh, Yadav · Apr 2025**

Mem0 presents a scalable production architecture for long-term memory. The system **dynamically extracts, consolidates, and retrieves salient information** from ongoing conversations rather than buffering full transcripts. An enhanced graph-based variant models relational structure between entities and concepts.

**Evaluated on LOCOMO benchmark vs. 6 baseline categories (including full-context replay, RAG, and OpenAI's memory system):**
- 26% relative improvement over OpenAI's native memory on LLM-as-a-Judge metric
- Graph-enhanced variant adds ~2% on top of base Mem0
- 91% lower p95 latency compared to full-context replay
- 90%+ token cost reduction vs. full-context methods

**Key insight:** Structured persistent memory outperforms full-context replay for long-term coherence. The extraction step — choosing *what* to save — is more important than the storage mechanism itself. Mem0 shows that targeted compression beats brute-force retention at production scale.

---

### P4 — Procedural Memory Is Not All You Need
**arXiv:2505.03434 · Wheeler, Jeunen · ACM UMAP '25 Workshop · May 2025**

A conceptual paper arguing that LLMs' default competence is **procedural memory** — pattern matching on training distributions. This works for "tame" environments where rules are fixed and feedback is clear. It fails catastrophically in **"wicked" environments** — shifting rules, ambiguous feedback, novel situations.

**The core argument:**
- LLMs are pre-trained pattern engines. Prompt engineering just selects which patterns to activate.
- Procedural memory alone cannot generalize to out-of-distribution problem shapes
- For adaptive agents, semantic memory (world models) and associative learning (concept connection) must be explicitly architectured in, not assumed from pretrained weights

**Recommended architecture:** Modular separation of:
1. Procedural module — task execution, tool use, pattern application
2. Semantic module — explicit world model, user model, domain knowledge
3. Associative module — analogical reasoning, cross-domain pattern transfer

**Implication:** An agent relying only on its base LLM for memory is brittle. Nous's multi-store Heart (facts, episodes, procedures, censors) maps directly to this recommendation.

---

### P5 — MAP: Modular Agentic Planner
**Nature Communications 2025 · Brain-inspired agentic architecture**

MAP frames long-horizon planning as a **coordination problem**, not a capability problem. LLMs can individually perform the cognitive sub-functions of planning (conflict detection, state prediction, task decomposition, goal evaluation) but struggle to orchestrate them autonomously in sequence.

MAP decomposes planning into five modules modeled on prefrontal cortex functions:
1. **Conflict Monitoring** — detects contradictions and inconsistencies in current state
2. **State Prediction** — forecasts consequences of candidate actions
3. **State Evaluation** — scores predicted states against goal criteria
4. **Task Decomposition** — breaks multi-step goals into primitive sub-tasks
5. **Task Coordination** — sequences sub-tasks, manages dependencies and re-planning

**Evaluated on:** Graph traversal, Tower of Hanoi, PlanBench, StrategyQA.

**Results:** Outperforms standard LLM prompting and competitive multi-agent planning baselines. Importantly, MAP works with smaller/cheaper LLMs when modules are well-separated — specialization compensates for raw model capability.

**Key insight:** The bottleneck in agentic planning is not intelligence, it is **coordination**. Modular decomposition with explicit handoffs between cognitive sub-functions enables capabilities that emerge from structure, not from scale.

---

### P6 — Structured Cognitive Loop (SCL)
**arXiv:2511.17673 · Nov 2025 · Open-source, GPT-4o demo**

SCL proposes a **five-phase modular cognitive loop** as an architectural standard for safe, traceable LLM agents:

```
Retrieval → Cognition → Control → Action → Memory
(R-CCAM)
```

Each phase is an explicit module with defined inputs and outputs. The critical addition is **Soft Symbolic Control** in the Control phase — a governance layer that applies symbolic (rule-based) constraints to probabilistic inference outputs before actions are committed. This preserves neural flexibility while enforcing policy compliance.

**Results:**
- Zero policy violations in evaluation scenarios
- Elimination of redundant/spurious tool calls
- Full decision traceability (every action can be traced to a reasoning step and a constraint check)

**Key insight:** The distinction between *cognition* (probabilistic inference) and *control* (symbolic constraint application) is doing a lot of work here. Separating these phases prevents the agent from rationalizing around constraints post-hoc — a pattern common in unconstrained CoT agents.

---

### P7 — Memory in the Age of AI Agents: A Survey
**arXiv:2512.13564 · 47 authors · Dec 2025 (updated Jan 2026)**

The field's most comprehensive unified treatment of agent memory. Organizes the space across three analytical lenses:

**Lens 1: Memory Forms** (how memory is physically represented)
- **Token-level** — explicit, discrete representations (files, databases, retrieved documents)
- **Parametric** — implicit memory encoded in model weights via fine-tuning
- **Latent** — compressed representations in hidden states (soft prompts, KV cache)

**Lens 2: Memory Functions** (what the memory is for)
- **Factual** — grounded knowledge about the world
- **Episodic** — records of specific past experiences
- **Procedural** — skills, workflows, how-to knowledge
- **Semantic** — concepts, relationships, world models
- **Working** — active in-context information for the current task

**Lens 3: Memory Processes** (how memory is operated on)
- **Encoding** — converting experience to stored representation
- **Consolidation** — moving from working to long-term storage, with compression/generalization
- **Retrieval** — surfacing relevant memory for the current context
- **Forgetting / Pruning** — removing stale, low-value, or contradictory memories
- **Evolution** — updating existing memories as new information arrives

**Notable research gaps identified:**
- No standard benchmark for longitudinal agent memory performance
- Most systems optimize retrieval in isolation; consolidation and evolution are underspecified
- Forgetting is treated as a degenerate case, not a designed capability
- Parametric + token-level hybrid systems remain an unsolved open problem

---

### P8 — ACC: Agent Cognitive Compressor
**arXiv:2601.11653 · Fouad Bousetouane · Jan 2026**

ACC introduces a **bio-inspired memory controller** designed to prevent memory-induced drift in long multi-turn workflows. The central insight: transcript replay (the dominant approach) causes agents to re-weight unverified, out-of-date, or contradictory content at each turn. Over hundreds of turns, this produces hallucination accumulation and behavioral drift.

**The ACC mechanism:**
- Replaces transcript replay with a **bounded internal state** — a fixed-size compressed representation of agent knowledge and task context
- The state is updated *online* at each turn, not re-read from scratch
- **Artifact recall** (re-surfacing specific tool outputs or document content) is explicitly separated from **state commitment** (what becomes part of the agent's persistent model)
- Only verified, agent-confirmed content is committed to the persistent state

**Evaluated across:** IT operations automation, cybersecurity response workflows, healthcare triage.

**Results:** Significantly lower hallucination rate and behavioral drift vs. transcript replay and RAG-based memory at 50+ turn horizons.

**Key insight:** The distinction between *recall* (temporary re-surfacing for a task) and *commitment* (permanent integration into the agent's state) is fundamental. Most systems treat these as the same operation. Separating them is what makes ACC robust to multi-turn noise.

---

### P9 — SYNAPSE: A Unified Memory Architecture
**arXiv:2601.02744 · Jan 2026**

SYNAPSE models agent memory as a **dynamic graph where relevance emerges from spreading activation** rather than pre-computed vector similarity. Memory items are nodes; semantic, temporal, and causal relationships are edges. Retrieval activates a seed node and propagates activation through the graph; lateral inhibition suppresses competing activations; temporal decay de-weights older, less-reinforced memories.

**Triple Hybrid Retrieval:** SYNAPSE fuses three signals:
1. Geometric embedding similarity (dense vector recall)
2. Spreading-activation graph traversal
3. Temporal recency weighting

**Solves the "Contextual Tunneling" problem:** Standard dense retrieval surfaces memories that are lexically similar but temporally or causally stale. SYNAPSE's graph traversal allows retrieval to "tunnel through" weak but structurally relevant connections.

**Evaluated on:** LoCoMo benchmark (longitudinal conversation memory).
**Results:** Outperforms SOTA on both temporal coherence and multi-hop reasoning questions.

**Key insight:** Relevance is not a static property of a memory item — it is a relational property that depends on context, recency, and causal structure. Vector similarity alone cannot model this. Graph-based spreading activation can.

---

## 2. Cross-Cutting Themes

### Theme A: The Memory Architecture Gap

All nine papers converge on a shared diagnosis: **current agent memory is dominated by simple context replay**. Full-transcript injection is the default because it is easy to implement, not because it is effective. As task horizons lengthen, its failure modes compound:

- P8 (ACC): transcript replay causes drift and hallucination accumulation at 50+ turns
- P3 (Mem0): full-context methods have 91% higher latency and 90% more token cost
- P2 (Episodic Memory): without episodic indexing, agents cannot reason about their own history
- P9 (SYNAPSE): context replay cannot model temporal or causal relevance

The implication for Nous: our Heart system (explicit multi-store memory) is architecturally correct. The gap is in memory *lifecycle* — consolidation, evolution, forgetting — which most current implementations skip.

---

### Theme B: Separation of Concerns in Memory Operations

Multiple papers independently arrive at the same structural principle: **memory operations need to be separated by function**, not bundled together.

| Concern | Paper | Separation identified |
|---------|-------|-----------------------|
| Recall vs. commitment | P8 (ACC) | Artifact recall ≠ state commitment |
| Encoding vs. retrieval | P7 (Survey) | Consolidation is its own operation |
| Memory vs. planning | P5 (MAP) | Cognitive sub-functions need explicit interfaces |
| Retrieval vs. governance | P6 (SCL) | Cognition ≠ control |
| Linking vs. storage | P1 (A-MEM) | Memory network evolution is a separate operation from storage |

This pattern suggests a design law: **every time a memory operation is bundled, a failure mode is hidden**. Nous's current learn_fact / record_decision / create_censor separation is a good start, but consolidation and evolution are still underdeveloped.

---

### Theme C: Graph Structures Outperform Flat Vectors for Long-Horizon Memory

Three papers (A-MEM, Mem0 graph variant, SYNAPSE) all demonstrate that **graph-based memory representations outperform flat vector stores** for tasks requiring multi-hop, temporal, or causal reasoning:

- A-MEM: linked Zettelkasten notes outperform k-NN lookup for associative recall
- Mem0: graph variant adds ~2% over already-strong base on LOCOMO
- SYNAPSE: graph traversal + lateral inhibition solves contextual tunneling entirely

The gain is clearest on temporal and multi-hop queries — the exact scenarios where long-lived agents operate. Flat vector retrieval plateaus; graph retrieval scales with the depth of relationships encoded.

**For Nous:** The current Heart uses Postgres with vector embeddings. A graph overlay — storing explicit links between facts, episodes, and decisions — would capture the structural relationships that embedding similarity misses.

---

### Theme D: Cognitive Neuroscience as Design Language

Five of the nine papers draw explicitly on cognitive neuroscience frameworks (P2, P4, P5, P6, P8). This is not metaphor — the structural insights are being translated into working architectures:

| Neuroscience concept | Paper | Architectural translation |
|---------------------|-------|--------------------------|
| Episodic vs. semantic memory | P2, P4, P7 | Separate stores with different indexing |
| Prefrontal cortex modules | P5 (MAP) | Conflict monitoring, state prediction, task decomposition |
| Working memory limits | P7, P8 | Bounded internal state, committed vs. retrieved |
| Spreading activation | P9 (SYNAPSE) | Graph traversal with lateral inhibition |
| Soft symbolic control | P6 (SCL) | Governance layer over probabilistic inference |

This validates the Nous design philosophy: Minsky's Society of Mind and cognitive science are not decorative framing — they are a source of proven structural solutions.

---

### Theme E: Forgetting as a First-Class Operation

The survey (P7) identifies forgetting as systematically neglected. But several papers show the consequences of not designing it explicitly:

- P1 (A-MEM): retroactive relinking implies some older representations become stale and need pruning
- P8 (ACC): information committed to the internal state without verification accumulates as noise
- P7 (Survey): "forgetting" is listed as an explicit memory process alongside encoding and retrieval

Current Nous behavior: facts are stored indefinitely, episodes are archived but not pruned, and contradictions are detected only when two facts are directly compared. There is no active expiry or confidence decay mechanism.

**The missing pattern:** A confidence decay function — facts and episodes that are not reinforced by new evidence should have their retrieval weight reduced over time, with deletion after sustained low confidence.

---

## 3. Implications for Nous

### 3.1 What We Already Have (Validated)

The nine papers collectively validate Nous's foundational choices:

| Nous feature | Validation |
|--------------|-----------|
| Multi-type Heart (facts, episodes, procedures, censors) | P2, P4, P7 — modular memory types are the right structure |
| Explicit censor system | P6 (SCL Soft Symbolic Control), P8 (ACC commitment gate) |
| Decision recording with deliberation | P5 (MAP explicit phase handoffs) |
| Episode summaries | P2 (episodic indexing), P7 (consolidation as a memory process) |
| Confidence scores on facts | P7, P8 (commitment gating implies confidence thresholds) |

### 3.2 What We Are Missing

**Gap 1 — Memory evolution (A-MEM)**
Nous currently appends facts. When new information arrives that updates a prior fact, we create a new fact — we do not update the old one's contextual representation. A-MEM's retroactive linking model suggests we should: when encoding a new fact, search for related facts and update their `tags`, `confidence`, or context fields if the new information revises them.

**Gap 2 — Episodic memory indexing (P2)**
Our episodes are stored with `started_at` and `ended_at` timestamps, but they are not retrievable by temporal sequence or episode-to-episode reasoning chains. The five properties from P2 — temporal indexing, instance-specificity, single-shot encodability, inspectability, compositionality — should be an explicit checklist against which we test our episode implementation.

**Gap 3 — Committed vs. retrieved memory (ACC)**
Nous does not distinguish between a fact surfaced into context for a turn (retrieval) and a fact the agent has confirmed as accurate and integrated into its persistent model (commitment). Everything that gets encoded via `learn_fact` has the same status. ACC's architecture suggests we should add a `committed: bool` field and treat uncommitted facts as working memory, not persistent memory.

**Gap 4 — Memory lifecycle / forgetting (P7, ACC)**
Confidence decay is not implemented. Facts and episodes accumulate indefinitely. The Memory Lifecycle feature (F008) addresses archiving, but not active pruning based on staleness signals. We need a decay model: facts not reinforced within N interactions should have their retrieval weight reduced; after M interactions at low confidence, they should be candidates for deletion.

**Gap 5 — Graph overlay for relational retrieval (A-MEM, Mem0, SYNAPSE)**
Our Postgres schema supports vector embeddings for semantic similarity, but has no explicit edge representation between memory items. Related facts, causal episode chains, and decision-to-fact links are implied by content but not structurally queryable. A lightweight edges table (source_id, target_id, relationship_type, strength) would unlock multi-hop retrieval without requiring a full graph database.

**Gap 6 — Planning module coordination (MAP)**
MAP's finding — that LLMs can execute cognitive sub-functions individually but struggle to coordinate them — maps directly onto the Nous frame system. Our frames switch the cognitive mode (conversation, analysis, coding, etc.) but they do not structure the sub-phases *within* an agentic task: conflict check → state prediction → action selection → memory update. Explicitly sequencing these in the agent loop for high-stakes tasks would mirror MAP's gains.

### 3.3 Priority Assessment

| Gap | Effort | Value | Priority |
|-----|--------|-------|----------|
| G4 — Memory lifecycle / forgetting | Medium | High | ⭐⭐⭐ — aligns with F008 |
| G3 — Committed vs. retrieved memory | Low | High | ⭐⭐⭐ — small schema change, large correctness gain |
| G2 — Episodic indexing | Low | High | ⭐⭐⭐ — P2's five properties are a direct spec checklist |
| G1 — Memory evolution (relinking) | Medium | Medium | ⭐⭐ — deferred to F010+ |
| G5 — Graph overlay | High | High | ⭐⭐ — powerful but a significant architectural investment |
| G6 — Planning loop phases | Medium | Medium | ⭐⭐ — relevant for F011+ agentic task flows |

---

## 4. Actionable Next Steps

1. **Audit episode implementation against P2's five properties** — check if Nous episodes are temporally indexable, instance-specific, single-shot encodable, inspectable, and compositional. Open a follow-up issue for each unmet property.

2. **Add `committed` flag to facts table** (Gap 3) — separates retrieval artifacts from persistent knowledge. Simple migration: `ALTER TABLE facts ADD COLUMN committed boolean DEFAULT true`. Mark facts from automated extraction as `committed = false` until confirmed.

3. **Spec F008 memory lifecycle update** — incorporate P7's forgetting model and P8's commitment gating. Add: confidence decay function, staleness signals, candidate-for-deletion state.

4. **Investigate lightweight graph edges** (Gap 5) — prototype an `edges` table in Postgres. Link: fact→fact (semantic), episode→episode (temporal chain), decision→fact (evidence), fact→decision (informed-by). Test against multi-hop retrieval queries.

5. **Memory evolution via relinking** (Gap 1) — on `learn_fact`, run a similarity search and propose tag/context updates to related facts above a confidence threshold. Flag for human review rather than auto-apply initially.

6. **Benchmark Nous on LOCOMO** — Mem0, SYNAPSE, and others all use LOCOMO for longitudinal memory evaluation. Running the benchmark on Nous would produce a concrete performance baseline and identify failure modes.

7. **Read SYNAPSE full paper** and assess whether a Postgres-native spreading activation retrieval algorithm is feasible, or whether a dedicated graph DB (e.g., Apache AGE for Postgres) is warranted.
