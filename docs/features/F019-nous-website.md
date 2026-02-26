# F019 — Nous Website

> **Status:** Planned
> **Priority:** P2
> **Depends on:** F018 (Agent Identity) — ship before launch
> **Estimated effort:** ~1 day design + ~2 days build
> **Target URL:** nous-framework.ai (or nous.ai / getnous.dev — TBD)

## Problem

Nous is a serious open-source framework with a genuinely novel foundation — Minsky's Society of Mind as a first-class architecture. But its only public front door right now is a GitHub README.

A README serves people who already found you. A website converts curious passers-by into actual adopters. For an open-source framework targeting developer adoption, that gap matters enormously.

The risk of waiting: other frameworks (LangChain, CrewAI, AutoGen) continue to dominate discovery simply because they have polished public presence — not because they're architecturally stronger.

## Goal

A fast, minimal, developer-first website that:

1. Communicates what Nous is in under 10 seconds
2. Earns credibility with technical developers through honest depth
3. Provides a clear path from curiosity → running agent
4. Cross-links with `cognition-engines.ai` as the companion decision layer

## Audience

**Primary:** Software engineers building AI agents who are frustrated with the shallowness of current frameworks (LangChain-style prompt plumbing). They want something architecturally honest.

**Secondary:** Technical leads and CTAs evaluating frameworks for team adoption. They need to understand the "why" quickly, trust the architecture, and hand it off to their devs.

## Positioning

**One-liner:**
> *"An AI agent framework that thinks, learns, and grows — grounded in Minsky's Society of Mind."*

**What Nous is NOT:**
- Not another prompt-chaining library
- Not a thin wrapper around OpenAI
- Not a cloud service you pay for

**What Nous IS:**
- An opinionated open-source framework
- Grounded in 40 years of cognitive science (Minsky, 1986)
- Embeds decision memory, structured recall, self-monitoring, and calibration as first-class concepts
- Runs entirely in your infrastructure

## Site Architecture

```
/                   → Homepage (hero + why + loop + quick start + CTA)
/docs               → Documentation (mirrors /docs in repo)
/concepts           → Deep dives: K-Lines, Censors, Calibration, Frames, B-Brain
/blog               → Posts (optional, Phase 2)
/cognition-engines  → Bridge page — relationship to cognition-engines.ai
```

---

## Page Designs

### 1. Homepage

#### Hero Section

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   [Nous project image — "Minds from Mindless Stuff"]│
│                                                     │
│   AI agents that think, learn, and grow.            │
│                                                     │
│   Nous is an open-source agent framework grounded   │
│   in Minsky's Society of Mind — with structured     │
│   memory, decision intelligence, and self-monitoring│
│   built in from the start.                         │
│                                                     │
│   [Get Started →]          [GitHub ↗]               │
│                                                     │
│   v0.1.0  ·  Apache 2.0  ·  11,800 lines  ·  424 tests │
└─────────────────────────────────────────────────────┘
```

**Design notes:**
- Use the existing `docs/nous-project-image.png` — it's striking and sets tone
- Social proof row (version, license, lines, tests) builds instant credibility with developers
- Two CTAs: one to docs (conversion), one to GitHub (credibility)

---

#### "Why Nous?" Section

Three columns, each grounded in a real failure mode of existing frameworks:

**Column 1: Beyond Stateless Reactors**
> Current agents receive a prompt, generate a response, and forget. Even agents with "memory" just store and retrieve text — there's no structure, no learning, no growth. Nous gives agents memory that mirrors how minds actually work.

**Column 2: Decision Intelligence, Not Just Storage**
> Nous agents record every significant decision with reasoning, confidence, and outcome. They track calibration. They learn to trust their own estimates. When they face the same situation again, they know what worked last time.

**Column 3: A Framework That Has a Why**
> Every design decision in Nous traces back to Minsky's Society of Mind (1986). K-Lines, Censors, Frames, B-Brains, Parallel Bundles — these aren't metaphors. They're the actual architectural components.

---

#### The Nous Loop

Visual treatment of the 7-step cognitive cycle. Not a wall of text — a diagram with hover/click to expand each step.

```
SENSE → FRAME → RECALL → DELIBERATE → ACT → MONITOR → LEARN
```

One-line description under each step:

| Step | Description |
|------|-------------|
| SENSE | Receive input — a message, an event, a timer |
| FRAME | Interpret through a cognitive frame. What kind of problem is this? |
| RECALL | Activate K-Lines — context bundles that reconstruct the relevant mental state |
| DELIBERATE | Query past decisions, check guardrails, record intent before acting |
| ACT | Execute. The B-brain watches the A-brain work. |
| MONITOR | Did the action match the intent? Were there surprises? |
| LEARN | Update memory, calibration, K-lines, and guardrails |

---

#### Memory Architecture Section

Use the existing mermaid diagram from the README, rendered as a visual (not code block). Four layers with brief description:

- **Slow (Identity)** — Who the agent is. Character, values, protocols.
- **Medium (Knowledge)** — Facts, K-Lines, Episodes. Accumulated expertise.
- **Fast (Working)** — Current turn context. What the agent is thinking right now.
- **Persistent (Intelligence)** — Decisions and calibration. How the agent improves.

> *"Most frameworks give agents fast memory. Nous gives agents all four layers."*

---

#### Concepts Grid

Six cards, one per core Minsky concept implemented in Nous:

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  K-Lines     │  │   Censors    │  │   Frames     │
│  Ch 8        │  │   Ch 9       │  │   Ch 25      │
│              │  │              │  │              │
│ Context      │  │ Guardrails   │  │ Active       │
│ bundles with │  │ that block   │  │ interpretation│
│ level-bands  │  │ before harm  │  │ lens         │
└──────────────┘  └──────────────┘  └──────────────┘

┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  B-Brains   │  │  Calibration │  │  Parallel    │
│  Ch 6        │  │              │  │  Bundles     │
│              │  │              │  │  Ch 18       │
│ Self-monitor │  │ Confidence   │  │ Multiple     │
│ layer that   │  │ tracking +   │  │ reasons >    │
│ watches agent│  │ Brier scores │  │ one chain    │
└──────────────┘  └──────────────┘  └──────────────┘
```

Each card links to `/concepts/{name}` for a deep dive.

---

#### Quick Start Section

Show the minimum viable code to get a Nous agent running. Target: 5 minutes from zero.

```bash
# Clone and start
git clone https://github.com/tfatykhov/nous
cd nous
cp .env.example .env  # add your ANTHROPIC_API_KEY
docker compose up
```

```bash
# Talk to your agent
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello — what do you remember about me?"}'
```

> Your agent is now running with structured memory, decision recording, and self-monitoring. Everything it learns persists across sessions.

Link: `[Read the full quickstart docs →]`

---

#### Relationship to Cognition Engines

Short section, not a full pitch — just establishing the connection clearly:

> Nous uses [Cognition Engines](https://cognition-engines.ai) as its decision memory substrate — the same decision intelligence principles, proven independently, now embedded as Nous's Brain organ. The shared asset is the philosophy, not the codebase. Nous Brain is a purpose-built embedded implementation optimized for in-process use with zero network overhead.

---

#### Stats Bar

```
~11,800 lines of Python  ·  424 tests  ·  18 Postgres tables  ·  12 REST endpoints  ·  v0.1.0
```

---

#### Footer

- GitHub link
- Cognition Engines link
- Apache 2.0 license note
- *"Built with Minsky and too much coffee ☕"*

---

### 2. /concepts Pages

One page per core concept. Template:

```
# K-Lines

> Minsky, Society of Mind, Chapter 8

## The Idea
[1-2 para plain English]

## Minsky's Original Insight
[Quote + brief context]

## How Nous Implements It
[Code snippet or diagram]

## Why It Matters for Agents
[Practical consequence — what breaks without this]
```

Pages to build:
- `/concepts/k-lines`
- `/concepts/censors`
- `/concepts/frames`
- `/concepts/b-brains`
- `/concepts/calibration`
- `/concepts/parallel-bundles`
- `/concepts/paperts-principle`

---

### 3. /cognition-engines Bridge Page

Exists to explain the relationship clearly — important for SEO and for developers who find one project and wonder about the other.

Content:
- What Cognition Engines is (standalone decision server, MCP integration)
- What Nous is (full agent framework, embedded Brain)
- How they relate (same philosophy, independent codebases)
- When to use which (CE: existing agent stack via MCP. Nous: new agent from scratch)

---

## Technical Spec

### Stack Recommendation

**Option A: Astro + Tailwind (recommended)**
- Static site generation — fast, cheap to host
- MDX support for concept docs
- Easy to maintain alongside the repo
- Deploy to Vercel/Netlify in minutes

**Option B: Docusaurus**
- Better if docs are the primary goal
- More opinionated, harder to make visually distinctive
- Solid choice if the site is mainly `/docs`

**Option C: plain HTML/CSS**
- Fastest to ship, hardest to maintain
- No build toolchain overhead
- Fine for MVP if design is simple

**Recommendation:** Astro. Docs-first but design-flexible. Minsky deserves a site that doesn't look like every other GitHub project.

### Hosting

- Vercel (free tier, instant deploys from GitHub)
- Custom domain — `nous-framework.ai` or `getnous.dev`
- HTTPS via Vercel automatically

### Assets Needed

- [ ] `docs/nous-project-image.png` — already exists, use in hero
- [ ] Architecture diagram — render the mermaid from README as SVG
- [ ] Memory layers diagram — render mermaid from README as SVG
- [ ] Loop diagram — custom SVG of the 7-step cycle
- [ ] Minsky portrait or book cover — check licensing (MIT Press)
- [ ] OG image for social sharing (1200×630)

---

## Content Principles

1. **Show, don't tell** — code snippets over adjectives
2. **Credit the source** — Minsky gets cited, not vaguely referenced
3. **Honest about status** — v0.1.0, actively built, not "production ready for everyone"
4. **Developer voice** — no marketing speak. "Nous gives agents structured memory" not "unlock the power of next-generation AI"
5. **Short paragraphs** — developers skim. Every paragraph should survive being skipped.

---

## Launch Sequence

### Phase 1 — MVP (ship after F018)
- [ ] Homepage only
- [ ] Quick start links to GitHub README for now
- [ ] Stats bar, hero, loop section, quick start, footer
- [ ] Domain purchased and DNS configured

### Phase 2 — Content Complete
- [ ] `/concepts` pages (all 7)
- [ ] `/docs` (mirror or link to GitHub docs)
- [ ] `/cognition-engines` bridge page
- [ ] OG images and social metadata

### Phase 3 — Growth
- [ ] `/blog` — launch post, Minsky deep dives, build log
- [ ] Analytics (privacy-respecting — Plausible or Fathom)
- [ ] Community links (Discord? GitHub Discussions?)

---

## Success Metrics

- **GitHub stars** — primary signal for developer interest
- **Clone-to-docker-up rate** — tracked via docs analytics
- **Inbound issues/PRs** — quality signal (are people actually using it?)
- **Search visibility** — "Minsky agent framework", "Society of Mind AI agent"

---

## Open Questions

1. **Domain name** — `nous-framework.ai`, `getnous.dev`, or something else? `nous.ai` is almost certainly taken.
2. **Docs hosting** — embedded in site or link to GitHub? Embedded is better UX but more maintenance.
3. **Blog from day one?** — A good launch post (explaining the Minsky connection) could drive HackerNews / dev.to traction. Worth the effort?
4. **Video?** — A 90-second demo (agent running, showing memory, decisions, calibration) could be very effective. Worth building before launch?
5. **Should cognition-engines.ai and nous-framework.ai visually relate?** — Shared design language would signal they're part of the same project ecosystem.
