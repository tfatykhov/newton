# 015: Deep-Thinking Ratio (DTR) — Measuring Real Reasoning Effort

**Paper:** Peng, Tan, Zhao, Chen, Lin, Go, Meng (Google / UVA). [arXiv:2602.13517](https://arxiv.org/html/2602.13517v1)
**Date reviewed:** 2026-02-23
**Relevance:** High — directly applicable to Nous confidence calibration and reasoning quality assessment

## Core Idea

Raw token count has a **negative** correlation (r=-0.59) with accuracy in reasoning tasks. Longer CoT ≠ better reasoning. Instead, measure **deep-thinking tokens** — tokens whose internal predictions don't stabilize until the final ~15% of model layers. The ratio of these tokens (DTR) has a strong **positive** correlation (r=0.683) with accuracy.

## How It Works

### Deep-Thinking Tokens

For each generated token, project intermediate layer hidden states into vocabulary space using the unembedding matrix. Measure Jensen-Shannon Divergence (JSD) between each layer's distribution and the final layer's distribution.

- **Early-settling tokens**: predictions converge in shallow layers (e.g., "and", "is", "the") — low thinking effort
- **Deep-thinking tokens**: predictions keep changing until deep layers (e.g., answer digits, operator completions) — high thinking effort

### The Algorithm

1. For each token at generation step t, compute JSD between layer l's distribution and final layer's distribution
2. Find settling depth c_t = first layer where JSD drops below threshold g (default 0.5)
3. If c_t is in the final ρ fraction of layers (default ρ=0.85, meaning last 15%), it's a deep-thinking token
4. DTR = count of deep-thinking tokens / total tokens

### Think@n Strategy

Instead of Self-Consistency (generate n samples, majority vote):
1. Start generating n candidate responses
2. After just 50 tokens, calculate DTR for each prefix
3. **Early-halt** candidates with low DTR (unpromising)
4. Continue only high-DTR candidates
5. **Result: ~50% cost reduction** with equal or better accuracy

## Key Results

| Metric | Avg Correlation with Accuracy |
|--------|-------------------------------|
| Token count | r = -0.59 (negative!) |
| Reverse token count | r = +0.59 (post-hoc, not causal) |
| Log probability | r = +0.22 to +0.61 (inconsistent) |
| Self-Certainty | r = +0.39 to +0.60 (inconsistent) |
| **DTR** | **r = +0.683** (robust across models) |

On GPT-OSS-120B-medium:
- AIME 2025: Think@n matches Cons@n accuracy (92.7%) at ~50% cost
- GPQA-Diamond: Think@n matches at ~50% cost
- Consistent across GPT-OSS, DeepSeek-R1, Qwen3 model families

## Key Findings

1. **"Longer is better" is wrong**: Token count ANTI-correlates with accuracy. Overthinking degrades performance.
2. **DTR is model-agnostic**: Works across GPT-OSS (20B, 120B), DeepSeek-R1-70B, Qwen3-30B
3. **Early prediction possible**: DTR from just 50-token prefix predicts final answer quality
4. **Hyperparameters are stable**: g=0.5, ρ=0.85 works well across all tested settings
5. **Functional vs templated tokens**: "boxed", "return" settle early; answer tokens, operator completions settle late

## Connections to Nous

### 1. Confidence Calibration (Brain)

**Current approach**: Confidence is manually assigned (0-1 float) by the agent.

**DTR opportunity**: If Nous has access to intermediate layer states (via Anthropic API or open-weight models), DTR could provide an **objective confidence signal**:
- High DTR on reasoning tokens → model was "thinking hard" → potentially more reliable
- Low DTR on answer tokens → model was on "autopilot" → flag for lower confidence
- This could auto-calibrate confidence instead of relying on self-reported values

**Implementation path**: Not possible with Anthropic API (no layer access). But for open-weight model deployments (DeepSeek, Qwen), Nous could compute DTR as a confidence input.

### 2. Reasoning Quality Monitor (MonitorEngine)

**Current approach**: MonitorEngine uses structural heuristics (errors → surprise 0.9, tool errors → 0.3, clean → 0.0).

**DTR opportunity**: For open-weight models, DTR on the response could be a **reasoning quality signal**:
- Low DTR on a complex question → superficial reasoning → flag for re-examination
- High DTR throughout → sustained deep thinking → higher trust in result
- Could trigger re-generation if DTR is below threshold for high-stakes decisions

### 3. Think@n for Decision Making

**Current approach**: One LLM call per turn.

**DTR opportunity**: For critical decisions (stakes=high/critical), implement Think@n:
1. Generate 3-5 candidate responses
2. Score each by DTR (from 50-token prefix if using open-weight)
3. Select highest-DTR response
4. Cost: ~1.5x a single call (vs 3-5x for full self-consistency)

### 4. Post-Turn Assessment Enhancement

The Monitor's `assess()` could incorporate DTR-inspired heuristics even without layer access:
- **Response length vs complexity mismatch**: Very long response to a simple question → likely overthinking (DTR paper confirms this)
- **Answer position**: If the "real answer" appears early then gets revised multiple times → potential overthinking signal
- These are surface-level proxies that align with DTR's findings

### 5. Censor Evolution (M1.3)

DTR findings suggest a new guardrail type:
```
"Overthinking Censor": When response length exceeds 3x median for similar-stakes decisions,
flag as potential overthinking. Don't block, but reduce confidence automatically.
```

## Design Decisions for Nous

### D1: Abstract the confidence signal
Don't hardcode DTR. Instead, create a `ReasoningQualityProvider` interface:
```python
class ReasoningQualityProvider(ABC):
    async def score(self, response_tokens, model_internals=None) -> float:
        """Score reasoning quality 0-1. Higher = more genuine reasoning."""
```
- Default implementation: heuristic (length ratio, answer stability)
- DTR implementation: requires layer access (open-weight only)
- Future: model-specific implementations

### D2: Surface-level DTR proxy for API models
Since we can't access Anthropic's layer states, build proxy metrics:
- Token count vs similar past decisions (is this abnormally long?)
- Answer revision count (how many times did the conclusion change?)
- Reasoning density (ratio of substantive tokens to filler)

### D3: Think@n as configurable strategy
Add to config:
```python
think_at_n: int = 1  # Number of candidates per turn (1 = no sampling)
think_at_n_stakes: str = "critical"  # Minimum stakes to trigger multi-sampling
```

## Limitations

1. **Requires layer access**: Core DTR needs intermediate hidden states — not available via Claude API
2. **Compute overhead**: Computing JSD for every token across all layers adds ~20-30% overhead per token
3. **Not tested on tool use**: All benchmarks are math/science reasoning, not agentic tool-use scenarios
4. **Correlation ≠ causation**: High DTR correlates with accuracy but doesn't mean forcing deeper thinking helps
5. **Prefix DTR is approximate**: 50-token prefix prediction works but is less reliable than full-sequence DTR

## Actionable Next Steps

1. **Create research note** ✅ (this file)
2. **Add ReasoningQualityProvider interface** to Nous cognitive schemas (future feature)
3. **Implement surface-level proxy** in MonitorEngine.assess() — flag overthinking based on length anomalies
4. **Spec Think@n** as F009 or similar feature for critical decision multi-sampling
5. **Track DTR research** — as Anthropic/OpenAI expose more model internals, DTR could become directly usable
